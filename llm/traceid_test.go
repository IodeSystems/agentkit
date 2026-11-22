package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

var traceMsgs = []Message{{Role: "user", Content: "x"}}

// headerRecorder is a server that records the headers of every request and
// answers a plain chat reply carrying a body id.
func headerRecorder(t *testing.T) (*httptest.Server, func() []http.Header) {
	t.Helper()
	var mu sync.Mutex
	var seen []http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Clone())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/embeddings" {
			fmt.Fprint(w, `{"data":[{"index":0,"embedding":[0.1]}]}`)
			return
		}
		fmt.Fprint(w, `{"id":"chatcmpl-abc","model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	return srv, func() []http.Header {
		mu.Lock()
		defer mu.Unlock()
		return append([]http.Header(nil), seen...)
	}
}

func TestEmbedSendsTraceIDFromContext(t *testing.T) {
	srv, seen := headerRecorder(t)
	c := NewClient(srv.URL, "", "m")
	if _, err := c.Embed(WithTraceID(context.Background(), "thr-1"), "e", []string{"a"}); err != nil {
		t.Fatal(err)
	}
	if h := seen(); len(h) != 1 || h[0].Get("X-Trace-Id") != "thr-1" {
		t.Errorf("headers = %v", h)
	}
}

func TestDiscoverContextSendsTraceIDFromContext(t *testing.T) {
	srv, seen := headerRecorder(t)
	c := NewClient(srv.URL, "", "m")
	_, _ = c.DiscoverContext(WithTraceID(context.Background(), "thr-2"))
	h := seen()
	if len(h) == 0 {
		t.Fatal("no request")
	}
	for i, hh := range h {
		if hh.Get("X-Trace-Id") != "thr-2" {
			t.Errorf("request %d X-Trace-Id = %q", i, hh.Get("X-Trace-Id"))
		}
	}
}

func TestChatOptsTraceIDBeatsContextTraceID(t *testing.T) {
	srv, seen := headerRecorder(t)
	c := NewClient(srv.URL, "", "m")
	var got []Usage
	c.OnUsage = func(u Usage) { got = append(got, u) }
	ctx := WithTraceID(context.Background(), "from-ctx")
	if _, err := c.ChatWithOpts(ctx, traceMsgs, nil, &ChatOpts{TraceID: "from-opts"}); err != nil {
		t.Fatal(err)
	}
	if _, err := c.ChatWithOpts(ctx, traceMsgs, nil, nil); err != nil {
		t.Fatal(err)
	}
	h := seen()
	if h[0].Get("X-Trace-Id") != "from-opts" || h[1].Get("X-Trace-Id") != "from-ctx" {
		t.Errorf("headers = %v", h)
	}
	if len(got) != 2 || got[0].TraceID != "from-opts" || got[1].TraceID != "from-ctx" {
		t.Errorf("usage trace ids = %+v", got)
	}
}

func TestUsageCarriesResponseIDNonStreaming(t *testing.T) {
	srv, _ := headerRecorder(t)
	c := NewClient(srv.URL, "", "m")
	var got []Usage
	c.OnUsage = func(u Usage) { got = append(got, u) }
	if _, err := c.ChatWithOpts(context.Background(), traceMsgs, nil, &ChatOpts{TraceID: "t"}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ResponseID != "chatcmpl-abc" || got[0].TraceID != "t" {
		t.Errorf("usage = %+v", got)
	}
}

func streamTrace(t *testing.T, body string, hdr map[string]string) Usage {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		for k, v := range hdr {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	var got []Usage
	c.OnUsage = func(u Usage) { got = append(got, u) }
	ch, err := c.ChatStream(context.Background(), traceMsgs, nil, &ChatOpts{TraceID: "st"})
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	if len(got) != 1 {
		t.Fatalf("OnUsage called %d times", len(got))
	}
	return got[0]
}

func TestUsageCarriesFirstChunkIDOnStreamedReply(t *testing.T) {
	u := streamTrace(t, "data: {\"id\":\"gen-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"+
		"data: {\"id\":\"gen-2\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"+
		"data: [DONE]\n\n", nil)
	if u.ResponseID != "gen-1" || u.TraceID != "st" {
		t.Errorf("usage = %+v", u)
	}
}

func TestHeaderOnlyUsageCarriesResponseIDAndTraceID(t *testing.T) {
	u := streamTrace(t, "data: {\"id\":\"gen-9\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
		map[string]string{"x-ratelimit-remaining-requests": "3"})
	if u.ResponseID != "gen-9" || u.TraceID != "st" {
		t.Errorf("usage = %+v", u)
	}
}

func TestChatOptsHeadersSentAndCredentialsNotOverridable(t *testing.T) {
	srv, seen := headerRecorder(t)
	c := NewClient(srv.URL, "real-key", "m")
	h := http.Header{}
	h.Set("x-openrouter-trace-job", "j1")
	h.Set("Authorization", "Bearer evil")
	h.Set("Content-Type", "text/plain")
	if _, err := c.ChatWithOpts(context.Background(), traceMsgs, nil, &ChatOpts{Headers: h}); err != nil {
		t.Fatal(err)
	}
	got := seen()[0]
	if got.Get("x-openrouter-trace-job") != "j1" {
		t.Errorf("extra header not sent: %v", got)
	}
	if got.Get("Authorization") != "Bearer real-key" || got.Get("Content-Type") != "application/json" {
		t.Errorf("credential or body type overridden: %v", got)
	}
}
