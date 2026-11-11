package llm

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The whole point, end to end: corrallm stamps its overhead on a SERVED
// response, and the client must carry it as far as Usage. Before this, a
// request that queued nine seconds and was then served looked exactly like a
// slow generation.
func TestServerTimingReachesUsageOverTheWire(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(HeaderQueuedMS, "9000")
		w.Header().Set(HeaderLoadMS, "1200")
		w.Header().Set(HeaderUpstreamTTFBMS, "180")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var usage *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if !usage.Server.Reported {
		t.Fatal("Server.Reported false — the headers were dropped between response and Usage")
	}
	if usage.Server.QueuedMS != 9000 || usage.Server.LoadMS != 1200 {
		t.Errorf("Server = %+v, want queued 9000 / load 1200", usage.Server)
	}
	if usage.Server.OverheadMS() != 10200 {
		t.Errorf("OverheadMS = %d, want 10200", usage.Server.OverheadMS())
	}
	// LatencyMS is a local measurement of a fast loopback call, so the overhead
	// the server claims exceeds it. WorkMS must floor at zero rather than go
	// negative — the assertion that matters is that it did not report the full
	// latency as work.
	if got := usage.WorkMS(); got != 0 {
		t.Errorf("WorkMS = %d, want 0 for a claimed overhead larger than local latency", got)
	}
}

// An endpoint that reports nothing must leave Reported false, so a host can
// tell "no overhead" from "no information".
func TestPlainEndpointReportsNoTiming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var usage *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if usage.Server.Reported {
		t.Error("Reported must stay false against an endpoint that sends no timing headers")
	}
	if usage.WorkMS() != usage.LatencyMS {
		t.Errorf("WorkMS = %d, want the full LatencyMS %d", usage.WorkMS(), usage.LatencyMS)
	}
}

// A lane that failed over answers with a model the caller did not ask for;
// without the served model and the proxy's request id on Usage, an A/B across
// backends cannot say which backend produced which reply.
func TestServedModelAndRequestIDReachUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(HeaderServed, "swift-nvfp4")
		w.Header().Set(HeaderRequestID, "tkt-42")
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), `"stream":true`) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "swift")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var usage *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil || usage.ServedModel != "swift-nvfp4" || usage.RequestID != "tkt-42" {
		t.Errorf("stream usage = %+v, want served swift-nvfp4, request id tkt-42", usage)
	}

	c = NewClient(srv.URL, "", "swift")
	var got *Usage
	c.OnUsage = func(u Usage) { got = &u }
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got == nil || got.ServedModel != "swift-nvfp4" || got.RequestID != "tkt-42" {
		t.Errorf("non-stream usage = %+v", got)
	}
}

// attributionServer answers both shapes with the given headers and body model,
// so each provider case below differs only in what it sends.
func attributionServer(hdr map[string]string, model string) *httptest.Server {
	mf := ""
	if model != "" {
		mf = fmt.Sprintf(`"model":%q,`, model)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k, v := range hdr {
			w.Header().Set(k, v)
		}
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), `"stream":true`) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{%s"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, mf)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Model rides the content chunk; the usage chunk omits it, as on
		// OpenAI with include_usage.
		fmt.Fprintf(w, "data: {%s\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n", mf)
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

// attributionOf runs one stream and one non-stream call and returns the Usage
// each produced.
func attributionOf(t *testing.T, url string) (stream, plain Usage) {
	t.Helper()
	msgs := []Message{{Role: "user", Content: "x"}}
	c := NewClient(url, "", "asked-for")
	ch, err := c.ChatStream(context.Background(), msgs, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for chunk := range ch {
		if chunk.Usage != nil {
			stream = *chunk.Usage
		}
	}
	c = NewClient(url, "", "asked-for")
	c.OnUsage = func(u Usage) { plain = u }
	if _, _, err := c.Chat(context.Background(), msgs, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	return stream, plain
}

func wantAttribution(t *testing.T, stream, plain Usage, served, id string) {
	t.Helper()
	for name, u := range map[string]Usage{"stream": stream, "non-stream": plain} {
		if u.ServedModel != served || u.RequestID != id {
			t.Errorf("%s: served %q id %q, want %q / %q", name, u.ServedModel, u.RequestID, served, id)
		}
	}
}

// OpenAI sends x-request-id and the dated snapshot it resolved the alias to.
// Dropping either leaves the host unable to name the answering model or file a
// support ticket.
func TestOpenAIAttributionFromRequestIDHeaderAndBodyModel(t *testing.T) {
	srv := attributionServer(map[string]string{"X-Request-Id": "req_abc"}, "gpt-4o-2024-08-06")
	defer srv.Close()
	s, p := attributionOf(t, srv.URL)
	wantAttribution(t, s, p, "gpt-4o-2024-08-06", "req_abc")
}

// OpenRouter names the routed model in the body and sends no request header;
// the body model must still reach Usage and the id must stay empty.
func TestOpenRouterAttributionFromBodyModelOnly(t *testing.T) {
	srv := attributionServer(nil, "anthropic/claude-sonnet-4")
	defer srv.Close()
	s, p := attributionOf(t, srv.URL)
	wantAttribution(t, s, p, "anthropic/claude-sonnet-4", "")
}

// LiteLLM's only per-request id is x-litellm-call-id.
func TestLiteLLMAttributionFromCallIDHeader(t *testing.T) {
	srv := attributionServer(map[string]string{"X-Litellm-Call-Id": "call-7"}, "")
	defer srv.Close()
	s, p := attributionOf(t, srv.URL)
	wantAttribution(t, s, p, "", "call-7")
}

// A corrallm in front of an upstream that also names itself: the proxy's
// header says which backend answered, the body may be the upstream's own name
// or an echo of the request. The header must win for both fields.
func TestCorrallmHeadersBeatBodyModelAndVendorRequestID(t *testing.T) {
	srv := attributionServer(map[string]string{
		HeaderServed: "swift-nvfp4", HeaderRequestID: "tkt-1", "X-Request-Id": "req_vendor",
	}, "asked-for")
	defer srv.Close()
	s, p := attributionOf(t, srv.URL)
	wantAttribution(t, s, p, "swift-nvfp4", "tkt-1")
}

// A plain endpoint with no headers and no body model reports nothing, and one
// that only echoes a body model reports just that.
func TestPlainEndpointAttribution(t *testing.T) {
	srv := attributionServer(nil, "")
	defer srv.Close()
	s, p := attributionOf(t, srv.URL)
	wantAttribution(t, s, p, "", "")

	srv2 := attributionServer(nil, "local-gguf")
	defer srv2.Close()
	s, p = attributionOf(t, srv2.URL)
	wantAttribution(t, s, p, "local-gguf", "")
}

// Anthropic's Messages API sends request-id as a header and the model in
// message_start / the message body; the shim must carry the model through the
// conversion to the OpenAI shape or ServedModel is lost on this path.
func TestAnthropicMessagesAttribution(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Request-Id", "req_011C")
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(b), `"stream":true`) {
			fmt.Fprint(w, `{"id":"m","model":"claude-opus-5-5-20260101","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range []string{
			`{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5-20260101","usage":{"input_tokens":1,"output_tokens":1}}}`,
			`{"type":"message_stop"}`,
		} {
			fmt.Fprintf(w, "event: x\ndata: %s\n\n", e)
		}
	}))
	defer srv.Close()
	msgs := []Message{{Role: "user", Content: "x"}}
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	var stream Usage
	ch, err := c.ChatStream(context.Background(), msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for chunk := range ch {
		if chunk.Usage != nil {
			stream = *chunk.Usage
		}
	}
	c = NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	var plain Usage
	c.OnUsage = func(u Usage) { plain = u }
	if _, _, err := c.Chat(context.Background(), msgs, nil); err != nil {
		t.Fatal(err)
	}
	wantAttribution(t, stream, plain, "claude-opus-5-5-20260101", "req_011C")
}
