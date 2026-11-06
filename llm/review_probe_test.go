package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestCountTokens_TransportFailureOnFirstProbeIsNotCachedAsAbsent(t *testing.T) {
	resetTokenizeCache()
	var down atomic.Bool
	down.Store(true)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if down.Load() {
			if hj, ok := w.(http.Hijacker); ok {
				conn, _, _ := hj.Hijack()
				conn.Close()
				return
			}
		}
		fmt.Fprint(w, `{"tokens":[1,2,3]}`)
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	if _, ok := c.CountTokens(context.Background(), "x"); ok {
		t.Fatal("counted while the server was dropping connections")
	}
	down.Store(false)
	if n, ok := c.CountTokens(context.Background(), "x"); !ok || n != 3 {
		t.Fatalf("n=%d ok=%v after recovery; a transport fault was remembered as no tokenizer", n, ok)
	}
}

func TestCountTokens_AuthFailureIsNotCachedAsAbsent(t *testing.T) {
	resetTokenizeCache()
	var status atomic.Int32
	status.Store(401)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s := int(status.Load()); s != 200 {
			http.Error(w, "no", s)
			return
		}
		fmt.Fprint(w, `{"tokens":[1,2]}`)
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	if _, ok := c.CountTokens(context.Background(), "x"); ok {
		t.Fatal("counted through a 401")
	}
	status.Store(200)
	if n, ok := c.CountTokens(context.Background(), "x"); !ok || n != 2 {
		t.Fatalf("n=%d ok=%v; a 401 was remembered as no tokenizer", n, ok)
	}
}

func TestKeyBudget_AuthAndServerErrorsAreNotCachedAsUnsupported(t *testing.T) {
	for _, code := range []int{401, 403, 500, 503} {
		resetBudgetRoutes()
		var status atomic.Int32
		status.Store(int32(code))
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if s := int(status.Load()); s != 200 {
				http.Error(w, "no", s)
				return
			}
			fmt.Fprint(w, `{"info":{"spend":1.5,"max_budget":10}}`)
		}))
		c := NewClient(srv.URL, "k", "m")
		if _, err := c.KeyBudget(context.Background()); err == nil || errors.Is(err, ErrBudgetUnsupported) {
			t.Fatalf("status %d: err = %v; want the status error, not ErrBudgetUnsupported", code, err)
		}
		status.Store(200)
		if _, err := c.KeyBudget(context.Background()); err != nil {
			t.Fatalf("status %d: after recovery err = %v; the failure was remembered as unsupported", code, err)
		}
		srv.Close()
	}
}

func TestProbes_SendClientHeaders(t *testing.T) {
	resetTokenizeCache()
	seen := map[string]string{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen[r.URL.Path] = r.Header.Get("X-Gateway-Key")
		switch r.URL.Path {
		case "/tokenize":
			fmt.Fprint(w, `{"tokens":[1]}`)
		case "/props":
			fmt.Fprint(w, `{"n_ctx":4096}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	c.Header = http.Header{"X-Gateway-Key": {"secret"}}
	if _, ok := c.CountTokens(context.Background(), "x"); !ok {
		t.Fatal("no count")
	}
	if _, ok := c.ContextWindow(context.Background()); !ok {
		t.Fatal("no window")
	}
	for _, p := range []string{"/tokenize", "/props"} {
		if seen[p] != "secret" {
			t.Errorf("%s: X-Gateway-Key = %q; Client.Header was not applied", p, seen[p])
		}
	}
}

func TestCountPrompt_AnthropicProbeSendsClientHeaders(t *testing.T) {
	resetTokenizeCache()
	var got string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/messages/count_tokens" {
			got = r.Header.Get("X-Gateway-Key")
			fmt.Fprint(w, `{"input_tokens":9}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	c.Header = http.Header{"X-Gateway-Key": {"secret"}}
	if _, ok := c.CountPrompt(context.Background(), "s", nil); !ok {
		t.Fatal("no count")
	}
	if got != "secret" {
		t.Errorf("X-Gateway-Key = %q; Client.Header was not applied", got)
	}
}
