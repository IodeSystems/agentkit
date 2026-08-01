package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

func resetTokenizeCache() {
	tokenizeRoutes.Range(func(k, _ any) bool { tokenizeRoutes.Delete(k); return true })
}

// llama.cpp answers /tokenize beside /v1, with {"tokens":[...]}.
func TestCountTokens_LlamaCppRoute(t *testing.T) {
	resetTokenizeCache()
	var hits []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits = append(hits, r.URL.Path)
		if r.URL.Path != "/tokenize" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(`{"tokens":[1,2,3,4,5]}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL+"/v1", "", "m")
	n, ok := c.CountTokens(context.Background(), "some text")
	if !ok || n != 5 {
		t.Fatalf("CountTokens = %d, %v; want 5, true (tried %v)", n, ok, hits)
	}
}

// vLLM answers /v1/tokenize with {"count":N}.
func TestCountTokens_VLLMRoute(t *testing.T) {
	resetTokenizeCache()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/tokenize" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(`{"count":11,"tokens":[1,2],"max_model_len":4096}`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	if n, ok := c.CountTokens(context.Background(), "x"); !ok || n != 11 {
		t.Fatalf("CountTokens = %d, %v; want 11, true", n, ok)
	}
}

// corrallm forwards llama.cpp's route under /upstream/<model>/.
func TestCountTokens_UpstreamPassthrough(t *testing.T) {
	resetTokenizeCache()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/upstream/qwen/tokenize" {
			// Everything else is the proxy's single-page app.
			w.Write([]byte(`<!doctype html><html><body>corrallm</body></html>`))
			return
		}
		w.Write([]byte(`{"tokens":[1,2,3]}`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "qwen")
	if n, ok := c.CountTokens(context.Background(), "x"); !ok || n != 3 {
		t.Fatalf("CountTokens = %d, %v; want 3, true", n, ok)
	}
}

// The failure that makes a status check useless: a proxy with a UI answers 200
// and HTML for every unknown path, so /tokenize "exists" everywhere. Only the
// body distinguishes a tokenizer from an index page.
func TestCountTokens_HTMLCatchAllIsNotATokenizer(t *testing.T) {
	resetTokenizeCache()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("<!doctype html>\n<html lang=\"en\"><head><title>corrallm</title></head></html>"))
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	if n, ok := c.CountTokens(context.Background(), "x"); ok {
		t.Fatalf("an HTML catch-all was accepted as a tokenizer (n=%d)", n)
	}
}

// An endpoint with no tokenizer must cost ONE discovery, not one per call.
func TestCountTokens_AbsenceIsRememberedNotRetried(t *testing.T) {
	resetTokenizeCache()
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		http.Error(w, "no", http.StatusNotFound)
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	for i := 0; i < 5; i++ {
		if _, ok := c.CountTokens(context.Background(), "x"); ok {
			t.Fatal("reported a tokenizer that is not there")
		}
	}
	if calls > 3 {
		t.Fatalf("%d probe requests for 5 calls — absence is not being remembered", calls)
	}
}

func TestContextWindow_FromProps(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/props" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(`{"total_slots":1,"default_generation_settings":{"n_ctx":180224}}`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	n, ok := c.ContextWindow(context.Background())
	if !ok || n != 180224 {
		t.Fatalf("ContextWindow = %d, %v; want 180224, true", n, ok)
	}
}

func TestContextWindow_AbsentIsNotAnError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`<!doctype html><html></html>`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL+"/v1", "", "m")
	if n, ok := c.ContextWindow(context.Background()); ok {
		t.Fatalf("an HTML page reported a context window of %d", n)
	}
}

func TestTokenizeCandidatesCoverTheKnownShapes(t *testing.T) {
	c := NewClient("https://host/v1", "", "qwen")
	var urls []string
	for _, r := range c.tokenizeCandidates() {
		urls = append(urls, r.url)
	}
	for _, want := range []string{
		"https://host/tokenize",               // bare llama.cpp
		"https://host/upstream/qwen/tokenize", // corrallm passthrough
		"https://host/v1/tokenize",            // vLLM
	} {
		if !slices.Contains(urls, want) {
			t.Errorf("candidate %q missing from %v", want, urls)
		}
	}
}
