package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	if _, ok := c.CountTokens(context.Background(), "x"); ok {
		t.Fatal("reported a tokenizer that is not there")
	}
	// Whatever discovery cost, it is paid ONCE. Asserted as "does not grow"
	// rather than against a fixed number, which was one-probe-per-candidate and
	// therefore broke the moment a candidate was added — a true property
	// reported as a regression.
	afterFirst := calls
	for i := 0; i < 4; i++ {
		if _, ok := c.CountTokens(context.Background(), "x"); ok {
			t.Fatal("reported a tokenizer that is not there")
		}
	}
	if calls != afterFirst {
		t.Fatalf("probes grew %d -> %d across 5 calls — absence is not being remembered",
			afterFirst, calls)
	}
	if afterFirst == 0 {
		t.Fatal("no probe was made at all; the test proves nothing")
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

// anthropicStub serves only Anthropic's count_tokens shape — every other
// candidate 404s — and records the last body it was handed.
func anthropicStub(t *testing.T, n int, seen *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages/count_tokens" {
			http.Error(w, "no", http.StatusNotFound)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var got map[string]any
		if err := json.Unmarshal(body, &got); err != nil {
			t.Errorf("unparseable request: %s", body)
		}
		*seen = got
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"input_tokens":%d}`, n)
	}))
}

// CountPrompt must send Anthropic's OWN tool spelling. The two formats express
// the same schema differently (`input_schema` vs `parameters`), and counting the
// wrong spelling returns a plausible number for a prompt that was never sent.
func TestCountPrompt_SendsAnthropicToolShape(t *testing.T) {
	resetTokenizeCache()
	var seen map[string]any
	ts := anthropicStub(t, 551, &seen)
	defer ts.Close()

	var td ToolDef
	td.Type = "function"
	td.Function.Name = "search"
	td.Function.Description = "search the corpus"
	td.Function.Parameters = map[string]any{"type": "object"}

	c := NewClient(ts.URL+"/v1", "", "claude-haiku-4-5")
	n, ok := c.CountPrompt(context.Background(), "you are an agent", []ToolDef{td})
	if !ok || n != 551 {
		t.Fatalf("CountPrompt = %d, %v; want 551, true", n, ok)
	}
	if seen["system"] != "you are an agent" {
		t.Errorf("system not sent: %v", seen["system"])
	}
	tools, _ := seen["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("want 1 tool, got %v", seen["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if _, ok := tool["input_schema"]; !ok {
		t.Errorf("tool sent in OpenAI shape; Anthropic needs input_schema: %v", tool)
	}
	if _, ok := tool["parameters"]; ok {
		t.Errorf("OpenAI's `parameters` leaked into an Anthropic request: %v", tool)
	}
}

// An empty tool list must not send `tools` at all: an empty array still turns
// the provider's tool-use preamble on, which measured ~497 tokens — so a prompt
// that declares no tools would be charged for tooling it does not have.
func TestCountPrompt_NoToolsSendsNoToolsKey(t *testing.T) {
	resetTokenizeCache()
	var seen map[string]any
	ts := anthropicStub(t, 20, &seen)
	defer ts.Close()

	c := NewClient(ts.URL+"/v1", "", "claude-haiku-4-5")
	if _, ok := c.CountPrompt(context.Background(), "you are an agent", nil); !ok {
		t.Fatal("CountPrompt failed on a system-only prompt")
	}
	if _, present := seen["tools"]; present {
		t.Errorf("sent a tools key for a prompt with no tools: %v", seen)
	}
}

// A structured-only endpoint has no answer to "how many tokens is this string".
// Reporting one would be a count of the whole envelope, which is not what the
// caller asked about.
func TestCountTokens_RefusesAStructuredOnlyEndpoint(t *testing.T) {
	resetTokenizeCache()
	var seen map[string]any
	ts := anthropicStub(t, 551, &seen)
	defer ts.Close()

	c := NewClient(ts.URL+"/v1", "", "claude-haiku-4-5")
	if n, ok := c.CountTokens(context.Background(), "hello world"); ok {
		t.Errorf("raw-string count reported on a structured-only endpoint: %d", n)
	}
	// ...but the endpoint IS a counter, and CountPrompt must still work.
	if _, ok := c.CountPrompt(context.Background(), "hello world", nil); !ok {
		t.Error("CountPrompt refused an endpoint that answers count_tokens")
	}
}
