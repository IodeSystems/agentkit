package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestDiscoverContext_FindsBoundary stands up a server that accepts prompts up
// to a token threshold and rejects larger ones with a context-overflow 400 —
// DiscoverContext should converge near the threshold.
func TestDiscoverContext_FindsBoundary(t *testing.T) {
	const limitTokens = 4096 // server accepts prompts up to ~this many "word " tokens
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		// ~1 token per "word " (5 chars).
		toks := len(req.Messages[0].Content) / 5
		if toks > limitTokens {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":{"message":"This model's maximum context length is 4096 tokens"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "test-model")
	got, err := c.DiscoverContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// Converges to the largest accepted size, within the 256-token resolution.
	if got > limitTokens || limitTokens-got > 256 {
		t.Fatalf("discovered %d, want within 256 below %d", got, limitTokens)
	}
}

func TestDiscoverContext_PropagatesNonOverflowError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	if _, err := c.DiscoverContext(context.Background()); err == nil {
		t.Fatal("a non-overflow error (401) should propagate, not look like a boundary")
	}
}

func TestBodyIsContextOverflow(t *testing.T) {
	for _, s := range []string{
		"This model's maximum context length is 8192 tokens",
		"the input exceeds n_ctx",
		"prompt is too long",
		"reduce the length of the messages",
	} {
		if !bodyIsContextOverflow(s) {
			t.Errorf("should detect overflow: %q", s)
		}
	}
	if bodyIsContextOverflow("invalid api key") {
		t.Error("must not treat an auth error as a context overflow")
	}
}
