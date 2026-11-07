package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// A 400 that says the prompt does not fit is typed, carries the endpoint's own
// counts, and is sent once: the same bytes fail the same way.
func TestContextOverflow400IsTypedWithCounts(t *testing.T) {
	for name, tc := range map[string]struct {
		status         int
		body           string
		prompt, window int
	}{
		"llama.cpp":              {400, `{"error":{"code":400,"message":"request (89653 tokens) exceeds the available context size (86016 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":89653,"n_ctx":86016}}`, 89653, 86016},
		"llama.cpp message only": {400, `{"error":{"code":400,"message":"request (89653 tokens) exceeds the available context size (86016 tokens), try increasing it","type":"exceed_context_size_error"}}`, 89653, 86016},
		"openai":                 {400, `{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens. Please reduce the length of the messages.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}`, 130512, 128000},
		"vllm":                   {400, `{"object":"error","message":"This model's maximum context length is 4096 tokens. However, you requested 5000 tokens (4000 in the messages, 1000 in the completion).","type":"BadRequestError","code":400}`, 4000, 4096},
		"anthropic":              {400, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 208310 tokens > 200000 maximum"}}`, 208310, 200000},
		"413 no counts":          {413, `{"error":{"code":"context_length_exceeded","message":"too big"}}`, 0, 0},
	} {
		t.Run(name, func(t *testing.T) {
			var hits int64
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt64(&hits, 1)
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			_, _, err := NewClient(srv.URL, "k", "m").Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil)
			var tl *InputTooLargeError
			if !errors.As(err, &tl) {
				t.Fatalf("want *InputTooLargeError, got %T: %v", err, err)
			}
			if tl.Status != tc.status || tl.PromptTokens != tc.prompt || tl.ContextTokens != tc.window {
				t.Errorf("got status %d prompt %d window %d, want %d %d %d",
					tl.Status, tl.PromptTokens, tl.ContextTokens, tc.status, tc.prompt, tc.window)
			}
			if n := atomic.LoadInt64(&hits); n != 1 {
				t.Errorf("made %d attempts; an overflow must not be resent unchanged", n)
			}
		})
	}
}

// Other 400s stay untyped.
func TestOrdinary400IsNotAnOverflow(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"tools[0].function.name is required","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()
	_, _, err := NewClient(srv.URL, "k", "m").Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil)
	var tl *InputTooLargeError
	if err == nil || errors.As(err, &tl) {
		t.Fatalf("a schema 400 is not an overflow: %v", err)
	}
}
