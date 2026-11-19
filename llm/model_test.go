package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Model is the body's own model field, distinct from ServedModel's
// header-then-first-seen semantics. The three cases the specs name:
// non-streaming body, a stream whose chunks name different models, and a
// response that names none.

// The request names an alias; the response names the concrete model it
// resolved to. Model must carry the response's name, not the request's.
func TestModelFromNonStreamingBody(t *testing.T) {
	const requested = "local-alias"
	const served = "/home/nthalk/.corrallm/models/Swift-1.5.gguf"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":%q,"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, served)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", requested)
	var got *Usage
	c.OnUsage = func(u Usage) { got = &u }
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got == nil {
		t.Fatal("OnUsage was not called")
	}
	if got.Model != served {
		t.Errorf("Model = %q, want the response's %q (not the request's %q)", got.Model, served, requested)
	}
}

// A stream where an early chunk names one model and the last chunk carrying a
// model names a different one: Model must be the LAST one.
func TestModelFromLastStreamChunk(t *testing.T) {
	const first = "early-model"
	const last = "late-model"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"model\":%q,\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n", first)
		fmt.Fprintf(w, "data: {\"model\":%q,\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n", last)
		// The usage chunk carries no model of its own, as on OpenAI with
		// include_usage.
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	msgs := []Message{{Role: "user", Content: "x"}}

	var streamUsage *Usage
	c := NewClient(srv.URL, "", "alias")
	ch, err := c.ChatStream(context.Background(), msgs, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for chunk := range ch {
		if chunk.Usage != nil {
			streamUsage = chunk.Usage
		}
	}
	if streamUsage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if streamUsage.Model != last {
		t.Errorf("final-chunk Model = %q, want the last %q (not the first %q)", streamUsage.Model, last, first)
	}

	var onUsage Usage
	c = NewClient(srv.URL, "", "alias")
	c.OnUsage = func(u Usage) { onUsage = u }
	ch, err = c.ChatStream(context.Background(), msgs, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for range ch {
	}
	if onUsage.Model != last {
		t.Errorf("OnUsage Model = %q, want the last %q", onUsage.Model, last)
	}
}

// A response that names no model must leave Model empty.
func TestModelEmptyWhenResponseNamesNone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	var usage *Usage
	c := NewClient(srv.URL, "", "alias")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if usage.Model != "" {
		t.Errorf("Model = %q, want empty when the response names no model", usage.Model)
	}
}
