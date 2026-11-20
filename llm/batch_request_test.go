package llm

// Tests for the batch request path: ChatOpts.Batch routes to the provider's
// batch endpoint carrying the completion window, and the real-time path is
// byte-identical to the pre-batch client.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestBatchPath_PostsToBatchEndpointWithWindow asserts that a ChatWithOpts call
// with Batch:true posts to the /v1/chat/completions/batch URL and the body
// carries completion_window in seconds.
func TestBatchPath_PostsToBatchEndpointWithWindow(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"usage":{"prompt_tokens":1,"completion_tokens":1}}]}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "m")
	out, _, err := c.ChatWithOpts(context.Background(),
		[]Message{{Role: "user", Content: "x"}}, nil,
		&ChatOpts{Batch: true, CompletionWindow: 24 * time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if out != "ok" {
		t.Errorf("out = %q; want ok", out)
	}
	if gotPath != "/v1/chat/completions/batch" {
		t.Errorf("path = %q; want /v1/chat/completions/batch", gotPath)
	}
	cw, ok := gotBody["completion_window"]
	if !ok {
		t.Fatal("completion_window missing from batch request body")
	}
	// 24h = 86400 seconds
	if cw != float64(86400) {
		t.Errorf("completion_window = %v; want 86400", cw)
	}
}

// TestBatchPath_NoWindowOmitsField asserts that a batch call with
// CompletionWindow <= 0 does not add a completion_window field to the body.
func TestBatchPath_NoWindowOmitsField(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "m")
	_, _, err := c.ChatWithOpts(context.Background(),
		[]Message{{Role: "user", Content: "x"}}, nil,
		&ChatOpts{Batch: true})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/chat/completions/batch" {
		t.Errorf("path = %q; want /v1/chat/completions/batch", gotPath)
	}
	if _, ok := gotBody["completion_window"]; ok {
		t.Errorf("completion_window present; want absent when window is zero")
	}
}

// TestBatchPath_MessagesAPI_PostsToMessagesBatches asserts the Messages API
// variant: a client with MessagesAPI=true posts to /v1/messages/batches and
// the completion_window field is present after the shim transform.
func TestBatchPath_MessagesAPI_PostsToMessagesBatches(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"usage":{"prompt_tokens":1,"completion_tokens":1}}]}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "m")
	c.MessagesAPI = true
	_, _, err := c.ChatWithOpts(context.Background(),
		[]Message{{Role: "user", Content: "x"}}, nil,
		&ChatOpts{Batch: true, CompletionWindow: 6 * time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/v1/messages/batches" {
		t.Errorf("path = %q; want /v1/messages/batches", gotPath)
	}
	cw, ok := gotBody["completion_window"]
	if !ok {
		t.Fatal("completion_window missing from Messages-API batch request body")
	}
	if cw != float64(21600) {
		t.Errorf("completion_window = %v; want 21600", cw)
	}
}

// TestRealTimePath_BodyUnchanged asserts that a non-batch Chat call posts to
// the standard /v1/chat/completions URL and the body has no completion_window
// field — i.e. the wire format is identical to the pre-batch client.
func TestRealTimePath_BodyUnchanged(t *testing.T) {
	var gotPath string
	var gotBody []byte
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody, _ = io.ReadAll(r.Body)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"usage":{"prompt_tokens":5,"completion_tokens":3}}]}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "m")
	out, _, err := c.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hello"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out != "ok" {
		t.Errorf("out = %q; want ok", out)
	}
	// Standard path, not batch.
	if gotPath != "/v1/chat/completions" {
		t.Errorf("path = %q; want /v1/chat/completions", gotPath)
	}
	// No completion_window field in the body.
	var body map[string]any
	if err := json.Unmarshal(gotBody, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if _, ok := body["completion_window"]; ok {
		t.Errorf("completion_window present in real-time body; must be absent")
	}
	// The body must carry the standard fields: model and messages.
	if body["model"] != "m" {
		t.Errorf("model = %v; want m", body["model"])
	}
	if _, ok := body["messages"]; !ok {
		t.Errorf("messages missing from real-time body")
	}
}

// TestRealTimePath_BodyIdenticalToExplicitNonBatch asserts that the zero-value
// ChatOpts (the pre-batch default) produces the same wire body as a
// ChatOpts{} with no batch fields set.
func TestRealTimePath_BodyIdenticalToExplicitNonBatch(t *testing.T) {
	var body1, body2 []byte
	var n int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if n == 0 {
			body1 = body
		} else {
			body2 = body
		}
		n++
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "m")
	msgs := []Message{{Role: "user", Content: "compare"}}

	// Call 1: default (no opts).
	_, _, _ = c.Chat(context.Background(), msgs, nil)
	// Call 2: explicit non-batch opts.
	_, _, _ = c.ChatWithOpts(context.Background(), msgs, nil, &ChatOpts{})

	if len(body1) == 0 || len(body2) == 0 {
		t.Fatal("server did not receive both requests")
	}
	if string(body1) != string(body2) {
		t.Errorf("bodies differ:\n default:  %s\n explicit: %s", body1, body2)
	}
}
