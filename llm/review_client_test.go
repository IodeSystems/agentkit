package llm

// Regression coverage for transport-level faults in client.go: a stream line
// larger than a scanner buffer, a stream that ends without any terminator, SSE
// field spelling, request fields dropped on the non-streaming path, and retry
// clamping/parsing.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestReadStream_LineLargerThan64KiBIsNotAnError(t *testing.T) {
	big := strings.Repeat("x", 70*1024)
	body := `data: {"choices":[{"delta":{"content":"` + big + `"}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	var got strings.Builder
	for _, c := range collect(t, body) {
		if c.Error != "" {
			t.Fatalf("stream error on a 70 KiB data line: %.120s", c.Error)
		}
		got.WriteString(c.Content)
	}
	if got.Len() != len(big) {
		t.Fatalf("content length = %d; want %d", got.Len(), len(big))
	}
}

func TestReadStream_BodyEndingWithNoTerminatorIsReportedTruncated(t *testing.T) {
	body := `data: {"choices":[{"delta":{"content":"half an ans"}}]}` + "\n\n"
	var errs []string
	for _, c := range collect(t, body) {
		if c.Error != "" {
			errs = append(errs, c.Error)
		}
		if c.Done {
			t.Fatalf("a body with no [DONE] and no finish_reason must not report Done")
		}
	}
	if len(errs) != 1 || !strings.Contains(errs[0], "TRUNCATED") {
		t.Fatalf("errors = %q; want one saying the provider truncated the stream", errs)
	}
}

func TestReadStream_DataFieldWithoutSpaceIsAccepted(t *testing.T) {
	body := `data:{"choices":[{"delta":{"content":"hi"}}]}` + "\n\n" +
		`data:{"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
		"data:[DONE]\n\n"
	var got string
	var done bool
	for _, c := range collect(t, body) {
		got += c.Content
		done = done || c.Done
	}
	if got != "hi" || !done {
		t.Fatalf("content=%q done=%v; want hi, true", got, done)
	}
}

func TestChatWithOpts_ForwardsConstraintsAndIdentity(t *testing.T) {
	var gotBody map[string]any
	var trace string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		trace = r.Header.Get("X-Trace-Id")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	_, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "x"}}, nil,
		&ChatOpts{ToolChoice: "required", Grammar: "root ::= \"a\"",
			ResponseFormat: map[string]any{"type": "json_object"}, TraceID: "thread-7"})
	if err != nil {
		t.Fatal(err)
	}
	if gotBody["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v; want required", gotBody["tool_choice"])
	}
	if gotBody["grammar"] != "root ::= \"a\"" {
		t.Errorf("grammar = %v; dropped", gotBody["grammar"])
	}
	if gotBody["response_format"] == nil {
		t.Errorf("response_format dropped")
	}
	if trace != "thread-7" {
		t.Errorf("X-Trace-Id = %q; want thread-7", trace)
	}
}

func TestRetry5xx_RetryAfterPastBudgetIsClampedNotRefused(t *testing.T) {
	var n int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Retry-After", "30")
			w.WriteHeader(503)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	c.RetryBudget = 300 * time.Millisecond
	start := time.Now()
	out, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil)
	if err != nil || out != "ok" {
		t.Fatalf("out=%q err=%v; want the clamped retry to succeed", out, err)
	}
	if time.Since(start) > 5*time.Second {
		t.Fatalf("slept %s; the wait must be clamped to the budget", time.Since(start))
	}
}

func TestRetryAfterFrom_AcceptsFractionalSecondsAndHTTPDate(t *testing.T) {
	d, ok := retryAfterFrom(http.Header{}, []byte(`{"error":{"retry_after":1.5}}`))
	if !ok || d != 1500*time.Millisecond {
		t.Errorf("body float: d=%v ok=%v; want 1.5s", d, ok)
	}
	d, ok = retryAfterFrom(http.Header{}, []byte(`{"retry_after":2.5}`))
	if !ok || d != 2500*time.Millisecond {
		t.Errorf("top-level float: d=%v ok=%v; want 2.5s", d, ok)
	}
	h := http.Header{}
	h.Set("Retry-After", time.Now().Add(30*time.Second).UTC().Format(http.TimeFormat))
	d, ok = retryAfterFrom(h, nil)
	if !ok || d < 25*time.Second || d > 31*time.Second {
		t.Errorf("HTTP-date: d=%v ok=%v; want about 30s", d, ok)
	}
}
