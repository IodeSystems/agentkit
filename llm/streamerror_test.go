package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// sseServer answers each POST with the next body in bodies (the last repeats),
// as a 200 event stream, and counts the requests.
func sseServer(t *testing.T, hits *atomic.Int32, bodies ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(hits.Add(1)) - 1
		if n >= len(bodies) {
			n = len(bodies) - 1
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, bodies[n])
	}))
	t.Cleanup(srv.Close)
	return srv
}

const (
	okStream  = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	errEvent  = "data: {\"error\":{\"message\":\"upstream failed\",\"code\":502,\"metadata\":{\"provider_name\":\"Google\",\"raw\":\"boom\"}}}\n\n"
	contErr   = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n" + errEvent
	badReqErr = "data: {\"error\":{\"message\":\"bad input\",\"code\":400,\"type\":\"invalid_request_error\"}}\n\n"
)

func drainStream(t *testing.T, c *Client) (text string, errs []StreamChunk) {
	t.Helper()
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	for chunk := range ch {
		text += chunk.Content
		if chunk.Error != "" {
			errs = append(errs, chunk)
		}
	}
	return
}

// A transient error event ahead of any output must be invisible to the caller:
// one stream, the final text, one usage report, the retry narrated once.
func TestStreamError_BeforeAnyOutputIsRetriedInvisibly(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, errEvent, okStream)
	c := NewClient(srv.URL, "", "m")
	var kinds []RetryKind
	c.OnRetry = func(e RetryEvent) { kinds = append(kinds, e.Kind) }
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v; want hi, none", text, errs)
	}
	if hits.Load() != 2 {
		t.Errorf("server hits = %d; want 2", hits.Load())
	}
	if got := fmt.Sprint(kinds); got != "[stream recovered]" {
		t.Errorf("retry events = %s; want [stream recovered]", got)
	}
}

// Once content reached the caller a resend would regenerate the reply from
// scratch under text the caller already holds: no retry, typed error.
func TestStreamError_AfterContentIsNotRetriedAndKeepsDiagnosis(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, contErr, okStream)
	text, errs := drainStream(t, NewClient(srv.URL, "", "m"))
	if hits.Load() != 1 || text != "partial" || len(errs) != 1 {
		t.Fatalf("hits=%d text=%q errs=%d; want 1, partial, 1", hits.Load(), text, len(errs))
	}
	var se *StreamError
	if !errors.As(errs[0].Err(), &se) || se.Code != "502" || !strings.Contains(string(se.Metadata), "provider_name") {
		t.Fatalf("typed error = %+v", errs[0].StreamErr)
	}
	for _, want := range []string{"upstream failed", "code 502", "Google", "boom"} {
		if !strings.Contains(errs[0].Error, want) {
			t.Errorf("Error() %q lacks %q", errs[0].Error, want)
		}
	}
}

// A 400 would fail identically on a resend: surface it at once.
func TestStreamError_NonRetryableCodeIsNotRetried(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, badReqErr, okStream)
	_, errs := drainStream(t, NewClient(srv.URL, "", "m"))
	if hits.Load() != 1 || len(errs) != 1 || errs[0].StreamErr == nil || errs[0].StreamErr.Type != "invalid_request_error" {
		t.Fatalf("hits=%d errs=%+v; want 1 call and a typed 400", hits.Load(), errs)
	}
}

// With the budget spent the retry must stop and hand back the typed error
// instead of sleeping past the ceiling.
func TestStreamError_RetryBudgetExhaustedReturnsTypedError(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, errEvent)
	c := NewClient(srv.URL, "", "m")
	c.RetryBudget = 200 * time.Millisecond
	_, errs := drainStream(t, c)
	if len(errs) != 1 || errs[0].StreamErr == nil || errs[0].StreamErr.Code != "502" {
		t.Fatalf("errs=%+v; want one typed 502", errs)
	}
	if hits.Load() < 2 {
		t.Errorf("hits = %d; want at least one retry before giving up", hits.Load())
	}
}

// A retry_after inside the error sets the sleep; the budget gate sees it.
func TestStreamError_RetryAfterInErrorBeyondBudgetGivesUpAtOnce(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, "data: {\"error\":{\"message\":\"slow down\",\"code\":429,\"retry_after\":60}}\n\n", okStream)
	c := NewClient(srv.URL, "", "m")
	c.RetryBudget = time.Second
	_, errs := drainStream(t, c)
	if hits.Load() != 1 || len(errs) != 1 || errs[0].StreamErr.RetryAfter != time.Minute {
		t.Fatalf("hits=%d errs=%+v", hits.Load(), errs)
	}
}

// Providers send code as a number (OpenRouter) or a string (OpenAI-style).
func TestParseStreamError_CodeNumberOrString(t *testing.T) {
	for _, tc := range []struct {
		raw, code string
		retry     bool
	}{
		{`{"message":"m","code":503}`, "503", true},
		{`{"message":"m","code":"429"}`, "429", true},
		{`{"message":"m","code":"rate_limit_exceeded"}`, "rate_limit_exceeded", true},
		{`{"message":"m","code":"invalid_api_key","type":"auth"}`, "invalid_api_key", false},
		{`{"message":"Provider is overloaded"}`, "", true},
		{`"plain string"`, "", false},
	} {
		se := parseStreamError([]byte(tc.raw))
		if se == nil || se.Code != tc.code || se.Retryable() != tc.retry {
			t.Errorf("%s -> %+v retryable=%v; want code %q retryable %v", tc.raw, se, se != nil && se.Retryable(), tc.code, tc.retry)
		}
	}
	if parseStreamError([]byte(`null`)) != nil || parseStreamError([]byte(`{}`)) != nil {
		t.Error("null or empty error object parsed as an error")
	}
}
