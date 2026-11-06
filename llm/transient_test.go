package llm

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"syscall"
	"testing"
)

// The verbatim error from a run that died mid-deploy. It was originally read as
// the agent failing the task; the serving process had restarted nine minutes
// later, which is what it actually was.
const observedCancel = "agent: chat: stream error: stream ID 49; CANCEL; received from peer"

func TestTransientUpstream(t *testing.T) {
	transient := []string{
		observedCancel,
		"http2: server sent GOAWAY and closed the connection",
		`{"error":{"message":"no backend available"}}`,
		"read tcp 10.0.0.1:443: connection reset by peer",
		"dial tcp 127.0.0.1:8080: connect: connection refused",
		"unexpected EOF",
		"llm: upstream returned 503",
		// Verbatim from three benchmark runs lost to a saturated slots:1 model.
		"agent: chat: llm: retry budget 5m0s exhausted after 5m40s of 429 backpressure",
		"provider returned 429",
		"rate limit exceeded",
	}
	for _, s := range transient {
		if !TransientUpstream(errors.New(s)) {
			t.Errorf("should be transient: %q", s)
		}
		if TransientUpstreamReason(errors.New(s)) == "" {
			t.Errorf("transient errors must carry a reason: %q", s)
		}
	}

	// Everything the CALLER is responsible for must stay visible. Classifying a
	// bad request as infrastructure is how a real defect gets retried forever.
	terminal := []string{
		"llm: TRUNCATED TOOL CALL",
		`Failed to parse tool call arguments as JSON: missing closing quote`,
		"agent: max turns (40) exceeded",
		"llm: provider returned 400",
		"context deadline exceeded is a net.Error, but 401 is not",
	}
	for _, s := range terminal[:4] {
		if TransientUpstream(errors.New(s)) {
			t.Errorf("should NOT be transient: %q", s)
		}
	}

	if TransientUpstream(nil) {
		t.Error("nil is not an error")
	}
	if !TransientUpstream(fmt.Errorf("wrapped: %w", syscall.ECONNRESET)) {
		t.Error("should unwrap to the syscall errno")
	}
}

func TestTransientReasonNamesTheFault(t *testing.T) {
	got := TransientUpstreamReason(errors.New(observedCancel))
	if got == "" {
		t.Fatal("no reason")
	}
	t.Logf("reason: %s", got)
	for _, want := range []string{"restarted", "stream"} {
		if !strings.Contains(got, want) {
			t.Errorf("reason should mention %q, got %q", want, got)
		}
	}
}

func TestTransientUpstream_CallerCancellationAndDeadlineAreNeverTransient(t *testing.T) {
	for name, err := range map[string]error{
		"canceled":         &url.Error{Op: "Post", URL: "http://x", Err: context.Canceled},
		"deadline":         &url.Error{Op: "Post", URL: "http://x", Err: context.DeadlineExceeded},
		"wrapped canceled": fmt.Errorf("agent: chat: %w", &url.Error{Op: "Post", URL: "http://x", Err: context.Canceled}),
		"wrapped deadline": fmt.Errorf("agent: chat: %w", context.DeadlineExceeded),
	} {
		if TransientUpstream(err) {
			t.Errorf("%s: a caller's cancellation or deadline is not the server going away", name)
		}
	}
}

func TestTransientUpstream_StatusCodesMatchStructurallyNotAsBareNumbers(t *testing.T) {
	for _, s := range []string{
		"llm: status 400: 502 tokens too many",
		"llm: status 400: prompt has 429 messages",
		"llm: status 422: field 504 is invalid",
	} {
		if TransientUpstream(errors.New(s)) {
			t.Errorf("a number inside a 4xx body is not a gateway status: %q", s)
		}
	}
	for _, s := range []string{
		"llm: status 502",
		"llm: status 503: backend down",
		"llm: status 504 (after 4 retries)",
		"llm: retry budget 5m0s exhausted after 3s (last status 503)",
		"llm: status 429: slow down",
	} {
		if !TransientUpstream(errors.New(s)) {
			t.Errorf("real error string must stay transient: %q", s)
		}
	}
}
