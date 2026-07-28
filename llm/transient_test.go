package llm

import (
	"errors"
	"fmt"
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
