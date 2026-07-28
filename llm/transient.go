package llm

import (
	"errors"
	"net"
	"strings"
	"syscall"
)

// TransientUpstream reports whether an error is the SERVER going away rather
// than anything wrong with the request.
//
// The distinction is not cosmetic. A benchmark that counts a redeploy as a task
// failure reports the wrong number, and an agent that treats one as a bad request
// rewrites a prompt that was fine. Measured case: a run died at 21:42:53 with
//
//	agent: chat: stream error: stream ID 49; CANCEL; received from peer
//
// and the serving process's start time was 21:51:55 — a build-and-restart window
// that killed every in-flight stream at its front. Two runs of a five-run arm
// were lost that way and looked, in the results table, exactly like the model
// failing the task.
//
// Deliberately narrow. It matches only faults that say the connection or the
// backend died: an HTTP/2 CANCEL or GOAWAY, a reset or refused connection, an EOF
// mid-stream, and the gateway statuses. A 400, a 500 from argument parsing, or a
// model that produced nonsense are all the caller's problem and must stay
// visible.
func TransientUpstream(err error) bool {
	if err == nil {
		return false
	}
	// Connection-level faults, without string matching where the type is known.
	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.EPIPE) {
		return true
	}
	var ne net.Error
	if errors.As(err, &ne) {
		return true
	}
	return transientText(err.Error())
}

// transientText matches the wire-level signatures that only reach us as text:
// http2 stream errors surface through the transport as strings, and a proxy's
// own body ("no backend available") is not an error type at all.
func transientText(s string) bool {
	s = strings.ToLower(s)
	for _, sig := range []string{
		"cancel; received from peer", // http2 RST_STREAM on a killed server
		"goaway",                     // graceful shutdown, mid-deploy
		"stream error",
		"connection reset",
		"connection refused",
		"broken pipe",
		"unexpected eof",
		"server closed",
		"no backend available", // corrallm with the model not yet up
		"502", "503", "504",
	} {
		if strings.Contains(s, sig) {
			return true
		}
	}
	return false
}

// TransientUpstreamReason names WHICH transient fault fired, for a log or a
// benchmark's result row. Empty when the error is not transient.
//
// It exists so a run can be marked infrastructure-invalid with the evidence
// attached, rather than a human reading stderr hours later and guessing — which
// is how the deploy above was originally misread as the agent failing.
func TransientUpstreamReason(err error) string {
	if !TransientUpstream(err) {
		return ""
	}
	s := strings.ToLower(err.Error())
	switch {
	case strings.Contains(s, "cancel; received from peer"):
		return "upstream cancelled the stream (server restarted or was killed mid-request)"
	case strings.Contains(s, "goaway"):
		return "upstream sent GOAWAY (graceful shutdown, likely a deploy)"
	case strings.Contains(s, "no backend available"):
		return "upstream has no backend for this model (not yet loaded, or being replaced)"
	case strings.Contains(s, "connection refused"):
		return "upstream refused the connection (process down)"
	case strings.Contains(s, "connection reset"), strings.Contains(s, "broken pipe"):
		return "upstream dropped the connection"
	case strings.Contains(s, "unexpected eof"):
		return "upstream closed the response early"
	case strings.Contains(s, "503"), strings.Contains(s, "502"), strings.Contains(s, "504"):
		return "upstream gateway error (backend unavailable)"
	}
	return "upstream transport fault"
}
