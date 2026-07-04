package agent

import (
	"encoding/json"
	"fmt"
)

// Lifting — async tool results.
//
// A synchronous tool answers inline. A tool that can't (it kicked off an
// upstream job, an approval, a long shell) instead returns the LIFT wire
// shape as its result string:
//
//	{"pending": true, "correlation_id": "X", "ttl_s": 30}
//
// The dispatcher recognizes it (ParseLiftRequest), records the pending call
// keyed by correlation_id, and substitutes PendingResult(...) so the model
// sees a clear "this is pending, wrap up" message instead of the placeholder.
// The turn ends normally; the SESSION STAYS ACTIVE. When the upstream
// completes, the host injects the real payload as a KindToolResult / notice
// Entry keyed by the same tool_call_id, and the next Turn reconciles.
//
// agent owns only the wire shape + the canonical messages — it is NOT a
// goroutine park. Storage (the pending table), the inbound completion
// endpoint, and deadline GC stay with the host, which already has a durable
// session + inbox to redeliver on. This keeps lifting event-driven rather
// than blocking a turn.

// LiftRequest is the parsed async-tool opt-in.
type LiftRequest struct {
	Pending       bool   `json:"pending"`
	CorrelationID string `json:"correlation_id"`
	TTLSeconds    int    `json:"ttl_s"`
}

// ParseLiftRequest reports whether a tool result opted into async semantics.
// ok is true only when the result is a JSON object with pending=true and a
// non-empty correlation_id. Cheap-rejects non-JSON without allocating.
func ParseLiftRequest(result string) (LiftRequest, bool) {
	if len(result) == 0 || result[0] != '{' {
		return LiftRequest{}, false
	}
	var lr LiftRequest
	if err := json.Unmarshal([]byte(result), &lr); err != nil {
		return LiftRequest{}, false
	}
	if !lr.Pending || lr.CorrelationID == "" {
		return LiftRequest{}, false
	}
	return lr, true
}

// PendingResult is the canonical result string a dispatcher substitutes for a
// lifted call so the model stops acting on the placeholder and wraps up. The
// host may use its own wording; this is a sensible default for new consumers.
func PendingResult(correlationID, toolCallID string, ttlSeconds int) string {
	msg := fmt.Sprintf(
		"This tool call is now PENDING (correlation_id=%s, tool_call_id=%s). "+
			"The real result will arrive as a follow-up event when the upstream completes. "+
			"Do not retry this call. End your Turn or proceed with unrelated work; "+
			"the next Turn will see the lifted result.",
		correlationID, toolCallID)
	if ttlSeconds > 0 {
		msg += fmt.Sprintf(" Deadline: %ds.", ttlSeconds)
	}
	return msg
}
