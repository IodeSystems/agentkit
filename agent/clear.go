package agent

import (
	"encoding/json"
)

// Notification lifecycle — supersede + clear.
//
// Injected notifications are cheap to publish but expensive to CONSUME: a
// stale in-flight notice (an MCP LSP "bad build" for a file that now compiles)
// is re-validated by the model every Turn until claimed — pure waste. Two
// primitives keyed on a notice's group key close that:
//
//   - SUPERSEDE: publishing a new notice for the same (type, group key)
//     replaces the prior UNSHOWN one (newest wins) instead of stacking, so
//     the inbox holds at most one live notice per key. This is a merge
//     strategy on the host's publish path — agent only names the wire shape.
//   - CLEAR: retract the notice entirely when the condition resolves. A tool
//     result or an integration callback carries the ClearRequest shape; the
//     host removes matching unshown deliveries. Zero model cost.
//
// The revalidate WAITER (a cheap pre-Turn check that self-clears a stale
// notice) is a host-scheduler concern — it composes with ClearRequest by
// issuing a clear when its check reports "resolved".

// ClearRequest is the wire shape a tool result / integration callback returns
// to retract prior unshown notices for a group key:
//
//	{"clear": true, "group_by": "file", "key": "src/main.go"}
//
// GroupBy names the notice field that partitions notices; Key is the value to
// clear. Both required.
type ClearRequest struct {
	Clear   bool   `json:"clear"`
	GroupBy string `json:"group_by"`
	Key     string `json:"key"`
}

// ParseClearRequest reports whether a result opted into a clear. ok is true
// only for a JSON object with clear=true and non-empty group_by + key.
func ParseClearRequest(result string) (ClearRequest, bool) {
	if len(result) == 0 || result[0] != '{' {
		return ClearRequest{}, false
	}
	var cr ClearRequest
	if err := json.Unmarshal([]byte(result), &cr); err != nil {
		return ClearRequest{}, false
	}
	if !cr.Clear || cr.GroupBy == "" || cr.Key == "" {
		return ClearRequest{}, false
	}
	return cr, true
}
