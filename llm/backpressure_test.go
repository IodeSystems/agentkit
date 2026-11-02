package llm

import (
	"net/http"
	"testing"
	"time"
)

// A refusal names its type, and a wait longer than the retry loop's clamp.
func TestBackpressureKeepsTheErrorTypeAndTheAskedWait(t *testing.T) {
	body := []byte(`{"error":{"message":"refused","type":"budget_exhausted","retry_after":1066299}}`)
	bp, _ := backpressureFrom(http.Header{}, body)
	if bp.Type != "budget_exhausted" {
		t.Errorf("type %q", bp.Type)
	}
	if bp.AskedWait != 1066299*time.Second {
		t.Errorf("asked wait %s", bp.AskedWait)
	}
}
