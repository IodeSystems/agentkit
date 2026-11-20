package llm

import (
	"encoding/json"
	"testing"
)

func TestBatchRate_ClaudeIsHalfPrice(t *testing.T) {
	if got := BatchRate(DialectAnthropic); got != 0.5 {
		t.Errorf("BatchRate(%q) = %v, want 0.5", DialectAnthropic, got)
	}
}

func TestBatchRate_UnknownDialectIsStandardPrice(t *testing.T) {
	for _, d := range []string{DialectLlamaCpp, DialectOpenRouter, DialectOpenAI, DialectAuto, "unknown"} {
		if got := BatchRate(d); got != 1.0 {
			t.Errorf("BatchRate(%q) = %v, want 1.0", d, got)
		}
	}
}

func TestUsage_BatchedDistinguishesBatchFromRealtime(t *testing.T) {
	// A batch completion and a real-time completion for the same parts.
	batchUsage := Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
		Batch:            true,
	}
	realtimeUsage := Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	if !batchUsage.Batched() {
		t.Error("batch Usage.Batched() = false, want true")
	}
	if realtimeUsage.Batched() {
		t.Error("real-time Usage.Batched() = true, want false")
	}

	// The batch field names the call as batch at its rate.
	if !batchUsage.Batch {
		t.Error("batch Usage.Batch = false, want true")
	}
	if realtimeUsage.Batch {
		t.Error("real-time Usage.Batch = true, want false")
	}

	// A caller can compute the batch cost from the standard rate.
	stdCost := 0.01 // arbitrary standard per-token rate
	batchCost := stdCost * BatchRate(DialectAnthropic)
	if batchCost != 0.005 {
		t.Errorf("batch cost = %v, want 0.005 (50%% of %v)", batchCost, stdCost)
	}
}

// TestUsage_BatchSurvivesJSONRoundTrip pins the reading path: Usage reaches a
// caller decoded from the wire (the OnUsage callback and the final
// StreamChunk are JSON-decoded), so the field must survive that round-trip and
// keep the two distinguishable by it. A batch Usage marshals with "batch":true;
// a real-time Usage omits the key entirely (omitempty), and decoding both
// leaves batchUsage distinguishable from realtimeUsage.
func TestUsage_BatchSurvivesJSONRoundTrip(t *testing.T) {
	batch := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, Batch: true}
	realtime := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}

	batchJSON, err := json.Marshal(batch)
	if err != nil {
		t.Fatalf("marshal batch: %v", err)
	}
	realtimeJSON, err := json.Marshal(realtime)
	if err != nil {
		t.Fatalf("marshal real-time: %v", err)
	}

	var backBatch, backRealtime Usage
	if err := json.Unmarshal(batchJSON, &backBatch); err != nil {
		t.Fatalf("unmarshal batch: %v", err)
	}
	if err := json.Unmarshal(realtimeJSON, &backRealtime); err != nil {
		t.Fatalf("unmarshal real-time: %v", err)
	}

	if !backBatch.Batched() {
		t.Error("batch Usage lost Batch after JSON round-trip")
	}
	if backRealtime.Batched() {
		t.Error("real-time Usage gained Batch after JSON round-trip")
	}
	// The field is present in the batch wire form and absent in the real-time
	// one — that asymmetry is what distinguishes the two on the wire.
	if !jsonContainsKey(t, batchJSON, "batch") {
		t.Error("batch Usage JSON does not name \"batch\"")
	}
	if jsonContainsKey(t, realtimeJSON, "batch") {
		t.Error("real-time Usage JSON names \"batch\" (want omitted)")
	}
}

// jsonContainsKey reports whether raw is a JSON object that has the named key
// at the top level.
func jsonContainsKey(t *testing.T, raw []byte, key string) bool {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal object: %v", err)
	}
	_, ok := m[key]
	return ok
}
