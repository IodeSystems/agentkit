package llm

import (
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
