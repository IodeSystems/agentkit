package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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

// TestBatchRate_ABatchCallIsCheaperThanRealtime pins the "at its rate" half of
// the spec: a batch Usage does not just name the call as batch, it is batch AT
// ITS RATE — for a dialect with a published discount the batch rate is strictly
// below the standard per-token rate, so the same tokens cost less. Fails if
// BatchRate(DialectAnthropic) were raised to 1.0 (a batch flag with no discount).
func TestBatchRate_ABatchCallIsCheaperThanRealtime(t *testing.T) {
	std := 1.0 // the standard per-token rate, as the shim prices it
	if BatchRate(DialectAnthropic) >= BatchRate(DialectAuto) {
		t.Errorf("BatchRate(%q) = %v is not below the standard rate %v",
			DialectAnthropic, BatchRate(DialectAnthropic), std)
	}
	// The same 150 tokens cost less in batch than in real-time.
	tokens := 150
	batchCost := float64(tokens) * BatchRate(DialectAnthropic)
	realtimeCost := float64(tokens) * BatchRate(DialectAuto)
	if batchCost >= realtimeCost {
		t.Errorf("%d tokens: batch cost %v is not below real-time cost %v",
			tokens, batchCost, realtimeCost)
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

// A batch completion is accounted at the batch rate: the shim's standard
// cost (usage.cost in the body) is scaled by the dialect's BatchRate, and the
// Usage is marked batch. A real-time completion for the same tokens is
// byte-identical to the pre-batch value: cost unscaled, Batch false.
func TestBatchCost_ScaledByDialectRate(t *testing.T) {
	const stdCost = 0.02 // the standard per-completion cost in the body
	body := fmt.Sprintf(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"model":"claude-opus-5-5","usage":{"prompt_tokens":1000,"completion_tokens":500,"total_tokens":1500,"cost":%g}}`, stdCost)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	var saw Usage
	c.OnUsage = func(u Usage) { saw = u }

	// Batch: cost is scaled by BatchRate(DialectAnthropic) = 0.5.
	if _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Batch: true}); err != nil {
		t.Fatal(err)
	}
	if !saw.Batch {
		t.Error("batch Usage.Batch = false, want true")
	}
	wantBatch := stdCost * BatchRate(DialectAnthropic)
	if saw.Cost.Amount != wantBatch {
		t.Errorf("batch cost = %v, want %v (0.5 x %v)", saw.Cost.Amount, wantBatch, stdCost)
	}
	if !saw.Cost.Reported {
		t.Error("batch Usage.Cost.Reported = false, want true")
	}

	// Real-time: cost is the standard body cost, unscaled, Batch false.
	saw = Usage{}
	if _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil {
		t.Fatal(err)
	}
	if saw.Batch {
		t.Error("real-time Usage.Batch = true, want false")
	}
	if saw.Cost.Amount != stdCost {
		t.Errorf("real-time cost = %v, want %v (unscaled)", saw.Cost.Amount, stdCost)
	}
	if !saw.Cost.Reported {
		t.Error("real-time Usage.Cost.Reported = false, want true")
	}
}

// A batch completion on a dialect with no published discount (BatchRate 1.0)
// reports the standard cost unscaled, and is still marked batch.
func TestBatchCost_NoDiscountDialect(t *testing.T) {
	const stdCost = 0.03
	body := fmt.Sprintf(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"model":"gpt-4","usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"cost":%g}}`, stdCost)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/v1", "k", "gpt-4")
	var saw Usage
	c.OnUsage = func(u Usage) { saw = u }

	if _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Batch: true}); err != nil {
		t.Fatal(err)
	}
	if !saw.Batch {
		t.Error("batch Usage.Batch = false, want true")
	}
	// No discount: cost is the standard body cost, unscaled.
	if saw.Cost.Amount != stdCost {
		t.Errorf("batch cost = %v, want %v (BatchRate 1.0, unscaled)", saw.Cost.Amount, stdCost)
	}
}
