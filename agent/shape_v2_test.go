package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// shapeTarget is the token ceiling the Shaper reshapes to: budget minus
// headroom, with a fallback when the headroom would meet/exceed the budget.
func TestShapeTarget(t *testing.T) {
	cases := []struct{ budget, headroom, want int }{
		{100000, 0, 90000}, // 0 → default 10k headroom
		{1000, 200, 800},   // explicit headroom
		{1000, -1, 1000},   // headroom disabled → hard budget
		{1000, 5000, 1000}, // headroom ≥ budget → fallback
		{5000, 0, 5000},    // default 10k ≥ budget → fallback
	}
	for _, c := range cases {
		got := ShaperPolicy{BudgetTokens: c.budget, LODHeadroomTokens: c.headroom}.shapeTarget()
		if got != c.want {
			t.Errorf("shapeTarget(budget=%d, headroom=%d) = %d; want %d", c.budget, c.headroom, got, c.want)
		}
	}
}

// A mid-Turn compaction must surface BOTH via the OnCompaction callback and in
// TurnResult.Compactions, and the Turn must report token usage (cumulative
// Total billed + the current Active window).
func TestTurn_SurfacesCompactionAndUsage(t *testing.T) {
	store := &memStore{}
	big := strings.Repeat("lorem ipsum ", 30) // ~360 chars ≈ 90 tokens each
	for i := 0; i < 4; i++ {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: KindAssistant, Content: big, CreatedAt: int64(i + 1),
		})
	}
	store.entries = append(store.entries, Entry{ID: "q", Kind: KindUser, Content: "recap?", CreatedAt: 10})

	// Tiny budget, headroom disabled → LOD can't fit → must compact. The
	// Shaper's runner returns the summary.
	shaper := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 80, PreserveLastMessages: 1, LODTruncateAboveChars: 40, LODHeadroomTokens: -1},
	}
	// The Session's runner returns a reply + usage.
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "here is the recap"}, {Usage: &llm.Usage{PromptTokens: 20, CompletionTokens: 4}}},
	}}

	var cbCompactions []CompactionInfo
	var lastUsage TokenUsage
	var clock int64 = 100
	sess := &Session{
		SessionID:    "s",
		Store:        store,
		Runner:       runner,
		Build:        shaper.Build,
		Now:          func() int64 { clock++; return clock },
		OnCompaction: func(ci CompactionInfo) { cbCompactions = append(cbCompactions, ci) },
		OnUsage:      func(u TokenUsage) { lastUsage = u },
		MaxTurns:     3,
	}
	res, err := sess.Turn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply != "here is the recap" {
		t.Errorf("reply = %q", res.Reply)
	}
	// Surfaced both ways.
	if len(cbCompactions) == 0 || len(res.Compactions) == 0 {
		t.Fatalf("compaction not surfaced: callback=%d result=%d", len(cbCompactions), len(res.Compactions))
	}
	ci := res.Compactions[0]
	if ci.Summary != "SUMMARY" {
		t.Errorf("summary = %q; want SUMMARY", ci.Summary)
	}
	if ci.SubsumedCount != 4 {
		t.Errorf("subsumed = %d; want 4", ci.SubsumedCount)
	}
	if !(ci.TokensBefore > ci.TokensAfter) {
		t.Errorf("expected TokensBefore(%d) > TokensAfter(%d)", ci.TokensBefore, ci.TokensAfter)
	}
	// Usage: Total = billed prompt+completion; Active = current window > 0.
	if res.Usage.Total != 24 {
		t.Errorf("Total = %d; want 24", res.Usage.Total)
	}
	if res.Usage.Active <= 0 {
		t.Errorf("Active = %d; want > 0", res.Usage.Active)
	}
	if lastUsage != res.Usage {
		t.Errorf("OnUsage %+v != TurnResult.Usage %+v", lastUsage, res.Usage)
	}
}
