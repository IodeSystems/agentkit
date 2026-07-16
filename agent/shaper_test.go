package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/iodesystems/agentkit/llm"
)

// ── classifier ──────────────────────────────────────────────────────

func TestClassifyPristineCount(t *testing.T) {
	cases := []struct {
		name     string
		kinds    []EntryKind
		maxMsgs  int
		maxTools int
		want     int
	}{
		{"empty", nil, 5, 5, 0},
		{
			"all messages under budget",
			[]EntryKind{KindUser, KindAssistant, KindUser, KindAssistant},
			5, 5, 4,
		},
		{
			"messages over budget — stop at budget edge",
			[]EntryKind{
				KindUser, KindAssistant,
				KindUser, KindAssistant,
				KindUser, KindAssistant,
			},
			3, 0, 3,
		},
		{
			"mixed messages + tools — both budgets respected",
			[]EntryKind{KindToolCall, KindAssistant, KindToolResult, KindUser},
			2, 1, 3,
		},
		{
			"compaction markers always pass through",
			[]EntryKind{KindCompaction, KindUser},
			1, 0, 2,
		},
		{
			"return short-circuits on first over-budget walking back",
			[]EntryKind{KindNotification, KindUser, KindAssistant},
			1, 0, 1,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			entries := make([]Entry, len(c.kinds))
			for i, k := range c.kinds {
				entries[i] = Entry{ID: uuid.New().String(), Kind: k}
			}
			got := classifyPristineCount(entries, c.maxMsgs, c.maxTools)
			if got != c.want {
				t.Errorf("classifyPristineCount(%v, msgs=%d, tools=%d) = %d, want %d",
					c.kinds, c.maxMsgs, c.maxTools, got, c.want)
			}
		})
	}
}

// ── LOD stub format ─────────────────────────────────────────────────

// TestLodStub_BackPointsToEvent — the stub must embed the source entry id so
// a future read-truncated-field tool call can recover the original payload.
// This is the "toolcall lifting" mechanism.
func TestLodStub_BackPointsToEvent(t *testing.T) {
	e := Entry{
		ID:      "abc-123",
		Kind:    KindToolResult,
		Content: strings.Repeat("x", 10_000),
	}
	stub := lodStub(e, 4000)

	if !strings.Contains(stub, "event_id=abc-123") {
		t.Errorf("stub missing event_id back-pointer: %q", stub)
	}
	if !strings.Contains(stub, "len=10000") {
		t.Errorf("stub missing original length: %q", stub)
	}
	if len(stub) >= len(e.Content) {
		t.Errorf("stub did not shrink content: %d >= %d", len(stub), len(e.Content))
	}
	if !strings.Contains(stub, "…") {
		t.Errorf("stub missing ellipsis sentinel: %q", stub)
	}
}

// ── manual Compact ──────────────────────────────────────────────────

// markerCount counts KindCompaction entries currently in the store.
func markerCount(s *memStore) int {
	n := 0
	for _, e := range s.entries {
		if e.Kind == KindCompaction {
			n++
		}
	}
	return n
}

func newCompactShaper(store *memStore) *Shaper {
	return &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}, {{Content: "SUMMARY2"}}}},
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1, LODTruncateAboveChars: 40},
	}
}

// Compact folds a built-up history regardless of budget: older entries are
// subsumed into a KindCompaction marker even though the (huge) budget is nowhere
// near saturated.
func TestCompact_FoldsHistory(t *testing.T) {
	store := &memStore{}
	kinds := []EntryKind{KindUser, KindAssistant, KindToolCall, KindToolResult, KindAssistant}
	for i, k := range kinds {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: k, Content: "event content here", CreatedAt: int64(i + 1),
		})
	}
	// pristine tail = last user/assistant msg only (PreserveLastMessages:1) → the
	// trailing KindAssistant. Everything before is foldable.
	store.entries = append(store.entries, Entry{ID: "z", Kind: KindUser, Content: "recap?", CreatedAt: 100})

	sh := newCompactShaper(store)
	info, did, err := sh.Compact(context.Background(), "s")
	if err != nil {
		t.Fatal(err)
	}
	if !did {
		t.Fatal("expected didCompact=true")
	}
	if info.SubsumedCount <= 0 {
		t.Errorf("SubsumedCount = %d; want > 0", info.SubsumedCount)
	}
	if info.Summary != "SUMMARY" {
		t.Errorf("Summary = %q; want SUMMARY", info.Summary)
	}
	if markerCount(store) != 1 {
		t.Fatalf("marker count = %d; want 1", markerCount(store))
	}
	// The folded (pre-pristine) rows must be gone from the store, replaced by
	// the marker.
	for _, e := range store.entries {
		if e.ID == "a" || e.ID == "b" {
			t.Errorf("subsumed entry %s still present after compact", e.ID)
		}
	}
}

// Compact on a session that is nothing but pristine tail returns (zero, false,
// nil) and writes no marker.
func TestCompact_PristineOnly(t *testing.T) {
	store := &memStore{}
	// PreserveLastMessages large enough that everything is pristine.
	store.entries = append(store.entries,
		Entry{ID: "a", Kind: KindUser, Content: "hi", CreatedAt: 1},
		Entry{ID: "b", Kind: KindAssistant, Content: "hello", CreatedAt: 2},
	)
	sh := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 10},
	}
	info, did, err := sh.Compact(context.Background(), "s")
	if err != nil {
		t.Fatal(err)
	}
	if did {
		t.Errorf("expected didCompact=false, got true")
	}
	if info != (CompactionInfo{}) {
		t.Errorf("expected zero CompactionInfo, got %+v", info)
	}
	if markerCount(store) != 0 {
		t.Errorf("marker written on pristine-only session: %d", markerCount(store))
	}
}

// Compacting twice does not re-summarize the existing marker: the second call
// folds only whatever new non-marker rows remain in the older region, and never
// re-subsumes the marker itself.
func TestCompact_TwiceDoesNotRefoldMarker(t *testing.T) {
	store := &memStore{}
	kinds := []EntryKind{KindUser, KindAssistant, KindUser, KindAssistant}
	for i, k := range kinds {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: k, Content: "content", CreatedAt: int64(i + 1),
		})
	}
	store.entries = append(store.entries, Entry{ID: "z", Kind: KindUser, Content: "recap?", CreatedAt: 100})

	sh := newCompactShaper(store)
	if _, did, err := sh.Compact(context.Background(), "s"); err != nil || !did {
		t.Fatalf("first compact: did=%v err=%v", did, err)
	}
	if markerCount(store) != 1 {
		t.Fatalf("after first compact marker count = %d; want 1", markerCount(store))
	}

	// Second call: only the marker + pristine tail remain in the older region;
	// the marker must NOT be re-summarized → didCompact=false, still one marker.
	info, did, err := sh.Compact(context.Background(), "s")
	if err != nil {
		t.Fatalf("second compact err: %v", err)
	}
	if did {
		t.Errorf("second compact re-folded: did=true info=%+v", info)
	}
	if markerCount(store) != 1 {
		t.Errorf("marker count after second compact = %d; want 1", markerCount(store))
	}
}

// ── AlwaysLOD ───────────────────────────────────────────────────────

// AlwaysLOD forces LOD truncation of oversized older entries on every Build even
// when the context is well under budget; without it, the same under-budget Build
// leaves the entry pristine.
func TestAlwaysLOD(t *testing.T) {
	big := strings.Repeat("x", 5000)
	seed := func() *memStore {
		s := &memStore{}
		s.entries = append(s.entries,
			Entry{ID: "a", Kind: KindToolResult, Content: big, ToolCallID: "tc1", CreatedAt: 1},
			Entry{ID: "b", Kind: KindUser, Content: "and now?", CreatedAt: 2},
		)
		return s
	}
	// PreserveLastMessages:1 keeps only "b" pristine; "a" is older → LOD-eligible.
	// Huge budget so no compaction fires.
	pol := ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1, LODTruncateAboveChars: 100}

	// Without AlwaysLOD: under budget → pristine, full content survives.
	shOff := &Shaper{Store: seed(), Policy: pol}
	offMsgs, err := shOff.Build(context.Background(), "s", "")
	if err != nil {
		t.Fatal(err)
	}
	if !containsFull(offMsgs, big) {
		t.Errorf("AlwaysLOD=false: expected full oversized content to survive under budget")
	}

	// With AlwaysLOD: truncated even under budget.
	polOn := pol
	polOn.AlwaysLOD = true
	shOn := &Shaper{Store: seed(), Policy: polOn}
	onMsgs, err := shOn.Build(context.Background(), "s", "")
	if err != nil {
		t.Fatal(err)
	}
	if containsFull(onMsgs, big) {
		t.Errorf("AlwaysLOD=true: oversized older entry was NOT truncated")
	}
	if !anyContains(onMsgs, "[truncated event_id=a") {
		t.Errorf("AlwaysLOD=true: expected LOD stub back-pointer for entry a")
	}
}

func containsFull(msgs []llm.Message, s string) bool {
	return anyContains(msgs, s)
}

func anyContains(msgs []llm.Message, s string) bool {
	for _, m := range msgs {
		if strings.Contains(m.Content, s) {
			return true
		}
	}
	return false
}
