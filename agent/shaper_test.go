package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

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
				entries[i] = Entry{ID: NewID(), Kind: k}
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
	stub := lodStub(e, 4000, LODHead, "", "")

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

// An unset BudgetTokens means "nobody told me the window size", NOT "the window
// is zero tokens". Read the second way, every Build is over budget and compacts
// — measured in the wild as 45 compactions in 29 minutes, one per turn, each
// spending a summarization call to fold history that had just been folded, and
// leaving 5 of 50 entries alive.
func TestBuild_ZeroBudgetNeverCompacts(t *testing.T) {
	store := &memStore{}
	kinds := []EntryKind{KindUser, KindAssistant, KindToolCall, KindToolResult, KindAssistant}
	for i, k := range kinds {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: k, Content: strings.Repeat("event content here ", 50),
			CreatedAt: int64(i + 1),
		})
	}
	sh := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 0, PreserveLastMessages: 1, LODTruncateAboveChars: 40},
	}
	for turn := 0; turn < 3; turn++ {
		if _, err := sh.Build(context.Background(), "s", "sys"); err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		if n := markerCount(store); n != 0 {
			t.Fatalf("turn %d: unbudgeted Build compacted (%d markers)", turn, n)
		}
	}
	// And the history is rendered whole, not folded away.
	msgs, err := sh.Build(context.Background(), "s", "sys")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) < len(kinds) {
		t.Errorf("history was dropped: %d messages for %d entries", len(msgs), len(kinds))
	}
}

// A real budget still compacts — the fix must not disable the feature.
func TestBuild_RealBudgetStillCompacts(t *testing.T) {
	store := &memStore{}
	for i := 0; i < 6; i++ {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: KindUser, Content: strings.Repeat("x", 4000),
			CreatedAt: int64(i + 1),
		})
	}
	sh := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}, {{Content: "SUMMARY2"}}}},
		Policy: ShaperPolicy{BudgetTokens: 500, LODHeadroomTokens: -1, PreserveLastMessages: 1,
			LODTruncateAboveChars: 100},
	}
	if _, err := sh.Build(context.Background(), "s", "sys"); err != nil {
		t.Fatal(err)
	}
	if markerCount(store) == 0 {
		t.Error("a genuinely over-budget session should still compact")
	}
}

// summarizeErr must carry the turn's budget cause out so the jsonl says WHY the
// fold died, not just "context canceled". A named cause (errTurnBudget in
// dun) reads as the real reason; a bare context.Canceled cause (ctrl-C, or the
// parent going down) says so instead of masquerading as a transient fault.
func TestSummarizeErr_CarriesCause(t *testing.T) {
	// Named cause: the budget clock set it.
	causeErr := fmt.Errorf("the turn ran past its 30m budget")
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(context.Canceled)
	cancel(causeErr)
	got := summarizeErr(ctx, ctx.Err())
	if !strings.Contains(got.Error(), "compaction aborted") {
		t.Errorf("want 'compaction aborted', got %q", got)
	}
	if !strings.Contains(got.Error(), "30m budget") {
		t.Errorf("want the cause named, got %q", got)
	}

	// Bare Canceled cause: says it's ctrl-C / parent-down, not a fault.
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	cancel2()
	got2 := summarizeErr(ctx2, ctx2.Err())
	if !strings.Contains(got2.Error(), "no turn-budget cause") {
		t.Errorf("want 'no turn-budget cause', got %q", got2)
	}

	// A non-cancel error is returned unchanged (a real 500 keeps its message).
	ctx3, cancel3 := context.WithCancel(context.Background())
	defer cancel3()
	plain := fmt.Errorf("upstream 500")
	if got3 := summarizeErr(ctx3, plain); got3 != plain {
		t.Errorf("non-cancel error must pass through unchanged, got %v", got3)
	}
}

// The Pause hook must be called around the compaction LLM call and resumed
// afterwards, so a slow endpoint + a long fold cannot burn the turn's budget.
func TestSummarize_PausesClock(t *testing.T) {
	store := &memStore{}
	kinds := []EntryKind{KindUser, KindAssistant, KindToolCall, KindToolResult, KindAssistant}
	for i, k := range kinds {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: k, Content: "event content here", CreatedAt: int64(i + 1),
		})
	}
	store.entries = append(store.entries, Entry{ID: "z", Kind: KindUser, Content: "recap?", CreatedAt: 100})

	var paused, resumed bool
	sh := newCompactShaper(store)
	sh.Pause = func() func() {
		paused = true
		return func() { resumed = true }
	}
	if _, _, err := sh.Compact(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	if !paused {
		t.Error("Pause was not called around the compaction LLM call")
	}
	if !resumed {
		t.Error("the resume func was not called after the compaction")
	}
}

// A tool that answers at the BOTTOM keeps its bottom.
//
// `lodStub` kept a head only, so a truncated test suite lost the one line that
// says whether it passed: suites put first failures at the top and the verdict
// at the end. Recorded as a known tradeoff on 2026-09-06 and deliberately not
// gold-plated then — "a head+tail option on lodStub would close it".
func TestATailAnsweringToolKeepsItsVerdict(t *testing.T) {
	body := "FAIL TestOne\n" + strings.Repeat("noise\n", 4000) + "SUITE FAILED: 1 of 900\n"
	e := Entry{ID: "e1", Kind: KindToolResult, ToolName: "test.run", Content: body}

	if head := lodStub(e, 2000, LODHead, "test.run", ""); strings.Contains(head, "SUITE FAILED") {
		t.Fatal("the fixture does not discriminate: the verdict is inside the head")
	}
	ends := lodStub(e, 2000, LODEnds, "test.run", "")
	if !strings.Contains(ends, "SUITE FAILED: 1 of 900") {
		t.Errorf("the verdict was truncated away: %q", last(ends, 200))
	}
	if !strings.Contains(ends, "FAIL TestOne") {
		t.Errorf("the first failure was dropped; both ends is the point: %q", ends[:200])
	}
	// The elision is COUNTED, so a model can tell a gap from a join.
	if !strings.Contains(ends, "characters") {
		t.Errorf("the stub does not say how much it removed: %q", ends)
	}

	tail := lodStub(e, 2000, LODTail, "test.run", "")
	if !strings.Contains(tail, "SUITE FAILED") {
		t.Errorf("a tail-keeping stub lost the tail: %q", tail)
	}
}

// A stub says what the call ASKED, not only what came back.
//
// The head of a truncated result is the top of whatever the tool returned — for
// a file its first lines, for a listing an arbitrary few — and never why the
// session went looking. The arguments answer that for nothing: the model
// already wrote them and they are already in the log.
func TestAStubCarriesTheCallThatProducedIt(t *testing.T) {
	e := Entry{ID: "e2", Kind: KindToolResult, ToolName: "dir.grep",
		Content: strings.Repeat("match\n", 5000)}
	stub := lodStub(e, 1000, LODHead, "dir.grep", `{"pattern":"func TestD1"}`)

	if !strings.Contains(stub, "dir.grep") {
		t.Errorf("the stub does not name the tool: %q", stub)
	}
	if !strings.Contains(stub, "func TestD1") {
		t.Errorf("the stub does not say what was asked, which is the whole point: %q", stub)
	}
	if !strings.Contains(stub, "event_id=e2") {
		t.Errorf("the back-pointer was lost: %q", stub)
	}
}

func last(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
