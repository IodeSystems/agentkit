package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// orphanResults counts tool messages whose call is not in the rendered list: a
// provider rejects the request on the first one.
func orphanResults(msgs []llm.Message) int {
	calls := map[string]bool{}
	n := 0
	for _, m := range msgs {
		for _, tc := range m.ToolCalls {
			calls[tc.ID] = true
		}
		if m.Role == "tool" && !calls[m.ToolCallID] {
			n++
		}
	}
	return n
}

// A fold must not leave a tool result in the kept tail while the call that asked
// for it is summarized away: the provider rejects a tool message with no call.
func TestCompact_NeverSplitsACallFromItsResult(t *testing.T) {
	store := &memStore{entries: []Entry{
		{ID: "u", Kind: KindUser, Content: "go", CreatedAt: 1},
		{ID: "c", Kind: KindToolCall, ToolCallID: "c1", ToolName: "t", Content: "{}", CreatedAt: 2},
		{ID: "r", Kind: KindToolResult, ToolCallID: "c1", ToolName: "t", Content: "out", CreatedAt: 3},
		{ID: "a", Kind: KindAssistant, Content: "done", CreatedAt: 4},
	}}
	sh := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1, PreserveLastToolCalls: 1},
	}
	if _, _, err := sh.Compact(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	entries, _ := store.Context(context.Background(), "s")
	sortEntries(entries)
	msgs := groupMessages("", entries, func(_ int, e Entry) string { return e.Content }, false, ToolFormatNative)
	if n := orphanResults(msgs); n != 0 {
		t.Fatalf("%d orphan tool result(s) after fold: %+v", n, msgs)
	}
}

// A second fold must replace the first marker, not stack beside it: otherwise
// markers accumulate and each new summary restates the previous ones.
func TestCompact_SecondFoldReplacesTheFirstMarker(t *testing.T) {
	store := &memStore{}
	add := func(id string, k EntryKind, at int64) {
		store.entries = append(store.entries, Entry{ID: id, Kind: k, Content: "event " + id, CreatedAt: at})
	}
	add("u1", KindUser, 1)
	add("a1", KindAssistant, 2)
	add("u2", KindUser, 3)
	sh := newCompactShaper(store)
	if _, did, err := sh.Compact(context.Background(), "s"); err != nil || !did {
		t.Fatalf("first fold: did=%v err=%v", did, err)
	}
	add("u3", KindUser, 10)
	add("a3", KindAssistant, 11)
	add("u4", KindUser, 12)
	if _, did, err := sh.Compact(context.Background(), "s"); err != nil || !did {
		t.Fatalf("second fold: did=%v err=%v", did, err)
	}
	if n := markerCount(store); n != 1 {
		t.Fatalf("markers = %d, want 1", n)
	}
}

// A lone marker with nothing new beside it is not a fold: no summarizer call.
func TestCompact_OnlyAMarkerOlderIsNothingToFold(t *testing.T) {
	store := &memStore{entries: []Entry{
		{ID: "m", Kind: KindCompaction, Content: "SUMMARY", CreatedAt: 1},
		{ID: "u", Kind: KindUser, Content: "x", CreatedAt: 2},
	}}
	if _, did, err := newCompactShaper(store).Compact(context.Background(), "s"); err != nil || did {
		t.Fatalf("did=%v err=%v, want no fold", did, err)
	}
}

// A compaction marker is the only record of what it folded: stubbing it for
// length, aged or not, destroys that. A user message is not a tool result and is
// not bounded by MaxToolResultChars.
func TestRender_NeverStubsAMarkerNorBoundsANonToolResult(t *testing.T) {
	long := strings.Repeat("x", 5000)
	entries := []Entry{
		{ID: "m", Kind: KindCompaction, Content: long},
		{ID: "u", Kind: KindUser, Content: long},
		{ID: "a", Kind: KindAssistant, Content: "ok"},
	}
	// Aged under LOD: the marker is older than the tail.
	lod := &Shaper{Policy: ShaperPolicy{LODTruncateAboveChars: 100}}
	if m := lod.render(entries, 2, "", true)[0]; strings.Contains(m.Content, "[truncated") {
		t.Fatalf("marker stubbed by LOD: %.60s", m.Content)
	}
	// Preserved tail under MaxToolResultChars.
	capped := &Shaper{Policy: ShaperPolicy{MaxToolResultChars: 100}}
	for _, m := range capped.render(entries, 3, "", false) {
		if strings.Contains(m.Content, "[truncated") {
			t.Fatalf("%s message stubbed by MaxToolResultChars: %.60s", m.Role, m.Content)
		}
	}
}

// The two TokensBefore numbers describe the same moment; they must agree.
func TestCompact_InfoAndStartReportTheSameTokensBefore(t *testing.T) {
	store := &memStore{}
	for i, k := range []EntryKind{KindUser, KindAssistant, KindUser} {
		store.entries = append(store.entries, Entry{ID: string(rune('a' + i)), Kind: k, Content: strings.Repeat("word ", 50), CreatedAt: int64(i + 1)})
	}
	var start CompactionStart
	ctx := withCompactionStartSink(context.Background(), func(s CompactionStart) { start = s })
	info, did, err := newCompactShaper(store).Compact(ctx, "s")
	if err != nil || !did {
		t.Fatalf("did=%v err=%v", did, err)
	}
	if start.TokensBefore != info.TokensBefore || info.TokensBefore == 0 {
		t.Fatalf("start=%d info=%d", start.TokensBefore, info.TokensBefore)
	}
}

// A multimodal message keeps its body in Parts with Content empty; counting only
// Content made it free and the shaper under-measured the window.
func TestMessageTokens_CountsPartsAndImages(t *testing.T) {
	m := llm.Message{Role: "user", Parts: []llm.ContentPart{llm.TextPart(strings.Repeat("x", 300)), llm.ImagePart("https://x/y.png")}}
	got := messageTokens(CharsByThree{}, m)
	if want := 100 + ImageTokenEstimate; got < want {
		t.Fatalf("tokens = %d, want >= %d", got, want)
	}
	sh := &Shaper{Estimate: CharsByThree{}}
	if sh.tokens([]llm.Message{m}) != got {
		t.Fatal("Shaper.tokens and messageTokens disagree")
	}
	s := &Session{Estimate: CharsByThree{}}
	if s.estimateTokens([]llm.Message{m}) != got {
		t.Fatal("Session.estimateTokens and messageTokens disagree")
	}
}
