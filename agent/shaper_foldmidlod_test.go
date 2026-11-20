package agent

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// midLODShaper: dun's policy on a session whose results are above
// LODTruncateAboveChars, so a stub is visible.
func midLODShaper(mid bool) (*memStore, *Shaper) {
	st := reportedSession(12_000)
	sh := reportedShaper(st, 55)
	sh.Policy.FoldMidLOD = mid
	sh.Policy.BudgetTokens = 100_000 // stubbed history alone is over target: it folds
	return st, sh
}

func stubbed(m llm.Message) bool {
	return m.Role == "tool" && strings.HasPrefix(m.Content, "[truncated event_id=")
}

// foldThenBuild folds once, then builds again under target and fails if that
// second build folds too.
func foldThenBuild(t *testing.T, sh *Shaper) (CompactionInfo, []llm.Message) {
	t.Helper()
	_, infos := buildWithInfo(t, sh, "")
	if len(infos) != 1 {
		t.Fatalf("%d folds, want 1", len(infos))
	}
	msgs, again := buildWithInfo(t, sh, "")
	if len(again) != 0 {
		t.Fatalf("the build after the fold folded again (%d)", len(again))
	}
	return infos[0], msgs
}

// After a fold, the kept band is full of results that were sent whole. With
// FoldMidLOD the under-target build stubs the old ones; the pristine tail (the
// result the session is about to act on) stays verbatim.
func TestFoldMidLOD_StubsOldResultsUnderTargetButNotThePristineTail(t *testing.T) {
	_, sh := midLODShaper(true)
	_, msgs := foldThenBuild(t, sh)
	var tools []llm.Message
	for _, m := range msgs {
		if m.Role == "tool" {
			tools = append(tools, m)
		}
	}
	n := 0
	for _, m := range tools {
		if stubbed(m) {
			n++
		}
	}
	if n == 0 {
		t.Fatal("no old tool result was stubbed under target after the fold")
	}
	for _, m := range tools[len(tools)-2:] { // PreserveLastToolCalls 4: two pairs
		if stubbed(m) {
			t.Fatalf("a pristine-tail result was stubbed: %.60q", m.Content)
		}
	}
	if k := orphanResults(msgs); k != 0 {
		t.Fatalf("%d orphan tool result(s): stubs must keep calls with their results", k)
	}
}

// Off, nothing changes: the same fold's next build sends every result whole,
// and with FoldMidLOD on but no marker in the context the build is identical.
func TestFoldMidLOD_OffOrBeforeAnyFoldIsTheOldRender(t *testing.T) {
	_, sh := midLODShaper(false)
	_, msgs := foldThenBuild(t, sh)
	for _, m := range msgs {
		if stubbed(m) {
			t.Fatalf("FoldMidLOD off stubbed a result under target: %.60q", m.Content)
		}
	}

	small := func(mid bool) []llm.Message {
		st := reportedSession(12_000)
		st.entries = st.entries[:40] // under target, no fold
		sh := reportedShaper(st, 55)
		sh.Policy.FoldMidLOD = mid
		msgs, infos := buildWithInfo(t, sh, "sys")
		if len(infos) != 0 {
			t.Fatal("folded under target")
		}
		return msgs
	}
	if !reflect.DeepEqual(small(false), small(true)) {
		t.Fatal("with no marker in the context, FoldMidLOD changed the render")
	}
}

// The boundary is stepped, so the prefix the cache holds breaks once per
// LODStep entries, not every round as the pristine tail moves: while the
// boundary stays where it is, each round's messages are a prefix of the next
// round's.
func TestFoldMidLOD_BoundaryIsSteppedSoTheCachedPrefixHolds(t *testing.T) {
	st, sh := midLODShaper(true)
	foldThenBuild(t, sh)
	boundary := func() int {
		es := append([]Entry(nil), st.entries...)
		sortEntries(es)
		return sh.Policy.lodBoundary(len(es) - classifyPristineCount(es, 6, 4))
	}
	prev, _ := buildWithInfo(t, sh, "")
	prevB := boundary()
	held, stepped := 0, 0
	at := int64(0) // the store keeps the marker last; the newest is the max
	for _, e := range st.entries {
		at = max(at, e.CreatedAt)
	}
	for round := 0; round < 10; round++ {
		id := fmt.Sprintf("n%d", round)
		for _, e := range []Entry{
			{Kind: KindAssistant, Content: "next " + strings.Repeat("t", 200)},
			{Kind: KindToolCall, ToolCallID: id, ToolName: "exec", Content: `{"command":"x"}`},
			{Kind: KindToolResult, ToolCallID: id, ToolName: "exec", Content: strings.Repeat("r", 6000)},
		} {
			at++
			e.CreatedAt, e.ID = at, fmt.Sprintf("z%04d", at)
			st.entries = append(st.entries, e)
		}
		msgs, infos := buildWithInfo(t, sh, "")
		if len(infos) != 0 {
			t.Fatalf("round %d folded; the test needs every round under target", round)
		}
		b := boundary()
		if b == prevB {
			if len(msgs) < len(prev) || !reflect.DeepEqual(msgs[:len(prev)], prev) {
				t.Fatalf("round %d: boundary %d did not move, yet the prefix changed", round, b)
			}
			held++
		} else {
			stepped++
		}
		prev, prevB = msgs, b
	}
	t.Logf("%d rounds kept the prefix, %d stepped", held, stepped)
	if held == 0 || stepped == 0 {
		t.Fatalf("held %d, stepped %d: the test must see both", held, stepped)
	}
}

// The band is sized with the render the next build sends; with old results
// stubbed it holds more of them.
func TestFoldMidLOD_FoldKeepPctKeepsMoreEntries(t *testing.T) {
	_, off := midLODShaper(false)
	_, on := midLODShaper(true)
	a, _ := foldThenBuild(t, off)
	b, _ := foldThenBuild(t, on)
	t.Logf("kept off %d (%d tok after), on %d (%d tok after)", a.KeptCount, a.TokensAfter, b.KeptCount, b.TokensAfter)
	if b.KeptCount <= a.KeptCount {
		t.Fatalf("kept %d with FoldMidLOD, %d without: the band did not widen", b.KeptCount, a.KeptCount)
	}
}

// Over random sessions with FoldMidLOD on, the fold's numbers still describe
// the next build: TokensAfter is what it sends, the parts sum to it, and the
// band never leaves the next build over target.
func TestFoldMidLOD_TokensAfterIsTheNextBuildOnRandomSessions(t *testing.T) {
	for seed := int64(0); seed < 150; seed++ {
		r := rand.New(rand.NewSource(seed))
		st := randSession(r)
		sh := reportedShaper(st, 55)
		sh.Policy.FoldMidLOD = true
		sh.Policy.BudgetTokens = 60_000 + r.Intn(200_000)
		sh.Policy.FoldUserIntoToolResult = r.Intn(2) == 0
		sh.Policy.PinFirstUser = r.Intn(2) == 0
		_, infos := buildWithInfo(t, sh, "")
		if len(infos) == 0 {
			continue
		}
		ci := infos[len(infos)-1]
		next, again := buildWithInfo(t, sh, "")
		if len(again) != 0 {
			t.Errorf("seed %d: the build after the fold folded again", seed)
			continue
		}
		if got := sh.tokens(next); got != ci.TokensAfter {
			t.Errorf("seed %d: TokensAfter %d, next build sends %d", seed, ci.TokensAfter, got)
		}
		if got := ci.SummaryTokens + ci.PinnedTokens + ci.KeptTokens; got != ci.TokensAfter {
			t.Errorf("seed %d: parts %d, TokensAfter %d", seed, got, ci.TokensAfter)
		}
	}
}
