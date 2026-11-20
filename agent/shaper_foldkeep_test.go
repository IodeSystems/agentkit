package agent

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// sumRunner answers every fold with a summary of n characters and records the
// requests, so a test can compare what two folds asked the summarizer.
type sumRunner struct {
	n    int
	reqs [][]llm.Message
}

func (r *sumRunner) ChatStream(_ context.Context, msgs []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	r.reqs = append(r.reqs, append([]llm.Message(nil), msgs...))
	ch := make(chan llm.StreamChunk, 2)
	ch <- llm.StreamChunk{Content: "SUMMARY " + strings.Repeat("s", r.n)}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

// reportedSession: 434 entries, ~281k tokens.
func reportedSession(resultChars int) *memStore {
	st := &memStore{}
	at := int64(1)
	add := func(e Entry) {
		e.CreatedAt = at
		e.ID = fmt.Sprintf("e%04d", at)
		at++
		st.entries = append(st.entries, e)
	}
	add(Entry{Kind: KindUser, Content: "do the thing"})
	for i := 0; len(st.entries) < 434; i++ {
		id := fmt.Sprintf("c%d", i)
		add(Entry{Kind: KindToolCall, ToolCallID: id, ToolName: "exec", Content: `{"command":"` + strings.Repeat("a", 590) + `"}`})
		add(Entry{Kind: KindToolResult, ToolCallID: id, ToolName: "exec", Content: fmt.Sprintf("r%d ", i) + strings.Repeat("r", resultChars)})
		add(Entry{Kind: KindAssistant, Content: fmt.Sprintf("a%d ", i) + strings.Repeat("t", 1880)})
	}
	return st
}

// reportedShaper: a 322k window, budget = window - 34,816 (dun's reserve).
func reportedShaper(st *memStore, pct int) *Shaper {
	return &Shaper{
		Store:  st,
		Runner: &sumRunner{n: 8000 * 3},
		Policy: ShaperPolicy{
			BudgetTokens: 322_000 - 34_816, PreserveLastMessages: 6, PreserveLastToolCalls: 4,
			LODTruncateAboveChars: 4000, MaxToolResultChars: DefaultMaxToolResultChars,
			PinLatestUser: true, FoldKeepPct: pct,
		},
	}
}

func buildWithInfo(t *testing.T, sh *Shaper, system string) ([]llm.Message, []CompactionInfo) {
	t.Helper()
	var infos []CompactionInfo
	ctx := context.WithValue(context.Background(), compactionSinkKey{}, func(ci CompactionInfo) { infos = append(infos, ci) })
	msgs, err := sh.Build(ctx, "s", system)
	if err != nil {
		t.Fatal(err)
	}
	return msgs, infos
}

func TestFoldKeepPct_ReportedCase(t *testing.T) {
	system := strings.Repeat("y", 30_000) // ~10k tokens of prefix
	for _, pct := range []int{0, 55} {
		st := reportedSession(3990)
		sh := reportedShaper(st, pct)
		_, target := sh.limits(context.Background())
		newest := st.entries[len(st.entries)-1].Content
		msgs, infos := buildWithInfo(t, sh, system)
		if len(infos) != 1 {
			t.Fatalf("pct %d: %d folds, want 1", pct, len(infos))
		}
		ci := infos[0]
		after := sh.tokens(msgs)
		t.Logf("pct %d: target %d; folded %d entries: %d -> %d tokens (%d%% of target; info after %d), kept %d entries (%d tokens)",
			pct, target, ci.SubsumedCount, ci.TokensBefore, after, after*100/target, ci.TokensAfter, ci.KeptCount, ci.KeptTokens)
		if ci.TokensBefore <= target {
			t.Fatalf("before %d not over target %d", ci.TokensBefore, target)
		}
		if orphanResults(msgs) != 0 {
			t.Fatal("orphan tool result")
		}
		if !containsFull(msgs, newest) {
			t.Fatal("newest entry not verbatim")
		}
		if pct == 55 {
			if after < target*50/100 || after > target*60/100 {
				t.Fatalf("after %d is %d%% of target, want 50-60%%", after, after*100/target)
			}
			// No loop: the next builds do not fold until the prompt grows past target.
			for i := 0; i < 3; i++ {
				if _, again := buildWithInfo(t, sh, system); len(again) != 0 {
					t.Fatalf("build %d folded again", i)
				}
			}
		}
	}
}

func TestFoldKeepPct_HugeNewestResultFallsBack(t *testing.T) {
	st := reportedSession(3990)
	// The newest result alone is over the band (MaxToolResultChars off).
	for i := len(st.entries) - 1; i >= 0; i-- {
		if st.entries[i].Kind == KindToolResult {
			st.entries[i].Content = strings.Repeat("h", 3*170_000)
			break
		}
	}
	sh := reportedShaper(st, 55)
	sh.Policy.MaxToolResultChars = 0
	_, target := sh.limits(context.Background())
	floor := classifyPristineCount(st.entries, 6, 4)
	msgs, infos := buildWithInfo(t, sh, "")
	if len(infos) != 1 {
		t.Fatalf("%d folds, want 1", len(infos))
	}
	after := sh.tokens(msgs)
	if infos[0].KeptCount != floor || after > target || orphanResults(msgs) != 0 {
		t.Fatalf("kept %d (floor %d), after %d over target %d, or orphans", infos[0].KeptCount, floor, after, target)
	}
}

// With a band that holds no more than the pristine tail, the fold is the one
// the policy made without FoldKeepPct.
func TestFoldKeepPct_BandNoWiderThanPristineTailFoldsAsToday(t *testing.T) {
	mk := func(pct int) CompactionInfo {
		st := reportedSession(3990)
		st.entries = st.entries[:40]
		sh := reportedShaper(st, pct)
		sh.Policy.BudgetTokens = 30_000 // target 20k: the band holds less than the floor
		_, infos := buildWithInfo(t, sh, "")
		if len(infos) == 0 {
			t.Fatal("no fold")
		}
		return infos[0]
	}
	a, b := mk(0), mk(55)
	if a.SubsumedCount != b.SubsumedCount || a.KeptCount != b.KeptCount {
		t.Fatalf("pct 0 %+v, pct 55 %+v", a, b)
	}
}

// 0 must not change a fold: same entries subsumed, same request to the
// summarizer, and the pristine tail is what stays.
func TestFoldKeepPct_ZeroFoldsByteIdenticalToUnset(t *testing.T) {
	run := func(set bool) (CompactionInfo, []string, [][]llm.Message) {
		st := reportedSession(3990)
		sh := reportedShaper(st, 0)
		if !set {
			sh.Policy = ShaperPolicy{
				BudgetTokens: sh.Policy.BudgetTokens, PreserveLastMessages: 6, PreserveLastToolCalls: 4,
				LODTruncateAboveChars: 4000, MaxToolResultChars: DefaultMaxToolResultChars, PinLatestUser: true,
			}
		}
		_, infos := buildWithInfo(t, sh, "sys")
		if len(infos) != 1 {
			t.Fatalf("%d folds, want 1", len(infos))
		}
		kept, _ := st.Context(context.Background(), "s")
		var ids []string
		for _, e := range kept {
			if e.Kind == KindCompaction {
				ids = append(ids, "marker:"+e.Content) // the marker's ID is random
				continue
			}
			ids = append(ids, e.ID)
		}
		return infos[0], ids, sh.Runner.(*sumRunner).reqs
	}
	ai, aids, areq := run(true)
	bi, bids, breq := run(false)
	floor := classifyPristineCount(reportedSession(3990).entries, 6, 4)
	if ai.KeptCount != floor || bi.KeptCount != floor {
		t.Fatalf("kept %d / %d, want the pristine tail %d", ai.KeptCount, bi.KeptCount, floor)
	}
	if ai.SubsumedCount != bi.SubsumedCount || !reflect.DeepEqual(aids, bids) || !reflect.DeepEqual(areq, breq) {
		t.Fatal("FoldKeepPct 0 folded differently from the field unset")
	}
}

// The kept part, the summary at its cap and the prefix stay inside the band:
// the band is what keeps the next build from folding again at once.
func TestFoldKeepPct_KeptPlusSummaryCapStaysWithinBand(t *testing.T) {
	const pct = 55
	system := strings.Repeat("y", 30_000)
	st := reportedSession(3990)
	sh := reportedShaper(st, pct)
	sh.Estimate = Default()
	_, target := sh.limits(context.Background())
	prefix := 0
	{
		all := append([]Entry(nil), st.entries...)
		sortEntries(all)
		floor := classifyPristineCount(all, 6, 4)
		before := sh.tokens(sh.render(all, floor, system, false))
		prefix = before - sh.tokens(sh.render(all, len(all), "", false))
	}
	_, infos := buildWithInfo(t, sh, system)
	if len(infos) != 1 {
		t.Fatalf("%d folds, want 1", len(infos))
	}
	band := target * pct / 100
	if got := prefix + SummaryCap(SummaryMaxTokens) + infos[0].KeptTokens; got > band {
		t.Fatalf("prefix %d + summary cap %d + kept %d = %d over the band %d", prefix, SummaryCap(SummaryMaxTokens), infos[0].KeptTokens, got, band)
	}
	if infos[0].KeptCount <= classifyPristineCount(st.entries, 6, 4) {
		t.Fatalf("kept %d, no more than the pristine tail", infos[0].KeptCount)
	}
}

// Words typed while a tool ran sit between its call and its result. A suffix
// that begins at that user entry would keep the result and fold the call; the
// widening must pull the call in at every band size.
func TestFoldKeepPct_WidenedSuffixNeverOrphansAResult(t *testing.T) {
	for pct := 20; pct <= 90; pct += 5 {
		st := &memStore{}
		at := int64(1)
		add := func(e Entry) {
			e.CreatedAt = at
			e.ID = fmt.Sprintf("e%04d", at)
			at++
			st.entries = append(st.entries, e)
		}
		add(Entry{Kind: KindUser, Content: "go"})
		for i := 0; i < 60; i++ {
			id := fmt.Sprintf("c%d", i)
			add(Entry{Kind: KindToolCall, ToolCallID: id, ToolName: "t", Content: "{}"})
			add(Entry{Kind: KindUser, Content: fmt.Sprintf("aside %d %s", i, strings.Repeat("u", 400))})
			add(Entry{Kind: KindToolResult, ToolCallID: id, ToolName: "t", Content: strings.Repeat("r", 3000)})
		}
		sh := &Shaper{
			Store: st, Runner: &sumRunner{n: 300},
			Policy: ShaperPolicy{BudgetTokens: 30_000, PreserveLastMessages: 1, PreserveLastToolCalls: 1, FoldKeepPct: pct},
		}
		msgs, infos := buildWithInfo(t, sh, "")
		if len(infos) == 0 {
			t.Fatalf("pct %d: no fold", pct)
		}
		if n := orphanResults(msgs); n != 0 {
			t.Fatalf("pct %d: %d orphan tool result(s): %v", pct, n, roles(msgs))
		}
	}
}

// TokensAfter was measured with LOD forced, so a long tool result that LOD
// shortens made it smaller than what the next build sent (dun: 90,127
// reported, 97.8k sent). It must equal the next build's prompt tokens, with
// FoldMidLOD (the next build stubs its older part) and without.
func TestTokensAfterEqualsNextBuildAfterFoldWithLongToolResult(t *testing.T) {
	for _, mid := range []bool{false, true} {
		st := reportedSession(8000)  // above LODTruncateAboveChars
		sh := reportedShaper(st, 55) // keeps entries beyond the pristine tail
		sh.Policy.BudgetTokens = 100_000
		sh.Policy.FoldMidLOD = mid
		_, infos := buildWithInfo(t, sh, "")
		if len(infos) != 1 {
			t.Fatalf("mid %v: %d folds, want 1", mid, len(infos))
		}
		next, err := sh.Build(context.Background(), "s", "")
		if err != nil {
			t.Fatal(err)
		}
		if got := sh.tokens(next); got != infos[0].TokensAfter {
			t.Fatalf("mid %v: TokensAfter %d, next build sends %d", mid, infos[0].TokensAfter, got)
		}
	}
}
