package agent

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func randSession(r *rand.Rand) *memStore {
	st := &memStore{}
	at := int64(1)
	add := func(e Entry) {
		e.CreatedAt = at
		if r.Intn(10) != 0 {
			at++
		}
		e.ID = fmt.Sprintf("e%05d", len(st.entries))
		st.entries = append(st.entries, e)
	}
	add(Entry{Kind: KindUser, Content: "do the thing " + strings.Repeat("u", r.Intn(4000))})
	for c := 0; len(st.entries) < 300+r.Intn(300); c++ {
		switch r.Intn(12) {
		case 0:
			add(Entry{Kind: KindUser, Content: "more " + strings.Repeat("u", r.Intn(3000))})
			continue
		case 1:
			add(Entry{Kind: KindNotification, Tag: "n", Content: strings.Repeat("n", r.Intn(3000))})
			continue
		}
		add(Entry{Kind: KindAssistant, Content: strings.Repeat("t", r.Intn(2000))})
		n := 1 + r.Intn(3)
		var ids []string
		for k := 0; k < n; k++ {
			id := fmt.Sprintf("c%d_%d", c, k)
			ids = append(ids, id)
			add(Entry{Kind: KindToolCall, ToolCallID: id, ToolName: "exec", Content: `{"command":"` + strings.Repeat("a", r.Intn(3000)) + `"}`})
		}
		if r.Intn(6) == 0 {
			add(Entry{Kind: KindUser, Content: "aside " + strings.Repeat("u", r.Intn(500))})
		}
		for _, id := range ids {
			sz := r.Intn(12000)
			if r.Intn(15) == 0 {
				sz = 70_000 + r.Intn(100_000) // above MaxToolResultChars
			}
			add(Entry{Kind: KindToolResult, ToolCallID: id, ToolName: "exec", Content: strings.Repeat("r", sz)})
		}
	}
	return st
}

// With dun's policy (FoldKeepPct 55, LOD 4000, MaxToolResultChars 64k, pins,
// FoldUserIntoToolResult either way) and results above both cut-offs, the two
// renders share one basis: after = marker + pinned + kept, and the next build
// sends exactly TokensAfter. Guards that the render basis is NOT the cause of
// dun's "kept 54.4k, after 48.6k".
func TestFold_KeptTokensNeverExceedsTokensAfterOnOneStore(t *testing.T) {
	folds, widened := 0, 0
	for seed := int64(0); seed < 300; seed++ {
		r := rand.New(rand.NewSource(seed))
		st := randSession(r)
		sh := reportedShaper(st, 55)
		sh.Policy.BudgetTokens = 60_000 + r.Intn(200_000)
		sh.Policy.FoldUserIntoToolResult = r.Intn(2) == 0
		sh.Policy.PinFirstUser = r.Intn(2) == 0
		floor := classifyPristineCount(st.entries, 6, 4)
		_, infos := buildWithInfo(t, sh, strings.Repeat("y", r.Intn(30000)))
		next, _ := sh.Build(context.Background(), "s", "")
		for k, ci := range infos {
			folds++
			if ci.KeptCount > floor {
				widened++
			}
			if ci.TokensAfter < ci.KeptTokens {
				t.Errorf("seed %d fold %d: kept %d tokens > after %d", seed, k, ci.KeptTokens, ci.TokensAfter)
			}
			if k == len(infos)-1 && sh.tokens(next) != ci.TokensAfter {
				t.Errorf("seed %d: TokensAfter %d, next build sends %d", seed, ci.TokensAfter, sh.tokens(next))
			}
		}
	}
	t.Logf("%d folds, %d kept more than the pristine floor", folds, widened)
	if widened == 0 {
		t.Fatal("no fold exercised the FoldKeepPct band")
	}
}

// raceRunner runs a second, floor-only fold (Shaper.Compact: no target, so no
// band — what dun's boundaryFold calls) on the same store while the first
// fold's summary is in flight.
type raceRunner struct {
	sumRunner
	other *Shaper
	fired bool
}

func (r *raceRunner) ChatStream(ctx context.Context, msgs []llm.Message, tools []llm.ToolDef, o *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	if !r.fired {
		r.fired = true
		if _, _, err := r.other.Compact(context.Background(), "s"); err != nil {
			return nil, err
		}
	}
	return r.sumRunner.ChatStream(ctx, msgs, tools, o)
}

// KeptTokens is measured on the snapshot compactOldest loaded before the
// summary call; TokensAfter on the store after it. Anything that removes kept
// entries in between (a second fold, a drop) leaves KeptTokens describing
// entries that are no longer sent. Reproduces dun's shape: kept > after.
func TestFold_KeptTokensDescribeTheSnapshotNotTheStoreAfterAConcurrentFold(t *testing.T) {
	st := reportedSession(3990)
	sh := reportedShaper(st, 55)
	other := reportedShaper(st, 55)
	other.Runner = &sumRunner{n: 300}
	sh.Runner = &raceRunner{sumRunner: sumRunner{n: 8000 * 3}, other: other}
	_, infos := buildWithInfo(t, sh, strings.Repeat("y", 30_000))
	if len(infos) == 0 {
		t.Fatal("no fold")
	}
	ci := infos[0]
	next, _ := sh.Build(context.Background(), "s", "")
	t.Logf("kept %d (%d tok), after %d, next build %d", ci.KeptCount, ci.KeptTokens, ci.TokensAfter, sh.tokens(next))
	if ci.TokensAfter < ci.KeptTokens {
		t.Fatalf("TokensAfter %d < KeptTokens %d: kept was measured on a snapshot the store no longer holds",
			ci.TokensAfter, ci.KeptTokens)
	}
}

// The after-numbers split TokensAfter with nothing left over and nothing
// counted twice, and TokensBefore is on TokensAfter's basis: the system
// prompt is in the number that decided the fold (CompactionStart) and in
// neither of the reported ones. A before that counts the system prompt and an
// after that does not over-report what a fold removed by the system prompt.
func TestFold_AfterPartsSumToTokensAfterAndBeforeExcludesTheSystemPrompt(t *testing.T) {
	for _, fold := range []bool{false, true} {
		st := reportedSession(3990)
		sh := reportedShaper(st, 55)
		sh.Policy.PinFirstUser = true
		sh.Policy.FoldUserIntoToolResult = fold
		sh.Estimate = Default()
		system := strings.Repeat("y", 30_000)

		snap := append([]Entry(nil), st.entries...)
		sortEntries(snap)
		wantBefore := sh.tokens(sh.render(snap, classifyPristineCount(snap, 6, 4), "", false))
		sysTokens := messageTokens(CharsByThree{}, llm.Message{Role: "system", Content: system})

		var start CompactionStart
		var infos []CompactionInfo
		ctx := context.WithValue(context.Background(), compactionSinkKey{}, func(ci CompactionInfo) { infos = append(infos, ci) })
		ctx = withCompactionStartSink(ctx, func(cs CompactionStart) {
			if start.TokensBefore == 0 {
				start = cs
			}
		})
		if _, err := sh.Build(ctx, "s", system); err != nil {
			t.Fatal(err)
		}
		if len(infos) == 0 {
			t.Fatal("no fold")
		}
		ci := infos[0]
		if got := ci.SummaryTokens + ci.PinnedTokens + ci.KeptTokens; got != ci.TokensAfter {
			t.Errorf("fold=%v: summary %d + pinned %d + kept %d = %d, TokensAfter %d",
				fold, ci.SummaryTokens, ci.PinnedTokens, ci.KeptTokens, got, ci.TokensAfter)
		}
		if ci.SummaryTokens <= 0 || ci.PinnedTokens <= 0 || ci.KeptTokens <= 0 {
			t.Errorf("fold=%v: summary %d, pinned %d, kept %d: each part must be measured",
				fold, ci.SummaryTokens, ci.PinnedTokens, ci.KeptTokens)
		}
		if ci.TokensBefore != wantBefore {
			t.Errorf("fold=%v: TokensBefore %d, conversation render %d", fold, ci.TokensBefore, wantBefore)
		}
		if start.TokensBefore != ci.TokensBefore+sysTokens {
			t.Errorf("fold=%v: start %d, info %d + system %d", fold, start.TokensBefore, ci.TokensBefore, sysTokens)
		}
	}
}

// A round that only calls tools has no Content: counted by role alone, the
// estimate of a tool-heavy window was low by every call's arguments.
func TestMessageTokens_CountsNativeToolCalls(t *testing.T) {
	m := llm.Message{Role: "assistant"}
	bare := messageTokens(CharsByThree{}, m)
	var tc llm.ToolCall
	tc.Function.Name = "exec"
	tc.Function.Arguments = strings.Repeat("a", 300)
	m.ToolCalls = []llm.ToolCall{tc}
	if got, want := messageTokens(CharsByThree{}, m), bare+2+100; got != want {
		t.Fatalf("got %d tokens, want %d (role + name + arguments)", got, want)
	}
}
