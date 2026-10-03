package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// reheadFixture is a session whose older tool results are large enough that
// the window is over budget even with LOD.
func reheadFixture() *memStore {
	s := &memStore{}
	for i := 0; i < 6; i++ {
		s.entries = append(s.entries,
			Entry{ID: "u" + string(rune('a'+i)), Kind: KindUser, Content: strings.Repeat("big ", 400), CreatedAt: int64(i + 1)})
	}
	s.entries = append(s.entries, Entry{ID: "z", Kind: KindUser, Content: "now", CreatedAt: 100})
	return s
}

func reheadShaper(s *memStore, rehead func(context.Context, string) bool) (*Shaper, *scriptRunner) {
	r := &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}, {{Content: "SUMMARY2"}}}}
	return &Shaper{
		Store: s, Runner: r, Rehead: rehead,
		Policy: ShaperPolicy{BudgetTokens: 600, LODHeadroomTokens: -1, PreserveLastMessages: 1},
	}, r
}

// A re-heading that brings the window under budget replaces the fold: Build
// asks it once, re-reads the context, and calls no summarizer.
func TestReheadThatFitsReplacesTheFold(t *testing.T) {
	s := reheadFixture()
	calls := 0
	sh, _ := reheadShaper(s, func(context.Context, string) bool {
		calls++
		s.mu.Lock()
		defer s.mu.Unlock()
		for i := range s.entries[:len(s.entries)-1] {
			s.entries[i].Content = "[stub]"
		}
		return true
	})
	msgs, err := sh.Build(context.Background(), "s", "sys")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("rehead called %d times, want once", calls)
	}
	if markerCount(s) != 0 {
		t.Fatal("a fold ran although the re-heading fit the window")
	}
	if !strings.Contains(msgs[len(msgs)-1].Content, "now") {
		t.Fatalf("the newest entry is not whole: %+v", msgs[len(msgs)-1])
	}
}

// A re-heading that changes nothing, or not enough, leaves the fold to run as
// before — and is asked only once per Build.
func TestReheadThatDoesNotFitStillFolds(t *testing.T) {
	for _, changed := range []bool{false, true} {
		s := reheadFixture()
		calls := 0
		sh, _ := reheadShaper(s, func(context.Context, string) bool { calls++; return changed })
		if _, err := sh.Build(context.Background(), "s", "sys"); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("changed=%v: rehead called %d times, want once", changed, calls)
		}
		if markerCount(s) == 0 {
			t.Fatalf("changed=%v: no fold although the window is still over budget", changed)
		}
	}
}

// Under budget, the re-heading is never asked: appending to the tail keeps the
// cached prefix, and nothing should rewrite it.
func TestReheadIsNotAskedUnderBudget(t *testing.T) {
	s := &memStore{entries: []Entry{{ID: "a", Kind: KindUser, Content: "small", CreatedAt: 1}}}
	sh, _ := reheadShaper(s, func(context.Context, string) bool { t.Fatal("asked under budget"); return false })
	if _, err := sh.Build(context.Background(), "s", "sys"); err != nil {
		t.Fatal(err)
	}
}
