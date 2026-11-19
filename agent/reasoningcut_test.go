package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func reasoningOnly(think string, finish llm.StopReason) []llm.StreamChunk {
	return []llm.StreamChunk{{Reasoning: think}, {Done: true, StopReason: finish}}
}

func cutNotices(st *memStore) []Entry {
	var out []Entry
	for _, e := range st.entries {
		if e.Kind == KindNotification && e.Tag == ReasoningCutTag {
			out = append(out, e)
		}
	}
	return out
}

func TestReasoningOnlyRoundIsRepairedNotEndedAsReply(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		reasoningOnly("thinking about the parcel", ""),
		{{Content: "The answer."}},
	}}
	var hooks []ReasoningCut
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnReasoningCut: func(c ReasoningCut) { hooks = append(hooks, c) }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "The answer." {
		t.Fatalf("Turn = %q, %v; want the repair round's reply", res.Reply, err)
	}
	n := cutNotices(store)
	if len(n) != 1 || !strings.Contains(n[0].Content, "thinking about the parcel") ||
		!strings.Contains(n[0].Content, "Do not reason again") {
		t.Fatalf("notices = %+v", n)
	}
	if len(runner.seen) != 2 {
		t.Errorf("rounds = %d, want 2", len(runner.seen))
	}
	if len(hooks) != 2 || hooks[0].Repaired || !hooks[1].Repaired {
		t.Errorf("hook calls = %+v, want Repaired false then true", hooks)
	}
	for _, e := range store.entries {
		if e.Kind == KindAssistant && strings.TrimSpace(e.Content) == "" {
			t.Error("an empty assistant entry was persisted")
		}
	}
}

func TestLengthCutEmptyReplyWithReasoningIsNotTheOverflowPath(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		reasoningOnly("long thought", llm.StopReasonLength),
		{{Content: "done"}},
	}}
	var hook ReasoningCut
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnOverflow: func(context.Context, Overflow) OverflowDecision {
			t.Error("OnOverflow ran for a reasoning cut")
			return OverflowDecision{}
		},
		OnReasoningCut: func(c ReasoningCut) {
			if !c.Repaired {
				hook = c
			}
		}}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "done" {
		t.Fatalf("Turn = %q, %v", res.Reply, err)
	}
	if hook.Finish != llm.StopReasonLength {
		t.Errorf("Finish = %q, want length", hook.Finish)
	}
}

func TestLengthCutInsideToolCallStillTakesTheOverflowPath(t *testing.T) {
	store := &memStore{}
	cut := &llm.ToolCall{ID: "c1", Type: "function"}
	cut.Function.Name = "exec"
	cut.Function.Arguments = `{"command":"x`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Reasoning: "think"}, {ToolCall: cut}, {Done: true, StopReason: llm.StopReasonLength, TruncatedToolCalls: []string{"exec"}}},
		{{Content: "ok"}},
	}}
	over, rc := 0, 0
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		Dispatch:       func(context.Context, llm.ToolCall) (string, error) { return "ran", nil },
		OnOverflow:     func(context.Context, Overflow) OverflowDecision { over++; return OverflowDecision{} },
		OnReasoningCut: func(ReasoningCut) { rc++ }}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if over != 1 || rc != 0 || len(cutNotices(store)) != 0 {
		t.Errorf("overflow=%d reasoningCut=%d notices=%d; want 1,0,0", over, rc, len(cutNotices(store)))
	}
}

func TestEmptyRepairRoundEndsTurnReasoningCut(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		reasoningOnly("a", ""), reasoningOnly("b", ""), {{Content: "never"}},
	}}
	var end TurnEnd
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnTurnEnd: func(e TurnEnd) { end = e }}
	_, err := s.Turn(context.Background())
	var rce *ReasoningCutError
	if !errors.As(err, &rce) {
		t.Fatalf("err = %v, want *ReasoningCutError", err)
	}
	if end.Reason != TurnReasoningCut || !end.Reason.Failed() {
		t.Errorf("reason = %q", end.Reason)
	}
	if len(runner.seen) != 2 {
		t.Errorf("rounds = %d, want exactly 2", len(runner.seen))
	}
}

func TestRepairRoundSendsReasoningOffAndOnlyThatRound(t *testing.T) {
	store := &memStore{}
	call := &llm.ToolCall{ID: "c1", Type: "function"}
	call.Function.Name = "t"
	call.Function.Arguments = `{}`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		reasoningOnly("think", ""),
		{{ToolCall: call}},
		{{Content: "end"}},
	}}
	budget := 2500
	host := &llm.ChatOpts{ReasoningBudgetTokens: &budget, Effort: llm.EffortHigh}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 6, ChatOpts: host,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil }}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(runner.opts) != 3 {
		t.Fatalf("rounds = %d", len(runner.opts))
	}
	r := runner.opts[1]
	if r.Effort != llm.EffortNone || r.ReasoningBudgetTokens != nil {
		t.Errorf("repair round opts = %+v, want reasoning off", r)
	}
	for _, i := range []int{0, 2} {
		o := runner.opts[i]
		if o.Effort != llm.EffortHigh || o.ReasoningBudgetTokens == nil {
			t.Errorf("round %d opts = %+v, want the host's", i, o)
		}
	}
	if s.ChatOpts != host || host.Effort != llm.EffortHigh || host.ReasoningBudgetTokens == nil {
		t.Errorf("s.ChatOpts was mutated: %+v", host)
	}
}

func TestNoReasoningCutRepairKeepsTodaysBehaviour(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{reasoningOnly("think", "")}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, NoReasoningCutRepair: true,
		OnReasoningCut: func(ReasoningCut) { t.Error("hook ran with the repair off") }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "" || len(runner.seen) != 1 || len(cutNotices(store)) != 0 {
		t.Errorf("reply=%q err=%v rounds=%d", res.Reply, err, len(runner.seen))
	}
}

func TestEmptyReplyWithoutReasoningIsUntouched(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "  "}}}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnReasoningCut: func(ReasoningCut) { t.Error("hook ran without reasoning") }}
	if _, err := s.Turn(context.Background()); err != nil || len(runner.seen) != 1 {
		t.Errorf("err=%v rounds=%d", err, len(runner.seen))
	}
}

func TestReasoningTailIsBoundedAndValidUTF8(t *testing.T) {
	long := strings.Repeat("é世", 3000) // 5 bytes per pair
	tail := reasoningTail(long)
	if !strings.HasPrefix(tail, "…") {
		t.Error("a cut tail must start with …")
	}
	body := strings.TrimPrefix(tail, "…")
	if len(body) > reasoningTailBytes || len(body) < reasoningTailBytes-3 {
		t.Errorf("tail body = %d bytes", len(body))
	}
	if !validUTF8(tail) {
		t.Error("tail is not valid UTF-8")
	}
	if got := reasoningTail("short"); got != "short" {
		t.Errorf("short reasoning = %q", got)
	}
}

func validUTF8(s string) bool { return strings.ToValidUTF8(s, "") == s }
