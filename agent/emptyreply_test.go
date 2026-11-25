package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func emptyRound(finish, native string) []llm.StreamChunk {
	return []llm.StreamChunk{{Done: true, FinishReason: finish, NativeFinishReason: native}}
}

func countNotices(st *memStore, tag string) int {
	n := 0
	for _, e := range st.entries {
		if e.Kind == KindNotification && e.Tag == tag {
			n++
		}
	}
	return n
}

func TestEmptyRoundWithoutReasoningIsRepairedOnce(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", "MALFORMED_FUNCTION_CALL"),
		{{Content: "The answer."}},
	}}
	// Resends and the malformed-call repair off: this test is about the
	// shared repair behind them.
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, MaxProviderFailRetries: -1,
		MaxMalformedCallRepairs: -1, OnReasoningCut: func(ReasoningCut) { t.Error("reasoning-cut hook ran without reasoning") }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "The answer." || len(runner.seen) != 2 {
		t.Fatalf("reply=%q err=%v rounds=%d", res.Reply, err, len(runner.seen))
	}
	if countNotices(store, EmptyReplyTag) != 1 {
		t.Errorf("empty-reply notices = %d, want 1", countNotices(store, EmptyReplyTag))
	}
	for _, e := range store.entries {
		if e.Kind == KindAssistant && e.Content == "" {
			t.Error("an empty assistant entry was persisted")
		}
	}
}

func TestTwoEmptyRoundsFailWithEmptyReplyError(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", "X"), emptyRound("stop", "MALFORMED_FUNCTION_CALL"), {{Content: "never"}},
	}}
	var end TurnEnd
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, MaxProviderFailRetries: -1,
		MaxMalformedCallRepairs: -1, OnTurnEnd: func(e TurnEnd) { end = e }}
	_, err := s.Turn(context.Background())
	var ee *EmptyReplyError
	if !errors.As(err, &ee) || !errors.Is(err, ErrEmptyReply) {
		t.Fatalf("err = %v, want *EmptyReplyError matching ErrEmptyReply", err)
	}
	if ee.FinishReason != "stop" || ee.NativeFinishReason != "MALFORMED_FUNCTION_CALL" {
		t.Errorf("finish = %q/%q, want the repair round's", ee.FinishReason, ee.NativeFinishReason)
	}
	if end.Reason != TurnEmptyReply || !end.Reason.Failed() || len(runner.seen) != 2 {
		t.Errorf("reason=%q rounds=%d", end.Reason, len(runner.seen))
	}
}

func TestEmptyRepairLeavesReasoningOptsAlone(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{emptyRound("stop", ""), {{Content: "ok"}}}}
	budget := 2500
	host := &llm.ChatOpts{ReasoningBudgetTokens: &budget, Effort: llm.EffortHigh}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, ChatOpts: host}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	for i, o := range runner.opts {
		if o.Effort != llm.EffortHigh || o.ReasoningBudgetTokens == nil {
			t.Errorf("round %d opts = %+v, want the host's reasoning settings", i, o)
		}
	}
}

func TestEmptyLengthRoundKeepsOverflowPath(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Done: true, StopReason: llm.StopReasonLength}},
	}}
	over := 0
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnOverflow: func(context.Context, Overflow) OverflowDecision { over++; return OverflowDecision{} }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "" || over != 1 || len(runner.seen) != 1 || countNotices(store, EmptyReplyTag) != 0 {
		t.Errorf("reply=%q err=%v overflow=%d rounds=%d", res.Reply, err, over, len(runner.seen))
	}
}

func TestReasoningCutAndEmptyRepairShareOneBudget(t *testing.T) {
	// Reasoning cut repaired, then an empty round later in the same Turn.
	store := &memStore{}
	call := &llm.ToolCall{ID: "c1", Type: "function"}
	call.Function.Name = "t"
	call.Function.Arguments = `{}`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		reasoningOnly("think", ""),
		{{ToolCall: call}},
		emptyRound("stop", ""),
		{{Content: "never"}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 6,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil }}
	_, err := s.Turn(context.Background())
	if !errors.Is(err, ErrEmptyReply) || len(runner.seen) != 3 {
		t.Errorf("err=%v rounds=%d, want ErrEmptyReply after 3 rounds", err, len(runner.seen))
	}

	// And the other order: empty repaired, then a reasoning-only round.
	store = &memStore{}
	runner = &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", ""),
		{{ToolCall: call}},
		reasoningOnly("think", ""),
		{{Content: "never"}},
	}}
	s = &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 6,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil }}
	_, err = s.Turn(context.Background())
	var rce *ReasoningCutError
	if !errors.As(err, &rce) || len(runner.seen) != 3 {
		t.Errorf("err=%v rounds=%d, want ReasoningCutError after 3 rounds", err, len(runner.seen))
	}
}
