package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// noSleep fails the test if a resend backoff starts: a malformed call must
// never be resent.
func noSleep(t *testing.T) func(context.Context, time.Duration) error {
	return func(context.Context, time.Duration) error { t.Error("a malformed call was resent"); return nil }
}

func TestMalformedCallIsRepairedNotResent(t *testing.T) {
	for _, finish := range []string{"stop", "error"} {
		store := &memStore{}
		runner := &scriptRunner{turns: [][]llm.StreamChunk{
			{{Reasoning: "thinking"}, {Done: true, FinishReason: finish, NativeFinishReason: "malformed_function_call"}},
			{{Content: "The answer."}},
		}}
		host := &llm.ChatOpts{Effort: llm.EffortHigh}
		s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, ChatOpts: host,
			sleep:          noSleep(t),
			OnReasoningCut: func(ReasoningCut) { t.Error("reasoning-cut repair ran for a malformed call") }}
		res, err := s.Turn(context.Background())
		if err != nil || res.Reply != "The answer." || len(runner.seen) != 2 {
			t.Fatalf("finish %q: reply=%q err=%v rounds=%d", finish, res.Reply, err, len(runner.seen))
		}
		var notice string
		for _, e := range store.entries {
			if e.Kind == KindNotification && e.Tag == MalformedCallTag {
				notice = e.Content
			}
		}
		if countNotices(store, MalformedCallTag) != 1 || !strings.Contains(notice, "malformed_function_call") ||
			countNotices(store, EmptyReplyTag) != 0 {
			t.Errorf("finish %q: notices malformed=%d empty=%d content=%q", finish,
				countNotices(store, MalformedCallTag), countNotices(store, EmptyReplyTag), notice)
		}
		if runner.opts[1].Effort != llm.EffortHigh {
			t.Errorf("finish %q: repair round effort = %q, want the host's opts unchanged", finish, runner.opts[1].Effort)
		}
	}
}

func TestMalformedCallRepairsSpentEndWithEmptyReplyError(t *testing.T) {
	store := &memStore{}
	var turns [][]llm.StreamChunk
	for range 4 {
		turns = append(turns, emptyRound("stop", "MALFORMED_FUNCTION_CALL"))
	}
	turns = append(turns, []llm.StreamChunk{{Content: "never"}})
	runner := &scriptRunner{turns: turns}
	var end TurnEnd
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 10, sleep: noSleep(t),
		OnTurnEnd: func(e TurnEnd) { end = e }}
	_, err := s.Turn(context.Background())
	var ee *EmptyReplyError
	if !errors.As(err, &ee) || !errors.Is(err, ErrEmptyReply) || ee.NativeFinishReason != "MALFORMED_FUNCTION_CALL" {
		t.Fatalf("err = %v, want *EmptyReplyError with the native reason", err)
	}
	if end.Reason != TurnEmptyReply || len(runner.seen) != 4 || countNotices(store, MalformedCallTag) != 3 ||
		countNotices(store, EmptyReplyTag) != 0 {
		t.Errorf("reason=%q rounds=%d malformed notices=%d empty notices=%d, want 3 repairs and no shared repair",
			end.Reason, len(runner.seen), countNotices(store, MalformedCallTag), countNotices(store, EmptyReplyTag))
	}
}

func TestMalformedCallWithVisibleTextIsAReply(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "Here it is."}, {Done: true, FinishReason: "stop", NativeFinishReason: "MALFORMED_FUNCTION_CALL"}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: noSleep(t)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "Here it is." || len(runner.seen) != 1 || countNotices(store, MalformedCallTag) != 0 {
		t.Errorf("reply=%q err=%v rounds=%d notices=%d", res.Reply, err, len(runner.seen), countNotices(store, MalformedCallTag))
	}
}

func TestNegativeMaxMalformedCallRepairsKeepsTheResend(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", "MALFORMED_FUNCTION_CALL"), {{Content: "ok"}},
	}}
	var waits []time.Duration
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		MaxMalformedCallRepairs: -1, sleep: recordSleep(&waits)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "ok" || len(waits) != 1 || countNotices(store, MalformedCallTag) != 0 {
		t.Errorf("reply=%q err=%v resends=%d notices=%d, want the old resend", res.Reply, err, len(waits),
			countNotices(store, MalformedCallTag))
	}
}

func TestMalformedCallBudgetIsApartFromTheSharedRepair(t *testing.T) {
	// The shared repair is spent on a plain empty round; the repair round is a
	// malformed call and still gets its own repair instead of ending the Turn.
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", ""),
		emptyRound("error", "MALFORMED_FUNCTION_CALL"),
		{{Content: "ok"}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: noSleep(t)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "ok" || countNotices(store, EmptyReplyTag) != 1 || countNotices(store, MalformedCallTag) != 1 {
		t.Errorf("reply=%q err=%v empty=%d malformed=%d", res.Reply, err,
			countNotices(store, EmptyReplyTag), countNotices(store, MalformedCallTag))
	}
}

func TestMalformedCallIgnoresNoReasoningCutRepair(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", "MALFORMED_FUNCTION_CALL"), {{Content: "ok"}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, NoReasoningCutRepair: true, sleep: noSleep(t)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "ok" || countNotices(store, MalformedCallTag) != 1 {
		t.Errorf("reply=%q err=%v notices=%d, want the malformed-call repair", res.Reply, err, countNotices(store, MalformedCallTag))
	}
}
