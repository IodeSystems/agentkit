package agent

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// failRound is a reasoning-only round that OpenRouter reports as a stop with
// the upstream's failure in native_finish_reason.
func failRound(think, native string) []llm.StreamChunk {
	return []llm.StreamChunk{{Reasoning: think}, {Done: true, FinishReason: "stop", NativeFinishReason: native}}
}

// recordSleep returns a sleeper that records each wait and never sleeps.
func recordSleep(waits *[]time.Duration) func(context.Context, time.Duration) error {
	return func(_ context.Context, d time.Duration) error { *waits = append(*waits, d); return nil }
}

func TestMalformedResponseIsResentNotRepaired(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		failRound("thinking", "MALFORMED_RESPONSE"),
		{{Content: "The answer."}},
	}}
	budget := 2500
	host := &llm.ChatOpts{ReasoningBudgetTokens: &budget, Effort: llm.EffortHigh}
	var waits []time.Duration
	var events []ProviderFailRetry
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, ChatOpts: host,
		sleep:               recordSleep(&waits),
		OnProviderFailRetry: func(r ProviderFailRetry) { events = append(events, r) },
		OnReasoningCut:      func(ReasoningCut) { t.Error("reasoning-cut repair ran for a provider failure") }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "The answer." || len(runner.seen) != 2 {
		t.Fatalf("reply=%q err=%v rounds=%d", res.Reply, err, len(runner.seen))
	}
	if !reflect.DeepEqual(runner.seen[0], runner.seen[1]) {
		t.Errorf("resend differs from the failed request:\n%v\n%v", runner.seen[0], runner.seen[1])
	}
	for i, o := range runner.opts {
		if o.Effort != llm.EffortHigh || o.ReasoningBudgetTokens == nil {
			t.Errorf("round %d opts = %+v, want reasoning left on", i, o)
		}
	}
	if n := countNotices(store, ReasoningCutTag) + countNotices(store, EmptyReplyTag); n != 0 {
		t.Errorf("%d repair notices persisted, want 0", n)
	}
	if len(events) != 1 || events[0].Attempt != 1 || events[0].Max != 10 ||
		events[0].NativeFinishReason != "MALFORMED_RESPONSE" || events[0].FinishReason != "stop" ||
		len(waits) != 1 || events[0].Wait != waits[0] {
		t.Errorf("events=%+v waits=%v, want one event whose Wait is the slept wait", events, waits)
	}
}

func TestOtherWithWhitespaceTextIsResentAndNothingPersisted(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "  \n"}, {Done: true, FinishReason: "stop", NativeFinishReason: "OTHER"}},
		{{Content: "ok"}},
	}}
	var waits []time.Duration
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: recordSleep(&waits)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "ok" || len(waits) != 1 {
		t.Fatalf("reply=%q err=%v resends=%d", res.Reply, err, len(waits))
	}
	if !reflect.DeepEqual(runner.seen[0], runner.seen[1]) {
		t.Error("the whitespace reply entered the resent request")
	}
}

func TestCleanNativeStopIsNotResent(t *testing.T) {
	for _, native := range []string{"end_turn", "End_Turn", "STOP", "eos"} {
		store := &memStore{}
		runner := &scriptRunner{turns: [][]llm.StreamChunk{emptyRound("stop", native), {{Content: "ok"}}}}
		var waits []time.Duration
		s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: recordSleep(&waits)}
		if _, err := s.Turn(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(waits) != 0 || countNotices(store, EmptyReplyTag) != 1 {
			t.Errorf("native %q: resends=%d empty-reply notices=%d, want the empty-reply repair",
				native, len(waits), countNotices(store, EmptyReplyTag))
		}
	}
}

func TestVisibleTextWithMalformedResponseIsKeptAsReply(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "Partial but real."}, {Done: true, FinishReason: "stop", NativeFinishReason: "MALFORMED_RESPONSE"}},
	}}
	var waits []time.Duration
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: recordSleep(&waits)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "Partial but real." || len(waits) != 0 || len(runner.seen) != 1 {
		t.Errorf("reply=%q err=%v resends=%d rounds=%d", res.Reply, err, len(waits), len(runner.seen))
	}
}

func TestProviderFailResendsBackOffThenFallToReasoningCutRepair(t *testing.T) {
	store := &memStore{}
	var turns [][]llm.StreamChunk
	for range 11 {
		turns = append(turns, failRound("think", "MALFORMED_RESPONSE"))
	}
	turns = append(turns, []llm.StreamChunk{{Content: "repaired"}})
	runner := &scriptRunner{turns: turns}
	var waits []time.Duration
	var events []ProviderFailRetry
	cuts := 0
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 3, sleep: recordSleep(&waits),
		OnProviderFailRetry: func(r ProviderFailRetry) { events = append(events, r) },
		OnReasoningCut:      func(c ReasoningCut) { cuts++ }}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "repaired" {
		t.Fatalf("reply=%q err=%v; resends must not spend MaxTurns", res.Reply, err)
	}
	want := []time.Duration{1, 2, 4, 8, 16, 32, 64, 128, 256, 300}
	if len(waits) != len(want) || len(events) != len(want) {
		t.Fatalf("waits=%v events=%d, want %d", waits, len(events), len(want))
	}
	for i, w := range want {
		lo := w * time.Second
		hi := lo + time.Duration(providerFailJitter*float64(lo))
		if waits[i] < lo || waits[i] > hi || events[i].Attempt != i+1 || events[i].Wait != waits[i] {
			t.Errorf("resend %d: wait=%v event=%+v, want [%v, %v]", i+1, waits[i], events[i], lo, hi)
		}
	}
	if len(cutNotices(store)) != 1 || cuts != 2 || runner.opts[11].Effort != llm.EffortNone || runner.opts[10].Effort == llm.EffortNone {
		t.Errorf("cut notices=%d hooks=%d, want the reasoning-cut repair after the resends",
			len(cutNotices(store)), cuts)
	}
}

func TestProviderFailDuringRepairRoundKeepsTheRepairPending(t *testing.T) {
	// A plain empty round is repaired; the repair round itself is a provider
	// failure. The resend is the repair round again, not a failed repair.
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		emptyRound("stop", ""),
		emptyRound("stop", "MALFORMED_RESPONSE"),
		{{Content: "ok"}},
	}}
	var waits []time.Duration
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5, sleep: recordSleep(&waits)}
	res, err := s.Turn(context.Background())
	if err != nil || res.Reply != "ok" || len(waits) != 1 || countNotices(store, EmptyReplyTag) != 1 {
		t.Fatalf("reply=%q err=%v resends=%d notices=%d", res.Reply, err, len(waits), countNotices(store, EmptyReplyTag))
	}
	if !reflect.DeepEqual(runner.seen[1], runner.seen[2]) {
		t.Error("the resent repair round differs from the failed one")
	}
}

func TestCancelDuringProviderFailBackoffEndsTurnCanceled(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{failRound("t", "MALFORMED_RESPONSE"), {{Content: "never"}}}}
	ctx, cancel := context.WithCancel(context.Background())
	var end TurnEnd
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		sleep:     func(ctx context.Context, d time.Duration) error { cancel(); return sleepCtx(ctx, d) },
		OnTurnEnd: func(e TurnEnd) { end = e }}
	_, err := s.Turn(ctx)
	if !errors.Is(err, context.Canceled) || end.Reason != TurnCanceled || len(runner.seen) != 1 {
		t.Errorf("err=%v reason=%q rounds=%d", err, end.Reason, len(runner.seen))
	}
	if len(store.entries) != 0 {
		t.Errorf("entries persisted: %+v", store.entries)
	}
}

func TestNegativeMaxProviderFailRetriesKeepsOldBehavior(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{failRound("t", "MALFORMED_RESPONSE"), {{Content: "ok"}}}}
	var waits []time.Duration
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		MaxProviderFailRetries: -1, sleep: recordSleep(&waits)}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(waits) != 0 || len(cutNotices(store)) != 1 {
		t.Errorf("resends=%d cut notices=%d, want the reasoning-cut repair at once", len(waits), len(cutNotices(store)))
	}
}
