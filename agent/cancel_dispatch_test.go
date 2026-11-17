package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// ctxStore fails Append when the ctx it is given is done, like a Store backed
// by a database driver that honours cancellation.
type ctxStore struct{ memStore }

func (c *ctxStore) Append(ctx context.Context, id string, e Entry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return c.memStore.Append(ctx, id, e)
}

func threeCalls() *scriptRunner {
	return &scriptRunner{turns: [][]llm.StreamChunk{
		{
			{ToolCall: toolCall("c1", "t", `{}`)},
			{ToolCall: toolCall("c2", "t", `{}`)},
			{ToolCall: toolCall("c3", "t", `{}`)},
		},
		{{Content: "must not be reached"}},
	}}
}

// assertPaired fails unless every KindToolCall has exactly one KindToolResult.
func assertPaired(t *testing.T, entries []Entry) map[string]string {
	t.Helper()
	calls, results := map[string]bool{}, map[string]string{}
	for _, e := range entries {
		switch e.Kind {
		case KindToolCall:
			calls[e.ToolCallID] = true
		case KindToolResult:
			if _, dup := results[e.ToolCallID]; dup {
				t.Fatalf("two results for %s", e.ToolCallID)
			}
			results[e.ToolCallID] = e.Content
		}
	}
	for id := range calls {
		if _, ok := results[id]; !ok {
			t.Fatalf("orphan tool call %s: no result", id)
		}
	}
	return results
}

func runCancelTurn(t *testing.T, store Store, cancelIn func(cancel context.CancelFunc, n int)) (dispatched []string, runner *scriptRunner, reason TurnEndReason, err error) {
	t.Helper()
	runner = threeCalls()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := &Session{
		SessionID: "s1", System: "sys", Store: store, Runner: runner,
		Now:       func() int64 { return 100 },
		OnTurnEnd: func(e TurnEnd) { reason = e.Reason },
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			dispatched = append(dispatched, tc.ID)
			cancelIn(cancel, len(dispatched))
			return "real " + tc.ID, nil
		},
	}
	_, err = s.Turn(ctx)
	return
}

func TestTurn_CancelDuringCallOneAnswersRemainingCalls(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})
	d, runner, reason, err := runCancelTurn(t, store, func(cancel context.CancelFunc, n int) {
		if n == 1 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if reason != TurnCanceled {
		t.Fatalf("reason = %v", reason)
	}
	if len(d) != 1 || d[0] != "c1" {
		t.Fatalf("dispatched %v, want only c1", d)
	}
	if runner.i != 1 {
		t.Fatalf("chat rounds = %d, want 1", runner.i)
	}
	r := assertPaired(t, store.entries)
	if r["c1"] != "real c1" || r["c2"] != errCanceledBeforeRun || r["c3"] != errCanceledBeforeRun {
		t.Fatalf("results = %v", r)
	}
}

func TestTurn_CancelWithCtxHonouringStoreStillPersistsAllResults(t *testing.T) {
	store := &ctxStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})
	d, _, reason, err := runCancelTurn(t, store, func(cancel context.CancelFunc, n int) {
		if n == 1 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled (not a persist error)", err)
	}
	if reason != TurnCanceled || len(d) != 1 {
		t.Fatalf("reason=%v dispatched=%v", reason, d)
	}
	r := assertPaired(t, store.entries)
	if r["c1"] != "real c1" || r["c2"] != errCanceledBeforeRun || r["c3"] != errCanceledBeforeRun {
		t.Fatalf("results = %v", r)
	}
}

// Cancel lands after the model round returned and before the first dispatch.
// The store honours cancellation: the calls must still be persisted (the round
// happened) and every one answered.
func TestTurn_CancelBeforeFirstDispatchAnswersEveryCall(t *testing.T) {
	store := &ctxStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// Cancel as the model round's stream ends.
	base := threeCalls()
	runner := &cancelAfterRunner{scriptRunner: base, cancel: cancel}
	var dispatched int
	var reason TurnEndReason
	s := &Session{
		SessionID: "s1", System: "sys", Store: store, Runner: runner,
		Now:       func() int64 { return 100 },
		OnTurnEnd: func(e TurnEnd) { reason = e.Reason },
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			dispatched++
			return "x", nil
		},
	}
	_, err := s.Turn(ctx)
	if !errors.Is(err, context.Canceled) || reason != TurnCanceled {
		t.Fatalf("err=%v reason=%v", err, reason)
	}
	if dispatched != 0 {
		t.Fatalf("dispatched %d calls after cancel", dispatched)
	}
	r := assertPaired(t, store.entries)
	if len(r) != 3 {
		t.Fatalf("results = %v", r)
	}
	for id, c := range r {
		if c != errCanceledBeforeRun {
			t.Fatalf("%s = %q", id, c)
		}
	}
}

// cancelAfterRunner cancels once the scripted round has been handed back.
type cancelAfterRunner struct {
	*scriptRunner
	cancel context.CancelFunc
}

func (r *cancelAfterRunner) ChatStream(ctx context.Context, m []llm.Message, td []llm.ToolDef, o *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	ch, err := r.scriptRunner.ChatStream(ctx, m, td, o)
	r.cancel()
	return ch, err
}
