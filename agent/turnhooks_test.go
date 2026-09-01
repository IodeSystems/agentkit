package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// call builds a ToolCall; Function is an anonymous struct, so it cannot be
// written as a composite literal.
func call(id, name string) *llm.ToolCall {
	tc := &llm.ToolCall{ID: id, Type: "function"}
	tc.Function.Name = name
	tc.Function.Arguments = "{}"
	return tc
}

// collect wires both hooks and returns pointers to what they saw. A Turn must
// report exactly one end, whatever exit it took.
func collect(s *Session) (*[]TurnStart, *[]TurnEnd) {
	var starts []TurnStart
	var ends []TurnEnd
	s.OnTurnStart = func(t TurnStart) { starts = append(starts, t) }
	s.OnTurnEnd = func(t TurnEnd) { ends = append(ends, t) }
	return &starts, &ends
}

func onlyEnd(t *testing.T, ends *[]TurnEnd) TurnEnd {
	t.Helper()
	if len(*ends) != 1 {
		t.Fatalf("OnTurnEnd fired %d times, want exactly 1", len(*ends))
	}
	return (*ends)[0]
}

// The ordinary exit. Reason must distinguish it from every failure, and Err must
// be nil — a host keying on Err alone cannot tell this from a terminal tool.
func TestTurnEndReportsReplyOnCleanExit(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "done."}}}}}
	starts, ends := collect(s)

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if len(*starts) != 1 {
		t.Fatalf("OnTurnStart fired %d times, want 1", len(*starts))
	}
	if (*starts)[0].SessionID != "s" {
		t.Errorf("start SessionID = %q, want s", (*starts)[0].SessionID)
	}
	e := onlyEnd(t, ends)
	if e.Reason != TurnReply {
		t.Errorf("Reason = %q, want %q", e.Reason, TurnReply)
	}
	if e.Err != nil {
		t.Errorf("Err = %v, want nil on a clean exit", e.Err)
	}
	if e.Reason.Failed() {
		t.Error("a reply is not a failure")
	}
	if e.Rounds != 1 {
		t.Errorf("Rounds = %d, want 1", e.Rounds)
	}
}

// A terminal tool closing the session is ALSO a nil-error exit. This is the
// case that makes the reason load-bearing: without it, a host cannot tell a
// session that ended deliberately from one that merely stopped talking.
func TestTurnEndDistinguishesTerminalToolFromReply(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{
			{ToolCall: call("1", "ship")},
		}}},
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			return "", ErrSessionClosed
		}}
	_, ends := collect(s)

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	e := onlyEnd(t, ends)
	if e.Reason != TurnTerminalTool {
		t.Errorf("Reason = %q, want %q", e.Reason, TurnTerminalTool)
	}
	if e.Err != nil {
		t.Errorf("Err = %v, want nil", e.Err)
	}
	if e.Reason.Failed() {
		t.Error("a terminal tool is a clean exit")
	}
}

// Running out of rounds is a failure with its own reason, not a generic error.
func TestTurnEndReportsMaxTurns(t *testing.T) {
	poll := []llm.StreamChunk{{ToolCall: call("1", "poll")}}
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 2,
		Runner:   &scriptRunner{turns: [][]llm.StreamChunk{poll, poll, poll}},
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "still going", nil }}
	_, ends := collect(s)

	if _, err := s.Turn(context.Background()); err == nil {
		t.Fatal("Turn: want an error when MaxTurns is exceeded")
	}
	e := onlyEnd(t, ends)
	if e.Reason != TurnMaxTurns {
		t.Errorf("Reason = %q, want %q", e.Reason, TurnMaxTurns)
	}
	if e.Err == nil {
		t.Error("Err must carry the failure")
	}
	if !e.Reason.Failed() {
		t.Error("max_turns is a failure")
	}
	if e.Rounds != 2 {
		t.Errorf("Rounds = %d, want 2 (MaxTurns)", e.Rounds)
	}
}

// ctxRunner fails the way a real transport does when the caller goes away: it
// returns the context's error. scriptRunner ignores ctx entirely, so it can
// never reach this path.
type ctxRunner struct{}

func (ctxRunner) ChatStream(ctx context.Context, _ []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ch := make(chan llm.StreamChunk, 1)
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

// A canceled context is separated from TurnError deliberately: it is usually a
// host shutting down, and a host that logs it as a fault cries wolf. The
// classification reads ctx, not the error's text, because cancellation arrives
// wrapped in whatever call happened to fail — here, "agent: chat: %w".
func TestTurnEndSeparatesCancellationFromError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5, Runner: ctxRunner{}}
	_, ends := collect(s)

	_, err := s.Turn(ctx)
	if err == nil {
		t.Fatal("Turn: want an error from a canceled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Turn err = %v, want it to wrap context.Canceled", err)
	}
	e := onlyEnd(t, ends)
	if e.Reason != TurnCanceled {
		t.Errorf("Reason = %q, want %q — a cancellation must not read as a fault",
			e.Reason, TurnCanceled)
	}
	if !errors.Is(e.Err, context.Canceled) {
		t.Errorf("Err = %v, want the cancellation", e.Err)
	}
}

// A transport failure that is NOT a cancellation must still classify as
// TurnError. Without this, the ctx check could classify everything as canceled
// and the cancellation test above would still pass.
func TestTurnEndReportsPlainTransportError(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: failRunner{errors.New("connection refused")}}
	_, ends := collect(s)

	if _, err := s.Turn(context.Background()); err == nil {
		t.Fatal("Turn: want the transport error")
	}
	e := onlyEnd(t, ends)
	if e.Reason != TurnError {
		t.Errorf("Reason = %q, want %q", e.Reason, TurnError)
	}
	if !e.Reason.Failed() {
		t.Error("a transport error is a failure")
	}
}

// failRunner always fails, with no cancellation involved.
type failRunner struct{ err error }

func (r failRunner) ChatStream(context.Context, []llm.Message, []llm.ToolDef, *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	return nil, r.err
}

// The hooks are optional. A Session that sets neither must behave exactly as
// before — this is the guard against the wiring becoming load-bearing.
func TestTurnHooksAreOptional(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "done."}}}}}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "done." {
		t.Errorf("Reply = %q", res.Reply)
	}
}

// toolRound is a round that calls a tool AND reports the proxy's overhead.
func toolRound(id, name string, queuedMS, loadMS int64) []llm.StreamChunk {
	u := &llm.Usage{PromptTokens: 10, CompletionTokens: 2}
	u.Server = llm.ServerTiming{QueuedMS: queuedMS, LoadMS: loadMS, Reported: true}
	return []llm.StreamChunk{{ToolCall: call(id, name)}, {Done: true, Usage: u}}
}

// replyRound is a final round: content, no tool calls, so the loop ends.
func replyRound(content string, queuedMS, loadMS int64) []llm.StreamChunk {
	u := &llm.Usage{PromptTokens: 10, CompletionTokens: 2}
	u.Server = llm.ServerTiming{QueuedMS: queuedMS, LoadMS: loadMS, Reported: true}
	return []llm.StreamChunk{{Content: content}, {Done: true, Usage: u}}
}

// A turn's cost is the sum of its round-trips, so overhead must accumulate
// across rounds — a host budgets the turn, not the round. Two rounds here: a
// tool call that queued 4.5s, then the reply that queued 3s.
func TestTurnEndSumsServerOverheadAcrossRounds(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{
			toolRound("1", "look", 4000, 500),
			replyRound("done.", 3000, 0),
		}},
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil }}
	_, ends := collect(s)

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	e := onlyEnd(t, ends)
	if e.Rounds != 2 {
		t.Fatalf("Rounds = %d, want 2 — this test is meaningless with one round", e.Rounds)
	}
	if !e.ServerTimingSeen {
		t.Fatal("ServerTimingSeen false — the proxy's report was dropped before TurnEnd")
	}
	if e.ServerOverheadMS != 7500 {
		t.Errorf("ServerOverheadMS = %d, want 7500 (4000+500 then 3000+0)", e.ServerOverheadMS)
	}
	if e.Duration <= 0 {
		t.Error("Duration must be set")
	}
}

// The distinction a host cannot afford to lose: an endpoint that reports
// nothing must leave ServerTimingSeen false, so nobody subtracts an overhead
// that was never measured.
func TestTurnEndDoesNotInventOverhead(t *testing.T) {
	s := &Session{SessionID: "s", Store: &memStore{}, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "done."}}}}}
	_, ends := collect(s)

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	e := onlyEnd(t, ends)
	if e.ServerTimingSeen {
		t.Error("ServerTimingSeen must be false when nothing reported")
	}
	if e.ServerOverheadMS != 0 {
		t.Errorf("ServerOverheadMS = %d, want 0", e.ServerOverheadMS)
	}
}
