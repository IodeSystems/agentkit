package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// callTurn is one round in which the model makes the same call again.
func callTurn(id string) []llm.StreamChunk {
	tc := &llm.ToolCall{ID: id, Type: "function"}
	tc.Function.Name = "check_status"
	tc.Function.Arguments = `{"job":"j1"}`
	return []llm.StreamChunk{{ToolCall: tc}}
}

func repeatingTurns(n int) [][]llm.StreamChunk {
	turns := make([][]llm.StreamChunk, 0, n)
	for i := 0; i < n; i++ {
		turns = append(turns, callTurn(fmt.Sprintf("c%d", i)))
	}
	return turns
}

// The same call returning the same answer forever is a loop, and MaxTurns (100)
// is far too generous a backstop for it.
func TestIdenticalExchangeStopsTheTurn(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: repeatingTurns(30)}
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 30,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			return "still running", nil
		},
	}

	_, err := s.Turn(context.Background())
	if err == nil {
		t.Fatal("an identical call/result exchange repeated 30 rounds never stopped the Turn")
	}
	if !strings.Contains(err.Error(), "tool-call loop") {
		t.Errorf("error does not name the cause: %v", err)
	}
	// Default MaxRepeatedExchanges=5 → warn at 5, stop at 10.
	if len(runner.seen) != 10 {
		t.Errorf("made %d rounds, want 10 (warn at 5, stop at twice that)", len(runner.seen))
	}
	var notice string
	for _, e := range store.entries {
		if e.Kind == KindNotification && e.Tag == "tool_loop" {
			notice = e.Content
		}
	}
	if !strings.Contains(notice, "check_status") {
		t.Errorf("the model was never told which tool was looping: %q", notice)
	}
}

// The false positive that would make this guard unusable: a poll IS a repeated
// identical call, and killing one is killing a correctly-waiting job. Only the
// result distinguishes them.
func TestPollingWithChangingResultsIsNotALoop(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: append(repeatingTurns(20), []llm.StreamChunk{{Content: "finished"}})}
	n := 0
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 25,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			n++
			return fmt.Sprintf("running, %d%% complete", n*5), nil
		},
	}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("a poll whose result advances was killed as a loop: %v", err)
	}
	if res.Reply != "finished" {
		t.Errorf("Reply = %q, want the poll to have run to completion", res.Reply)
	}
	for _, e := range store.entries {
		if e.Kind == KindNotification && e.Tag == "tool_loop" {
			t.Fatal("a progressing poll was warned about as a loop")
		}
	}
}

func TestRepeatedExchangeGuardIsDisablable(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: repeatingTurns(12)}
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 12,
		MaxRepeatedExchanges: -1,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			return "same", nil
		},
	}

	_, err := s.Turn(context.Background())
	if err == nil || !strings.Contains(err.Error(), "max turns") {
		t.Fatalf("with the guard off, only MaxTurns should stop the loop; got %v", err)
	}
}

// OnToolCalls hook: a host injects tool calls that appear as if the model
// made them. The Turn loop persists + dispatches them normally.
func TestOnToolCalls_InjectsForcedCall(t *testing.T) {
	store := &memStore{}
	// Model returns a plain text reply with no tool calls.
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "done with the task"}},
	}}

	var dispatched []llm.ToolCall
	var injected bool
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			dispatched = append(dispatched, tc)
			return fmt.Sprintf("tool %q executed", tc.Function.Name), nil
		},
	}
	s.OnToolCalls = func(tcs []llm.ToolCall) []llm.ToolCall {
		if injected {
			return tcs
		}
		injected = true
		forced := &llm.ToolCall{
			ID:   "call_forced_ship",
			Type: "function",
		}
		forced.Function.Name = "ship"
		forced.Function.Arguments = `{"mode":"push"}`
		return append(tcs, *forced)
	}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if !strings.Contains(res.Reply, "done with the task") {
		t.Errorf("reply = %q; want model's text", res.Reply)
	}

	// The forced tool call should have been dispatched.
	if len(dispatched) != 1 {
		t.Fatalf("dispatched %d tool calls, want 1", len(dispatched))
	}
	if dispatched[0].Function.Name != "ship" {
		t.Fatalf("dispatched tool = %q, want ship", dispatched[0].Function.Name)
	}
	if !strings.Contains(dispatched[0].Function.Arguments, "push") {
		t.Errorf("arguments = %q; want push mode", dispatched[0].Function.Arguments)
	}

	// History should show: assistant(text) → assistant(tool_calls: ship) → tool(result).
	var kinds []string
	for _, e := range store.entries {
		kinds = append(kinds, string(e.Kind))
	}
	// Expect: assistant (text reply), tool_call (ship), tool (result).
	if len(kinds) < 3 {
		t.Fatalf("history has %d entries (%v), want at least 3", len(kinds), kinds)
	}
	if kinds[0] != "assistant" {
		t.Errorf("first entry kind = %q, want assistant", kinds[0])
	}
	if kinds[1] != "tool_call" {
		t.Errorf("second entry kind = %q, want tool_call", kinds[1])
	}
	if kinds[2] != "tool_result" {
		t.Errorf("third entry kind = %q, want tool", kinds[2])
	}
}

// OnToolCalls with existing tool calls: the forced call is added alongside
// the model's own calls (multi-tool-call).
func TestOnToolCalls_AddsToExistingCalls(t *testing.T) {
	store := &memStore{}
	tc1 := &llm.ToolCall{ID: "call_model_1", Type: "function"}
	tc1.Function.Name = "node_query"
	tc1.Function.Arguments = `{"selector":"func"}`

	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{ToolCall: tc1}},
		{{Content: "thanks"}}, // second round after tool results
	}}

	var dispatched []string
	var injected2 bool
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			dispatched = append(dispatched, tc.Function.Name)
			return "ok", nil
		},
	}
	s.OnToolCalls = func(tcs []llm.ToolCall) []llm.ToolCall {
		if injected2 {
			return tcs
		}
		injected2 = true
		forced := &llm.ToolCall{
			ID:   "call_forced_ship",
			Type: "function",
		}
		forced.Function.Name = "ship"
		forced.Function.Arguments = `{"mode":"verify"}`
		return append(tcs, *forced)
	}

	_, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}

	// Both tool calls should have been dispatched in the same round.
	if len(dispatched) != 2 {
		t.Fatalf("dispatched %d calls (%v), want 2", len(dispatched), dispatched)
	}
	// Model's call first, forced call second (OnToolCalls appended).
	if dispatched[0] != "node_query" {
		t.Errorf("first dispatched = %q, want node_query", dispatched[0])
	}
	if dispatched[1] != "ship" {
		t.Errorf("second dispatched = %q, want ship", dispatched[1])
	}
}
