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
