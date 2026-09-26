package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// Reasoning is kept on the entry of the round that produced it — the reply, or
// the first call of a round with no prose — counted, and never replayed.
func TestTurnRecordsReasoningAndNeverReplaysIt(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		// Round 1: reasoning, then two calls and no prose.
		{
			{Reasoning: "I should look "}, {Reasoning: "first."},
			{ToolCall: call("1", "look")}, {ToolCall: call("2", "look")},
			{Usage: &llm.Usage{CompletionTokens: 50}},
		},
		// Round 2: reasoning and a reply.
		{
			{Reasoning: "done thinking"}, {Content: "answer"},
			{Usage: &llm.Usage{CompletionTokens: 20}},
		},
	}}
	s := &Session{SessionID: "s", Store: store, MaxTurns: 5, Runner: runner,
		Estimate: CharsByFour{},
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil }}
	var end TurnEnd
	s.OnTurnEnd = func(e TurnEnd) { end = e }
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}

	var calls, replies []Entry
	for _, e := range store.entries {
		switch e.Kind {
		case KindToolCall:
			calls = append(calls, e)
		case KindAssistant:
			replies = append(replies, e)
		}
	}
	if len(calls) != 2 || len(replies) != 1 {
		t.Fatalf("got %d calls, %d replies", len(calls), len(replies))
	}
	if calls[0].Reasoning != "I should look first." || calls[0].Usage == nil ||
		calls[0].Usage.ReasoningSource != ReasoningEstimated || calls[0].Usage.Generated != 50 {
		t.Errorf("first call of a prose-less round must carry its reasoning and usage: %+v %+v", calls[0], calls[0].Usage)
	}
	if calls[1].Reasoning != "" || calls[1].Usage != nil {
		t.Errorf("second call must not repeat the round's reasoning/usage: %+v", calls[1])
	}
	if replies[0].Reasoning != "done thinking" || replies[0].Usage.ReasoningSource != ReasoningEstimated {
		t.Errorf("reply lost its reasoning: %+v %+v", replies[0], replies[0].Usage)
	}
	// "I should look first." is 20 chars -> 5 by chars/4 rounded up; "done thinking" 13 -> 4.
	if end.Usage.Reasoning != 5+4 || end.Usage.ReasoningEstimated != 2 || end.Usage.ReasoningReported != 0 {
		t.Errorf("turn tally = %+v", end.Usage)
	}
	for _, msgs := range runner.seen {
		for _, m := range msgs {
			if strings.Contains(m.Content, "should look") || strings.Contains(m.Content, "done thinking") {
				t.Fatalf("reasoning was replayed to the model: %+v", m)
			}
		}
	}
}

// A provider that neither counts nor streams reasoning leaves it NOT REPORTED —
// an empty source, not a zero that reads as "no reasoning happened".
func TestReasoningNotReportedIsUnknownNotZero(t *testing.T) {
	s := &Session{}
	eu := s.entryUsage(&llm.Usage{CompletionTokens: 100}, "")
	if eu == nil || eu.ReasoningSource != "" || eu.Reasoning != 0 {
		t.Errorf("want unknown reasoning, got %+v", eu)
	}
	n := 7
	eu = s.entryUsage(&llm.Usage{CompletionTokens: 100,
		CompletionTokensDetails: &llm.CompletionTokensDetails{ReasoningTokens: &n}}, "some text")
	if eu.ReasoningSource != ReasoningReported || eu.Reasoning != 7 {
		t.Errorf("a reported count must win over an estimate: %+v", eu)
	}
	// An estimate never exceeds the completion it is part of.
	eu = s.entryUsage(&llm.Usage{CompletionTokens: 2}, strings.Repeat("x", 300))
	if eu.Reasoning != 2 {
		t.Errorf("estimate not capped at completion: %+v", eu)
	}
}
