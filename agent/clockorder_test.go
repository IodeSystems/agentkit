package agent

import (
	"context"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// A round in which the model says something AND makes a call — the ordinary
// shape, and the one that exposed the bug.
func sayAndCall(text, id string) []llm.StreamChunk {
	tc := &llm.ToolCall{ID: id, Type: "function"}
	tc.Function.Name = "look"
	tc.Function.Arguments = `{}`
	return []llm.StreamChunk{{Content: text}, {ToolCall: tc}}
}

// History is ordered by CreatedAt, so every Append in the loop has to use the
// SAME clock. The assistant reply used time.Now() while everything else used
// the session's Now, which on a host that supplies one (a counter, a fake
// clock) sorted every reply after the entire rest of the conversation.
//
// The visible symptom was llama.cpp rejecting "2 or more assistant messages at
// the end of the list", but the reordering had already corrupted every turn:
// the model was shown tool calls with no reasoning in front of them.
func TestAssistantEntriesUseTheSessionClock(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		sayAndCall("first thought", "c1"),
		sayAndCall("second thought", "c2"),
		{{Content: "done"}},
	}}
	var tick int64
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 10,
		Now:      func() int64 { tick++; return tick },
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil },
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}

	entries, err := store.Context(context.Background(), "s")
	if err != nil {
		t.Fatal(err)
	}
	sortEntries(entries)

	for _, e := range entries {
		if e.CreatedAt > tick {
			t.Fatalf("%s entry has CreatedAt=%d, beyond the session clock's %d — it used a "+
				"different clock and will sort outside the conversation", e.Kind, e.CreatedAt, tick)
		}
	}

	// The ordering that matters: an assistant reply must sit with the round it
	// came from, not after everything.
	kinds := make([]EntryKind, 0, len(entries))
	for _, e := range entries {
		kinds = append(kinds, e.Kind)
	}
	for i := 1; i < len(kinds); i++ {
		if kinds[i] == KindAssistant && kinds[i-1] == KindAssistant {
			t.Fatalf("two assistant entries adjacent — llama.cpp rejects that outright: %v", kinds)
		}
	}
	if kinds[0] != KindAssistant {
		t.Errorf("first entry is %s; the first round's reply should lead the history: %v", kinds[0], kinds)
	}
}

// The same guarantee at the message level, which is what actually reaches the
// provider.
func TestRenderedMessagesDoNotEndWithTwoAssistants(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		sayAndCall("a", "c1"),
		sayAndCall("b", "c2"),
		{{Content: "c"}},
	}}
	var tick int64
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 10,
		Now:      func() int64 { tick++; return tick },
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ok", nil },
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	msgs, err := DefaultContextBuilder(context.Background(), store, "s", "")
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(msgs); i++ {
		if msgs[i].Role == "assistant" && msgs[i-1].Role == "assistant" {
			roles := make([]string, len(msgs))
			for j, m := range msgs {
				roles[j] = m.Role
			}
			t.Fatalf("consecutive assistant messages sent to the provider: %v", roles)
		}
	}
}
