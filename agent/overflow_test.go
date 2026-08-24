package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// cutTurn is what the transport hands back when the provider ran out of room:
// whatever arrived, then a Done chunk saying why. names is the calls the cut
// left unparseable — empty means it landed in prose.
func cutTurn(content string, names ...string) []llm.StreamChunk {
	return []llm.StreamChunk{
		{Content: content},
		{Done: true, StopReason: llm.StopReasonLength, TruncatedToolCalls: names,
			Usage: &llm.Usage{PromptTokens: 109000, CompletionTokens: 79000}},
	}
}

// A cut in PROSE would otherwise end the Turn on half a sentence: there is no
// tool call to answer, so nothing in the loop notices. With a host policy the
// round is re-run, and the note it returned is in the context the retry is
// built from.
func TestOverflowInProseRetriesWithTheNote(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		cutTurn("Here is the plan: first we"),
		{{Content: "Here is the plan, briefly: do X."}},
	}}
	var saw Overflow
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnOverflow: func(_ context.Context, o Overflow) OverflowDecision {
			saw = o
			return OverflowDecision{Note: "you ran out of room; be brief", Retry: true}
		}}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "Here is the plan, briefly: do X." {
		t.Errorf("Reply = %q, want the retried round", res.Reply)
	}
	if saw.Attempt != 1 || saw.InToolCall() {
		t.Errorf("Overflow = %+v; want attempt 1, in prose", saw)
	}
	if saw.PromptTokens != 109000 || saw.Generated != 79000 {
		t.Errorf("the host must get the PROVIDER's counts, not estimates: %+v", saw)
	}
	var note string
	for _, e := range store.entries {
		if e.Kind == KindNotification && e.Tag == OverflowTag {
			note = e.Content
		}
	}
	if note != "you ran out of room; be brief" {
		t.Errorf("note = %q; the host's text must reach the model", note)
	}
}

// A cut TOOL CALL already has a recovery path — the call is persisted with a
// placeholder and answered with an error result. Re-running the round on top of
// that asks the model the same question twice, so the note rides alongside the
// result instead.
func TestOverflowInToolCallDoesNotRerunTheRound(t *testing.T) {
	store := &memStore{}
	cut := &llm.ToolCall{ID: "c1", Type: "function"}
	cut.Function.Name = "exec"
	cut.Function.Arguments = `{"command":"python3 - <<`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{
			{ToolCall: cut},
			{Done: true, StopReason: llm.StopReasonLength, TruncatedToolCalls: []string{"exec"}},
		},
		{{Content: "ok, smaller now"}},
	}}
	rounds := 0
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "ran", nil },
		OnOverflow: func(_ context.Context, o Overflow) OverflowDecision {
			rounds++
			if !o.InToolCall() {
				t.Errorf("a cut call must report itself as one: %+v", o)
			}
			return OverflowDecision{Note: "cut mid-call", Retry: true}
		}}

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if rounds != 1 {
		t.Errorf("overflow fired %d times; want 1", rounds)
	}
	// The pair the model needs to learn from: its own call, and a result saying
	// why it could not run.
	var call, result string
	for _, e := range store.entries {
		switch e.Kind {
		case KindToolCall:
			call = e.Content
		case KindToolResult:
			result = e.Content
		}
	}
	if call == "" || result == "" {
		t.Fatalf("the truncated call must stay paired: call=%q result=%q", call, result)
	}
	if strings.Contains(call, "python3 - <<") {
		t.Error("unparseable arguments must never be persisted verbatim")
	}
}

// Retrying without the host changing anything reproduces the cut exactly. The
// cap is what stops that being an infinite, expensive loop.
func TestOverflowRetriesAreCapped(t *testing.T) {
	store := &memStore{}
	var turns [][]llm.StreamChunk
	for i := 0; i < 8; i++ {
		turns = append(turns, cutTurn("still too long"))
	}
	s := &Session{SessionID: "s", Store: store, Runner: &scriptRunner{turns: turns},
		MaxTurns: 20, MaxOverflowRetries: 2,
		OnOverflow: func(context.Context, Overflow) OverflowDecision {
			return OverflowDecision{Note: "be brief", Retry: true}
		}}

	_, err := s.Turn(context.Background())
	if err == nil {
		t.Fatal("a policy that never helps must fail the Turn, not loop on it")
	}
	if !strings.Contains(err.Error(), "cut generation for room") {
		t.Errorf("error should name the cause: %v", err)
	}
}

// With no policy installed nothing changes: the cut is reported by the
// transport and the loop behaves exactly as it did before.
func TestOverflowWithoutAPolicyIsInert(t *testing.T) {
	store := &memStore{}
	s := &Session{SessionID: "s", Store: store, MaxTurns: 5,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{cutTurn("half a sentence")}}}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "half a sentence" {
		t.Errorf("Reply = %q", res.Reply)
	}
	for _, e := range store.entries {
		if e.Tag == OverflowTag {
			t.Error("no policy means no note")
		}
	}
}
