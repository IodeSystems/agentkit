package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// llm.ToolCall's Function is an anonymous struct, so build one by assignment.
func toolCall(id, name, args string) *llm.ToolCall {
	tc := &llm.ToolCall{ID: id, Type: "function"}
	tc.Function.Name = name
	tc.Function.Arguments = args
	return tc
}

// End-to-end through Turn: a tool call whose arguments were cut off mid-stream
// must be ANSWERED, not executed and not dropped.
//
// Four things have to hold at once, and each has burned this code before:
//
//	1. the tool never runs        — half arguments means a half write
//	2. the persisted call parses  — an unparseable one poisons the session,
//	                                 rejected at parse time by every later request
//	3. a paired tool_result exists — a dropped call leaves an orphan result
//	4. the loop continues         — so the model can retry, which is the point
func TestTurn_TruncatedToolCallIsAnsweredNotExecuted(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "write the file", CreatedAt: 1})

	truncated := `{"path":"Big.java","content":"public class Big {`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{ToolCall: toolCall("call_1", "write_file", truncated)}},
		{{Content: "retried smaller"}}, // the loop must get here
	}}

	var dispatched []string
	s := &Session{
		SessionID: "s1",
		System:    "sys",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 100 },
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			dispatched = append(dispatched, tc.Function.Name)
			return "wrote it", nil
		},
	}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("a truncated call must not fail the turn: %v", err)
	}

	// 1. never executed
	if len(dispatched) != 0 {
		t.Fatalf("dispatched a call with half its arguments: %v", dispatched)
	}
	// 4. the loop continued to a second round
	if res.Reply != "retried smaller" {
		t.Fatalf("loop did not continue; reply=%q", res.Reply)
	}

	var call, result *Entry
	for i := range store.entries {
		switch store.entries[i].Kind {
		case KindToolCall:
			call = &store.entries[i]
		case KindToolResult:
			result = &store.entries[i]
		}
	}
	if call == nil || result == nil {
		t.Fatalf("call/result pair missing: call=%v result=%v", call, result)
	}
	// 3. paired by id
	if call.ToolCallID != "call_1" || result.ToolCallID != "call_1" {
		t.Errorf("pair broken: call=%q result=%q", call.ToolCallID, result.ToolCallID)
	}
	// 2. the persisted call is parseable — and an OBJECT, which is what a
	// provider walks to render historical calls.
	var obj map[string]any
	if err := json.Unmarshal([]byte(call.Content), &obj); err != nil {
		t.Fatalf("persisted arguments would poison the session: %v\n%s", err, call.Content)
	}
	if !strings.Contains(call.Content, "Big.java") {
		t.Error("the model's partial work should be salvageable from the stand-in")
	}
	// The result must tell the model what to do differently, and say the two
	// things it cannot infer: the call did not run, and it was cut off (retry
	// smaller) rather than malformed (retry correctly).
	for _, want := range []string{"truncation detected", "NOT executed", "smaller"} {
		if !strings.Contains(result.Content, want) {
			t.Errorf("tool result should mention %q, got: %s", want, result.Content)
		}
	}
	// Both sides of the exchange agree on why.
	if why, _ := obj["_error"].(string); why != result.Content {
		t.Errorf("persisted call and its result disagree:\n call:   %q\n result: %q", why, result.Content)
	}
}

// A well-formed call still dispatches — the guard must not cost the normal path.
func TestTurn_ValidToolCallStillDispatches(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{ToolCall: toolCall("call_1", "write_file", `{"path":"a.java"}`)}},
		{{Content: "done"}},
	}}
	var dispatched []string
	s := &Session{
		SessionID: "s1", System: "sys", Store: store, Runner: runner,
		Now: func() int64 { return 100 },
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			dispatched = append(dispatched, tc.Function.Arguments)
			return "wrote it", nil
		},
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(dispatched) != 1 || dispatched[0] != `{"path":"a.java"}` {
		t.Fatalf("valid call did not reach the dispatcher verbatim: %v", dispatched)
	}
}
