package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/agent/toolfmt"
	"github.com/iodesystems/agentkit/llm"
)

// toolCallChunk builds a StreamChunk carrying one tool call.
func toolCallChunk(id, name, args string) llm.StreamChunk {
	tc := &llm.ToolCall{ID: id}
	tc.Function.Name = name
	tc.Function.Arguments = args
	return llm.StreamChunk{ToolCall: tc}
}

// TestEncodeToolResult_TransformsStoredResult verifies that a non-nil
// EncodeToolResult re-encodes the raw tool result BEFORE it lands in the
// KindToolResult entry (so the stored + rendered form is the encoded one).
func TestEncodeToolResult_TransformsStoredResult(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})

	const rawJSON = `[{"sym":"S.Start","class":"method"},{"sym":"main","class":"func"}]`

	// Turn 1: model calls a tool. Turn 2: model replies (no tool calls) → done.
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{toolCallChunk("c1", "lookup", "{}")},
		{{Content: "done"}},
	}}
	s := &Session{
		SessionID: "s1",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 1 },
		Dispatch: func(_ context.Context, _ llm.ToolCall) (string, error) {
			return rawJSON, nil
		},
		EncodeToolResult: toolfmt.EncodeCSV,
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}

	var result *Entry
	for i := range store.entries {
		if store.entries[i].Kind == KindToolResult {
			result = &store.entries[i]
		}
	}
	if result == nil {
		t.Fatal("no KindToolResult entry stored")
	}
	if result.Content == rawJSON {
		t.Fatal("EncodeToolResult was not applied; content is raw JSON")
	}
	want := toolfmt.EncodeCSV(rawJSON)
	if result.Content != want {
		t.Fatalf("stored content = %q; want CSV %q", result.Content, want)
	}
	if !strings.Contains(result.Content, "class,sym") {
		t.Errorf("stored content is not CSV:\n%s", result.Content)
	}
}

// TestEncodeToolResult_NilPassthrough verifies backward-compatibility: nil hook
// stores the raw result unchanged.
func TestEncodeToolResult_NilPassthrough(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})

	const rawJSON = `[{"a":1}]`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{toolCallChunk("c1", "lookup", "{}")},
		{{Content: "done"}},
	}}
	s := &Session{
		SessionID: "s1",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 1 },
		Dispatch: func(_ context.Context, _ llm.ToolCall) (string, error) {
			return rawJSON, nil
		},
		// EncodeToolResult nil → passthrough.
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, e := range store.entries {
		if e.Kind == KindToolResult && e.Content != rawJSON {
			t.Fatalf("nil hook changed content: %q", e.Content)
		}
	}
}
