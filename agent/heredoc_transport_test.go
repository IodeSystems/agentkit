package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func wfTool() llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = "write_file"
	td.Function.Description = "Write a file."
	_ = json.Unmarshal([]byte(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`), &td.Function.Parameters)
	return td
}

// A call arriving as CONTENT is dispatched like any other, with real argument
// types, and the body survives bytes the native format loses.
func TestHeredocTransport_ParsesCallsFromContent(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "write it", CreatedAt: 1})

	body := "before\n</parameter>\n<parameter=path>\nafter"
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "@@call write_file\n{path: \"notes.md\", content: ~~~md\n" +
			body + "\n~~~md\n}\n@@end"}},
		{{Content: "done"}},
	}}

	var got llm.ToolCall
	s := &Session{
		SessionID: "s1", System: "sys", Store: store, Runner: runner,
		Tools: []llm.ToolDef{wfTool()}, ToolFormat: ToolFormatHeredoc,
		Now: func() int64 { return 100 },
		Dispatch: func(_ context.Context, tc llm.ToolCall) (string, error) {
			got = tc
			return "wrote it", nil
		},
	}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply != "done" {
		t.Fatalf("loop did not continue: %q", res.Reply)
	}
	if got.Function.Name != "write_file" {
		t.Fatalf("tool not dispatched: %+v", got)
	}
	var a map[string]any
	if err := json.Unmarshal([]byte(got.Function.Arguments), &a); err != nil {
		t.Fatal(err)
	}
	if a["content"] != body {
		t.Errorf("body mangled:\n got %q\nwant %q", a["content"], body)
	}
	if a["path"] != "notes.md" {
		t.Errorf("path = %#v", a["path"])
	}
}

// The tools array must NOT be sent (llama.cpp refuses a grammar beside it), the
// grammar and stop MUST be, and the system prompt has to teach the format since
// the tool list no longer travels in `tools`.
func TestHeredocTransport_SendsGrammarNotTools(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "go", CreatedAt: 1})
	runner := &recordingRunner{scriptRunner: scriptRunner{
		turns: [][]llm.StreamChunk{{{Content: "nothing to do"}}}}}
	s := &Session{
		SessionID: "s1", System: "sys", Store: store, Runner: runner,
		Tools: []llm.ToolDef{wfTool()}, ToolFormat: ToolFormatHeredoc,
		Now:      func() int64 { return 100 },
		Dispatch: func(context.Context, llm.ToolCall) (string, error) { return "", nil },
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(runner.tools) != 0 {
		t.Errorf("tools must not be sent with a grammar: %+v", runner.tools)
	}
	if runner.opts == nil || runner.opts.Grammar == "" {
		t.Fatal("grammar not sent")
	}
	if len(runner.opts.Stop) == 0 {
		t.Error("stop sequence not sent; a completed grammar does not force EOS")
	}
	sys := runner.msgs[0].Content
	if !strings.Contains(sys, "write_file") || !strings.Contains(sys, llm.HeredocOpen) {
		t.Errorf("system prompt should carry the tool list and the format:\n%s", sys)
	}
}

// Replayed history must be shown in the dialect the model writes, not re-rendered
// through the provider's native tool_calls.
func TestHeredocTransport_ReplaysHistoryAsText(t *testing.T) {
	store := &memStore{}
	store.entries = []Entry{
		{ID: "u1", Kind: KindUser, Content: "write it", CreatedAt: 1},
		{ID: "c1", Kind: KindToolCall, ToolName: "write_file", ToolCallID: "x1",
			Content: `{"path":"a.md","content":"hi"}`, CreatedAt: 2},
		{ID: "r1", Kind: KindToolResult, ToolName: "write_file", ToolCallID: "x1",
			Content: "wrote it", CreatedAt: 3},
	}
	msgs, err := DefaultContextBuilderFormat(context.Background(), store, "s1", "sys",
		ToolFormatHeredoc)
	if err != nil {
		t.Fatal(err)
	}
	var sawCallText, sawNativeCalls bool
	for _, m := range msgs {
		if strings.Contains(m.Content, llm.CallPrefix+"write_file") {
			sawCallText = true
		}
		if len(m.ToolCalls) > 0 {
			sawNativeCalls = true
		}
	}
	if !sawCallText {
		t.Error("a stored call should replay as heredoc text")
	}
	if sawNativeCalls {
		t.Error("a stored call must NOT replay as native tool_calls in heredoc mode")
	}
}
