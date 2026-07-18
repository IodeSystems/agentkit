package agent

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/iodesystems/agentkit/llm"
)

// SlotSystemNote is auto-appended to the system prompt when tools are attached,
// skipped when there are none, and suppressible via OmitSlotInstructions.
func TestTurn_AppendsSlotNoteWhenTools(t *testing.T) {
	sysOf := func(s *Session) string {
		store := &memStore{}
		store.queue(Entry{ID: "u1", Kind: KindUser, Content: "hi", CreatedAt: 1})
		s.SessionID, s.Store, s.Now = "s1", store, func() int64 { return 1 }
		s.Runner = &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "ok"}}}}
		if _, err := s.Turn(context.Background()); err != nil {
			t.Fatal(err)
		}
		for _, m := range s.Runner.(*scriptRunner).seen[0] {
			if m.Role == "system" {
				return m.Content
			}
		}
		return ""
	}

	tools := []llm.ToolDef{{Type: "function"}}
	if got := sysOf(&Session{System: "BASE", Tools: tools}); !strings.Contains(got, SlotSystemNote) || !strings.HasPrefix(got, "BASE") {
		t.Fatalf("expected BASE + slot note, got %q", got)
	}
	if got := sysOf(&Session{System: "BASE"}); strings.Contains(got, SlotSystemNote) {
		t.Fatalf("no tools → no slot note, got %q", got)
	}
	if got := sysOf(&Session{System: "BASE", Tools: tools, OmitSlotInstructions: true}); strings.Contains(got, SlotSystemNote) {
		t.Fatalf("OmitSlotInstructions should suppress the note, got %q", got)
	}
}

// End to end: a tool result's section is transcluded into the next reply.
// TurnResult.Reply is expanded (host displays full bytes); the STORED assistant
// entry keeps the placeholder (model replay stays lean).
func TestTurn_TranscludesToolOutput(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "show it", CreatedAt: 1})

	// Turn iter 1: model calls a tool. iter 2: model references its output by
	// placeholder instead of retyping the 3 rows.
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{toolCallChunk("c1", "query", "{}")},
		{{Content: "Here are the results:\n{report}"}},
	}}
	s := &Session{
		SessionID: "s1",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 1 },
		Dispatch: func(_ context.Context, _ llm.ToolCall) (string, error) {
			return "<report>a\nb\nc</report>", nil
		},
	}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if want := "Here are the results:\na\nb\nc"; res.Reply != want {
		t.Fatalf("Reply = %q, want expanded %q", res.Reply, want)
	}
	// The persisted assistant entry stays RAW.
	var asst *Entry
	for i := range store.entries {
		if store.entries[i].Kind == KindAssistant {
			asst = &store.entries[i]
		}
	}
	if asst == nil {
		t.Fatal("no assistant entry stored")
	}
	if asst.Content != "Here are the results:\n{report}" {
		t.Fatalf("stored reply should keep placeholder, got %q", asst.Content)
	}
}

// captureSlots must expose {OUTPUT} for any result and a named slot per
// well-formed section, with tags stripped from both.
func TestCaptureSlots_sectionsAndOutput(t *testing.T) {
	slots := captureSlots("before <stdout>line1\nline2</stdout> after <stderr>oops</stderr>")
	if slots["stdout"] != "line1\nline2" {
		t.Fatalf("stdout slot = %q", slots["stdout"])
	}
	if slots["stderr"] != "oops" {
		t.Fatalf("stderr slot = %q", slots["stderr"])
	}
	// OUTPUT keeps surrounding prose but unwraps the section tags.
	if got, want := slots[slotOutput], "before line1\nline2 after oops"; got != want {
		t.Fatalf("OUTPUT = %q, want %q", got, want)
	}
}

// A result with no sections still yields OUTPUT = the whole result verbatim.
func TestCaptureSlots_noSections(t *testing.T) {
	slots := captureSlots("just plain text {not a slot}")
	if slots[slotOutput] != "just plain text {not a slot}" {
		t.Fatalf("OUTPUT = %q", slots[slotOutput])
	}
	if len(slots) != 1 {
		t.Fatalf("expected only OUTPUT, got %v", slots)
	}
}

// A mismatched open/close pair (<a>...</b>) is NOT a section — guards against
// RE2's lack of backreferences silently mis-capturing.
func TestCaptureSlots_mismatchedTagsIgnored(t *testing.T) {
	slots := captureSlots("<a>x</b>")
	if _, ok := slots["a"]; ok {
		t.Fatalf("mismatched tag captured as slot: %v", slots)
	}
	if slots[slotOutput] != "<a>x</b>" {
		t.Fatalf("OUTPUT should be untouched, got %q", slots[slotOutput])
	}
}

// expandSlots substitutes known placeholders and leaves unknown braces alone —
// the safety property that keeps JSON/code from being mangled.
func TestExpandSlots_knownOnly(t *testing.T) {
	slots := map[string]string{"table": "ROWS", slotOutput: "ALL"}
	got := expandSlots(`see {table}; whole {OUTPUT}; config {"k":1}; missing {ghost}`, slots)
	want := `see ROWS; whole ALL; config {"k":1}; missing {ghost}`
	if got != want {
		t.Fatalf("expand = %q, want %q", got, want)
	}
}

func TestExpandSlots_emptyIsNoop(t *testing.T) {
	if got := expandSlots("{OUTPUT}", nil); got != "{OUTPUT}" {
		t.Fatalf("empty slots should be a no-op, got %q", got)
	}
}

// truncateForModel keeps a head+tail around a marker, stays under budget-ish,
// names the sections, and never splits a rune.
func TestTruncateForModel(t *testing.T) {
	// Under limit → unchanged.
	if got := truncateForModel("short", 100); got != "short" {
		t.Fatalf("under-limit should pass through, got %q", got)
	}
	// Over limit → marker + both ends preserved.
	body := strings.Repeat("A", 300) + "<tbl>x</tbl>" + strings.Repeat("Z", 300)
	got := truncateForModel(body, 120)
	if !strings.Contains(got, "truncated") || !strings.Contains(got, "{OUTPUT}") {
		t.Fatalf("marker missing: %q", got)
	}
	if !strings.Contains(got, "{tbl}") {
		t.Fatalf("section name not advertised: %q", got)
	}
	if !strings.HasPrefix(got, "AAA") || !strings.HasSuffix(got, "ZZZ") {
		t.Fatalf("head/tail not preserved: %q", got)
	}
	if len(got) >= len(body) {
		t.Fatalf("truncation did not shrink: %d >= %d", len(got), len(body))
	}
}

func TestTruncateForModel_runeSafe(t *testing.T) {
	body := strings.Repeat("世", 200) // 3 bytes each
	got := truncateForModel(body, 90)
	if !utf8.ValidString(got) {
		t.Fatalf("truncation split a rune: %q", got)
	}
}

// Truncation is MODEL-view only: the stored result stays complete and {OUTPUT}
// surfaces it whole even with MaxToolResultChars set well below its length.
func TestTurn_TruncationKeepsOutputComplete(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u1", Kind: KindUser, Content: "dump", CreatedAt: 1})

	complete := strings.Repeat("data ", 100) // 500 chars
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{toolCallChunk("c1", "dump", "{}")},
		{{Content: "Here:\n{OUTPUT}"}},
	}}
	s := &Session{
		SessionID:          "s1",
		Store:              store,
		Runner:             runner,
		Now:                func() int64 { return 1 },
		MaxToolResultChars: 80,
		Dispatch: func(_ context.Context, _ llm.ToolCall) (string, error) {
			return complete, nil
		},
	}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// User-facing reply carries the COMPLETE result.
	if res.Reply != "Here:\n"+complete {
		t.Fatalf("Reply missing complete output: %q", res.Reply)
	}
	// Stored tool result is complete (truncation is view-only).
	for i := range store.entries {
		if store.entries[i].Kind == KindToolResult && store.entries[i].Content != complete {
			t.Fatalf("stored result was truncated: %q", store.entries[i].Content)
		}
	}
	// The MODEL's second-round view WAS truncated.
	last := runner.seen[len(runner.seen)-1]
	var toolMsg string
	for _, m := range last {
		if m.Role == "tool" {
			toolMsg = m.Content
		}
	}
	if len(toolMsg) >= len(complete) || !strings.Contains(toolMsg, "truncated") {
		t.Fatalf("model view was not truncated: %q", toolMsg)
	}
}

// ExpandEntries splices a tool result's slots into a later assistant reply,
// leaving the reply's Entry the display copy while other kinds pass through.
func TestExpandEntries_history(t *testing.T) {
	entries := []Entry{
		{Kind: KindUser, Content: "run it"},
		{Kind: KindToolResult, Content: "<report>42 rows</report>"},
		{Kind: KindAssistant, Content: "Done. Results:\n{report}"},
	}
	out := ExpandEntries(entries)
	if out[2].Content != "Done. Results:\n42 rows" {
		t.Fatalf("assistant not expanded: %q", out[2].Content)
	}
	// Input not mutated.
	if entries[2].Content != "Done. Results:\n{report}" {
		t.Fatalf("ExpandEntries mutated its input: %q", entries[2].Content)
	}
	// Non-assistant entries untouched.
	if out[1].Content != "<report>42 rows</report>" {
		t.Fatalf("tool result changed: %q", out[1].Content)
	}
}
