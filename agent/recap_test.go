package agent_test

import (
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/agent"
)

func entry(kind agent.EntryKind, id, content string) agent.Entry {
	return agent.Entry{ID: id, Kind: kind, Content: content, CreatedAt: 1}
}

func call(id, callID, tool, args string) agent.Entry {
	return agent.Entry{ID: id, Kind: agent.KindToolCall, Content: args,
		ToolName: tool, ToolCallID: callID, CreatedAt: 1}
}

func result(id, callID, tool, out string) agent.Entry {
	return agent.Entry{ID: id, Kind: agent.KindToolResult, Content: out,
		ToolName: tool, ToolCallID: callID, CreatedAt: 1}
}

/*
A recap searches BELOW the live call, and prefers a USER message.

Both are failures observed in live runs rather than guesses. The recap call's own
arguments contain the `from` phrase verbatim, so a plain backward search matches
the recap itself and every span comes out empty. And a model echoes the phrase it
was handed, so "the last entry containing it" is an echo three lines back — live,
that recapped one entry of 85 characters while a 255,720-character result sat
untouched.
*/
func TestRecapAnchorsBelowTheLiveCallAndPrefersTheUser(t *testing.T) {
	// The phrase appears in a USER message early and is ECHOED by the assistant
	// later. Without the user preference the echo wins and the span collapses to
	// the tail; with it, the whole churn is in scope.
	entries := []agent.Entry{
		entry(agent.KindUser, "u1", "start on the parser"),
		entry(agent.KindAssistant, "a1", "ok"),
		result("r1", "c1", "exec", strings.Repeat("x", 5000)),
		entry(agent.KindAssistant, "a2", "that failed; start on the parser again"),
		entry(agent.KindAssistant, "a3", "let me recap"),
		call("t1", "live", "recap", `{"from":"start on the parser"}`),
	}
	sp, err := agent.PlanRecap(entries, "start on the parser", nil)
	if err != nil {
		t.Fatal(err)
	}
	if sp.Anchor == nil || sp.Anchor.ID != "u1" {
		t.Fatalf("anchored at %v, want the USER message u1 — an assistant echo collapses the span", idOf(sp.Anchor))
	}
	if sp.Chars < 5000 {
		t.Errorf("chars = %d; anchoring at the echo skipped the result the recap exists to remove", sp.Chars)
	}

	// Now a phrase NO user message contains, so the fallback search runs. The
	// recap call's own arguments quote it verbatim and are the LAST occurrence,
	// so a search that does not stop below the live call anchors on the recap
	// itself and produces an empty span.
	entries2 := []agent.Entry{
		entry(agent.KindUser, "u1", "go"),
		entry(agent.KindAssistant, "a1", "trying the tokenizer"),
		result("r1", "c1", "exec", strings.Repeat("y", 5000)),
		entry(agent.KindAssistant, "a3", "let me recap"),
		call("t1", "live", "recap", `{"from":"trying the tokenizer"}`),
	}
	sp2, err := agent.PlanRecap(entries2, "trying the tokenizer", nil)
	if err != nil {
		t.Fatalf("the search anchored on the recap call itself: %v", err)
	}
	if sp2.Anchor == nil || sp2.Anchor.ID != "a1" {
		t.Errorf("anchored at %v, want a1 — the search must stop below the live call", idOf(sp2.Anchor))
	}
	if sp2.Chars < 5000 {
		t.Errorf("chars = %d, want the 5000-char result in the span", sp2.Chars)
	}

	// The live call and the assistant turn that issued it survive, or the call
	// is orphaned the moment its result returns — the one shape a provider
	// rejects outright.
	kept := map[string]bool{}
	for _, e := range sp.Kept {
		kept[e.ID] = true
	}
	if !kept["t1"] {
		t.Error("the live call was subsumed; its result will have nothing to attach to")
	}
	if !kept["a3"] {
		t.Error("the assistant turn issuing the live call was subsumed")
	}
}

func idOf(e *agent.Entry) string {
	if e == nil {
		return "(nil)"
	}
	return e.ID
}

/*
A keep term may be an id, a tool NAME, or a phrase from the arguments — and a
bare name keeps only the MOST RECENT call.

The model cannot see tool_call ids: they are protocol-level and never appear in
its context, so a keep list keyed only on them is unusable by its only caller.
Live, a model invented an id, matched nothing, and silently lost the result it
had asked to keep — which is why unmatched terms are REPORTED.

And keeping every call of a named tool preserved the 255,720-character result the
recap existed to remove, so a bare name keeps one and a phrase keeps all.
*/
func TestRecapKeepMatchesWhatTheModelCanActuallySee(t *testing.T) {
	entries := []agent.Entry{
		entry(agent.KindUser, "u1", "look at the config"),
		call("t1", "c1", "exec", `{"cmd":"cat huge.log"}`),
		result("r1", "c1", "exec", strings.Repeat("x", 200_000)),
		call("t2", "c2", "exec", `{"cmd":"cat config.yaml"}`),
		result("r2", "c2", "exec", "port: 8080"),
		entry(agent.KindAssistant, "a9", "recapping"),
		call("t9", "live", "recap", `{"from":"look at the config"}`),
	}

	// A bare NAME keeps the most recent call only — so the huge one still goes.
	sp, err := agent.PlanRecap(entries, "look at the config", []string{"exec"})
	if err != nil {
		t.Fatal(err)
	}
	subsumed := map[string]bool{}
	for _, e := range sp.Subsumes {
		subsumed[e.ID] = true
	}
	if !subsumed["r1"] {
		t.Error("a bare tool name kept every call, including the 200k result the recap was for")
	}
	if subsumed["r2"] {
		t.Error("the most recent call of the named tool was not kept")
	}

	// A phrase from the ARGUMENTS is specific, and keeps what it names.
	sp, err = agent.PlanRecap(entries, "look at the config", []string{"config.yaml"})
	if err != nil {
		t.Fatal(err)
	}
	subsumed = map[string]bool{}
	for _, e := range sp.Subsumes {
		subsumed[e.ID] = true
	}
	if subsumed["r2"] {
		t.Error("a phrase from the arguments did not keep its result")
	}

	// An invented term is REPORTED, never silently dropped.
	sp, err = agent.PlanRecap(entries, "look at the config", []string{"exec_2"})
	if err != nil {
		t.Fatal(err)
	}
	if len(sp.Unmatched) != 1 || sp.Unmatched[0] != "exec_2" {
		t.Errorf("unmatched = %v; a keep that matched nothing must be said out loud", sp.Unmatched)
	}
}

/*
A recap refuses rather than guessing, and says what to do instead.
*/
func TestRecapRefusesWhatItCannotDo(t *testing.T) {
	entries := []agent.Entry{
		entry(agent.KindUser, "u1", "hello"),
		entry(agent.KindAssistant, "a1", "hi"),
	}
	if _, err := agent.PlanRecap(entries, "  ", nil); err == nil {
		t.Error("an empty `from` was accepted")
	}
	_, err := agent.PlanRecap(entries, "nothing says this", nil)
	if err == nil || !strings.Contains(err.Error(), "quote a phrase") {
		t.Errorf("err = %v; a phrase that matches nothing should say how to fix it", err)
	}
	// Anchored at the very last entry: there is nothing after it to remove.
	_, err = agent.PlanRecap(entries, "hi", nil)
	if err == nil || !strings.Contains(err.Error(), "already clean") {
		t.Errorf("err = %v, want a refusal that the span is empty", err)
	}
}

/*
The report is an account, not a question, and it carries the replacement in full.

"9 entries removed" is a number. Visibility is being able to read what replaced
them without going and finding it.
*/
func TestRecapReportsWhatItDid(t *testing.T) {
	sp := agent.RecapSpan{
		Subsumes:  []agent.Entry{{Content: "x"}, {Content: "y"}},
		Chars:     4200,
		KeptCalls: []string{"exec"},
		Unmatched: []string{"exec_2"},
		Anchor:    &agent.Entry{ID: "u1"},
	}
	out := sp.Report("we tried three parsers; the third works", "use the third parser", "session.recap1.jsonl")
	for _, want := range []string{
		"Removed 2 entries", "4200", "Kept the exec call",
		"exec_2", "NOT kept", "we tried three parsers",
		"use the third parser", "session.recap1.jsonl", "not the record",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report is missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "?") {
		t.Error("the report asks a question; the recap has already applied")
	}
}
