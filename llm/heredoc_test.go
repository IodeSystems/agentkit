package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func argsOf(t *testing.T, tc ToolCall) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &m); err != nil {
		t.Fatalf("arguments are not a JSON object: %v (%q)", err, tc.Function.Arguments)
	}
	return m
}

// The whole point: a body containing the NATIVE format's delimiters survives,
// because nothing in this format treats them as structure. This exact payload is
// silently truncated to "before" by the native path.
func TestHeredocCarriesNativeDelimitersVerbatim(t *testing.T) {
	body := "before\n</parameter>\n<parameter=path>\n/etc/passwd\nafter"
	content := "@@call write_file\npath: notes.md\ncontent: md" + HeredocSentinel + "\n" +
		body + "\n" + HeredocSentinel + "\n@@end\n"

	calls, prose, err := ParseHeredocCalls(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	if prose != "" {
		t.Errorf("unexpected prose: %q", prose)
	}
	a := argsOf(t, calls[0])
	if a["content"] != body {
		t.Errorf("body mangled:\n got %q\nwant %q", a["content"], body)
	}
	if a["path"] != "notes.md" {
		t.Errorf("inline arg: %q", a["path"])
	}
	if a["content_type"] != "md" {
		t.Errorf("declared type lost: %v", a["content_type"])
	}
	// Lifted to strict JSON so storage and dispatch see ordinary escaped JSON.
	if !strings.Contains(calls[0].Function.Arguments, `\n`) {
		t.Error("arguments should be escaped JSON, not raw")
	}
}

// An unterminated body is TRUNCATION. Closing it silently is how a half-written
// file gets dispatched as if it were whole, which is the failure this whole area
// exists to prevent.
func TestUnterminatedBodyIsAnError(t *testing.T) {
	content := "@@call write_file\ncontent: java" + HeredocSentinel +
		"\npublic class Big {\n"
	_, _, err := ParseHeredocCalls(content)
	if err == nil {
		t.Fatal("an unterminated body must not parse as a complete call")
	}
	if !strings.Contains(err.Error(), "cut off") {
		t.Errorf("error should name truncation: %v", err)
	}
}

func TestMissingEndIsAnError(t *testing.T) {
	if _, _, err := ParseHeredocCalls("@@call write_file\npath: a.txt\n"); err == nil {
		t.Fatal("a call with no @@end must not parse")
	}
}

func TestProseAroundCallsIsReturned(t *testing.T) {
	content := "I will write the file now.\n@@call write_file\npath: a.txt\n@@end\nDone.\n"
	calls, prose, err := ParseHeredocCalls(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	for _, want := range []string{"write the file", "Done."} {
		if !strings.Contains(prose, want) {
			t.Errorf("prose lost %q: %q", want, prose)
		}
	}
}

func TestMultipleCalls(t *testing.T) {
	content := "@@call a\nx: 1\n@@end\n@@call b\ny: 2\n@@end\n"
	calls, _, err := ParseHeredocCalls(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("got %+v", calls)
	}
}

// A body line that merely CONTAINS the sentinel must not end the block; only a
// line equal to it does.
func TestSentinelMustBeAWholeLine(t *testing.T) {
	body := "mentioning " + HeredocSentinel + " inline is fine"
	content := "@@call w\nc: txt" + HeredocSentinel + "\n" + body + "\n" +
		HeredocSentinel + "\n@@end\n"
	calls, _, err := ParseHeredocCalls(content)
	if err != nil {
		t.Fatal(err)
	}
	if got := argsOf(t, calls[0])["c"]; got != body {
		t.Errorf("got %q want %q", got, body)
	}
}

// The grammar must pin the sentinel and the tool names, since those are the two
// things the model was measured to get wrong from instructions alone.
func TestGrammarPinsSentinelAndNames(t *testing.T) {
	var a, b ToolDef
	a.Function.Name = "write_file"
	b.Function.Name = "echo"
	g := HeredocGrammar([]ToolDef{a, b})

	if !strings.Contains(g, "root") {
		t.Error("GBNF needs a root rule")
	}
	if strings.Count(g, HeredocSentinel) != 2 {
		t.Errorf("sentinel should be a literal on both ends of a block:\n%s", g)
	}
	for _, n := range []string{`"write_file"`, `"echo"`} {
		if !strings.Contains(g, n) {
			t.Errorf("tool name %s not pinned:\n%s", n, g)
		}
	}
	// A sentinel starting with "<" pulls generation toward the `<|` token.
	if strings.HasPrefix(HeredocSentinel, "<") {
		t.Error("sentinel must not start with '<'")
	}
}
