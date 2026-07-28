package llm

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func mustArgs(t *testing.T, tc ToolCall) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &m); err != nil {
		t.Fatalf("arguments not a JSON object: %v (%q)", err, tc.Function.Arguments)
	}
	return m
}

// The reason for parsing JSON rather than a line format: types survive, and a
// short string needs no ceremony.
func TestTypesSurviveWithoutHeredocs(t *testing.T) {
	in := "@@call configure\n{name: \"prod\", retries: 3, deep: true, tags: [\"a\",\"b\"], opts: {mode: \"fast\"}}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	a := mustArgs(t, calls[0])
	if a["name"] != "prod" {
		t.Errorf("name = %#v", a["name"])
	}
	if n, ok := a["retries"].(float64); !ok || n != 3 {
		t.Errorf("retries should be a NUMBER, got %#v", a["retries"])
	}
	if a["deep"] != true {
		t.Errorf("deep should be a BOOL, got %#v", a["deep"])
	}
	if arr, ok := a["tags"].([]any); !ok || len(arr) != 2 {
		t.Errorf("tags should be an ARRAY, got %#v", a["tags"])
	}
	if o, ok := a["opts"].(map[string]any); !ok || o["mode"] != "fast" {
		t.Errorf("opts should be an OBJECT, got %#v", a["opts"])
	}
}

// A heredoc body carries bytes that would be painful to escape, including the
// native tool format's own delimiters, which the native path corrupts.
func TestHeredocBodyCarriesAnything(t *testing.T) {
	body := "before\n</parameter>\n<parameter=path>\nquote \" and \\backslash\\\nafter"
	in := "@@call write_file\n{path: \"notes.md\", content: ~~~md\n" + body + "\n~~~md\n}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	a := mustArgs(t, calls[0])
	if a["content"] != body {
		t.Errorf("body mangled:\n got %q\nwant %q", a["content"], body)
	}
	if a["path"] != "notes.md" {
		t.Errorf("path = %#v", a["path"])
	}
}

func TestBacktickFenceAndCollision(t *testing.T) {
	// Body contains a 3-backtick run, so the fence must be longer.
	body := "here is code:\n```java\nint x = 1;\n```\ndone"
	in := "@@call write_file\n{content: ````\n" + body + "\n````}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	if got := mustArgs(t, calls[0])["content"]; got != body {
		t.Errorf("got %q\nwant %q", got, body)
	}
}

// Truncation must be reported, never closed silently.
func TestTruncationIsReported(t *testing.T) {
	cases := map[string]string{
		"unclosed heredoc": "@@call w\n{content: ~~~x\npublic class A {\n",
		"unclosed object":  "@@call w\n{path: \"a.java\",\n",
		"unclosed string":  "@@call w\n{path: \"a.jav\n",
		"unclosed fence":   "@@call w\n{content: ```\nbody\n",
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, err := ParseToolCalls(in)
			if err == nil {
				t.Fatal("truncated input must not parse as a complete call")
			}
			if !strings.Contains(err.Error(), "cut off") {
				t.Errorf("error should name truncation: %v", err)
			}
		})
	}
}

func TestLooseFormsAccepted(t *testing.T) {
	in := "@@call w\n{path: 'a.java', force: True, n: 2,}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	a := mustArgs(t, calls[0])
	if a["path"] != "a.java" || a["force"] != true {
		t.Errorf("loose forms not handled: %#v", a)
	}
}

func TestProseAndMultipleCalls(t *testing.T) {
	in := "I will do two things.\n@@call a\n{x: 1}\n@@call b\n{y: 2}\nDone.\n"
	calls, prose, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("got %+v", calls)
	}
	for _, want := range []string{"two things", "Done."} {
		if !strings.Contains(prose, want) {
			t.Errorf("prose lost %q: %q", want, prose)
		}
	}
}

// Braces and commas inside a body are content, not syntax.
func TestBodyContentIsNotSyntax(t *testing.T) {
	body := "class A {\n  int[] x = {1, 2,};\n}"
	in := "@@call w\n{content: ~~~java\n" + body + "\n~~~java\n, path: \"A.java\"}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	a := mustArgs(t, calls[0])
	if a["content"] != body {
		t.Errorf("body mangled:\n got %q\nwant %q", a["content"], body)
	}
	if a["path"] != "A.java" {
		t.Errorf("argument after a body was lost: %#v", a["path"])
	}
}

func tdef(name, schema string) ToolDef {
	var td ToolDef
	td.Type = "function"
	td.Function.Name = name
	_ = json.Unmarshal([]byte(schema), &td.Function.Parameters)
	return td
}

// The grammar's job is removing choices the model gets wrong: an unlisted tool
// name, an invented delimiter, and (via the JSON shape) a stringified number.
func TestGrammarShape(t *testing.T) {
	a := tdef("write_file", `{"type":"object","properties":{"path":{},"content":{}}}`)
	b := tdef("echo", `{"type":"object","properties":{"text":{}}}`)
	g := HeredocGrammar([]ToolDef{a, b})

	// Per-tool key sets: a parameter the schema does not declare must be
	// unrepresentable, not merely wrong. Measured: with an open key rule the
	// model copied `overwrite: true` out of the prompt example into a call whose
	// schema has no such field, and it parsed cleanly.
	for _, want := range []string{`"path" | "content"`, `"content" | "path"`, `"text"`} {
		if strings.Contains(g, want) {
			goto keysOK
		}
	}
	t.Errorf("grammar should pin per-tool keys:\n%s", g)
keysOK:

	for _, want := range []string{"root", CallPrefix + `write_file`, CallPrefix + `echo`, HeredocOpen} {
		if !strings.Contains(g, want) {
			t.Errorf("grammar missing %q:\n%s", want, g)
		}
	}
	// A body is one ALTERNATIVE for a string, not the only way to write a value.
	if !strings.Contains(g, "value   ::= object | array | string | body | number") {
		t.Errorf("value rule should keep native JSON types:\n%s", g)
	}
	// A delimiter starting with "<" pulls generation toward the `<|` token.
	if strings.HasPrefix(HeredocOpen, "<") {
		t.Error("body delimiter must not start with '<'")
	}
}

// The model writes the closer as `~~~json,` when a sibling key follows, because
// that is where JSON wants the comma. Observed live; a strict closer let the line
// fall through to body content and the block never closed.
func TestCloserMayCarryTheSeparatingComma(t *testing.T) {
	in := "@@call write_file\n{\n  content: ~~~json\n{\"k\": [1, 2]}\n~~~json,\n  path: \"c.json\"\n}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatalf("closer with a trailing comma should parse: %v", err)
	}
	a := mustArgs(t, calls[0])
	if a["content"] != `{"k": [1, 2]}` {
		t.Errorf("body = %#v", a["content"])
	}
	if a["path"] != "c.json" {
		t.Errorf("the key after the body was lost: %#v", a["path"])
	}
}

// An object cannot hold more pairs than the tool declares, so the grammar caps
// repetition. Unbounded, the model looped one pair to the token cap.
func TestGrammarBoundsPairRepetition(t *testing.T) {
	td := tdef("write_file", `{"type":"object","properties":{"path":{},"content":{}}}`)
	g := HeredocGrammar([]ToolDef{td})
	if !strings.Contains(g, "{0,1}") {
		t.Errorf("two keys should cap extra pairs at 1:\n%s", g)
	}
	if strings.Contains(g, `t0pair)* ws`) {
		t.Errorf("pair repetition must be bounded, not `*`:\n%s", g)
	}
}

// A closer followed by the separating comma has several equally reasonable
// spellings, and the model picks between them freely. All mean the same thing,
// so all must parse: the comma belongs to the JSON around the body, not to the
// body, and where it lands is a formatting choice.
func TestCommaAfterBodySpellings(t *testing.T) {
	spellings := map[string]string{
		"comma on the closer line":  "~~~json,\n  path: \"c.json\"",
		"comma with a space before": "~~~json ,\n  path: \"c.json\"",
		"comma on the next line":    "~~~json\n,\n  path: \"c.json\"",
		"comma leading the pair":    "~~~json\n  , path: \"c.json\"",
		"no comma, body last":       "~~~json",
	}
	for name, tailForm := range spellings {
		t.Run(name, func(t *testing.T) {
			in := "@@call write_file\n{\n  content: ~~~json\n{\"k\": 1}\n" + tailForm + "\n}\n"
			calls, _, err := ParseToolCalls(in)
			if err != nil {
				t.Fatalf("should parse: %v\ninput:\n%s", err, in)
			}
			a := mustArgs(t, calls[0])
			if a["content"] != `{"k": 1}` {
				t.Errorf("body = %#v", a["content"])
			}
			if name != "no comma, body last" && a["path"] != "c.json" {
				t.Errorf("key after the body lost: %#v", a["path"])
			}
		})
	}
}

// A reply with no call must come back untouched. Appending a terminator to
// ordinary prose put text in front of the user that the model never wrote.
func TestCloseHeredocLeavesProseAlone(t *testing.T) {
	for _, s := range []string{"done", "I could not find the file.", ""} {
		if got := CloseHeredoc(s); got != s {
			t.Errorf("CloseHeredoc(%q) = %q, want unchanged", s, got)
		}
	}
	started := "@@call w\n{a: 1}"
	if got := CloseHeredoc(started); !strings.Contains(got, HeredocEnd) {
		t.Errorf("a started call should be closed: %q", got)
	}
}

// The grammar must permit a BATCH: the parser always accepted many calls, and
// restricting generation to one silently gave up the native format's parallel
// calls. Independent work then costs one round trip per call.
func TestGrammarAllowsBoundedBatch(t *testing.T) {
	g := HeredocGrammar([]ToolDef{tdef("read_file", `{"type":"object","properties":{"path":{}}}`)})
	if !strings.Contains(g, "anycall1 anycall2?") {
		t.Errorf("root should allow more than one call:\n%s", g)
	}
	// Bounded, not `+`: unbounded repetition is what made the model re-emit the
	// same call to the token cap.
	if strings.Contains(g, "anycall+") || strings.Contains(g, "anycall*") {
		t.Errorf("batch must be bounded, not unbounded:\n%s", g)
	}
	if strings.Contains(g, fmt.Sprintf("anycall%d", MaxBatchedCalls+1)) {
		t.Errorf("batch exceeded MaxBatchedCalls:\n%s", g)
	}
}

// Parsing a batch yields separate calls with their own arguments.
func TestParsesABatch(t *testing.T) {
	in := "@@call read_file\n{path: \"a.txt\"}\n@@end\n" +
		"@@call read_file\n{path: \"b.txt\"}\n@@end\n" +
		"@@call list_dir\n{path: \".\", recursive: true}\n@@end\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 3 {
		t.Fatalf("want 3 calls, got %d", len(calls))
	}
	if calls[2].Function.Name != "list_dir" {
		t.Errorf("third call = %q", calls[2].Function.Name)
	}
	var a map[string]any
	if err := json.Unmarshal([]byte(calls[1].Function.Arguments), &a); err != nil {
		t.Fatal(err)
	}
	if a["path"] != "b.txt" {
		t.Errorf("second call args = %#v", a)
	}
}
