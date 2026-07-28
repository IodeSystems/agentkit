package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// The exact shape that poisoned a real session: a large newText argument cut
// off mid-string by a full context window.
func truncatedArgs() string {
	return `{"node":"app/src/test/java/com/termux/app/ScrollKeysTest.java","newText":"package com.termux.app;\n\npublic class ScrollKeysTest {\n\n    @Test\n   `
}

func TestTruncatedArgumentsAreNotValidJSON(t *testing.T) {
	if json.Valid([]byte(truncatedArgs())) {
		t.Fatal("fixture is supposed to be unparseable")
	}
	// and a well-formed one must pass
	if !json.Valid([]byte(`{"node":"a.java","newText":"ok"}`)) {
		t.Fatal("valid arguments rejected")
	}
}

// invalidArgs is the single gate before dispatch. What it says matters as much
// as whether it fires: truncation and a syntax error call for different retries.
func TestInvalidArgsClassification(t *testing.T) {
	cases := []struct {
		name  string
		args  string
		want  string // substring the result must contain; "" means dispatchable
		alsoN string // substring it must NOT contain
	}{
		{name: "well-formed object", args: `{"a":1}`, want: ""},
		{name: "no arguments at all", args: ``, want: ""},
		{name: "whitespace only", args: "  \n", want: ""},
		{name: "truncated mid-string", args: truncatedArgs(), want: "truncation detected"},
		{name: "bare open brace", args: `{`, want: "truncation detected"},
		{name: "key with no value", args: `{"a":`, want: "truncation detected"},
		{
			name:  "syntax error is not truncation",
			args:  `{"a": }`,
			want:  "malformed json",
			alsoN: "truncation detected",
		},
		// Valid JSON that is not an object passes json.Valid but satisfies no
		// tool schema, and renders as zero parameters under the Qwen template.
		{name: "bare string", args: `"hello"`, want: "must be a JSON object"},
		{name: "array", args: `[1,2]`, want: "must be a JSON object"},
		{name: "number", args: `42`, want: "must be a JSON object"},
		{name: "null", args: `null`, want: "must be a JSON object"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, got := checkArgs(c.args)
			if c.want == "" {
				if got != "" {
					t.Fatalf("should dispatch, got refusal: %s", got)
				}
				return
			}
			if !strings.Contains(got, c.want) {
				t.Fatalf("want %q in result, got %q", c.want, got)
			}
			if c.alsoN != "" && strings.Contains(got, c.alsoN) {
				t.Fatalf("result should not claim %q: %s", c.alsoN, got)
			}
			if !strings.Contains(got, "NOT executed") {
				t.Errorf("the model must be told the call did not run: %s", got)
			}
		})
	}
}

// The placeholder must be VALID JSON — that is the invariant that stops a
// session being poisoned — while still carrying the tail so the model knows
// where its own output stopped.
func TestRejectedArgsPlaceholderIsValidJSON(t *testing.T) {
	raw := `{"node":"a.java","newText":"package com.termux.app;` + strings.Repeat("x", 5000) + "UNIQUE_TAIL"
	_, why := checkArgs(raw)
	if why == "" {
		t.Fatal("fixture should have been refused")
	}
	got := rejectedArgsPlaceholder(raw, why)

	if !json.Valid([]byte(got)) {
		t.Fatalf("placeholder is not valid JSON — it would poison the session:\n%s", got)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatal(err)
	}
	// A provider renders arguments by iterating keys; it must be an object.
	if e, _ := m["_error"].(string); e != why {
		t.Errorf("persisted call and tool result must give the same reason:\n call: %q\n result: %q", e, why)
	}
	if n, ok := m["_raw_chars"].(float64); !ok || int(n) != len(raw) {
		t.Errorf("should record the original size, got %v", m["_raw_chars"])
	}
	tail, _ := m["_tail"].(string)
	if !strings.Contains(tail, "UNIQUE_TAIL") {
		t.Error("tail dropped — that is the part that says where to resume")
	}
	if len(got) > 2000 {
		t.Errorf("placeholder should stay bounded, got %d chars", len(got))
	}
}

// Preserving the CALL (rather than dropping it) is what keeps the exchange
// paired: the loop records a tool_result against the same id regardless, so a
// dropped call would leave an orphan result.
func TestPlaceholderKeepsCallPairable(t *testing.T) {
	raw := `{"a":"unterminated`
	got := rejectedArgsPlaceholderFor(raw)
	if !json.Valid([]byte(got)) {
		t.Fatal("must be parseable")
	}
	if strings.Contains(got, `"unterminated`) && !strings.Contains(got, `_tail`) {
		t.Error("raw unterminated fragment must only appear inside the escaped _tail field")
	}
}
