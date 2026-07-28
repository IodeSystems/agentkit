package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// The repair primitive, tested at its own layer. The agent-level gate that uses
// it (refusing a call, wording the tool result) is tested in agent/.
func TestRepairLooseJSON(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		outcome RepairOutcome
		want    map[string]any
	}{
		{"trailing comma", `{"a":1,}`, RepairOK, map[string]any{"a": 1.0}},
		{"single quotes", `{'a':'b'}`, RepairOK, map[string]any{"a": "b"}},
		{"unquoted keys", `{a:"b"}`, RepairOK, map[string]any{"a": "b"}},
		{"python literals", `{"a":True,"b":None}`, RepairOK, map[string]any{"a": true, "b": nil}},
		{"code fence", "```json\n{\"a\":1}\n```", RepairOK, map[string]any{"a": 1.0}},
		// Already valid: nothing to repair, so the caller's own handling stands.
		{"already valid", `{"a":1}`, RepairFailed, nil},
		{"not json at all", `hello`, RepairFailed, nil},
		// Truncation must NEVER be repaired: closing the braces would invent an
		// ending the model never wrote.
		{"truncated string", `{"a":"unterminated`, RepairTruncated, nil},
		{"truncated object", `{"a":`, RepairTruncated, nil},
		{"bare open brace", `{`, RepairTruncated, nil},
		{"loose AND truncated", `{a:'unterminated`, RepairTruncated, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fixed, _, outcome := RepairLooseJSON(c.in)
			if outcome != c.outcome {
				t.Fatalf("outcome = %v, want %v (fixed=%q)", outcome, c.outcome, fixed)
			}
			if c.outcome != RepairOK {
				return
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(fixed), &got); err != nil {
				t.Fatalf("repair produced non-JSON %q: %v", fixed, err)
			}
			for k, want := range c.want {
				if got[k] != want {
					t.Errorf("key %q = %#v, want %#v", k, got[k], want)
				}
			}
		})
	}
}

// String contents are content, not syntax: a Java snippet full of braces and
// trailing commas must survive the scanner untouched.
func TestRepairLeavesStringsAlone(t *testing.T) {
	inner := "class A {\n  int[] x = {1, 2,};\n  String s = \"don't\";\n}"
	b, _ := json.Marshal(map[string]string{"c": inner})
	loose := string(b[:len(b)-1]) + ",}"
	fixed, _, outcome := RepairLooseJSON(loose)
	if outcome != RepairOK {
		t.Fatalf("outcome=%v", outcome)
	}
	var got map[string]string
	if err := json.Unmarshal([]byte(fixed), &got); err != nil {
		t.Fatal(err)
	}
	if got["c"] != inner {
		t.Errorf("string content mangled:\n got %q\nwant %q", got["c"], inner)
	}
}

func TestIsTruncationErr(t *testing.T) {
	if !IsTruncationErr(DecodeErr(`{"a":`)) {
		t.Error("an input that stops early is truncation")
	}
	if IsTruncationErr(DecodeErr(`{"a": }`)) {
		t.Error("a syntax error is not truncation")
	}
	if IsTruncationErr(DecodeErr(`{"a":1}`)) {
		t.Error("valid JSON is not truncation")
	}
	if !strings.Contains(DecodeErr(`{"a": }`).Error(), "invalid character") {
		t.Error("syntax errors should name the offending byte")
	}
}
