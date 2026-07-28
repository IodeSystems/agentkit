package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// Loose-but-whole arguments are repaired rather than refused. Refusing costs a
// full regeneration, and generated tokens are the priciest channel.
func TestCheckArgsRepairsLooseButWholeJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]any
	}{
		{
			name: "trailing comma",
			in:   `{"path":"a.java","n":3,}`,
			want: map[string]any{"path": "a.java", "n": 3.0},
		},
		{
			name: "single quotes",
			in:   `{'path':'a.java'}`,
			want: map[string]any{"path": "a.java"},
		},
		{
			name: "unquoted keys",
			in:   `{path:"a.java",n:3}`,
			want: map[string]any{"path": "a.java", "n": 3.0},
		},
		{
			name: "python literals",
			in:   `{"path":"a.java","force":True,"prev":None}`,
			want: map[string]any{"path": "a.java", "force": true, "prev": nil},
		},
		{
			name: "markdown code fence",
			in:   "```json\n{\"path\":\"a.java\"}\n```",
			want: map[string]any{"path": "a.java"},
		},
		{
			name: "several at once",
			in:   "```\n{path:'a.java', force:True,}\n```",
			want: map[string]any{"path": "a.java", "force": true},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			args, refusal := checkArgs(c.in)
			if refusal != "" {
				t.Fatalf("refused a repairable call: %s", refusal)
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(args), &got); err != nil {
				t.Fatalf("repair produced non-JSON %q: %v", args, err)
			}
			for k, want := range c.want {
				if got[k] != want {
					t.Errorf("key %q = %#v, want %#v (full: %s)", k, got[k], want, args)
				}
			}
			if len(got) != len(c.want) {
				t.Errorf("key count %d, want %d: %s", len(got), len(c.want), args)
			}
		})
	}
}

// THE LINE THAT MUST NOT MOVE. A truncated payload is never repaired: closing
// its braces would hand a dispatcher a call the model never finished writing,
// with a half-written file inside it. Truncation is diagnosed first and repair
// never sees it.
func TestCheckArgsNeverRepairsTruncation(t *testing.T) {
	truncated := []string{
		`{"path":"a.java","content":"public class Big {`,
		`{"path":"a.java",`,
		`{`,
		`{"path":"a.java","items":[1,2,`,
		// Loose AND truncated: the looseness must not buy it a repair.
		`{path:'a.java', content:'public class Big {`,
	}
	for _, in := range truncated {
		t.Run(in[:min(len(in), 24)], func(t *testing.T) {
			args, refusal := checkArgs(in)
			if refusal == "" {
				t.Fatalf("REPAIRED A TRUNCATED CALL — it would be dispatched: %q", args)
			}
			if !strings.Contains(refusal, "truncation detected") {
				t.Errorf("should be reported as truncation, got: %s", refusal)
			}
			if args != in {
				t.Errorf("truncated input must be returned untouched, got %q", args)
			}
		})
	}
}

// Content inside a string literal is content, not syntax. A Java snippet full of
// braces, quotes and trailing commas must survive the scanner untouched.
func TestRepairLeavesStringContentAlone(t *testing.T) {
	inner := "class A {\n  int[] x = {1, 2,};\n  String s = \"don't\";\n}"
	b, err := json.Marshal(map[string]string{"path": "A.java", "content": inner})
	if err != nil {
		t.Fatal(err)
	}
	// Well-formed already: must pass through byte-identical, no repair attempted.
	args, refusal := checkArgs(string(b))
	if refusal != "" {
		t.Fatalf("refused valid arguments: %s", refusal)
	}
	if args != string(b) {
		t.Fatalf("valid arguments were rewritten:\n got %s\nwant %s", args, b)
	}
	// And with a genuine trailing comma bolted on, only that comma changes.
	loose := string(b[:len(b)-1]) + ",}"
	args, refusal = checkArgs(loose)
	if refusal != "" {
		t.Fatalf("refused a repairable call: %s", refusal)
	}
	var got map[string]string
	if err := json.Unmarshal([]byte(args), &got); err != nil {
		t.Fatal(err)
	}
	if got["content"] != inner {
		t.Errorf("string content was mangled:\n got %q\nwant %q", got["content"], inner)
	}
}

// A repair that cannot produce a strict JSON object degrades to the ordinary
// refusal. It must never fall through to a dispatch.
func TestUnrepairableFallsBackToRefusal(t *testing.T) {
	for _, in := range []string{
		`{"path": <function>}`,
		`not json at all`,
		`{"a" "b"}`,
	} {
		args, refusal := checkArgs(in)
		if refusal == "" {
			t.Errorf("%q was accepted as %q", in, args)
		}
	}
}

// Repair must not rescue a non-object: valid JSON that is not a map still
// renders zero parameters under the Qwen template.
func TestRepairDoesNotRescueNonObjects(t *testing.T) {
	for _, in := range []string{`"hello"`, `[1,2]`, `42`} {
		if _, refusal := checkArgs(in); !strings.Contains(refusal, "must be a JSON object") {
			t.Errorf("%q: got %q", in, refusal)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// rejectedArgsPlaceholderFor is the test-side pairing of a refusal with its
// stand-in, matching what the turn loop does.
func rejectedArgsPlaceholderFor(raw string) string {
	_, why := checkArgs(raw)
	return rejectedArgsPlaceholder(raw, why)
}
