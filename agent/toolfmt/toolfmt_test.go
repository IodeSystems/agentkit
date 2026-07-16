package toolfmt

import (
	"strings"
	"testing"
)

const prose = "--- FAIL: TestX\n  foo.go:12: expected 3 got 4\nFAIL"
const malformed = `{"file":"a.go","lang":` // truncated

func TestEncodeTightC(t *testing.T) {
	// Uniform object array → `[N]header` count anchor + dense rows.
	uni := `[{"id":1,"name":"a","yrs":12},{"id":2,"name":"b","yrs":8}]`
	if got := EncodeTightC(uni); got != "[2]id,name,yrs\n1,a,12\n2,b,8" {
		t.Errorf("uniform table w/ count anchor:\n%s", got)
	}
	// Non-uniform array (differing key sets) → loose, NOT a sparse ,, table.
	nonuni := `[{"id":1,"name":"a"},{"id":2,"role":"x"}]`
	if got := EncodeTightC(nonuni); strings.Contains(got, "\n") || !strings.HasPrefix(got, "[{") {
		t.Errorf("non-uniform array should be loose (no table):\n%s", got)
	}
	// A row with a missing column is NOT uniform → loose (no empty cells).
	if got := EncodeTightC(`[{"a":1,"b":2},{"a":3}]`); strings.Contains(got, ",,") || strings.Contains(got, "\n") {
		t.Errorf("missing-cell array must not become a sparse table:\n%s", got)
	}
	// Object with mixed children → per-key: scalar inline, nested object as
	// ORDER-PRESERVING loose braces, uniform array child as a count-anchor table.
	nested := `{"order":"O6","cust":{"id":9,"tier":"gold"},"items":[{"sku":"x","q":2}]}`
	if got := EncodeTightC(nested); got != "order:O6\ncust:{id:9,tier:gold}\nitems\n[1]sku,q\nx,2" {
		t.Errorf("nested per-key dispatch:\n%s", got)
	}
	// A uniform table nested INSIDE an object is still a table, not flattened.
	if got := EncodeTightC(`{"employees":[{"id":1,"name":"a"},{"id":2,"name":"b"}]}`); got != "employees\n[2]id,name\n1,a\n2,b" {
		t.Errorf("table inside object must survive:\n%s", got)
	}
	// A genuinely-nested object VALUE keeps key order (not alphabetized).
	if got := EncodeTightC(`{"a":{"z":1,"m":2,"b":3}}`); got != "a:{z:1,m:2,b:3}" {
		t.Errorf("nested object order must be preserved:\n%s", got)
	}
	// All-scalar object still inlines (the cheap win).
	if got := EncodeTightC(`{"n":42,"s":"hi"}`); got != "n:42,s:hi" {
		t.Errorf("all-scalar object should inline: %q", got)
	}
	// Passthrough of non-JSON.
	if EncodeTightC(prose) != prose || EncodeTightC(malformed) != malformed {
		t.Error("tightc should passthrough prose / malformed JSON")
	}
}

// TestEncodeTightC_Reversible locks the leading-bracket quoting rule: a value or
// key that begins with '[' or '{' must be QUOTED, else a decoder would misread it
// as a nested array/object. Covers real cases (regexes, markdown links).
func TestEncodeTightC_Reversible(t *testing.T) {
	cases := map[string]string{
		`{"pattern":"[a-z]+"}`:   `pattern:"[a-z]+"`,
		`{"md":"[t](u)"}`:        `md:"[t](u)"`,
		`{"v":"{x"}`:             `v:"{x"`,
		`{"[2]x":1,"z":2}`:       `"[2]x":1,z:2`,
		`{"num":"42","real":42}`: `num:"42",real:42`,
		`{"t":"true","b":true}`:  `t:"true",b:true`,
	}
	for in, want := range cases {
		if got := EncodeTightC(in); got != want {
			t.Errorf("EncodeTightC(%s) = %q, want %q", in, got, want)
		}
	}
}
