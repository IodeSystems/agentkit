package toolfmt

import (
	"encoding/json"
	"strings"
	"testing"
)

// polyLSP is the canonical poly-lsp-style structure result: an object with two
// scalar fields and one uniform array of flat objects (nested `@` line-range).
const polyLSP = `{"file":"a.go","lang":"go","#":[{"sym":"S.Start","class":"method","@":[22,35]},{"sym":"main","class":"func","@":[42,50]}]}`

const prose = "--- FAIL: TestX\n  foo.go:12: expected 3 got 4\nFAIL"

const malformed = `{"file":"a.go","lang":` // truncated

// topArray is a top-level uniform array of flat objects.
const topArray = `[{"sym":"S.Start","class":"method"},{"sym":"main","class":"func"}]`

// nested is not clean flat tabular (nested object value) → CSV passthrough.
const nested = `{"a":{"b":1},"c":[1,2,3]}`

// stdPayload is the coordinator's 3-row standard payload (dotted path in
// Server.Start exercises "hierarchy as paths").
const stdPayload = `{"file":"server.go","lang":"go","#":[{"sym":"Server","class":"struct","@":[10,40]},{"sym":"Server.Start","class":"method","@":[22,35]},{"sym":"main","class":"func","@":[42,50]}]}`

func TestEncodeTOON(t *testing.T) {
	got := EncodeTOON(polyLSP)
	if !strings.Contains(got, "file: a.go") || !strings.Contains(got, "lang: go") {
		t.Errorf("TOON missing scalar fields:\n%s", got)
	}
	if !strings.Contains(got, "#[2]{@,class,sym}:") {
		t.Errorf("TOON missing header:\n%s", got)
	}
	for _, want := range []string{"[22,35],method,S.Start", "[42,50],func,main"} {
		if !strings.Contains(got, want) {
			t.Errorf("TOON missing row %q:\n%s", want, got)
		}
	}
	if len(got) >= len(polyLSP) {
		t.Errorf("TOON not shorter: %d >= %d", len(got), len(polyLSP))
	}
	if EncodeTOON(prose) != prose {
		t.Error("TOON should passthrough prose")
	}
	if EncodeTOON(malformed) != malformed {
		t.Error("TOON should passthrough malformed JSON")
	}
}

func TestEncodeTOONTopArray(t *testing.T) {
	got := EncodeTOON(topArray)
	if !strings.HasPrefix(got, "[2]{class,sym}:") {
		t.Errorf("TOON top-array header wrong:\n%s", got)
	}
	if !strings.Contains(got, "method,S.Start") || !strings.Contains(got, "func,main") {
		t.Errorf("TOON top-array rows missing:\n%s", got)
	}
}

func TestEncodeCSV(t *testing.T) {
	got := EncodeCSV(polyLSP)
	for _, want := range []string{"# file=a.go", "# lang=go", "@,class,sym", `"[22,35]",method,S.Start`, `"[42,50]",func,main`} {
		if !strings.Contains(got, want) {
			t.Errorf("CSV missing %q:\n%s", want, got)
		}
	}
	if len(got) >= len(polyLSP) {
		t.Errorf("CSV not shorter: %d >= %d", len(got), len(polyLSP))
	}
	if EncodeCSV(prose) != prose {
		t.Error("CSV should passthrough prose")
	}
	if EncodeCSV(malformed) != malformed {
		t.Error("CSV should passthrough malformed JSON")
	}
}

func TestEncodeCSVTopArray(t *testing.T) {
	got := EncodeCSV(topArray)
	lines := strings.Split(got, "\n")
	if len(lines) != 3 {
		t.Fatalf("CSV top-array want 3 lines, got %d:\n%s", len(lines), got)
	}
	if lines[0] != "class,sym" {
		t.Errorf("CSV header wrong: %q", lines[0])
	}
}

func TestEncodeCSVFallback(t *testing.T) {
	// Nested / non-flat → passthrough raw JSON unchanged.
	if EncodeCSV(nested) != nested {
		t.Errorf("CSV should passthrough nested JSON, got:\n%s", EncodeCSV(nested))
	}
	// Scalar top-level → passthrough.
	if EncodeCSV(`42`) != `42` {
		t.Error("CSV should passthrough scalar JSON")
	}
}

func TestEncodeLoose(t *testing.T) {
	got := EncodeLoose(polyLSP)

	// Loose JSON must re-parse to the SAME data (info-preserving). A tolerant
	// re-quoting recovers strict JSON, but the simplest proof is structural:
	// keys/safe values are bare, structure + numbers intact.
	for _, want := range []string{
		"file:a.go",    // safe key + safe value, both bare
		"lang:go",      // bare
		"class:method", // bare key + value
		"sym:S.Start",  // S.Start safe (dot allowed in value)
		"class:func",   // second row keeps per-row keys (not tabular)
		"sym:main",     //
		"[22,35]",      // numbers bare, structure intact
		"[42,50]",      //
		"#:[",          // widened rule: `#` has no structural delimiter → BARE
		"@:[",          // widened rule: `@` → BARE
	} {
		if !strings.Contains(got, want) {
			t.Errorf("loose missing %q:\n%s", want, got)
		}
	}
	// The widened rule must NOT quote `#`/`@` keys.
	if strings.Contains(got, `"#":`) || strings.Contains(got, `"@":`) {
		t.Errorf("loose should emit #/@ keys bare (no useless quotes):\n%s", got)
	}
	// Per-row keys retained (structural, NOT tabular like TOON).
	if strings.Count(got, "sym:") != 2 {
		t.Errorf("loose should keep per-row keys (want 2 sym:), got %d:\n%s", strings.Count(got, "sym:"), got)
	}
	// Shorter than strict JSON (quote tax removed).
	if len(got) >= len(polyLSP) {
		t.Errorf("loose not shorter: %d >= %d", len(got), len(polyLSP))
	}
	// Still valid... loose is NOT valid strict JSON, but must be recoverable.
	// Assert the risky cases stay QUOTED.
	mix := `{"cmd":"go test -v","region":"us-west-2","empty":"","numish":"42","boolish":"true","nullish":"null"}`
	lm := EncodeLoose(mix)
	for _, want := range []string{
		`cmd:"go test -v"`, // whitespace → quoted
		"region:us-west-2", // safe → bare
		`empty:""`,         // empty → quoted
		`numish:"42"`,      // number-ambiguous → quoted
		`boolish:"true"`,   // bool-ambiguous → quoted
		`nullish:"null"`,   // null-ambiguous → quoted
	} {
		if !strings.Contains(lm, want) {
			t.Errorf("loose mix missing %q:\n%s", want, lm)
		}
	}
	// Hyphen/digit keys are safe now (no structural delimiter) → BARE.
	if k := EncodeLoose(`{"a-b":1,"123":2}`); !strings.Contains(k, "a-b:1") || !strings.Contains(k, "123:2") {
		t.Errorf("hyphen/digit key should be bare: %s", k)
	}
	// A key with a real structural delimiter / whitespace stays quoted.
	if k := EncodeLoose(`{"a b":1,"c:d":2}`); !strings.Contains(k, `"a b":1`) || !strings.Contains(k, `"c:d":2`) {
		t.Errorf("delimiter/whitespace key must stay quoted: %s", k)
	}
	if EncodeLoose(prose) != prose {
		t.Error("loose should passthrough prose")
	}
	if EncodeLoose(malformed) != malformed {
		t.Error("loose should passthrough malformed JSON")
	}
}

// ratio returns len(out)/len(raw) as a percentage.
func ratio(out, raw string) float64 { return 100 * float64(len(out)) / float64(len(raw)) }

// hasEscapedBlob reports whether s contains a re-serialized JSON value as an
// escaped string (the forbidden `\"`-blob path).
func hasEscapedBlob(s string) bool { return strings.Contains(s, `\"`) }

// ACCEPTANCE 1: uniform poly-lsp → header + rows table, ≤ ~55% of JSON.
func TestEncodeTight_Uniform(t *testing.T) {
	got := EncodeTight(stdPayload)
	want := "file server.go\n" +
		"lang go\n" +
		"#\n" +
		"sym,class,@\n" +
		"Server,struct,[10,40]\n" +
		"Server.Start,method,[22,35]\n" +
		"main,func,[42,50]"
	if got != want {
		t.Fatalf("tight uniform mismatch:\n got:\n%s\n want:\n%s", got, want)
	}
	if r := ratio(got, stdPayload); r > 60 {
		t.Errorf("uniform tight %.0f%% of JSON; want ≤ ~55%%", r)
	}
	if toon := EncodeTOON(stdPayload); len(got) >= len(toon) {
		t.Errorf("tight (%d) not shorter than toon (%d)", len(got), len(toon))
	}
	if hasEscapedBlob(got) {
		t.Errorf("tight emitted an escaped JSON blob:\n%s", got)
	}
}

// ACCEPTANCE 2: semi-uniform events → SPARSE UNION TABLE with ,, empty cells,
// < 70% of JSON, never an escaped string.
func TestEncodeTight_SemiUniformSparseTable(t *testing.T) {
	raw := `{"events":[{"ts":1,"level":"info","msg":"start"},{"ts":2,"level":"warn","msg":"slow","dur":300},{"ts":3,"level":"error","msg":"boom","code":9,"trace":"x.go:1"}]}`
	got := EncodeTight(raw)
	// One union header naming every key seen.
	if !strings.Contains(got, "ts,level,msg,dur,code,trace") {
		t.Errorf("missing union header:\n%s", got)
	}
	// Missing keys → empty cells (,,).
	for _, row := range []string{"1,info,start,,,", "2,warn,slow,300,,", "3,error,boom,,9,x.go:1"} {
		if !strings.Contains(got, row) {
			t.Errorf("missing sparse row %q:\n%s", row, got)
		}
	}
	if hasEscapedBlob(got) {
		t.Fatalf("semi-uniform emitted an escaped JSON blob (FORBIDDEN):\n%s", got)
	}
	if r := ratio(got, raw); r >= 70 {
		t.Errorf("semi-uniform tight %.0f%% of JSON; want < 70%%", r)
	}
}

// ACCEPTANCE 3: nightmare (non-uniform + nested + delimiter-laden values) →
// per-element recursion, \t-indent nesting, quoted delimiter values, ≤ ~100% of
// JSON, and NO escaped JSON blob.
func TestEncodeTight_Nightmare(t *testing.T) {
	raw := `{"id":42,"tags":["a","b,c"],"items":[{"k":1},{"k":2,"v":"x y"},{"deep":{"meta":{"x":1,"y":2}}}],"note":"a, b | c","meta":{"deep":{"x":9}}}`
	got := EncodeTight(raw)
	if hasEscapedBlob(got) {
		t.Fatalf("nightmare emitted an escaped JSON blob (FORBIDDEN):\n%s", got)
	}
	if r := ratio(got, raw); r > 100 {
		t.Errorf("nightmare tight %.0f%% of JSON; want ≤ ~100%%", r)
	}
	// Genuine nesting via TAB indentation (meta.deep.x).
	if !strings.Contains(got, "meta\n\tdeep\n\t\tx 9") {
		t.Errorf("nightmare missing \\t-indented nesting for meta.deep.x:\n%s", got)
	}
	// Delimiter-laden value quoted (comma + pipe + spaces).
	if !strings.Contains(got, `note "a, b | c"`) {
		t.Errorf("nightmare should quote delimiter-laden value:\n%s", got)
	}
	// Scalar array with a comma element → bracketed + inner quote.
	if !strings.Contains(got, `tags [a,"b,c"]`) {
		t.Errorf("nightmare scalar-array rendering wrong:\n%s", got)
	}
	// Heterogeneous items array → per-element (blank-separated), NOT a broken table.
	if !strings.Contains(got, "k 1\n\nk 2, v \"x y\"") {
		t.Errorf("nightmare items should render per-element:\n%s", got)
	}
}

// Per-strategy unit checks.
func TestEncodeTight_Strategies(t *testing.T) {
	// 1 & 6: all-scalar object → inline; literal-ambiguous strings quoted (type).
	if got := EncodeTight(`{"n":42,"s":"hi","b":true,"z":null,"num":"42"}`); got != `n 42, s hi, b true, z null, num "42"` {
		t.Errorf("all-scalar inline / literal-ambiguity: %q", got)
	}
	// 2: prose (multiline JSON string) → verbatim, no quotes/escaping.
	if got := EncodeTight("\"line1\\nline2\""); got != "line1\nline2" {
		t.Errorf("prose passthrough: %q", got)
	}
	// 3: scalar array → inline brackets.
	if got := EncodeTight(`{"tags":["a","b"]}`); got != "tags [a,b]" {
		t.Errorf("scalar array: %q", got)
	}
	// 7: nested object → \t per level.
	if got := EncodeTight(`{"a":{"b":{"c":1}}}`); got != "a\n\tb\n\t\tc 1" {
		t.Errorf("nested \\t indent: %q", got)
	}
	// 8: empties → explicit markers.
	if got := EncodeTight(`[]`); got != "[]" {
		t.Errorf("empty array: %q", got)
	}
	if got := EncodeTight(`{}`); got != "{}" {
		t.Errorf("empty object: %q", got)
	}
	// non-JSON prose + malformed → passthrough.
	if EncodeTight(prose) != prose {
		t.Error("tight should passthrough prose")
	}
	if EncodeTight(malformed) != malformed {
		t.Error("tight should passthrough malformed JSON")
	}
}

func TestEncodeJSONTOON(t *testing.T) {
	got := EncodeJSONTOON(polyLSP)

	// Envelope must still be valid JSON.
	var env map[string]any
	if err := json.Unmarshal([]byte(got), &env); err != nil {
		t.Fatalf("json-toon envelope not valid JSON: %v\n%s", err, got)
	}
	// Scalars preserved as JSON values.
	if env["file"] != "a.go" || env["lang"] != "go" {
		t.Errorf("json-toon lost scalar fields: %v", env)
	}
	// The `#` field is now a TOON table STRING.
	hash, ok := env["#"].(string)
	if !ok {
		t.Fatalf("json-toon `#` not a string: %T", env["#"])
	}
	if !strings.HasPrefix(hash, "[2]{@,class,sym}: ") {
		t.Errorf("json-toon `#` header wrong: %q", hash)
	}
	for _, want := range []string{"[22,35],method,S.Start", "[42,50],func,main", " | "} {
		if !strings.Contains(hash, want) {
			t.Errorf("json-toon `#` missing %q: %q", want, hash)
		}
	}
	if len(got) >= len(polyLSP) {
		t.Errorf("json-toon not shorter: %d >= %d", len(got), len(polyLSP))
	}
	if EncodeJSONTOON(prose) != prose {
		t.Error("json-toon should passthrough prose")
	}
	if EncodeJSONTOON(malformed) != malformed {
		t.Error("json-toon should passthrough malformed JSON")
	}
}

func TestEncodeJSONTOONNested(t *testing.T) {
	// Uniform arrays nested deeper than top level are still compressed.
	raw := `{"outer":{"rows":[{"a":1,"b":2},{"a":3,"b":4}]}}`
	got := EncodeJSONTOON(raw)
	var env map[string]any
	if err := json.Unmarshal([]byte(got), &env); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, got)
	}
	outer := env["outer"].(map[string]any)
	rows, ok := outer["rows"].(string)
	if !ok {
		t.Fatalf("nested `rows` not compressed to string: %T", outer["rows"])
	}
	if !strings.HasPrefix(rows, "[2]{a,b}: ") {
		t.Errorf("nested toon string wrong: %q", rows)
	}
}
