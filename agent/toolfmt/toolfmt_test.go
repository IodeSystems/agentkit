package toolfmt

import (
	"encoding/json"
	"fmt"
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
	want := "file:server.go\n" +
		"lang:go\n" +
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
// \t-indent nesting, quoted delimiter values, loose-JSON array fallback,
// ≤ ~100% of JSON, and NO escaped JSON blob.
func TestEncodeTight_Nightmare(t *testing.T) {
	raw := `{"id":42,"tags":["a","b,c"],"items":[{"k":1},{"k":2,"v":"x y"},{"deep":{"meta":{"x":1,"y":2}}}],"note":"a, b | c","meta":{"deep":{"x":9}}}`
	got := EncodeTight(raw)
	if hasEscapedBlob(got) {
		t.Fatalf("nightmare emitted an escaped JSON blob (FORBIDDEN):\n%s", got)
	}
	if r := ratio(got, raw); r > 100 {
		t.Errorf("nightmare tight %.0f%% of JSON; want ≤ ~100%%", r)
	}
	// Genuine nesting via TAB indentation (meta.deep.x), scalar leaf via colon.
	if !strings.Contains(got, "meta\n\tdeep\n\t\tx:9") {
		t.Errorf("nightmare missing \\t-indented nesting for meta.deep.x:\n%s", got)
	}
	// Delimiter-laden value quoted (comma + pipe + spaces), colon delimiter.
	if !strings.Contains(got, `note:"a, b | c"`) {
		t.Errorf("nightmare should quote delimiter-laden value:\n%s", got)
	}
	// Scalar array with a comma element → bracketed + inner quote.
	if !strings.Contains(got, `tags:[a,"b,c"]`) {
		t.Errorf("nightmare scalar-array rendering wrong:\n%s", got)
	}
	// The heterogeneous/nested items array does NOT factor into a table, so tight
	// falls back to self-delimiting LOOSE JSON, inline via `items:[...]`.
	if !strings.Contains(got, `items:[{k:1},{k:2,v:"x y"},{deep:{meta:{x:1,y:2}}}]`) {
		t.Errorf("nightmare items should fall back to loose JSON inline:\n%s", got)
	}
}

// TestEncodeTight_RecoverableNotAmbiguous is the direct answer to "can a reader
// tell an array element from a sibling field?": a heterogeneous array must be
// emitted as one self-delimiting (loose-JSON) token, never as blank-separated
// indented blocks that collide with the parent's sibling separators.
func TestEncodeTight_RecoverableNotAmbiguous(t *testing.T) {
	// Array with a NESTED cell (can't be a flat table), plus a sibling scalar
	// field after it — the case that used to produce ambiguous per-element blocks.
	got := EncodeTight(`{"rows":[{"a":1},{"b":{"x":2}}],"after":7}`)
	// The array is one self-delimiting token, inline under its key.
	if !strings.Contains(got, "rows:[{a:1},{b:{x:2}}]") {
		t.Fatalf("nested-cell array should fall back to self-delimiting loose JSON:\n%s", got)
	}
	// `after` is a plain root field — unambiguously NOT an array element.
	if !strings.Contains(got, "after:7") {
		t.Errorf("sibling field lost:\n%s", got)
	}
	// No blank-line-separated bare blocks that could be misread as array elements.
	if strings.Contains(got, "\n\na:") || strings.Contains(got, "\n\nb:") {
		t.Errorf("tight must not emit ambiguous per-element blocks:\n%s", got)
	}
}

// TestEncodeTight_ArrayProbe: the table-vs-loose choice is decided by MEASURED
// bytes, not a fixed sparsity threshold. A sparse (33%-filled) array whose shared
// keys still make the union table smaller must render as the table; a very-sparse
// array with short unique keys must render as loose. In both cases tight emits the
// strictly smaller of the two recoverable candidates.
func TestEncodeTight_ArrayProbe(t *testing.T) {
	// 33% fill, but factoring the (repeated/long) keys into one header beats the
	// ,, empty-cell cost → TABLE wins even though it is "mostly empty".
	sparse := `[{"user_id":1},{"status":"active"},{"flagged":true},{"user_id":2}]`
	gotSparse := EncodeTight(sparse)
	loose := EncodeLoose(sparse)           // the loose candidate the probe compares against
	if strings.HasPrefix(gotSparse, "[") { // loose starts with '['; a table does not
		t.Errorf("sparse-but-overlapping array should PROBE to the table, got loose:\n%s", gotSparse)
	}
	if len(gotSparse) >= len(loose) {
		t.Errorf("probe must pick the smaller: table %d should be < loose %d", len(gotSparse), len(loose))
	}
	if !strings.HasPrefix(gotSparse, "user_id,status,flagged\n") {
		t.Errorf("expected union-table header:\n%s", gotSparse)
	}

	// Very sparse, short unique keys → the empties dominate, loose wins.
	vs := `[{"a":1},{"b":2},{"c":3},{"d":4},{"e":5}]`
	gotVS := EncodeTight(vs)
	if gotVS != "[{a:1},{b:2},{c:3},{d:4},{e:5}]" {
		t.Errorf("very-sparse array should PROBE to loose, got:\n%s", gotVS)
	}
}

// TestEncodeLift covers the lift add-in: dedup of REPEATED subtrees, the
// never-worse-than-tight probe, and passthrough. Lift is dedup-only — a subtree
// that appears once is never hoisted (see the un-nest case below), which is what
// keeps candidate collection O(n) instead of probing every unique subnode.
func TestEncodeLift(t *testing.T) {
	// (a) DEDUP: an identical nested subtree under two keys is hoisted once and
	// referenced twice — strictly smaller than tight.
	dedup := `{"groupA":{"users":[{"id":1,"role":"admin"},{"id":2,"role":"user"}]},"groupB":{"users":[{"id":1,"role":"admin"},{"id":2,"role":"user"}]}}`
	gotD := EncodeLift(dedup)
	if len(gotD) >= len(EncodeTight(dedup)) {
		t.Errorf("dedup lift (%d) should beat tight (%d):\n%s", len(gotD), len(EncodeTight(dedup)), gotD)
	}
	if strings.Count(gotD, "$0") < 3 { // 2 references + 1 definition anchor
		t.Errorf("expected $0 referenced twice + defined once:\n%s", gotD)
	}
	if !strings.Contains(gotD, "$0\n") { // a `$0` definition anchor on its own line
		t.Errorf("expected a $0 definition block:\n%s", gotD)
	}

	// (b) SINGLE OCCURRENCE is NOT hoisted: a good table trapped inside a
	// heterogeneous (loose) array appears once, so dedup-only lift leaves it as
	// tight. (The old un-nest optimization hoisted it, but that required probing
	// every unique subnode — the O(n²) source — for ~0 real-corpus gain.)
	unnest := `{"items":[{"k":1},{"rows":[{"a":1,"b":2},{"a":3,"b":4},{"a":5,"b":6},{"a":7,"b":8}]}]}`
	if EncodeLift(unnest) != EncodeTight(unnest) {
		t.Errorf("single-occurrence subtree must reduce to tight:\n%s", EncodeLift(unnest))
	}

	// (c) DEDUP works even on LARGE input (the O(n²) probe would have hung here):
	// 200 rows sharing one repeated nested subtree hoist to a single $0.
	var b strings.Builder
	b.WriteString(`{"rows":[`)
	for i := 0; i < 200; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, `{"i":%d,"meta":{"kind":"widget","tier":"gold","region":"us-west-2"}}`, i)
	}
	b.WriteString(`]}`)
	big := b.String()
	gotBig := EncodeLift(big)
	if len(gotBig) >= len(EncodeTight(big)) {
		t.Errorf("repeated-subtree lift (%d) should beat tight (%d) on large input", len(gotBig), len(EncodeTight(big)))
	}
	if !strings.Contains(gotBig, "$0") {
		t.Errorf("large repeated subtree should hoist to $0")
	}

	// NEVER WORSE than tight: for any payload, lift ≤ tight (byte-probed).
	for _, raw := range []string{stdPayload, dedup, unnest, big,
		`{"file":"a.go","#":[{"sym":"S","class":"m","@":[1,2]},{"sym":"main","class":"f","@":[3,4]}]}`,
		`{"events":[{"ts":1,"level":"info"},{"ts":2,"level":"warn"}]}`} {
		if len(EncodeLift(raw)) > len(EncodeTight(raw)) {
			t.Errorf("lift must never exceed tight for %q", raw)
		}
	}

	// No repetition / nothing trapped → lift reduces to EXACTLY tight.
	plain := `{"file":"a.go","#":[{"sym":"S","class":"m","@":[1,2]},{"sym":"main","class":"f","@":[3,4]}]}`
	if EncodeLift(plain) != EncodeTight(plain) {
		t.Errorf("lift should equal tight when nothing is worth hoisting")
	}

	// Passthrough.
	if EncodeLift(prose) != prose {
		t.Error("lift should passthrough prose")
	}
	if EncodeLift(malformed) != malformed {
		t.Error("lift should passthrough malformed JSON")
	}
}

// TestEncodeTightC covers the comprehension-tuned variant: count-anchored uniform
// tables, non-uniform arrays and nested objects dropping to ORDER-PRESERVING loose.
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
	// All-scalar object still inlines (tightc keeps tight's cheap win).
	if got := EncodeTightC(`{"n":42,"s":"hi"}`); got != "n:42,s:hi" {
		t.Errorf("all-scalar object should inline: %q", got)
	}
	// Passthrough.
	if EncodeTightC(prose) != prose || EncodeTightC(malformed) != malformed {
		t.Error("tightc should passthrough prose / malformed JSON")
	}
}

// Per-strategy unit checks.
func TestEncodeTight_Strategies(t *testing.T) {
	// 1 & 6: all-scalar object → inline `key:val,key:val`; literal-ambiguous
	// strings quoted (type preserved).
	if got := EncodeTight(`{"n":42,"s":"hi","b":true,"z":null,"num":"42"}`); got != `n:42,s:hi,b:true,z:null,num:"42"` {
		t.Errorf("all-scalar inline / literal-ambiguity: %q", got)
	}
	// 2: prose (multiline JSON string) → verbatim, no quotes/escaping.
	if got := EncodeTight("\"line1\\nline2\""); got != "line1\nline2" {
		t.Errorf("prose passthrough: %q", got)
	}
	// 3: scalar array → inline brackets, colon delimiter.
	if got := EncodeTight(`{"tags":["a","b"]}`); got != "tags:[a,b]" {
		t.Errorf("scalar array: %q", got)
	}
	// 7: nested object → \t per level, scalar leaf via colon.
	if got := EncodeTight(`{"a":{"b":{"c":1}}}`); got != "a\n\tb\n\t\tc:1" {
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
