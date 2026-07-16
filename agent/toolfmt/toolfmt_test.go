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

func TestEncodeYAML(t *testing.T) {
	got := EncodeYAML(polyLSP)
	for _, want := range []string{"file: a.go", "lang: go", "sym: S.Start", "class: method", "sym: main"} {
		if !strings.Contains(got, want) {
			t.Errorf("YAML missing %q:\n%s", want, got)
		}
	}
	// numbers stay bare, not quoted strings
	if strings.Contains(got, `"22"`) {
		t.Errorf("YAML quoted a number:\n%s", got)
	}
	if EncodeYAML(prose) != prose {
		t.Error("YAML should passthrough prose")
	}
	if EncodeYAML(malformed) != malformed {
		t.Error("YAML should passthrough malformed JSON")
	}
}

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
