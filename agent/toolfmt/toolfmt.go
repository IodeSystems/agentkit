// Package toolfmt provides pluggable, information-preserving re-encoders for
// tool-call RESULTS before they enter an LLM's context. The goal is fewer
// tokens for the same data: JSON is verbose (repeated keys, punctuation), so a
// uniform array of flat objects is far cheaper as TOON/CSV/loose/tight.
//
// Every encoder shares one contract:
//
//   - Input that is NOT valid JSON (prose — file contents, `go test` output,
//     plain text) is returned UNCHANGED. Only structured JSON is transformed.
//   - The transform is information-preserving: an LLM reading the encoded form
//     has every datum the JSON had. CSV is the one exception — it only handles
//     clean flat tabular data and PASSES THROUGH the raw JSON otherwise
//     (documented per-function).
//
// JSON itself is the identity encoding — there is no EncodeJSON; the caller
// simply leaves Session.EncodeToolResult nil.
//
// Column ordering: for tabular output (TOON/CSV/json-toon) columns are emitted
// in sorted key order for determinism (Go map iteration + json.Marshal do not
// preserve JSON insertion order). Object key order in the json-toon envelope is
// likewise sorted by json.Marshal. This is a deliberate, documented choice —
// the encoders aim for terse + unambiguous + information-preserving, not
// byte-identical key ordering with the source.
package toolfmt

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// EncodeTOON re-encodes a JSON tool result as Token-Oriented Object Notation:
// a uniform array of flat objects states its keys ONCE as a header, then emits
// one comma-joined row per element — eliminating the per-row key repetition
// that makes JSON verbose.
//
// Handled shapes:
//   - Top-level uniform array of flat objects → `[N]{col1,col2,...}:` header +
//     one row per line.
//   - Top-level object with scalar fields plus exactly one array-valued field
//     that is a uniform array of flat objects → scalar fields as `key: val`
//     lines, then `<arrayKey>[N]{cols}:` header + rows.
//
// Anything else (non-uniform, deeply nested, scalar) falls back to compact JSON,
// which is fully information-preserving. Non-JSON input passes through unchanged.
func EncodeTOON(raw string) string {
	v, ok := parseJSON(raw)
	if !ok {
		return raw
	}

	switch t := v.(type) {
	case []any:
		if cols, rows, ok := tabulate(t); ok {
			return toonMultiline("", cols, rows)
		}
	case map[string]any:
		arrKey, arr, scalars, ok := singleArrayObject(t)
		if ok {
			if cols, rows, uok := tabulate(arr); uok {
				var b strings.Builder
				for _, kv := range scalars {
					b.WriteString(kv[0].(string))
					b.WriteString(": ")
					b.WriteString(scalarString(kv[1]))
					b.WriteByte('\n')
				}
				b.WriteString(toonMultiline(arrKey, cols, rows))
				return b.String()
			}
		}
	}

	// Fallback: compact JSON, still fully information-preserving.
	return compactJSON(v)
}

// EncodeCSV re-encodes a JSON tool result as CSV — the tersest form for flat
// uniform tabular data. CSV is TOON's rows without the `[N]{cols}` header/count
// decoration (an RFC4180 header row instead), so it shares the same row
// tabulation.
//
// Handled shapes:
//   - Top-level uniform array of flat objects → header row + data rows
//     (RFC4180 quoting; nested array/object cells JSON-stringified).
//   - Top-level object with scalar fields plus exactly one array-valued field
//     that is a uniform array of flat objects (the poly-lsp `{"#":[...]}` /
//     `{"matches":[...]}` shape) → the scalar fields become `# key=val`
//     comment lines, then the array is emitted as CSV.
//
// Any shape that is NOT a clean uniform flat array (heterogeneous, deeply
// nested, scalar) PASSES THROUGH the raw JSON unchanged — CSV only handles flat
// tabular data. Non-JSON input passes through unchanged.
func EncodeCSV(raw string) string {
	v, ok := parseJSON(raw)
	if !ok {
		return raw
	}

	switch t := v.(type) {
	case []any:
		if cols, rows, uok := tabulate(t); uok {
			return csvBlock(nil, cols, rows)
		}
	case map[string]any:
		_, arr, scalars, ok := singleArrayObject(t)
		if ok {
			if cols, rows, uok := tabulate(arr); uok {
				comments := make([]string, len(scalars))
				for i, kv := range scalars {
					comments[i] = fmt.Sprintf("# %s=%s", kv[0].(string), scalarString(kv[1]))
				}
				return csvBlock(comments, cols, rows)
			}
		}
	}

	// Not flat tabular → passthrough raw JSON (CSV can't represent it).
	return raw
}

// EncodeLoose re-encodes a JSON tool result as "loose JSON": the EXACT JSON
// structure (braces, brackets, commas, colons, per-row keys) with the quote tax
// removed where unambiguous. Object keys that are safe identifiers and string
// values that are safe bare tokens lose their quotes; everything structural
// stays. This sits between strict JSON and TOON — keys stay per-row (unlike
// TOON's header-once tabular form) but the quote overhead is gone. Non-JSON
// input passes through unchanged.
//
// Bare-emit rule (only USEFUL quotes survive; recoverability wins):
//   - A KEY is bare unless it is empty or contains a structural JSON delimiter
//     (" : , { } [ ]) or whitespace. So #, @, foo.bar, 123, us-west-2 → BARE.
//   - A string VALUE is bare unless it is empty, contains one of those
//     delimiters or whitespace, OR is literal-ambiguous with a bare token (it
//     equals true/false/null or parses as a JSON number) — those keep quotes
//     because the quote is USEFUL: it preserves string-vs-number/bool TYPE on
//     read-back. Every other value goes bare.
func EncodeLoose(raw string) string {
	v, ok := parseJSON(raw)
	if !ok {
		return raw
	}
	var b strings.Builder
	writeLoose(&b, v)
	return b.String()
}

func writeLoose(b *strings.Builder, v any) {
	switch t := v.(type) {
	case map[string]any:
		// Sorted keys for determinism.
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			if looseSafeKey(k) {
				b.WriteString(k)
			} else {
				b.WriteString(quoteJSON(k))
			}
			b.WriteByte(':')
			writeLoose(b, t[k])
		}
		b.WriteByte('}')
	case []any:
		b.WriteByte('[')
		for i, el := range t {
			if i > 0 {
				b.WriteByte(',')
			}
			writeLoose(b, el)
		}
		b.WriteByte(']')
	case string:
		if looseSafeValue(t) {
			b.WriteString(t)
		} else {
			b.WriteString(quoteJSON(t))
		}
	default:
		// number / bool / nil — already bare, and json.Marshal renders them
		// exactly (nil → null, bools, numbers).
		out, err := json.Marshal(v)
		if err == nil {
			b.Write(out)
		} else {
			b.WriteString(scalarString(v))
		}
	}
}

// looseSafeKey reports whether an object key can be emitted bare: non-empty and
// free of structural JSON delimiters and whitespace. (No literal-ambiguity check
// — a key is always a string, so a bare 123 or true key is unambiguous.)
func looseSafeKey(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v' {
			return false
		}
		switch r {
		case ',', ':', '{', '}', '[', ']', '"':
			return false
		}
	}
	return true
}

// looseSafeValue reports whether a string value can be emitted bare: non-empty,
// no whitespace, no structural/delimiter char, and not ambiguous with a
// number/bool/null literal (whose quote is USEFUL — it preserves type).
func looseSafeValue(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v' {
			return false
		}
		switch r {
		case ',', ':', '{', '}', '[', ']', '"':
			return false
		}
	}
	// Ambiguous with a real scalar literal on read-back → keep quotes.
	switch s {
	case "true", "false", "null":
		return false
	}
	if looksLikeNumber(s) {
		return false
	}
	return true
}

// looksLikeNumber reports whether s would parse as a JSON number literal.
func looksLikeNumber(s string) bool {
	dec := json.NewDecoder(strings.NewReader(s))
	var n json.Number
	if err := dec.Decode(&n); err != nil {
		return false
	}
	return !dec.More()
}

// quoteJSON renders s as a JSON string literal (with quotes + escaping).
func quoteJSON(s string) string {
	out, err := json.Marshal(s)
	if err != nil {
		return `"` + s + `"`
	}
	return string(out)
}

// EncodeJSONTOON re-encodes a JSON tool result as an EMBEDDED hybrid: the OUTER
// value stays valid JSON, but any field (recursively) whose value is a uniform
// array of flat objects is replaced by an INLINE TOON STRING of the tabular
// form `[N]{cols}: row | row | ...`. The model keeps a familiar, always-
// parseable JSON envelope while the expensive repeated-key arrays — where the
// tokens actually are — get compressed. Scalar / non-uniform / non-array fields
// stay normal JSON. The envelope is re-marshaled compact. Non-JSON input passes
// through unchanged.
func EncodeJSONTOON(raw string) string {
	v, ok := parseJSON(raw)
	if !ok {
		return raw
	}
	out, err := json.Marshal(jsonToon(v))
	if err != nil {
		return raw
	}
	return string(out)
}

// jsonToon walks a decoded JSON value, replacing every uniform-flat-object array
// with its inline TOON string and recursing into everything else.
func jsonToon(v any) any {
	switch t := v.(type) {
	case []any:
		if cols, rows, ok := tabulate(t); ok {
			return toonInline(cols, rows)
		}
		out := make([]any, len(t))
		for i, el := range t {
			out[i] = jsonToon(el)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = jsonToon(val)
		}
		return out
	default:
		return v
	}
}

// --- shared tabular emission ---

// toonMultiline renders `<name>[N]{cols}:` then one comma-joined row per line.
// name may be empty for a top-level array.
func toonMultiline(name string, cols []string, rows [][]string) string {
	var b strings.Builder
	b.WriteString(toonHeader(name, cols, len(rows)))
	for _, r := range rows {
		b.WriteByte('\n')
		b.WriteString(strings.Join(r, ","))
	}
	return b.String()
}

// toonInline renders `[N]{cols}: row | row | ...` on a single line — the form
// embedded as a JSON string value by EncodeJSONTOON.
func toonInline(cols []string, rows [][]string) string {
	joined := make([]string, len(rows))
	for i, r := range rows {
		joined[i] = strings.Join(r, ",")
	}
	return toonHeader("", cols, len(rows)) + " " + strings.Join(joined, " | ")
}

// toonHeader builds the `<name>[N]{cols}:` prefix shared by both TOON forms.
func toonHeader(name string, cols []string, n int) string {
	return name + "[" + strconv.Itoa(n) + "]{" + strings.Join(cols, ",") + "}:"
}

// csvBlock renders optional comment lines, a header row, and one data row per
// element using RFC4180 quoting.
func csvBlock(comments []string, cols []string, rows [][]string) string {
	var b bytes.Buffer
	for _, c := range comments {
		b.WriteString(c)
		b.WriteByte('\n')
	}
	w := csv.NewWriter(&b)
	_ = w.Write(cols)
	for _, r := range rows {
		_ = w.Write(r)
	}
	w.Flush()
	return strings.TrimRight(b.String(), "\n")
}

// tabulate reports whether arr is a non-empty array of flat objects sharing the
// same key set, and returns the sorted column list + each element's cells in
// column order. Shared by TOON, CSV, and json-toon so the row rendering lives
// in one place. Cell rendering: scalars bare, nested array/object as compact
// JSON (e.g. [22,35]).
func tabulate(arr []any) (cols []string, rows [][]string, ok bool) {
	if len(arr) == 0 {
		return nil, nil, false
	}
	keySet := map[string]bool{}
	maps := make([]map[string]any, 0, len(arr))
	for i, el := range arr {
		m, isObj := el.(map[string]any)
		if !isObj {
			return nil, nil, false
		}
		if i == 0 {
			for k := range m {
				keySet[k] = true
				cols = append(cols, k)
			}
			sort.Strings(cols)
		} else {
			if len(m) != len(keySet) {
				return nil, nil, false
			}
			for k := range m {
				if !keySet[k] {
					return nil, nil, false
				}
			}
		}
		maps = append(maps, m)
	}
	rows = make([][]string, len(maps))
	for i, m := range maps {
		cells := make([]string, len(cols))
		for j, c := range cols {
			cells[j] = cellString(m[c])
		}
		rows[i] = cells
	}
	return cols, rows, true
}

// cellString renders a single value: scalars bare, nested array/object as
// compact JSON.
func cellString(v any) string {
	switch v.(type) {
	case []any, map[string]any:
		out, err := json.Marshal(v)
		if err == nil {
			return string(out)
		}
	}
	return scalarString(v)
}

// --- shared helpers ---

// parseJSON reports whether raw is valid JSON and returns the decoded value.
func parseJSON(raw string) (any, bool) {
	dec := json.NewDecoder(strings.NewReader(raw))
	// Keep numbers as their source literal — without this, a large integer id
	// (28457823) decodes to float64 and scalarString formats it back in
	// scientific notation (2.8457823e+07), corrupting the token the LLM reads.
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, false
	}
	// Reject trailing junk (a bare prose line that happens to start with a
	// number/word would otherwise decode a prefix).
	if dec.More() {
		return nil, false
	}
	return v, true
}

// singleArrayObject reports whether m is an object with exactly one array-valued
// field and any number of scalar fields (no other objects/arrays). It returns
// the array field's key + value and the scalar fields sorted by key.
func singleArrayObject(m map[string]any) (arrKey string, arr []any, scalars [][2]any, ok bool) {
	arrCount := 0
	var scalarKeys []string
	scalarVals := map[string]any{}
	for k, v := range m {
		switch vv := v.(type) {
		case []any:
			arrCount++
			arrKey = k
			arr = vv
		case map[string]any:
			return "", nil, nil, false // nested object sibling → not this shape
		default:
			scalarKeys = append(scalarKeys, k)
			scalarVals[k] = v
		}
	}
	if arrCount != 1 {
		return "", nil, nil, false
	}
	sort.Strings(scalarKeys)
	for _, k := range scalarKeys {
		scalars = append(scalars, [2]any{k, scalarVals[k]})
	}
	return arrKey, arr, scalars, true
}

// scalarString renders a scalar JSON value (string, number, bool, nil) as a
// bare string.
func scalarString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case json.Number:
		return t.String()
	case bool:
		return strconv.FormatBool(t)
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	default:
		out, err := json.Marshal(v)
		if err == nil {
			return string(out)
		}
		return fmt.Sprintf("%v", v)
	}
}
