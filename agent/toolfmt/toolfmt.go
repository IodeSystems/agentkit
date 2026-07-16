// Package toolfmt provides pluggable, information-preserving re-encoders for
// tool-call RESULTS before they enter an LLM's context. The goal is fewer
// tokens for the same data: JSON is verbose (repeated keys, punctuation), so a
// uniform array of flat objects is far cheaper as YAML/TOON/CSV.
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

	"gopkg.in/yaml.v3"
)

// EncodeYAML re-encodes a JSON tool result as YAML. YAML works for ANY JSON
// value and is LLM-native, so this never falls back except on non-JSON input,
// which passes through unchanged.
func EncodeYAML(raw string) string {
	v, ok := parseJSON(raw)
	if !ok {
		return raw
	}
	out, err := yaml.Marshal(v)
	if err != nil {
		return raw
	}
	return strings.TrimRight(string(out), "\n")
}

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
// Anything else (non-uniform, deeply nested, scalar) falls back to YAML, which
// is still terse and fully information-preserving. Non-JSON input passes
// through unchanged.
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

	// Fallback: terse YAML, still fully information-preserving.
	return EncodeYAML(raw)
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
