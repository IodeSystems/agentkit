package toolfmt

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
)

// This file holds the shared leaf machinery that EncodeTightC builds on: the
// order-preserving JSON parse (omap/parseOrdered), scalar/cell rendering, the
// bare-vs-quoted token rules, and the small shape predicates. The tightc
// dispatch itself lives in tightc.go.

// --- cell / scalar rendering ---

// renderCell renders a value at a SINGLE-LINE position (table cell, inline-object
// value, scalar-array element): scalars bare/quoted, scalar arrays bracketed,
// anything richer as compact JSON (recoverable fallback).
func renderCell(v any) string {
	switch t := v.(type) {
	case string:
		if strings.ContainsAny(t, "\n\r") || !tightValSafe(t) {
			return quoteJSON(t)
		}
		return t
	case nil, bool, json.Number, float64:
		return renderScalar(t)
	case []any:
		if isScalarArray(t) {
			return renderScalarArray(t)
		}
		// Last-resort single-line: RAW compact JSON, never quoted/escaped
		// (escaping a JSON value is strictly bigger than the original).
		return compactJSON(v)
	default:
		return compactJSON(v)
	}
}

// renderScalar renders a NON-string scalar bare: null/bools/numbers.
func renderScalar(v any) string {
	switch t := v.(type) {
	case nil:
		return "null"
	case bool:
		return strconv.FormatBool(t)
	case json.Number:
		return t.String()
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	default:
		return scalarString(v)
	}
}

// renderInlineObject renders `key:val,key:val` on one line.
func renderInlineObject(m *omap) string {
	parts := make([]string, len(m.keys))
	for i, k := range m.keys {
		parts[i] = tightKeyTok(k) + ":" + renderCell(m.vals[k])
	}
	return strings.Join(parts, ",")
}

// renderScalarArray renders [a,b,c] with comma-joined single-line cells.
func renderScalarArray(arr []any) string {
	parts := make([]string, len(arr))
	for i, e := range arr {
		parts[i] = renderCell(e)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// --- token safety ---

// tightValSafe reports whether a string VALUE emits bare: non-empty, not opening
// with a structural bracket/brace, no whitespace or active delimiter (comma /
// pipe / quote), and not literal-ambiguous with true/false/null/a number.
func tightValSafe(s string) bool {
	if s == "" {
		return false
	}
	// A value whose first byte is a structural opener would be misread as a
	// nested array/object by a decoder (e.g. the string "[a-z]+" or "{x"); quote
	// it so the leading bracket/brace is unambiguously literal.
	if s[0] == '[' || s[0] == '{' {
		return false
	}
	for _, r := range s {
		switch r {
		case ' ', '\t', '\n', '\r', '\f', '\v', ',', '|', '"':
			return false
		}
	}
	switch s {
	case "true", "false", "null":
		return false
	}
	return !looksLikeNumber(s)
}

func tightValTok(s string) string {
	if tightValSafe(s) {
		return s
	}
	return quoteJSON(s)
}

// tightKeySafe reports whether a KEY emits bare: non-empty, not opening with a
// structural bracket/brace, no whitespace or active delimiter (comma / colon /
// pipe / quote — colon delimits key:value, so a key can't contain one).
func tightKeySafe(s string) bool {
	if s == "" {
		return false
	}
	// A bare key beginning with '[' (e.g. "[2]x") collides with the tightc
	// `[N]` table-count marker on a label line; '{' would open a nested object.
	if s[0] == '[' || s[0] == '{' {
		return false
	}
	for _, r := range s {
		switch r {
		case ' ', '\t', '\n', '\r', '\f', '\v', ',', ':', '|', '"':
			return false
		}
	}
	return true
}

func tightKeyTok(s string) string {
	if tightKeySafe(s) {
		return s
	}
	return quoteJSON(s)
}

// --- shape helpers ---

func isScalar(v any) bool {
	switch v.(type) {
	case nil, string, bool, float64, json.Number:
		return true
	default:
		return false
	}
}

func isScalarArray(arr []any) bool {
	for _, e := range arr {
		if !isScalar(e) {
			return false
		}
	}
	return true
}

// inlineable reports whether a value renders as a single-line cell: a scalar or
// an array of scalars.
func inlineable(v any) bool {
	if isScalar(v) {
		return true
	}
	arr, ok := v.([]any)
	return ok && isScalarArray(arr)
}

func allScalars(arr []any) bool {
	for _, e := range arr {
		if !isScalar(e) {
			return false
		}
	}
	return true
}

// joinBlocks joins rendered sibling blocks, inserting a blank line only AFTER a
// multiline block (a table/nested block whose end must be delimited so its last
// row can't be mistaken for the next key). A single-line block needs no trailer.
func joinBlocks(blocks []string) string {
	var out []string
	for i, b := range blocks {
		if i > 0 && strings.Contains(blocks[i-1], "\n") {
			out = append(out, "")
		}
		out = append(out, b)
	}
	return strings.Join(out, "\n")
}

// compactJSON renders any value (including *omap, which preserves key order) as
// compact JSON — the recoverable fallback for sub-parts the scheme can't flatten.
func compactJSON(v any) string {
	out, err := json.Marshal(v)
	if err != nil {
		return scalarString(v)
	}
	return string(out)
}

// --- order-preserving JSON parse ---

// omap is a JSON object that remembers key insertion order (json.Unmarshal into
// map[string]any loses it, and tightc's column/field order is load-bearing for
// comprehension).
type omap struct {
	keys []string
	vals map[string]any
}

// MarshalJSON emits the object in insertion order so compactJSON fallbacks keep
// the original field order.
func (m *omap) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, k := range m.keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		b.Write(kb)
		b.WriteByte(':')
		vb, err := json.Marshal(m.vals[k])
		if err != nil {
			return nil, err
		}
		b.Write(vb)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// parseOrdered decodes raw JSON preserving object key order (objects → *omap,
// arrays → []any, numbers → json.Number). Reports ok=false for non-JSON.
func parseOrdered(raw string) (any, bool) {
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	v, err := readOrdered(dec)
	if err != nil {
		return nil, false
	}
	if dec.More() {
		return nil, false
	}
	return v, true
}

func readOrdered(dec *json.Decoder) (any, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := t.(json.Delim); ok {
		switch d {
		case '{':
			m := &omap{vals: map[string]any{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key := kt.(string)
				val, err := readOrdered(dec)
				if err != nil {
					return nil, err
				}
				if _, seen := m.vals[key]; !seen {
					m.keys = append(m.keys, key)
				}
				m.vals[key] = val
			}
			if _, err := dec.Token(); err != nil { // consume '}'
				return nil, err
			}
			return m, nil
		case '[':
			arr := []any{}
			for dec.More() {
				val, err := readOrdered(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, val)
			}
			if _, err := dec.Token(); err != nil { // consume ']'
				return nil, err
			}
			return arr, nil
		}
	}
	return t, nil // string, json.Number, bool, nil
}
