package toolfmt

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
)

// EncodeTight is the aggressive synthesis of every token-saving principle,
// co-optimizing tokens AND comprehension. It is NOT one format but a RECURSIVE
// PER-NODE STRATEGY DISPATCHER: renderNode classifies each node by its content
// SHAPE and routes to the best-fit strategy, and each strategy recursively calls
// renderNode for its children. The optimal representation of a tree is the
// best-fit strategy chosen independently per subtree.
//
// Strategy ladder (first that applies to a node wins):
//
//  1. scalar (string/number/bool/null) → bare token; a string is quoted ONLY if
//     unsafe (contains the active delimiter/whitespace, empty, or is
//     literal-ambiguous with true/false/null/a number — that quote is USEFUL, it
//     preserves type). numbers/bools/null are bare.
//  2. long/multiline string (prose — file contents, logs) → passthrough VERBATIM
//     (never tabularize prose).
//  3. array of scalars → inline [a,b,c].
//  4. array of objects, uniform OR semi-uniform → SPARSE UNION TABLE: the union
//     of keys as a one-time header, comma-separated rows, ,, empty cells for
//     missing keys (uniform is the degenerate case: union = the common keys).
//  5. array that does NOT factor into a clean table (heterogeneous keys, nested
//     cells, mixed) → RAW compact JSON. There is no unambiguous flat layout for
//     such an array — a hand-rolled indent/blank scheme collides with the
//     parent's sibling separators, so a reader can't tell an element from a
//     sibling field. Tight refuses to guess: JSON is self-delimiting and fully
//     recoverable. Terseness yields to recoverability here by design.
//  6. object, all values inline-able (scalar / scalar-array) → inline
//     `key val, key val`.
//  7. object with nested children → each key on its own line; an array child
//     stays FLAT under the key label (a table, or a self-delimiting JSON array),
//     a nested OBJECT is indented ONE TAB per level (\t, never spaces).
//  8. empty array/object → explicit terse marker [] / {}.
//
// Separators (documented, non-colliding):
//   - table column / row-cell separator: COMMA (so an empty cell ,, is
//     unambiguous). A value containing a comma is quoted.
//   - inline-object pair separator: COMMA-SPACE `, ` (distinct from the bare
//     table comma).
//   - a scalar array keeps BRACKETS with comma-joined values: [10,40].
//   - genuine object nesting: ONE TAB per depth level (strategy 7). Path-encoded
//     hierarchy (dotted keys like "Server.Start") stays FLAT — no indentation.
//
// Invariant: recoverability first — every datum, and which key/position it
// belongs to, is recoverable from delimiters + the union header + indentation
// alone (never a count). Where a strategy would be ambiguous it drops to the
// next, more explicit one. Non-JSON input passes through unchanged.
func EncodeTight(raw string) string {
	v, ok := parseOrdered(raw)
	if !ok {
		return raw
	}
	return renderNode(v, 0)
}

// renderNode dispatches a decoded JSON node to the first applicable strategy.
func renderNode(v any, depth int) string {
	switch t := v.(type) {
	case string:
		if strings.ContainsAny(t, "\n\r") { // strategy 2: prose verbatim
			return t
		}
		return tightValTok(t) // strategy 1
	case nil, bool, json.Number, float64:
		return renderScalar(t) // strategy 1
	case []any:
		return renderArray(t, depth)
	case *omap:
		return renderObject(t, depth)
	default:
		return compactJSON(v)
	}
}

// renderArray routes an array: empty (8) → scalars (3) → object union table (4)
// → per-element (5).
func renderArray(arr []any, depth int) string {
	if len(arr) == 0 {
		return "[]" // strategy 8
	}
	if allScalars(arr) {
		return renderScalarArray(arr) // strategy 3
	}
	if allObjects(arr) {
		if cols, ok := unionTable(arr); ok {
			return renderTable(cols, arr) // strategy 4
		}
	}
	// strategy 5: an array that does NOT factor into a clean table (heterogeneous
	// keys, nested cells, mixed scalars+objects) has no unambiguous flat layout —
	// a hand-rolled indent/blank-line scheme collides with the parent's own
	// sibling separators, so a reader can't tell an element from a sibling field.
	// Recoverability wins over terseness: emit RAW compact JSON, which is
	// self-delimiting (brackets bound the array, braces bound each object) and
	// unambiguously recoverable. depth is irrelevant to a single self-closed token.
	return compactJSON(arr)
}

// renderObject routes an object: empty (8) → all-inline (6) → nested (7).
func renderObject(m *omap, depth int) string {
	if len(m.keys) == 0 {
		return "{}" // strategy 8
	}
	allInline := true
	for _, k := range m.keys {
		if !inlineable(m.vals[k]) {
			allInline = false
			break
		}
	}
	if allInline {
		return renderInlineObject(m) // strategy 6
	}
	return renderNestedObject(m, depth) // strategy 7
}

// renderInlineObject renders `key val, key val` on one line (strategy 6).
func renderInlineObject(m *omap) string {
	parts := make([]string, len(m.keys))
	for i, k := range m.keys {
		parts[i] = tightKeyTok(k) + " " + renderCell(m.vals[k])
	}
	return strings.Join(parts, ", ")
}

// renderNestedObject renders one key per line; array children stay flat, object
// children indent one tab; sibling blocks are blank-line separated when either
// is multiline so a table can't bleed into the next key (strategy 7).
func renderNestedObject(m *omap, depth int) string {
	blocks := make([]string, 0, len(m.keys))
	for _, k := range m.keys {
		val := m.vals[k]
		if inlineable(val) {
			// scalar / scalar-array → `key val` on this line.
			blocks = append(blocks, tightKeyTok(k)+" "+renderCell(val))
			continue
		}
		child := renderNode(val, depth+1)
		if _, isArr := val.([]any); isArr {
			// array child (table / per-element) → FLAT under the key label.
			blocks = append(blocks, tightKeyTok(k)+"\n"+child)
		} else {
			// nested OBJECT → key label + child indented ONE TAB (genuine nesting).
			blocks = append(blocks, tightKeyTok(k)+"\n"+indentTab(child))
		}
	}
	return joinBlocks(blocks)
}

// renderTable renders the union header + one comma-separated row per element,
// empty cells for missing keys (strategy 4).
func renderTable(cols []string, arr []any) string {
	hdr := make([]string, len(cols))
	for i, c := range cols {
		hdr[i] = tightKeyTok(c)
	}
	lines := make([]string, 0, len(arr)+1)
	lines = append(lines, strings.Join(hdr, ","))
	for _, el := range arr {
		m := el.(*omap)
		cells := make([]string, len(cols))
		for i, c := range cols {
			if v, ok := m.vals[c]; ok {
				cells[i] = renderCell(v)
			} else {
				cells[i] = "" // missing key → empty cell (,,)
			}
		}
		lines = append(lines, strings.Join(cells, ","))
	}
	return strings.Join(lines, "\n")
}

// renderScalarArray renders [a,b,c] with comma-joined single-line cells.
func renderScalarArray(arr []any) string {
	parts := make([]string, len(arr))
	for i, e := range arr {
		parts[i] = renderCell(e)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// renderCell renders a value at a SINGLE-LINE position (table cell, inline-object
// value, scalar-array element): scalars bare/quoted, scalar arrays bracketed,
// anything richer as quoted compact JSON (recoverable fallback).
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
		// (escaping a JSON value is strictly bigger than the original — forbidden).
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

// unionTable computes the union of keys across a non-empty array of objects
// (first-seen order). It reports ok=false — deferring to per-element rendering —
// when any cell holds a nested (non-inline) value, or the union table would be
// more than 60% empty cells.
func unionTable(arr []any) (cols []string, ok bool) {
	seen := map[string]bool{}
	present := 0
	for _, el := range arr {
		m, isObj := el.(*omap)
		if !isObj {
			return nil, false
		}
		for _, k := range m.keys {
			if !inlineable(m.vals[k]) {
				return nil, false // nested cell → per-element
			}
			if !seen[k] {
				seen[k] = true
				cols = append(cols, k)
			}
			present++
		}
	}
	if len(cols) == 0 {
		return nil, false
	}
	fill := float64(present) / float64(len(arr)*len(cols))
	if fill < 0.40 { // >60% empty → too heterogeneous for a table
		return nil, false
	}
	return cols, true
}

// --- token safety ---

// tightValSafe reports whether a string VALUE emits bare: non-empty, no
// whitespace or active delimiter (comma / pipe / quote), and not
// literal-ambiguous with true/false/null/a number.
func tightValSafe(s string) bool {
	if s == "" {
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

// tightKeySafe reports whether a KEY emits bare: non-empty, no whitespace or
// active delimiter. (No literal-ambiguity check — a key is always a string, so a
// bare 123/true key is unambiguous.)
func tightKeySafe(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		switch r {
		case ' ', '\t', '\n', '\r', '\f', '\v', ',', '|', '"':
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

func allObjects(arr []any) bool {
	for _, e := range arr {
		if _, ok := e.(*omap); !ok {
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

// indentTab prefixes every line of s with one tab.
func indentTab(s string) string {
	return "\t" + strings.ReplaceAll(s, "\n", "\n\t")
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

// --- order-preserving JSON parse (tight only) ---

// omap is a JSON object that remembers key insertion order (json.Unmarshal into
// map[string]any loses it, and tight's column/field order is load-bearing for
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
