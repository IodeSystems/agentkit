package toolfmt

import (
	"encoding/json"
	"strconv"
	"strings"
)

// EncodeTightC is TIGHT tuned for COMPREHENSION on small / quantized models,
// after the bonsai accuracy benchmark showed plain tight's token wins came at a
// real comprehension cost (aggregation/filtering/nested all degraded vs JSON and
// TOON). It keeps tight's cheap wins (bare tokens, unquoted-unless-needed, inline
// key:val objects, uniform tables) but spends a few tokens back where the data
// showed the model losing structure:
//
//  1. ROW-COUNT ANCHOR on every table — `[N]col1,col2` header. bonsai answered
//     "how many rows" questions with an EMPTY string on tight (no count to read)
//     but correctly on TOON (which declares [N]). A count is not math the model
//     computes; it is a structural anchor it reads. ~1 token per table.
//  2. UNIFORM-ONLY tables. tight's SPARSE union table (`,,` empty cells for
//     missing keys) forced the model to count commas to realign columns — the
//     exact place filtering/aggregation misfired. Here a table is used ONLY when
//     every row has every column; any irregularity drops to loose (explicit
//     braces + keys on every value), which measured BETTER than tight on the
//     nested dataset.
//  3. NESTED OBJECTS render as loose JSON (explicit {}), not tab-indentation.
//     Indentation is positional structure the model has to track; braces are
//     explicit. loose beat tight by ~10 points on nested data.
//
// Recoverability is unchanged (delimiters + header + [N] + loose braces). Non-JSON
// passes through. It is deliberately LESS token-minimal than tight — the bet is
// that on small models the comprehension gain outweighs the extra tokens.
func EncodeTightC(raw string) string {
	v, ok := parseOrdered(raw)
	if !ok {
		return raw
	}
	return renderNodeC(v, 0)
}

// renderNodeC is the comprehension dispatcher (mirrors renderNode; diverges on
// arrays and nested objects).
func renderNodeC(v any, depth int) string {
	switch t := v.(type) {
	case string:
		if strings.ContainsAny(t, "\n\r") {
			return t
		}
		return tightValTok(t)
	case nil, bool, json.Number, float64:
		return renderScalar(t)
	case []any:
		return renderArrayC(t, depth)
	case *omap:
		return renderObjectC(t, depth)
	default:
		return compactJSON(v)
	}
}

// renderArrayC: empty → [] ; scalars → [a,b,c] ; a UNIFORM object array (every
// row has every column) → count-anchored table ; anything else → loose.
func renderArrayC(arr []any, depth int) string {
	if len(arr) == 0 {
		return "[]"
	}
	if allScalars(arr) {
		return renderScalarArray(arr)
	}
	if cols, ok := uniformColumns(arr); ok {
		return renderTableC(cols, arr)
	}
	return looseOrderedC(arr) // irregular / nested → explicit braces, order kept
}

// renderObjectC: empty → {} ; all-inline → key:val,key:val ; otherwise per-key
// dispatch — a scalar stays inline, an ARRAY child renders through renderArrayC
// (so a uniform table inside an object is still a count-anchored table, NOT
// flattened to loose), and a genuinely-nested OBJECT child renders as explicit
// loose braces (no tab-indentation).
func renderObjectC(m *omap, depth int) string {
	if len(m.keys) == 0 {
		return "{}"
	}
	allInline := true
	for _, k := range m.keys {
		if !inlineable(m.vals[k]) {
			allInline = false
			break
		}
	}
	if allInline {
		return renderInlineObject(m)
	}
	blocks := make([]string, 0, len(m.keys))
	for _, k := range m.keys {
		val := m.vals[k]
		key := tightKeyTok(k)
		switch {
		case inlineable(val):
			blocks = append(blocks, key+":"+renderCell(val))
		case isArrayVal(val):
			child := renderArrayC(val.([]any), depth+1)
			if strings.Contains(child, "\n") {
				blocks = append(blocks, key+"\n"+child) // table → label + block
			} else {
				blocks = append(blocks, key+":"+child) // single-line loose array
			}
		default:
			blocks = append(blocks, key+":"+looseOrderedC(val)) // nested object → loose
		}
	}
	return joinBlocks(blocks)
}

func isArrayVal(v any) bool {
	_, ok := v.([]any)
	return ok
}

// looseOrderedC renders a value as compact loose JSON — explicit {}/[] structure,
// keys/values unquoted where safe — while PRESERVING object key order (the plain
// `loose`/EncodeLoose path re-sorts keys via a map; field order is load-bearing
// for comprehension, so tightc keeps it).
func looseOrderedC(v any) string {
	switch t := v.(type) {
	case *omap:
		parts := make([]string, len(t.keys))
		for i, k := range t.keys {
			parts[i] = tightKeyTok(k) + ":" + looseOrderedC(t.vals[k])
		}
		return "{" + strings.Join(parts, ",") + "}"
	case []any:
		parts := make([]string, len(t))
		for i, e := range t {
			parts[i] = looseOrderedC(e)
		}
		return "[" + strings.Join(parts, ",") + "]"
	case string:
		if strings.ContainsAny(t, "\n\r") || !tightValSafe(t) {
			return quoteJSON(t)
		}
		return t
	default:
		return renderScalar(t)
	}
}

// renderTableC renders a `[N]header` count anchor line, then one comma-separated
// row per element. N is the row count. Every row has every column (uniform), so
// there are no empty cells to realign.
func renderTableC(cols []string, arr []any) string {
	hdr := make([]string, len(cols))
	for i, c := range cols {
		hdr[i] = tightKeyTok(c)
	}
	lines := make([]string, 0, len(arr)+1)
	lines = append(lines, "["+strconv.Itoa(len(arr))+"]"+strings.Join(hdr, ","))
	for _, el := range arr {
		m := el.(*omap)
		cells := make([]string, len(cols))
		for i, c := range cols {
			cells[i] = renderCell(m.vals[c])
		}
		lines = append(lines, strings.Join(cells, ","))
	}
	return strings.Join(lines, "\n")
}

// uniformColumns returns the column order iff arr is a non-empty array of objects
// that ALL share the exact same key set (so a dense table has no missing cells).
// Column order follows the first row. Reports ok=false on any structural
// irregularity (non-object, nested cell, differing key set) so the caller drops
// to loose.
func uniformColumns(arr []any) (cols []string, ok bool) {
	first, isObj := arr[0].(*omap)
	if !isObj || len(first.keys) == 0 {
		return nil, false
	}
	for _, k := range first.keys {
		if !inlineable(first.vals[k]) {
			return nil, false
		}
	}
	cols = first.keys
	for _, el := range arr[1:] {
		m, isObj := el.(*omap)
		if !isObj || len(m.keys) != len(cols) {
			return nil, false
		}
		for _, c := range cols {
			v, present := m.vals[c]
			if !present || !inlineable(v) {
				return nil, false
			}
		}
	}
	return cols, true
}
