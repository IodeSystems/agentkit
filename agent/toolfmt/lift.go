package toolfmt

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// liftRef is a hoisted-subtree reference token (e.g. "$0") that renders bare.
// It flows through the tight renderer as a scalar-for-layout value (see isScalar
// and the liftRef cases in renderNode/renderCell).
type liftRef string

// MarshalJSON lets a liftRef survive a compactJSON/loose fallback: it serializes
// as the quoted token, and the loose pass then strips the quote (the token has no
// structural char), so a reference inside a JSON-fallback sub-part stays bare and
// recoverable rather than breaking the blob.
func (r liftRef) MarshalJSON() ([]byte, error) { return json.Marshal(string(r)) }

// liftMinSize is the smallest canonical (compact-JSON) length worth hoisting.
// Hoisting anything tinier can't beat the reference overhead; the top-level byte
// PROBE is the real guard, this just avoids pointless work.
const liftMinSize = 8

// liftMaxInput is a coarse backstop on the lift probe. Candidate collection is
// now dedup-only (see liftRender): a single O(n) counting pass keeps only the
// subtree shapes that repeat, so all-distinct bulk data (100 unique rows) yields
// zero candidates and returns instantly — the O(n²) blow-up is gone. The residual
// cost is the incremental accept loop, which re-renders once per ACCEPTED
// (repeated) shape — normally a handful. This guard only trips on a pathological
// input that is both huge AND has many distinct repeated shapes; above this
// plain-tight size EncodeLift skips lifting and returns tight. Lift is monotone,
// so the guard is lossless — it bounds worst-case time, never correctness.
const liftMaxInput = 4 << 20 // 4 MiB

// EncodeLift is tight WITH a lifting add-in: a repeated subtree (an object or
// array that appears identically two or more times) is emitted ONCE as a named
// definition ($0, $1, …) and referenced at each use site, so the bytes are paid
// once. This trades a little locality for a "jump" — which an LLM resolves via
// attention (content-addressable lookup), far cheaper than the counting tight
// already avoids. Lifting is a pure win only when a subtree actually repeats or
// is large, so EncodeLift is byte-PROBED against plain tight and returns the
// smaller: it is NEVER worse than tight, and reduces to exactly tight when
// nothing is worth hoisting. Non-JSON input passes through unchanged.
//
// Output shape: the tight body (with $n references) then a blank line then the
// definitions, each `$n:<inline value>` or `$n` + block, blank-line separated.
// Recoverability: every $n reference resolves to exactly one definition anchored
// at a `$n` line; the reference is a jump, never a computed offset.
func EncodeLift(raw string) string {
	v, ok := parseOrdered(raw)
	if !ok {
		return raw
	}
	plain := renderNode(v, 0)
	if len(plain) > liftMaxInput {
		return plain // too large to probe affordably; lift would reduce to tight anyway
	}
	if lifted, ok := liftRender(v); ok && len(lifted) < len(plain) {
		return lifted
	}
	return plain
}

// liftRender hoists every repeated subtree and returns the lifted rendering, or
// ok=false when nothing repeats (so EncodeLift keeps plain tight).
//
// Candidates are subtrees whose canonical appears TWO OR MORE times — dedup is
// the only unbounded win, and it is also what makes this affordable: a single
// occurrence-counting pass collapses the candidate set from "every subnode"
// (which made the probe O(n²) on bulk uniform data — 100 distinct rows = 100
// full-tree re-renders) down to the handful of shapes that actually repeat.
// Uniform data with all-distinct rows yields ZERO candidates and returns
// instantly. (The earlier single-occurrence "un-nest a loose-trapped table"
// case is dropped: it required probing every unique subnode — the source of the
// blow-up — and saved ~0 tokens on real corpora.)
func liftRender(v any) (string, bool) {
	counts := map[string]int{}
	var seenOrder []string
	countCandidates(v, counts, &seenOrder)
	var order []string // repeated candidate canonicals, first-appearance order
	for _, c := range seenOrder {
		if counts[c] >= 2 {
			order = append(order, c)
		}
	}
	if len(order) == 0 {
		return "", false
	}

	// Try repeated candidates largest-first (a container before its contents),
	// keeping a lift only if it strictly reduces the total. This MEASURES the
	// dedup saving instead of estimating it: a subtree paid once (its `$n`
	// definition) plus a short ref at each of its N sites, accepted exactly when
	// that nets fewer bytes than inlining it N times. Largest-first lets an outer
	// repeated container subsume inner ones before they're considered separately.
	// Monotone: the result is never larger than plain tight.
	cands := append([]string(nil), order...)
	sort.SliceStable(cands, func(i, j int) bool { return len(cands[i]) > len(cands[j]) })

	accepted := map[string]bool{}
	best := renderNode(v, 0) // plain tight baseline
	improved := false
	for _, c := range cands {
		accepted[c] = true
		if out, ok := renderLifted(v, accepted, order); ok && len(out) < len(best) {
			best = out
			improved = true
		} else {
			delete(accepted, c)
		}
	}
	if !improved {
		return "", false
	}
	return best, true
}

// renderLifted renders v with every canonical in accepted hoisted to a `$n`
// definition (numbered in first-appearance order), returning ok=false if nothing
// is accepted.
func renderLifted(v any, accepted map[string]bool, order []string) (string, bool) {
	refOf := map[string]string{}
	var defsOrder []string
	for _, c := range order {
		if accepted[c] {
			refOf[c] = "$" + strconv.Itoa(len(defsOrder))
			defsOrder = append(defsOrder, c)
		}
	}
	if len(defsOrder) == 0 {
		return "", false
	}

	body := renderNode(substAll(v, refOf), 0)
	defs := make([]string, len(defsOrder))
	for i, c := range defsOrder {
		node, _ := parseOrdered(c) // re-decode the canonical subtree
		content := renderNode(substChildren(node, refOf), 0)
		if strings.Contains(content, "\n") {
			defs[i] = refOf[c] + "\n" + content // multiline (table): label + block
		} else {
			defs[i] = refOf[c] + ":" + content // single-line: $n:value
		}
	}
	return body + "\n\n" + strings.Join(defs, "\n\n"), true
}

// countCandidates tallies how many times each object/array subnode canonical
// (large enough to be worth hoisting) appears, recording first-appearance order.
// One O(n) traversal; callers keep only canonicals with count>=2.
func countCandidates(v any, counts map[string]int, order *[]string) {
	switch t := v.(type) {
	case *omap:
		if c := compactJSON(v); len(c) >= liftMinSize {
			if counts[c] == 0 {
				*order = append(*order, c)
			}
			counts[c]++
		}
		for _, k := range t.keys {
			countCandidates(t.vals[k], counts, order)
		}
	case []any:
		if c := compactJSON(v); len(c) >= liftMinSize {
			if counts[c] == 0 {
				*order = append(*order, c)
			}
			counts[c]++
		}
		for _, e := range t {
			countCandidates(e, counts, order)
		}
	}
}

// substAll replaces a node with its reference if the WHOLE node is hoisted,
// otherwise rebuilds it with substituted children.
func substAll(node any, refOf map[string]string) any {
	switch node.(type) {
	case *omap, []any:
		if ref, ok := refOf[compactJSON(node)]; ok {
			return liftRef(ref)
		}
		return substChildren(node, refOf)
	default:
		return node
	}
}

// substChildren rebuilds a node's children through substAll but leaves the node
// itself in place (used when rendering a definition's own body).
func substChildren(node any, refOf map[string]string) any {
	switch t := node.(type) {
	case *omap:
		out := &omap{keys: append([]string(nil), t.keys...), vals: make(map[string]any, len(t.keys))}
		for _, k := range t.keys {
			out.vals[k] = substAll(t.vals[k], refOf)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = substAll(e, refOf)
		}
		return out
	default:
		return node
	}
}
