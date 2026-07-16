// Package toolfmt provides a token-lean, information-preserving re-encoder for
// tool-call RESULTS before they enter an LLM's context. JSON is verbose (repeated
// keys, punctuation); EncodeTightC restates a uniform array of flat objects as a
// count-anchored table and drops needless quotes, cutting tokens for the same
// data while staying reversible.
//
// Contract:
//   - Input that is NOT valid JSON (prose — file contents, `go test` output,
//     plain text) is returned UNCHANGED. Only structured JSON is transformed.
//   - The transform is information-preserving and reversible: an LLM (or a
//     decoder) reading the encoded form recovers every datum and its key/position.
//
// JSON itself is the identity encoding — there is no EncodeJSON; the caller
// simply leaves Session.EncodeToolResult nil. tightc (EncodeTightC) is the one
// non-identity format, tuned for comprehension on small/quantized models.
package toolfmt

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// looksLikeNumber reports whether s parses as a single JSON number (so a bare
// numeric-looking string value must be quoted to preserve its string type).
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

// scalarString renders a scalar value as its bare string form (no quotes).
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
