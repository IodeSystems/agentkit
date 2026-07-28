package agent

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
)

// repairLooseArgs turns near-JSON tool arguments into JSON, or reports that it
// could not. It returns the repaired text, the list of repairs applied (for the
// log), and whether anything usable came out.
//
// WHY REPAIR AT ALL: rejecting a call costs a full regeneration of its
// arguments, and generated tokens are the most expensive channel there is —
// roughly an order of magnitude over cached prompt. A trailing comma is not
// worth re-writing a 4KB file for.
//
// WHAT IT WILL NEVER DO: close an unterminated string, brace or bracket. That
// is the whole discipline here. Off-the-shelf JSON-repair routines auto-close,
// which on a TRUNCATED payload fabricates a call the model never finished
// writing and hands it to a dispatcher as though it were complete — a half-
// written file, a delete with half a filter. Truncation is diagnosed BEFORE
// repair is attempted (see checkArgs) and is never sent here. Repair is for
// syntax the model got wrong, never for output that stopped early.
//
// Every repair is also validated afterwards: anything that does not come out as
// a strict JSON object is discarded and the call is refused as if no repair had
// been attempted. A botched repair therefore degrades to the existing refusal,
// never to a wrong dispatch.
func repairLooseArgs(s string) (string, []string, repairOutcome) {
	var repairs []string
	out := s
	if stripped, ok := stripCodeFence(out); ok {
		out = stripped
		repairs = append(repairs, "removed a markdown code fence")
	}
	rewritten, rs, unterminated := rewriteLoose(out)
	repairs = append(repairs, rs...)
	rewritten = strings.TrimSpace(rewritten)

	// Looseness can HIDE truncation. Go's decoder rejects `{path:'a.java',
	// content:'class Big {` at the unquoted key — a syntax error — and never
	// reaches the unterminated string at the end, so the caller's first-pass
	// classification says "malformed" when the real fault is "cut off". Getting
	// this wrong is not merely cosmetic: it tells the model to retry CORRECTLY
	// when it needs to retry SMALLER, and it will produce the same oversized
	// call again. Once quoting is normalized the truncation is visible, so it is
	// re-classified here rather than at the caller.
	if unterminated {
		return "", nil, repairTruncated
	}
	if rewritten != "" && rewritten[0] == '{' && json.Valid([]byte(rewritten)) {
		if len(repairs) == 0 {
			return "", nil, repairFailed // nothing was wrong with it here
		}
		return rewritten, repairs, repairOK
	}
	if isTruncationErr(decodeErr(rewritten)) {
		return "", nil, repairTruncated
	}
	return "", nil, repairFailed
}

// repairOutcome is what a repair attempt concluded. repairTruncated is separate
// from repairFailed because the two produce different advice to the model.
type repairOutcome int

const (
	repairFailed repairOutcome = iota
	repairOK
	repairTruncated
)

// decodeErr reports why s is not JSON, or nil if it is.
func decodeErr(s string) error {
	var v json.RawMessage
	return json.NewDecoder(strings.NewReader(s)).Decode(&v)
}

// isTruncationErr distinguishes "the input stopped early" from "the input is
// wrong". A *json.SyntaxError means the decoder read a byte it could not accept;
// an EOF means it ran out while still expecting more.
func isTruncationErr(err error) bool {
	if err == nil {
		return false
	}
	var se *json.SyntaxError
	if errors.As(err, &se) {
		return false
	}
	return errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)
}

// stripCodeFence removes a ```…``` wrapper, which models emit when they treat
// the arguments field as a place to write a code block.
func stripCodeFence(s string) (string, bool) {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") {
		return s, false
	}
	// Drop the opening fence and any language tag on that line.
	if nl := strings.IndexByte(t, '\n'); nl >= 0 {
		t = t[nl+1:]
	} else {
		return s, false
	}
	if end := strings.LastIndex(t, "```"); end >= 0 {
		t = t[:end]
	}
	return t, true
}

// rewriteLoose applies the per-token repairs. It is a scanner rather than a
// regex pass because every rule must be suppressed inside string literals: a
// trailing comma in prose, or a brace in a Java snippet, is content.
func rewriteLoose(s string) (string, []string, bool) {
	var b strings.Builder
	b.Grow(len(s))
	var repairs []string
	seen := map[string]bool{}
	note := func(r string) {
		if !seen[r] {
			seen[r] = true
			repairs = append(repairs, r)
		}
	}

	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '"':
			j := scanQuoted(s, i, '"')
			if j < 0 {
				// An unterminated literal is truncation, not bad syntax. Report
				// it rather than trying to close it.
				return "", nil, true
			}
			b.WriteString(s[i:j])
			i = j

		case c == '\'':
			lit, j, ok := convertSingleQuoted(s, i)
			if !ok {
				return "", nil, true // unterminated single-quoted literal
			}
			b.WriteString(lit)
			i = j
			note("converted single-quoted strings to double-quoted")

		case c == ',':
			k := i + 1
			for k < len(s) && isJSONSpace(s[k]) {
				k++
			}
			if k < len(s) && (s[k] == '}' || s[k] == ']') {
				note("dropped a trailing comma")
				i++
				continue
			}
			b.WriteByte(c)
			i++

		case isIdentStart(c):
			j := i
			for j < len(s) && isIdentChar(s[j]) {
				j++
			}
			word := s[i:j]
			switch word {
			case "True", "False", "None":
				b.WriteString(map[string]string{"True": "true", "False": "false", "None": "null"}[word])
				note("converted Python literals (True/False/None)")
				i = j
				continue
			}
			// An identifier immediately before a colon is an unquoted key.
			k := j
			for k < len(s) && isJSONSpace(s[k]) {
				k++
			}
			if k < len(s) && s[k] == ':' {
				b.WriteString(`"` + word + `"`)
				note("quoted unquoted object keys")
				i = j
				continue
			}
			b.WriteString(word)
			i = j

		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String(), repairs, false
}

// scanQuoted returns the index just past the literal starting at i, or -1 if it
// is unterminated.
func scanQuoted(s string, i int, quote byte) int {
	for j := i + 1; j < len(s); j++ {
		switch s[j] {
		case '\\':
			j++
		case quote:
			return j + 1
		}
	}
	return -1
}

// convertSingleQuoted rewrites 'text' as "text", moving the escaping across:
// \' becomes ', and a bare " must become \".
func convertSingleQuoted(s string, i int) (string, int, bool) {
	end := scanQuoted(s, i, '\'')
	if end < 0 {
		return "", 0, false
	}
	inner := s[i+1 : end-1]
	var b strings.Builder
	b.WriteByte('"')
	for k := 0; k < len(inner); k++ {
		switch {
		case inner[k] == '\\' && k+1 < len(inner) && inner[k+1] == '\'':
			b.WriteByte('\'')
			k++
		case inner[k] == '\\' && k+1 < len(inner):
			b.WriteByte('\\')
			b.WriteByte(inner[k+1])
			k++
		case inner[k] == '"':
			b.WriteString(`\"`)
		default:
			b.WriteByte(inner[k])
		}
	}
	b.WriteByte('"')
	return b.String(), end, true
}

func isJSONSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isIdentStart(c byte) bool {
	return c == '_' || c == '$' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentChar(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}
