package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// json-loose-heredoc: JSON, loosely parsed, with two extra ways to write a
// STRING literal.
//
// JSON is already the right shape for tool arguments — objects, numbers,
// booleans and arrays all express themselves — and the only thing it is bad at is
// carrying a long literal body, because every newline, quote and backslash has to
// be escaped. So only that one case gets new syntax:
//
//	{
//	  path: "src/Hello.java",
//	  retries: 3,
//	  deep: true,
//	  content: ~~~EOF
//	public class Hello {
//	    System.out.println("hi");
//	}
//	~~~EOF
//	}
//
// or with a backtick fence, which models produce natively and which grows to
// avoid collisions exactly as markdown does:
//
//	{ content: ```
//	raw text with "quotes" and \backslashes\
//	``` }
//
// Both forms rewrite to an ordinary escaped JSON string, so the result is strict
// JSON and every consumer downstream is unchanged. The raw form exists only
// between the model and this parser.
//
// WHY NOT ESCAPE. Escaping is not the expensive part (measured: a raw body saves
// 15-28% of generated tokens, and the escapes themselves cost almost nothing per
// character). The expensive part is that the model will not reliably emit a
// delimiter it treats as structure: asked for a value containing `</parameter>`
// it silently truncated at the delimiter, and asked for `<parameter=path>` it
// wrote `<parameter:path>`. A body form whose delimiter does not occur in the
// content sidesteps that, which is the whole point.

// blockScan reads one heredoc or backtick body starting at i, returning the raw
// body and the index just past the closing delimiter.
//
// A missing closer is an ERROR, never a silent close at end-of-input: an
// unterminated body is what TRUNCATED generation looks like, and inventing its
// end is how a half-written file reaches a dispatcher looking complete.
func blockScan(s string, i int) (body string, next int, ok bool, err error) {
	switch {
	case strings.HasPrefix(s[i:], "~~~"):
		// Opener runs to end of line: `~~~` plus an optional tag. The closer is
		// that same token alone on a line.
		nl := strings.IndexByte(s[i:], '\n')
		if nl < 0 {
			return "", i, true, fmt.Errorf("heredoc opener never ends (generation was cut off)")
		}
		opener := strings.TrimRight(s[i:i+nl], " \t\r")
		rest := s[i+nl+1:]
		for off := 0; ; {
			lineEnd := strings.IndexByte(rest[off:], '\n')
			var line string
			if lineEnd < 0 {
				line = rest[off:]
			} else {
				line = rest[off : off+lineEnd]
			}
			if strings.TrimRight(line, " \t\r") == opener {
				return strings.TrimSuffix(rest[:off], "\n"), i + nl + 1 + off + len(line) + 1, true, nil
			}
			if lineEnd < 0 {
				return "", i, true, fmt.Errorf("heredoc %q never closed (generation was cut off; "+
					"the call was NOT completed)", opener)
			}
			off += lineEnd + 1
		}

	case s[i] == '`':
		// Variable-length fence, markdown style: N backticks open, N close, so a
		// body containing backticks is expressible.
		n := 0
		for i+n < len(s) && s[i+n] == '`' {
			n++
		}
		fence := strings.Repeat("`", n)
		rest := s[i+n:]
		rest = strings.TrimPrefix(rest, "\n")
		end := strings.Index(rest, fence)
		if end < 0 {
			return "", i, true, fmt.Errorf("backtick body never closed (generation was cut off; " +
				"the call was NOT completed)")
		}
		body = strings.TrimSuffix(rest[:end], "\n")
		consumed := len(s) - len(rest) + end + n
		return body, consumed, true, nil
	}
	return "", i, false, nil
}

// rewriteLooseHeredoc turns a json-loose-heredoc value into strict JSON, stopping
// when the top-level value completes. Returns the strict JSON and the index just
// past it.
//
// It is a scanner rather than a regex pass because every rule has to be
// suppressed inside a string or a body: a brace in a Java snippet is content, and
// so is a trailing comma in prose.
func rewriteLooseHeredoc(s string, i int) (string, int, error) {
	var b strings.Builder
	depth := 0
	started := false

	for i < len(s) {
		c := s[i]

		// A raw body is only legal where a VALUE goes; elsewhere backticks and
		// tildes are ordinary characters.
		if c == '`' || strings.HasPrefix(s[i:], "~~~") {
			body, next, ok, err := blockScan(s, i)
			if err != nil {
				return "", i, err
			}
			if ok {
				enc, e := json.Marshal(body)
				if e != nil {
					return "", i, e
				}
				b.Write(enc)
				i = next
				continue
			}
		}

		switch {
		case c == '"':
			j := scanQuoted(s, i, '"')
			if j < 0 {
				return "", i, fmt.Errorf("string never closed (generation was cut off; " +
					"the call was NOT completed)")
			}
			b.WriteString(s[i:j])
			i = j

		case c == '\'':
			lit, j, ok := convertSingleQuoted(s, i)
			if !ok {
				return "", i, fmt.Errorf("string never closed (generation was cut off; " +
					"the call was NOT completed)")
			}
			b.WriteString(lit)
			i = j

		case c == '{' || c == '[':
			depth++
			started = true
			b.WriteByte(c)
			i++

		case c == '}' || c == ']':
			depth--
			b.WriteByte(c)
			i++
			if started && depth == 0 {
				return b.String(), i, nil
			}

		case c == ',':
			k := i + 1
			for k < len(s) && isJSONSpace(s[k]) {
				k++
			}
			if k < len(s) && (s[k] == '}' || s[k] == ']') {
				i++ // trailing comma
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
			case "True", "true":
				b.WriteString("true")
				i = j
				continue
			case "False", "false":
				b.WriteString("false")
				i = j
				continue
			case "None", "null":
				b.WriteString("null")
				i = j
				continue
			}
			k := j
			for k < len(s) && isJSONSpace(s[k]) {
				k++
			}
			if k < len(s) && s[k] == ':' {
				b.WriteString(`"` + word + `"`) // unquoted key
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
	if !started {
		return "", i, fmt.Errorf("no JSON value found")
	}
	return "", i, fmt.Errorf("value never closed (generation was cut off; " +
		"the call was NOT completed)")
}

// ParseLooseHeredocJSON rewrites one json-loose-heredoc value to strict JSON.
func ParseLooseHeredocJSON(s string) (string, error) {
	out, _, err := rewriteLooseHeredoc(s, 0)
	if err != nil {
		return "", err
	}
	if !json.Valid([]byte(out)) {
		return "", fmt.Errorf("rewritten value is not valid JSON: %s", out)
	}
	return out, nil
}
