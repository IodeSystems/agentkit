package llm

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// A tool-call format that carries argument values as RAW heredoc bodies instead
// of escaped JSON strings, parsed out of the model's ordinary content.
//
// WHY THIS EXISTS. The native XML tool format cannot carry a value containing its
// own delimiter, and it fails silently. Measured against Qwen3 with the server's
// parser bypassed, asking for the value "before\n</parameter>\nafter" produces
// this entire generation:
//
//	<tool_call>
//	<function=echo>
//	<parameter=text>
//	before
//	</parameter>
//	</function>
//	</tool_call>
//
// The model wrote "before", closed the parameter, and stopped. It never emitted
// the delimiter and never emitted "after". A related case corrupts instead of
// truncating: asked for code containing `<parameter=path>` it writes
// `<parameter:path>`, one character different, in its own reasoning block before
// any tool call exists. Run through an interpreter, that program executes and
// returns a plausible WRONG answer.
//
// So this is not a parser bug and no parser fix reaches it. To the model those
// byte sequences are structure, and it will not quote its own syntax. The only
// lever is a delimiter that does not occur in the content being handled, and the
// only way to stop the model choosing one badly is to not let it choose:
// HeredocGrammar pins the sentinel and the sampler enforces it.
//
// COST. Freeform content is the ONLY place a grammar is available: llama.cpp
// refuses `grammar` alongside `tools` ("Cannot specify grammar with tools"),
// measured by ProbeToolSurface as GrammarWithToolsAccepted=false. Giving up the
// native format is therefore what BUYS sampler-level enforcement rather than
// what loses it.

// HeredocSentinel terminates a raw body. It is deliberately long and
// unpronounceable: a body line must equal it exactly to end the block, so the
// only failure mode is content that contains this exact line. Short sentinels
// like "EOF" or "END" appear in real files and shell scripts.
//
// It must not begin with "<". `<|` is a single token in Qwen-family vocabularies,
// and a delimiter starting with "<" pulls generation toward it: asked to emit
// `<<END` the model produced `<|<END`. Measured alternatives — `~~~END` and
// `@@END` reproduced exactly, `<<END` and `:::END` did not.
const HeredocSentinel = "~~~AGENTKIT_EOF_7F3A"

// HeredocOpen introduces a raw body: `key: <ext>` + this + newline.
const HeredocOpen = "~~~"

// HeredocEnd terminates a call block, and doubles as the stop sequence.
//
// A grammar does not force EOS when it completes: measured with `root ::= call`,
// the model produced a valid call and then repeated it to the token cap. Pass
// HeredocStop as ChatOpts.Stop and re-append HeredocEnd before parsing, since the
// provider strips the matched stop string.
const HeredocEnd = "@@end"

// HeredocStop is the stop sequence to pair with HeredocGrammar.
func HeredocStop() []string { return []string{HeredocEnd} }

// CloseHeredoc re-appends the terminator a stop sequence consumed, so the text
// can be parsed. It is deliberately explicit rather than making the parser
// tolerate a missing terminator: an unterminated block is how TRUNCATED output
// looks, and the parser must keep refusing that.
func CloseHeredoc(raw string) string {
	if strings.Contains(raw, HeredocEnd) {
		return raw
	}
	return strings.TrimRight(raw, "\n") + "\n" + HeredocEnd + "\n"
}

// HeredocTypes is the closed set of body type tags the grammar admits. Closed on
// purpose: an open [a-z]+ let the model place an argument VALUE in the tag slot.
// `json` is load-bearing (it makes the body a real value rather than a string);
// the rest are descriptive and only reach the tool as "<key>_type".
var HeredocTypes = []string{
	"go", "java", "js", "json", "md", "py", "sh", "sql", "txt", "xml", "yaml",
}

func quoteAll(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = `"` + s + `"`
	}
	return out
}

// CallPrefix introduces a tool call. Everything after it, up to the end of the
// JSON value, is the call's arguments as json-loose-heredoc.
const CallPrefix = "@@call "

// ParseToolCalls extracts tool calls from model content, plus any surrounding
// prose.
//
// The shape is a prefix line naming the tool, then ONE json-loose-heredoc object:
//
//	@@call write_file
//	{ path: "a.java", content: ~~~EOF
//	public class A {}
//	~~~EOF
//	}
//
// Arguments arrive as strict JSON with real types, so a tool whose schema wants a
// number or an object gets one. That is the reason for parsing JSON rather than a
// line format: only STRING bodies are awkward to escape, so only string bodies
// get alternate syntax, and `{name: "prod", retries: 3}` stays two obvious pairs
// instead of two heredocs.
//
// The object self-terminates, so no end marker is needed to parse. An UNCLOSED
// object, string or body is reported as truncation rather than closed silently.
func ParseToolCalls(content string) (calls []ToolCall, prose string, err error) {
	var rest strings.Builder
	i := 0
	for {
		idx := strings.Index(content[i:], CallPrefix)
		if idx < 0 {
			rest.WriteString(content[i:])
			break
		}
		rest.WriteString(content[i : i+idx])
		j := i + idx + len(CallPrefix)

		nl := strings.IndexByte(content[j:], '\n')
		if nl < 0 {
			return nil, rest.String(), fmt.Errorf("llm: %s with no arguments (generation was cut off)", CallPrefix)
		}
		name := strings.TrimSpace(content[j : j+nl])
		if name == "" {
			return nil, rest.String(), fmt.Errorf("llm: %s with no tool name", CallPrefix)
		}

		args, next, e := rewriteLooseHeredoc(content, j+nl+1)
		if e != nil {
			return nil, rest.String(), fmt.Errorf("llm: %s: %w", name, e)
		}
		if !json.Valid([]byte(args)) {
			return nil, rest.String(), fmt.Errorf("llm: %s: rewritten arguments are not valid JSON: %s", name, args)
		}
		if strings.TrimSpace(args)[0] != '{' {
			return nil, rest.String(), fmt.Errorf("llm: %s: arguments must be a JSON object, got %s", name, args)
		}
		var tc ToolCall
		tc.Type = "function"
		tc.Function.Name = name
		tc.Function.Arguments = args
		calls = append(calls, tc)
		i = next
	}
	return calls, strings.TrimSpace(rest.String()), nil
}

// HeredocGrammar constrains generation to `@@call <name>` plus one
// json-loose-heredoc object, for exactly these tools.
//
// The grammar earns its keep by removing choices the model was measured to get
// wrong. It cannot emit an unlisted tool name. It cannot invent a body delimiter
// (asked to produce `<<END` it produced `<|<END`, because `<|` is one token).
// And the object is a real JSON object, so a value that is a number or a nested
// object is expressible without a type tag.
//
// Pass it as ChatOpts.Grammar with NO tools set: llama.cpp refuses a grammar
// alongside `tools`, which is exactly why this format is parsed from content.
// Pair it with ChatOpts.Stop — a completed grammar does NOT force EOS, and
// without a stop the model re-emits the same call to the token cap.
func HeredocGrammar(tools []ToolDef) string {
	var b strings.Builder
	var alts []string
	var rules strings.Builder

	named := make([]ToolDef, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name != "" {
			named = append(named, t)
		}
	}
	sort.Slice(named, func(i, j int) bool { return named[i].Function.Name < named[j].Function.Name })

	for n, t := range named {
		id := fmt.Sprintf("t%d", n)
		alts = append(alts, id+"call")
		keys := paramKeys(t)
		// The terminator is part of the grammar so ChatOpts.Stop can match it.
		// Measured: without it, an object whose pair-list allows repetition let
		// the model emit `path`/`content` over and over to the token cap. `}`
		// was always legal, but so was another pair, and nothing forced the end.
		fmt.Fprintf(&rules, "%scall  ::= \"%s%s\\n\" %sobj \"\\n%s\"\n",
			id, CallPrefix, t.Function.Name, id, HeredocEnd)
		if len(keys) == 0 {
			fmt.Fprintf(&rules, "%sobj   ::= \"{\" ws \"}\"\n", id)
			continue
		}
		// Per-tool key set. Measured: with an open `key ::= [a-zA-Z_]...` the
		// model copied `overwrite: true` out of the prompt's EXAMPLE into a call
		// whose schema has no such parameter, and it parsed cleanly. A closed set
		// makes a hallucinated argument unrepresentable rather than merely wrong.
		// Bounded, not `*`. Measured: with unlimited pairs the model emitted
		// `path`/`content` over and over to the token cap — `}` was always legal
		// but so was another pair, and nothing forced the end. An object cannot
		// hold more pairs than the tool has parameters, so cap it there and the
		// object must close.
		fmt.Fprintf(&rules, "%sobj   ::= \"{\" ws %spair (ws \",\" ws %spair){0,%d} ws \",\"? ws \"}\"\n",
			id, id, id, len(keys)-1)
		fmt.Fprintf(&rules, "%spair  ::= %skey ws \":\" ws value\n", id, id)
		fmt.Fprintf(&rules, "%skey   ::= %s\n", id, strings.Join(quoteAll(keys), " | "))
	}
	if len(alts) == 0 {
		alts = []string{"anycall"}
		rules.WriteString(`anycall ::= "` + CallPrefix + `" [a-zA-Z_] [a-zA-Z0-9_]* "\n" object "\n` + HeredocEnd + `"` + "\n")
		rules.WriteString(`object  ::= "{" ws (pair (ws "," ws pair)*)? ws "}"` + "\n")
		rules.WriteString(`pair    ::= key ws ":" ws value` + "\n")
		rules.WriteString(`key     ::= [a-zA-Z_] [a-zA-Z0-9_]*` + "\n")
	}

	b.WriteString("root    ::= " + strings.Join(alts, " | ") + "\n")
	b.WriteString(rules.String())
	// A raw body is a STRING alternative, not a replacement for JSON: numbers,
	// booleans, arrays and nested objects stay themselves, so a tool whose schema
	// wants an integer receives one.
	b.WriteString(`value   ::= object | array | string | body | number | "true" | "false" | "null"` + "\n")
	b.WriteString(`object  ::= "{" ws (opair (ws "," ws opair)*)? ws "}"` + "\n")
	b.WriteString(`opair   ::= okey ws ":" ws value` + "\n")
	b.WriteString(`okey    ::= "\"" [^"]* "\"" | [a-zA-Z_] [a-zA-Z0-9_]*` + "\n")
	b.WriteString(`array   ::= "[" ws (value (ws "," ws value)*)? ws "]"` + "\n")
	b.WriteString(`string  ::= "\"" ([^"\\] | "\\" .)* "\""` + "\n")
	// The tag is REQUIRED, not optional. Measured: asked to write a body that
	// itself contains a bare `~~~` line, the model opened with a bare `~~~` and
	// the body closed on its own content. A mandatory tag means the closer is
	// `~~~md`, which ordinary content does not contain — the same reason markdown
	// grows its fences. Failure was safe (a truncation error, never a silently
	// corrupted call), but it lost the call.
	b.WriteString(`body    ::= "` + HeredocOpen + `" tag "\n" line* "` + HeredocOpen + `" tag ","? "\n"` + "\n")
	b.WriteString(`tag     ::= [a-zA-Z0-9_]+` + "\n")
	b.WriteString(`line    ::= [^\n]* "\n"` + "\n")
	b.WriteString(`number  ::= "-"? [0-9]+ ("." [0-9]+)?` + "\n")
	b.WriteString(`ws      ::= [ \t\n]*` + "\n")
	return b.String()
}

// paramKeys lists a tool's declared parameter names, sorted.
func paramKeys(t ToolDef) []string {
	schema, ok := t.Function.Parameters.(map[string]any)
	if !ok {
		return nil
	}
	props, _ := schema["properties"].(map[string]any)
	out := make([]string, 0, len(props))
	for k := range props {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func HeredocSystemPrompt(tools []ToolDef) string {
	var b strings.Builder
	// Short on purpose. The grammar already forbids unknown tool names and any
	// other delimiter, so restating those spends tokens on rules the sampler
	// enforces anyway. What a grammar CANNOT express is the intent: that a body
	// is literal and must never be escaped. That is what this says.
	b.WriteString("Call a tool by emitting this and nothing else:\n\n")
	b.WriteString(CallPrefix + "write_file\n" +
		"{\n" +
		"  path: \"src/Hello.java\",\n" +
		"  overwrite: true,\n" +
		"  content: " + HeredocOpen + "java\n" +
		"public class Hello {\n" +
		"    System.out.println(\"hi\");\n" +
		"}\n" +
		HeredocOpen + "java\n" +
		"}\n" +
		HeredocEnd + "\n\n")
	b.WriteString("The arguments are ordinary JSON, so numbers, booleans, arrays " +
		"and nested objects are written as themselves.\n")
	b.WriteString("For a STRING that is long or contains quotes, newlines or " +
		"backslashes, use a " + HeredocOpen + "tag body instead of a quoted string " +
		"and write the text EXACTLY as it should be, escaping nothing. Close it " +
		"with the same " + HeredocOpen + "tag on its own line.\n")
	b.WriteString("ALWAYS put a tag after " + HeredocOpen + " (its type: md, java, " +
		"json, sh). A bare " + HeredocOpen + " would be closed by any " + HeredocOpen +
		" the text itself contains.\n")
	b.WriteString("Short plain strings stay quoted: `path: \"a.java\"`.\n\n")
	if len(tools) > 0 {
		b.WriteString("Tools:\n")
		for _, t := range tools {
			fmt.Fprintf(&b, "- %s(%s): %s\n",
				t.Function.Name, strings.Join(paramNames(t), ", "), t.Function.Description)
		}
	}
	return b.String()
}

// paramNames lists a tool's parameters, required ones first and marked. The full
// JSON Schema is deliberately NOT dumped: it duplicates what the grammar and the
// example already convey, and on a multi-tool loop the schemas dominate the
// prompt. A name and a required-marker is what the model acts on.
func paramNames(t ToolDef) []string {
	schema, ok := t.Function.Parameters.(map[string]any)
	if !ok {
		return nil
	}
	props, _ := schema["properties"].(map[string]any)
	req := map[string]bool{}
	if rs, ok := schema["required"].([]any); ok {
		for _, r := range rs {
			if s, ok := r.(string); ok {
				req[s] = true
			}
		}
	}
	var out []string
	for name := range props {
		out = append(out, name)
	}
	sort.Strings(out)
	for i, name := range out {
		if !req[name] {
			out[i] = name + "?"
		}
	}
	return out
}
