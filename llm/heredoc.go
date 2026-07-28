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

// ParseHeredocCalls extracts tool calls from model content, returning the calls
// and whatever prose surrounded them.
//
// Arguments are lifted to a strict JSON object, so what reaches a dispatcher and
// what is persisted are ordinary escaped JSON. The raw form exists only between
// the model and this function, which is the same division the native format uses
// (and the reason escaping costs 0 tokens: it never reaches the tokenizer).
//
// An argument is a raw block (`key: <ext>~~~SENTINEL`, then lines, then the
// sentinel alone). A block's declared extension is recorded under "<key>_type"
// when present, which is how a tool learns a body is xml vs json without sniffing.
//
// A one-line `key: value` is also accepted, because a model that skips the block
// form for a short value has still said something unambiguous. HeredocGrammar
// deliberately does NOT offer that choice: given it, the model put a MULTI-line
// value inline and the body's own lines were then read as further parameters.
// Parse leniently, generate strictly.
func ParseHeredocCalls(content string) (calls []ToolCall, prose string, err error) {
	lines := strings.Split(content, "\n")
	var rest []string
	for i := 0; i < len(lines); {
		name, ok := strings.CutPrefix(strings.TrimSpace(lines[i]), "@@call ")
		if !ok {
			rest = append(rest, lines[i])
			i++
			continue
		}
		tc, next, e := parseOneCall(strings.TrimSpace(name), lines, i+1)
		if e != nil {
			return nil, strings.Join(rest, "\n"), e
		}
		calls = append(calls, tc)
		i = next
	}
	return calls, strings.TrimSpace(strings.Join(rest, "\n")), nil
}

func parseOneCall(name string, lines []string, i int) (ToolCall, int, error) {
	var tc ToolCall
	if name == "" {
		return tc, i, fmt.Errorf("llm: @@call with no tool name")
	}
	args := map[string]any{}
	for i < len(lines) {
		line := lines[i]
		if strings.TrimSpace(line) == "@@end" {
			i++
			tc.Type = "function"
			tc.Function.Name = name
			b, err := json.Marshal(args)
			if err != nil {
				return tc, i, fmt.Errorf("llm: encoding %s arguments: %w", name, err)
			}
			tc.Function.Arguments = string(b)
			return tc, i, nil
		}
		key, val, found := strings.Cut(line, ":")
		if !found {
			// A stray line inside a call block. Skip rather than fail: the model
			// sometimes prefixes a blank line, and refusing the whole call over
			// whitespace would throw away real work.
			i++
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if ext, sentinel, isBlock := cutHeredocOpen(val); isBlock {
			body, next, e := readBody(lines, i+1, sentinel)
			if e != nil {
				return tc, next, fmt.Errorf("llm: %s.%s: %w", name, key, e)
			}
			args[key] = body
			if ext != "" {
				args[key+"_type"] = ext
			}
			i = next
			continue
		}
		args[key] = val
		i++
	}
	return tc, i, fmt.Errorf("llm: @@call %s is missing its @@end", name)
}

// cutHeredocOpen recognises `<ext>~~~SENTINEL` and returns the extension and the
// sentinel that will terminate the body.
func cutHeredocOpen(val string) (ext, sentinel string, ok bool) {
	idx := strings.Index(val, HeredocOpen)
	if idx < 0 {
		return "", "", false
	}
	ext = strings.TrimSpace(val[:idx])
	sentinel = strings.TrimSpace(val[idx:])
	if sentinel == HeredocOpen {
		return "", "", false // "~~~" with no sentinel is not a block opener
	}
	return ext, sentinel, true
}

// readBody consumes lines until one equals sentinel exactly. A missing sentinel
// is reported as an error rather than closed silently: an unterminated body is
// TRUNCATED output, and inventing its end is how a half-written file gets
// dispatched as though it were complete.
func readBody(lines []string, i int, sentinel string) (string, int, error) {
	var body []string
	for ; i < len(lines); i++ {
		if lines[i] == sentinel || strings.TrimRight(lines[i], "\r") == sentinel {
			return strings.Join(body, "\n"), i + 1, nil
		}
		body = append(body, lines[i])
	}
	return "", i, fmt.Errorf("body never terminated by %q (generation was cut off; "+
		"the call was NOT completed)", sentinel)
}

// HeredocGrammar builds a GBNF that constrains generation to this format for
// exactly these tools. The sampler cannot emit a tool name that is not listed or
// a sentinel that is not the pinned one, which removes the two things the model
// was measured to get wrong when merely instructed: it drifted on the delimiter
// (`<<END` became `<|<END`) and ignored some formats entirely.
//
// Pass the result as ChatOpts.Grammar with NO tools set. llama.cpp rejects the
// combination, which is why this format is parsed from content.
func HeredocGrammar(tools []ToolDef) string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name != "" {
			names = append(names, `"`+t.Function.Name+`"`)
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		names = []string{`[a-zA-Z_] [a-zA-Z0-9_]*`}
	}
	var b strings.Builder
	b.WriteString("root      ::= call+\n")
	b.WriteString(`call      ::= "@@call " name "\n" param+ "@@end\n"` + "\n")
	b.WriteString("name      ::= " + strings.Join(names, " | ") + "\n")
	// EVERY argument is a block, with no inline alternative. Measured: given the
	// choice, the model put a multi-line value inline, after which the body's own
	// lines were re-read as further `key: value` parameters and the call was
	// garbage. Removing the alternative removes the mistake — that is what a
	// grammar is FOR, and it is only available here because llama.cpp refuses a
	// grammar alongside `tools`.
	b.WriteString("param     ::= key \": \" block\n")
	b.WriteString("key       ::= [a-zA-Z_] [a-zA-Z0-9_]*\n")
	// The sentinel is a literal on both ends, so the model cannot choose it.
	b.WriteString(`block     ::= ext? "` + HeredocSentinel + `\n" bodyline* "` +
		HeredocSentinel + `\n"` + "\n")
	b.WriteString("ext       ::= [a-z]+\n")
	b.WriteString(`bodyline  ::= [^\n]* "\n"` + "\n")
	return b.String()
}

// HeredocSystemPrompt describes the format to the model. The grammar enforces the
// shape; this explains the intent, which the grammar cannot (a grammar can force
// a sentinel but not convey that the body needs no escaping).
func HeredocSystemPrompt(tools []ToolDef) string {
	var b strings.Builder
	b.WriteString("To call a tool, emit a call block and nothing else. EVERY argument " +
		"is a raw block terminated by " + HeredocSentinel + ".\n\n")
	b.WriteString("Example, calling write_file with two arguments:\n\n")
	// A CONCRETE example, not a placeholder skeleton. Measured twice: given
	// `<arg>` or `<ext>` to fill in, the model emits those literally.
	b.WriteString("@@call write_file\n" +
		"path: " + HeredocSentinel + "\n" +
		"src/Hello.java\n" +
		HeredocSentinel + "\n" +
		"content: java" + HeredocSentinel + "\n" +
		"public class Hello {\n" +
		"    System.out.println(\"hi\");\n" +
		"}\n" +
		HeredocSentinel + "\n" +
		"@@end\n\n")
	b.WriteString("The word after the colon (\"java\" above) names the body type and " +
		"is optional.\n")
	b.WriteString("Inside a block write the text EXACTLY as it should be: no backslash " +
		"escapes, no doubled backslashes, no quoting.\n")
	b.WriteString("A block ends at a line that is exactly " + HeredocSentinel + ".\n\n")
	if len(tools) > 0 {
		b.WriteString("Tools:\n")
		for _, t := range tools {
			schema, _ := json.Marshal(t.Function.Parameters)
			fmt.Fprintf(&b, "- %s: %s\n  %s\n", t.Function.Name, t.Function.Description, schema)
		}
	}
	return b.String()
}
