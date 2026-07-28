package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// mcpshellPath resolves the interpreter used to VERIFY a generated script:
// MCPSHELL if set, else whatever is on PATH.
func mcpshellPath() string {
	if p := os.Getenv("MCPSHELL"); p != "" {
		return p
	}
	return "mcpshell"
}

// Escape-depth ladder. Each level nests one more layer of quoting, which is where
// the backslashes double: a string, holding JSON, holding JSON, holding JSON.
// Every level has a distinct deterministic output, so a corrupted script is
// caught by BEHAVIOUR, not just by a byte diff.
// AUTHORED, not copied. The model is told WHAT the string must contain, in
// prose, and has to derive the escaping itself. That is the shape of a real edit
// driven through eval, and it is where escape depth is claimed to bite: copying a
// script is transcription, authoring one is arithmetic on backslashes.
//
// Every expected length was verified against the interpreter first. Two of my
// hand-counted values were wrong, which would have been reported as model
// failures.
var ladder = []struct {
	depth int
	code  string // the SPEC shown to the model, plus the expected output
	want  string
}{
	{1, "a string holding exactly the two characters backslash and n", "2"},
	{2, `a string containing exactly: say "hi"  (the double quotes are part of it)`, "8"},
	{3, `a string containing exactly this JSON: {"k": "v"}`, "10"},
	{4, `a string containing exactly this JSON: {"k": "{\"n\": 1}"}`, "19"},
	{5, `a Windows path string: C:\Users\dev`, "12"},
}

func run(code string) string {
	f, err := os.CreateTemp("", "esc*.js")
	if err != nil {
		return "ERR " + err.Error()
	}
	defer os.Remove(f.Name())
	_, _ = f.WriteString(code)
	_ = f.Close()
	out, _ := exec.Command(mcpshellPath(), "-f", f.Name()).CombinedOutput()
	return strings.TrimSpace(string(out))
}

func evalTool() llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = "eval"
	td.Function.Description = "Evaluate mcpshell code and return its result."
	_ = json.Unmarshal([]byte(`{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}`), &td.Function.Parameters)
	return td
}

func ask(c *llm.Client, code string, heredoc bool, temp float64) (string, error) {
	tools := []llm.ToolDef{evalTool()}
	// The declaration requirement is stated because mcpshell is a JS SUBSET that
	// rejects a bare assignment. Without it, nine of eleven failures in the first
	// run were "'s' is not defined" — the benchmark was measuring prompt quality,
	// not escaping, and one cell read 0/3 while every escape in it was correct.
	user := "Call eval with mcpshell code that builds " + code +
		". Declare the variable with `let` (a bare assignment is a syntax error), " +
		"and make the LAST expression that variable's .length so the number is " +
		"the output. No other output."
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if !heredoc {
		opts := &llm.ChatOpts{ToolChoice: "required", Temperature: &temp, MaxTokens: 900}
		ch, err := c.ChatStream(ctx, []llm.Message{{Role: "user", Content: user}}, tools, opts)
		if err != nil {
			return "", err
		}
		var got string
		for chunk := range ch {
			if chunk.ToolCall != nil {
				var a struct{ Code string }
				if e := json.Unmarshal([]byte(chunk.ToolCall.Function.Arguments), &a); e == nil {
					got = a.Code
				} else {
					return "", fmt.Errorf("args unparseable: %v", e)
				}
			}
			if chunk.Error != "" {
				return "", fmt.Errorf("%s", chunk.Error)
			}
		}
		return got, nil
	}

	opts := &llm.ChatOpts{Grammar: llm.HeredocGrammar(tools), Temperature: &temp,
		MaxTokens: 900, Stop: llm.HeredocStop()}
	ch, err := c.ChatStream(ctx, []llm.Message{
		{Role: "system", Content: llm.HeredocSystemPrompt(tools)},
		{Role: "user", Content: user},
	}, nil, opts)
	if err != nil {
		return "", err
	}
	var raw string
	for chunk := range ch {
		raw += chunk.Content
		if chunk.Error != "" {
			return "", fmt.Errorf("%s", chunk.Error)
		}
	}
	calls, _, e := llm.ParseToolCalls(llm.CloseHeredoc(raw))
	if e != nil {
		return "", e
	}
	if len(calls) == 0 {
		return "", fmt.Errorf("no call")
	}
	var a struct{ Code string }
	if e := json.Unmarshal([]byte(calls[0].Function.Arguments), &a); e != nil {
		return "", e
	}
	return a.Code, nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func main() {
	c := llm.NewClient("https://llm.iodesystems.com/v1", "", "Qwen3-6-27B-MPT")
	reps := 3
	type cell struct{ outOK, langErr, wrong, fail int }
	res := map[string]*cell{}
	key := func(d int, h bool) string {
		if h {
			return fmt.Sprintf("d%d-heredoc", d)
		}
		return fmt.Sprintf("d%d-native", d)
	}

	for _, l := range ladder {
		truth := l.want
		fmt.Printf("case a%d want=%q  (%s)\n", l.depth, truth, l.code)
		for _, h := range []bool{false, true} {
			k := key(l.depth, h)
			res[k] = &cell{}
			for r := 0; r < reps; r++ {
				temp := 0.0
				if r > 0 {
					temp = 0.4
				}
				got, err := ask(c, l.code, h, temp)
				if err != nil {
					res[k].fail++
					fmt.Printf("   %-14s rep%d ERROR %v\n", k, r, err)
					continue
				}

				out := run(got)
				switch {
				case out == truth:
					res[k].outOK++
				case strings.Contains(out, "is not defined"),
					strings.Contains(out, "Syntax error"):
					// The model wrote invalid mcpshell, which says nothing about
					// whether it escaped correctly. Counted apart so it cannot be
					// mistaken for an escaping failure.
					res[k].langErr++
					fmt.Printf("   %-14s rep%d LANG  %s\n      code: %s\n",
						k, r, firstLine(out), got)
				default:
					res[k].wrong++
					fmt.Printf("   %-14s rep%d WRONG out=%q want=%q\n      code: %s\n",
						k, r, firstLine(out), truth, got)
				}
			}
		}
	}

	fmt.Printf("\n%-6s %-22s %-22s\n", "case", "native ok/wrong/lang", "heredoc ok/wrong/lang")
	for _, l := range ladder {
		n, hd := res[key(l.depth, false)], res[key(l.depth, true)]
		fmt.Printf("a%-5d %d/%d/%d%-16s %d/%d/%d\n", l.depth,
			n.outOK, n.wrong, n.langErr, "", hd.outOK, hd.wrong, hd.langErr)
	}
	fmt.Printf("\nout of %d reps. WRONG = escaping produced the wrong string.\n"+
		"LANG = invalid mcpshell, which says nothing about escaping.\n", reps)
}
