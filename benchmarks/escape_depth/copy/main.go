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
var ladder = []struct {
	depth int
	code  string
}{
	{1, `let a = "hi"; a + "!"`},
	{2, `let a = "say \"hi\""; a.length`},
	{3, `let a = "{\"k\": \"v\"}"; a.length`},
	{4, `let a = "{\"k\": \"{\\\"n\\\": 1}\"}"; a.length`},
	{5, `let a = "{\"k\": \"{\\\"n\\\": \\\"{\\\\\\\"z\\\\\\\": 2}\\\"}\"}"; a.length`},
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
	user := "Call eval with EXACTLY this code. It is every line between the " +
		"markers, excluding them. Reproduce it character for character.\n" +
		"@@@BEGIN\n" + code + "\n@@@END"
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

func main() {
	c := llm.NewClient("https://llm.iodesystems.com/v1", "", "Qwen3-6-27B-MPT")
	reps := 3
	type cell struct{ codeOK, outOK, fail int }
	res := map[string]*cell{}
	key := func(d int, h bool) string {
		if h {
			return fmt.Sprintf("d%d-heredoc", d)
		}
		return fmt.Sprintf("d%d-native", d)
	}

	for _, l := range ladder {
		truth := run(l.code)
		fmt.Printf("depth %d truth=%q\n", l.depth, truth)
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
				if strings.TrimSpace(got) == strings.TrimSpace(l.code) {
					res[k].codeOK++
				}
				if out := run(got); out == truth {
					res[k].outOK++
				} else {
					fmt.Printf("   %-14s rep%d out=%q want=%q\n", k, r, out, truth)
					if strings.TrimSpace(got) != strings.TrimSpace(l.code) {
						fmt.Printf("      code: %s\n      want: %s\n", got, l.code)
					}
				}
			}
		}
	}

	fmt.Printf("\n%-8s %-22s %-22s\n", "depth", "native code/out/err", "heredoc code/out/err")
	for _, l := range ladder {
		n, hd := res[key(l.depth, false)], res[key(l.depth, true)]
		fmt.Printf("d%-7d %d/%d/%d%-16s %d/%d/%d\n", l.depth,
			n.codeOK, n.outOK, n.fail, "", hd.codeOK, hd.outOK, hd.fail)
	}
	fmt.Printf("\n(each cell is out of %d reps: exact-code / correct-output / errors)\n", reps)
}
