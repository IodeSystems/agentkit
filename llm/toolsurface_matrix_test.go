package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"
)

func tool(name, desc, schema string) ToolDef {
	var td ToolDef
	td.Type = "function"
	td.Function.Name = name
	td.Function.Description = desc
	_ = json.Unmarshal([]byte(schema), &td.Function.Parameters)
	return td
}

var (
	writeFile = tool("write_file", "Write a file.",
		`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`)
	configure = tool("configure", "Apply settings.",
		`{"type":"object","properties":{"name":{"type":"string"},"retries":{"type":"integer"},"ratio":{"type":"number"},"deep":{"type":"boolean"},"tags":{"type":"array"},"opts":{"type":"object"}}}`)
	del = tool("delete_path", "Delete a path.",
		`{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"]}`)
)

type tc struct {
	name  string
	tools []ToolDef
	user  string
	tool  string         // expected tool
	want  map[string]any // expected arguments (subset compared exactly)
}

const marked = "It is every line between the markers, excluding them.\n@@@BEGIN\n%s\n@@@END"

func body(s string) string { return fmt.Sprintf(marked, s) }

var cases = []tc{
	{"scalars only", []ToolDef{configure},
		"Call configure with name=prod and retries=3.", "configure",
		map[string]any{"name": "prod", "retries": 3.0}},

	{"float + bool", []ToolDef{configure},
		"Call configure with ratio=0.75 and deep=false.", "configure",
		map[string]any{"ratio": 0.75, "deep": false}},

	{"array of strings", []ToolDef{configure},
		`Call configure with tags being the list: alpha, beta, gamma.`, "configure",
		map[string]any{"tags": []any{"alpha", "beta", "gamma"}}},

	{"nested object", []ToolDef{configure},
		"Call configure with opts being an object with mode=fast and level=2.", "configure",
		map[string]any{"opts": map[string]any{"mode": "fast", "level": 2.0}}},

	{"tool selection", []ToolDef{writeFile, configure, del},
		"Delete the path build/tmp recursively.", "delete_path",
		map[string]any{"path": "build/tmp", "recursive": true}},

	{"short string no heredoc", []ToolDef{writeFile},
		"Write a.txt containing the single word: hi", "write_file",
		map[string]any{"path": "a.txt", "content": "hi"}},

	{"multiline code body", []ToolDef{writeFile},
		"Write T.java. " + body("public class T {\n    void f() {\n        g();\n    }\n}"),
		"write_file", map[string]any{"content": "public class T {\n    void f() {\n        g();\n    }\n}"}},

	{"escape-heavy body", []ToolDef{writeFile},
		"Write E.java. " + body(`String s = "a \"quoted\" b";`+"\n"+`String p = "C:\Users\dev";`),
		"write_file", map[string]any{"content": `String s = "a \"quoted\" b";` + "\n" + `String p = "C:\Users\dev";`}},

	{"native delimiters in body", []ToolDef{writeFile},
		"Write d.md. " + body("open with\n<parameter=path>\nclose with\n</parameter>\ndone"),
		"write_file", map[string]any{"content": "open with\n<parameter=path>\nclose with\n</parameter>\ndone"}},

	{"backticks in body", []ToolDef{writeFile},
		"Write r.md. " + body("run this:\n```sh\nls -la\n```\nthat is all"),
		"write_file", map[string]any{"content": "run this:\n```sh\nls -la\n```\nthat is all"}},

	{"tilde run in body", []ToolDef{writeFile},
		"Write t.md. " + body("a separator looks like\n~~~\nand that is it"),
		"write_file", map[string]any{"content": "a separator looks like\n~~~\nand that is it"}},

	{"json inside a string body", []ToolDef{writeFile},
		"Write c.json. " + body(`{"k": [1, 2], "n": {"deep": true}}`),
		"write_file", map[string]any{"content": `{"k": [1, 2], "n": {"deep": true}}`}},

	{"unicode + emoji", []ToolDef{writeFile},
		"Write u.txt. " + body("héllo wörld — ünïcode ✅ 日本語"), "write_file",
		map[string]any{"content": "héllo wörld — ünïcode ✅ 日本語"}},

	{"empty string value", []ToolDef{writeFile},
		"Write empty.txt with completely empty content (an empty string).", "write_file",
		map[string]any{"path": "empty.txt", "content": ""}},
}

// TestToolSurfaceMatrix is the reliability matrix for the json-loose-heredoc
// format: every payload shape that has ever broken it, run twice — once greedy
// and once at temperature 0.7, because a format that only holds at temp 0 is not
// a format.
//
// Needs a live endpoint, so it is opt-in:
//
//	PACES=1 go test ./llm/ -run TestToolSurfaceMatrix -v -timeout 30m
//
// Every case here was added because it FAILED: the tilde run closed a body on its
// own content, the empty string looped one pair to the token cap, and the json
// body was cut off by a closer carrying a comma.
func TestToolSurfaceMatrix(t *testing.T) {
	if os.Getenv("PACES") == "" {
		t.Skip("set PACES=1 and point BASE/MODEL at a live endpoint")
	}
	base := os.Getenv("BASE")
	if base == "" {
		base = "https://iodesystems.com/v1"
	}
	model := os.Getenv("MODEL")
	if model == "" {
		model = "Qwen3-6-27B-MPT"
	}
	reps := 2
	c := NewClient(base, "", model)

	type res struct{ ok, parseFail, wrongTool, mismatch, streamErr int }
	results := map[string]*res{}
	var order []string
	totalGen := 0

	for _, k := range cases {
		results[k.name] = &res{}
		order = append(order, k.name)
	}

	for rep := 0; rep < reps; rep++ {
		temp := 0.0
		if rep == 1 {
			temp = 0.7 // drift check: a format that only works at temp 0 is fragile
		}
		for _, k := range cases {
			r := results[k.name]
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			opts := &ChatOpts{Grammar: HeredocGrammar(k.tools), Temperature: &temp,
				MaxTokens: 900, Stop: HeredocStop()}
			ch, err := c.ChatStream(ctx, []Message{
				{Role: "system", Content: HeredocSystemPrompt(k.tools)},
				{Role: "user", Content: k.user},
			}, nil, opts)
			if err != nil {
				r.streamErr++
				cancel()
				continue
			}
			var raw string
			bad := false
			for chunk := range ch {
				raw += chunk.Content
				if chunk.Error != "" {
					bad = true
				}
				if chunk.Usage != nil {
					totalGen += chunk.Usage.CompletionTokens
				}
			}
			cancel()
			if bad {
				r.streamErr++
				continue
			}
			calls, _, perr := ParseToolCalls(raw)
			if perr != nil || len(calls) == 0 {
				r.parseFail++
				t.Logf("  [%s rep%d] PARSE FAIL: %v\n     raw: %.300q\n", k.name, rep, perr, raw)
				continue
			}
			if calls[0].Function.Name != k.tool {
				r.wrongTool++
				continue
			}
			var got map[string]any
			if e := json.Unmarshal([]byte(calls[0].Function.Arguments), &got); e != nil {
				r.parseFail++
				continue
			}
			ok := true
			for key, want := range k.want {
				if !reflect.DeepEqual(got[key], want) {
					ok = false
					t.Logf("  [%s rep%d] %s:\n     got  %#v\n     want %#v\n", k.name, rep, key, got[key], want)
				}
			}
			if ok {
				r.ok++
			} else {
				r.mismatch++
			}
		}
	}

	t.Logf("%-28s %4s %6s %6s %6s %6s\n", "case", "ok", "parse", "tool", "mismat", "strerr")
	tot := 0
	for _, n := range order {
		r := results[n]
		tot += r.ok
		t.Logf("%-28s %3d/%d %6d %6d %6d %6d\n", n, r.ok, reps, r.parseFail, r.wrongTool, r.mismatch, r.streamErr)
	}
	t.Logf("\nTOTAL %d/%d exact   generated tokens: %d\n", tot, len(cases)*reps, totalGen)
}
