package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// toolDef is a tiny constructor for an OpenAI-format tool. params is a JSON
// Schema (map[string]any) describing the arguments.
func toolDef(name, desc string, params map[string]any) llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = name
	td.Function.Description = desc
	td.Function.Parameters = params
	return td
}

// obj/prop are small JSON-Schema builders so tool defs read declaratively.
func obj(required []string, props map[string]any) map[string]any {
	return map[string]any{"type": "object", "required": required, "properties": props}
}
func prop(typ, desc string) map[string]any { return map[string]any{"type": typ, "description": desc} }

// localTools is the demo's hand-written tool set — pure Go, no MCP process
// required. This is the "local tool call" story: the model requests a call,
// the dispatcher runs Go code, the result feeds back into the loop.
var localTools = []llm.ToolDef{
	toolDef("get_weather", "Return the current weather for a city.",
		obj([]string{"city"}, map[string]any{
			"city":  prop("string", "City name, e.g. Denver"),
			"units": prop("string", "\"c\" or \"f\" (default c)"),
		})),
	toolDef("add", "Add two numbers and return the sum.",
		obj([]string{"a", "b"}, map[string]any{
			"a": prop("number", "first addend"),
			"b": prop("number", "second addend"),
		})),
}

// dispatchLocal executes a local tool call and returns the string the model
// sees as the result. Note the contract: errors meant for the MODEL (unknown
// tool, bad args) are formatted INTO the result string, not returned as a Go
// error — a returned error aborts the whole Turn. This keeps the loop alive so
// the model can correct itself.
func dispatchLocal(_ context.Context, tc llm.ToolCall) (string, error) {
	switch tc.Function.Name {
	case "get_weather":
		var args struct {
			City  string `json:"city"`
			Units string `json:"units"`
		}
		if err := json.Unmarshal([]byte(orEmpty(tc.Function.Arguments)), &args); err != nil {
			return fmt.Sprintf("ERROR: bad arguments: %v", err), nil
		}
		units := strings.ToLower(args.Units)
		temp := "18°C"
		if units == "f" {
			temp = "64°F"
		}
		return fmt.Sprintf("Weather in %s: %s, partly cloudy, wind 8 km/h.", args.City, temp), nil
	case "add":
		var args struct {
			A float64 `json:"a"`
			B float64 `json:"b"`
		}
		if err := json.Unmarshal([]byte(orEmpty(tc.Function.Arguments)), &args); err != nil {
			return fmt.Sprintf("ERROR: bad arguments: %v", err), nil
		}
		return fmt.Sprintf("%g", args.A+args.B), nil
	default:
		return fmt.Sprintf("ERROR: unknown tool %q", tc.Function.Name), nil
	}
}

func orEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "{}"
	}
	return s
}
