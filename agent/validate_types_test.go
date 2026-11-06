package agent

import (
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func toolWith(props map[string]any, required ...string) []llm.ToolDef {
	var d llm.ToolDef
	d.Type = "function"
	d.Function.Name = "t"
	d.Function.Parameters = map[string]any{"type": "object", "properties": props, "required": required}
	return []llm.ToolDef{d}
}

// One property declared as a type list ("integer","null") made the whole schema
// fail to parse, which dropped `required` too: a call missing a required field
// passed.
func TestSchemaValidator_TypeListDoesNotDisableRequired(t *testing.T) {
	sv := NewSchemaValidator(toolWith(map[string]any{
		"n":    map[string]any{"type": []any{"integer", "null"}},
		"name": map[string]any{"type": "string"},
	}, "name"))
	if err := sv.ValidateArgs("t", `{"n":3}`); err == nil || !strings.Contains(err.Error(), "name") {
		t.Fatalf("missing required not reported: %v", err)
	}
	if err := sv.ValidateArgs("t", `{"name":"x","n":null}`); err != nil {
		t.Fatalf("null for [integer,null] rejected: %v", err)
	}
	if err := sv.ValidateArgs("t", `{"name":"x","n":"3"}`); err == nil {
		t.Fatal("string accepted for [integer,null]")
	}
}

// 1.5 is a number, not an integer.
func TestSchemaValidator_IntegerRejectsFraction(t *testing.T) {
	sv := NewSchemaValidator(toolWith(map[string]any{"n": map[string]any{"type": "integer"}}))
	if err := sv.ValidateArgs("t", `{"n":1.5}`); err == nil {
		t.Fatal("1.5 accepted as integer")
	}
	for _, ok := range []string{`{"n":2}`, `{"n":2.0}`, `{"n":-7}`, `{"n":1e3}`} {
		if err := sv.ValidateArgs("t", ok); err != nil {
			t.Errorf("%s rejected: %v", ok, err)
		}
	}
}
