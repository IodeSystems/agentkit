package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// Validation — the schema fix loop.
//
// Structured-output roles need their tool-call arguments to actually match
// the tool's schema. tool_choice=required forces the model to CALL a tool,
// but not to fill it correctly. Validator closes that gap: a bad call is
// rejected with a fix instruction as its tool result, the session stays
// active, and the model retries — the same hand-rolled rejection pattern
// hosts already write per-tool, made generic.
//
// ValidatingDispatcher wraps any ToolDispatcher with this check, so it drops
// into the existing loop with no Turn surgery.

// Validator gates a tool call before dispatch. A nil error means accept; a
// non-nil error's message is fed back to the model as the tool result (the
// fix instruction).
type Validator interface {
	ValidateArgs(toolName, argsJSON string) error
}

// ValidatorFunc adapts a function to Validator.
type ValidatorFunc func(toolName, argsJSON string) error

func (f ValidatorFunc) ValidateArgs(toolName, argsJSON string) error { return f(toolName, argsJSON) }

// ValidatingDispatcher returns a ToolDispatcher that validates arguments
// before delegating to inner. On validation failure it returns the fix
// instruction as the (non-error) result — keeping the session active so the
// model can retry — and never calls inner. A nil validator is a pass-through.
func ValidatingDispatcher(inner ToolDispatcher, v Validator) ToolDispatcher {
	if v == nil {
		return inner
	}
	return func(ctx context.Context, tc llm.ToolCall) (string, error) {
		if err := v.ValidateArgs(tc.Function.Name, tc.Function.Arguments); err != nil {
			return fmt.Sprintf(
				"INVALID arguments for %s: %v. Fix the arguments and call %s again.",
				tc.Function.Name, err, tc.Function.Name), nil
		}
		return inner(ctx, tc)
	}
}

// SchemaValidator is the lightweight default Validator, built from the same
// tool defs the session advertises. It is deliberately dependency-free and
// conservative — it catches the common structured-output failures without a
// full JSON-Schema engine:
//
//   - arguments must be a JSON object
//   - every `required` property named in the tool's schema must be present
//     and non-null
//   - a present property whose schema declares a primitive `type`
//     (string/number/integer/boolean/array/object) must not be the wrong
//     JSON kind
//
// Unknown tools pass (the host may dispatch tools it didn't declare). For
// stricter guarantees a consumer plugs its own Validator (e.g. a real
// JSON-Schema library) — the interface is the seam.
type SchemaValidator struct {
	schemas map[string]toolSchema
}

type toolSchema struct {
	required []string
	types    map[string]string // property name → declared JSON type (if any)
}

// NewSchemaValidator indexes the required-keys + declared types from each
// tool def's parameters schema. Tool defs whose parameters aren't a
// standard object schema simply contribute no constraints.
func NewSchemaValidator(tools []llm.ToolDef) *SchemaValidator {
	sv := &SchemaValidator{schemas: make(map[string]toolSchema, len(tools))}
	for _, t := range tools {
		sv.schemas[t.Function.Name] = parseToolSchema(t.Function.Parameters)
	}
	return sv
}

func parseToolSchema(params any) toolSchema {
	ts := toolSchema{types: map[string]string{}}
	// params is typically map[string]any from JSON, or a Go value the caller
	// built. Normalize via a JSON round-trip so both shapes work.
	raw, err := json.Marshal(params)
	if err != nil {
		return ts
	}
	var schema struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			Type string `json:"type"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		return ts
	}
	ts.required = schema.Required
	for name, p := range schema.Properties {
		if p.Type != "" {
			ts.types[name] = p.Type
		}
	}
	return ts
}

func (sv *SchemaValidator) ValidateArgs(toolName, argsJSON string) error {
	ts, known := sv.schemas[toolName]
	if !known {
		return nil
	}
	args := strings.TrimSpace(argsJSON)
	if args == "" {
		args = "{}"
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(args), &obj); err != nil {
		return fmt.Errorf("arguments are not a JSON object: %v", err)
	}
	var missing []string
	for _, r := range ts.required {
		v, ok := obj[r]
		if !ok || string(v) == "null" {
			missing = append(missing, r)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("missing required field(s): %s", strings.Join(missing, ", "))
	}
	var typeErrs []string
	for name, want := range ts.types {
		v, ok := obj[name]
		if !ok || string(v) == "null" {
			continue // absent optional field — fine
		}
		if got := jsonKind(v); got != "" && !typeMatches(want, got) {
			typeErrs = append(typeErrs, fmt.Sprintf("%s should be %s, got %s", name, want, got))
		}
	}
	if len(typeErrs) > 0 {
		sort.Strings(typeErrs)
		return fmt.Errorf("%s", strings.Join(typeErrs, "; "))
	}
	return nil
}

// jsonKind returns the JSON-Schema kind of a raw value, or "" if unknowable.
func jsonKind(v json.RawMessage) string {
	s := strings.TrimSpace(string(v))
	if s == "" {
		return ""
	}
	switch s[0] {
	case '"':
		return "string"
	case '{':
		return "object"
	case '[':
		return "array"
	case 't', 'f':
		return "boolean"
	case 'n':
		return "" // null handled by the caller
	default:
		return "number"
	}
}

func typeMatches(want, got string) bool {
	switch want {
	case "integer", "number":
		return got == "number"
	default:
		return want == got
	}
}
