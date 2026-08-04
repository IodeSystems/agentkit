package mcpmgr

import (
	"encoding/json"
	"strings"
	"testing"
)

// A nil Required (an MCP tool with no required fields) must NOT marshal to
// `"required": null` — that's invalid JSON Schema and llama.cpp rejects the
// whole chat request ("type must be array, but is null"). normalizeSchema
// defaults nil sub-fields to their empty forms.
func TestNormalizeSchemaFillsNils(t *testing.T) {
	s := normalizeSchema(nil)
	if s["type"] != "object" {
		t.Errorf("type = %v; want object", s["type"])
	}
	if s["properties"] == nil {
		t.Error("properties is nil; want an empty object")
	}
	if s["required"] == nil {
		t.Error("required is nil; want an empty array")
	}

	// The whole point is what it marshals to.
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "null") {
		t.Errorf("marshalled to %s; want no nulls", b)
	}
}

// normalizeSchema normalizes, it does not reconstruct. Every keyword it does
// not know about has to come out the other side untouched — the reason this
// package stopped decoding inputSchema into a fixed struct in the first place.
func TestNormalizeSchemaPreservesUnknownKeywords(t *testing.T) {
	in := map[string]any{
		"type":                 "object",
		"$schema":              "https://json-schema.org/draft/2020-12/schema",
		"additionalProperties": false,
		"$defs":                map[string]any{"id": map[string]any{"type": "string"}},
		"properties": map[string]any{
			"who": map[string]any{"$ref": "#/$defs/id", "description": "the subject"},
		},
		"required": []any{"who"},
	}
	s := normalizeSchema(in)

	for _, k := range []string{"$schema", "additionalProperties", "$defs"} {
		if _, ok := s[k]; !ok {
			t.Errorf("%s was dropped", k)
		}
	}
	if s["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v; want false", s["additionalProperties"])
	}
	props := s["properties"].(map[string]any)
	who := props["who"].(map[string]any)
	if who["$ref"] != "#/$defs/id" || who["description"] != "the subject" {
		t.Errorf("properties.who = %v; want it carried through verbatim", who)
	}
	if req := s["required"].([]any); len(req) != 1 || req[0] != "who" {
		t.Errorf("required = %v; want [who]", s["required"])
	}
}

// The input map belongs to the caller (it came off the wire); normalizing must
// not scribble defaults into it.
func TestNormalizeSchemaDoesNotMutateInput(t *testing.T) {
	in := map[string]any{"type": "object"}
	normalizeSchema(in)
	if _, ok := in["required"]; ok {
		t.Error("normalizeSchema wrote back into its argument")
	}
}
