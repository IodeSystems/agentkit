package mcpmgr

import "testing"

// A nil Required (an MCP tool with no required fields) must NOT marshal to
// `"required": null` — that's invalid JSON Schema and llama.cpp rejects the
// whole chat request ("type must be array, but is null"). normalizeSchema
// defaults nil sub-fields to their empty forms.
func TestNormalizeSchema(t *testing.T) {
	// no required, no properties → empty [] / {}, type defaulted.
	s := normalizeSchema("", nil, nil)
	if s["type"] != "object" {
		t.Errorf("type = %v; want object", s["type"])
	}
	req, ok := s["required"].([]string)
	if !ok || req == nil {
		t.Errorf("required = %#v; want non-nil []string", s["required"])
	}
	if len(req) != 0 {
		t.Errorf("required = %v; want empty", req)
	}
	props, ok := s["properties"].(map[string]any)
	if !ok || props == nil {
		t.Errorf("properties = %#v; want non-nil map", s["properties"])
	}

	// populated schema is passed through unchanged.
	p := map[string]any{"city": map[string]any{"type": "string"}}
	s = normalizeSchema("object", p, []string{"city"})
	if got := s["required"].([]string); len(got) != 1 || got[0] != "city" {
		t.Errorf("required = %v; want [city]", got)
	}
	if _, ok := s["properties"].(map[string]any)["city"]; !ok {
		t.Errorf("properties lost the city field")
	}
}
