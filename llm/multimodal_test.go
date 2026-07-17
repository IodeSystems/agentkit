package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// A plain text message must marshal to the bare-string content shape — byte
// identical to before Parts existed. This is the invariant that keeps every
// existing caller unaffected.
func TestMessage_StringContentUnchanged(t *testing.T) {
	b, err := json.Marshal(Message{Role: "user", Content: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(b); got != `{"role":"user","content":"hello"}` {
		t.Fatalf("string path changed: %s", got)
	}
}

// With Parts set, content becomes the OpenAI multimodal array and the Content
// string is dropped.
func TestMessage_MultimodalArray(t *testing.T) {
	m := Message{
		Role:    "user",
		Content: "ignored when Parts set",
		Parts: []ContentPart{
			TextPart("what does this page say?"),
			ImageData("image/png", []byte{0x89, 0x50, 0x4e, 0x47}), // "‰PNG"
		},
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)

	// content must be an array, not the string.
	if strings.Contains(got, `"ignored when Parts set"`) {
		t.Fatalf("Content string leaked into multimodal message: %s", got)
	}
	if !strings.Contains(got, `"type":"text","text":"what does this page say?"`) {
		t.Errorf("missing text part: %s", got)
	}
	if !strings.Contains(got, `"type":"image_url"`) || !strings.Contains(got, `"url":"data:image/png;base64,iVBORw=="`) {
		t.Errorf("missing/badly-encoded image part: %s", got)
	}

	// Round-trip through a generic decode to prove content is a JSON array.
	var probe struct {
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &probe); err != nil {
		t.Fatal(err)
	}
	if len(probe.Content) == 0 || probe.Content[0] != '[' {
		t.Fatalf("content is not a JSON array: %s", probe.Content)
	}
}

// ImageData encodes raw bytes as a base64 data URI with the given MIME type.
func TestImageData_DataURI(t *testing.T) {
	p := ImageData("image/jpeg", []byte("hi"))
	if p.Type != "image_url" || p.ImageURL == nil {
		t.Fatalf("wrong part: %+v", p)
	}
	if p.ImageURL.URL != "data:image/jpeg;base64,aGk=" {
		t.Errorf("data URI = %q", p.ImageURL.URL)
	}
}

// Tool messages and assistant tool_calls still marshal correctly alongside the
// new field (Parts empty → untouched).
func TestMessage_ToolFieldsUnaffected(t *testing.T) {
	b, _ := json.Marshal(Message{Role: "tool", Content: "42", ToolCallID: "call_1"})
	if got := string(b); got != `{"role":"tool","content":"42","tool_call_id":"call_1"}` {
		t.Fatalf("tool message shape changed: %s", got)
	}
}
