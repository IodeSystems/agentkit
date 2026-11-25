package agent

import (
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func imgPart(url string) llm.ContentPart {
	return llm.ContentPart{Type: "image_url", ImageURL: &llm.ImageURL{URL: url}}
}

// A user entry carrying Parts must render as a MULTIMODAL message so the
// provider receives an OpenAI content array. Without this the parts are
// unreachable: llm.Message has supported Parts for a while, but nothing could
// get them through a session, so a vision probe silently degraded to text.
func TestRenderEntry_UserPartsBecomeMultimodal(t *testing.T) {
	e := Entry{
		Kind:    KindUser,
		Content: "What shape is this?",
		Parts: []llm.ContentPart{
			{Type: "text", Text: "What shape is this?"},
			imgPart("data:image/png;base64,iVBORw=="),
		},
	}
	m := renderEntry(e, e.Content, false, ToolFormatNative)
	if len(m.Parts) != 2 {
		t.Fatalf("Parts not carried through: got %d", len(m.Parts))
	}
	if m.Content != "" {
		t.Errorf("Content must not be set alongside Parts (it would be sent instead): %q", m.Content)
	}
	if m.Role != "user" {
		t.Errorf("role = %q, want user", m.Role)
	}
}

// The overwhelming majority of entries have no Parts and must render exactly as
// before — this change is additive or it is a regression for every existing host.
func TestRenderEntry_NoPartsUnchanged(t *testing.T) {
	e := Entry{Kind: KindUser, Content: "plain text"}
	m := renderEntry(e, e.Content, false, ToolFormatNative)
	if len(m.Parts) != 0 {
		t.Errorf("unexpected Parts on a text entry: %+v", m.Parts)
	}
	if m.Content != "plain text" {
		t.Errorf("Content = %q", m.Content)
	}
}

// When the shaper substitutes a stub (LOD truncation) or a compaction summary,
// `content` differs from e.Content. Re-attaching the image there would defeat
// the very truncation that decided to drop it — and would send a full-size
// image inside a window the shaper is trying to shrink.
func TestRenderEntry_SubstitutedContentDropsParts(t *testing.T) {
	e := Entry{
		Kind:    KindUser,
		Content: "What shape is this?",
		Parts:   []llm.ContentPart{imgPart("data:image/png;base64,iVBORw==")},
	}
	m := renderEntry(e, "[truncated]", false, ToolFormatNative)
	if len(m.Parts) != 0 {
		t.Errorf("parts survived an LOD stub — truncation defeated: %+v", m.Parts)
	}
	if m.Content != "[truncated]" {
		t.Errorf("Content = %q, want the stub", m.Content)
	}
}

// Assistant turns are text: an assistant entry with Parts must ignore them
// rather than emit a message the provider rejects. (Tool results with Parts
// render multimodally: toolresult_parts_test.go.)
func TestRenderEntry_AssistantIgnoresParts(t *testing.T) {
	e := Entry{
		Kind:    KindAssistant,
		Content: "said hi",
		Parts:   []llm.ContentPart{imgPart("data:image/png;base64,iVBORw==")},
	}
	m := renderEntry(e, e.Content, false, ToolFormatNative)
	if len(m.Parts) != 0 {
		t.Errorf("assistant rendered multimodally: %+v", m.Parts)
	}
	if m.Role != "assistant" || m.Content != "said hi" {
		t.Errorf("assistant mangled: %+v", m)
	}
}
