package llm

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func toolCall(id string) ToolCall {
	var tc ToolCall
	tc.ID, tc.Type = id, "function"
	tc.Function.Name, tc.Function.Arguments = "shot", `{}`
	return tc
}

// chatWire is the "messages" array a chat-format client sends, decoded.
func chatWire(t *testing.T, c *Client, msgs []Message) []map[string]any {
	t.Helper()
	body, _ := c.chatBody(context.Background(), msgs, nil, nil, false)
	b, err := json.Marshal(body["messages"])
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

var png = ImageData("image/png", []byte{0x89, 0x50, 0x4e, 0x47})

// A tool message on the chat format is text only: an image left inside it is
// refused or flattened by the server. It must move to one user message after the
// run, with Carl's exact label in its place.
func TestToolMedia_SingleImageMovesToUserMessageAfterRun(t *testing.T) {
	c := NewClient("http://x/v1", "k", "m")
	got := chatWire(t, c, []Message{
		{Role: "user", Content: "look"},
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1")}},
		{Role: "tool", ToolCallID: "c1", Parts: []ContentPart{TextPart("took it"), png}},
		{Role: "user", Content: "and?"},
	})
	if len(got) != 5 {
		t.Fatalf("want 5 messages, got %d: %v", len(got), got)
	}
	if got[2]["content"] != "took it\n\nimage/png is attached in the next user message" {
		t.Fatalf("tool content %q", got[2]["content"])
	}
	if got[2]["tool_call_id"] != "c1" {
		t.Fatalf("tool_call_id lost: %v", got[2])
	}
	u := got[3]
	parts, _ := u["content"].([]any)
	if u["role"] != "user" || len(parts) != 1 || parts[0].(map[string]any)["type"] != "image_url" {
		t.Fatalf("want one user message holding the image alone, got %v", u)
	}
	if got[4]["content"] != "and?" {
		t.Fatalf("the caller's next message must follow the carried media: %v", got[4])
	}
}

// Parallel calls answer as one run; nothing may sit between two of its tool
// messages, so all media goes in ONE user message after the last, numbered.
func TestToolMedia_SeveralImagesAcrossParallelCallsAreNumbered(t *testing.T) {
	c := NewClient("http://x/v1", "k", "m")
	jpg := ImageData("image/jpeg", []byte{1})
	got := chatWire(t, c, []Message{
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("a"), toolCall("b")}},
		{Role: "tool", ToolCallID: "a", Parts: []ContentPart{png}},
		{Role: "tool", ToolCallID: "b", Parts: []ContentPart{TextPart("two"), jpg, png}},
	})
	if len(got) != 4 || got[1]["role"] != "tool" || got[2]["role"] != "tool" || got[3]["role"] != "user" {
		t.Fatalf("want assistant, tool, tool, user; got %v", got)
	}
	if got[1]["content"] != "image 1: image/png is attached in the next user message" {
		t.Fatalf("first tool %q", got[1]["content"])
	}
	want := "two\n\nimage 2: image/jpeg is attached in the next user message\n\n" +
		"image 3: image/png is attached in the next user message"
	if got[2]["content"] != want {
		t.Fatalf("second tool %q", got[2]["content"])
	}
	parts := got[3]["content"].([]any)
	var seq []string
	for _, p := range parts {
		m := p.(map[string]any)
		if m["type"] == "text" {
			seq = append(seq, m["text"].(string))
		} else {
			seq = append(seq, m["type"].(string))
		}
	}
	if strings.Join(seq, ",") != "image 1:,image_url,image 2:,image_url,image 3:,image_url" {
		t.Fatalf("user parts %v", seq)
	}
}

// The Messages API takes audio nowhere and the shim refuses it, which would fail
// the request over one tool's output. It is omitted, and a label says so.
func TestToolMedia_AudioOnMessagesAPIIsOmittedWithLabel(t *testing.T) {
	c, got := fakeMessages(t, evStart, evStop)
	msgChunks(t, c, []Message{
		{Role: "user", Content: "listen"},
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1")}},
		{Role: "tool", ToolCallID: "c1", Parts: []ContentPart{AudioData("wav", []byte{1, 2})}},
	}, nil)
	b, _ := json.Marshal((*got)["messages"])
	s := string(b)
	if !strings.Contains(s, `[audio/wav not sent: this endpoint does not accept audio in tool results]`) {
		t.Fatalf("no omission label on the wire: %s", s)
	}
	if strings.Contains(s, "input_audio") || strings.Contains(s, "AQI=") {
		t.Fatalf("audio reached the wire: %s", s)
	}
}

// tool_result content takes document blocks: a PDF stays inside it.
func TestToolMedia_PDFOnMessagesAPIIsNativeInToolResult(t *testing.T) {
	c, got := fakeMessages(t, evStart, evStop)
	msgChunks(t, c, []Message{
		{Role: "user", Content: "read"},
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1")}},
		{Role: "tool", ToolCallID: "c1", Content: "here", Parts: []ContentPart{PDFData("a.pdf", []byte("%PDF"))}},
	}, nil)
	msgs := (*got)["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)
	tr := last["content"].([]any)[0].(map[string]any)
	if tr["type"] != "tool_result" {
		t.Fatalf("want tool_result, got %v", tr)
	}
	inner := tr["content"].([]any)
	if len(inner) != 2 || inner[0].(map[string]any)["text"] != "here" || inner[1].(map[string]any)["type"] != "document" {
		t.Fatalf("want Content as text then the document inside the tool_result: %v", inner)
	}
	if len(msgs) != 3 {
		t.Fatalf("no extra user turn on the Messages API: %v", msgs)
	}
}

// A model the record knows takes no images gets a label, on either format.
// Unknown (no fact) sends.
func TestToolMedia_CapabilityGateOmitsKnownUnsupported(t *testing.T) {
	c := NewClient("http://x/v1", "k", "m")
	msgs := []Message{
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1")}},
		{Role: "tool", ToolCallID: "c1", Parts: []ContentPart{png}},
	}
	if got := chatWire(t, c, msgs); len(got) != 3 {
		t.Fatalf("unknown modality must send: %v", got)
	}
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{
		CapModalityPrefix + "image": {Value: "false", Source: SourceHost}}})
	got := chatWire(t, c, msgs)
	if len(got) != 2 {
		t.Fatalf("no user message for an omitted part: %v", got)
	}
	if got[1]["content"] != "[image/png not sent: this model does not accept image input]" {
		t.Fatalf("label %q", got[1]["content"])
	}
}

func TestToolMedia_CallerSliceUnchanged(t *testing.T) {
	msgs := []Message{
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1")}},
		{Role: "tool", ToolCallID: "c1", Content: "x", Parts: []ContentPart{TextPart("t"), png}},
	}
	before := make([]Message, len(msgs))
	copy(before, msgs)
	before[1].Parts = append([]ContentPart(nil), msgs[1].Parts...)
	_ = routeToolMedia(msgs, wireChat, Capabilities{})
	_ = routeToolMedia(msgs, wireMessages, Capabilities{})
	if !reflect.DeepEqual(msgs, before) {
		t.Fatalf("caller's slice mutated: %+v", msgs)
	}
	plain := []Message{{Role: "user", Content: "hi"}, {Role: "tool", Content: "r"}}
	if out := routeToolMedia(plain, wireChat, Capabilities{}); &out[0] != &plain[0] {
		t.Fatal("a list with no tool media must pass through as is")
	}
}

// Prompt caching needs round N's messages to render byte-identically as the
// head of round N+1.
func TestToolMedia_RoundNIsPrefixOfRoundNPlus1(t *testing.T) {
	c := NewClient("http://x/v1", "k", "m")
	round1 := []Message{
		{Role: "user", Content: "look"},
		{Role: "assistant", ToolCalls: []ToolCall{toolCall("a"), toolCall("b")}},
		{Role: "tool", ToolCallID: "a", Parts: []ContentPart{png}},
		{Role: "tool", ToolCallID: "b", Parts: []ContentPart{png}},
	}
	round2 := append(append([]Message(nil), round1...),
		Message{Role: "assistant", ToolCalls: []ToolCall{toolCall("c")}},
		Message{Role: "tool", ToolCallID: "c", Parts: []ContentPart{png}},
	)
	w1, w2 := chatWire(t, c, round1), chatWire(t, c, round2)
	b1, _ := json.Marshal(w1)
	b2, _ := json.Marshal(w2[:len(w1)])
	if string(b1) != string(b2) {
		t.Fatalf("round 1 is not a prefix of round 2:\n%s\n%s", b1, b2)
	}
	if w2[len(w2)-2]["content"] != "image/png is attached in the next user message" {
		t.Fatalf("a new run numbers on its own: %v", w2[len(w2)-2])
	}
}

func TestToolMedia_PartMime(t *testing.T) {
	for _, tc := range []struct {
		p    ContentPart
		want string
	}{
		{ImagePart("https://e/x.png"), "image"},
		{VideoData("video/mp4", []byte{1}), "video/mp4"},
		{AudioData("MP3", []byte{1}), "audio/mp3"},
		{ContentPart{Type: "file", File: &FileRef{Filename: "r.pdf", FileID: "f1"}}, "application/pdf"},
		{ContentPart{Type: "file", File: &FileRef{FileID: "f1"}}, "file"},
		{ContentPart{Type: "input_hologram"}, "input_hologram"},
	} {
		if got := partMime(tc.p); got != tc.want {
			t.Errorf("partMime(%s) = %q, want %q", tc.p.Type, got, tc.want)
		}
	}
}

// A kind this package does not know moves on the chat format and is omitted,
// with a label, on the Messages API (whose shim refuses unknown types).
func TestToolMedia_UnknownKindRoutes(t *testing.T) {
	p := ContentPart{Type: "input_hologram"}
	if toolPartRoute(wireChat, p) != routeMove || toolPartRoute(wireMessages, p) != routeOmit {
		t.Fatal("unknown kind routing")
	}
	zip := FileData("application/zip", "a.zip", []byte{1})
	if toolPartRoute(wireMessages, zip) != routeOmit {
		t.Fatal("a file the shim refuses must be omitted, not sent")
	}
}
