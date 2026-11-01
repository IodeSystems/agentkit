package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// fakeMessages answers /v1/messages with events, and keeps the request.
func fakeMessages(t *testing.T, events ...string) (*Client, *map[string]any) {
	t.Helper()
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			var head struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal([]byte(e), &head)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", head.Type, e)
		}
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	return c, &got
}

const (
	evStart = `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"x","usage":{"input_tokens":10,"cache_read_input_tokens":900,"cache_creation_input_tokens":90,"output_tokens":1}}}`
	evStop  = `{"type":"message_stop"}`
)

func msgChunks(t *testing.T, c *Client, msgs []Message, opts *ChatOpts) []StreamChunk {
	t.Helper()
	ch, err := c.ChatStream(context.Background(), msgs, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	var out []StreamChunk
	for chunk := range ch {
		out = append(out, chunk)
	}
	return out
}

func TestMessagesAPI_TextAndUsage(t *testing.T) {
	var saw Usage
	c, got := fakeMessages(t, evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`, evStop)
	c.OnUsage = func(u Usage) { saw = u }
	var text string
	var u *Usage
	for _, chunk := range msgChunks(t, c, []Message{{Role: "system", Content: "S"}, {Role: "user", Content: "hi"}}, &ChatOpts{ReasoningEffort: "none"}) {
		text += chunk.Content
		if chunk.Usage != nil {
			u = chunk.Usage
		}
	}
	if text != "hello" {
		t.Errorf("text %q", text)
	}
	if u == nil || u.PromptTokens != 1000 || u.CompletionTokens != 7 || u.CachedPromptTokens() != 900 || u.CacheCreationInputTokens != 90 {
		t.Fatalf("usage %+v", u)
	}
	if saw.PromptTokens != 1000 {
		t.Errorf("OnUsage %+v", saw)
	}
	req := *got
	if req["stream"] != true || req["cache_control"] == nil || req["max_tokens"].(float64) != anthropicMaxTokens {
		t.Errorf("request %v", req)
	}
	if req["output_config"].(map[string]any)["effort"] != "low" {
		t.Errorf("off is the lowest effort: %v", req["output_config"])
	}
	if sys := req["system"].([]any)[0].(map[string]any); sys["text"] != "S" || sys["cache_control"] == nil {
		t.Errorf("system %v", sys)
	}
	for _, field := range []string{"temperature", "tools", "thinking"} {
		if _, ok := req[field]; ok {
			t.Errorf("%s was sent", field)
		}
	}
}

func TestMessagesAPI_ToolCalls(t *testing.T) {
	c, _ := fakeMessages(t, evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"exec","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t2","name":"noargs","input":{}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`, evStop)
	var calls []ToolCall
	for _, chunk := range msgChunks(t, c, []Message{{Role: "user", Content: "go"}}, nil) {
		if chunk.ToolCall != nil {
			calls = append(calls, *chunk.ToolCall)
		}
	}
	if len(calls) != 2 || calls[0].ID != "t1" || calls[0].Function.Name != "exec" || calls[0].Function.Arguments != `{"command":"ls"}` {
		t.Fatalf("calls %+v", calls)
	}
	if calls[1].Function.Arguments != "{}" {
		t.Errorf("no arguments: %q", calls[1].Function.Arguments)
	}
}

// max_tokens is finish_reason length, which a caller recovers from.
func TestMessagesAPI_MaxTokens(t *testing.T) {
	c, _ := fakeMessages(t, evStart,
		`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":7}}`, evStop)
	var stop StopReason
	for _, chunk := range msgChunks(t, c, []Message{{Role: "user", Content: "go"}}, nil) {
		if chunk.StopReason != "" {
			stop = chunk.StopReason
		}
	}
	if stop != StopReasonLength {
		t.Errorf("stop %q", stop)
	}
}

func TestMessagesAPI_Refusal(t *testing.T) {
	c, _ := fakeMessages(t, evStart,
		`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"cyber","explanation":"no"}},"usage":{"output_tokens":1}}`, evStop)
	var msg string
	for _, chunk := range msgChunks(t, c, []Message{{Role: "user", Content: "go"}}, nil) {
		msg += chunk.Error
	}
	if !strings.Contains(msg, "refused") || !strings.Contains(msg, "cyber") {
		t.Errorf("error %q", msg)
	}
}

// The loop watcher sees this stream as it sees any other.
func TestMessagesAPI_LoopIsCut(t *testing.T) {
	block := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the parcel runs along the east line of lot seven.\n"}}`
	events := []string{evStart}
	for range 60 {
		events = append(events, block)
	}
	c, _ := fakeMessages(t, events...)
	var stop StopReason
	for _, chunk := range msgChunks(t, c, []Message{{Role: "user", Content: "go"}}, nil) {
		if chunk.Done {
			stop = chunk.StopReason
		}
	}
	if stop != StopReasonRepetition {
		t.Errorf("stop %q", stop)
	}
}

func TestMessagesAPI_SchemaAndChat(t *testing.T) {
	c, got := fakeMessages(t, evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop)
	var td ToolDef
	td.Function.Name, td.Function.Description = "exec", "run"
	td.Function.Parameters = map[string]any{"type": "object", "properties": map[string]any{"command": map[string]any{"type": "string"}}, "required": []string{"command"}}
	text, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, []ToolDef{td}, nil)
	if err != nil || text != "ok" {
		t.Fatalf("%q %v", text, err)
	}
	tool := (*got)["tools"].([]any)[0].(map[string]any)
	schema := tool["input_schema"].(map[string]any)
	if tool["name"] != "exec" || schema["properties"] == nil || len(schema["required"].([]any)) != 1 {
		t.Errorf("tool %v", tool)
	}
}

func roles(m []anthropic.MessageParam) string {
	var out []string
	for _, x := range m {
		out = append(out, string(x.Role)[:1])
	}
	return strings.Join(out, "")
}

func msgCall(id string) ToolCall {
	tc := ToolCall{ID: id, Type: "function"}
	tc.Function.Name, tc.Function.Arguments = "exec", `{"command":"ls"}`
	return tc
}

// A transcript dun folded or interrupted still makes a valid request.
func TestAnthropicMessages_Order(t *testing.T) {
	for name, tc := range map[string]struct {
		in   []Message
		want string
	}{
		"plain":         {[]Message{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}, {Role: "user", Content: "c"}}, "uau"},
		"two users":     {[]Message{{Role: "user", Content: "a"}, {Role: "user", Content: "b"}}, "u"},
		"starts fold":   {[]Message{{Role: "assistant", Content: "a"}, {Role: "user", Content: "b"}}, "uau"},
		"ends assist":   {[]Message{{Role: "user", Content: "a"}, {Role: "assistant", Content: "b"}}, "uau"},
		"notice inside": {[]Message{{Role: "user", Content: "a"}, {Role: "assistant", ToolCalls: []ToolCall{msgCall("1"), msgCall("2")}}, {Role: "tool", ToolCallID: "1", Content: "x"}, {Role: "user", Content: "[note] n"}, {Role: "tool", ToolCallID: "2", Content: "y"}}, "uau"},
		"no result":     {[]Message{{Role: "user", Content: "a"}, {Role: "assistant", ToolCalls: []ToolCall{msgCall("1")}}, {Role: "user", Content: "b"}}, "uau"},
		"no call":       {[]Message{{Role: "tool", ToolCallID: "gone", Content: "x"}, {Role: "user", Content: "b"}}, "u"},
		"empty reply":   {[]Message{{Role: "user", Content: "a"}, {Role: "assistant"}, {Role: "user", Content: "b"}}, "u"},
	} {
		if got := roles(anthropicMessages(tc.in)); got != tc.want {
			t.Errorf("%s: %s, want %s", name, got, tc.want)
		}
	}
	// Results lead the user message, whatever came between them.
	m := anthropicMessages([]Message{{Role: "user", Content: "a"}, {Role: "assistant", ToolCalls: []ToolCall{msgCall("1"), msgCall("2")}},
		{Role: "tool", ToolCallID: "1", Content: "x"}, {Role: "user", Content: "[note] n"}, {Role: "tool", ToolCallID: "2", Content: "y"}})
	blocks := m[2].Content
	if blocks[0].OfToolResult == nil || blocks[1].OfToolResult == nil || blocks[2].OfText == nil {
		t.Errorf("order %+v", blocks)
	}
	// A call with no result gets one.
	m = anthropicMessages([]Message{{Role: "user", Content: "a"}, {Role: "assistant", ToolCalls: []ToolCall{msgCall("1")}}, {Role: "user", Content: "b"}})
	if r := m[2].Content[0].OfToolResult; r == nil || r.ToolUseID != "1" || !r.IsError.Value {
		t.Errorf("missing result %+v", m[2].Content[0])
	}
}

func TestAnthropicMessages_Image(t *testing.T) {
	m := anthropicMessages([]Message{{Role: "user", Parts: []ContentPart{TextPart("see"), {Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64,QUJD"}}}}})
	img := m[0].Content[1].OfImage
	if img == nil || img.Source.OfBase64 == nil || img.Source.OfBase64.Data != "QUJD" || string(img.Source.OfBase64.MediaType) != "image/png" {
		t.Errorf("image %+v", m[0].Content[1])
	}
}

func TestAnthropicEffort(t *testing.T) {
	if e := anthropicEffort("claude-haiku-4-5-20251001", "high"); e != "" {
		t.Errorf("an old model got %q", e)
	}
	if e := anthropicEffort("claude-opus-5-5", "high"); e != "high" {
		t.Errorf("effort %q", e)
	}
	if e := anthropicEffort("claude-opus-5-5", ""); e != "" {
		t.Errorf("default %q", e)
	}
}

// A forced ToolChoice is forced where the model takes it, and on the models that
// 400 on it, auto narrowed to the named tool rather than a failed request.
func TestMessagesAPI_ForcedToolChoice(t *testing.T) {
	defs := func(names ...string) []ToolDef {
		var out []ToolDef
		for _, n := range names {
			var td ToolDef
			td.Function.Name = n
			td.Function.Parameters = map[string]any{"type": "object", "properties": map[string]any{}}
			out = append(out, td)
		}
		return out
	}
	named := `{"type":"function","function":{"name":"answer"}}`
	for _, tc := range []struct {
		model, choice string
		want          string // tool_choice as JSON, "" for not sent
		tools         string
	}{
		{"claude-haiku-5-5", "required", `{"type":"any"}`, "exec,answer"},
		{"claude-haiku-5-5", named, `{"name":"answer","type":"tool"}`, "exec,answer"},
		{"claude-haiku-5-5", "none", `{"type":"none"}`, "exec,answer"},
		{"claude-haiku-5-5", "auto", "", "exec,answer"},
		{"claude-haiku-5-5", "", "", "exec,answer"},
		{"claude-haiku-5-5", `{"type":"function"}`, "", "exec,answer"},
		{"claude-opus-5-5", named, `{"type":"auto"}`, "answer"},
		{"claude-sonnet-5-5", named, `{"type":"auto"}`, "answer"},
		{"claude-fable-5-1", named, `{"type":"auto"}`, "answer"},
		{"claude-mythos-5-1", named, `{"type":"auto"}`, "answer"},
		{"claude-opus-5-5", "required", `{"type":"auto"}`, "exec,answer"},
		{"claude-opus-5-5", `{"type":"function","function":{"name":"gone"}}`, `{"type":"auto"}`, "exec,answer"},
		{"claude-opus-5-5", "none", `{"type":"none"}`, "exec,answer"},
	} {
		c, got := fakeMessages(t, evStart,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop)
		c.model = tc.model
		if _, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, defs("exec", "answer"), &ChatOpts{ToolChoice: tc.choice}); err != nil {
			t.Fatalf("%s %s: %v", tc.model, tc.choice, err)
		}
		req := *got
		var choice string
		if v, ok := req["tool_choice"]; ok {
			b, _ := json.Marshal(v)
			choice = string(b)
		}
		var names []string
		for _, x := range req["tools"].([]any) {
			names = append(names, x.(map[string]any)["name"].(string))
		}
		if choice != tc.want || strings.Join(names, ",") != tc.tools {
			t.Errorf("%s %s: tool_choice %s tools %v, want %s %s", tc.model, tc.choice, choice, names, tc.want, tc.tools)
		}
	}
}
