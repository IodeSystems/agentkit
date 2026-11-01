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
	if req["stream"] != true || req["max_tokens"].(float64) != 64000 {
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

// Not streaming, with the key and version a Messages endpoint asks for.
func TestMessagesAPI_Chat(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	text, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err != nil || text != "ok" {
		t.Fatalf("%q %v", text, err)
	}
	if got.Get("x-api-key") != "k" || got.Get("anthropic-version") == "" || got.Get("Authorization") != "" {
		t.Errorf("headers %v", got)
	}
}

// A 429 goes through the client's own retries, as on any other provider.
func TestMessagesAPI_Retries(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	if text, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil || text != "ok" || calls != 2 {
		t.Errorf("%q %v after %d calls", text, err, calls)
	}
}

// OpenRouter is asked to cache Claude, and nothing else is.
func TestOpenRouterCachesClaude(t *testing.T) {
	for _, tc := range []struct {
		url, model string
		want       bool
	}{
		{"https://openrouter.ai/api/v1", "anthropic/claude-opus-5-5", true},
		{"https://openrouter.ai/api/v1", "deepseek/x", false},
		{"http://localhost:8080/v1", "claude-opus-5-5", false},
	} {
		body := map[string]any{}
		NewClient(tc.url, "k", tc.model).applySampling(body, nil)
		if _, got := body["cache_control"]; got != tc.want {
			t.Errorf("%s %s: %v", tc.url, tc.model, got)
		}
	}
}
