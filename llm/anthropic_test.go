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

	shim "github.com/iodesystems/openai-claude-api-shim"
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
	evStart = `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5","usage":{"input_tokens":10,"cache_read_input_tokens":900,"cache_creation_input_tokens":90,"output_tokens":1}}}`
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
	for _, chunk := range msgChunks(t, c, []Message{{Role: "system", Content: "S"}, {Role: "user", Content: "hi"}}, &ChatOpts{Effort: "none"}) {
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
	// 10 new at $4, 900 read at $0.20, 90 written at $5, 7 out at $20, per MTok.
	want := (10*4 + 900*0.20 + 90*5 + 7*20) / 1e6
	if !saw.Cost.Reported || saw.Cost.Currency != CurrencyUSDEstimate || saw.Cost.Amount < want*0.999 || saw.Cost.Amount > want*1.001 {
		t.Errorf("cost %+v, want %v USD", saw.Cost, want)
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

// A strict tool carries strict true to the Claude wire: the OpenAI-shaped
// body's function.strict is forwarded by the shim to the Messages API's
// tool.strict, which a Sonnet 5.5 request needs.
func TestMessagesAPI_StrictToolCarriesStrictTrue(t *testing.T) {
	c, got := fakeMessages(t, evStart,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop)
	strict := true
	var td ToolDef
	td.Type = "function"
	td.Function.Name = "exec"
	td.Function.Description = "run"
	td.Function.Parameters = map[string]any{"type": "object"}
	td.Function.Strict = &strict
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "go"}}, []ToolDef{td}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	toolsArr, ok := (*got)["tools"].([]any)
	if !ok || len(toolsArr) != 1 {
		t.Fatalf("tools %v", (*got)["tools"])
	}
	tool := toolsArr[0].(map[string]any)
	if tool["name"] != "exec" {
		t.Fatalf("tool %v", tool)
	}
	if tool["strict"] != true {
		t.Errorf("strict not carried to the Claude wire: %v", tool)
	}
	if _, present := tool["input_schema"]; !present {
		t.Errorf("input_schema missing: %v", tool)
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
	res, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err != nil || res.Content != "ok" {
		t.Fatalf("%q %v", res.Content, err)
	}
	if got.Get("x-api-key") != "k" || got.Get("anthropic-version") == "" || got.Get("Authorization") != "" {
		t.Errorf("headers %v", got)
	}
}

// Citation metadata the model attached to a document answer surfaces on the
// result as a field separate from the text: the caller reads the reply and the
// citations as distinct things.
func TestMessagesAPI_CitationsSurface(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"the east line runs here.","citations":[{"type":"citation","document_index":1,"cited_text":"east line","start_char_index":10,"end_char_index":20}]}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	res, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "the east line runs here." {
		t.Errorf("content %q", res.Content)
	}
	if len(res.Citations) != 1 {
		t.Fatalf("citations %v", res.Citations)
	}
	got := res.Citations[0]
	if got.DocumentIndex != 1 || got.CitationText != "east line" || got.StartCharIndex != 10 || got.EndCharIndex != 20 {
		t.Errorf("citation %+v", got)
	}
}

// A reply with no citations leaves the field nil, not an empty slice.
func TestMessagesAPI_NoCitationsIsNil(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	res, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Citations != nil {
		t.Errorf("citations %v, want nil", res.Citations)
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
	if res, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil || res.Content != "ok" || calls != 2 {
		t.Errorf("%q %v after %d calls", res.Content, err, calls)
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

// A host's price wins.
func TestMessagesAPI_ClaudePrices(t *testing.T) {
	c, _ := fakeMessages(t, `{"type":"message_start","message":{"id":"m","model":"claude-opus-5-5","usage":{"input_tokens":1000000}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}`, evStop)
	c.ClaudePrices = shim.Prices{"claude-opus-5-5": {Input: 1}}
	var saw Usage
	c.OnUsage = func(u Usage) { saw = u }
	msgChunks(t, c, []Message{{Role: "user", Content: "hi"}}, nil)
	if saw.Cost.Amount != 1 || saw.Cost.Currency != CurrencyUSDEstimate {
		t.Errorf("cost %+v", saw.Cost)
	}
}

// opus-latest goes out as the newest Opus, and the list is asked once.
func TestMessagesAPI_Alias(t *testing.T) {
	lists := 0
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/models") {
			lists++
			fmt.Fprint(w, `{"data":[{"id":"claude-opus-5","line":"opus","created_at":"2026-07-24T00:00:00Z"},{"id":"claude-opus-5-5","line":"opus","created_at":"2026-09-21T00:00:00Z"}]}`)
			return
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &sent)
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "opus-latest")
	c.MessagesAPI = true
	for range 2 {
		if _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if sent["model"] != "claude-opus-5-5" || lists != 1 {
		t.Errorf("sent %v after %d lists", sent["model"], lists)
	}
}

// A model list that cannot be read is asked again after a pause, not on every
// call: each ask is a blocking GET held under the alias lock.
func TestMessagesAPI_FailedAliasResolutionIsNotRetriedEveryCall(t *testing.T) {
	lists := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/models") {
			lists++
			http.Error(w, "down", 500)
			return
		}
		fmt.Fprint(w, `{"id":"m","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "opus-latest")
	c.MessagesAPI = true
	for range 4 {
		if got := c.wireModel(context.Background()); got != "opus-latest" {
			t.Fatalf("wireModel = %q; an unresolved alias goes out as it is", got)
		}
	}
	if lists != 1 {
		t.Errorf("model list asked %d times for 4 calls; want 1 inside the failure window", lists)
	}
}

// A person's words typed while a tool ran arrive as assistant(tool_calls) →
// tool → user. Anthropic wants the text in the SAME user turn as the
// tool_result blocks, after them. If the shim ever stops merging, the host's
// mid-turn interjection becomes a request the API refuses (two user turns in a
// row, or text before the result).
func TestMessagesAPI_UserAfterToolResultSharesTheResultsTurn(t *testing.T) {
	c, got := fakeMessages(t, evStart, evStop)
	var tc ToolCall
	tc.ID, tc.Type = "c1", "function"
	tc.Function.Name, tc.Function.Arguments = "read", `{}`
	msgChunks(t, c, []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []ToolCall{tc}},
		{Role: "tool", ToolCallID: "c1", Content: "result"},
		{Role: "user", Content: "also check b"},
	}, nil)
	msgs, _ := (*got)["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("want user, assistant, user (3 turns); got %d: %v", len(msgs), msgs)
	}
	last := msgs[2].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("last turn role %v", last["role"])
	}
	blocks := last["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("want tool_result + text in one turn, got %v", blocks)
	}
	if blocks[0].(map[string]any)["type"] != "tool_result" || blocks[1].(map[string]any)["type"] != "text" {
		t.Fatalf("tool_result must lead, text follow: %v", blocks)
	}
}

// A trailing assistant message does not reach the wire on a no-prefill model
// (claude-sonnet-5-5). The shim's repair() appends a user "Continue." turn,
// so the last message on the wire is always user.
func TestMessagesAPI_TrailingAssistantDoesNotReachSonnet55(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\nevent: message_stop\ndata: %s\n\n", evStart, evStop)
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL+"/v1", "k", "claude-sonnet-5-5")
	c.MessagesAPI = true
	msgChunks(t, c, []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", Content: "I'll handle that."},
	}, nil)
	msgs, ok := got["messages"].([]any)
	if !ok || len(msgs) == 0 {
		t.Fatalf("messages %v", got["messages"])
	}
	last := msgs[len(msgs)-1].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("last wire message role = %v, want user (trailing assistant was not repaired)", last["role"])
	}
}

// A forced tool_choice (named tool) becomes auto on claude-sonnet-5-5,
// which answers a 400 to a forced choice. The shim's unforced() converts it.
func TestMessagesAPI_ForcedToolChoiceBecomesAutoForSonnet55(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\nevent: message_stop\ndata: %s\n\n", evStart, evStop)
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL+"/v1", "k", "claude-sonnet-5-5")
	c.MessagesAPI = true
	strict := true
	var td ToolDef
	td.Type = "function"
	td.Function.Name = "exec"
	td.Function.Description = "run"
	td.Function.Parameters = map[string]any{"type": "object"}
	td.Function.Strict = &strict
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "go"}},
		[]ToolDef{td}, &ChatOpts{ToolChoice: `{"type":"tool","function":{"name":"exec"}}`})
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	tc, ok := got["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice %v", got["tool_choice"])
	}
	if tc["type"] != "auto" {
		t.Errorf("tool_choice type = %v, want auto (forced choice on Sonnet 5.5)", tc["type"])
	}
}

// The same for "required" (any): a forced any also becomes auto.
func TestMessagesAPI_RequiredToolChoiceBecomesAutoForSonnet55(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "event: message_start\ndata: %s\n\nevent: message_stop\ndata: %s\n\n", evStart, evStop)
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL+"/v1", "k", "claude-sonnet-5-5")
	c.MessagesAPI = true
	strict := true
	var td ToolDef
	td.Type = "function"
	td.Function.Name = "exec"
	td.Function.Description = "run"
	td.Function.Parameters = map[string]any{"type": "object"}
	td.Function.Strict = &strict
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "go"}},
		[]ToolDef{td}, &ChatOpts{ToolChoice: "required"})
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	tc, ok := got["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice %v", got["tool_choice"])
	}
	if tc["type"] != "auto" {
		t.Errorf("tool_choice type = %v, want auto (required on Sonnet 5.5)", tc["type"])
	}
}
