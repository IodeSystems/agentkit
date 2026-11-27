package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	shim "github.com/iodesystems/openai-claude-api-shim"
)

// bodyServer answers chat requests from replies in order (the last repeats) and
// keeps every request body.
type bodyServer struct {
	mu     sync.Mutex
	bodies [][]byte
}

type reply struct {
	status int
	body   string
}

func newBodyServer(t *testing.T, replies ...reply) (*httptest.Server, *bodyServer) {
	t.Helper()
	bs := &bodyServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bs.mu.Lock()
		n := len(bs.bodies)
		bs.bodies = append(bs.bodies, b)
		bs.mu.Unlock()
		if n >= len(replies) {
			n = len(replies) - 1
		}
		rp := replies[n]
		if rp.status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(rp.status)
			fmt.Fprint(w, rp.body)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, rp.body)
	}))
	t.Cleanup(srv.Close)
	return srv, bs
}

func (bs *bodyServer) messages(t *testing.T, i int) []map[string]any {
	t.Helper()
	bs.mu.Lock()
	defer bs.mu.Unlock()
	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(bs.bodies[i], &body); err != nil {
		t.Fatalf("body %d: %v", i, err)
	}
	return body.Messages
}

func replayDrain(t *testing.T, c *Client, msgs []Message) []StreamChunk {
	t.Helper()
	ch, err := c.ChatStream(context.Background(), msgs, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var out []StreamChunk
	for chunk := range ch {
		out = append(out, chunk)
	}
	return out
}

var replayHistory = []Message{
	{Role: "user", Content: "go"},
	{Role: "assistant", Content: "calling", Reasoning: "plan: call f",
		ReasoningOpaque: json.RawMessage(`[{"type":"reasoning.text","text":"plan: call f","signature":"S1","format":"anthropic-claude-v1","index":0}]`),
		ToolCalls: []ToolCall{{ID: "t1", Type: "function", Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{"f", "{}"}}}},
	{Role: "tool", ToolCallID: "t1", Content: "ok"},
}

func TestReplayFieldsUnsetBodyIsByteIdentical(t *testing.T) {
	srv, bs := newBodyServer(t, reply{200, okStream})
	c := NewClient(srv.URL, "k", "m")
	plain := []Message{{Role: "user", Content: "go"}, {Role: "assistant", Content: "x"}, {Role: "user", Content: "y"}}
	replayDrain(t, c, plain)
	b, err := json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(bs.bodies[0]), `"messages":`+string(b)) {
		t.Errorf("messages are not the plain marshal:\n%s\n%s", bs.bodies[0], b)
	}
	if strings.Contains(string(bs.bodies[0]), "reasoning") {
		t.Errorf("a body with no reasoning set names reasoning: %s", bs.bodies[0])
	}
}

func TestReplayDeepSeekSendsReasoningContentOnAssistantOnly(t *testing.T) {
	srv, bs := newBodyServer(t, reply{200, okStream})
	c := NewClient(srv.URL, "k", "deepseek-v4") // llamacpp dialect: the OpenAI-compatible default
	replayDrain(t, c, replayHistory)
	msgs := bs.messages(t, 0)
	if got := msgs[1]["reasoning_content"]; got != "plan: call f" {
		t.Errorf("assistant reasoning_content = %v", got)
	}
	for i, m := range msgs {
		if _, ok := m["reasoning_details"]; ok {
			t.Errorf("messages[%d] carries reasoning_details on an OpenAI-compatible endpoint", i)
		}
		if _, ok := m["reasoning_content"]; ok && i != 1 {
			t.Errorf("messages[%d] (%v) carries reasoning_content", i, m["role"])
		}
	}
	if replayHistory[1].replay != nil {
		t.Error("the caller's message was written")
	}
}

func TestReplayOpenRouterSendsDetailsElseText(t *testing.T) {
	srv, bs := newBodyServer(t, reply{200, okStream})
	c := NewClient(srv.URL, "k", "deepseek/deepseek-v4")
	c.SetReasoningDialect(DialectOpenRouter)
	textOnly := append([]Message(nil), replayHistory...)
	textOnly = append(textOnly, Message{Role: "assistant", Content: "done", Reasoning: "wrap up"})
	replayDrain(t, c, textOnly)
	msgs := bs.messages(t, 0)
	d, ok := msgs[1]["reasoning_details"].([]any)
	if !ok || len(d) != 1 || d[0].(map[string]any)["signature"] != "S1" {
		t.Errorf("reasoning_details = %v", msgs[1]["reasoning_details"])
	}
	if _, ok := msgs[1]["reasoning"]; ok {
		t.Errorf("both reasoning and reasoning_details sent")
	}
	if msgs[3]["reasoning"] != "wrap up" {
		t.Errorf("text-only assistant: reasoning = %v", msgs[3]["reasoning"])
	}
}

func TestReplayDroppedOnAnthropicCompatEndpoint(t *testing.T) {
	srv, bs := newBodyServer(t, reply{200, okStream})
	c := NewClient(srv.URL, "k", "claude-opus-5-5") // Anthropic's OpenAI-compatible endpoint
	replayDrain(t, c, replayHistory)
	if strings.Contains(string(bs.bodies[0]), "reasoning_") || strings.Contains(string(bs.bodies[0]), "plan: call f") {
		t.Errorf("reasoning sent to the compat endpoint: %s", bs.bodies[0])
	}
}

func TestReplayUnsignedThinkingNeverReachesMessagesAPI(t *testing.T) {
	c, got := fakeMessages(t, evStart, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, evStop)
	msgs := []Message{{Role: "user", Content: "go"},
		{Role: "assistant", Content: "x", Reasoning: "unsigned thought"},
		{Role: "assistant", Content: "y", Reasoning: "t", ReasoningOpaque: json.RawMessage(`[{"type":"reasoning.text","text":"no sig","index":0}]`)},
		{Role: "user", Content: "more"}}
	msgChunks(t, c, msgs, nil)
	b, _ := json.Marshal(*got)
	if strings.Contains(string(b), "thinking") || strings.Contains(string(b), "unsigned thought") || strings.Contains(string(b), "no sig") {
		t.Errorf("unsigned thinking reached Anthropic: %s", b)
	}
}

// shimKeepsSignatures reports whether the shim in the build carries a thinking
// block's signature into its OpenAI shape (reasoning_details).
func shimKeepsSignatures() bool {
	out, err := shim.Response([]byte(`{"content":[{"type":"thinking","thinking":"t","signature":"S"}]}`), shim.StreamOptions{})
	return err == nil && strings.Contains(string(out), `"signature":"S"`)
}

// The full loop on the Messages API: a streamed thinking block's signature
// comes back on the Done chunk, and sending it on the assistant message puts
// the signed block first in that turn.
func TestReplayAnthropicSignatureRoundTripsThroughTheShim(t *testing.T) {
	if !shimKeepsSignatures() {
		t.Skip("the required openai-claude-api-shim drops thinking signatures; needs the reasoning-replay release")
	}
	c, got := fakeMessages(t, evStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"need "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"f"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIGXYZ"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"ENC"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`, evStop)
	var reasoning string
	var opaque json.RawMessage
	var calls []ToolCall
	for _, ch := range msgChunks(t, c, []Message{{Role: "user", Content: "go"}}, nil) {
		reasoning += ch.Reasoning
		if len(ch.ReasoningOpaque) > 0 {
			opaque = ch.ReasoningOpaque
		}
		if ch.ToolCall != nil {
			calls = append(calls, *ch.ToolCall)
		}
	}
	if reasoning != "need f" {
		t.Errorf("reasoning = %q (doubled by the details?)", reasoning)
	}
	if !strings.Contains(string(opaque), `"signature":"SIGXYZ"`) || !strings.Contains(string(opaque), `"data":"ENC"`) {
		t.Fatalf("opaque = %s", opaque)
	}

	next := []Message{{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: calls, Reasoning: reasoning, ReasoningOpaque: opaque},
		{Role: "tool", ToolCallID: calls[0].ID, Content: "ok"}}
	msgChunks(t, c, next, nil)
	var req struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	b, _ := json.Marshal(*got)
	if err := json.Unmarshal(b, &req); err != nil {
		t.Fatal(err)
	}
	a := req.Messages[1].Content
	if len(a) != 3 || a[0]["type"] != "thinking" || a[0]["signature"] != "SIGXYZ" || a[0]["thinking"] != "need f" ||
		a[1]["type"] != "redacted_thinking" || a[1]["data"] != "ENC" || a[2]["type"] != "tool_use" {
		t.Errorf("assistant content = %v", a)
	}
}

func TestReplayRefusalIsLearnedResentAndSkippedAfter(t *testing.T) {
	refusal := `{"error":{"message":"The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.","type":"invalid_request_error"}}`
	srv, bs := newBodyServer(t, reply{400, refusal}, reply{200, okStream})
	c := NewClient(srv.URL, "k", "deepseek-reasoner")
	var trs []Translation
	c.OnTranslate = func(tr Translation) { trs = append(trs, tr) }
	chunks := replayDrain(t, c, replayHistory)
	for _, ch := range chunks {
		if ch.Error != "" {
			t.Fatalf("not resent: %s", ch.Error)
		}
	}
	if len(bs.bodies) != 2 {
		t.Fatalf("%d requests, want the refused one and one resend", len(bs.bodies))
	}
	if !strings.Contains(string(bs.bodies[0]), "reasoning_content") || strings.Contains(string(bs.bodies[1]), "reasoning_content") {
		t.Errorf("resend still carries reasoning_content:\n%s\n%s", bs.bodies[0], bs.bodies[1])
	}
	if v, _ := c.Capabilities().fact(CapReasoningReplay); v != "refused" {
		t.Errorf("not learned: %+v", c.Capabilities().Facts)
	}
	replayDrain(t, c, replayHistory)
	if len(bs.bodies) != 3 || strings.Contains(string(bs.bodies[2]), "reasoning_content") {
		t.Errorf("a later request sent the refused field again")
	}
	if len(trs) == 0 || trs[len(trs)-1].Key != "messages.reasoning" || trs[len(trs)-1].Wanted != "reasoning_content" {
		t.Errorf("skip not reported: %+v", trs)
	}
}

func TestReplayMissingReasoningIsNotARefusalOfTheField(t *testing.T) {
	body := `{"error":{"message":"Missing ` + "`reasoning_content`" + ` field in the assistant message at message index 1","type":"invalid_request_error"}}`
	sent := map[string]any{"messages": []Message{{Role: "assistant", replay: &replayWire{ReasoningContent: "x"}}}}
	for _, u := range refusalFacts(400, body, sent) {
		if u.Key == CapReasoningReplay {
			t.Errorf("a body that asks for reasoning taught %+v", u)
		}
	}
}

func TestReplayRefusalNeedsTheFieldToHaveBeenSent(t *testing.T) {
	body := `{"error":{"message":"Please remove the reasoning_content from your message"}}`
	if us := refusalFacts(400, body, map[string]any{"messages": []Message{{Role: "user"}}}); len(us) != 0 {
		t.Errorf("taught %+v with no reasoning sent", us)
	}
}

func TestReplayAnthropicBadSignatureIsARefusal(t *testing.T) {
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: Invalid ` + "`signature`" + ` in ` + "`thinking`" + ` block"}}`
	sent := map[string]any{"messages": []Message{{Role: "assistant", replay: &replayWire{ReasoningDetails: json.RawMessage(`[]`)}}}}
	us := refusalFacts(400, body, sent)
	if len(us) != 1 || us[0].Key != CapReasoningReplay {
		t.Errorf("facts = %+v", us)
	}
}

func TestMergeReasoningDetailsJoinsFragmentsOfOneItem(t *testing.T) {
	var acc []reasoningItem
	for _, d := range []string{
		`[{"type":"reasoning.text","text":"a ","format":"anthropic-claude-v1","index":0}]`,
		`[{"type":"reasoning.text","text":"b","format":"anthropic-claude-v1","index":0}]`,
		`[{"type":"reasoning.text","text":"","signature":"SIG","format":"anthropic-claude-v1","index":0}]`,
		`[{"type":"reasoning.encrypted","data":"D","format":"anthropic-claude-v1","index":1}]`,
		`[{"type":"reasoning.summary","summary":"s"}]`,
	} {
		acc = mergeReasoningDetails(acc, parseReasoningDetails(json.RawMessage(d)))
	}
	var got []map[string]any
	if err := json.Unmarshal(marshalReasoningDetails(acc), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0]["text"] != "a b" || got[0]["signature"] != "SIG" || got[1]["data"] != "D" || got[2]["summary"] != "s" {
		t.Errorf("merged = %v", got)
	}
}

func TestStreamDoneCarriesMergedReasoningDetailsWithoutDoublingText(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"reasoning\":\"th\",\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"th\",\"index\":0}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning\":\"ink\",\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"ink\",\"index\":0}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"\",\"signature\":\"Z\",\"index\":0}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	srv, _ := newBodyServer(t, reply{200, stream})
	c := NewClient(srv.URL, "k", "m")
	var text string
	var opaque json.RawMessage
	for _, ch := range replayDrain(t, c, []Message{{Role: "user", Content: "x"}}) {
		text += ch.Reasoning
		if ch.Done && len(ch.ReasoningOpaque) > 0 {
			opaque = ch.ReasoningOpaque
		}
	}
	if text != "think" {
		t.Errorf("reasoning = %q", text)
	}
	if string(opaque) != `[{"index":0,"signature":"Z","text":"think","type":"reasoning.text"}]` {
		t.Errorf("opaque = %s", opaque)
	}
}

func TestChatResultCarriesReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"content":"a","reasoning_content":"why","reasoning_details":[{"type":"reasoning.text","text":"why","signature":"S","index":0}]},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)
	res, err := NewClient(srv.URL, "k", "m").Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Reasoning != "why" || !strings.Contains(string(res.ReasoningOpaque), `"signature":"S"`) {
		t.Errorf("result = %+v", res)
	}
}
