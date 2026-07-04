package llm

// Coverage for the streaming-delta accumulator. OpenAI's tool-call
// streaming protocol fragments arguments across many chunks keyed by
// `index`; readStream must reassemble before emitting, otherwise the
// harness dispatches one tool call per character of arguments (the
// failure mode caught in the first real-LLM scenario run).

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestReadStream_AccumulatesToolCallFragments(t *testing.T) {
	// Mimic OpenAI's streaming format: one tool call sliced into
	// fragments across multiple SSE events.
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant"}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"propose_task","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"purpose\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"read the"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" README\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	chunks := collect(t, body)

	// Count tool calls.
	var calls []ToolCall
	for _, c := range chunks {
		if c.ToolCall != nil {
			calls = append(calls, *c.ToolCall)
		}
	}
	if len(calls) != 1 {
		t.Fatalf("got %d tool calls; want 1 (fragments must accumulate)", len(calls))
	}
	tc := calls[0]
	if tc.ID != "call_1" {
		t.Errorf("id = %q; want call_1", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("type = %q; want function", tc.Type)
	}
	if tc.Function.Name != "propose_task" {
		t.Errorf("name = %q; want propose_task", tc.Function.Name)
	}
	const wantArgs = `{"purpose":"read the README"}`
	if tc.Function.Arguments != wantArgs {
		t.Errorf("arguments = %q; want %q", tc.Function.Arguments, wantArgs)
	}
}

func TestReadStream_TwoParallelToolCalls(t *testing.T) {
	// LLM emits two tool_calls in one response — different `index`
	// values, fragments interleaved on the wire.
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"propose_task","arguments":"{\"p\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"propose_task","arguments":"{\"p\":"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"first\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"second\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	chunks := collect(t, body)
	var calls []ToolCall
	for _, c := range chunks {
		if c.ToolCall != nil {
			calls = append(calls, *c.ToolCall)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("got %d tool calls; want 2", len(calls))
	}
	if calls[0].ID != "a" || calls[1].ID != "b" {
		t.Errorf("ids = %q,%q; want a,b", calls[0].ID, calls[1].ID)
	}
	if calls[0].Function.Arguments != `{"p":"first"}` {
		t.Errorf("call 0 args = %q", calls[0].Function.Arguments)
	}
	if calls[1].Function.Arguments != `{"p":"second"}` {
		t.Errorf("call 1 args = %q", calls[1].Function.Arguments)
	}
}

func TestReadStream_ContentStillStreams(t *testing.T) {
	// Content tokens must still arrive one-at-a-time for UI streaming.
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
		`data: {"choices":[{"delta":{"content":", "}}]}`,
		`data: {"choices":[{"delta":{"content":"world."}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	chunks := collect(t, body)
	var content strings.Builder
	for _, c := range chunks {
		content.WriteString(c.Content)
	}
	if content.String() != "Hello, world." {
		t.Errorf("content = %q; want %q", content.String(), "Hello, world.")
	}
}

func TestReadStream_UsageOnlyChunk(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"hi"}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")
	chunks := collect(t, body)
	var usage *Usage
	for _, c := range chunks {
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	if usage == nil {
		t.Fatalf("no usage chunk")
	}
	if usage.TotalTokens != 15 {
		t.Errorf("total = %d; want 15", usage.TotalTokens)
	}
}

// collect spins up an httptest server returning `body` over the
// streaming endpoint, dials it via the client, and drains the channel.
func collect(t *testing.T, body string) []StreamChunk {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(ts.Close)

	c := NewClient(ts.URL, "", "test-model")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var out []StreamChunk
	for chunk := range ch {
		out = append(out, chunk)
	}
	return out
}
