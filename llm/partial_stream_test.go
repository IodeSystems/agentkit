package llm

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"
)

// A stream that ends mid-tool-call must NOT yield a ToolCall. This is the
// origin of the poisoned-session bug: the provider never sends a malformed
// call, the client manufactures one from a half-filled accumulator.
func TestAbruptStreamDoesNotEmitToolCall(t *testing.T) {
	// deltas building `{"path":"a.txt","content":"unterm` … then nothing
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"a.txt\",\"content\":\"unterm"}}]}}]}`,
		"", // stream just stops: no finish_reason, no [DONE]
	}, "\n\n")

	c := &Client{}
	ch := make(chan StreamChunk, 16)
	go c.readStream(context.Background(), io.NopCloser(strings.NewReader(sse)), ch, time.Now())

	var tools []ToolCall
	var errs []StreamChunk
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
		if chunk.Error != "" {
			errs = append(errs, chunk)
		}
	}
	if len(tools) != 0 {
		t.Fatalf("emitted %d tool call(s) from an incomplete stream: %+v", len(tools), tools)
	}
	if len(errs) != 1 {
		t.Fatalf("expected one error chunk, got %d", len(errs))
	}
	if !strings.Contains(errs[0].Error, "incomplete") {
		t.Errorf("error should say the call is incomplete: %q", errs[0].Error)
	}
	// The work must still be recoverable by the caller.
	if len(errs[0].PartialToolCalls) != 1 {
		t.Fatalf("partial not handed back: %+v", errs[0].PartialToolCalls)
	}
	if !strings.Contains(errs[0].PartialToolCalls[0].Function.Arguments, "a.txt") {
		t.Error("partial arguments were dropped")
	}
}

// A stream that DOES complete must still emit the tool call normally.
func TestCompleteStreamStillEmitsToolCall(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"a.txt\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	c := &Client{}
	ch := make(chan StreamChunk, 16)
	go c.readStream(context.Background(), io.NopCloser(strings.NewReader(sse)), ch, time.Now())

	var tools []ToolCall
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Function.Name != "write_file" {
		t.Errorf("wrong tool: %+v", tools[0])
	}
}

// The real-world shape: the provider terminates CLEANLY (finish_reason=length,
// then [DONE]) but the tool-call arguments were cut off mid-string. Measured
// verbatim from llama.cpp. A stream that ends properly is NOT evidence that the
// payload is whole, and this is the path the original poisoning came through.
//
// The call IS emitted, unlike the abrupt-end case above. The two are different
// failures: an abrupt end is a broken transport, where the whole response is
// suspect and the request should be retried; a clean finish_reason=length is the
// model overrunning its own budget, where the response is intact and the right
// answer is a tool_result telling it so. Emitting lets agent.Session refuse the
// dispatch and reply with that result, which is what lets the model retry
// smaller. Swallowing it here would abort the turn with the model none the wiser.
func TestCleanFinishWithTruncatedArgumentsIsEmittedForTheCallerToRefuse(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"finish_reason":null,"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"Big.java\",\"content\":\"public class Big {"}}]}}]}`,
		`data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	c := &Client{}
	ch := make(chan StreamChunk, 16)
	go c.readStream(context.Background(), io.NopCloser(strings.NewReader(sse)), ch, time.Now())

	var tools []ToolCall
	var errs []StreamChunk
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
		if chunk.Error != "" {
			errs = append(errs, chunk)
		}
	}
	if len(errs) != 0 {
		t.Fatalf("must not error the turn — the caller answers this call: %+v", errs)
	}
	if len(tools) != 1 {
		t.Fatalf("expected the truncated call to be handed to the caller, got %d", len(tools))
	}
	if tools[0].Function.Name != "write_file" || tools[0].ID != "c1" {
		t.Errorf("identity must survive so the result can be paired to it: %+v", tools[0])
	}
	if !strings.Contains(tools[0].Function.Arguments, "Big.java") {
		t.Error("partial arguments should be handed back verbatim for salvage")
	}
	// Reported faithfully — still unparseable. The caller, not the transport,
	// decides what to do about that.
	if json.Valid([]byte(tools[0].Function.Arguments)) {
		t.Error("fixture should be unparseable")
	}
}
