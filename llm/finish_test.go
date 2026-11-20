package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// finish_reason and native_finish_reason reach the caller verbatim, and
// reasoning_details is read only when it is the sole carrier of reasoning.

// runSSE serves the given data payloads and returns every chunk plus the
// Usage OnUsage saw.
func runSSE(t *testing.T, payloads ...string) ([]StreamChunk, *Usage) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, p := range payloads {
			fmt.Fprintf(w, "data: %s\n\n", p)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	var got *Usage
	c.OnUsage = func(u Usage) { got = &u }
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var chunks []StreamChunk
	for chunk := range ch {
		chunks = append(chunks, chunk)
	}
	return chunks, got
}

func doneChunk(t *testing.T, chunks []StreamChunk) StreamChunk {
	t.Helper()
	for _, c := range chunks {
		if c.Done {
			return c
		}
	}
	t.Fatal("no Done chunk")
	return StreamChunk{}
}

// A "stop" finish with a native word was reduced to nothing: only length and
// content_filter survived. Both words must reach the Done chunk and the Usage,
// even when the usage arrives in a LATER chunk than the finish.
func TestFinishReasonAndNativeKeptOnStop(t *testing.T) {
	chunks, usage := runSSE(t,
		`{"choices":[{"delta":{"content":"hi"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop","native_finish_reason":"STOP"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	d := doneChunk(t, chunks)
	if d.FinishReason != "stop" || d.NativeFinishReason != "STOP" {
		t.Errorf("Done chunk = %q/%q, want stop/STOP", d.FinishReason, d.NativeFinishReason)
	}
	if usage == nil || usage.FinishReason != "stop" || usage.NativeFinishReason != "STOP" {
		t.Errorf("OnUsage usage = %+v, want stop/STOP", usage)
	}
	var lateUsage *Usage
	for _, c := range chunks {
		if c.Usage != nil {
			lateUsage = c.Usage
		}
	}
	if lateUsage == nil || lateUsage.FinishReason != "stop" {
		t.Errorf("usage chunk after the finish lost FinishReason: %+v", lateUsage)
	}
}

// OpenRouter reports a malformed call as finish_reason "error" (or "stop")
// with the upstream word in native_finish_reason. Losing it hides why the
// reply is empty.
func TestNativeFinishReasonMalformedFunctionCallKept(t *testing.T) {
	chunks, _ := runSSE(t,
		`{"choices":[{"delta":{},"finish_reason":"error","native_finish_reason":"MALFORMED_FUNCTION_CALL"}]}`)
	d := doneChunk(t, chunks)
	if d.FinishReason != "error" || d.NativeFinishReason != "MALFORMED_FUNCTION_CALL" {
		t.Errorf("Done chunk = %q/%q", d.FinishReason, d.NativeFinishReason)
	}
}

// The non-streaming body fills the same Usage fields.
func TestFinishReasonFromNonStreamingBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop","native_finish_reason":"STOP"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	var got *Usage
	c.OnUsage = func(u Usage) { got = &u }
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatal(err)
	}
	if got == nil || got.FinishReason != "stop" || got.NativeFinishReason != "STOP" {
		t.Errorf("usage = %+v, want stop/STOP", got)
	}
}

func reasoningOf(chunks []StreamChunk) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString(c.Reasoning)
	}
	return b.String()
}

// A stream with only reasoning_details lost all its reasoning. Text and
// summary count; the encrypted blob does not.
func TestReasoningDetailsOnlyIsRead(t *testing.T) {
	chunks, _ := runSSE(t,
		`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"think "},{"type":"reasoning.summary","summary":"sum "},{"type":"reasoning.encrypted","data":"SECRET"}]}}]}`,
		`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`)
	if got := reasoningOf(chunks); got != "think sum " {
		t.Errorf("reasoning = %q, want %q", got, "think sum ")
	}
}

// OpenRouter sends the same text in reasoning and reasoning_details. Reading
// both doubled it.
func TestReasoningDetailsNotDoubledWithReasoning(t *testing.T) {
	chunks, _ := runSSE(t,
		`{"choices":[{"delta":{"reasoning":"think","reasoning_details":[{"type":"reasoning.text","text":"think"}]}}]}`,
		`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`)
	if got := reasoningOf(chunks); got != "think" {
		t.Errorf("reasoning = %q, want %q (not doubled)", got, "think")
	}
}
