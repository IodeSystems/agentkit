package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The OpenAI spec says `arguments` is a STRING containing JSON. Some providers
// send the object directly. Before this, that shape was not tolerated — and the
// streaming path failed SILENTLY, which is the part that mattered: no call, no
// error, no log, indistinguishable from a model that said nothing.
func TestStreamingAcceptsObjectArguments(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"write_file","arguments":{"path":"a.java","n":3}}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`, "",
	}, "\n\n")

	c := &Client{}
	ch := make(chan StreamChunk, 16)
	go c.readStream(context.Background(), io.NopCloser(strings.NewReader(sse)), ch, time.Now())

	var tools []ToolCall
	var errs []string
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
		if chunk.Error != "" {
			errs = append(errs, chunk.Error)
		}
	}
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(tools) != 1 {
		t.Fatalf("object arguments were dropped: got %d calls", len(tools))
	}
	// Normalized to the string form the rest of the stack works in, and it must
	// survive as something agent.Session will dispatch.
	var got map[string]any
	if err := json.Unmarshal([]byte(tools[0].Function.Arguments), &got); err != nil {
		t.Fatalf("arguments not usable as JSON: %v (%q)", err, tools[0].Function.Arguments)
	}
	if got["path"] != "a.java" {
		t.Errorf("arguments corrupted: %v", got)
	}
}

// The non-streaming path decodes into ToolCall directly, so it needs the same
// tolerance — there an object used to fail the decode of the WHOLE response.
func TestNonStreamingAcceptsObjectArguments(t *testing.T) {
	body := `{"choices":[{"message":{"content":"","tool_calls":[
		{"id":"c1","type":"function","function":{"name":"f","arguments":{"path":"a.java"}}},
		{"id":"c2","type":"function","function":{"name":"g","arguments":"{\"path\":\"b.java\"}"}}
	]}}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	_, calls, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("object arguments killed the whole response: %v", err)
	}
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
	// Both shapes normalize to the same string form.
	for _, tc := range calls {
		if !json.Valid([]byte(tc.Function.Arguments)) {
			t.Errorf("%s: not valid JSON: %q", tc.Function.Name, tc.Function.Arguments)
		}
	}
	if !strings.Contains(calls[0].Function.Arguments, "a.java") ||
		!strings.Contains(calls[1].Function.Arguments, "b.java") {
		t.Errorf("arguments mismatched: %q / %q",
			calls[0].Function.Arguments, calls[1].Function.Arguments)
	}
}

// Absent/null arguments mean "no arguments" — a real case for a tool with no
// parameters. It must not be confused with unparseable, which is a failure.
func TestNormalizeArgsAbsentIsNotAFailure(t *testing.T) {
	for _, raw := range []string{``, `null`, `  null `} {
		if got := normalizeArgs(json.RawMessage(raw)); got != "" {
			t.Errorf("normalizeArgs(%q) = %q, want empty", raw, got)
		}
	}
}

// The spec-compliant path is the common one and must be untouched: fragments
// arrive as JSON strings and concatenate into the raw argument text.
func TestNormalizeArgsUnquotesStringFragments(t *testing.T) {
	var sb strings.Builder
	for _, frag := range []string{`"{\"path\":\""`, `"a.java"`, `"\"}"`} {
		sb.WriteString(normalizeArgs(json.RawMessage(frag)))
	}
	if got := sb.String(); got != `{"path":"a.java"}` {
		t.Fatalf("reassembled fragments = %q", got)
	}
}
