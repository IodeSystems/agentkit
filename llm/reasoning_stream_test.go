package llm

import (
	"strings"
	"testing"
)

// The shape llama.cpp b11191 sent through corrallm on 2026-09-26, trimmed:
// reasoning arrives as delta.reasoning_content, content after it, and the usage
// chunk carries NO completion_tokens_details.
func TestReadStream_ReasoningContentIsSeparateAndUnreported(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}`,
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":"User asks 17*3."}}]}`,
		`data: {"choices":[{"index":0,"delta":{"reasoning_content":" 51.\n"}}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"51"}}]}`,
		`data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}`,
		`data: {"choices":[],"usage":{"completion_tokens":40,"prompt_tokens":65,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":0}}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var reasoning, content strings.Builder
	var usage *Usage
	for _, c := range collect(t, body) {
		reasoning.WriteString(c.Reasoning)
		content.WriteString(c.Content)
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	if got := reasoning.String(); got != "User asks 17*3. 51.\n" {
		t.Errorf("reasoning = %q", got)
	}
	if got := content.String(); got != "51" {
		t.Errorf("content = %q; reasoning must not leak into it", got)
	}
	if usage == nil {
		t.Fatal("no usage")
	}
	if n, ok := usage.ReportedReasoningTokens(); ok {
		t.Errorf("reasoning reported as %d; the provider sent no count, so it is unknown", n)
	}
}

// The OpenAI shape: a count in usage.completion_tokens_details, and the
// delta.reasoning spelling some providers use for the text.
func TestReadStream_ReasoningTokensReported(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"reasoning":"think"}}]}`,
		`data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":30,"total_tokens":35,"completion_tokens_details":{"reasoning_tokens":27}}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var reasoning strings.Builder
	var usage *Usage
	for _, c := range collect(t, body) {
		reasoning.WriteString(c.Reasoning)
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	if reasoning.String() != "think" {
		t.Errorf("reasoning = %q; want delta.reasoning read too", reasoning.String())
	}
	n, ok := usage.ReportedReasoningTokens()
	if !ok || n != 27 {
		t.Errorf("reasoning tokens = %d, %v; want 27, true", n, ok)
	}
	// A reported zero is a fact, not an absence.
	zero := 0
	u := &Usage{CompletionTokensDetails: &CompletionTokensDetails{ReasoningTokens: &zero}}
	if n, ok := u.ReportedReasoningTokens(); !ok || n != 0 {
		t.Errorf("reported zero = %d, %v; want 0, true", n, ok)
	}
	if _, ok := (*Usage)(nil).ReportedReasoningTokens(); ok {
		t.Error("nil usage reported reasoning")
	}
}
