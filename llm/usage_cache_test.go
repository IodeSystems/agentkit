package llm

import (
	"encoding/json"
	"testing"
)

// llama-server (and therefore corrallm) reports cache hits in OpenAI's NESTED
// shape. agentkit only ever declared Anthropic's flat fields, so this
// unmarshalled into nothing: cached tokens read as 0 and prompt_tokens looked
// like real work on every turn. Summing that across a conversation bills a
// stable prefix once per turn and makes the tool schema — the single most
// cacheable thing in the context — look like the dominant cost.
func TestUsageParsesOpenAICachedTokens(t *testing.T) {
	// Verbatim from corrallm -> llama-server, second identical request.
	raw := `{"completion_tokens":5,"prompt_tokens":83,"total_tokens":88,
	         "prompt_tokens_details":{"cached_tokens":79}}`
	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatal(err)
	}
	if got := u.CachedPromptTokens(); got != 79 {
		t.Errorf("CachedPromptTokens = %d, want 79 (OpenAI nested shape)", got)
	}
	// The number that reflects work actually done: 83 sent, 79 cached, 4 new.
	if got := u.NewPromptTokens(); got != 4 {
		t.Errorf("NewPromptTokens = %d, want 4", got)
	}
}

// Anthropic's flat shape must keep working.
func TestUsageParsesAnthropicCacheFields(t *testing.T) {
	raw := `{"prompt_tokens":1000,"completion_tokens":10,"total_tokens":1010,
	         "cache_read_input_tokens":900}`
	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatal(err)
	}
	if got := u.CachedPromptTokens(); got != 900 {
		t.Errorf("CachedPromptTokens = %d, want 900 (Anthropic flat shape)", got)
	}
	if got := u.NewPromptTokens(); got != 100 {
		t.Errorf("NewPromptTokens = %d, want 100", got)
	}
}

// A provider that reports no cache info at all: every prompt token is new.
func TestUsageWithoutCacheInfo(t *testing.T) {
	var u Usage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":50,"completion_tokens":5}`), &u); err != nil {
		t.Fatal(err)
	}
	if u.CachedPromptTokens() != 0 || u.NewPromptTokens() != 50 {
		t.Errorf("no cache info: cached=%d new=%d, want 0/50", u.CachedPromptTokens(), u.NewPromptTokens())
	}
}
