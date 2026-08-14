package llm

import (
	"context"
	"os"
	"testing"
	"time"
)

func llamaClient() *Client  { return NewClient("http://box:8111/v1", "", "Qwen3.8-27B") }
func claudeClient() *Client { return NewClient("http://box:8111/v1", "", "claude-haiku-4-5") }

// TestReasoningDialect: the dialect comes from the model id or the host, because
// there is nothing to probe — both dialects share /v1/chat/completions.
func TestReasoningDialect(t *testing.T) {
	if llamaClient().anthropicDialect() {
		t.Error("a llama.cpp model must not be treated as Anthropic")
	}
	if !claudeClient().anthropicDialect() {
		t.Error("a claude-* model must be treated as Anthropic")
	}
	// Reached directly rather than through corrallm: the model id carries no
	// claude- prefix but the host is unambiguous.
	if !NewClient("https://api.anthropic.com/v1", "", "whatever").anthropicDialect() {
		t.Error("api.anthropic.com must be treated as Anthropic regardless of model id")
	}
}

// TestBudgetMapsPerDialect is the point of the whole file: each provider gets
// ITS field and never the other's. Neither rejects a foreign field — both
// answer 200 and drop it — so a mix-up is invisible at runtime and only a test
// can catch it.
func TestBudgetMapsPerDialect(t *testing.T) {
	n := 1024

	body := map[string]any{}
	llamaClient().applyReasoning(body, &ChatOpts{
		ReasoningBudgetTokens:  &n,
		ReasoningBudgetMessage: "wrap up",
	})
	if body["reasoning_budget_tokens"] != 1024 {
		t.Errorf("llama.cpp budget = %v, want 1024", body["reasoning_budget_tokens"])
	}
	if body["reasoning_budget_message"] != "wrap up" {
		t.Errorf("llama.cpp budget message = %v", body["reasoning_budget_message"])
	}
	if _, ok := body["thinking"]; ok {
		t.Error("Anthropic's thinking block leaked onto a llama.cpp request")
	}

	abody := map[string]any{}
	claudeClient().applyReasoning(abody, &ChatOpts{
		ReasoningBudgetTokens:  &n,
		ReasoningBudgetMessage: "wrap up",
	})
	th, ok := abody["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("Anthropic thinking block missing: %+v", abody)
	}
	if th["type"] != "enabled" || th["budget_tokens"] != 1024 {
		t.Errorf("thinking = %+v, want {enabled 1024}", th)
	}
	if _, ok := abody["reasoning_budget_tokens"]; ok {
		t.Error("llama.cpp's field leaked onto an Anthropic request")
	}
	// Anthropic ends thinking without announcing it; sending a message it
	// cannot honour would imply the model gets told, and it does not.
	if _, ok := abody["reasoning_budget_message"]; ok {
		t.Error("budget message sent to a provider that has no such concept")
	}
}

// TestBudgetUnsetSendsNothing: an absent budget must leave the request alone, or
// every caller with no opinion silently acquires one.
func TestBudgetUnsetSendsNothing(t *testing.T) {
	for name, c := range map[string]*Client{"llama": llamaClient(), "anthropic": claudeClient()} {
		body := map[string]any{}
		c.applyReasoning(body, &ChatOpts{})
		if len(body) != 0 {
			t.Errorf("%s: unset budget wrote %+v", name, body)
		}
		c.applyReasoning(body, nil)
		if len(body) != 0 {
			t.Errorf("%s: nil opts wrote %+v", name, body)
		}
	}
}

// TestZeroAndNegativeBudget: 0 and -1 are MEANINGFUL to llama.cpp (end now /
// disabled) and unrepresentable on Anthropic, which rejects budget_tokens < 1.
// The pointer exists precisely so these stay distinct from "no opinion".
func TestZeroAndNegativeBudget(t *testing.T) {
	for _, n := range []int{0, -1} {
		body := map[string]any{}
		llamaClient().applyReasoning(body, &ChatOpts{ReasoningBudgetTokens: &n})
		if body["reasoning_budget_tokens"] != n {
			t.Errorf("llama.cpp must forward %d verbatim, got %v", n, body["reasoning_budget_tokens"])
		}
		abody := map[string]any{}
		claudeClient().applyReasoning(abody, &ChatOpts{ReasoningBudgetTokens: &n})
		if _, ok := abody["thinking"]; ok {
			t.Errorf("Anthropic must not be sent budget_tokens=%d; it rejects <1", n)
		}
	}
}

// TestEnableThinkingMergesKwargs: the toggle rides in chat_template_kwargs,
// which a caller may already be using. Assigning instead of merging would drop
// theirs silently.
func TestEnableThinkingMergesKwargs(t *testing.T) {
	on := true
	body := map[string]any{"chat_template_kwargs": map[string]any{"mine": "kept"}}
	llamaClient().applyReasoning(body, &ChatOpts{EnableThinking: &on})
	kw, ok := body["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("chat_template_kwargs = %T", body["chat_template_kwargs"])
	}
	if kw["enable_thinking"] != true {
		t.Errorf("enable_thinking = %v, want true", kw["enable_thinking"])
	}
	if kw["mine"] != "kept" {
		t.Error("caller's own chat_template_kwargs were clobbered")
	}

	// Anthropic has no toggle — a budget IS the toggle — so this must not
	// invent a kwarg its endpoint would ignore.
	abody := map[string]any{}
	claudeClient().applyReasoning(abody, &ChatOpts{EnableThinking: &on})
	if _, ok := abody["chat_template_kwargs"]; ok {
		t.Error("chat_template_kwargs sent to the Anthropic dialect")
	}
}

// TestApplySamplingCarriesReasoning: the knobs must reach the body through the
// SAME path the sampling params take, or one of the two body-builders forgets
// them.
func TestApplySamplingCarriesReasoning(t *testing.T) {
	n := 512
	body := map[string]any{}
	llamaClient().applySampling(body, &ChatOpts{ReasoningBudgetTokens: &n})
	if body["reasoning_budget_tokens"] != 512 {
		t.Errorf("budget did not survive applySampling: %+v", body)
	}
}

// TestWithAutoReasoningBudgetKeepsExplicit: an explicit budget is the caller's
// decision and auto must not overwrite it.
func TestWithAutoReasoningBudgetKeepsExplicit(t *testing.T) {
	n := 77
	opts := &ChatOpts{ReasoningBudgetTokens: &n}
	// No server here, so the auto path cannot succeed; the point is that it does
	// not try to, and does not clear what is already set.
	got := llamaClient().WithAutoReasoningBudget(t.Context(), opts, "sys", nil, 0)
	if got.ReasoningBudgetTokens == nil || *got.ReasoningBudgetTokens != 77 {
		t.Errorf("explicit budget overwritten: %+v", got.ReasoningBudgetTokens)
	}
	if got.ReasoningBudgetMessage != "" {
		t.Error("a default message was added to a caller-set budget")
	}
}

// TestWithAutoReasoningBudgetUnreachableServer: when the window cannot be
// learned, send NO budget. A guess can only truncate, and the failure would land
// on the request rather than here.
func TestWithAutoReasoningBudgetUnreachableServer(t *testing.T) {
	c := NewClient("http://127.0.0.1:1/v1", "", "m") // nothing listens on port 1
	opts := c.WithAutoReasoningBudget(t.Context(), nil, "sys", nil, 0)
	if opts.ReasoningBudgetTokens != nil {
		t.Errorf("budget invented without a known window: %v", *opts.ReasoningBudgetTokens)
	}
}

// TestLive_AutoReasoningBudget sizes a budget against a real endpoint.
// Guarded: LLM_LIVE=1 LLM_URL=... LLM_MODEL=... go test -run TestLive_AutoReasoningBudget ./llm
//
// Worth running against a real server because both halves are discovery: the
// window comes from /props (or corrallm's /upstream/<model>/props) and the
// prompt size from whichever tokenizer the endpoint turns out to have. Neither
// can be faked convincingly in a unit test, and a wrong window silently
// produces a budget that overruns the context.
func TestLive_AutoReasoningBudget(t *testing.T) {
	if os.Getenv("LLM_LIVE") == "" {
		t.Skip("set LLM_LIVE=1 (+ LLM_URL/LLM_MODEL) to size against a real endpoint")
	}
	url := os.Getenv("LLM_URL")
	if url == "" {
		url = "http://127.0.0.1:8111"
	}
	model := os.Getenv("LLM_MODEL")
	if model == "" {
		model = "Qwen3.8-27B"
	}
	c := NewClient(url, os.Getenv("LLM_KEY"), model)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	window, ok := c.ContextWindow(ctx)
	if !ok {
		t.Fatalf("no context window from %s (%s)", url, model)
	}
	t.Logf("window=%d", window)

	budget, ok := c.AutoReasoningBudget(ctx, "You are a helpful assistant.", nil, 4096)
	if !ok {
		t.Fatal("could not size a budget despite a known window")
	}
	t.Logf("budget=%d (%.0f%% of window)", budget, 100*float64(budget)/float64(window))
	if budget >= window {
		t.Errorf("budget %d must be smaller than the window %d", budget, window)
	}
}
