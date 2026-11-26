package llm

import (
	"context"
	"encoding/json"
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

// TestEffortMergesKwargs: the toggle rides in chat_template_kwargs, which a
// caller may already be using. Assigning instead of merging would drop theirs
// silently.
func TestEffortMergesKwargs(t *testing.T) {
	body := map[string]any{"chat_template_kwargs": map[string]any{"mine": "kept"}}
	llamaClient().applyReasoning(body, &ChatOpts{Effort: EffortHigh})
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

	// Anthropic has no toggle (a budget IS the toggle), so this must not
	// invent a kwarg its endpoint would ignore.
	abody := map[string]any{}
	claudeClient().applyReasoning(abody, &ChatOpts{Effort: EffortHigh})
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

// OpenRouter takes one `reasoning` object, and none of llama.cpp's fields.
// low/medium are the native effort (decision 2), a caller's budget replaces it.
func TestApplyReasoning_OpenRouter(t *testing.T) {
	c := NewClient("https://openrouter.ai/api/v1", "", "deepseek/deepseek-r1")
	n := 2048
	for name, tc := range map[string]struct {
		opts ChatOpts
		want string
	}{
		"effort":       {ChatOpts{Effort: EffortHigh}, `{"effort":"high"}`},
		"low":          {ChatOpts{Effort: EffortLow}, `{"effort":"low"}`},
		"budget":       {ChatOpts{Effort: EffortLow, ReasoningBudgetTokens: &n}, `{"max_tokens":2048}`},
		"none":         {ChatOpts{Effort: EffortNone}, `{"enabled":false}`},
		"none+budget":  {ChatOpts{Effort: EffortNone, ReasoningBudgetTokens: &n}, `{"enabled":false}`},
		"unset":        {ChatOpts{}, ``},
		"budget alone": {ChatOpts{ReasoningBudgetTokens: &n}, `{"max_tokens":2048}`},
	} {
		body := map[string]any{}
		c.applyReasoning(body, &tc.opts)
		got := ""
		if r, ok := body["reasoning"]; ok {
			b, _ := json.Marshal(r)
			got = string(b)
		}
		if got != tc.want {
			t.Errorf("%s: reasoning %s, want %s", name, got, tc.want)
		}
		if _, ok := body["chat_template_kwargs"]; ok {
			t.Errorf("%s: chat_template_kwargs sent to OpenRouter", name)
		}
		if _, ok := body["reasoning_budget_tokens"]; ok {
			t.Errorf("%s: reasoning_budget_tokens sent to OpenRouter", name)
		}
	}
}

// A named dialect wins over detection: the openai one sends only an
// effort, a LiteLLM proxy's address names nothing.
func TestReasoningDialect_Named(t *testing.T) {
	n := 4096
	c := NewClient("https://litellm.corp.example/v1", "", "deepseek-1017")
	body := map[string]any{}
	c.applyReasoning(body, &ChatOpts{Effort: EffortMedium, ReasoningBudgetTokens: &n})
	if _, ok := body["chat_template_kwargs"]; !ok {
		t.Error("auto on an unnamed address is llama.cpp's dialect")
	}
	c.SetReasoningDialect(DialectOpenAI)
	body = map[string]any{}
	c.applyReasoning(body, &ChatOpts{Effort: EffortMedium, ReasoningBudgetTokens: &n})
	if body["reasoning_effort"] != "medium" || len(body) != 1 {
		t.Errorf("openai: %v", body)
	}
	c.SetReasoningDialect(DialectOpenRouter)
	body = map[string]any{}
	c.applyReasoning(body, &ChatOpts{Effort: EffortHigh})
	if r, _ := body["reasoning"].(map[string]any); r["effort"] != "high" {
		t.Errorf("openrouter: %v", body)
	}
}

// goldenBodies is the wire for every dialect x level with an empty record: the
// table in effort.go, as bytes. A cell that changes is a wire change for every
// host, so it is spelled out rather than derived.
func TestEffortGoldenBodies(t *testing.T) {
	mk := func(dialect, url, model string, msgs bool) *Client {
		c := NewClient(url, "", model)
		c.SetReasoningDialect(dialect)
		c.MessagesAPI = msgs
		return c
	}
	llama := mk(DialectLlamaCpp, "http://x/v1", "Qwen3.8", false)
	anth := mk(DialectAnthropic, "http://x/v1", "claude-opus-4-6", false)
	orr := mk(DialectOpenRouter, "http://x/v1", "m", false)
	oai := mk(DialectOpenAI, "http://x/v1", "m", false)
	msg := mk(DialectAnthropic, "http://x/v1", "claude-opus-4-6", true)
	old := mk(DialectAnthropic, "http://x/v1", "claude-haiku-4-5", true)
	cases := []struct {
		name string
		c    *Client
		e    Effort
		want string
	}{
		{"llama unset", llama, EffortUnset, `{}`},
		{"llama none", llama, EffortNone, `{"chat_template_kwargs":{"enable_thinking":false},"reasoning_budget_tokens":0}`},
		{"llama minimal", llama, EffortMinimal, `{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget_tokens":512}`},
		{"llama low", llama, EffortLow, `{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget_tokens":1024}`},
		{"llama medium", llama, EffortMedium, `{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget_tokens":4096}`},
		{"llama high", llama, EffortHigh, `{"chat_template_kwargs":{"enable_thinking":true}}`},
		{"llama xhigh", llama, EffortXHigh, `{"chat_template_kwargs":{"enable_thinking":true}}`},
		{"llama max", llama, EffortMax, `{"chat_template_kwargs":{"enable_thinking":true}}`},

		{"anthropic unset", anth, EffortUnset, `{}`},
		{"anthropic none", anth, EffortNone, `{}`},
		{"anthropic minimal", anth, EffortMinimal, `{"thinking":{"budget_tokens":1024,"type":"enabled"}}`},
		{"anthropic low", anth, EffortLow, `{"thinking":{"budget_tokens":1024,"type":"enabled"}}`},
		{"anthropic medium", anth, EffortMedium, `{"thinking":{"budget_tokens":4096,"type":"enabled"}}`},
		{"anthropic high", anth, EffortHigh, `{"thinking":{"budget_tokens":16384,"type":"enabled"}}`},
		{"anthropic xhigh", anth, EffortXHigh, `{"thinking":{"budget_tokens":32768,"type":"enabled"}}`},
		{"anthropic max", anth, EffortMax, `{"thinking":{"budget_tokens":65536,"type":"enabled"}}`},

		{"openrouter unset", orr, EffortUnset, `{}`},
		{"openrouter none", orr, EffortNone, `{"reasoning":{"enabled":false}}`},
		{"openrouter minimal", orr, EffortMinimal, `{"reasoning":{"effort":"minimal"}}`},
		{"openrouter low", orr, EffortLow, `{"reasoning":{"effort":"low"}}`},
		{"openrouter medium", orr, EffortMedium, `{"reasoning":{"effort":"medium"}}`},
		{"openrouter high", orr, EffortHigh, `{"reasoning":{"effort":"high"}}`},
		{"openrouter xhigh", orr, EffortXHigh, `{"reasoning":{"effort":"xhigh"}}`},
		{"openrouter max", orr, EffortMax, `{"reasoning":{"effort":"max"}}`},

		{"openai unset", oai, EffortUnset, `{}`},
		{"openai none", oai, EffortNone, `{"reasoning_effort":"none"}`},
		{"openai minimal", oai, EffortMinimal, `{"reasoning_effort":"minimal"}`},
		{"openai low", oai, EffortLow, `{"reasoning_effort":"low"}`},
		{"openai medium", oai, EffortMedium, `{"reasoning_effort":"medium"}`},
		{"openai high", oai, EffortHigh, `{"reasoning_effort":"high"}`},
		{"openai xhigh", oai, EffortXHigh, `{"reasoning_effort":"xhigh"}`},
		{"openai max", oai, EffortMax, `{"reasoning_effort":"max"}`},

		{"messages unset", msg, EffortUnset, `{}`},
		{"messages none", msg, EffortNone, `{"reasoning_effort":"low"}`},
		{"messages minimal", msg, EffortMinimal, `{"reasoning_effort":"low"}`},
		{"messages low", msg, EffortLow, `{"reasoning_effort":"low"}`},
		{"messages medium", msg, EffortMedium, `{"reasoning_effort":"medium"}`},
		{"messages high", msg, EffortHigh, `{"reasoning_effort":"high"}`},
		{"messages xhigh", msg, EffortXHigh, `{"reasoning_effort":"xhigh"}`},
		{"messages max", msg, EffortMax, `{"reasoning_effort":"max"}`},
		{"messages old model", old, EffortHigh, `{}`},
	}
	for _, tc := range cases {
		body := map[string]any{}
		tc.c.applyReasoning(body, &ChatOpts{Effort: tc.e})
		b, _ := json.Marshal(body)
		if string(b) != tc.want {
			t.Errorf("%s: %s, want %s", tc.name, b, tc.want)
		}
	}
}

// TestEffortBudgetOverridesLevelBudget: an explicit cap (aw4 sends 2500) beats
// the budget the level implies, but never revives thinking that none turned off,
// and the budget message rides with whichever budget goes out.
func TestEffortBudgetOverridesLevelBudget(t *testing.T) {
	n := 2500
	body := map[string]any{}
	llamaClient().applyReasoning(body, &ChatOpts{Effort: EffortMedium, ReasoningBudgetTokens: &n, ReasoningBudgetMessage: "stop"})
	if body["reasoning_budget_tokens"] != 2500 || body["reasoning_budget_message"] != "stop" {
		t.Errorf("llama override: %v", body)
	}
	body = map[string]any{}
	llamaClient().applyReasoning(body, &ChatOpts{Effort: EffortNone, ReasoningBudgetTokens: &n})
	if body["reasoning_budget_tokens"] != 0 {
		t.Errorf("none must keep budget 0: %v", body)
	}
	body = map[string]any{}
	claudeClient().applyReasoning(body, &ChatOpts{Effort: EffortHigh, ReasoningBudgetTokens: &n})
	if th, _ := body["thinking"].(map[string]any); th["budget_tokens"] != 2500 {
		t.Errorf("anthropic override: %v", body)
	}
}

// The openai dialect sends the standard's values as given, off included.
func TestReasoningDialect_OpenAISendsStandardValues(t *testing.T) {
	c := NewClient("https://litellm.corp.example/v1", "", "glm")
	c.SetReasoningDialect(DialectOpenAI)
	for _, v := range []Effort{"none", "minimal", "low", "medium", "high", "xhigh"} {
		body := map[string]any{}
		c.applyReasoning(body, &ChatOpts{Effort: v})
		if body["reasoning_effort"] != string(v) || len(body) != 1 {
			t.Errorf("%s: %v", v, body)
		}
	}
}
