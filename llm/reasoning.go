package llm

import (
	"context"
	"strings"
)

// Thinking budgets.
//
// A reasoning model spends tokens BEFORE it answers, and the two providers this
// client reaches disagree about how to bound that spend:
//
//	llama.cpp   reasoning_budget_tokens: N   (+ reasoning_budget_message)
//	Anthropic   thinking: {type, budget_tokens}
//
// Neither rejects the other's field. Both answer 200 and drop it — measured
// 2026-08-14 against llama.cpp b10380 and Anthropic's OpenAI-compat endpoint,
// each sent the wrong field and each returned a normal completion. That is the
// whole reason this file exists: a hand-written body with a plausible-looking
// `reasoning_budget` cannot be told apart from one that worked, so the mapping
// has to happen somewhere that a typo is a compile error.

// reasoningDialect is which shape an endpoint understands.
//
// Detected from the model id and the base URL rather than probed, unlike
// tokenizeRoute: there is nothing to probe for. Both dialects share ONE route
// (/v1/chat/completions) and both silently accept the other's field, so a probe
// would come back 200 either way and learn nothing. The model id is the honest
// signal — corrallm's own passthrough is keyed on `claude-*` globs, and a
// request that reaches Anthropic at all had to name a claude model to get there.
func (c *Client) anthropicDialect() bool {
	return strings.HasPrefix(c.model, "claude-") ||
		strings.Contains(c.baseURL, "api.anthropic.com")
}

// applyReasoning maps the reasoning knobs in opts onto an outgoing body in
// whichever dialect this endpoint speaks.
//
// Called from applySampling so both the streaming and non-streaming paths get
// it from one place; adding a third body-builder without routing it through
// there would silently lose the budget.
func (c *Client) applyReasoning(body map[string]any, opts *ChatOpts) {
	if opts == nil {
		return
	}
	anth := c.anthropicDialect()

	// Anthropic has no separate thinking toggle: the presence of a `thinking`
	// block with a budget IS the toggle, so EnableThinking alone cannot be
	// expressed and is folded into the budget branch below.
	if opts.EnableThinking != nil && !anth {
		// chat_template_kwargs is a passthrough to the server's Jinja template,
		// so this only means anything to a template that reads it (Qwen3.x
		// does). Merged rather than assigned: a caller may already have put
		// other kwargs here, and clobbering them would be a silent loss.
		kw, _ := body["chat_template_kwargs"].(map[string]any)
		if kw == nil {
			kw = map[string]any{}
		}
		kw["enable_thinking"] = *opts.EnableThinking
		body["chat_template_kwargs"] = kw
	}

	if opts.ReasoningBudgetTokens == nil {
		return
	}
	n := *opts.ReasoningBudgetTokens

	if anth {
		// A non-positive budget is "no budget" here, not "think for zero
		// tokens": Anthropic rejects budget_tokens < 1, and llama.cpp's 0
		// ("end immediately") has no equivalent. Emitting nothing leaves the
		// account default in charge, which is the closest honest mapping.
		if n > 0 {
			body["thinking"] = map[string]any{"type": "enabled", "budget_tokens": n}
		}
		return
	}

	body["reasoning_budget_tokens"] = n
	if opts.ReasoningBudgetMessage != "" {
		// Worth sending on every budgeted request. The cut is HARD: measured on
		// Qwen3.8-27B, reasoning stopped mid-word and the message was appended
		// to the fragment ("...first has 1 hourBudget reached. Give the final
		// answer now."). Without it the model gets an amputation and no
		// instruction. llama.cpp PR #25961 adds a soft ratio that warns at a
		// fraction of the budget and waits for a newline; until that merges,
		// this message is the only thing telling the model what happened.
		body["reasoning_budget_message"] = opts.ReasoningBudgetMessage
	}
}

// DefaultBudgetMessage is what to say at the cut when the caller has no opinion.
// Imperative and short: it is injected mid-thought, so it competes with whatever
// the model was in the middle of saying.
const DefaultBudgetMessage = "Thinking budget reached. Stop reasoning and give your final answer now."

// reasoningShare is the fraction of the free window handed to thinking.
//
// 2/3 is Qwen3.8's own published ratio — its card recommends 262,144 tokens of
// reasoning against 131,072 of final response. Those absolutes are quoted
// "within the 1M context length" and do not survive being moved to a smaller
// window; the RATIO does, which is the only part worth carrying.
const reasoningShare = 2.0 / 3.0

// AutoReasoningBudget sizes a thinking budget from what is actually left in the
// context window, and ok=false when it cannot be sized honestly.
//
// reserve is what the caller wants kept back for the final answer BEYOND the
// share arithmetic — tool results still to come, a compaction margin, whatever
// the caller knows and this package does not.
//
// ok=false is a fact, not an error, and the caller should send no budget at all
// rather than guess one: an unbudgeted request thinks as much as it wants, which
// is right when the window is unknown. A guessed budget can only truncate.
func (c *Client) AutoReasoningBudget(ctx context.Context, system string, tools []ToolDef, reserve int) (int, bool) {
	window, ok := c.ContextWindow(ctx)
	if !ok || window <= 0 {
		return 0, false
	}
	used, ok := c.CountPrompt(ctx, system, tools)
	if !ok {
		// An estimate here would be worse than nothing. Overshooting the prompt
		// size inflates the budget and the model runs past the window; the
		// failure lands on the request, not on this function, so it would be
		// diagnosed as anything but a bad guess.
		return 0, false
	}
	free := window - used - reserve
	if free <= 0 {
		return 0, false
	}
	budget := int(float64(free) * reasoningShare)
	if budget <= 0 {
		return 0, false
	}
	return budget, true
}

// WithAutoReasoningBudget fills in ReasoningBudgetTokens on opts when it is
// unset, leaving a caller's explicit budget alone.
//
// Deliberately does NOT set EnableThinking. Whether to think is a decision about
// the task; how long to think is a decision about the window, and only the
// second one is arithmetic this package can do.
func (c *Client) WithAutoReasoningBudget(ctx context.Context, opts *ChatOpts, system string, tools []ToolDef, reserve int) *ChatOpts {
	if opts == nil {
		opts = &ChatOpts{}
	}
	if opts.ReasoningBudgetTokens != nil {
		return opts
	}
	n, ok := c.AutoReasoningBudget(ctx, system, tools, reserve)
	if !ok {
		return opts
	}
	opts.ReasoningBudgetTokens = &n
	if opts.ReasoningBudgetMessage == "" {
		opts.ReasoningBudgetMessage = DefaultBudgetMessage
	}
	return opts
}
