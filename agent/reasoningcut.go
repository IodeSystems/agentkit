package agent

// Reasoning-cut repair.
//
// A round can end with reasoning and nothing else: no text, no tool call. The
// usual cause is a thinking budget (llama-server forces </think> at
// reasoning_budget_tokens) followed by max_tokens running out right after, or a
// model that reasons after a failed tool call and stops. Before this file such
// a round persisted nothing, dropped its reasoning, and ended the Turn as
// TurnReply, as if the model had answered. Hosts saw a silent empty reply.
//
// The repair is the same shape as the repetition re-prompt: tell the model what
// happened with a notification, and run one more round. Two things differ.
// The notification carries the tail of the reasoning, because the reasoning
// itself is not replayed (Entry.Reasoning is stored, never sent) and the model
// would otherwise start from zero. And the repair round runs with reasoning
// OFF, because a model that is allowed to think again walks into the same wall.
//
// An assistant entry with empty content is NOT persisted for the cut round:
// llama.cpp refuses two consecutive assistant messages and some providers
// refuse empty assistant content. The notification is the record.

import (
	"fmt"
	"unicode/utf8"

	"github.com/iodesystems/agentkit/llm"
)

// ReasoningCutTag marks the notification entries this produces, so a host can
// render them apart from real notifications.
const ReasoningCutTag = "reasoning-cut"

// reasoningTailBytes bounds how much of the cut reasoning the notice repeats.
// Enough for the model to see where it was heading; small enough that the
// repair does not itself eat the answer room that was just short.
const reasoningTailBytes = 2000

// ReasoningCut is one round that produced reasoning and no text and no tool
// call, as reported to Session.OnReasoningCut.
type ReasoningCut struct {
	// Tokens is the provider's reasoning-token count when the usage carries
	// one, else the session's estimate from the streamed reasoning text.
	Tokens int
	// Finish is the round's stop reason as the stream reported it; "" when the
	// stream named none (an ordinary stop).
	Finish llm.StopReason
	// Repaired is false when the cut is detected, before the repair round runs.
	// It is true on a second call, once the repair round produced text or a
	// tool call. A cut that is not repaired never gets the second call: the
	// Turn ends with *ReasoningCutError instead.
	Repaired bool
}

// ReasoningCutError is returned by Turn (TurnResult.Reason == TurnReasoningCut)
// when a cut round was not recovered: the repair round was empty too, or the
// repair already ran once in this Turn.
type ReasoningCutError struct {
	Tokens int
	Finish llm.StopReason
}

func (e *ReasoningCutError) Error() string {
	return fmt.Sprintf("agent: the model reasoned (%d tokens, finish %q) and produced no answer and no tool call, "+
		"and the repair did not recover it", e.Tokens, e.Finish)
}

// reasoningTail is the last reasoningTailBytes of r, cut forward to a rune
// boundary so the notice stays valid UTF-8, prefixed with "…" when cut.
func reasoningTail(r string) string {
	if len(r) <= reasoningTailBytes {
		return r
	}
	t := r[len(r)-reasoningTailBytes:]
	for len(t) > 0 && !utf8.RuneStart(t[0]) {
		t = t[1:]
	}
	return "…" + t
}

func reasoningCutNotice(reasoning string) string {
	return "Your reasoning was cut before you answered. Its last part was:\n" + reasoningTail(reasoning) +
		"\nDo not reason again. Answer now: make a tool call or give your final answer."
}

// reasoningOff returns a copy of o with the effort set to none and the host's
// budget cleared. llm maps none per dialect (llama.cpp: enable_thinking false
// and budget 0, which ends thinking at once even for a template that ignores
// the flag; Anthropic: no thinking block; openai/openrouter: their off form)
// and clamps it by the capability record, so a thinking-only model gets the
// lowest effort it accepts instead of a certain 400. A host budget is cleared
// because it is a cap on thinking, and this round asks for none.
func reasoningOff(o *llm.ChatOpts) *llm.ChatOpts {
	var c llm.ChatOpts
	if o != nil {
		c = *o
	}
	c.Effort = llm.EffortNone
	c.ReasoningBudgetTokens = nil
	c.ReasoningBudgetMessage = ""
	return &c
}
