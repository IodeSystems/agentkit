package agent

import "fmt"

// overflow.go — a round the provider ended because it ran out of room.
//
// finish_reason="length" is not a transport fault and not a model decision: the
// response hit a ceiling. Which ceiling, and what to do about it, are questions
// only the HOST can answer, because only the host knows the window it is working
// against and whether the prompt or the response is the part that grew.
//
// So this is a report, not a policy. The Session surfaces the cut, the host
// decides between shedding context and telling the model to be briefer, and the
// text it returns is appended to the conversation the same way the repetition
// notice is: as a tail-kept notification, adjacent to whatever the cut landed
// next to.
//
// That adjacency is the whole delivery mechanism, and it covers both shapes the
// cut takes. Cut inside a TOOL CALL: the call is persisted with a placeholder
// and answered with an error result, and this notice lands beside that result —
// the model reads the failure and the reason for it in the same breath. Cut in
// PROSE: there is no call to answer and the turn would otherwise end on a
// half-finished sentence, so the notice is what the retry is prompted from, and
// what a following user message arrives on top of.

// OverflowTag marks the notification entries this produces, so a host can render
// them apart from real notifications — and drop them once they are stale, which
// they are the moment a round completes without a cut.
const OverflowTag = "context-overflow"

// Overflow is one length-cut round, as reported to Session.OnOverflow.
type Overflow struct {
	// PromptTokens and Generated are the PROVIDER's counts for the round that
	// was cut, or 0 when it reported none. Provider numbers rather than
	// estimates on purpose: the host is about to decide whether the prompt or
	// the response filled the window, and that decision is exactly the one an
	// estimate gets wrong.
	PromptTokens int
	Generated    int

	// TruncatedToolCalls names the calls whose arguments the cut left
	// unparseable. Empty means the cut landed in prose — see Retry, which is
	// only consulted in that case.
	TruncatedToolCalls []string

	// Attempt counts consecutive cut rounds in this Turn, from 1. A host uses it
	// to escalate: hint first, shed context after that.
	Attempt int
}

// InToolCall reports whether the cut landed mid tool call, which is the case
// where the model already gets an error result to answer.
func (o Overflow) InToolCall() bool { return len(o.TruncatedToolCalls) > 0 }

// String is the one-line form for a log.
func (o Overflow) String() string {
	where := "in the reply"
	if o.InToolCall() {
		where = fmt.Sprintf("mid tool call (%v)", o.TruncatedToolCalls)
	}
	return fmt.Sprintf("generation cut off %s after %d tokens, on a %d-token prompt (attempt %d)",
		where, o.Generated, o.PromptTokens, o.Attempt)
}

// OverflowDecision is what the host wants done about a cut.
type OverflowDecision struct {
	// Note is appended to the conversation as a tail-kept notification tagged
	// OverflowTag. Empty appends nothing.
	Note string
	// Retry re-runs the round after the note is appended. Only consulted when
	// the cut landed in PROSE: a cut tool call is already answered with an error
	// result, and re-running the round on top of that would ask the model the
	// same question twice.
	//
	// The point of retrying is that the host has CHANGED something first —
	// compacted the history, lowered a budget, queued the note. Retrying without
	// changing anything reproduces the cut exactly, which is the loop this whole
	// file exists to break.
	Retry bool
}
