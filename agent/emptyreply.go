package agent

// Empty-reply repair.
//
// A round can end with nothing at all: no text, no tool call, no reasoning.
// Seen live: Gemini through OpenRouter returns such rounds with finish_reason
// "stop" and native_finish_reason "MALFORMED_FUNCTION_CALL". Before this file
// the Turn ended as TurnReply with Reply "" and a nil error, so a host could
// not tell "the model had nothing to say" from "the provider dropped the
// answer".
//
// The repair is the reasoning-cut repair (reasoningcut.go) without the
// reasoning: one notification, one more round. It shares that repair's budget:
// ONE repair per Turn in total, whichever kind of empty round comes first. Two
// differences. The round runs with the host's opts UNCHANGED: reasoning did
// not cause this, and a thinking-only model refuses reasoning off. And a round
// that is still empty ends the Turn with *EmptyReplyError, never Reply "".
//
// A length cut is NOT an empty reply. An empty round with finish "length" and
// no reasoning is a provider ceiling, and the overflow path (overflow.go) owns
// it: the host decides what to shed. A repair note there would ask a model that
// ran out of room to answer in the same room.
//
// A native reason that is not a clean stop (MALFORMED_FUNCTION_CALL is one) is
// first resent unchanged, with backoff (providerfail.go); this repair runs only
// once those resends are spent, or with Session.MaxProviderFailRetries < 0.
//
// As in the reasoning cut, no empty assistant entry is persisted:
// llama.cpp refuses two consecutive assistant messages and some providers
// refuse empty assistant content. The notification is the record.

import (
	"errors"
	"fmt"
)

// EmptyReplyTag marks the notification entries this produces, so a host can
// render them apart from real notifications.
const EmptyReplyTag = "empty-reply"

// ErrEmptyReply matches an *EmptyReplyError under errors.Is.
var ErrEmptyReply = errors.New("agent: the model returned an empty reply")

// EmptyReplyError is returned by Turn (TurnResult.Reason == TurnEmptyReply)
// when a round had no text, no tool call and no reasoning, and the repair did
// not recover it: the repair round was empty too, the repair was already spent
// in this Turn, or Session.NoReasoningCutRepair is set.
type EmptyReplyError struct {
	// FinishReason and NativeFinishReason are the empty round's provider
	// finish_reason and native_finish_reason, verbatim ("" when not sent).
	FinishReason       string
	NativeFinishReason string
}

func (e *EmptyReplyError) Error() string {
	return fmt.Sprintf("model returned an empty reply (finish=%s, native=%s)", e.FinishReason, e.NativeFinishReason)
}

// Is makes errors.Is(err, ErrEmptyReply) true for any *EmptyReplyError.
func (e *EmptyReplyError) Is(target error) bool { return target == ErrEmptyReply }

// repairKindT says which empty round a pending repair answers. It decides which
// error a still-empty repair round returns, and whether the round sends
// reasoning off (only the reasoning-cut repair does).
type repairKindT int

const (
	repairReasoning repairKindT = iota
	repairEmpty
)

func emptyReplyNotice(finish string) string {
	if finish == "" {
		finish = "none"
	}
	return "Your last reply was empty: no text and no tool call (finish_reason: " + finish + "). Reply again."
}
