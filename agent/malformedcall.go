package agent

// Malformed-call repair.
//
// Gemini (seen through OpenRouter) ends a round with native_finish_reason
// MALFORMED_FUNCTION_CALL when the model wrote a tool call the provider could
// not parse. The round arrives empty: no text, no tool call. OpenRouter maps
// it to finish "stop" (the provider-failure resend in providerfail.go would
// take it) or to finish "error" (the empty-reply repair would take it). Both
// are the wrong cure. Seen live: the identical request gets the identical bad
// call, so ten resends only delay the failure by about 13.5 minutes, and the
// generic "Reply again." notice does not say what was wrong.
//
// The model made the error, so the model is told what it was. An empty round
// with this native reason (any case, any finish) appends a notification tagged
// MalformedCallTag that names the cause and the usual fixes, and runs one more
// round with the host's opts UNCHANGED (reasoning is not the cause). The causes
// reported for Gemini: a Python-style call (default_api.fn(...)) instead of a
// JSON object, unescaped newlines or quotes inside a string, and very long
// arguments (over about 1000 characters).
//
// Budget: Session.MaxMalformedCallRepairs per Turn (0 → 3, negative → off),
// its OWN budget, apart from the one repair the reasoning cut and the empty
// reply share and from the provider-failure resends. A malformed call is a
// thing a model can fix on a second or third try; one try is too few. When
// the budget is spent the Turn ends TurnEmptyReply with *EmptyReplyError
// (carrying the native reason) at once: the shared repair's "Reply again." is
// weaker than the notice the model already ignored three times.
//
// Off (negative) is the behavior before this file: the round is a provider
// failure and is resent, then goes to the shared repair. NoReasoningCutRepair
// does not turn this off; MaxMalformedCallRepairs is its own switch.
//
// A round WITH visible text is a reply, whatever its native reason says. No
// empty assistant entry is persisted, for the reasons in emptyreply.go.

import "strings"

// MalformedCallTag marks the notification entries this produces, so a host can
// render them apart from real notifications.
const MalformedCallTag = "malformed-call"

const defaultMalformedCallRepairs = 3

// malformedCall reports whether a native_finish_reason says the model wrote a
// tool call the provider could not parse.
func malformedCall(native string) bool {
	return strings.EqualFold(native, "MALFORMED_FUNCTION_CALL")
}

func malformedCallNotice(native string) string {
	return "Your last tool call was malformed (native_finish_reason " + native + "). " +
		"Call the tool by its exact name, with the arguments as one JSON object that matches its schema " +
		"— not code such as default_api.fn(...). Escape newlines and quotes inside strings. " +
		"Keep long text arguments short or split them."
}
