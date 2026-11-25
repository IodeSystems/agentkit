package agent

// Provider-failure resend.
//
// OpenRouter maps an upstream failure it does not recognize to finish_reason
// "stop" and puts the real reason in native_finish_reason. Seen live: Gemini
// streams reasoning only, then fails with MALFORMED_RESPONSE, and the round
// arrives as finish "stop", native "MALFORMED_RESPONSE", no text, no tool call.
// To the rest of Turn that is a reasoning-only round, and the reasoning-cut
// repair (reasoningcut.go) would answer it with a notice and a round with
// reasoning off. That is the wrong cure: the model did nothing wrong, the
// provider dropped the answer. The fix is to send the SAME request again.
//
// So a round that is empty (no non-whitespace text, no tool call; reasoning
// does NOT count, since it is never the answer), with finish "stop" and a
// native reason that is not a clean stop, is discarded and resent: nothing is
// persisted, no notice is appended, the opts are those the round sent
// (a pending repair round stays pending, and is resent as the repair round).
// It does not spend the Turn's one empty-round repair. Resends back off
// exponentially (1s doubling, each sleep capped at 5m, additive jitter) and are
// bounded per Turn by Session.MaxProviderFailRetries (total, not consecutive).
// A reason that repeats stops them sooner: after 2 resends in a row that came
// back with the same native reason (case-insensitive), the third such round is
// not resent. A fault that survives two resends unchanged is not transient, and
// the full chain would only delay the repair by minutes. A different reason
// starts the streak again; so does any round that is not a provider failure.
// When resends stop, the round goes to the existing repairs unchanged.
//
// MALFORMED_FUNCTION_CALL is NOT resent while Session.MaxMalformedCallRepairs
// is on: the model wrote a bad call, and the same request gets the same bad
// call. It has its own repair (malformedcall.go).
//
// The predicate is an ALLOWLIST of clean native reasons, on purpose. A clean
// reason missing from it costs a few resends of a round that was empty anyway;
// a failure reason missing from a denylist ends the Turn silently with a
// repair notice blaming the model. OpenRouter does not publish the set of
// native reasons it passes through, so the list is the one pi uses (commit
// 9b911b4, packages/ai/src/utils/openrouter-finish-reason.ts). Deliberately NOT
// on it: Gemini "OTHER" and DeepSeek "insufficient_system_resource", which the
// providers document as failures. A round WITH visible text and a failure
// native reason is kept as an ordinary reply: the text is the answer.

import (
	"context"
	"math/rand"
	"strings"
	"time"
)

// cleanNativeStops are native_finish_reason values that mean the model stopped
// on its own. Lower case; compared case-insensitively.
var cleanNativeStops = map[string]bool{
	"stop":          true,
	"end_turn":      true,
	"stop_sequence": true,
	"eos":           true,
	"complete":      true,
	"completed":     true,
	"pause_turn":    true,
	"compaction":    true,
	"end_of_stream": true,
}

// providerFailed reports whether an empty round (the caller has checked no
// visible text and no tool call) is a provider failure dressed as a stop: finish
// "stop" and a native reason that is present and not a clean stop. An empty
// native reason is not evidence of anything, so it is not a failure.
func providerFailed(finish, native string) bool {
	if finish != "stop" || native == "" {
		return false
	}
	return !cleanNativeStops[strings.ToLower(native)]
}

const (
	defaultProviderFailRetries = 10
	providerFailBase           = time.Second
	providerFailCap            = 5 * time.Minute
	// maxProvFailStreak is how many resends in a row one native reason gets.
	maxProvFailStreak = 2
)

// providerFailJitter is the additive jitter fraction, the same rule as llm's
// jittered(): add a random [0, fraction) of the sleep, never subtract. Many
// sessions behind one proxy hit the same upstream fault at once; an exact
// schedule would wake them together.
var providerFailJitter = 0.20

// providerFailDelay is the sleep before resend number attempt (1-based):
// 1s, 2s, 4s, ... capped at 5m, plus jitter.
func providerFailDelay(attempt int) time.Duration {
	d := providerFailCap
	if attempt-1 < 20 { // 1s<<20 is far past the cap; avoids overflow
		if b := providerFailBase << (attempt - 1); b < d {
			d = b
		}
	}
	if providerFailJitter > 0 {
		d += time.Duration(rand.Float64() * providerFailJitter * float64(d))
	}
	return d
}

// sleepCtx waits d or until ctx is done, whichever is first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ProviderFailRetry is one resend of a round the provider failed, delivered to
// Session.OnProviderFailRetry BEFORE the backoff sleep. A host shows it: a full
// chain of resends waits about 13.5 minutes, and without it the session looks
// frozen.
type ProviderFailRetry struct {
	// Attempt is the 1-based resend about to happen; Max is the Turn's cap.
	Attempt, Max int
	// FinishReason and NativeFinishReason are the failed round's, verbatim
	// (FinishReason is always "stop").
	FinishReason, NativeFinishReason string
	// Wait is the sleep before the resend, jitter included.
	Wait time.Duration
}
