package llm

import (
	"fmt"
	"strings"
	"time"
)

// RetryKind names WHY the client is backing off, so a UI can say something more
// useful than "retrying".
type RetryKind string

const (
	// RetryTransport — the request never got a response: the gateway is down,
	// restarting, or unreachable.
	RetryTransport RetryKind = "transport"
	// Retry429 — backpressure. The server will serve, just not yet.
	Retry429 RetryKind = "429"
	// Retry5xx — the upstream answered with a server error.
	Retry5xx RetryKind = "5xx"
	// RetryRecovered — a later attempt succeeded. Fires once, so a UI that put up
	// a "retrying" banner knows to take it down.
	RetryRecovered RetryKind = "recovered"
	// RetryGiveUp — the client stopped retrying (attempt cap, wall-clock budget,
	// or a fault that no amount of waiting changes). The call's error follows.
	RetryGiveUp RetryKind = "giveup"
)

// RetryEvent is one report from the client's backoff loop, delivered to
// Client.OnRetry.
//
// It exists because the retry logic was already correct and completely
// INVISIBLE: every decision was a log.Printf, and a caller whose logs go
// anywhere but the user's screen (a TUI, a daemon, anything with a log file)
// shows a frozen cursor for up to the whole retry budget and then one error
// line. The wait is the part the user needs narrated — it is the only time the
// agent is doing nothing and the only time a human might reasonably intervene.
//
// Every field is filled on a best-effort basis; a consumer should render Reason
// (always set) and treat the rest as detail.
type RetryEvent struct {
	Kind RetryKind
	// Attempt is the 1-based attempt this event describes: the one that just
	// failed, or — for RetryRecovered — the one that succeeded.
	Attempt int
	// Status is the HTTP status, 0 on the transport path (no response at all).
	Status int
	// Err is the transport error, nil on the status paths.
	Err error
	// Body is the first line of the server's own message, when it sent one. The
	// single most actionable field on a 5xx (see the "input too large" case in
	// postWithRetry) and empty far more often than not.
	Body string
	// Delay is how long the client will wait before the next attempt. 0 on
	// RetryRecovered / RetryGiveUp.
	Delay time.Duration
	// Elapsed is time spent in this call's retry loop so far.
	Elapsed time.Duration
	// Budget is the wall-clock ceiling for the whole loop (Client.RetryBudget).
	Budget time.Duration
	// Attempts5xx / Max5xx expose the SEPARATE attempt cap that bounds the 5xx
	// path, which is what actually ends most upstream outages — the budget rarely
	// gets a chance to. Both 0 outside the 5xx path.
	Attempts5xx, Max5xx int
	// BP carries the queue detail a fair-share proxy attached to a 429 — slots
	// busy, requests ahead, why. Non-nil only on the 429 path, and only worth
	// rendering when BP.Queued().
	BP *Backpressure
	// ServerAsked is true when Delay is what the SERVER instructed (Retry-After or
	// a backpressure body) rather than our own exponential schedule. Worth saying
	// out loud: it means the wait is a real estimate, not a guess.
	ServerAsked bool
	// Reason is a human-readable sentence: what failed and what happens next.
	Reason string
}

// Unbounded reports whether the retry budget is effectively "until the caller
// gives up" (Client.RetryBudget < 0), so a UI renders "no limit" instead of a
// nonsense century.
func (e RetryEvent) Unbounded() bool { return e.Budget >= unboundedRetryBudget }

// String renders the event as one line, the same shape the client's own log
// lines use.
func (e RetryEvent) String() string {
	var b strings.Builder
	b.WriteString(e.Reason)
	switch e.Kind {
	case RetryRecovered, RetryGiveUp:
	default:
		// Say WHOSE number the wait is: a server-supplied Retry-After is an
		// estimate from the thing that knows, our schedule is a guess.
		if e.ServerAsked {
			fmt.Fprintf(&b, " — the provider asked for %s", e.Delay.Round(100*time.Millisecond))
		} else {
			fmt.Fprintf(&b, " — retrying in %s", e.Delay.Round(100*time.Millisecond))
		}
	}
	if e.Elapsed > 0 {
		if e.Unbounded() {
			fmt.Fprintf(&b, " (%s elapsed)", e.Elapsed.Round(time.Second))
		} else {
			fmt.Fprintf(&b, " (%s of %s)", e.Elapsed.Round(time.Second), e.Budget)
		}
	}
	return b.String()
}

// retryReason classifies an event into a sentence. Kept separate from
// TransientUpstreamReason because that one classifies an ERROR STRING after the
// fact, while here the client knows exactly which branch it took.
func retryReason(e RetryEvent) string {
	switch e.Kind {
	case RetryRecovered:
		return fmt.Sprintf("provider recovered on attempt %d", e.Attempt)
	case RetryGiveUp:
		return "gave up retrying the provider"
	case RetryTransport:
		return fmt.Sprintf("cannot reach the provider (attempt %d): %v", e.Attempt, e.Err)
	case Retry429:
		// Prefer the proxy's own account of the queue. "4/4 slots busy, 2 waiting
		// ahead, queue-timeout" tells a user whether to wait or go do something
		// else; "429" tells them nothing.
		if e.BP != nil && e.BP.Queued() {
			return fmt.Sprintf("provider at capacity — %s (attempt %d)", e.BP, e.Attempt)
		}
		if e.BP != nil && e.BP.Reason != "" {
			return fmt.Sprintf("provider is holding the request off — %s (attempt %d)", e.BP.Reason, e.Attempt)
		}
		return fmt.Sprintf("provider is saturated — 429 backpressure (attempt %d)", e.Attempt)
	case Retry5xx:
		s := fmt.Sprintf("provider returned %d (5xx attempt %d/%d)", e.Status, e.Attempts5xx, e.Max5xx)
		if e.Body != "" {
			s += ": " + e.Body
		}
		return s
	}
	return fmt.Sprintf("provider retry (attempt %d)", e.Attempt)
}

// emitRetry delivers ev to OnRetry, filling Reason when the caller left it
// empty. Nil-safe on both the client and the hook, so every call site is a
// one-liner with no guard.
func (c *Client) emitRetry(ev RetryEvent) {
	if c == nil || c.OnRetry == nil {
		return
	}
	if ev.Reason == "" {
		ev.Reason = retryReason(ev)
	}
	c.OnRetry(ev)
}
