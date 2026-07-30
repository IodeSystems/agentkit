package llm

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Backpressure is the detail a fair-share proxy attaches to a 429.
//
// corrallm's proxy answers a saturated backend with a fully actionable 429 —
// Retry-After plus X-RateLimit-Capacity / -InFlight / -Waiting headers and a JSON
// body carrying the same numbers and a reason ("rejected", "queue-timeout",
// "spill", "exhausted"). Until now the client read the one field it needed to
// sleep and discarded the rest, so a user waiting out a queue was told "429" when
// the server had said "4 of 4 slots busy, 2 requests ahead of you, come back in
// 10s". That difference is the whole gap between a hang and a wait.
//
// Every field is optional: a plain OpenAI-style 429 fills in nothing but
// RetryAfter, and a consumer must treat zeros as "not reported".
type Backpressure struct {
	// Reason is the proxy's own classification. corrallm: "rejected" (queue full),
	// "queue-timeout" (waited and never got a slot), "spill" (routed elsewhere),
	// "exhausted" (budget/quota gone).
	Reason string
	// RetryAfter is when the server says to come back. Ok=false from
	// backpressureFrom means it did not say.
	RetryAfter time.Duration
	// Capacity / InFlight / Waiting describe the queue: total slots, slots busy,
	// and requests already waiting. Zero = not reported.
	Capacity, InFlight, Waiting int
	// Message is the server's human-readable line, when it sent one.
	Message string
}

// Queued reports whether the server described a queue we are standing in, i.e.
// whether the capacity numbers are worth showing a user.
func (b Backpressure) Queued() bool { return b.Capacity > 0 || b.Waiting > 0 }

// String renders the queue state as a phrase, empty when nothing was reported.
func (b Backpressure) String() string {
	var parts []string
	if b.Capacity > 0 {
		parts = append(parts, strconv.Itoa(b.InFlight)+"/"+strconv.Itoa(b.Capacity)+" slots busy")
	}
	if b.Waiting > 0 {
		parts = append(parts, strconv.Itoa(b.Waiting)+" waiting ahead")
	}
	if b.Reason != "" {
		parts = append(parts, b.Reason)
	}
	return strings.Join(parts, ", ")
}

// backpressureFrom reads whatever queue detail a 429 carried. ok reports whether
// the server gave an explicit Retry-After (header or body) — the same signal
// retryAfterFrom returns, so the caller keeps its own schedule when false while
// still getting any capacity numbers that were present.
//
// Headers win over the body for the counts: they are set unconditionally by the
// proxy, while the body can be replaced by an intermediary.
func backpressureFrom(h http.Header, body []byte) (Backpressure, bool) {
	var bp Backpressure
	bp.Capacity = headerInt(h, "X-RateLimit-Capacity")
	bp.InFlight = headerInt(h, "X-RateLimit-InFlight")
	bp.Waiting = headerInt(h, "X-RateLimit-Waiting")

	var parsed struct {
		RetryAfter int `json:"retry_after"`
		Error      struct {
			Message    string `json:"message"`
			Reason     string `json:"reason"`
			RetryAfter int    `json:"retry_after"`
			Capacity   int    `json:"capacity"`
			InFlight   int    `json:"in_flight"`
			Waiting    int    `json:"waiting"`
		} `json:"error"`
	}
	if len(body) > 0 && json.Unmarshal(body, &parsed) == nil {
		bp.Reason = parsed.Error.Reason
		bp.Message = parsed.Error.Message
		if bp.Capacity == 0 {
			bp.Capacity = parsed.Error.Capacity
		}
		if bp.InFlight == 0 {
			bp.InFlight = parsed.Error.InFlight
		}
		if bp.Waiting == 0 {
			bp.Waiting = parsed.Error.Waiting
		}
	}
	d, ok := retryAfterFrom(h, body)
	bp.RetryAfter = d
	return bp, ok
}

// headerInt reads a non-negative integer header, 0 when absent or unparseable.
func headerInt(h http.Header, name string) int {
	v := strings.TrimSpace(h.Get(name))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0
	}
	return n
}
