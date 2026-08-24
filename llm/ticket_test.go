package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// A fair-share proxy that hands out a ticket must get it back on the retry.
//
// Without the echo the server sees a stream of strangers: the caller that has
// been waiting 40 seconds is indistinguishable from one arriving now, and on a
// one-slot backend that is how a caller starves — every return is outranked by
// somebody's first attempt.
func TestTicketIsEchoedOnRetry(t *testing.T) {
	var mu sync.Mutex
	var seen []string // the ticket header on each attempt, in order

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("X-Corrallm-Request-Id"))
		n := len(seen)
		mu.Unlock()

		if n == 1 {
			// First attempt: turn it away and issue a ticket, the way corrallm
			// does — header and body both.
			w.Header().Set("Retry-After", "1")
			w.Header().Set("X-Corrallm-Ticket", "tkt_abc123")
			w.Header().Set("X-RateLimit-Capacity", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"reason": "queue-timeout", "retry_after": 1, "ticket": "tkt_abc123",
				},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	c.RetryBudget = 10 * time.Second
	resp, err := c.postWithRetry(context.Background(), srv.URL, []byte(`{}`), "")
	if err != nil {
		t.Fatalf("postWithRetry: %v", err)
	}
	defer resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(seen) < 2 {
		t.Fatalf("want at least 2 attempts, got %d", len(seen))
	}
	if seen[0] != "" {
		t.Errorf("first attempt already carried a ticket %q; we had not been given one", seen[0])
	}
	if seen[1] != "tkt_abc123" {
		t.Errorf("retry carried %q, want the ticket the server issued", seen[1])
	}
}

// The ticket comes off either the header or the body, because an intermediary
// can rewrite one and not the other.
func TestBackpressureReadsTheTicketFromEitherPlace(t *testing.T) {
	h := http.Header{}
	h.Set("X-Corrallm-Ticket", "from-header")
	bp, _ := backpressureFrom(h, []byte(`{"error":{"ticket":"from-body"}}`))
	if bp.Ticket != "from-header" {
		t.Errorf("ticket = %q, want the header to win", bp.Ticket)
	}

	bp, _ = backpressureFrom(http.Header{}, []byte(`{"error":{"ticket":"from-body"}}`))
	if bp.Ticket != "from-body" {
		t.Errorf("ticket = %q, want the body's when no header carried one", bp.Ticket)
	}

	bp, _ = backpressureFrom(http.Header{}, []byte(`{"error":{"reason":"rejected"}}`))
	if bp.Ticket != "" {
		t.Errorf("invented a ticket %q from a 429 that issued none", bp.Ticket)
	}
}
