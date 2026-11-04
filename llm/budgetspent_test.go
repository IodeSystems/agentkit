package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// A 402 (spent calendar budget) used to be answered as a 429 and slept on for
// retryAfterCeiling before the caller could fall back. It must come back at
// once: one request, no sleep, a typed error carrying the parsed body, and one
// RetryGiveUp event with Status 402 for hosts that fall back from the hook.
func TestBudgetSpent402IsNotRetriedOrSleptOn(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"error":{"message":"key k's budget per calendar day is spent","type":"budget_exhausted","retry_after":40000,`+
			`"budgets":[{"scope":"key:k","owner":"key k's budget","window":"usd/calendar-day","used":5.2,"limit":5,"room_at":"2026-11-05T00:00:00Z","summary":"spent $5.20 of $5"}]}}`)
	}))
	defer srv.Close()

	var evs []RetryEvent
	c := NewClient(srv.URL, "", "m")
	c.OnRetry = func(ev RetryEvent) { evs = append(evs, ev) }
	t0 := time.Now()
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	if resp != nil {
		resp.Body.Close()
		t.Fatal("got a response; want an error")
	}
	if d := time.Since(t0); d > 2*time.Second {
		t.Errorf("took %s; a 402 must not sleep", d)
	}
	if n := hits.Load(); n != 1 {
		t.Errorf("server hits = %d; want 1", n)
	}
	var spent *BudgetSpentError
	if !errors.As(err, &spent) {
		t.Fatalf("err = %v (%T); want *BudgetSpentError", err, err)
	}
	if spent.Status != 402 || spent.BP.Type != "budget_exhausted" || spent.BP.AskedWait != 40000*time.Second {
		t.Errorf("spent = %+v", spent)
	}
	if len(spent.BP.Budgets) != 1 || spent.BP.Budgets[0].Window != "usd/calendar-day" ||
		spent.BP.Budgets[0].Used != 5.2 || spent.BP.Budgets[0].RoomAt != "2026-11-05T00:00:00Z" {
		t.Errorf("budgets = %+v", spent.BP.Budgets)
	}
	if TransientUpstream(err) {
		t.Errorf("TransientUpstream(%v) = true; a spent budget is not the server going away", err)
	}
	if len(evs) != 1 {
		t.Fatalf("events = %d (%+v); want 1", len(evs), evs)
	}
	ev := evs[0]
	if ev.Kind != RetryGiveUp || ev.Status != 402 || ev.Delay != 0 || ev.BP == nil ||
		ev.BP.Type != "budget_exhausted" || len(ev.BP.Budgets) != 1 {
		t.Errorf("event = %+v", ev)
	}
}

// allocation_spent carries no retry_after: AskedWait stays zero.
func TestBudgetSpent402WithoutRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"error":{"message":"allocation spent","type":"allocation_spent","budgets":[]}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	_, err := c.postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	var spent *BudgetSpentError
	if !errors.As(err, &spent) {
		t.Fatalf("err = %v; want *BudgetSpentError", err)
	}
	if spent.BP.Type != "allocation_spent" || spent.BP.AskedWait != 0 {
		t.Errorf("bp = %+v", spent.BP)
	}
}
