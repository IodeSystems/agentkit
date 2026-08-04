package llm

// OnRetry tests. The retry loop was correct and mute: a caller whose logs don't
// land on the user's screen showed nothing for the whole retry budget and then
// one error. These assert the loop NARRATES itself — every wait, the recovery,
// and the give-up — because that stream is what a UI renders.

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestOnRetry_429ThenRecovered — a 429 followed by a 200 emits the backoff event
// and then exactly one "recovered", so a banner goes up and comes down.
func TestOnRetry_429ThenRecovered(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `ok`)
	}))
	defer srv.Close()

	var evs []RetryEvent
	c := NewClient(srv.URL, "", "m")
	c.OnRetry = func(ev RetryEvent) { evs = append(evs, ev) }
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()

	if len(evs) != 2 {
		t.Fatalf("events = %d (%v); want 2 (429 + recovered)", len(evs), evs)
	}
	if evs[0].Kind != Retry429 || evs[0].Attempt != 1 || evs[0].Status != 429 {
		t.Errorf("first event = %+v; want 429 on attempt 1", evs[0])
	}
	if evs[0].Reason == "" {
		t.Error("Reason is empty; a UI has nothing to render")
	}
	if evs[1].Kind != RetryRecovered || evs[1].Attempt != 2 {
		t.Errorf("second event = %+v; want recovered on attempt 2", evs[1])
	}
}

// TestOnRetry_429CarriesQueueDetail — corrallm's proxy answers a saturated
// backend with the whole picture (Retry-After + capacity/in-flight/waiting +
// a reason). All of it must reach the hook: "4/4 slots busy, 2 waiting ahead,
// queue-timeout" tells a user whether to wait; "429" tells them nothing.
func TestOnRetry_429CarriesQueueDetail(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) > 1 {
			w.WriteHeader(http.StatusOK)
			return
		}
		h := w.Header()
		h.Set("Retry-After", "1")
		h.Set("X-RateLimit-Capacity", "4")
		h.Set("X-RateLimit-InFlight", "4")
		h.Set("X-RateLimit-Waiting", "2")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"backend at capacity; retry after backoff",`+
			`"type":"backpressure","reason":"queue-timeout","retry_after":1,`+
			`"capacity":4,"in_flight":4,"waiting":2}}`)
	}))
	defer srv.Close()

	var evs []RetryEvent
	c := NewClient(srv.URL, "", "m")
	c.OnRetry = func(ev RetryEvent) { evs = append(evs, ev) }
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()

	if len(evs) == 0 {
		t.Fatal("no retry events")
	}
	ev := evs[0]
	if ev.BP == nil {
		t.Fatal("BP is nil; the queue detail was dropped")
	}
	if ev.BP.Capacity != 4 || ev.BP.InFlight != 4 || ev.BP.Waiting != 2 {
		t.Errorf("queue = %d/%d busy, %d waiting; want 4/4 and 2", ev.BP.InFlight, ev.BP.Capacity, ev.BP.Waiting)
	}
	if ev.BP.Reason != "queue-timeout" {
		t.Errorf("BP.Reason = %q; want queue-timeout", ev.BP.Reason)
	}
	if !ev.ServerAsked {
		t.Error("ServerAsked = false; the server sent Retry-After, so the wait is its estimate")
	}
	// The server's 1s, not our own schedule — plus additive jitter, which is
	// what keeps a queue-timeout from marching every rejected caller back in on
	// the same tick. Never below the server's ask; never more than the fraction
	// above it. See retryJitterFraction.
	if maxDelay := time.Second + time.Duration(retryJitterFraction*float64(time.Second)); ev.Delay < time.Second || ev.Delay > maxDelay {
		t.Errorf("Delay = %s; want the server's 1s plus jitter (≤ %s), not our schedule", ev.Delay, maxDelay)
	}
	for _, want := range []string{"4/4 slots busy", "2 waiting ahead", "queue-timeout"} {
		if !strings.Contains(ev.Reason, want) {
			t.Errorf("Reason = %q; want it to mention %q", ev.Reason, want)
		}
	}
	if !strings.Contains(ev.String(), "provider asked for") {
		t.Errorf("String() = %q; want it to credit the server's own wait", ev.String())
	}
}

// TestBackpressureFrom_PlainProviderReportsNothing — a stock OpenAI-style 429
// carries no queue detail, and the zero values must not be rendered as a queue
// of zero slots.
func TestBackpressureFrom_PlainProviderReportsNothing(t *testing.T) {
	bp, ok := backpressureFrom(http.Header{}, []byte(`{"error":{"message":"rate limited"}}`))
	if ok {
		t.Error("ok = true; nothing said when to retry")
	}
	if bp.Queued() {
		t.Errorf("Queued() = true for %+v; want no queue claim", bp)
	}
	if bp.String() != "" {
		t.Errorf("String() = %q; want empty", bp.String())
	}
}

// TestOnRetry_5xxCarriesServerMessage — the 5xx path reports the server's own
// first line and the attempt cap, then a giveup. The body is the actionable part
// (see the "input too large" case), so it must reach the hook, not just the log.
func TestOnRetry_5xxCarriesServerMessage(t *testing.T) {
	fastBackoff(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, "upstream model is not loaded")
	}))
	defer srv.Close()

	var evs []RetryEvent
	c := NewClient(srv.URL, "", "m")
	c.Retry5xxAttempts = 2
	c.OnRetry = func(ev RetryEvent) { evs = append(evs, ev) }
	if _, err := c.postChatWithRetry(context.Background(), []byte(`{}`), ""); err == nil {
		t.Fatal("postChatWithRetry: want error after the 5xx cap")
	}
	if len(evs) == 0 {
		t.Fatal("no retry events")
	}
	first := evs[0]
	if first.Kind != Retry5xx || first.Status != 502 {
		t.Errorf("first event = %+v; want 5xx/502", first)
	}
	if !strings.Contains(first.Body, "not loaded") {
		t.Errorf("Body = %q; want the server's message", first.Body)
	}
	if first.Max5xx != 2 || first.Attempts5xx != 1 {
		t.Errorf("attempts = %d/%d; want 1/2", first.Attempts5xx, first.Max5xx)
	}
	last := evs[len(evs)-1]
	if last.Kind != RetryGiveUp {
		t.Errorf("last event = %+v; want giveup", last)
	}
	if !strings.Contains(last.Reason, "gave up") {
		t.Errorf("giveup Reason = %q; want it to say so", last.Reason)
	}
}

// TestOnRetry_TransportDown — nothing is listening, so the transport path fires
// with Err set and Status 0. This is the case the user actually hits ("just a
// generic single failure to connect"): the events are the only sign the client
// is waiting rather than hung.
func TestOnRetry_TransportDown(t *testing.T) {
	fastBackoff(t)
	// A closed listener's port: connection refused, deterministically.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	var evs []RetryEvent
	c := NewClient(url, "", "m")
	ctx, cancel := context.WithCancel(context.Background())
	c.OnRetry = func(ev RetryEvent) {
		evs = append(evs, ev)
		if len(evs) >= 3 {
			cancel() // seen enough; don't sit out the whole budget
		}
	}
	_, _ = c.postChatWithRetry(ctx, []byte(`{}`), "")
	cancel()

	if len(evs) < 2 {
		t.Fatalf("events = %d; want the transport path to report each wait", len(evs))
	}
	if evs[0].Kind != RetryTransport || evs[0].Status != 0 || evs[0].Err == nil {
		t.Errorf("first event = %+v; want transport with Err set and no status", evs[0])
	}
	if evs[0].Delay <= 0 {
		t.Error("Delay = 0; a UI cannot count down without it")
	}
	if evs[1].Attempt <= evs[0].Attempt {
		t.Errorf("attempts not advancing: %d then %d", evs[0].Attempt, evs[1].Attempt)
	}
}

// TestOnRetry_NoHookNoPanic — the hook is optional and every call site is
// unguarded, so the nil path is the one that must not break.
func TestOnRetry_NoHookNoPanic(t *testing.T) {
	fastBackoff(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	c.Retry5xxAttempts = 1
	if _, err := c.postChatWithRetry(context.Background(), []byte(`{}`), ""); err == nil {
		t.Fatal("want error")
	}
}
