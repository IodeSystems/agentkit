package llm

import (
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const (
	queueTimeoutEvent = "data: {\"error\":{\"message\":\"backend at capacity; retry after backoff\",\"type\":\"backpressure\",\"reason\":\"queue-timeout\",\"code\":429,\"status\":429,\"retry_after\":0.1}}\n\ndata: [DONE]\n\n"
	noReasonEvent     = "data: {\"error\":{\"message\":\"backend at capacity; retry after backoff\",\"type\":\"backpressure\",\"code\":429,\"retry_after\":0.1}}\n\ndata: [DONE]\n\n"
)

// corrallm sends a queue-timeout as an error event inside a 200 stream. It must
// follow the 429 rule: immediate retries up to maxQueueTimeoutStreak in a row,
// then one real wait (the event's retry_after), then success.
func TestStreamError_QueueTimeoutEventRetriesImmediatelyLikeA429(t *testing.T) {
	saveStreak, saveJitter := maxQueueTimeoutStreak, immediateRetryJitterCeiling
	maxQueueTimeoutStreak, immediateRetryJitterCeiling = 3, 10*time.Millisecond
	t.Cleanup(func() { maxQueueTimeoutStreak, immediateRetryJitterCeiling = saveStreak, saveJitter })
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, queueTimeoutEvent, queueTimeoutEvent, queueTimeoutEvent, queueTimeoutEvent, okStream)
	c := NewClient(srv.URL, "", "m")
	var evs []RetryEvent
	c.OnRetry = func(e RetryEvent) { evs = append(evs, e) }
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v", text, errs)
	}
	if hits.Load() != 5 || len(evs) != 5 {
		t.Fatalf("hits=%d events=%d (%+v); want 5 and 5", hits.Load(), len(evs), evs)
	}
	for i := 0; i < 3; i++ {
		e := evs[i]
		if e.Kind != RetryStream || e.Delay > 20*time.Millisecond || e.ServerAsked || !strings.Contains(e.Reason, "immediately") {
			t.Errorf("event %d = %+v; want an immediate RetryStream", i, e)
		}
	}
	if e := evs[3]; e.Kind != RetryStream || e.Delay < 100*time.Millisecond || !e.ServerAsked {
		t.Errorf("event 3 = %+v; want a real wait honoring retry_after once the streak is spent", e)
	}
	if evs[4].Kind != RetryRecovered {
		t.Errorf("last event = %v; want recovered", evs[4].Kind)
	}
}

// Without reason "queue-timeout" the event keeps the retry_after sleep.
func TestStreamError_EventWithoutQueueTimeoutReasonKeepsRetryAfterSleep(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := sseServer(t, &hits, noReasonEvent, okStream)
	c := NewClient(srv.URL, "", "m")
	var evs []RetryEvent
	c.OnRetry = func(e RetryEvent) { evs = append(evs, e) }
	if text, errs := drainStream(t, c); text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v", text, errs)
	}
	if len(evs) == 0 || evs[0].Delay < 100*time.Millisecond || !evs[0].ServerAsked || strings.Contains(evs[0].Reason, "immediately") {
		t.Errorf("events = %#v; want first to sleep >= retry_after", evs)
	}
}

// A proxy that sends status but no code is still transient at 429.
func TestStreamError_StatusStandsInForMissingCode(t *testing.T) {
	se := parseStreamError([]byte(`{"message":"busy","reason":"queue-timeout","status":429}`))
	if se == nil || se.Reason != "queue-timeout" || se.Status != 429 || !se.Retryable() {
		t.Errorf("got %+v", se)
	}
}
