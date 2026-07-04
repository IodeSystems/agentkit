package llm

// Backoff-on-429 tests for postChatWithRetry. The helper hides
// provider rate-limit churn from every consumer of ChatStream / Chat
// — without it, a single 429 fails the whole Turn (and on iode's
// free tier 429s are routine). Cover:
//
//   - 429 → 200 within one retry returns the 200 body
//   - Retry-After header (numeric, ≤ retryMaxBackoff) shapes the sleep
//   - Repeated 429 surfaces a clean "rate limited" error after maxAttempts
//   - ctx cancellation mid-backoff aborts cleanly (no goroutine leak)
//   - Non-retryable status (e.g. 500) passes through immediately

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fastBackoff swaps the package-level schedule for a millisecond
// version so tests don't sit for real wall-clock seconds. Returns a
// cleanup the caller must defer to restore the defaults.
func fastBackoff(t *testing.T) {
	t.Helper()
	saveInit, saveMax := retryInitialBackoffSet(10*time.Millisecond, 50*time.Millisecond)
	t.Cleanup(func() {
		retryInitialBackoffSet(saveInit, saveMax)
	})
}

// retryInitialBackoffSet swaps the package's backoff vars and returns
// the prior values for restore. Test-only — production never mutates.
func retryInitialBackoffSet(initial, max time.Duration) (time.Duration, time.Duration) {
	prevI, prevM := retryInitialBackoff, retryMaxBackoff
	retryInitialBackoff, retryMaxBackoff = initial, max
	return prevI, prevM
}

// TestPostChatWithRetry_RetriesOn429 — first response is 429, second
// is 200. Helper must transparently surface the 200 to the caller.
func TestPostChatWithRetry_RetriesOn429(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `ok`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d; want 200", resp.StatusCode)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("server hits = %d; want 2 (one 429, one 200)", got)
	}
}

// TestPostChatWithRetry_KeepsRetryingUntilCtx — every response is
// 429; the helper does NOT give up on its own (busy-box reality:
// another minute is another shot). Only ctx cancellation breaks
// the loop. We cancel after a short window and assert ≥ several
// retries happened.
func TestPostChatWithRetry_KeepsRetryingUntilCtx(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()
	_, err := c.postChatWithRetry(ctx, []byte(`{}`), "test")
	if err == nil {
		t.Fatal("postChatWithRetry: nil error after ctx timeout; want ctx error")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v; want ctx.DeadlineExceeded / Canceled", err)
	}
	// fastBackoff is initial=10ms, max=50ms. In 250ms we should get
	// at least ~5 retries; assert ≥ 3 so we're not flaky on slow CI.
	if got := hits.Load(); got < 3 {
		t.Errorf("server hits = %d; want ≥ 3 retries within the ctx window", got)
	}
}

// TestPostChatWithRetry_RespectsRetryAfter — server sends Retry-After:
// 1; the helper waits at least that long (not retryInitialBackoff,
// which we've made 10ms in fastBackoff) before retrying. We don't
// assert an exact duration (system jitter), only that we waited
// ≥ the header value.
func TestPostChatWithRetry_RespectsRetryAfter(t *testing.T) {
	// Keep the test fast: cap Retry-After to ~50ms via raw header.
	// We can't use seconds (the protocol's unit) at sub-second
	// resolution, so simulate by setting a 1-second Retry-After and
	// bumping retryMaxBackoff above it so the helper doesn't clamp.
	// Then assert we waited at least ~900ms.
	saveInit, saveMax := retryInitialBackoffSet(10*time.Millisecond, 5*time.Second)
	defer retryInitialBackoffSet(saveInit, saveMax)

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	start := time.Now()
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	if elapsed < 900*time.Millisecond {
		t.Errorf("elapsed = %v; want ≥ ~1s honoring Retry-After: 1", elapsed)
	}
}

// TestPostChatWithRetry_RespectsRetryAfterCap — Retry-After: 999s
// must be clamped to retryAfterCeiling rather than blocking the
// Turn for ~17 minutes. The ceiling exists as a sanity guard against
// a misbehaving server; the fair-share proxy normally sends sub-
// minute values which pass through exactly.
func TestPostChatWithRetry_RespectsRetryAfterCap(t *testing.T) {
	saveInit, saveMax := retryInitialBackoffSet(10*time.Millisecond, 50*time.Millisecond)
	defer retryInitialBackoffSet(saveInit, saveMax)
	prevCeil := retryAfterCeiling
	retryAfterCeiling = 100 * time.Millisecond
	defer func() { retryAfterCeiling = prevCeil }()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", strconv.Itoa(999))
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	start := time.Now()
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	// Should have been clamped to retryAfterCeiling (100ms), not 999s.
	if elapsed > 1*time.Second {
		t.Errorf("elapsed = %v; want clamped to ~%s, not the 999s the server suggested", elapsed, retryAfterCeiling)
	}
}

// TestPostChatWithRetry_SendsAPIKeyAndTrace — the API key is sent as the
// Authorization Bearer (autowork3's scheduling identity to the llama-swap
// fork), and the trace id is forwarded for log correlation.
func TestPostChatWithRetry_SendsAPIKeyAndTrace(t *testing.T) {
	fastBackoff(t)
	var gotAuth, gotTrace atomic.Value
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth.Store(r.Header.Get("Authorization"))
		gotTrace.Store(r.Header.Get("X-Trace-Id"))
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "sk-aw3", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "thread-abc")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	if got, _ := gotAuth.Load().(string); got != "Bearer sk-aw3" {
		t.Errorf("Authorization = %q; want %q", got, "Bearer sk-aw3")
	}
	if got, _ := gotTrace.Load().(string); got != "thread-abc" {
		t.Errorf("X-Trace-Id = %q; want thread-abc", got)
	}
}

// TestPostChatWithRetry_OmitsTraceWhenEmpty — an ad-hoc probe with no trace
// id should not emit a blank X-Trace-Id header.
func TestPostChatWithRetry_OmitsTraceWhenEmpty(t *testing.T) {
	fastBackoff(t)
	var traceHeaderSeen atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := r.Header["X-Trace-Id"]; ok {
			traceHeaderSeen.Store(true)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	if traceHeaderSeen.Load() {
		t.Error("X-Trace-Id was emitted with empty trace id; want header omitted")
	}
}

// TestPostChatWithRetry_CtxCancelMidBackoff — cancel the ctx during
// the first backoff sleep. Helper must return ctx.Err() and not hang
// for the full sleep duration.
func TestPostChatWithRetry_CtxCancelMidBackoff(t *testing.T) {
	// Make the backoff long enough that we can cancel inside it.
	saveInit, saveMax := retryInitialBackoffSet(2*time.Second, 5*time.Second)
	defer retryInitialBackoffSet(saveInit, saveMax)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		_, err := c.postChatWithRetry(ctx, []byte(`{}`), "test")
		done <- err
	}()

	// Let the first 429 land, then cancel mid-sleep.
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v; want context.Canceled", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("helper hung past ctx cancel — should return ctx.Err() promptly")
	}
}

// TestPostChatWithRetry_4xxPassesThrough — non-429 4xx (400, 401,
// 403, 404 …) surfaces immediately without retries. Those signal
// "your request is wrong" — retrying won't fix it.
func TestPostChatWithRetry_4xxPassesThrough(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d; want 400 (passthrough)", resp.StatusCode)
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("server hits = %d; want 1 (no retry on 4xx)", got)
	}
}

// TestPostChatWithRetry_5xx_RetriesThenRecovers — 5xx triggers
// bounded retry. First two responses are 502 ("bad gateway");
// third is 200. Helper retries through the 5xxs and surfaces the
// 200. This is the exact shape that surfaced in the autowork3
// trial: a transient 502 from the fair-share proxy that we
// recovered from on a subsequent attempt.
func TestPostChatWithRetry_5xx_RetriesThenRecovers(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d; want 200 (recovered after 5xx)", resp.StatusCode)
	}
	if got := hits.Load(); got != 3 {
		t.Errorf("server hits = %d; want 3 (2 × 502, 1 × 200)", got)
	}
}

// TestPostChatWithRetry_5xx_GivesUpAfterMaxAttempts — sustained
// 5xx hits the bounded-retry cap and surfaces a clean error
// pinning the count, so a real outage doesn't hang the daemon.
func TestPostChatWithRetry_5xx_GivesUpAfterMaxAttempts(t *testing.T) {
	fastBackoff(t)
	saveMax := retry5xxMaxAttempts
	retry5xxMaxAttempts = 3
	defer func() { retry5xxMaxAttempts = saveMax }()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	_, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	if err == nil {
		t.Fatal("postChatWithRetry: nil error after sustained 5xx; want error")
	}
	if !strings.Contains(err.Error(), "status 502") {
		t.Errorf("err = %v; want it to mention status 502", err)
	}
	if got := hits.Load(); got != int32(retry5xxMaxAttempts) {
		t.Errorf("server hits = %d; want %d (retry5xxMaxAttempts cap)", got, retry5xxMaxAttempts)
	}
}

// TestPostChatWithRetry_429_DoesNotCountAgainst5xxCap — a 429 + 5xx
// burst together shouldn't accidentally trip the 5xx cap. The 429
// retries are independent (Retry-After driven). Verify by sending
// 4×429 then 1×500 then 1×200 with the 5xx cap set to 2 — the
// 429s don't increment the 5xx counter, the single 500 retries
// once into the 200, total recovers.
func TestPostChatWithRetry_429_DoesNotCountAgainst5xxCap(t *testing.T) {
	fastBackoff(t)
	saveMax := retry5xxMaxAttempts
	retry5xxMaxAttempts = 2
	defer func() { retry5xxMaxAttempts = saveMax }()

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		switch {
		case n <= 4:
			w.WriteHeader(http.StatusTooManyRequests)
		case n == 5:
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	resp, err := c.postChatWithRetry(context.Background(), []byte(`{}`), "test")
	if err != nil {
		t.Fatalf("postChatWithRetry: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d; want 200 (5xx cap should not have tripped on the single 500)", resp.StatusCode)
	}
}
