package llm

// Stall timeouts. A request that the network silently dropped (a laptop moved
// between networks, a NAT entry expired) does not fail: the socket stays open
// and the read blocks forever. The retry loop never starts, because it only
// runs after an attempt ENDS. Seen live: one request hung with the retry budget
// untouched. So every attempt carries a watchdog that ends it.
//
// Two limits, because the silence means different things:
//
//   - FirstByteTimeout: request sent -> first response byte. Long by default,
//     since llama.cpp sends nothing while it processes a long prompt; a short
//     value would kill requests that work.
//   - StallTimeout: gap between body bytes after the first one. A live
//     generation, or a proxy keep-alive comment, produces bytes; a dead link
//     does not.
//
// Not http.Client.Timeout: that caps the TOTAL time, and a stream may
// legitimately run for an hour. Not a per-line SSE timer either: the clock is
// reset at the io.Reader, below the SSE parser, so ": keep-alive" and
// ": OPENROUTER PROCESSING" comment lines (which the parser drops) still count
// as life.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	defaultFirstByteTimeout = 15 * time.Minute
	defaultStallTimeout     = 5 * time.Minute

	// HTTP/2 health check for the client-owned transport: ping a connection that
	// has been silent for http2PingAfter and close it if no answer arrives within
	// http2PingTimeout. It finds a dead link in under a minute instead of waiting
	// for TCP to give up.
	http2PingAfter   = 30 * time.Second
	http2PingTimeout = 15 * time.Second
)

// StallError is a request that went silent. It is transient: before anything
// reached the caller the client retries it within Client.RetryBudget; after
// output was forwarded it is returned, like StreamError, because a resend
// would regenerate the reply from scratch.
type StallError struct {
	// Phase is "first_byte" (no response byte at all) or "between_chunks"
	// (bytes stopped after the first one).
	Phase string
	// After is the silence that was allowed: the limit that fired.
	After time.Duration
}

func (e *StallError) Error() string {
	if e.Phase == "first_byte" {
		return fmt.Sprintf("llm: no response byte within %s (first-byte timeout)", e.After)
	}
	return fmt.Sprintf("llm: no data for %s mid-response (stall timeout)", e.After)
}

// firstByteTimeout is the effective limit: 0 = default; negative = off (0 here).
func (c *Client) firstByteTimeout() time.Duration {
	return effectiveTimeout(c.FirstByteTimeout, defaultFirstByteTimeout)
}

func (c *Client) stallTimeout() time.Duration {
	return effectiveTimeout(c.StallTimeout, defaultStallTimeout)
}

func effectiveTimeout(v, def time.Duration) time.Duration {
	switch {
	case v < 0:
		return 0
	case v == 0:
		return def
	}
	return v
}

// newTransport is the client-owned transport: the default one, plus HTTP/2
// pings. One per Client, so a dead connection of one client is not shared.
func newTransport() http.RoundTripper {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return http.DefaultTransport
	}
	t := base.Clone()
	t.HTTP2 = &http.HTTP2Config{SendPingTimeout: http2PingAfter, PingTimeout: http2PingTimeout}
	return t
}

// do is http.Client.Do with the watchdog. The attempt runs under its own
// context; the timer cancels it. A caller cancel always wins: the error is then
// the caller's, never a StallError. The response body must be closed, as
// always; Close releases the attempt's context.
func (c *Client) do(req *http.Request) (*http.Response, error) {
	fb, st := c.firstByteTimeout(), c.stallTimeout()
	if fb == 0 && st == 0 {
		return c.http.Do(req)
	}
	parent := req.Context()
	ctx, cancel := context.WithCancel(parent)
	w := &watchdog{parent: parent, cancel: cancel, firstByte: fb, stall: st}
	w.timer = time.AfterFunc(time.Hour, w.fire)
	w.timer.Stop()
	if fb > 0 {
		w.arm(fb)
	}
	resp, err := c.http.Do(req.WithContext(ctx))
	if err != nil {
		w.finish()
		if se := w.stalled(); se != nil {
			return nil, se
		}
		return nil, err
	}
	resp.Body = &watchBody{rc: resp.Body, w: w}
	return resp, nil
}

type watchdog struct {
	parent    context.Context
	cancel    context.CancelFunc
	firstByte time.Duration
	stall     time.Duration
	timer     *time.Timer

	mu    sync.Mutex
	armed bool
	got   bool // first byte seen
	done  bool
	fired *StallError
}

func (w *watchdog) arm(d time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.done || w.fired != nil {
		return
	}
	w.armed = true
	w.timer.Reset(d)
}

func (w *watchdog) disarm() {
	w.mu.Lock()
	w.armed = false
	w.timer.Stop()
	w.mu.Unlock()
}

func (w *watchdog) fire() {
	w.mu.Lock()
	if !w.armed || w.done || w.parent.Err() != nil {
		w.mu.Unlock()
		return
	}
	if w.got {
		w.fired = &StallError{Phase: "between_chunks", After: w.stall}
	} else {
		w.fired = &StallError{Phase: "first_byte", After: w.firstByte}
	}
	w.mu.Unlock()
	w.cancel()
}

// stalled is the StallError if the watchdog ended the attempt and the caller
// did not.
func (w *watchdog) stalled() *StallError {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.parent.Err() != nil {
		return nil
	}
	return w.fired
}

// finish stops the timer and releases the attempt's context.
func (w *watchdog) finish() {
	w.mu.Lock()
	w.done = true
	w.armed = false
	w.timer.Stop()
	w.mu.Unlock()
	w.cancel()
}

// watchBody times each Read, not the gap between Reads: a caller that is slow to
// consume (a full channel) must not read as a silent server. Before the first
// byte the timer armed in do keeps running.
type watchBody struct {
	rc io.ReadCloser
	w  *watchdog
}

func (b *watchBody) Read(p []byte) (int, error) {
	w := b.w
	w.mu.Lock()
	got := w.got
	w.mu.Unlock()
	if got && w.stall > 0 {
		w.arm(w.stall)
	}
	n, err := b.rc.Read(p)
	if n > 0 {
		w.mu.Lock()
		w.got = true
		w.mu.Unlock()
	}
	if n > 0 || err != nil {
		w.disarm()
	}
	if err != nil {
		if se := w.stalled(); se != nil {
			return n, se
		}
	}
	return n, err
}

func (b *watchBody) Close() error {
	err := b.rc.Close()
	b.w.finish()
	return err
}

// stallOf returns the *StallError in err's chain, or nil.
func stallOf(err error) *StallError {
	var se *StallError
	if errors.As(err, &se) {
		return se
	}
	return nil
}
