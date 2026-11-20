package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// stallServer runs handler(srv, w, r). Handlers
// that hang wait on hang(r), which ends when the client leaves or the test
// ends, so the server can always close.
type stallServer struct {
	*httptest.Server
	hits    atomic.Int32
	release chan struct{}
}

func newStallServer(t *testing.T, handler func(srv *stallServer, w http.ResponseWriter, r *http.Request)) *stallServer {
	t.Helper()
	s := &stallServer{release: make(chan struct{})}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler(s, w, r)
	}))
	t.Cleanup(s.Server.Close)
	t.Cleanup(func() { close(s.release) }) // runs first (LIFO): frees hung handlers
	return s
}

func (s *stallServer) hang(r *http.Request) {
	select {
	case <-r.Context().Done():
	case <-s.release:
	}
}

func flush(w http.ResponseWriter) { w.(http.Flusher).Flush() }

func stallClient(url string, firstByte, stall time.Duration) *Client {
	c := NewClient(url, "", "m")
	c.FirstByteTimeout, c.StallTimeout = firstByte, stall
	return c
}

func collectRetries(c *Client) func() []RetryEvent {
	var mu sync.Mutex
	var evs []RetryEvent
	c.OnRetry = func(ev RetryEvent) {
		mu.Lock()
		evs = append(evs, ev)
		mu.Unlock()
	}
	return func() []RetryEvent {
		mu.Lock()
		defer mu.Unlock()
		return append([]RetryEvent(nil), evs...)
	}
}

func hasKind(evs []RetryEvent, k RetryKind) bool {
	for _, ev := range evs {
		if ev.Kind == k {
			return true
		}
	}
	return false
}

// A server that accepts the request and never answers (a dropped network) must
// not hang the call: the watchdog ends the attempt as a stall, the request is
// resent, and the second attempt answers.
func TestStall_FirstByteBeforeHeadersIsRetried(t *testing.T) {
	fastBackoff(t)
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		if int(srv.hits.Add(1)) == 1 {
			srv.hang(r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, okStream)
	})
	c := stallClient(srv.URL, 100*time.Millisecond, 0)
	events := collectRetries(c)
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v; want the retried attempt's answer", text, errs)
	}
	if srv.hits.Load() != 2 {
		t.Errorf("hits = %d; want 2", srv.hits.Load())
	}
	if !hasKind(events(), RetryStall) {
		t.Errorf("events = %+v; want a RetryStall", events())
	}
}

// The retry is bounded: a server that never answers ends in an error when the
// budget is spent, not in an endless loop.
func TestStall_NeverAnsweringServerStopsAtRetryBudget(t *testing.T) {
	fastBackoff(t)
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		srv.hang(r)
	})
	c := stallClient(srv.URL, 50*time.Millisecond, 0)
	c.RetryBudget = 400 * time.Millisecond
	events := collectRetries(c)
	start := time.Now()
	_, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err == nil {
		t.Fatal("ChatStream returned nil; want the budget error")
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("took %s; the budget is 400ms", d)
	}
	if srv.hits.Load() < 2 {
		t.Errorf("hits = %d; want it retried before giving up", srv.hits.Load())
	}
	if !hasKind(events(), RetryGiveUp) {
		t.Errorf("events = %+v; want a giveup", events())
	}
}

// Headers arrive, the body never starts (a proxy that flushed headers, then the
// link died): still a first_byte stall, and still invisible to the caller.
func TestStall_HeadersThenNoByteIsRetried(t *testing.T) {
	fastBackoff(t)
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flush(w)
		if int(srv.hits.Add(1)) == 1 {
			srv.hang(r)
			return
		}
		fmt.Fprint(w, okStream)
	})
	c := stallClient(srv.URL, 100*time.Millisecond, 0)
	events := collectRetries(c)
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v", text, errs)
	}
	if srv.hits.Load() != 2 || !hasKind(events(), RetryStall) {
		t.Errorf("hits=%d events=%+v; want one stall retry", srv.hits.Load(), events())
	}
}

// One chunk, then silence: output already reached the caller, so a resend would
// regenerate it. The stall is returned as a typed between_chunks error and the
// request is NOT repeated.
func TestStall_AfterForwardedOutputIsReturnedNotRetried(t *testing.T) {
	fastBackoff(t)
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n")
		flush(w)
		srv.hang(r)
	})
	c := stallClient(srv.URL, 0, 100*time.Millisecond)
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var failure error
	for chunk := range ch {
		text += chunk.Content
		if e := chunk.Err(); e != nil {
			failure = e
		}
	}
	var se *StallError
	if !errors.As(failure, &se) || se.Phase != "between_chunks" || se.After != 100*time.Millisecond {
		t.Fatalf("failure = %v; want *StallError between_chunks after 100ms", failure)
	}
	if text != "par" {
		t.Errorf("text = %q; the forwarded output must stay", text)
	}
	if srv.hits.Load() != 1 {
		t.Errorf("hits = %d; a stall after output must not resend", srv.hits.Load())
	}
	if !TransientUpstream(failure) {
		t.Error("TransientUpstream = false; a stall is the link going away")
	}
}

// SSE comment lines are bytes. A proxy that sends ": keep-alive" while a model
// thinks must not be cut, even when the quiet-of-data time is far beyond the
// stall limit, and even before the first data event.
func TestStall_KeepAliveCommentsCountAsLife(t *testing.T) {
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flush(w)
		for i := 0; i < 10; i++ { // 500ms of comments, 5x the stall limit
			fmt.Fprint(w, ": keep-alive\n\n")
			flush(w)
			time.Sleep(50 * time.Millisecond)
		}
		fmt.Fprint(w, okStream)
	})
	c := stallClient(srv.URL, 100*time.Millisecond, 100*time.Millisecond)
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v; comments must keep the stream alive", text, errs)
	}
	if srv.hits.Load() != 1 {
		t.Errorf("hits = %d; want no retry", srv.hits.Load())
	}
}

// The caller's own cancel is the caller's error. It must never be dressed as a
// stall, which is transient and would be retried by an outer loop.
func TestStall_CallerCancelReturnsContextError(t *testing.T) {
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		srv.hang(r)
	})
	c := stallClient(srv.URL, 5*time.Second, 5*time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	_, err := c.ChatStream(ctx, []Message{{Role: "user", Content: "x"}}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v; want context.Canceled", err)
	}
	var se *StallError
	if errors.As(err, &se) {
		t.Errorf("err = %v; a caller cancel must not be a StallError", err)
	}
}

// A negative value turns a limit off: no first byte for longer than the stall
// limit is fine when the first-byte limit is off.
func TestStall_NegativeFirstByteDisablesIt(t *testing.T) {
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		time.Sleep(300 * time.Millisecond)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, okStream)
	})
	c := stallClient(srv.URL, -1, 50*time.Millisecond)
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 || srv.hits.Load() != 1 {
		t.Fatalf("text=%q errs=%v hits=%d; want one clean answer", text, errs, srv.hits.Load())
	}
}

// And the stall limit off: a gap between chunks longer than FirstByteTimeout is
// fine, since that limit covers only the wait for the first byte.
func TestStall_NegativeStallDisablesIt(t *testing.T) {
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		srv.hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"h\"}}]}\n\n")
		flush(w)
		time.Sleep(300 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"i\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	})
	c := stallClient(srv.URL, 100*time.Millisecond, -1)
	text, errs := drainStream(t, c)
	if text != "hi" || len(errs) != 0 {
		t.Fatalf("text=%q errs=%v", text, errs)
	}
}

// Non-streaming: nothing reaches the caller until the body is whole, so a stall
// while reading it is retried like one before the headers.
func TestStall_ChatBodyStallIsRetried(t *testing.T) {
	fastBackoff(t)
	srv := newStallServer(t, func(srv *stallServer, w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		flush(w)
		if int(srv.hits.Add(1)) == 1 {
			srv.hang(r)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	})
	c := stallClient(srv.URL, 100*time.Millisecond, 0)
	events := collectRetries(c)
	text, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil)
	if err != nil || text.Content != "hi" {
		t.Fatalf("text=%q err=%v", text.Content, err)
	}
	if srv.hits.Load() != 2 || !hasKind(events(), RetryStall) {
		t.Errorf("hits=%d events=%+v", srv.hits.Load(), events())
	}
}

func TestStall_DefaultsAndOff(t *testing.T) {
	c := NewClient("http://x", "", "m")
	if c.firstByteTimeout() != 15*time.Minute || c.stallTimeout() != 2*time.Minute {
		t.Errorf("defaults = %s, %s; want 15m, 2m", c.firstByteTimeout(), c.stallTimeout())
	}
	c.FirstByteTimeout, c.StallTimeout = -1, -time.Second
	if c.firstByteTimeout() != 0 || c.stallTimeout() != 0 {
		t.Error("negative must read as off")
	}
}

// The client owns its transport and asks HTTP/2 to ping a silent connection.
func TestStall_ClientTransportPingsHTTP2(t *testing.T) {
	c := NewClient("http://x", "", "m")
	tr, ok := c.http.Transport.(*http.Transport)
	if !ok || tr.HTTP2 == nil {
		t.Fatalf("transport = %T; want an *http.Transport with HTTP2 config", c.http.Transport)
	}
	if tr.HTTP2.SendPingTimeout != 30*time.Second || tr.HTTP2.PingTimeout != 15*time.Second {
		t.Errorf("HTTP2 = %+v", tr.HTTP2)
	}
	if tr == http.DefaultTransport {
		t.Error("shares http.DefaultTransport; must be its own clone")
	}
	if !strings.Contains((&StallError{Phase: "first_byte", After: time.Second}).Error(), "first-byte") {
		t.Error("StallError text lost its phase")
	}
}
