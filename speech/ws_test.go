package speech

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// bufPipe is a buffered in-memory pipe that supports partial reads
// and writes, matching the behaviour of a real TCP connection. Unlike
// net.Pipe (which requires exact-size matching) and io.Pipe (which
// blocks a Read until the full buffer is filled), a bufPipe returns
// whatever is available, up to the requested length.
type bufPipe struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	closed bool
	cond   *sync.Cond
}

func newBufPipe() *bufPipe {
	p := &bufPipe{}
	p.cond = sync.NewCond(&p.mu)
	return p
}

func (p *bufPipe) Read(b []byte) (int, error) {
	p.mu.Lock()
	for p.buf.Len() == 0 && !p.closed {
		p.cond.Wait()
	}
	if p.buf.Len() == 0 {
		p.mu.Unlock()
		if p.closed {
			return 0, io.EOF
		}
		return 0, io.ErrClosedPipe
	}
	n, _ := p.buf.Read(b)
	p.mu.Unlock()
	return n, nil
}

func (p *bufPipe) Write(b []byte) (int, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return 0, io.ErrClosedPipe
	}
	n, _ := p.buf.Write(b)
	p.cond.Broadcast()
	p.mu.Unlock()
	return n, nil
}

func (p *bufPipe) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.cond.Broadcast()
	p.mu.Unlock()
	return nil
}

// pipeEnd wraps a pair of bufPipes to form a bidirectional
// in-memory connection for tests.
type pipeEnd struct {
	in  *bufPipe // data written to the peer
	out *bufPipe // data written by the peer
}

func (p *pipeEnd) Read(b []byte) (int, error)  { return p.in.Read(b) }
func (p *pipeEnd) Write(b []byte) (int, error) { return p.out.Write(b) }
func (p *pipeEnd) Close() error {
	p.in.Close()
	p.out.Close()
	return nil
}
func (p *pipeEnd) LocalAddr() net.Addr          { return net.Addr(nil) }
func (p *pipeEnd) RemoteAddr() net.Addr         { return net.Addr(nil) }
func (p *pipeEnd) SetDeadline(time.Time) error  { return nil }
func (p *pipeEnd) SetReadDeadline(time.Time) error  { return nil }
func (p *pipeEnd) SetWriteDeadline(time.Time) error { return nil }

// testPipe creates a pair of in-memory pipe ends that behave like a
// TCP connection for the purposes of frame read/write.
func testPipe() (a, b *pipeEnd) {
	aToB := newBufPipe() // data flowing a → b
	bToA := newBufPipe() // data flowing b → a
	return &pipeEnd{in: bToA, out: aToB}, &pipeEnd{in: aToB, out: bToA}
}

// pipeConn wraps a net.Conn with a wsConn, for use in tests.
func pipeConn(t *testing.T, c net.Conn) *wsConn {
	t.Helper()
	return &wsConn{conn: c, br: c, closeCh: make(chan struct{})}
}

// --- frame round-trip over net.Pipe ---

// TestFrameRoundTrip writes a frame on one side of a net.Pipe and
// reads it back on the other, for each payload-length tier (7-bit,
// 16-bit, 64-bit) and each opcode.
func TestFrameRoundTrip(t *testing.T) {
	sizes := []int{0, 1, 125, 126, 65535, 65536}
	opcodes := []byte{opText, opBinary, opPing, opPong, opClose}

	for _, op := range opcodes {
		for _, n := range sizes {
			n := n
			op := op
			payload := make([]byte, n)
			for i := range payload {
				payload[i] = byte(i % 251)
			}

			t.Run(fmt.Sprintf("op=%d_len=%d", op, n), func(t *testing.T) {
				a, b := testPipe()
				defer a.Close()
				defer b.Close()

				wsc := pipeConn(t, a)
				rsc := pipeConn(t, b)

				type result struct {
					f   *frame
					err error
				}
				ch := make(chan result, 1)
				go func() {
					f, err := rsc.readFrame()
					ch <- result{f, err}
				}()

				if err := wsc.writeFrame(op, payload); err != nil {
					t.Fatalf("writeFrame: %v", err)
				}

				r := <-ch
				if r.err != nil {
					t.Fatalf("readFrame: %v", r.err)
				}
				if r.f.Opcode != op {
					t.Errorf("opcode = 0x%x, want 0x%x", r.f.Opcode, op)
				}
				if !r.f.Final {
					t.Error("Final = false, want true")
				}
				if !bytes.Equal(r.f.Payload, payload) {
					t.Errorf("payload differs: got %d bytes, want %d", len(r.f.Payload), len(payload))
				}
			})
		}
	}
}

// TestFrameMaskVerify builds a masked frame with a known key and
// checks that unmasking recovers the original payload.
func TestFrameMaskVerify(t *testing.T) {
	orig := []byte("hello world, this is a mask test")
	masked := make([]byte, len(orig))
	copy(masked, orig)
	var key [4]byte
	key[0], key[1], key[2], key[3] = 0xAA, 0xBB, 0xCC, 0xDD
	applyMask(masked, key)
	if bytes.Equal(masked, orig) {
		t.Fatal("mask did not change payload")
	}
	applyMask(masked, key)
	if !bytes.Equal(masked, orig) {
		t.Fatal("unmask did not recover original payload")
	}
}

// --- upgrade key / accept ---

func TestUpgradeKeyIsBase64(t *testing.T) {
	key, err := upgradeKey()
	if err != nil {
		t.Fatal(err)
	}
	if len(key) != 24 { // 16 bytes → 24 base64 chars
		t.Errorf("key length = %d, want 24", len(key))
	}
}

func TestExpectedAcceptMatchesRFC6455Example(t *testing.T) {
	// RFC 6455 §4.2.2 example.
	const clientKey = "dGhlIHNhbXBsZSBub25jZQ=="
	const want = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
	if got := expectedAccept(clientKey); got != want {
		t.Errorf("expectedAccept = %q, want %q", got, want)
	}
}

// --- handshake: success ---

// wsTestServer is a minimal WebSocket server for tests. It performs
// the 101 upgrade and then exposes the raw net.Conn so the test can
// exchange frames directly.
type wsTestServer struct {
	*httptest.Server
	acceptChan chan net.Conn
	status     int
	body       string
	authRequired bool
	expectedAuth string
}

func newWSTestServer(t *testing.T) *wsTestServer {
	t.Helper()
	s := &wsTestServer{
		acceptChan: make(chan net.Conn, 1),
		status:     http.StatusSwitchingProtocols,
	}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Auth rejection: use s.status so tests can exercise both
		// 401 and 403 paths.
		if s.authRequired && r.Header.Get("Authorization") != s.expectedAuth {
			st := s.status
			if st == http.StatusSwitchingProtocols {
				st = http.StatusUnauthorized
			}
			w.WriteHeader(st)
			io.WriteString(w, s.body)
			return
		}

		if s.status != http.StatusSwitchingProtocols {
			w.WriteHeader(s.status)
			io.WriteString(w, s.body)
			return
		}

		key := r.Header.Get("Sec-WebSocket-Key")
		if key == "" {
			http.Error(w, "missing Sec-WebSocket-Key", http.StatusBadRequest)
			return
		}

		hj, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijack not supported", http.StatusInternalServerError)
			return
		}
		conn, buf, err := hj.Hijack()
		if err != nil {
			return
		}
		defer conn.Close()

		resp := "HTTP/1.1 101 Switching Protocols\r\n" +
			"Upgrade: websocket\r\n" +
			"Connection: Upgrade\r\n" +
			"Sec-WebSocket-Accept: " + expectedAccept(key) + "\r\n" +
			"\r\n"
		if _, err := buf.Write([]byte(resp)); err != nil {
			return
		}
		buf.Flush()

		s.acceptChan <- conn
	}))
	return s
}

func (s *wsTestServer) Close() {
	s.Server.Close()
}

func TestOpenSuccessfulUpgrade(t *testing.T) {
	srv := newWSTestServer(t)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, err := New(Config{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := c.Open(context.Background(), "/v1/listen")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer stream.Close()

	select {
	case conn := <-srv.acceptChan:
		conn.Close()
	case <-time.After(2 * time.Second):
		t.Fatal("server did not receive upgraded connection")
	}
}

// TestOpenAuthFailure401 verifies that a 401 from the provider
// produces an error that names the authentication failure.
func TestOpenAuthFailure401(t *testing.T) {
	srv := newWSTestServer(t)
	defer srv.Close()
	srv.authRequired = true
	srv.expectedAuth = "token good-key"
	srv.status = http.StatusUnauthorized
	srv.body = "invalid API key"

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "bad-key", BaseURL: base})

	_, err := c.Open(context.Background(), "/v1/listen")
	if err == nil {
		t.Fatal("expected error for 401, got nil")
	}
	if !strings.Contains(err.Error(), "authentication failure") {
		t.Errorf("error does not name authentication failure: %v", err)
	}
}

// TestOpenAuthFailure403 verifies the same for 403.
func TestOpenAuthFailure403(t *testing.T) {
	srv := newWSTestServer(t)
	defer srv.Close()
	srv.authRequired = true
	srv.expectedAuth = "token good-key"
	srv.status = http.StatusForbidden
	srv.body = "access denied"

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "bad-key", BaseURL: base})

	_, err := c.Open(context.Background(), "/speak")
	if err == nil {
		t.Fatal("expected error for 403, got nil")
	}
	if !strings.Contains(err.Error(), "authentication failure") {
		t.Errorf("error does not name authentication failure: %v", err)
	}
}

// TestOpenMissingKey verifies that New rejects a Config with no
// APIKey.
func TestOpenMissingKey(t *testing.T) {
	_, err := New(Config{
		BaseURL: "wss://api.deepgram.com",
	})
	if err == nil {
		t.Fatal("expected error for missing API key")
	}
	if !strings.Contains(err.Error(), "APIKey") {
		t.Errorf("error does not name APIKey: %v", err)
	}
}

// TestOpenInvalidBaseURL verifies that New rejects a BaseURL that is
// not ws:// or wss://.
func TestOpenInvalidBaseURL(t *testing.T) {
	for _, bad := range []string{"", "http://x", "localhost", "not a url"} {
		_, err := New(Config{APIKey: "k", BaseURL: bad})
		if err == nil {
			t.Errorf("expected error for BaseURL %q", bad)
		}
	}
}

// TestNewValidConfig verifies that New succeeds with a valid config
// and returns a usable Client.
func TestNewValidConfig(t *testing.T) {
	c, err := New(Config{
		APIKey:  "my-key",
		BaseURL: "wss://api.deepgram.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	if c.apiKey != "my-key" {
		t.Errorf("apiKey = %q, want %q", c.apiKey, "my-key")
	}
	if c.base != "wss://api.deepgram.com" {
		t.Errorf("base = %q, want %q", c.base, "wss://api.deepgram.com")
	}
}

// --- ping/pong and close over net.Pipe ---

// TestPingPongRoundTrip verifies that a ping frame is answered with a
// pong frame carrying the same payload.
func TestPingPongRoundTrip(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	server := pipeConn(t, b)
	client := pipeConn(t, a)

	pingPayload := []byte("keepalive")

	type result struct {
		f   *frame
		err error
	}

	// Server sends ping; client reads it.
	ch1 := make(chan result, 1)
	go func() {
		f, err := client.readFrame()
		ch1 <- result{f, err}
	}()
	if err := server.writeFrame(opPing, pingPayload); err != nil {
		t.Fatal(err)
	}
	r1 := <-ch1
	if r1.err != nil {
		t.Fatal(r1.err)
	}
	if r1.f.Opcode != opPing {
		t.Fatalf("opcode = 0x%x, want opPing", r1.f.Opcode)
	}
	if !bytes.Equal(r1.f.Payload, pingPayload) {
		t.Error("ping payload mismatch")
	}

	// Client replies with pong; server reads it.
	ch2 := make(chan result, 1)
	go func() {
		f, err := server.readFrame()
		ch2 <- result{f, err}
	}()
	if err := client.writeFrame(opPong, pingPayload); err != nil {
		t.Fatal(err)
	}
	r2 := <-ch2
	if r2.err != nil {
		t.Fatal(r2.err)
	}
	if r2.f.Opcode != opPong {
		t.Fatalf("opcode = 0x%x, want opPong", r2.f.Opcode)
	}
	if !bytes.Equal(r2.f.Payload, pingPayload) {
		t.Error("pong payload mismatch")
	}
}

// TestCloseFrameRoundTrip verifies that a close frame is read back
// with its code and reason intact.
func TestCloseFrameRoundTrip(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	server := pipeConn(t, b)
	client := pipeConn(t, a)

	payload := makeClosePayload(CloseNormalClosure, "done")

	type result struct {
		f   *frame
		err error
	}
	ch := make(chan result, 1)
	go func() {
		f, err := client.readFrame()
		ch <- result{f, err}
	}()
	if err := server.writeFrame(opClose, payload); err != nil {
		t.Fatal(err)
	}
	r := <-ch
	if r.err != nil {
		t.Fatal(r.err)
	}
	if r.f.Opcode != opClose {
		t.Fatalf("opcode = 0x%x, want opClose", r.f.Opcode)
	}
	code, reason := parseClosePayload(r.f.Payload)
	if code != CloseNormalClosure {
		t.Errorf("code = %d, want %d", code, CloseNormalClosure)
	}
	if reason != "done" {
		t.Errorf("reason = %q, want %q", reason, "done")
	}
}

// TestStreamReadMessageHandlesPing verifies that Stream.ReadMessage
// transparently answers pings and returns the next data frame.
func TestStreamReadMessageHandlesPing(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	server := pipeConn(t, b)
	client := pipeConn(t, a)
	stream := &Stream{conn: client}

	type result struct {
		f   *frame
		err error
	}

	// Server: write a ping, consume the pong the client replies
	// with, then write a text frame. net.Pipe is fully synchronous —
	// a write blocks until the peer reads — so the pong must be
	// consumed before the text write, or both sides block on each
	// other (the client writing the pong, the server writing the
	// text) and the test deadlocks.
	pongCh := make(chan result, 1)
	go func() {
		if err := server.writeFrame(opPing, []byte("hi")); err != nil {
			pongCh <- result{nil, err}
			return
		}
		f, err := server.readFrame()
		if err != nil {
			pongCh <- result{f, err}
			return
		}
		pongCh <- result{f, nil}
		if err := server.writeFrame(opText, []byte("hello")); err != nil {
			pongCh <- result{nil, err}
			return
		}
	}()

	op, payload, err := stream.ReadMessage()
	if err != nil {
		t.Fatal(err)
	}
	if op != opText {
		t.Fatalf("opcode = 0x%x, want opText", op)
	}
	if string(payload) != "hello" {
		t.Errorf("payload = %q, want %q", payload, "hello")
	}

	// Verify the server got a pong in reply to its ping.
	select {
	case r := <-pongCh:
		if r.err != nil {
			t.Fatalf("server: %v", r.err)
		}
		if r.f.Opcode != opPong {
			t.Fatalf("server got opcode 0x%x, want opPong", r.f.Opcode)
		}
		if !bytes.Equal(r.f.Payload, []byte("hi")) {
			t.Error("pong payload mismatch")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for the pong")
	}
}

// TestStreamReadMessageCloseFrame verifies that a close frame from
// the peer produces a *CloseError.
func TestStreamReadMessageCloseFrame(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	server := pipeConn(t, b)
	client := pipeConn(t, a)
	stream := &Stream{conn: client}

	go func() {
		server.writeFrame(opClose, makeClosePayload(CloseGoingAway, "shutting down"))
	}()

	_, _, err := stream.ReadMessage()
	if err == nil {
		t.Fatal("expected CloseError, got nil")
	}
	ce, ok := err.(*CloseError)
	if !ok {
		t.Fatalf("expected *CloseError, got %T: %v", err, err)
	}
	if ce.Code != CloseGoingAway {
		t.Errorf("code = %d, want %d", ce.Code, CloseGoingAway)
	}
	if ce.Reason != "shutting down" {
		t.Errorf("reason = %q, want %q", ce.Reason, "shutting down")
	}
}

// TestStreamCloseIsIdempotent verifies that calling Close twice does
// not panic or error.
func TestStreamCloseIsIdempotent(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	client := pipeConn(t, a)
	stream := &Stream{conn: client}

	if err := stream.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	select {
	case <-stream.Done():
		// expected
	default:
		t.Error("Done channel not closed after Close")
	}
}

// TestSendAfterClose verifies that Send returns ErrNotConnected after
// the stream is closed.
func TestSendAfterClose(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	client := pipeConn(t, a)
	stream := &Stream{conn: client}

	stream.Close()

	err := stream.Send([]byte("data"))
	if err != ErrNotConnected {
		t.Errorf("Send after Close = %v, want ErrNotConnected", err)
	}
}

// --- 16-bit and 64-bit payload lengths ---

func TestLargeFrame16Bit(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	wsc := pipeConn(t, a)
	rsc := pipeConn(t, b)

	payload := make([]byte, 300) // > 125, uses 16-bit length
	for i := range payload {
		payload[i] = byte(i)
	}

	type result struct {
		f   *frame
		err error
	}
	ch := make(chan result, 1)
	go func() {
		f, err := rsc.readFrame()
		ch <- result{f, err}
	}()
	if err := wsc.writeFrame(opBinary, payload); err != nil {
		t.Fatal(err)
	}
	r := <-ch
	if r.err != nil {
		t.Fatal(r.err)
	}
	if !bytes.Equal(r.f.Payload, payload) {
		t.Error("16-bit frame payload mismatch")
	}
}

func TestLargeFrame64Bit(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	wsc := pipeConn(t, a)
	rsc := pipeConn(t, b)

	payload := make([]byte, 65536) // just over the 16-bit limit
	for i := range payload {
		payload[i] = byte(i % 256)
	}

	type result struct {
		f   *frame
		err error
	}
	ch := make(chan result, 1)
	go func() {
		f, err := rsc.readFrame()
		ch <- result{f, err}
	}()
	if err := wsc.writeFrame(opBinary, payload); err != nil {
		t.Fatal(err)
	}
	r := <-ch
	if r.err != nil {
		t.Fatal(r.err)
	}
	if len(r.f.Payload) != 65536 {
		t.Fatalf("payload length = %d, want 65536", len(r.f.Payload))
	}
	if !bytes.Equal(r.f.Payload, payload) {
		t.Error("64-bit frame payload mismatch")
	}
}

// --- URL helpers ---

func TestURLJoin(t *testing.T) {
	cases := []struct {
		base, path, want string
	}{
		{"wss://api.deepgram.com", "/v1/listen", "wss://api.deepgram.com/v1/listen"},
		{"wss://api.deepgram.com/", "/v1/listen", "wss://api.deepgram.com/v1/listen"},
		{"wss://api.deepgram.com", "v1/listen", "wss://api.deepgram.com/v1/listen"},
		{"wss://api.deepgram.com/", "v1/listen", "wss://api.deepgram.com/v1/listen"},
	}
	for _, tc := range cases {
		got := urlJoin(tc.base, tc.path)
		if got != tc.want {
			t.Errorf("urlJoin(%q, %q) = %q, want %q", tc.base, tc.path, got, tc.want)
		}
	}
}

func TestParseURLRejectsNonWS(t *testing.T) {
	for _, bad := range []string{"", "http://x", "localhost", "ftp://x"} {
		if _, err := parseURL(bad); err == nil {
			t.Errorf("parseURL(%q) = nil error, want error", bad)
		}
	}
}

func TestParseURLAcceptsWSAndWSS(t *testing.T) {
	for _, good := range []string{"ws://localhost:8080", "wss://api.deepgram.com/v1/listen"} {
		if _, err := parseURL(good); err != nil {
			t.Errorf("parseURL(%q) = %v, want nil", good, err)
		}
	}
}

// --- readHTTPResponse ---

func TestReadHTTPResponse(t *testing.T) {
	raw := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n" +
		"\r\n"
	br := bufio.NewReader(strings.NewReader(raw))
	code, header, err := readHTTPResponse(br)
	if err != nil {
		t.Fatal(err)
	}
	if code != 101 {
		t.Errorf("code = %d, want 101", code)
	}
	if len(header["Upgrade"]) == 0 || header["Upgrade"][0] != "websocket" {
		t.Errorf("Upgrade = %v", header["Upgrade"])
	}
	if len(header["Sec-WebSocket-Accept"]) == 0 || header["Sec-WebSocket-Accept"][0] != "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" {
		t.Errorf("Sec-WebSocket-Accept = %v", header["Sec-WebSocket-Accept"])
	}
}

func TestReadHTTPResponse401(t *testing.T) {
	raw := "HTTP/1.1 401 Unauthorized\r\n" +
		"Content-Length: 11\r\n" +
		"\r\n" +
		"invalid key"
	br := bufio.NewReader(strings.NewReader(raw))
	code, _, err := readHTTPResponse(br)
	if err != nil {
		t.Fatal(err)
	}
	if code != 401 {
		t.Errorf("code = %d, want 401", code)
	}
}

// --- verifyUpgradeResponse ---

func TestVerifyUpgradeResponse(t *testing.T) {
	key, _ := upgradeKey()
	hdr := map[string][]string{
		"Sec-WebSocket-Accept": {expectedAccept(key)},
		"Upgrade":              {"websocket"},
	}
	if err := verifyUpgradeResponse(101, hdr, key); err != nil {
		t.Errorf("valid upgrade: %v", err)
	}

	hdrBad := map[string][]string{
		"Sec-WebSocket-Accept": {"wrong"},
		"Upgrade":              {"websocket"},
	}
	if err := verifyUpgradeResponse(101, hdrBad, key); err == nil {
		t.Error("expected error for bad Sec-WebSocket-Accept")
	}

	if err := verifyUpgradeResponse(403, hdr, key); err == nil {
		t.Error("expected error for non-101 status")
	}
}

// --- unmaskedFrame / maskedFrame helpers ---

func TestUnmaskedFrameSmall(t *testing.T) {
	payload := []byte("hi")
	fr := unmaskedFrame(opText, payload)
	if fr[0] != 0x81 {
		t.Errorf("first byte = 0x%x, want 0x81", fr[0])
	}
	if fr[1] != 2 {
		t.Errorf("length byte = %d, want 2", fr[1])
	}
	if !bytes.Equal(fr[2:], payload) {
		t.Error("payload mismatch")
	}
}

func TestMaskedFrameSmall(t *testing.T) {
	payload := []byte("hi")
	fr := maskedFrame(opBinary, payload)
	if fr[0] != 0x82 {
		t.Errorf("first byte = 0x%x, want 0x82", fr[0])
	}
	if fr[1]&0x80 == 0 {
		t.Error("mask bit not set")
	}
	if fr[1]&0x7F != 2 {
		t.Errorf("length = %d, want 2", fr[1]&0x7F)
	}
	maskKey := fr[2:6]
	masked := fr[6:]
	orig := make([]byte, len(masked))
	for i := range masked {
		orig[i] = masked[i] ^ maskKey[i%4]
	}
	if !bytes.Equal(orig, payload) {
		t.Error("unmasked payload does not match original")
	}
}

// --- concurrency: simultaneous reads and writes ---

func TestConcurrentReadWrite(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	wsc := pipeConn(t, a)
	rsc := pipeConn(t, b)

	const n = 50
	var wg sync.WaitGroup
	errCh := make(chan error, 2)

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < n; i++ {
			payload := []byte(fmt.Sprintf("msg-%d", i))
			if err := wsc.writeFrame(opText, payload); err != nil {
				errCh <- err
				return
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < n; i++ {
			f, err := rsc.readFrame()
			if err != nil {
				errCh <- err
				return
			}
			want := fmt.Sprintf("msg-%d", i)
			if string(f.Payload) != want {
				errCh <- fmt.Errorf("msg %d: got %q, want %q", i, f.Payload, want)
				return
			}
		}
	}()

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

// --- CloseError formatting ---

func TestCloseErrorString(t *testing.T) {
	e := &CloseError{Code: CloseGoingAway, Reason: "server restarting"}
	s := e.Error()
	if !strings.Contains(s, "1001") {
		t.Errorf("CloseError string missing code: %s", s)
	}
	if !strings.Contains(s, "server restarting") {
		t.Errorf("CloseError string missing reason: %s", s)
	}
}

// --- makeClosePayload edge cases ---

func TestMakeClosePayloadZero(t *testing.T) {
	if p := makeClosePayload(0, ""); p != nil {
		t.Errorf("makeClosePayload(0, \"\") = %v, want nil", p)
	}
}

func TestMakeClosePayloadNormal(t *testing.T) {
	p := makeClosePayload(CloseNormalClosure, "bye")
	if len(p) < 2 {
		t.Fatal("payload too short")
	}
	code := CloseCode(binary.BigEndian.Uint16(p[:2]))
	if code != CloseNormalClosure {
		t.Errorf("code = %d, want %d", code, CloseNormalClosure)
	}
	if string(p[2:]) != "bye" {
		t.Errorf("reason = %q, want %q", p[2:], "bye")
	}
}

// --- parseClosePayload edge cases ---

func TestParseClosePayloadEmpty(t *testing.T) {
	code, reason := parseClosePayload(nil)
	if code != CloseNoStatus {
		t.Errorf("code = %d, want CloseNoStatus", code)
	}
	if reason != "" {
		t.Errorf("reason = %q, want empty", reason)
	}
}

// --- Open with context cancellation ---

func TestOpenContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	base := "ws://127.0.0.1:1"
	c, _ := New(Config{APIKey: "k", BaseURL: base})

	_, err := c.Open(ctx, "/v1/listen")
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

// --- TLS: wss:// handshake is attempted ---

func TestOpenWSSAttempt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	base := "wss://" + srv.Listener.Addr().String()
	c, err := New(Config{APIKey: "k", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = c.Open(context.Background(), "/v1/listen")
	if err == nil {
		t.Fatal("expected TLS error against plain HTTP server")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "tls") {
		t.Errorf("error does not mention TLS: %v", err)
	}
}

// --- Stream.Done after peer close ---

func TestStreamDoneAfterPeerClose(t *testing.T) {
	a, b := testPipe()
	defer a.Close()
	defer b.Close()

	server := pipeConn(t, b)
	client := pipeConn(t, a)
	stream := &Stream{conn: client}

	go func() {
		server.writeFrame(opClose, makeClosePayload(CloseNormalClosure, ""))
	}()

	_, _, _ = stream.ReadMessage()

	select {
	case <-stream.Done():
	case <-time.After(time.Second):
		t.Error("Done channel not closed after peer close")
	}
}
