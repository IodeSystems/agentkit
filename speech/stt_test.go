package speech

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"
)

// sttWSServer is a fake Deepgram STT WebSocket server for tests.
// After the 101 upgrade it reads frames from the client and responds:
//
//   - binary frames (audio) get the next text frame from responses
//   - close frames get a close frame back
//   - ping frames get a pong
//
// The server-side connection is delivered on acceptChan after the
// upgrade, so the test can start the response loop in a goroutine.
type sttWSServer struct {
	*httptest.Server
	acceptChan  chan net.Conn
	responses   []string // one JSON envelope per audio frame
	authRequired bool
}

func newSTTWSServer(t *testing.T, responses []string) *sttWSServer {
	t.Helper()
	s := &sttWSServer{
		acceptChan: make(chan net.Conn, 1),
		responses:  responses,
	}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

func (s *sttWSServer) Close() { s.Server.Close() }

// serve reads frames from the server-side connection and responds
// according to the s.responses list. It runs until a close frame is
// received or the connection is torn down.
func (s *sttWSServer) serve(conn net.Conn) {
	br := bufio.NewReader(conn)
	wsc := &wsConn{conn: conn, br: br, closeCh: make(chan struct{})}

	respIdx := 0
	for {
		f, err := wsc.readFrame()
		if err != nil {
			return
		}
		switch f.Opcode {
		case opBinary:
			if respIdx < len(s.responses) {
				_ = wsc.writeFrame(opText, []byte(s.responses[respIdx]))
				respIdx++
			}
		case opClose:
			_ = wsc.writeFrame(opClose, makeClosePayload(CloseNormalClosure, ""))
			return
		case opPing:
			_ = wsc.writeFrame(opPong, f.Payload)
		}
	}
}

// waitForEvent reads the next event from ch with a timeout.
func waitForEvent(t *testing.T, ch <-chan STTEvent) STTEvent {
	t.Helper()
	select {
	case ev, ok := <-ch:
		if !ok {
			t.Fatal("events channel closed before event arrived")
		}
		return ev
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for STTEvent")
		return STTEvent{}
	}
}

// --- F7: STT stream produces transcriptions ---

// TestSTTTranscriptions verifies that a caller who opens an STT
// stream and sends audio frames receives transcription events.
func TestSTTTranscriptions(t *testing.T) {
	// The fake server responds to each audio frame with a
	// sequence of Deepgram envelopes.
	responses := []string{
		// Metadata first.
		`{"type":"Metadata","model":"nova-2","duration":0}`,
		// A partial result.
		`{"type":"Results","speech_final":false,"duration":1.5,` +
			`"model":"nova-2","channel":{"alternatives":[` +
			`{"transcript":"hello","confidence":0.85}]}}`,
		// A final result.
		`{"type":"Results","speech_final":true,"duration":2.0,` +
			`"model":"nova-2","channel":{"alternatives":[` +
			`{"transcript":"hello world","confidence":0.92}]}}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, err := New(Config{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stt, err := OpenSTT(context.Background(), c, STTOptions{Model: "nova-2"})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}
	defer stt.Close()

	// Start the server-side response loop.
	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send three audio frames; each triggers one response envelope.
	for i := 0; i < 3; i++ {
		if err := stt.Write(make([]byte, 3200)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}

	// Read the three events.
	ev1 := waitForEvent(t, stt.Events())
	if ev1.Type != "metadata" {
		t.Errorf("event 1 type = %q, want metadata", ev1.Type)
	}

	ev2 := waitForEvent(t, stt.Events())
	if ev2.Type != "partial" {
		t.Errorf("event 2 type = %q, want partial", ev2.Type)
	}
	if ev2.Transcript != "hello" {
		t.Errorf("event 2 transcript = %q, want hello", ev2.Transcript)
	}
	if ev2.Confidence != 0.85 {
		t.Errorf("event 2 confidence = %v, want 0.85", ev2.Confidence)
	}

	ev3 := waitForEvent(t, stt.Events())
	if ev3.Type != "final" {
		t.Errorf("event 3 type = %q, want final", ev3.Type)
	}
	if ev3.Transcript != "hello world" {
		t.Errorf("event 3 transcript = %q, want hello world", ev3.Transcript)
	}
	if ev3.Confidence != 0.92 {
		t.Errorf("event 3 confidence = %v, want 0.92", ev3.Confidence)
	}
}

// --- F7: stream closes cleanly and a final result is available ---

// TestSTTCleanClose verifies that closing an STT stream ends it
// cleanly and makes the final result available.
func TestSTTCleanClose(t *testing.T) {
	responses := []string{
		`{"type":"Metadata","model":"nova-2","duration":0}`,
		`{"type":"Results","speech_final":true,"duration":3.5,` +
			`"model":"nova-2","channel":{"alternatives":[` +
			`{"transcript":"the quick brown fox","confidence":0.95}]}}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{Model: "nova-2"})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send two audio frames to get both responses.
	for i := 0; i < 2; i++ {
		if err := stt.Write(make([]byte, 3200)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}

	// Consume the events.
	waitForEvent(t, stt.Events()) // metadata
	ev := waitForEvent(t, stt.Events())
	if ev.Type != "final" {
		t.Fatalf("event type = %q, want final", ev.Type)
	}

	// Close the stream.
	if err := stt.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The final result is available.
	if got := stt.Final(); got != "the quick brown fox" {
		t.Errorf("Final() = %q, want %q", got, "the quick brown fox")
	}

	// The events channel is closed.
	select {
	case _, ok := <-stt.Events():
		if ok {
			t.Error("events channel not closed after Close")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for events channel to close")
	}

	// Done is closed.
	select {
	case <-stt.Done():
	case <-time.After(time.Second):
		t.Error("Done channel not closed after Close")
	}
}

// --- F7: STT usage records duration and model name ---

// TestSTTUsageRecordsDurationAndModel verifies that after a
// completed STT stream, Usage() returns the stream duration in
// seconds and the model name.
func TestSTTUsageRecordsDurationAndModel(t *testing.T) {
	responses := []string{
		`{"type":"Metadata","model":"en-us","duration":0}`,
		`{"type":"Results","speech_final":false,"duration":1.2,` +
			`"model":"en-us","channel":{"alternatives":[` +
			`{"transcript":"hi","confidence":0.8}]}}`,
		`{"type":"Results","speech_final":true,"duration":2.8,` +
			`"model":"en-us","channel":{"alternatives":[` +
			`{"transcript":"hi there","confidence":0.9}]}}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{Model: "en-us"})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send audio to trigger all three responses.
	for i := 0; i < 3; i++ {
		if err := stt.Write(make([]byte, 3200)); err != nil {
			t.Fatalf("Write %d: %v", i, err)
		}
	}

	// Consume the events so the read loop has updated usage.
	for i := 0; i < 3; i++ {
		waitForEvent(t, stt.Events())
	}

	// Close the stream.
	if err := stt.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	usage := stt.Usage()
	if usage.Model != "en-us" {
		t.Errorf("Usage().Model = %q, want en-us", usage.Model)
	}
	if usage.Duration != 2.8 {
		t.Errorf("Usage().Duration = %v, want 2.8", usage.Duration)
	}
}

// --- parseSTTEnvelope unit tests ---

func TestParseSTTEnvelopeResultsFinal(t *testing.T) {
	payload := []byte(`{"type":"Results","speech_final":true,"duration":1.5,` +
		`"model":"nova-2","channel":{"alternatives":[` +
		`{"transcript":"hello","confidence":0.9}]}}`)
	ev, model, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "final" {
		t.Errorf("type = %q, want final", ev.Type)
	}
	if ev.Transcript != "hello" {
		t.Errorf("transcript = %q, want hello", ev.Transcript)
	}
	if ev.Confidence != 0.9 {
		t.Errorf("confidence = %v, want 0.9", ev.Confidence)
	}
	if ev.Duration != 1.5 {
		t.Errorf("duration = %v, want 1.5", ev.Duration)
	}
	if model != "nova-2" {
		t.Errorf("model = %q, want nova-2", model)
	}
}

func TestParseSTTEnvelopeResultsPartial(t *testing.T) {
	payload := []byte(`{"type":"Results","speech_final":false,"duration":0.5,` +
		`"model":"nova-2","channel":{"alternatives":[` +
		`{"transcript":"hel","confidence":0.7}]}}`)
	ev, _, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "partial" {
		t.Errorf("type = %q, want partial", ev.Type)
	}
	if ev.Transcript != "hel" {
		t.Errorf("transcript = %q, want hel", ev.Transcript)
	}
}

func TestParseSTTEnvelopeMetadata(t *testing.T) {
	payload := []byte(`{"type":"Metadata","model":"nova-2","duration":0}`)
	ev, model, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "metadata" {
		t.Errorf("type = %q, want metadata", ev.Type)
	}
	if model != "nova-2" {
		t.Errorf("model = %q, want nova-2", model)
	}
}

func TestParseSTTEnvelopeUtteranceEnd(t *testing.T) {
	payload := []byte(`{"type":"UtteranceEnd","duration":2.0}`)
	ev, _, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "utterance_end" {
		t.Errorf("type = %q, want utterance_end", ev.Type)
	}
	if ev.Duration != 2.0 {
		t.Errorf("duration = %v, want 2.0", ev.Duration)
	}
}

func TestParseSTTEnvelopeSpeechStarted(t *testing.T) {
	payload := []byte(`{"type":"SpeechStarted"}`)
	ev, _, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "speech_started" {
		t.Errorf("type = %q, want speech_started", ev.Type)
	}
}

func TestParseSTTEnvelopeSpeechEnded(t *testing.T) {
	payload := []byte(`{"type":"SpeechEnded","duration":1.0}`)
	ev, _, ok := parseSTTEnvelope(payload)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if ev.Type != "speech_stopped" {
		t.Errorf("type = %q, want speech_stopped", ev.Type)
	}
	if ev.Duration != 1.0 {
		t.Errorf("duration = %v, want 1.0", ev.Duration)
	}
}

func TestParseSTTEnvelopeUnknownType(t *testing.T) {
	payload := []byte(`{"type":"SomethingElse"}`)
	_, _, ok := parseSTTEnvelope(payload)
	if ok {
		t.Error("ok = true, want false for unknown type")
	}
}

func TestParseSTTEnvelopeInvalidJSON(t *testing.T) {
	payload := []byte(`not json`)
	_, _, ok := parseSTTEnvelope(payload)
	if ok {
		t.Error("ok = true, want false for invalid JSON")
	}
}

// --- STTOptions normalise ---

func TestSTTOptionsNormaliseDefaults(t *testing.T) {
	o := STTOptions{}.normalise()
	if o.Model != "nova-2" {
		t.Errorf("Model = %q, want nova-2", o.Model)
	}
	if o.SampleRate != 16000 {
		t.Errorf("SampleRate = %d, want 16000", o.SampleRate)
	}
	if o.Encoding != "linear16" {
		t.Errorf("Encoding = %q, want linear16", o.Encoding)
	}
	if !o.Interim {
		t.Error("Interim = false, want true for zero-value options")
	}
}

func TestSTTOptionsNormalisePreservesExplicit(t *testing.T) {
	o := STTOptions{
		Model:      "whisper-large",
		SampleRate: 8000,
		Encoding:   "mulaw",
		Punctuate:  true,
		Interim:    false,
	}.normalise()
	if o.Model != "whisper-large" {
		t.Errorf("Model = %q, want whisper-large", o.Model)
	}
	if o.SampleRate != 8000 {
		t.Errorf("SampleRate = %d, want 8000", o.SampleRate)
	}
	if o.Encoding != "mulaw" {
		t.Errorf("Encoding = %q, want mulaw", o.Encoding)
	}
	if !o.Punctuate {
		t.Error("Punctuate = false, want true")
	}
	if o.Interim {
		t.Error("Interim = true, want false")
	}
}

// --- STT Close idempotency ---

func TestSTTCloseIdempotent(t *testing.T) {
	responses := []string{
		`{"type":"Metadata","model":"nova-2","duration":0}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send one audio frame to trigger the metadata response.
	if err := stt.Write(make([]byte, 3200)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	// Consume the metadata event.
	waitForEvent(t, stt.Events())

	// Close twice; the second call must not block or error.
	if err := stt.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := stt.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

// --- STT: server sends close frame (peer-initiated close) ---

func TestSTTPeerClose(t *testing.T) {
	// The server responds to the first audio frame with a final
	// result, then sends a close frame.
	srv := &sttWSServer{
		acceptChan: make(chan net.Conn, 1),
	}
	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Header.Get("Sec-WebSocket-Key")
		if key == "" {
			http.Error(w, "missing key", http.StatusBadRequest)
			return
		}
		hj, _ := w.(http.Hijacker)
		conn, buf, _ := hj.Hijack()

		resp := "HTTP/1.1 101 Switching Protocols\r\n" +
			"Upgrade: websocket\r\n" +
			"Connection: Upgrade\r\n" +
			"Sec-WebSocket-Accept: " + expectedAccept(key) + "\r\n" +
			"\r\n"
		buf.Write([]byte(resp))
		buf.Flush()

		srv.acceptChan <- conn
	}))
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{Model: "nova-2"})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}

	// Server: read the audio frame, send a final result, then close.
	go func() {
		br := bufio.NewReader(conn)
		wsc := &wsConn{conn: conn, br: br, closeCh: make(chan struct{})}
		// Read the audio frame.
		_, _ = wsc.readFrame()
		// Send a final result.
		_ = wsc.writeFrame(opText, []byte(
			`{"type":"Results","speech_final":true,"duration":1.0,`+
				`"model":"nova-2","channel":{"alternatives":[`+
				`{"transcript":"bye","confidence":0.9}]}}`))
		// Send a close frame.
		_ = wsc.writeFrame(opClose, makeClosePayload(CloseNormalClosure, ""))
	}()

	// Send one audio frame.
	if err := stt.Write(make([]byte, 3200)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	// The read loop should get the final result and then the close
	// frame, exiting cleanly.
	ev := waitForEvent(t, stt.Events())
	if ev.Type != "final" {
		t.Fatalf("event type = %q, want final", ev.Type)
	}
	if ev.Transcript != "bye" {
		t.Errorf("transcript = %q, want bye", ev.Transcript)
	}

	// Wait for the read loop to exit (Done is closed).
	select {
	case <-stt.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for Done after peer close")
	}

	if got := stt.Final(); got != "bye" {
		t.Errorf("Final() = %q, want bye", got)
	}
}

// --- STT: URL query parameters ---

func TestOpenSTTQueryParams(t *testing.T) {
	// Capture the request URL from the server.
	type captured struct {
		path  string
		query url.Values
	}
	capCh := make(chan captured, 1)

	srv := &sttWSServer{
		acceptChan: make(chan net.Conn, 1),
	}
	srv.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capCh <- captured{
			path:  r.URL.Path,
			query: r.URL.Query(),
		}
		key := r.Header.Get("Sec-WebSocket-Key")
		if key == "" {
			http.Error(w, "missing key", http.StatusBadRequest)
			return
		}
		hj, _ := w.(http.Hijacker)
		conn, buf, _ := hj.Hijack()
		resp := "HTTP/1.1 101 Switching Protocols\r\n" +
			"Upgrade: websocket\r\n" +
			"Connection: Upgrade\r\n" +
			"Sec-WebSocket-Accept: " + expectedAccept(key) + "\r\n" +
			"\r\n"
		buf.Write([]byte(resp))
		buf.Flush()
		srv.acceptChan <- conn
	}))
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{
		Model:      "whisper-large",
		SampleRate: 8000,
		Encoding:   "mulaw",
		Punctuate:  true,
		Interim:    true,
	})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}
	defer stt.Close()

	// Drain the accept channel so the server goroutine doesn't block.
	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
		conn.Close()
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}

	var cap captured
	select {
	case cap = <-capCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for captured request")
	}

	if cap.path != "/v1/listen" {
		t.Errorf("path = %q, want /v1/listen", cap.path)
	}
	q := cap.query
	if q.Get("model") != "whisper-large" {
		t.Errorf("model = %q, want whisper-large", q.Get("model"))
	}
	if q.Get("sample_rate") != "8000" {
		t.Errorf("sample_rate = %q, want 8000", q.Get("sample_rate"))
	}
	if q.Get("encoding") != "mulaw" {
		t.Errorf("encoding = %q, want mulaw", q.Get("encoding"))
	}
	if q.Get("punctuate") != "true" {
		t.Errorf("punctuate = %q, want true", q.Get("punctuate"))
	}
	if q.Get("interim_results") != "true" {
		t.Errorf("interim_results = %q, want true", q.Get("interim_results"))
	}
}

// --- STT: Write after Close returns an error ---

func TestSTTWriteAfterClose(t *testing.T) {
	responses := []string{
		`{"type":"Metadata","model":"nova-2","duration":0}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send one audio frame to trigger the metadata response.
	if err := stt.Write(make([]byte, 3200)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	waitForEvent(t, stt.Events())
	stt.Close()

	err = stt.Write(make([]byte, 3200))
	if err == nil {
		t.Error("Write after Close = nil error, want error")
	}
}

// --- STT: events channel is closed after stream ends ---

func TestSTTEventsChannelClosed(t *testing.T) {
	responses := []string{
		`{"type":"Metadata","model":"nova-2","duration":0}`,
	}
	srv := newSTTWSServer(t, responses)
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stt, err := OpenSTT(context.Background(), c, STTOptions{})
	if err != nil {
		t.Fatalf("OpenSTT: %v", err)
	}

	var conn net.Conn
	select {
	case conn = <-srv.acceptChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for server connection")
	}
	go srv.serve(conn)

	// Send one audio frame to trigger the metadata response.
	if err := stt.Write(make([]byte, 3200)); err != nil {
		t.Fatalf("Write: %v", err)
	}

	waitForEvent(t, stt.Events())
	stt.Close()

	// The events channel must be closed.
	select {
	case _, ok := <-stt.Events():
		if ok {
			t.Error("events channel still open after Close")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for events channel to close")
	}
}


