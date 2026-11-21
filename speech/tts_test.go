package speech

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ttsServer is a fake Deepgram TTS endpoint. It performs the 101
// upgrade, sends a start event, answers each {"type":"text"} envelope
// with a run of binary audio frames (the audio is a deterministic
// function of the text), and ends the session when the client closes:
// it sends a close event and answers the close frame.
type ttsServer struct {
	*httptest.Server
	acceptChan chan net.Conn
}

func newTTSServer(t *testing.T, audioFor func(text string) []byte) *ttsServer {
	t.Helper()
	s := &ttsServer{acceptChan: make(chan net.Conn, 1)}
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
		defer conn.Close()

		resp := "HTTP/1.1 101 Switching Protocols\r\n" +
			"Upgrade: websocket\r\n" +
			"Connection: Upgrade\r\n" +
			"Sec-WebSocket-Accept: " + expectedAccept(key) + "\r\n" +
			"\r\n"
		if _, err := buf.Write([]byte(resp)); err != nil {
			return
		}
		if err := buf.Flush(); err != nil {
			return
		}
		s.acceptChan <- conn

		// Start event: the model the server will use.
		if _, err := buf.Write(unmaskedFrame(opText, []byte(`{"type":"start","model":"aura-2-thalia-en"}`))); err != nil {
			return
		}
		buf.Flush()

		// Serve text envelopes until the client closes.
		for {
			op, payload, err := readServerFrame(conn)
			if err != nil {
				return
			}
			switch op {
			case opClose:
				// End the session: close event, then answer the close.
				_, _ = buf.Write(unmaskedFrame(opText, []byte(`{"type":"close"}`)))
				_, _ = buf.Write(unmaskedFrame(opClose, makeClosePayload(CloseNormalClosure, "")))
				buf.Flush()
				return
			case opPing:
				_, _ = buf.Write(unmaskedFrame(opPong, payload))
				buf.Flush()
			case opText:
				var env ttsText
				if err := json.Unmarshal(payload, &env); err != nil {
					return
				}
				if env.Type != "text" {
					continue
				}
				audio := audioFor(env.Text)
				// Deliver the audio in two frames to exercise the
				// buffering path in TTSStream.Read.
				half := len(audio) / 2
				_, _ = buf.Write(unmaskedFrame(opBinary, audio[:half]))
				_, _ = buf.Write(unmaskedFrame(opBinary, audio[half:]))
				buf.Flush()
			}
		}
	}))
	return s
}

func (s *ttsServer) Close() { s.Server.Close() }

// readServerFrame reads one unmasked frame off a hijacked server-side
// conn (server-to-client frames are unmasked, so so are the client's,
// except for the mask key: client frames ARE masked).
func readServerFrame(conn net.Conn) (byte, []byte, error) {
	var hdr [2]byte
	if _, err := io.ReadFull(conn, hdr[:]); err != nil {
		return 0, nil, err
	}
	op := hdr[0] & 0x0F
	masked := hdr[1]&0x80 != 0
	length := int(hdr[1] & 0x7F)
	switch length {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(conn, ext[:]); err != nil {
			return 0, nil, err
		}
		length = int(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err := io.ReadFull(conn, ext[:]); err != nil {
			return 0, nil, err
		}
		length = int(binary.BigEndian.Uint64(ext[:]))
	}
	var maskKey [4]byte
	if masked {
		if _, err := io.ReadFull(conn, maskKey[:]); err != nil {
			return 0, nil, err
		}
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(conn, payload); err != nil {
		return 0, nil, err
	}
	if masked {
		applyMask(payload, maskKey)
	}
	return op, payload, nil
}

// drainAudio reads audio from the stream until nothing new arrives
// within quiet, then returns what was read. The TTS session ends when
// the client closes the connection, so io.Copy would block forever; a
// quiet timeout is the way to tell "the server is done sending".
func drainAudio(t *testing.T, s *TTSStream, quiet time.Duration) []byte {
	t.Helper()
	var out bytes.Buffer
	buf := make([]byte, 4096)
	for {
		s.stream.conn.conn.SetReadDeadline(time.Now().Add(quiet))
		n, err := s.Read(buf)
		if n > 0 {
			out.Write(buf[:n])
		}
		if err == io.EOF {
			return out.Bytes()
		}
		if err != nil {
			if isTimeout(err) {
				continue // quiet period: no more audio is coming
			}
			t.Fatalf("Read: %v", err)
		}
	}
}

func isTimeout(err error) bool {
	var ne *net.OpError
	return errors.As(err, &ne) && ne.Timeout()
}

// TestTTSStreamProducesAudio is the [F7] "TTS stream produces audio"
// scenario: open a stream, send text, and the audio frames the
// provider sends come out of Read.
func TestTTSStreamProducesAudio(t *testing.T) {
	const text = "hello from deepgram"
	var want []byte
	srv := newTTSServer(t, func(got string) []byte {
		if got != text {
			t.Errorf("server received text %q, want %q", got, text)
		}
		// Deterministic pseudo-audio: 512 bytes derived from the text.
		want = make([]byte, 512)
		for i := range want {
			want[i] = text[i%len(text)]
		}
		return want
	})
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, err := New(Config{APIKey: "test-key", BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := c.OpenTTS(context.Background(), TTSConfig{Model: "aura-2-thalia-en"})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := stream.SendText(text); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	got := drainAudio(t, stream, 250*time.Millisecond)
	if !bytes.Equal(got, want) {
		t.Errorf("audio = %d bytes, want %d bytes", len(got), len(want))
	}
}

// TestTTSUsageRecordsCharCount is the [F7] "TTS usage records
// character count" scenario: after the stream is used, Usage()
// reports the characters billed and the model.
func TestTTSUsageRecordsCharCount(t *testing.T) {
	const text1 = "first phrase" // 12 runes
	const text2 = "héllo wörld"  // 11 runes (é and ö are single runes)
	srv := newTTSServer(t, func(string) []byte { return []byte{0, 1, 2} })
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stream, err := c.OpenTTS(context.Background(), TTSConfig{Model: "aura-2-thalia-en"})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := stream.SendText(text1); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := stream.SendText(text2); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	u := stream.Usage()
	if u.Chars != 12+11 {
		t.Errorf("Usage().Chars = %d, want %d", u.Chars, 12+11)
	}
	if u.Model != "aura-2-thalia-en" {
		t.Errorf("Usage().Model = %q, want %q", u.Model, "aura-2-thalia-en")
	}
}

// TestTTSUsageEmptyBeforeStart checks the zero value: a stream that
// has sent nothing bills nothing, and the default model is named.
func TestTTSUsageEmptyBeforeStart(t *testing.T) {
	srv := newTTSServer(t, func(string) []byte { return nil })
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stream, err := c.OpenTTS(context.Background(), TTSConfig{})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	defer stream.Close()

	u := stream.Usage()
	if u.Chars != 0 {
		t.Errorf("Usage().Chars = %d, want 0", u.Chars)
	}
	if u.Model != ttsDefaultModel {
		t.Errorf("Usage().Model = %q, want %q", u.Model, ttsDefaultModel)
	}
}

// TestTTSReadDrivesStartEagerly verifies that Read alone (with no
// explicit Start) drives the handshake, delivers the audio, and
// records the model from the start event.
func TestTTSReadDrivesStartEagerly(t *testing.T) {
	srv := newTTSServer(t, func(string) []byte { return []byte("audio-bytes") })
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stream, err := c.OpenTTS(context.Background(), TTSConfig{})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	defer stream.Close()

	if err := stream.SendText("eager"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	got := drainAudio(t, stream, 250*time.Millisecond)
	if string(got) != "audio-bytes" {
		t.Errorf("audio = %q, want %q", got, "audio-bytes")
	}
	if stream.Usage().Model != "aura-2-thalia-en" {
		t.Errorf("Usage().Model = %q, want the server's model", stream.Usage().Model)
	}
}

// TestTTSReadReturnsEOFAfterClose verifies that once the session is
// closed and the buffer drained, Read returns io.EOF.
func TestTTSReadReturnsEOFAfterClose(t *testing.T) {
	srv := newTTSServer(t, func(string) []byte { return []byte("abc") })
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stream, err := c.OpenTTS(context.Background(), TTSConfig{})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if err := stream.SendText("closing"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	got := drainAudio(t, stream, 250*time.Millisecond)
	if string(got) != "abc" {
		t.Errorf("audio = %q, want %q", got, "abc")
	}

	// The server answered our close with its close event; the stream
	// is done and a further Read is io.EOF.
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Give the close event a moment to arrive, then the buffer is
	// empty and done is set: Read must be io.EOF, not an error.
	deadline := time.Now().Add(2 * time.Second)
	for {
		_, err := stream.Read(make([]byte, 16))
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Read after close = %v, want io.EOF", err)
		}
		if time.Now().After(deadline) {
			t.Fatal("Read after close never returned io.EOF")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestTTSOpenSendsModelInQuery verifies that the model is in the
// /speak query string, which is where Deepgram expects it, and that
// the default model is used when none is named.
func TestTTSOpenSendsModelInQuery(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		http.Error(w, "path captured", http.StatusInternalServerError)
	}))
	defer srv.Close()

	base := "ws://" + srv.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})
	_, _ = c.OpenTTS(context.Background(), TTSConfig{Model: "aura-asteria-en"})
	_, _ = c.OpenTTS(context.Background(), TTSConfig{})

	if len(paths) != 2 {
		t.Fatalf("got %d requests, want 2", len(paths))
	}
	if !strings.Contains(paths[0], "/speak?") || !strings.Contains(paths[0], "model=aura-asteria-en") {
		t.Errorf("request 1 path = %q, want /speak with model=aura-asteria-en", paths[0])
	}
	if !strings.Contains(paths[1], "model="+ttsDefaultModel) {
		t.Errorf("request 2 path = %q, want /speak with the default model", paths[1])
	}
}

// TestTTSStreamCloseIsIdempotent verifies Close can be called twice.
func TestTTSStreamCloseIsIdempotent(t *testing.T) {
	srv := newTTSServer(t, func(string) []byte { return nil })
	defer srv.Close()

	base := "ws://" + srv.Server.Listener.Addr().String()
	c, _ := New(Config{APIKey: "test-key", BaseURL: base})

	stream, err := c.OpenTTS(context.Background(), TTSConfig{})
	if err != nil {
		t.Fatalf("OpenTTS: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
