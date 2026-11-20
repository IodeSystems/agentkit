// Package speech is a Deepgram STT/TTS client.
//
// The provider protocol — the Deepgram WebSocket handshake and frame
// framing, the JSON envelopes, and per-stream billing — lives here.
// Audio capture, playback, echo cancellation, endpointing and turn
// handling are host concerns and are not in this package.
//
// The WebSocket client (ws.go) is a hand-written RFC 6455 client built
// on the standard library only. It is minimal by design: it opens a
// connection, exchanges frames, and closes. It does not implement
// subprotocols, extensions, compression, fragmentation, or an
// application-level ping policy — those are the responsibility of the
// STT/TTS layers that sit on top of it.
package speech

import (
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"sync"
)

// Opcodes, per RFC 6455 §5.5.
const (
	opContinuation = 0x0
	opText         = 0x1
	opBinary       = 0x2
	// opClose, opPing, opPong are control frames.
	opClose = 0x8
	opPing  = 0x9
	opPong  = 0xA
)

// The RFC 6455 magic GUID, used in the Sec-WebSocket-Accept
// computation (§4.2.2).
const wsMagicGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// ErrNotConnected is returned by send/receive operations after the
// connection has been closed.
var ErrNotConnected = errors.New("speech: connection is closed")

// ErrProtocol is returned when the peer violates the WebSocket
// protocol in a way that has no specific error.
var ErrProtocol = errors.New("speech: protocol violation")

// CloseCode is the 16-bit status code in a close frame (§7.4).
type CloseCode uint16

// Standard close codes (RFC 6455 §7.4.2).
const (
	CloseNormalClosure   CloseCode = 1000
	CloseGoingAway       CloseCode = 1001
	CloseProtocolError   CloseCode = 1002
	CloseUnsupportedData CloseCode = 1003
	CloseNoStatus        CloseCode = 1005
	CloseAbnormalClosure CloseCode = 1006
	CloseInvalidPayload  CloseCode = 1007
	ClosePolicyViolation CloseCode = 1008
	CloseMessageTooBig   CloseCode = 1009
)

// CloseError is returned by ReadMessage when the server sends a close
// frame. It carries the status code and the reason string from the
// frame so a caller can distinguish a clean close from a protocol
// error.
type CloseError struct {
	Code   CloseCode
	Reason string
}

func (e *CloseError) Error() string {
	return fmt.Sprintf("speech: connection closed by peer (code %d: %s)", e.Code, e.Reason)
}

// frame is one decoded WebSocket frame.
type frame struct {
	Opcode       byte
	Final        bool
	Payload      []byte
}

// wsConn is the transport-level connection. It is deliberately kept
// separate from the higher-level Client so that the STT/TTS layers can
// add their own state without touching the frame logic.
type wsConn struct {
	conn net.Conn
	br   io.Reader
	wmu  sync.Mutex // serialises writes

	closeOnce sync.Once
	closeCh   chan struct{}
	closed    bool
}

// readFrame reads one frame from the connection. Control frames
// (ping, pong, close) are returned as-is; the caller is responsible
// for replying to pings and handling close.
func (c *wsConn) readFrame() (*frame, error) {
	var hdr [2]byte
	if _, err := io.ReadFull(c.br, hdr[:]); err != nil {
		if c.closed {
			return nil, ErrNotConnected
		}
		if errors.Is(err, io.EOF) {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("speech: read frame header: %w", err)
	}

	f := &frame{
		Opcode: hdr[0] & 0x0F,
		Final:  hdr[0]&0x80 != 0,
	}

	masked := hdr[1]&0x80 != 0
	length := uint64(hdr[1] & 0x7F)

	switch length {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(c.br, ext[:]); err != nil {
			return nil, fmt.Errorf("speech: read 16-bit length: %w", err)
		}
		length = uint64(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err := io.ReadFull(c.br, ext[:]); err != nil {
			return nil, fmt.Errorf("speech: read 64-bit length: %w", err)
		}
		length = binary.BigEndian.Uint64(ext[:])
		if length > 1<<63-1 {
			return nil, fmt.Errorf("%w: payload length %d overflows int64", ErrProtocol, length)
		}
	}

	if length > 1<<24 {
		return nil, fmt.Errorf("%w: payload too large (%d bytes)", ErrProtocol, length)
	}

	var maskKey [4]byte
	if masked {
		if _, err := io.ReadFull(c.br, maskKey[:]); err != nil {
			return nil, fmt.Errorf("speech: read mask key: %w", err)
		}
	}

	f.Payload = make([]byte, length)
	if _, err := io.ReadFull(c.br, f.Payload); err != nil {
		return nil, fmt.Errorf("speech: read payload: %w", err)
	}

	if masked {
		applyMask(f.Payload, maskKey)
	}

	return f, nil
}

// writeFrame writes one frame to the connection. Client-to-server
// frames MUST be masked (RFC 6455 §5.3); the mask key is generated
// here.
func (c *wsConn) writeFrame(op byte, payload []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()

	if c.closed {
		return ErrNotConnected
	}

	var maskKey [4]byte
	if _, err := rand.Read(maskKey[:]); err != nil {
		return fmt.Errorf("speech: generate mask key: %w", err)
	}

	var hdr []byte
	hdr = append(hdr, 0x80|op) // FIN=1

	length := len(payload)
	switch {
	case length < 126:
		hdr = append(hdr, 0x80|byte(length))
	case length < 65536:
		hdr = append(hdr, 0x80|126)
		var ext [2]byte
		binary.BigEndian.PutUint16(ext[:], uint16(length))
		hdr = append(hdr, ext[:]...)
	default:
		hdr = append(hdr, 0x80|127)
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(length))
		hdr = append(hdr, ext[:]...)
	}

	hdr = append(hdr, maskKey[:]...)

	masked := make([]byte, len(payload))
	copy(masked, payload)
	applyMask(masked, maskKey)

	if _, err := c.conn.Write(hdr); err != nil {
		return fmt.Errorf("speech: write frame header: %w", err)
	}
	if len(masked) > 0 {
		if _, err := c.conn.Write(masked); err != nil {
			return fmt.Errorf("speech: write payload: %w", err)
		}
	}
	return nil
}

// sendControl sends a control frame (ping, pong, close).
func (c *wsConn) sendControl(op byte, payload []byte) error {
	return c.writeFrame(op, payload)
}

// closeConn marks the connection as closed and tears down the
// underlying network connection. Safe to call multiple times.
func (c *wsConn) closeConn() {
	c.closeOnce.Do(func() {
		c.closed = true
		close(c.closeCh)
		c.conn.Close()
	})
}

// CloseErrorf is a helper for building a CloseError from a close-frame
// payload.
func parseClosePayload(payload []byte) (CloseCode, string) {
	if len(payload) < 2 {
		return CloseNoStatus, ""
	}
	code := CloseCode(binary.BigEndian.Uint16(payload[:2]))
	reason := string(payload[2:])
	return code, reason
}

// makeClosePayload builds the byte payload for a close frame.
func makeClosePayload(code CloseCode, reason string) []byte {
	if code == 0 {
		return nil
	}
	var buf bytes.Buffer
	var ext [2]byte
	binary.BigEndian.PutUint16(ext[:], uint16(code))
	buf.Write(ext[:])
	buf.WriteString(reason)
	return buf.Bytes()
}

// --- handshake ---

// upgradeKey generates a new 128-bit Sec-WebSocket-Key per RFC 6455
// §4.1.
func upgradeKey() (string, error) {
	var key [16]byte
	if _, err := rand.Read(key[:]); err != nil {
		return "", fmt.Errorf("speech: generate websocket key: %w", err)
	}
	return base64.StdEncoding.EncodeToString(key[:]), nil
}

// expectedAccept computes the Sec-WebSocket-Accept value for a given
// client key (§4.2.2).
func expectedAccept(clientKey string) string {
	h := sha1.New()
	h.Write([]byte(clientKey + wsMagicGUID))
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

// verifyUpgradeResponse checks the 101 response from the server.
// It returns an error if the response is not a valid WebSocket
// upgrade.
func verifyUpgradeResponse(statusCode int, header map[string][]string, clientKey string) error {
	if statusCode != 101 {
		return fmt.Errorf("speech: websocket upgrade failed: status %d", statusCode)
	}

	accept := header["Sec-WebSocket-Accept"]
	if len(accept) == 0 || accept[0] != expectedAccept(clientKey) {
		return fmt.Errorf("%w: server returned invalid Sec-WebSocket-Accept", ErrProtocol)
	}

	upgrade := header["Upgrade"]
	if len(upgrade) == 0 || strings.ToLower(upgrade[0]) != "websocket" {
		return fmt.Errorf("%w: server did not upgrade to websocket", ErrProtocol)
	}

	return nil
}

// --- frame helpers used by tests ---

// unmaskedFrame builds an unmasked frame (server → client).
// Exported within the package for test fixtures.
func unmaskedFrame(op byte, payload []byte) []byte {
	var buf bytes.Buffer
	buf.WriteByte(0x80 | op)
	length := len(payload)
	switch {
	case length < 126:
		buf.WriteByte(byte(length))
	case length < 65536:
		buf.WriteByte(126)
		var ext [2]byte
		binary.BigEndian.PutUint16(ext[:], uint16(length))
		buf.Write(ext[:])
	default:
		buf.WriteByte(127)
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(length))
		buf.Write(ext[:])
	}
	buf.Write(payload)
	return buf.Bytes()
}

// maskedFrame builds a masked frame (client → server).
// Exported within the package for test fixtures.
func maskedFrame(op byte, payload []byte) []byte {
	var maskKey [4]byte
	// Deterministic mask for test reproducibility.
	maskKey[0] = 0x11
	maskKey[1] = 0x22
	maskKey[2] = 0x33
	maskKey[3] = 0x44

	var buf bytes.Buffer
	buf.WriteByte(0x80 | op)
	length := len(payload)
	switch {
	case length < 126:
		buf.WriteByte(0x80 | byte(length))
	case length < 65536:
		buf.WriteByte(0x80 | 126)
		var ext [2]byte
		binary.BigEndian.PutUint16(ext[:], uint16(length))
		buf.Write(ext[:])
	default:
		buf.WriteByte(0x80 | 127)
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(length))
		buf.Write(ext[:])
	}
	buf.Write(maskKey[:])
	masked := make([]byte, len(payload))
	copy(masked, payload)
	applyMask(masked, maskKey)
	buf.Write(masked)
	return buf.Bytes()
}

// applyMask XORs buf in place with the 4-byte mask key, repeating the
// key as needed (RFC 6455 §5.3).
func applyMask(buf []byte, key [4]byte) {
	for i := range buf {
		buf[i] ^= key[i%4]
	}
}

// urlJoin joins a base URL and a path, ensuring exactly one slash
// between them.
func urlJoin(base, path string) string {
	if strings.HasSuffix(base, "/") && strings.HasPrefix(path, "/") {
		return base + path[1:]
	}
	if !strings.HasSuffix(base, "/") && !strings.HasPrefix(path, "/") {
		return base + "/" + path
	}
	return base + path
}

// parseURL validates and normalises a URL, returning an error if it
// is not a valid absolute ws:// or wss:// URL.
func parseURL(raw string) (*url.URL, error) {
	if !strings.HasPrefix(raw, "ws://") && !strings.HasPrefix(raw, "wss://") {
		return nil, fmt.Errorf("speech: URL must start with ws:// or wss://, got %q", raw)
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("speech: parse URL %q: %w", raw, err)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("speech: URL %q has no host", raw)
	}
	return u, nil
}
