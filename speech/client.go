package speech

import (
	"bufio"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
)

// Config configures a Client.
//
// APIKey is the provider's API key. It is required: a Client without
// one cannot authenticate a stream, and New returns an error rather
// than deferring the failure to the first Open. BaseURL is the
// provider's WebSocket root (for example
// "wss://api.deepgram.com"); it must be an absolute ws:// or wss://
// URL. HTTPClient is the *http.Client used for the upgrade request;
// nil means http.DefaultClient.
type Config struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

// Client opens and owns speech streams against a provider's
// WebSocket endpoint. It is safe for concurrent use: each Open dials
// its own connection.
type Client struct {
	apiKey string
	base   string
	httpc  *http.Client
}

// New validates cfg and returns a *Client. It fails on a missing API
// key or an invalid BaseURL, so a misconfigured host is caught at
// construction rather than at the first stream.
func New(cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("speech: Config.APIKey is required")
	}
	u, err := parseURL(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	httpc := cfg.HTTPClient
	if httpc == nil {
		httpc = http.DefaultClient
	}
	return &Client{
		apiKey: cfg.APIKey,
		base:   u.Scheme + "://" + u.Host + u.Path,
		httpc:  httpc,
	}, nil
}

// Stream is one open WebSocket connection carrying a provider
// session (an STT or TTS stream). Frames are exchanged as whole
// messages: Send and ReadMessage each handle exactly one frame, and
// control traffic (pings, the close handshake) is handled
// transparently.
//
// A Stream is not safe for concurrent use; a host reads and writes it
// from the goroutine that owns the session.
type Stream struct {
	conn *wsConn
}

// Open dials cfg.BaseURL + path and performs the WebSocket upgrade
// handshake. The provider's API key is passed as the Authorization
// header (the form Deepgram expects for its WebSocket endpoints).
//
// The upgrade is done by hand on a raw net.Conn rather than through
// http.Client.Do: after a 101 the http client swaps the underlying
// connection into resp.Body behind a buffered reader, and bytes that
// reader has already swallowed are no longer reachable — so the only
// safe way to take ownership of the stream is to perform the whole
// request by hand and parse the 101 ourselves.
//
// A refused upgrade — 401 or 403 from the provider, or a failed
// dial — is returned as an error that names the failure: an
// authentication refusal is reported as such, not as a generic
// connection error, so a host can distinguish "bad key" from "no
// route".
func (c *Client) Open(ctx context.Context, path string) (*Stream, error) {
	target := urlJoin(c.base, path)
	u, err := parseURL(target)
	if err != nil {
		return nil, err
	}

	key, err := upgradeKey()
	if err != nil {
		return nil, err
	}

	var d net.Dialer
	if c.httpc != nil && c.httpc.Timeout > 0 {
		d.Timeout = c.httpc.Timeout
	}
	conn, err := d.DialContext(ctx, "tcp", u.Host)
	if err != nil {
		return nil, fmt.Errorf("speech: dial %s: %w", u.Host, err)
	}
	defer conn.Close()

	if u.Scheme == "wss" {
		host := u.Host
		if _, _, split := net.SplitHostPort(u.Host); split == nil {
			host += ":443"
		}
		tlsCfg := &tls.Config{ServerName: u.Hostname(), MinVersion: tls.VersionTLS12}
		if c.httpc != nil && c.httpc.Transport != nil {
			if t, ok := c.httpc.Transport.(*http.Transport); ok && t.TLSClientConfig != nil {
				tlsCfg = t.TLSClientConfig.Clone()
				tlsCfg.ServerName = u.Hostname()
			}
		}
		tlsConn := tls.Client(conn, tlsCfg)
		if err := tlsConn.HandshakeContext(ctx); err != nil {
			return nil, fmt.Errorf("speech: TLS handshake with %s: %w", u.Host, err)
		}
		conn = tlsConn
	}

	reqLine := "GET " + u.RequestURI() + " HTTP/1.1\r\n"
	var hdrs strings.Builder
	hdrs.WriteString("Host: " + u.Host + "\r\n")
	hdrs.WriteString("Connection: Upgrade\r\n")
	hdrs.WriteString("Upgrade: websocket\r\n")
	hdrs.WriteString("Sec-WebSocket-Version: 13\r\n")
	hdrs.WriteString("Sec-WebSocket-Key: " + key + "\r\n")
	if c.apiKey != "" {
		hdrs.WriteString("Authorization: token " + c.apiKey + "\r\n")
	}
	hdrs.WriteString("\r\n")

	if _, err := conn.Write([]byte(reqLine + hdrs.String())); err != nil {
		return nil, fmt.Errorf("speech: send upgrade request: %w", err)
	}

	br := bufio.NewReader(conn)
	statusCode, header, err := readHTTPResponse(br)
	if err != nil {
		return nil, fmt.Errorf("speech: read upgrade response: %w", err)
	}

	// A refusal at the upgrade is a final answer: read the body (it
	// often carries the provider's reason) and report it. 401/403
	// are named as authentication failures.
	if statusCode != http.StatusSwitchingProtocols {
		var body []byte
		if cl, ok := header["Content-Length"]; ok && len(cl) > 0 {
			if n, err := strconv.Atoi(cl[0]); err == nil && n > 0 && n <= 1<<10 {
				body = make([]byte, n)
				io.ReadFull(br, body)
			}
		}
		switch statusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			return nil, fmt.Errorf("speech: authentication failure opening %s: %s: %s", path, http.StatusText(statusCode), strings.TrimSpace(string(body)))
		default:
			return nil, fmt.Errorf("speech: provider refused websocket upgrade for %s: %s: %s", path, http.StatusText(statusCode), strings.TrimSpace(string(body)))
		}
	}

	if err := verifyUpgradeResponse(statusCode, header, key); err != nil {
		return nil, err
	}

	wsc := &wsConn{
		conn:    conn,
		br:      br,
		closeCh: make(chan struct{}),
	}
	return &Stream{conn: wsc}, nil
}

// readHTTPResponse reads one HTTP/1.1 response head (status line and
// headers) off a connection, returning the status code and the
// header map. The reader is left positioned at the start of the body
// (which, for a 101, is the WebSocket stream itself).
func readHTTPResponse(br *bufio.Reader) (int, map[string][]string, error) {
	line, err := br.ReadString('\n')
	if err != nil {
		return 0, nil, err
	}
	fields := strings.Fields(line)
	if len(fields) < 2 || !strings.HasPrefix(fields[0], "HTTP/") {
		return 0, nil, fmt.Errorf("%w: malformed status line %q", ErrProtocol, strings.TrimSpace(line))
	}
	code, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, nil, fmt.Errorf("%w: bad status code in %q", ErrProtocol, strings.TrimSpace(line))
	}

	header := make(map[string][]string)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return 0, nil, fmt.Errorf("speech: read response header: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		idx := strings.IndexByte(line, ':')
		if idx < 0 {
			return 0, nil, fmt.Errorf("%w: header line %q has no colon", ErrProtocol, line)
		}
		k := strings.TrimSpace(line[:idx])
		v := strings.TrimSpace(line[idx+1:])
		header[k] = append(header[k], v)
	}
	return code, header, nil
}

// Send writes one binary frame. Audio and JSON envelopes both travel
// as binary frames on Deepgram's WebSocket endpoints.
func (s *Stream) Send(payload []byte) error {
	return s.conn.writeFrame(opBinary, payload)
}

// SendText writes one text frame.
func (s *Stream) SendText(payload string) error {
	return s.conn.writeFrame(opText, []byte(payload))
}

// ReadMessage reads the next application message: a text or binary
// frame, returned as (opcode, payload). Ping frames are answered with
// pong and skipped; pong frames are skipped. A close frame ends the
// stream: it is answered if we have not already sent our own, and a
// *CloseError is returned so the caller can see the code.
func (s *Stream) ReadMessage() (opcode byte, payload []byte, err error) {
	for {
		f, err := s.conn.readFrame()
		if err != nil {
			return 0, nil, err
		}
		switch f.Opcode {
		case opPing:
			if err := s.conn.sendControl(opPong, f.Payload); err != nil {
				return 0, nil, err
			}
		case opPong:
			// Nothing to do: the peer answered one of our pings.
		case opClose:
			code, reason := parseClosePayload(f.Payload)
			// Answer the close unless we closed first.
			_ = s.conn.sendControl(opClose, makeClosePayload(CloseNormalClosure, ""))
			s.conn.closeConn()
			return 0, nil, &CloseError{Code: code, Reason: reason}
		case opText, opBinary:
			return f.Opcode, f.Payload, nil
		default:
			return 0, nil, fmt.Errorf("%w: unknown opcode 0x%x", ErrProtocol, f.Opcode)
		}
	}
}

// Close sends a close frame with code 1000 and tears the connection
// down. It is idempotent.
func (s *Stream) Close() error {
	_ = s.conn.sendControl(opClose, makeClosePayload(CloseNormalClosure, ""))
	s.conn.closeConn()
	return nil
}

// Done is closed when the connection is torn down, by Close, by a
// close frame from the peer, or by a read/write error. A host that
// wants to know the stream ended without polling ReadMessage selects
// on it.
func (s *Stream) Done() <-chan struct{} {
	return s.conn.closeCh
}
