package mcpmgr

// The Streamable HTTP transport (MCP 2025-03-26 and later): every JSON-RPC
// message is a POST to one endpoint URL. The server answers a request with
// either one JSON body or an SSE stream that carries the response — and may
// carry notifications before it. A notification gets 202 Accepted.
//
// The server may hand out a session in the Mcp-Session-Id header of the
// initialize reply; every later request echoes it, and close() ends it with a
// DELETE. After initialize every request carries MCP-Protocol-Version.
//
// Not implemented: the optional GET stream for messages the server sends
// outside any request, server→client requests (sampling, elicitation — this
// client advertises none), resumption by Last-Event-ID, and the older
// HTTP+SSE transport (2024-11-05). Notifications that arrive on a request's
// stream are delivered.
//
// Headers are sent on every request; Authorization: Bearer <token> is how a
// static token authenticates.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const httpCloseTimeout = 3 * time.Second

type httpClient struct {
	url     string
	headers map[string]string
	hc      *http.Client

	mu              sync.Mutex
	sessionID       string
	protocolVersion string

	nextID atomic.Int64

	notifyMu sync.RWMutex
	onNotify func(method string, params map[string]any)

	done      chan struct{}
	closeOnce sync.Once
}

func newHTTPClient(url string, headers map[string]string) *httpClient {
	return &httpClient{url: url, headers: headers, hc: &http.Client{}, done: make(chan struct{})}
}

func (c *httpClient) setNotificationHandler(fn func(method string, params map[string]any)) {
	c.notifyMu.Lock()
	defer c.notifyMu.Unlock()
	c.onNotify = fn
}

func (c *httpClient) newRequest(ctx context.Context, method string, body []byte) (*http.Request, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.url, r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	c.mu.Lock()
	if c.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", c.sessionID)
	}
	if c.protocolVersion != "" {
		req.Header.Set("MCP-Protocol-Version", c.protocolVersion)
	}
	c.mu.Unlock()
	return req, nil
}

func (c *httpClient) post(ctx context.Context, msg any) (*http.Response, error) {
	select {
	case <-c.done:
		return nil, errTransportClosed
	default:
	}
	body, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	req, err := c.newRequest(ctx, http.MethodPost, body)
	if err != nil {
		return nil, err
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		switch resp.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			return nil, fmt.Errorf("mcp: HTTP %d from %s — the server refused the credentials: %s",
				resp.StatusCode, c.url, strings.TrimSpace(string(snippet)))
		case http.StatusNotFound:
			c.mu.Lock()
			expired := c.sessionID != ""
			c.mu.Unlock()
			if expired {
				return nil, fmt.Errorf("mcp: HTTP 404 from %s — the server ended the session; restart the server", c.url)
			}
		}
		return nil, fmt.Errorf("mcp: HTTP %d from %s: %s", resp.StatusCode, c.url, strings.TrimSpace(string(snippet)))
	}
	return resp, nil
}

func (c *httpClient) call(ctx context.Context, method string, params, out any) error {
	id := c.nextID.Add(1)
	resp, err := c.post(ctx, map[string]any{"jsonrpc": jsonrpcVersion, "id": id, "method": method, "params": params})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if method == "initialize" {
		if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
			c.mu.Lock()
			c.sessionID = sid
			c.mu.Unlock()
		}
	}

	var r *rpcResponse
	ct, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if ct == "text/event-stream" {
		r, err = c.readStream(resp.Body, fmt.Sprint(id))
	} else {
		r = &rpcResponse{}
		err = json.NewDecoder(resp.Body).Decode(r)
	}
	if err != nil {
		return fmt.Errorf("mcp: %s reply: %w", method, err)
	}
	if r.Error != nil {
		return r.Error
	}
	if method == "initialize" {
		var ir initializeResult
		if json.Unmarshal(r.Result, &ir) == nil && ir.ProtocolVersion != "" {
			c.mu.Lock()
			c.protocolVersion = ir.ProtocolVersion
			c.mu.Unlock()
		}
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(r.Result, out); err != nil {
		return fmt.Errorf("decode %s result: %w", method, err)
	}
	return nil
}

// readStream reads SSE events until the response to request id arrives,
// handing any notification before it to the handler.
func (c *httpClient) readStream(body io.Reader, id string) (*rpcResponse, error) {
	br := bufio.NewReader(body)
	var data strings.Builder
	for {
		line, err := br.ReadString('\n')
		line = strings.TrimRight(line, "\r\n")
		if line == "" && data.Len() > 0 {
			if r := c.handleEvent(data.String(), id); r != nil {
				return r, nil
			}
			data.Reset()
		} else if v, ok := strings.CutPrefix(line, "data:"); ok {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimPrefix(v, " "))
		}
		if err != nil {
			if data.Len() > 0 {
				if r := c.handleEvent(data.String(), id); r != nil {
					return r, nil
				}
			}
			if err == io.EOF {
				return nil, fmt.Errorf("the stream ended without a response")
			}
			return nil, err
		}
	}
}

// handleEvent returns the event's message when it is the response to id, and
// delivers it when it is a notification.
func (c *httpClient) handleEvent(data, id string) *rpcResponse {
	var msg struct {
		ID     json.RawMessage `json:"id"`
		Method string          `json:"method"`
		Params map[string]any  `json:"params"`
		Result json.RawMessage `json:"result"`
		Error  *rpcError       `json:"error"`
	}
	if json.Unmarshal([]byte(data), &msg) != nil {
		return nil
	}
	if msg.Method == "" {
		if idKey(msg.ID) == id {
			return &rpcResponse{ID: msg.ID, Result: msg.Result, Error: msg.Error}
		}
		return nil
	}
	if len(msg.ID) == 0 {
		c.notifyMu.RLock()
		fn := c.onNotify
		c.notifyMu.RUnlock()
		if fn != nil {
			fn(msg.Method, msg.Params)
		}
	}
	// A server→client request: this client advertised no capability that
	// would make one legal, so it is dropped.
	return nil
}

func (c *httpClient) notify(method string, params any) error {
	resp, err := c.post(context.Background(), map[string]any{"jsonrpc": jsonrpcVersion, "method": method, "params": params})
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.Body.Close()
}

// close ends the session with a DELETE, when the server gave one. A server
// that does not support it answers 405, which is fine.
func (c *httpClient) close() error {
	c.closeOnce.Do(func() {
		close(c.done)
		c.mu.Lock()
		sid := c.sessionID
		c.mu.Unlock()
		if sid == "" {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), httpCloseTimeout)
		defer cancel()
		req, err := c.newRequest(ctx, http.MethodDelete, nil)
		if err != nil {
			return
		}
		if resp, err := c.hc.Do(req); err == nil {
			resp.Body.Close()
		}
	})
	return nil
}
