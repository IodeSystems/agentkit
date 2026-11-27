package mcpmgr

// Server→client requests (roots/list, sampling, elicitation) and host→server
// notifications. MCP lets a server ask the client things; a client that never
// answers wedges any server that waits on the reply. Until a host sets a
// RequestHandler, every such request is answered -32601, as before.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// RequestHandler answers one server→client request. serverID is the
// MCPConfig.ID of the asking server. ctx is canceled when that server's
// connection closes or the server cancels this request (RequestID names it;
// a canceled request gets no reply). Return the result to send, or an error:
//   - ErrMethodNotFound: answered -32601 (the same as no handler);
//   - *RequestError: its Code, Message and Data go on the wire;
//   - any other error: -32603 with err.Error() as the message.
//
// Each call runs in its own goroutine, so a slow handler does not stall the
// connection's read loop; replies may go out in any order.
type RequestHandler func(ctx context.Context, serverID, method string, params json.RawMessage) (result any, err error)

type requestIDKey struct{}

// RequestID is the JSON-RPC id (raw JSON: a number or a quoted string) of the
// server→client request a RequestHandler is answering. A server may cancel it
// with notifications/cancelled, which cancels the handler's ctx.
func RequestID(ctx context.Context) (json.RawMessage, bool) {
	id, ok := ctx.Value(requestIDKey{}).(json.RawMessage)
	return id, ok
}

// errCancelledByServer is the cause of a handler ctx canceled by the server's
// notifications/cancelled. Per spec the receiver SHOULD NOT answer then.
var errCancelledByServer = errors.New("mcp: request cancelled by the server")

// inflight tracks the server→client requests a handler is still answering,
// by id, so notifications/cancelled can cancel the right one. The zero value
// is ready. Both transports embed one.
type inflight struct {
	mu sync.Mutex
	m  map[string]context.CancelCauseFunc
}

// begin makes the handler's own ctx (child of life, carrying RequestID) and
// registers the id; call it on the read goroutine so a cancel that
// follows on the wire finds the entry.
func (f *inflight) begin(life context.Context, id json.RawMessage) (context.Context, func()) {
	ctx, cancel := context.WithCancelCause(life)
	ctx = context.WithValue(ctx, requestIDKey{}, id)
	key := idKey(id)
	f.mu.Lock()
	if f.m == nil {
		f.m = map[string]context.CancelCauseFunc{}
	}
	f.m[key] = cancel
	f.mu.Unlock()
	return ctx, func() {
		f.mu.Lock()
		delete(f.m, key)
		f.mu.Unlock()
		cancel(nil)
	}
}

// serveInflight answers one request under ctx from begin; nil means do not reply.
func serveInflight(ctx context.Context, h connRequestHandler, method string, params json.RawMessage) map[string]any {
	body := answer(ctx, h, method, params)
	if errors.Is(context.Cause(ctx), errCancelledByServer) {
		return nil
	}
	return body
}

// cancel handles a notifications/cancelled from the server.
func (f *inflight) cancel(method string, params map[string]any) {
	if method != "notifications/cancelled" {
		return
	}
	raw, err := json.Marshal(params["requestId"])
	if err != nil {
		return
	}
	f.mu.Lock()
	c := f.m[idKey(raw)]
	f.mu.Unlock()
	if c != nil {
		c(errCancelledByServer)
	}
}

// ErrMethodNotFound, returned from a RequestHandler, declines a method: the
// server gets -32601, exactly as if no handler were set.
var ErrMethodNotFound = errors.New("mcp: method not found")

// RequestError is a JSON-RPC error a RequestHandler wants sent to the server
// with its own code (for example -32602 invalid params).
type RequestError struct {
	Code    int
	Message string
	Data    any
}

func (e *RequestError) Error() string { return fmt.Sprintf("%s (code %d)", e.Message, e.Code) }

// connRequestHandler is the transport-level form: the Manager binds serverID.
type connRequestHandler func(ctx context.Context, method string, params json.RawMessage) (any, error)

// answer runs h and returns the JSON-RPC reply body for one request: either
// {"result": ...} or {"error": ...}. A nil h, or ErrMethodNotFound, gives -32601.
func answer(ctx context.Context, h connRequestHandler, method string, params json.RawMessage) map[string]any {
	notFound := map[string]any{"error": map[string]any{
		"code":    -32601,
		"message": "method not found: " + method,
	}}
	if h == nil {
		return notFound
	}
	res, err := h(ctx, method, params)
	switch {
	case err == nil:
		if res == nil {
			res = map[string]any{}
		}
		return map[string]any{"result": res}
	case errors.Is(err, ErrMethodNotFound):
		return notFound
	}
	var re *RequestError
	if errors.As(err, &re) {
		e := map[string]any{"code": re.Code, "message": re.Message}
		if re.Data != nil {
			e["data"] = re.Data
		}
		return map[string]any{"error": e}
	}
	return map[string]any{"error": map[string]any{"code": -32603, "message": err.Error()}}
}

// SetRequestHandler registers the answerer for server→client requests. It is
// on the Manager, not the config, because one host policy (the roots it
// serves) answers for every server and the handler gets the server's id.
// Call before StartServer: handlers are attached at spawn.
func (m *Manager) SetRequestHandler(fn RequestHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onRequest = fn
}

// notifyTimeout bounds Manager.Notify, which takes no ctx.
const notifyTimeout = 30 * time.Second

// Notify sends a notification (for example "notifications/roots/list_changed")
// to a connected server, project- or thread-scoped by the same id CallTool
// takes. params nil sends an empty object.
func (m *Manager) Notify(serverID, method string, params any) error {
	srv, err := m.server(serverID)
	if err != nil {
		return err
	}
	if params == nil {
		params = map[string]any{}
	}
	ctx, cancel := context.WithTimeout(context.Background(), notifyTimeout)
	defer cancel()
	return srv.client.notify(ctx, method, params)
}
