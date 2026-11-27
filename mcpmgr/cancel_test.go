package mcpmgr

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A server that cancels its own request must see the handler's ctx canceled
// and get no reply, and the handler must be able to read the request id.
func TestServerCancelStopsHandlerAndSuppressesReply(t *testing.T) {
	m := NewManager()
	type seen struct {
		id       string
		hasID    bool
		canceled bool
	}
	res := make(chan seen, 1)
	m.SetRequestHandler(func(ctx context.Context, _, _ string, _ json.RawMessage) (any, error) {
		id, ok := RequestID(ctx)
		select {
		case <-ctx.Done():
			res <- seen{string(id), ok, true}
		case <-time.After(5 * time.Second):
			res <- seen{string(id), ok, false}
		}
		return map[string]any{}, nil
	})
	got := make(chan Notification, 16)
	m.SetNotificationHandler(func(n Notification) { got <- n })
	cmd, env := fakeServer(t, "cancel-in", "")
	if err := m.StartServer(context.Background(), MCPConfig{ID: "srv", Name: "srv", Command: cmd, Env: env, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	r := <-res
	if !r.canceled {
		t.Fatal("handler ctx was not canceled by notifications/cancelled")
	}
	if !r.hasID || r.id != `"c1"` {
		t.Errorf("RequestID = %q, %v; want \"c1\"", r.id, r.hasID)
	}
	// The server echoes any reply it gets as test/reply, in order. A round
	// trip after the handler returned proves none was written.
	if _, err := m.CallTool(context.Background(), "srv", "ping", nil); err != nil {
		t.Fatal(err)
	}
	for len(got) > 0 {
		if n := <-got; n.Method == "test/reply" {
			t.Errorf("a canceled request was answered: %+v", n.Params)
		}
	}
}

// A caller that gives up on tools/call must tell the server, with the id of
// that request.
func TestCallerCancelSendsNotificationsCancelled(t *testing.T) {
	m := NewManager()
	got := make(chan Notification, 16)
	m.SetNotificationHandler(func(n Notification) { got <- n })
	cmd, env := fakeServer(t, "cancel-out", "")
	if err := m.StartServer(context.Background(), MCPConfig{ID: "srv", Name: "srv", Command: cmd, Env: env, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := m.CallTool(ctx, "srv", "hold", nil); err == nil {
		t.Fatal("want a ctx error")
	}
	n := waitNote(t, got, "test/got")
	p, _ := n.Params["params"].(string)
	var cp struct {
		RequestID json.Number `json:"requestId"`
		Reason    string      `json:"reason"`
	}
	if err := json.Unmarshal([]byte(p), &cp); err != nil || cp.RequestID == "" {
		t.Fatalf("bad cancel params %q: %v", p, err)
	}
	if !strings.Contains(cp.Reason, "deadline") {
		t.Errorf("reason = %q, want the ctx error", cp.Reason)
	}
}

// HTTP: server cancel inside the SSE stream cancels the handler; caller
// cancel POSTs notifications/cancelled, and never for initialize.
func TestHTTPCancelBothDirections(t *testing.T) {
	posts := make(chan map[string]any, 16)
	release := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		var msg map[string]any
		_ = json.NewDecoder(r.Body).Decode(&msg)
		method, _ := msg["method"].(string)
		idJSON, _ := json.Marshal(msg["id"])
		switch {
		case method == "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"jsonrpc":"2.0","id":` + string(idJSON) + `,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}}}}`))
		case method == "tools/list":
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {\"jsonrpc\":\"2.0\",\"id\":\"r1\",\"method\":\"roots/list\"}\n\n"))
			w.Write([]byte("data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":\"r1\"}}\n\n"))
			w.Write([]byte("data: {\"jsonrpc\":\"2.0\",\"id\":" + string(idJSON) + ",\"result\":{\"tools\":[]}}\n\n"))
		case method == "tools/call":
			w.Header().Set("Content-Type", "text/event-stream")
			w.(http.Flusher).Flush()
			select {
			case <-release:
			case <-r.Context().Done():
			}
		default:
			if msg["id"] == nil || method == "" {
				posts <- msg
			}
			w.WriteHeader(http.StatusAccepted)
		}
	}))
	defer ts.Close()
	defer close(release)

	m := NewManager()
	res := make(chan bool, 1)
	m.SetRequestHandler(func(ctx context.Context, _, _ string, _ json.RawMessage) (any, error) {
		_, ok := RequestID(ctx)
		select {
		case <-ctx.Done():
			res <- ok
		case <-time.After(5 * time.Second):
			res <- false
		}
		return map[string]any{}, nil
	})
	if err := m.StartServer(context.Background(), MCPConfig{ID: "h", Name: "h", URL: ts.URL, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(m.Close)
	if !<-res {
		t.Fatal("HTTP handler ctx not canceled, or RequestID missing")
	}
	// No reply POST for r1 may have been sent.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := m.CallTool(ctx, "h", "hold", nil); err == nil {
		t.Fatal("want a ctx error")
	}
	nextPost(t, posts, func(p map[string]any) bool {
		if p["id"] == "r1" {
			t.Errorf("a canceled request was answered: %v", p)
		}
		return p["method"] == "notifications/cancelled"
	})
}
