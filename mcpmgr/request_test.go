package mcpmgr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// startRootsServer starts the fake "roots" scenario. The server asks
// roots/list right after notifications/initialized; the client's answer comes
// back as a "test/reply" notification on the returned channel.
func startRootsServer(t *testing.T, m *Manager, caps map[string]any) <-chan Notification {
	t.Helper()
	got := make(chan Notification, 16)
	m.SetNotificationHandler(func(n Notification) { got <- n })
	cmd, env := fakeServer(t, "roots", "")
	if err := m.StartServer(context.Background(), MCPConfig{
		ID: "srv", Name: "srv", Command: cmd, Env: env, Timeout: 10, ClientCapabilities: caps,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(m.Close)
	return got
}

func waitNote(t *testing.T, ch <-chan Notification, method string) Notification {
	t.Helper()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case n := <-ch:
			if n.Method == method {
				return n
			}
		case <-timeout:
			t.Fatalf("no %s notification", method)
		}
	}
}

func TestRequestHandlerAnswerReachesTheServer(t *testing.T) {
	m := NewManager()
	var gotServer, gotMethod string
	m.SetRequestHandler(func(_ context.Context, serverID, method string, _ json.RawMessage) (any, error) {
		gotServer, gotMethod = serverID, method
		return map[string]any{"roots": []map[string]any{{"uri": "file:///w", "name": "w"}}}, nil
	})
	got := startRootsServer(t, m, nil)
	n := waitNote(t, got, "test/reply")
	if gotServer != "srv" || gotMethod != "roots/list" {
		t.Errorf("handler saw %q %q", gotServer, gotMethod)
	}
	if n.Params["id"] != `"r1"` || !strings.Contains(fmt.Sprint(n.Params["result"]), "file:///w") {
		t.Errorf("server did not get the handler's result: %+v", n.Params)
	}
}

func TestNoRequestHandlerStillAnswersMinus32601(t *testing.T) {
	m := NewManager()
	got := startRootsServer(t, m, nil)
	n := waitNote(t, got, "test/reply")
	if !strings.Contains(fmt.Sprint(n.Params["error"]), "-32601") {
		t.Errorf("a server request with no handler must get -32601, got %+v", n.Params)
	}
}

func TestRequestHandlerErrorsMapToJSONRPCCodes(t *testing.T) {
	for name, tc := range map[string]struct {
		err  error
		want string
	}{
		"sentinel declines as method-not-found": {ErrMethodNotFound, "-32601"},
		"typed error keeps its code":            {&RequestError{Code: -32602, Message: "bad uri"}, "-32602"},
		"plain error is internal error":         {errors.New("boom"), "-32603"},
	} {
		t.Run(name, func(t *testing.T) {
			m := NewManager()
			m.SetRequestHandler(func(context.Context, string, string, json.RawMessage) (any, error) { return nil, tc.err })
			got := startRootsServer(t, m, nil)
			n := waitNote(t, got, "test/reply")
			if !strings.Contains(fmt.Sprint(n.Params["error"]), tc.want) {
				t.Errorf("want %s on the wire, got %+v", tc.want, n.Params)
			}
		})
	}
}

// A handler that blocks must not stop the read loop: the notification below
// is read by the same loop that is waiting on the handler's goroutine.
func TestSlowRequestHandlerDoesNotStallTheReadLoop(t *testing.T) {
	m := NewManager()
	release := make(chan struct{})
	defer close(release)
	m.SetRequestHandler(func(ctx context.Context, _, _ string, _ json.RawMessage) (any, error) {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return map[string]any{}, nil
	})
	got := startRootsServer(t, m, nil)
	// tools/list answering proves the loop still reads while the handler waits.
	deadline := time.Now().Add(5 * time.Second)
	for !m.ServerReady("srv") && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !m.ServerReady("srv") {
		t.Fatal("a blocked request handler stalled the read loop")
	}
	if err := m.Notify("srv", "notifications/roots/list_changed", nil); err != nil {
		t.Fatal(err)
	}
	waitNote(t, got, "test/got")
}

func TestRequestHandlerCtxIsCanceledWhenTheServerCloses(t *testing.T) {
	m := NewManager()
	started := make(chan context.Context, 1)
	m.SetRequestHandler(func(ctx context.Context, _, _ string, _ json.RawMessage) (any, error) {
		started <- ctx
		<-ctx.Done()
		return nil, ctx.Err()
	})
	startRootsServer(t, m, nil)
	var ctx context.Context
	select {
	case ctx = <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("handler not called")
	}
	m.StopServer("srv")
	select {
	case <-ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("handler ctx outlived the connection")
	}
}

func TestHostNotifyReachesTheServer(t *testing.T) {
	m := NewManager()
	got := startRootsServer(t, m, nil)
	if err := m.Notify("srv", "notifications/roots/list_changed", nil); err != nil {
		t.Fatal(err)
	}
	waitNote(t, got, "test/got")
	if err := m.Notify("nope", "x", nil); err == nil {
		t.Error("Notify to an unknown server must fail")
	}
}

func TestClientCapabilitiesAreSentInInitialize(t *testing.T) {
	for name, tc := range map[string]struct {
		caps map[string]any
		want string
	}{
		"unset stays the empty object":     {nil, `{}`},
		"set is sent verbatim":             {map[string]any{"roots": map[string]any{"listChanged": true}}, `{"roots":{"listChanged":true}}`},
		"empty map stays the empty object": {map[string]any{}, `{}`},
	} {
		t.Run(name, func(t *testing.T) {
			m := NewManager()
			startRootsServer(t, m, tc.caps)
			info, ok := m.ServerInfo("srv")
			if !ok {
				t.Fatal("no server info")
			}
			var raw struct {
				Seen json.RawMessage `json:"clientCapsSeen"`
			}
			if err := json.Unmarshal(info.Raw, &raw); err != nil {
				t.Fatal(err)
			}
			if string(raw.Seen) != tc.want {
				t.Errorf("server saw capabilities %s, want %s", raw.Seen, tc.want)
			}
		})
	}
}

// httpRootsServer asks roots/list inside the tools/list SSE stream, and
// reports every message the client POSTs back (the answer, notifications).
func httpRootsServer(t *testing.T) (*httptest.Server, <-chan map[string]any) {
	posts := make(chan map[string]any, 16)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			return
		}
		var msg map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &msg)
		method, _ := msg["method"].(string)
		id, hasID := msg["id"]
		idJSON, _ := json.Marshal(id)
		switch {
		case method == "initialize":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"seen":%s}}`,
				idJSON, mustJSON(msg["params"].(map[string]any)["capabilities"]))
		case method == "tools/list":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"r1\",\"method\":\"roots/list\"}\n\n")
			fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"tools\":[]}}\n\n", idJSON)
		default:
			if !hasID || method == "" {
				posts <- msg
			}
			w.WriteHeader(http.StatusAccepted)
		}
	}))
	t.Cleanup(ts.Close)
	return ts, posts
}

func mustJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func nextPost(t *testing.T, posts <-chan map[string]any, pred func(map[string]any) bool) map[string]any {
	t.Helper()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case p := <-posts:
			if pred(p) {
				return p
			}
		case <-timeout:
			t.Fatal("expected POST never arrived")
		}
	}
}

func TestHTTP_RequestHandlerAnswerIsPostedBack(t *testing.T) {
	ts, posts := httpRootsServer(t)
	m := NewManager()
	m.SetRequestHandler(func(_ context.Context, serverID, method string, _ json.RawMessage) (any, error) {
		return map[string]any{"roots": []any{}, "via": serverID + method}, nil
	})
	if err := m.StartServer(context.Background(), MCPConfig{ID: "h", Name: "h", URL: ts.URL,
		ClientCapabilities: map[string]any{"roots": map[string]any{}}}); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	p := nextPost(t, posts, func(p map[string]any) bool { return p["id"] == "r1" })
	if res, _ := p["result"].(map[string]any); res["via"] != "hroots/list" {
		t.Errorf("handler result not posted back: %+v", p)
	}
	if err := m.Notify("h", "notifications/roots/list_changed", nil); err != nil {
		t.Fatal(err)
	}
	nextPost(t, posts, func(p map[string]any) bool { return p["method"] == "notifications/roots/list_changed" })
	info, _ := m.ServerInfo("h")
	if !strings.Contains(string(info.Raw), `"seen":{"roots":{}}`) {
		t.Errorf("capabilities not sent over HTTP: %s", info.Raw)
	}
}

func TestHTTP_NoRequestHandlerPostsMinus32601(t *testing.T) {
	ts, posts := httpRootsServer(t)
	m := NewManager()
	if err := m.StartServer(context.Background(), MCPConfig{ID: "h", Name: "h", URL: ts.URL}); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	p := nextPost(t, posts, func(p map[string]any) bool { return p["id"] == "r1" })
	if e, _ := p["error"].(map[string]any); e["code"] != float64(-32601) {
		t.Errorf("want -32601 posted back, got %+v", p)
	}
}
