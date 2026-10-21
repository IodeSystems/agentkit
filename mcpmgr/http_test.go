package mcpmgr

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeHTTPServer is a Streamable HTTP MCP server: bearer auth, a session id,
// JSON replies for some methods and SSE (with a notification first) for others.
type fakeHTTPServer struct {
	mu      sync.Mutex
	headers []http.Header
	deleted bool
}

func (f *fakeHTTPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.headers = append(f.headers, r.Header.Clone())
	f.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer tok" {
		http.Error(w, "bad token", http.StatusUnauthorized)
		return
	}
	if r.Method == http.MethodDelete {
		f.mu.Lock()
		f.deleted = true
		f.mu.Unlock()
		return
	}
	var msg struct {
		ID     json.RawMessage `json:"id"`
		Method string          `json:"method"`
		Params map[string]any  `json:"params"`
	}
	b, _ := io.ReadAll(r.Body)
	_ = json.Unmarshal(b, &msg)
	if len(msg.ID) == 0 {
		w.WriteHeader(http.StatusAccepted)
		return
	}
	reply := func(result any) string {
		out, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": result})
		return string(out)
	}
	switch msg.Method {
	case "initialize":
		w.Header().Set("Mcp-Session-Id", "s-1")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply(map[string]any{"protocolVersion": "2025-06-18",
			"capabilities": map[string]any{"tools": map[string]any{}, "resources": map[string]any{}, "prompts": map[string]any{}}}))
	case "tools/list":
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"data\":\"hello\"}}\n\n")
		fmt.Fprintf(w, "id: 1\ndata: %s\n\n", reply(map[string]any{"tools": []map[string]any{{"name": "search", "description": "find"}}}))
	case "tools/call":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply(map[string]any{"content": []map[string]any{{"type": "text", "text": "found " + fmt.Sprint(msg.Params["arguments"])}}}))
	case "resources/list":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply(map[string]any{"resources": []map[string]any{{"uri": "docs://guide", "name": "guide", "mimeType": "text/markdown"}}}))
	case "resources/read":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply(map[string]any{"contents": []map[string]any{{"uri": msg.Params["uri"], "text": "# Guide"}}}))
	case "prompts/list":
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, reply(map[string]any{"prompts": []map[string]any{{"name": "review", "arguments": []map[string]any{{"name": "path", "required": true}}}}}))
	case "prompts/get":
		w.Header().Set("Content-Type", "application/json")
		args, _ := msg.Params["arguments"].(map[string]any)
		fmt.Fprint(w, reply(map[string]any{"messages": []map[string]any{{"role": "user", "content": map[string]any{"type": "text", "text": "review " + fmt.Sprint(args["path"])}}}}))
	}
}

func TestHTTP_ToolsResourcesPromptsWithBearerAndSession(t *testing.T) {
	f := &fakeHTTPServer{}
	ts := httptest.NewServer(f)
	defer ts.Close()

	m := NewManager()
	var notes []Notification
	var nmu sync.Mutex
	m.SetNotificationHandler(func(n Notification) { nmu.Lock(); notes = append(notes, n); nmu.Unlock() })
	ctx := context.Background()
	cfg := MCPConfig{ID: "remote", Name: "remote", URL: ts.URL, Headers: map[string]string{"Authorization": "Bearer tok"}}
	if err := m.StartServer(ctx, cfg); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(m.GetTools()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if tools := m.GetTools(); len(tools) != 1 || tools[0].Name != "search" {
		t.Fatalf("tools over SSE: %+v (err %v)", tools, m.ServerReadyErr("remote"))
	}
	nmu.Lock()
	if len(notes) != 1 || notes[0].Text() != "hello" {
		t.Errorf("the notification before the response must be delivered: %+v", notes)
	}
	nmu.Unlock()
	if out, err := m.CallTool(ctx, "remote", "search", map[string]any{"q": "x"}); err != nil || !strings.HasPrefix(out, "found ") {
		t.Errorf("call: %q %v", out, err)
	}
	if _, ok := m.Capabilities("remote")["resources"]; !ok {
		t.Error("capabilities not kept")
	}
	rs, err := m.ListResources(ctx, "remote")
	if err != nil || len(rs) != 1 || rs[0].URI != "docs://guide" {
		t.Fatalf("resources: %+v %v", rs, err)
	}
	cs, err := m.ReadResource(ctx, "remote", "docs://guide")
	if err != nil || len(cs) != 1 || cs[0].Text != "# Guide" {
		t.Errorf("read: %+v %v", cs, err)
	}
	ps, err := m.ListPrompts(ctx, "remote")
	if err != nil || len(ps) != 1 || !ps[0].Arguments[0].Required {
		t.Errorf("prompts: %+v %v", ps, err)
	}
	_, msgs, err := m.GetPrompt(ctx, "remote", "review", map[string]string{"path": "a.go"})
	if err != nil || len(msgs) != 1 || msgs[0].Text() != "review a.go" {
		t.Errorf("prompt: %+v %v", msgs, err)
	}
	m.StopServer("remote")

	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.deleted {
		t.Error("close must DELETE the session")
	}
	if h := f.headers[0]; h.Get("Mcp-Session-Id") != "" || h.Get("MCP-Protocol-Version") != "" {
		t.Errorf("initialize must carry neither session nor version: %v", h)
	}
	if h := f.headers[len(f.headers)-1]; h.Get("Mcp-Session-Id") != "s-1" || h.Get("MCP-Protocol-Version") != "2025-06-18" {
		t.Errorf("later requests must echo the session and version: %v", h)
	}
}

func TestHTTP_BadTokenSaysSo(t *testing.T) {
	ts := httptest.NewServer(&fakeHTTPServer{})
	defer ts.Close()
	err := NewManager().StartServer(context.Background(), MCPConfig{ID: "r", Name: "r", URL: ts.URL,
		Headers: map[string]string{"Authorization": "Bearer wrong"}})
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "credentials") {
		t.Errorf("want a 401 that names the credentials, got %v", err)
	}
}
