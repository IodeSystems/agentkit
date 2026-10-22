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
	token   string // the bearer token accepted; "" means "tok"
}

const fakeChallenge = `Bearer resource_metadata="https://mcp.example/.well-known/oauth-protected-resource", scope="read"`

func (f *fakeHTTPServer) setToken(tok string) { f.mu.Lock(); f.token = tok; f.mu.Unlock() }

func (f *fakeHTTPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.headers = append(f.headers, r.Header.Clone())
	want := f.token
	f.mu.Unlock()
	if want == "" {
		want = "tok"
	}
	if r.Header.Get("Authorization") != "Bearer "+want {
		w.Header().Set("WWW-Authenticate", fakeChallenge)
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
	var ue *UnauthorizedError
	if !errors.Is(err, ErrUnauthorized) || !errors.As(err, &ue) || ue.WWWAuthenticate != fakeChallenge {
		t.Errorf("a static token's 401 must be an *UnauthorizedError with the challenge: %#v", err)
	}
}

// tokenBox is a caller's token cache: Authorization reads it, a refresh
// replaces it.
type tokenBox struct {
	mu        sync.Mutex
	tok       string
	reads     int
	refreshes int
	lastWWW   string
}

func (b *tokenBox) authorization(context.Context) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.reads++
	if b.tok == "" {
		return "", nil
	}
	return "Bearer " + b.tok, nil
}

// refreshTo returns an OnUnauthorized that installs tok.
func (b *tokenBox) refreshTo(tok string) func(context.Context, string) error {
	return func(_ context.Context, www string) error {
		b.mu.Lock()
		defer b.mu.Unlock()
		b.refreshes++
		b.lastWWW = www
		b.tok = tok
		return nil
	}
}

func TestHTTP_AuthorizationHookIsAskedEveryRequestAndOverridesTheStaticHeader(t *testing.T) {
	f := &fakeHTTPServer{}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "tok"}
	m := NewManager()
	cfg := MCPConfig{ID: "r", Name: "r", URL: ts.URL, Headers: map[string]string{"Authorization": "Bearer stale", "X-Keep": "1"},
		Authorization: box.authorization}
	if err := m.StartServer(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	if _, err := m.CallTool(context.Background(), "r", "search", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	m.StopServer("r")
	f.mu.Lock()
	defer f.mu.Unlock()
	for i, h := range f.headers {
		if h.Get("Authorization") != "Bearer tok" || h.Get("X-Keep") != "1" {
			t.Errorf("request %d: %v", i, h)
		}
	}
	box.mu.Lock()
	defer box.mu.Unlock()
	if box.reads < len(f.headers) {
		t.Errorf("the hook must run per request: %d reads for %d requests", box.reads, len(f.headers))
	}
}

func TestHTTP_EmptyAuthorizationSendsNoHeader(t *testing.T) {
	f := &fakeHTTPServer{}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{}
	err := NewManager().StartServer(context.Background(), MCPConfig{ID: "r", Name: "r", URL: ts.URL,
		Headers: map[string]string{"Authorization": "Bearer tok"}, Authorization: box.authorization})
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("want ErrUnauthorized, got %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if h := f.headers[0].Values("Authorization"); len(h) != 0 {
		t.Errorf("an empty hook value must send no Authorization, not the static one: %v", h)
	}
}

func TestHTTP_401OnInitializeRefreshesAndRetries(t *testing.T) {
	f := &fakeHTTPServer{token: "fresh"}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "expired"}
	m := NewManager()
	cfg := MCPConfig{ID: "r", Name: "r", URL: ts.URL, Authorization: box.authorization, OnUnauthorized: box.refreshTo("fresh")}
	if err := m.StartServer(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	defer m.StopServer("r")
	box.mu.Lock()
	defer box.mu.Unlock()
	if box.refreshes != 1 || box.lastWWW != fakeChallenge {
		t.Errorf("one refresh with the challenge: %d %q", box.refreshes, box.lastWWW)
	}
}

// The retry of a request whose answer is an SSE stream, in a session: it must
// carry the session id and protocol version, and the stream must be read.
func TestHTTP_401MidSessionRetriesAStreamingRequest(t *testing.T) {
	f := &fakeHTTPServer{}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "tok"}
	c := newHTTPClient(ts.URL, nil, newAuthHooks(box.authorization, box.refreshTo("tok2")))
	defer c.close()
	ctx := context.Background()
	if err := c.call(ctx, "initialize", map[string]any{}, nil); err != nil {
		t.Fatal(err)
	}
	f.setToken("tok2")
	var out struct {
		Tools []struct{ Name string } `json:"tools"`
	}
	if err := c.call(ctx, "tools/list", map[string]any{}, &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Name != "search" {
		t.Errorf("stream not read after the retry: %+v", out)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if n := len(f.headers); n != 3 {
		t.Fatalf("want initialize, the 401, the retry; got %d requests", n)
	}
	retry := f.headers[2]
	if retry.Get("Authorization") != "Bearer tok2" || retry.Get("Mcp-Session-Id") != "s-1" || retry.Get("MCP-Protocol-Version") != "2025-06-18" {
		t.Errorf("the retry must carry the new token and the session: %v", retry)
	}
}

func TestHTTP_401TwiceIsErrUnauthorizedWithTheChallenge(t *testing.T) {
	f := &fakeHTTPServer{token: "never"}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "a"}
	err := NewManager().StartServer(context.Background(), MCPConfig{ID: "r", Name: "r", URL: ts.URL,
		Authorization: box.authorization, OnUnauthorized: box.refreshTo("b")})
	var ue *UnauthorizedError
	if !errors.As(err, &ue) || !errors.Is(err, ErrUnauthorized) || ue.WWWAuthenticate != fakeChallenge {
		t.Fatalf("want *UnauthorizedError with the challenge, got %#v", err)
	}
	f.mu.Lock()
	n := len(f.headers)
	f.mu.Unlock()
	if n != 2 || box.refreshes != 1 {
		t.Errorf("retry exactly once: %d requests, %d refreshes", n, box.refreshes)
	}
}

func TestHTTP_RefreshErrorIsReturnedWithTheUnauthorized(t *testing.T) {
	f := &fakeHTTPServer{token: "never"}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "a"}
	errLogin := errors.New("login required")
	err := NewManager().StartServer(context.Background(), MCPConfig{ID: "r", Name: "r", URL: ts.URL,
		Authorization: box.authorization, OnUnauthorized: func(context.Context, string) error { return errLogin }})
	if !errors.Is(err, ErrUnauthorized) || !errors.Is(err, errLogin) {
		t.Fatalf("want both the 401 and the hook's error, got %v", err)
	}
}

// N requests that fail together on one expired token cause one refresh.
func TestHTTP_Concurrent401sRefreshOnce(t *testing.T) {
	f := &fakeHTTPServer{}
	ts := httptest.NewServer(f)
	defer ts.Close()
	box := &tokenBox{tok: "tok"}
	c := newHTTPClient(ts.URL, nil, newAuthHooks(box.authorization, box.refreshTo("tok2")))
	defer c.close()
	ctx := context.Background()
	if err := c.call(ctx, "initialize", map[string]any{}, nil); err != nil {
		t.Fatal(err)
	}
	f.setToken("tok2")
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- c.call(ctx, "tools/call", map[string]any{"name": "search"}, nil)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Error(err)
		}
	}
	box.mu.Lock()
	defer box.mu.Unlock()
	if box.refreshes != 1 {
		t.Errorf("want one refresh for one expired token, got %d", box.refreshes)
	}
}
