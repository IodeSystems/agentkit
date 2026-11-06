package mcpmgr

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func waitReady(t *testing.T, m *Manager, id string) {
	t.Helper()
	srv, err := m.server(id)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-srv.ready:
	case <-time.After(5 * time.Second):
		t.Fatal("tool discovery never finished")
	}
}

// A host that cancels the ctx it gave StartServer (a `defer cancel()`) must not
// kill the discovery that StartServer left running in the background.
func TestStartServer_DiscoverySurvivesCallerCancel(t *testing.T) {
	m := NewManager()
	defer m.Close()
	cmd, env := fakeServer(t, "delay-tools", "")
	ctx, cancel := context.WithCancel(context.Background())
	if err := m.StartServer(ctx, MCPConfig{ID: "d", Name: "d", Command: cmd, Env: env, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	cancel()
	waitReady(t, m, "d")
	if err := m.ServerReadyErr("d"); err != nil {
		t.Fatalf("discovery died with the caller's ctx: %v", err)
	}
	if len(m.GetTools()) == 0 {
		t.Fatal("no tools discovered")
	}
}

// One stderr line longer than any scanner cap must not stop the drain: the
// lines after it still reach the tail, and the child is never blocked on a
// full pipe.
func TestStderrTail_OverlongLineDoesNotStopDrain(t *testing.T) {
	tail := watchStderr(io.MultiReader(
		strings.NewReader(strings.Repeat("x", 3<<20)+"\n"),
		strings.NewReader("the real reason\n")), nil)
	tail.settle()
	got := tail.suffix()
	if !strings.Contains(got, "the real reason") {
		t.Fatalf("drain stopped at the long line: %.200q", got)
	}
	if len(got) > stderrLineMax*stderrTailLines {
		t.Fatalf("long line was not truncated: %d bytes", len(got))
	}
}

func aliveCount(pids []int) int {
	n := 0
	for _, p := range pids {
		if syscall.Kill(p, 0) == nil {
			n++
		}
	}
	return n
}

func readPids(t *testing.T, path string) []int {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var pids []int
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if n, err := strconv.Atoi(strings.TrimSpace(sc.Text())); err == nil {
			pids = append(pids, n)
		}
	}
	return pids
}

// Two concurrent StartServer calls for one id must leave exactly one live
// process: the loser's instance used to be overwritten in the map, unreachable
// and never closed.
func TestStartServer_ConcurrentSameIDLeavesOneProcess(t *testing.T) {
	m := NewManager()
	defer m.Close()
	exe, env := fakeServer(t, "happy", "")
	pidFile := filepath.Join(t.TempDir(), "pids")
	cfg := MCPConfig{
		ID: "dup", Name: "dup", Command: "sh", Timeout: 10,
		Args: []string{"-c", `echo $$ >> "$PIDS"; exec "$0"`, exe},
		Env:  append(env, "PIDS="+pidFile),
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.StartServer(context.Background(), cfg); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	pids := readPids(t, pidFile)
	if len(pids) != 2 {
		t.Fatalf("expected 2 spawns, got %v", pids)
	}
	deadline := time.Now().Add(5 * time.Second)
	for aliveCount(pids) > 1 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if n := aliveCount(pids); n != 1 {
		t.Fatalf("%d live processes for one server id, want 1", n)
	}
}

// The same race for the thread-scoped key.
func TestStartThreadServer_ConcurrentSameKeyLeavesOneProcess(t *testing.T) {
	m := NewManager()
	defer m.Close()
	exe, env := fakeServer(t, "happy", "")
	pidFile := filepath.Join(t.TempDir(), "pids")
	cfg := MCPConfig{
		ID: "i:t", Name: "dup", Command: "sh", Timeout: 10,
		Args: []string{"-c", `echo $$ >> "$PIDS"; exec "$0"`, exe},
		Env:  append(env, "PIDS="+pidFile),
	}
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.StartThreadServer(context.Background(), "i", "t", cfg); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	pids := readPids(t, pidFile)
	deadline := time.Now().Add(5 * time.Second)
	for aliveCount(pids) > 1 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if n := aliveCount(pids); len(pids) != 2 || n != 1 {
		t.Fatalf("spawns %v, %d live, want 2 spawns and 1 live", pids, n)
	}
}

// A server that never acknowledges notifications/initialized must fail the
// handshake at the configured timeout, not hang StartServer.
func TestHTTP_StalledInitializedNotificationFailsHandshake(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var msg struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &msg)
		if len(msg.ID) == 0 {
			select {
			case <-r.Context().Done():
			case <-time.After(6 * time.Second):
			}
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		out, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": msg.ID,
			"result": map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}}})
		w.Write(out)
	}))
	defer ts.Close()
	m := NewManager()
	defer m.Close()
	start := time.Now()
	err := m.StartServer(context.Background(), MCPConfig{ID: "s", Name: "s", URL: ts.URL, Timeout: 1})
	if err == nil {
		t.Fatal("handshake succeeded against a server that stalled the initialized notification")
	}
	if d := time.Since(start); d > 4*time.Second {
		t.Fatalf("StartServer took %v, want about the 1s timeout", d)
	}
}

// An image block must not put its base64 payload into the model's context.
func TestRenderContent_ImageShowsTypeAndSizeNotPayload(t *testing.T) {
	got := renderContent([]map[string]any{{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}})
	if got != "[image: image/png, 5 bytes]" {
		t.Fatalf("got %q", got)
	}
}

// A tools/call result carrying only structuredContent is a result, not an
// empty string.
func TestCallTool_StructuredContentOnlyRendersItsJSON(t *testing.T) {
	m := NewManager()
	defer m.Close()
	cmd, env := fakeServer(t, "happy", "")
	if err := m.StartServer(context.Background(), MCPConfig{ID: "s", Name: "s", Command: cmd, Env: env, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	out, err := m.CallTool(context.Background(), "s", "structured", nil)
	if err != nil {
		t.Fatal(err)
	}
	if out != `{"temp":21}` {
		t.Fatalf("got %q", out)
	}
}

// Close must not hold the manager lock while servers wind down (a wedged one
// takes seconds), and must leave nothing registered afterwards.
func TestClose_DoesNotBlockReadersAndClearsServers(t *testing.T) {
	m := NewManager()
	cmd, env := fakeServer(t, "hang", "")
	if err := m.StartServer(context.Background(), MCPConfig{ID: "h", Name: "h", Command: cmd, Env: env, Timeout: 10}); err != nil {
		t.Fatal(err)
	}
	closed := make(chan struct{})
	go func() { m.Close(); close(closed) }()
	time.Sleep(200 * time.Millisecond)
	probe := make(chan bool, 1)
	go func() { probe <- m.ServerStarted("h") }()
	select {
	case started := <-probe:
		if started {
			t.Error("ServerStarted true while Close is running")
		}
	case <-time.After(time.Second):
		t.Fatal("ServerStarted blocked behind Close")
	}
	<-closed
	if m.ServerStarted("h") {
		t.Error("ServerStarted true after Close")
	}
}

// A secret value with a newline would end its own line and start another key.
func TestRenderSecret_RejectsNewlineInjection(t *testing.T) {
	for _, format := range []string{"env", "properties"} {
		if _, err := RenderSecret(format, map[string]string{"A": "a\nB=evil"}); err == nil {
			t.Errorf("%s: value with newline accepted", format)
		}
		if _, err := RenderSecret(format, map[string]string{"A\nB": "x"}); err == nil {
			t.Errorf("%s: key with newline accepted", format)
		}
	}
	if _, err := RenderSecret("json", map[string]string{"A": "a\nB"}); err != nil {
		t.Errorf("json escapes newlines itself: %v", err)
	}
}

// A server→client request on an HTTP stream gets a -32601 reply, as on stdio,
// so a server that waits on it is not wedged.
func TestHTTP_ServerRequestAnsweredMethodNotFound(t *testing.T) {
	replied := make(chan map[string]any, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var msg struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Error  map[string]any  `json:"error"`
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &msg)
		if msg.Method == "" && len(msg.ID) > 0 { // the client's reply to our request
			replied <- msg.Error
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if len(msg.ID) == 0 {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if msg.Method == "tools/call" {
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			io.WriteString(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"sampling/createMessage\",\"params\":{}}\n\n")
			fl.Flush()
			select {
			case e := <-replied:
				replied <- e
			case <-time.After(3 * time.Second):
			}
			out, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": msg.ID,
				"result": map[string]any{"content": []any{map[string]any{"type": "text", "text": "ok"}}}})
			io.WriteString(w, "data: "+string(out)+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var result any = map[string]any{"tools": []any{}}
		if msg.Method == "initialize" {
			result = map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}}
		}
		out, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": result})
		w.Write(out)
	}))
	defer ts.Close()
	m := NewManager()
	defer m.Close()
	if err := m.StartServer(context.Background(), MCPConfig{ID: "s", Name: "s", URL: ts.URL, Timeout: 5}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.CallTool(context.Background(), "s", "x", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case e := <-replied:
		if code, _ := e["code"].(float64); code != -32601 {
			t.Fatalf("reply error %v, want code -32601", e)
		}
	default:
		t.Fatal("the server's request was never answered")
	}
}
