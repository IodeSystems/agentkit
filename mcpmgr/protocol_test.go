package mcpmgr

// Behaviour tests for the spawn → handshake → discover → call path, run
// against the in-repo fake server (fakeserver_test.go) so they need no
// external binary.
//
// Written against the CURRENT mark3labs/mcp-go-backed client, and deliberately
// asserting only what mcpmgr promises its callers — never a detail of the
// JSON-RPC plumbing underneath. That is what lets the client be swapped with
// these tests as the check on the swap rather than a casualty of it.
//
// Until this file existed the protocol layer had no coverage at all: every
// test in the package skipped unless poly-lsp-mcp happened to be on PATH.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// startFake spawns the named scenario as a project-scoped server and returns
// the manager. Uses context.Background() for the spawn: the tools-discovery
// goroutine borrows the caller's ctx, so a t.Cleanup'd cancel would race
// discovery to the finish line.
func startFake(t *testing.T, id, scenario string, timeout int) *Manager {
	t.Helper()
	m := NewManager()
	t.Cleanup(m.Close)
	cmd, env := fakeServer(t, scenario, "")
	if err := m.StartServer(context.Background(), MCPConfig{
		ID: id, Name: "fake-" + scenario, Command: cmd, Env: env, Timeout: timeout,
	}); err != nil {
		t.Fatalf("StartServer(%s): %v", scenario, err)
	}
	return m
}

// waitTools polls until discovery finishes, because GetTools deliberately
// returns nothing (rather than blocking) while a server is still listing.
func waitTools(t *testing.T, m *Manager, want int) []MCPTool {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if tools := m.GetTools(); len(tools) >= want {
			return tools
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("discovery did not yield %d tools in time (got %d)", want, len(m.GetTools()))
	return nil
}

func toolNamed(t *testing.T, tools []MCPTool, name string) MCPTool {
	t.Helper()
	for _, tl := range tools {
		if tl.Name == name {
			return tl
		}
	}
	t.Fatalf("tool %q not discovered; have %v", name, toolNames(tools))
	return MCPTool{}
}

func toolNames(tools []MCPTool) []string {
	var out []string
	for _, tl := range tools {
		out = append(out, tl.Name)
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------------------
// handshake + discovery
// ---------------------------------------------------------------------------

// The handshake is load-bearing and silent when skipped: a client that never
// sends `initialize` gets "client not initialized" on tools/list and the
// integration simply shows up with no tools at all.
func TestSpawnDiscoversTools(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	tools := waitTools(t, m, 3)

	got := toolNames(tools)
	want := []string{"echo", "ping", "rich"}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("tools = %v; want %v", got, want)
	}
	for _, tl := range tools {
		if tl.ServerID != "happy" {
			t.Errorf("%s: ServerID = %q; want happy", tl.Name, tl.ServerID)
		}
	}
	if d := toolNamed(t, tools, "echo").Description; d != "echo the input back" {
		t.Errorf("description = %q; want the server's", d)
	}
	if err := m.ServerReadyErr("happy"); err != nil {
		t.Errorf("ServerReadyErr = %v; want nil", err)
	}
}

// A tool with no arguments must still advertise a valid JSON Schema. A nil
// required marshals to `"required": null`, which llama.cpp rejects outright
// ("type must be array, but is null"), killing the whole chat request.
func TestSchemaNeverAdvertisesNulls(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	ping := toolNamed(t, waitTools(t, m, 3), "ping")

	if ping.InputSchema["type"] != "object" {
		t.Errorf("type = %v; want object", ping.InputSchema["type"])
	}
	if ping.InputSchema["properties"] == nil {
		t.Error("properties is nil; want an empty object")
	}
	if ping.InputSchema["required"] == nil {
		t.Error("required is nil; want an empty array")
	}
}

// ---------------------------------------------------------------------------
// calls
// ---------------------------------------------------------------------------

func TestCallToolRendersEveryContentKind(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	waitTools(t, m, 3)
	ctx := context.Background()

	t.Run("text", func(t *testing.T) {
		got, err := m.CallTool(ctx, "happy", "echo", map[string]any{"text": "hi"})
		if err != nil || got != "hi" {
			t.Fatalf("got %q, %v; want \"hi\", nil", got, err)
		}
	})

	// Multiple content blocks join with newlines rather than the caller
	// getting only the first one.
	t.Run("multiple blocks", func(t *testing.T) {
		got, err := m.CallTool(ctx, "happy", "multi", nil)
		if err != nil || got != "first\nsecond" {
			t.Fatalf("got %q, %v; want \"first\\nsecond\", nil", got, err)
		}
	})

	t.Run("image", func(t *testing.T) {
		got, err := m.CallTool(ctx, "happy", "picture", nil)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, "aGVsbG8=") || !strings.HasPrefix(got, "[image:") {
			t.Fatalf("got %q; want an [image: <data>] placeholder", got)
		}
	})

	// Anything that is neither text nor image must survive as JSON rather
	// than being silently dropped from the result.
	t.Run("other kinds fall back to JSON", func(t *testing.T) {
		got, err := m.CallTool(ctx, "happy", "audible", nil)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, "c291bmQ=") {
			t.Fatalf("got %q; want the audio payload preserved", got)
		}
	})

	// One JSON-RPC frame is one line, and a tool result can legitimately be
	// megabytes (a file read, a search dump). A line reader with a fixed
	// buffer would truncate or error here, and the failure would only show up
	// on the largest, least reproducible results.
	t.Run("multi-megabyte result", func(t *testing.T) {
		got, err := m.CallTool(ctx, "happy", "big", nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 4<<20 {
			t.Fatalf("got %d bytes; want %d", len(got), 4<<20)
		}
		if strings.Trim(got, "x") != "" {
			t.Error("payload corrupted in transit")
		}
	})

	// isError is the model's problem, not the transport's — but it must not
	// be mistaken for success, and the server's explanation has to survive.
	t.Run("isError", func(t *testing.T) {
		_, err := m.CallTool(ctx, "happy", "boom", nil)
		if err == nil {
			t.Fatal("isError result returned nil error")
		}
		if !strings.Contains(err.Error(), "it exploded") {
			t.Fatalf("err = %v; want the server's message", err)
		}
	})
}

// Every in-flight request is keyed by its JSON-RPC id. Get the correlation
// wrong and calls still "work" — they just return each other's answers, which
// no single-threaded test would ever notice.
func TestConcurrentCallsDoNotCrossWires(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	waitTools(t, m, 3)

	const n = 24
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := range n {
		wg.Go(func() {
			want := "payload-" + strconv.Itoa(i)
			got, err := m.CallTool(context.Background(), "happy", "echo",
				map[string]any{"text": want})
			if err != nil {
				errs <- err
			} else if got != want {
				errs <- fmt.Errorf("got %q; want %q", got, want)
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestCallToolUnknownServer(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if _, err := m.CallTool(context.Background(), "nope", "echo", nil); err == nil {
		t.Fatal("want an error for an unconnected server")
	}
}

// A caller's cancelled context must abandon the call promptly instead of
// parking on a response that is never coming.
func TestCallToolHonoursContext(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	waitTools(t, m, 3)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan error, 1)
	go func() {
		_, err := m.CallTool(ctx, "happy", "echo", map[string]any{"text": "x"})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("want an error from a cancelled context")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("CallTool ignored the cancelled context")
	}
}

// ---------------------------------------------------------------------------
// notifications
// ---------------------------------------------------------------------------

// Server→client push, including anything emitted DURING initialize: the sink
// is attached before the handshake, so a server that announces something in
// its first breath is not missed.
func TestNotificationsReachTheHandler(t *testing.T) {
	m := NewManager()
	defer m.Close()

	got := make(chan Notification, 8)
	m.SetNotificationHandler(func(n Notification) { got <- n })

	cmd, env := fakeServer(t, "notify", "")
	if err := m.StartServer(context.Background(), MCPConfig{
		ID: "n", Name: "fake-notify", Command: cmd, Env: env, Timeout: 10,
	}); err != nil {
		t.Fatal(err)
	}

	seen := map[string]Notification{}
	deadline := time.After(10 * time.Second)
	for len(seen) < 2 {
		select {
		case n := <-got:
			if n.Method != "notifications/message" {
				t.Errorf("method = %q; want notifications/message", n.Method)
			}
			if n.ServerID != "n" {
				t.Errorf("ServerID = %q; want n", n.ServerID)
			}
			switch {
			case strings.HasPrefix(n.Text(), "early"):
				seen["early"] = n
			case strings.HasPrefix(n.Text(), "late"):
				seen["late"] = n
			default:
				t.Errorf("unexpected notification text %q", n.Text())
			}
		case <-deadline:
			t.Fatalf("only saw %v", seen)
		}
	}

	if lvl := seen["early"].Level(); lvl != "warning" {
		t.Errorf("early level = %q; want warning", lvl)
	}
	if lvl := seen["late"].Level(); lvl != "info" {
		t.Errorf("late level = %q; want info", lvl)
	}
}

// A manager with no handler registered drops notifications instead of
// blocking the read loop or panicking.
func TestNotificationsWithoutHandlerAreHarmless(t *testing.T) {
	m := startFake(t, "n", "notify", 10)
	waitTools(t, m, 3)
	if _, err := m.CallTool(context.Background(), "n", "echo", map[string]any{"text": "ok"}); err != nil {
		t.Fatalf("unsolicited notifications broke the request path: %v", err)
	}
}

// ---------------------------------------------------------------------------
// failure reporting
// ---------------------------------------------------------------------------

// A server that refuses to run says why on stderr and exits. Without draining
// that pipe the caller gets only "transport closed" and the actual reason —
// usually a one-line fix — is discarded.
func TestSpawnFailureCarriesStderr(t *testing.T) {
	m := NewManager()
	defer m.Close()
	cmd, env := fakeServer(t, "stderr-die", "")
	err := m.StartServer(context.Background(), MCPConfig{
		ID: "d", Name: "fake-die", Command: cmd, Env: env, Timeout: 5,
	})
	if err == nil {
		t.Fatal("want an error from a server that exits during handshake")
	}
	if !strings.Contains(err.Error(), "MYSERVER_TOKEN") {
		t.Fatalf("err = %v; want the server's stderr included", err)
	}
}

func TestInitializeErrorIsReported(t *testing.T) {
	m := NewManager()
	defer m.Close()
	cmd, env := fakeServer(t, "bad-init", "")
	err := m.StartServer(context.Background(), MCPConfig{
		ID: "b", Name: "fake-badinit", Command: cmd, Env: env, Timeout: 5,
	})
	if err == nil {
		t.Fatal("want an error when initialize is rejected")
	}
	if !strings.Contains(err.Error(), "not configured") {
		t.Fatalf("err = %v; want the server's JSON-RPC error message", err)
	}
}

// Discovery is async: StartServer returning nil only means the handshake
// worked. A server that then stalls on tools/list would otherwise present as
// a silent absence of tools, with nothing to point at.
func TestSlowDiscoverySurfacesAsReadyErr(t *testing.T) {
	m := startFake(t, "s", "slow-tools", 1)

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if err := m.ServerReadyErr("s"); err != nil {
			if !strings.Contains(err.Error(), "list tools") {
				t.Fatalf("err = %v; want a list-tools failure", err)
			}
			if got := m.GetTools(); len(got) != 0 {
				t.Errorf("GetTools = %v; want none after a failed discovery", got)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("a stalled tools/list never surfaced as a ready error")
}

func TestServerStartedTracksLifecycle(t *testing.T) {
	m := startFake(t, "happy", "happy", 10)
	if !m.ServerStarted("happy") {
		t.Error("ServerStarted = false after StartServer")
	}
	m.StopServer("happy")
	if m.ServerStarted("happy") {
		t.Error("ServerStarted = true after StopServer")
	}
}

// ---------------------------------------------------------------------------
// teardown
// ---------------------------------------------------------------------------

// Closing stdin is a request, not a guarantee. A server that ignores both the
// closed pipe and SIGTERM has to be killed, or every restart leaks a process
// that still holds the port/lock/worktree the next one needs.
func TestCloseKillsAWedgedServer(t *testing.T) {
	if testing.Short() {
		t.Skip("walks the full graceful→SIGTERM→SIGKILL ladder")
	}
	pidFile := filepath.Join(t.TempDir(), "pid")

	m := NewManager()
	cmd, env := fakeServer(t, "hang", pidFile)
	if err := m.StartServer(context.Background(), MCPConfig{
		ID: "h", Name: "fake-hang", Command: cmd, Env: env, Timeout: 10,
	}); err != nil {
		t.Fatal(err)
	}

	pid := waitForPid(t, pidFile)
	if !processAlive(pid) {
		t.Fatalf("pid %d was never alive", pid)
	}

	start := time.Now()
	m.Close()
	t.Logf("Close() took %s", time.Since(start))

	// Reaping is what makes the process actually go away; poll briefly
	// rather than assuming it is instantaneous.
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		if !processAlive(pid) {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("pid %d survived Close()", pid)
}

func waitForPid(t *testing.T, path string) int {
	t.Helper()
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); {
		if b, err := os.ReadFile(path); err == nil && len(b) > 0 {
			pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
			if err == nil {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("child never wrote its pid file")
	return 0
}

// processAlive reports whether pid is a live, non-reaped process. Signal 0
// checks for existence without delivering anything.
func processAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}

// ---------------------------------------------------------------------------
// thread scoping
// ---------------------------------------------------------------------------

// Thread-scoped servers are the point of the two-map design: a per-worktree
// indexer must be reachable from its own thread only, and must not appear in
// the project-wide tool list.
func TestThreadScopedServerLifecycle(t *testing.T) {
	m := NewManager()
	defer m.Close()

	cmd, env := fakeServer(t, "happy", "")
	const integ, thread = "poly", "t-1"
	id := ThreadServerID(integ, thread)
	if err := m.StartThreadServer(context.Background(), integ, thread, MCPConfig{
		ID: id, Name: "fake-thread", Command: cmd, Env: env, Timeout: 10,
	}); err != nil {
		t.Fatal(err)
	}
	if !m.ThreadServerStarted(integ, thread) {
		t.Fatal("ThreadServerStarted = false after StartThreadServer")
	}

	deadline := time.Now().Add(10 * time.Second)
	var tools []MCPTool
	for time.Now().Before(deadline) {
		if tools = m.GetThreadTools(thread); len(tools) >= 3 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(tools) < 3 {
		t.Fatalf("GetThreadTools = %v; want the thread's tools", toolNames(tools))
	}

	// Project scope must not see them, and another thread must not either.
	if got := m.GetTools(); len(got) != 0 {
		t.Errorf("GetTools = %v; want none (server is thread-scoped)", toolNames(got))
	}
	if got := m.GetThreadTools("t-2"); len(got) != 0 {
		t.Errorf("GetThreadTools(t-2) = %v; want none", toolNames(got))
	}

	// CallTool routes on the composite id, not the project map.
	got, err := m.CallTool(context.Background(), id, "echo", map[string]any{"text": "scoped"})
	if err != nil || got != "scoped" {
		t.Fatalf("CallTool = %q, %v; want \"scoped\", nil", got, err)
	}

	m.StopThreadServers(thread)
	if m.ThreadServerStarted(integ, thread) {
		t.Error("ThreadServerStarted = true after StopThreadServers")
	}
	if _, err := m.CallTool(context.Background(), id, "echo", nil); err == nil {
		t.Error("want an error calling a stopped thread server")
	}
}
