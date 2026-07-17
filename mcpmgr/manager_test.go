package mcpmgr_test

// End-to-end test against a real poly-lsp-mcp binary. Skipped when
// the binary isn't built — `go test ./internal/mcpmgr` stays green on
// CI without the cross-repo build. Set POLY_LSP_MCP=/path/to/binary
// to point at an explicit build.
//
// What this catches: the "client not initialized" bug where
// mcpmgr.Manager.StartServer skipped the MCP initialize handshake,
// silently producing zero tools for every integration. Pre-fix the
// test fails with `mcp: initialize ...` or `ListTools` errors; post-
// fix it asserts the six poly-lsp-mcp tools land in GetTools.

import (
	"context"
	"os"
	"os/exec"
	"sort"
	"testing"
	"time"

	"github.com/iodesystems/agentkit/mcpmgr"
)

func polyBinary(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("POLY_LSP_MCP"); p != "" {
		return p
	}
	if p, err := exec.LookPath("poly-lsp-mcp"); err == nil {
		return p
	}
	if _, err := os.Stat("/tmp/poly-lsp-mcp"); err == nil {
		return "/tmp/poly-lsp-mcp"
	}
	t.Skip("poly-lsp-mcp binary not found (set POLY_LSP_MCP, build into PATH, or build to /tmp/poly-lsp-mcp)")
	return ""
}

// TestStartServer_DiscoversPolyLSPTools — load-bearing assertion for
// the initialize-handshake fix. With initialize skipped, GetTools
// returns nothing (the readyErr is logged as "client not
// initialized"). With initialize sent, the expected tools surface.
func TestStartServer_DiscoversPolyLSPTools(t *testing.T) {
	bin := polyBinary(t)
	root := t.TempDir()

	m := mcpmgr.NewManager()
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := mcpmgr.MCPConfig{
		ID:      "test-poly",
		Name:    "poly-lsp-mcp",
		Command: bin,
		Args:    []string{"mcp", "--root", root},
		Timeout: 10,
	}
	if err := m.StartServer(ctx, cfg); err != nil {
		t.Fatalf("StartServer: %v", err)
	}

	// Tool discovery is async (mcpmgr.StartServer kicks off a
	// goroutine for ListTools). Poll a few times rather than
	// inspecting unexported state.
	// poly-lsp-mcp's DEFAULT surface (its legacy 9-tool one is behind
	// --legacy-tools). The names are incidental here — this test is about
	// mcpmgr discovering tools from a real stdio MCP server at all.
	want := map[string]bool{
		"node_query": false,
		"node_read":  false,
		"node_edit":  false,
	}
	var got []string
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		tools := m.GetTools()
		if len(tools) >= len(want) {
			got = got[:0]
			for _, tl := range tools {
				if tl.ServerID == cfg.ID {
					got = append(got, tl.Name)
				}
			}
			if len(got) >= len(want) {
				break
			}
		}
		time.Sleep(100 * time.Millisecond)
	}

	if len(got) < len(want) {
		t.Fatalf("discovered %d tools after deadline, want at least %d: got=%v",
			len(got), len(want), got)
	}
	sort.Strings(got)
	for _, name := range got {
		if _, ok := want[name]; ok {
			want[name] = true
		}
	}
	var missing []string
	for name, seen := range want {
		if !seen {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Errorf("missing expected tools: %v (got: %v)", missing, got)
	}
}

// TestStartServer_FailsOnBadCommand — StartServer returns an error
// when the command can't be exec'd. Catches the case where an
// operator typos the binary path in `aw project integration add` and
// would otherwise see "client not initialized" much later.
func TestStartServer_FailsOnBadCommand(t *testing.T) {
	m := mcpmgr.NewManager()
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	err := m.StartServer(ctx, mcpmgr.MCPConfig{
		ID:      "bogus",
		Name:    "bogus",
		Command: "/does/not/exist/poly-lsp-mcp",
		Args:    []string{"mcp"},
		Timeout: 2,
	})
	if err == nil {
		t.Fatal("StartServer returned nil for missing binary; want error")
	}
}

// TestStartThreadServer_IsolatesByThread — two threads each get
// their own poly-lsp-mcp rooted at a distinct workspace. GetThreadTools
// for thread A returns only thread A's tools (with the right
// ServerID); thread B's tools are invisible to A. ThreadServerStarted
// reports correctly; StopThreadServers tears down only the named
// thread.
func TestStartThreadServer_IsolatesByThread(t *testing.T) {
	bin := polyBinary(t)
	rootA := t.TempDir()
	rootB := t.TempDir()

	m := mcpmgr.NewManager()
	defer m.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	integID := "lsp"
	cfgA := mcpmgr.MCPConfig{
		ID:      mcpmgr.ThreadServerID(integID, "tA"),
		Name:    "poly-A",
		Command: bin,
		Args:    []string{"mcp", "--root", rootA},
		Timeout: 10,
		Scope:   "thread",
	}
	cfgB := mcpmgr.MCPConfig{
		ID:      mcpmgr.ThreadServerID(integID, "tB"),
		Name:    "poly-B",
		Command: bin,
		Args:    []string{"mcp", "--root", rootB},
		Timeout: 10,
		Scope:   "thread",
	}

	if err := m.StartThreadServer(ctx, integID, "tA", cfgA); err != nil {
		t.Fatalf("StartThreadServer A: %v", err)
	}
	if err := m.StartThreadServer(ctx, integID, "tB", cfgB); err != nil {
		t.Fatalf("StartThreadServer B: %v", err)
	}

	if !m.ThreadServerStarted(integID, "tA") {
		t.Error("ThreadServerStarted(tA) = false; want true")
	}
	if !m.ThreadServerStarted(integID, "tB") {
		t.Error("ThreadServerStarted(tB) = false; want true")
	}
	if m.ThreadServerStarted(integID, "tC") {
		t.Error("ThreadServerStarted(tC) = true; want false (never started)")
	}

	// Poll for tool discovery on both. poly-lsp-mcp's default surface is 3
	// tools (node_query/node_read/node_edit); the 9-tool one is behind
	// --legacy-tools. The count is incidental — this test is about thread
	// isolation, not the surface.
	wantToolCount := 3
	deadline := time.Now().Add(8 * time.Second)
	var toolsA, toolsB []mcpmgr.MCPTool
	for time.Now().Before(deadline) {
		toolsA = nil
		toolsB = nil
		for _, tl := range m.GetThreadTools("tA") {
			if tl.ServerID == cfgA.ID {
				toolsA = append(toolsA, tl)
			}
		}
		for _, tl := range m.GetThreadTools("tB") {
			if tl.ServerID == cfgB.ID {
				toolsB = append(toolsB, tl)
			}
		}
		if len(toolsA) >= wantToolCount && len(toolsB) >= wantToolCount {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if len(toolsA) < wantToolCount {
		t.Errorf("thread A tools: got %d, want >=%d", len(toolsA), wantToolCount)
	}
	if len(toolsB) < wantToolCount {
		t.Errorf("thread B tools: got %d, want >=%d", len(toolsB), wantToolCount)
	}

	// Cross-tenancy: thread A's GetThreadTools must NOT include
	// tools whose ServerID is thread B's server.
	for _, tl := range m.GetThreadTools("tA") {
		if tl.ServerID == cfgB.ID {
			t.Errorf("thread A leaked thread B tool: %s", tl.Name)
		}
	}

	// StopThreadServers(tA) tears down only A; B keeps running.
	m.StopThreadServers("tA")
	if m.ThreadServerStarted(integID, "tA") {
		t.Error("after StopThreadServers(tA), still started")
	}
	if !m.ThreadServerStarted(integID, "tB") {
		t.Error("StopThreadServers(tA) also stopped tB; want B untouched")
	}
}

// TestThreadServerID_Composite — the convention is "<integID>:<threadID>".
// Other code depends on this format (resolveTool in the scheduler
// uses the ServerID to route dispatch); pinning it here protects
// against accidental shape changes.
func TestThreadServerID_Composite(t *testing.T) {
	got := mcpmgr.ThreadServerID("integ-1", "thread-abc")
	want := "integ-1:thread-abc"
	if got != want {
		t.Errorf("ThreadServerID = %q; want %q", got, want)
	}
}
