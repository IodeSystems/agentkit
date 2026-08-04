package mcpmgr

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// End to end through the real stack: a spawned MCP server pushes an
// unsolicited notification and the manager delivers it. Until this seam
// existed the manager was request/response only, and consumers routed around
// it — dun made sub-agents in-process purely so a child could reach a parent.
//
// Uses poly-lsp-mcp because it is the server that emits one: it watches the
// workspace and announces merge conflicts appearing.
func TestNotificationsReachTheManager(t *testing.T) {
	bin, err := exec.LookPath("poly-lsp-mcp")
	if err != nil {
		t.Skip("poly-lsp-mcp not on PATH")
	}
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module n\ngo 1.21\n")
	write("w.go", "package n\n\nfunc A() {}\n")

	got := make(chan Notification, 8)
	m := NewManager()
	m.SetNotificationHandler(func(n Notification) { got <- n })
	defer m.Close()

	if err := m.StartServer(context.Background(), MCPConfig{
		ID: "code", Command: bin, Args: []string{"mcp", "--root", dir}, Timeout: 30,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}

	// A conflict appears out of band — nobody asked the server anything.
	write("w.go", "package n\n\n<<<<<<< HEAD\nfunc A() {}\n=======\nfunc B() {}\n>>>>>>> feat\n")

	select {
	case n := <-got:
		if n.ServerID != "code" {
			t.Errorf("the notification should name its server; got %q", n.ServerID)
		}
		if n.Method != "notifications/message" {
			t.Errorf("method = %q", n.Method)
		}
		if n.Level() != "warning" {
			t.Errorf("a new conflict is a warning; got %q", n.Level())
		}
		if txt := n.Text(); txt == "" || !contains(txt, "merge conflict") {
			t.Errorf("text should describe the conflict; got %q", txt)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("no notification arrived — the server→client path is not connected")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
