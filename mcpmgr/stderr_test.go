package mcpmgr

import (
	"context"
	"strings"
	"testing"
	"time"
)

// A server that prints why it refuses to run and exits must surface that reason
// in the spawn error. Without the stderr tail the caller sees only the protocol
// layer's view — "transport closed" or EOF — and the actionable line is gone.
func TestSpawn_ErrorCarriesServerStderr(t *testing.T) {
	m := NewManager()
	defer m.Close()

	err := m.StartServer(context.Background(), MCPConfig{
		ID:      "refuser",
		Name:    "refuser",
		Command: "sh",
		Args:    []string{"-c", "echo 'no project name — pass --project' >&2; exit 1"},
		Timeout: 5,
	})
	if err == nil {
		t.Fatal("expected an error from a server that exits immediately")
	}
	if !strings.Contains(err.Error(), "no project name") {
		t.Fatalf("error dropped the server's own explanation: %v", err)
	}
}

// A server that says nothing must not grow a dangling "server stderr:" section.
func TestSpawn_QuietFailureHasNoStderrSection(t *testing.T) {
	m := NewManager()
	defer m.Close()

	err := m.StartServer(context.Background(), MCPConfig{
		ID:      "quiet",
		Name:    "quiet",
		Command: "sh",
		Args:    []string{"-c", "exit 1"},
		Timeout: 5,
	})
	if err == nil {
		t.Fatal("expected an error from a server that exits immediately")
	}
	if strings.Contains(err.Error(), "server stderr") {
		t.Fatalf("empty stderr should add nothing: %v", err)
	}
}

// The tail is bounded: a server that floods stderr before dying costs a fixed
// number of lines, and the LAST ones (the failure) are the ones kept.
func TestStderrTail_BoundedToLastLines(t *testing.T) {
	tail := watchStderr(strings.NewReader(strings.Repeat("noise\n", 200)+"the real reason\n"), nil)
	tail.settle()
	got := tail.suffix()
	if strings.Count(got, "\n") > stderrTailLines {
		t.Fatalf("tail exceeded %d lines: %q", stderrTailLines, got)
	}
	if !strings.Contains(got, "the real reason") {
		t.Fatalf("tail kept the wrong end: %q", got)
	}
}

// OnStderr hears a server's stderr while its handshake is still pending:
// the case where a start hangs and the tail is not reachable yet.
func TestOnStderr_DuringTheHandshake(t *testing.T) {
	m := NewManager()
	defer m.Close()
	heard := make(chan string, 8)
	started := make(chan error, 1)
	go func() {
		started <- m.StartServer(context.Background(), MCPConfig{
			ID: "slow", Name: "slow", Command: "sh", Timeout: 3,
			Args:     []string{"-c", "echo 'warming up' >&2; sleep 10"},
			OnStderr: func(line string) { heard <- line },
		})
	}()
	select {
	case line := <-heard:
		if line != "warming up" {
			t.Fatalf("heard %q", line)
		}
	case err := <-started:
		t.Fatalf("the start ended before stderr was heard: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("no stderr line during the handshake")
	}
	select {
	case <-started:
		t.Fatal("the handshake ended early: the line was not heard during it")
	default:
	}
}
