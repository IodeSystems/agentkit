package mcpmgr

import (
	"bufio"
	"io"
	"strings"
	"sync"
	"time"
)

// A spawned MCP server that refuses to run says why on stderr and exits. The
// stdio transport hands that pipe to nobody, so the handshake fails with the
// only thing the protocol layer can see — "transport closed" — and the actual
// reason is discarded. That turns a one-line fix into a debugging session.
//
// stderrTail drains the pipe (which also keeps a chatty server from blocking on
// a full pipe buffer) and keeps the last few lines so a spawn error can carry
// them. Bounded on both axes: a server that dies after logging a megabyte of
// progress should still cost a fixed amount of memory.
const (
	stderrTailLines = 20
	stderrLineMax   = 500
	// settleWait is how long a failed handshake waits for the dying process to
	// finish writing its complaint. The write and the transport error race —
	// without a short wait the tail is often empty precisely when it matters.
	settleWait = 250 * time.Millisecond
)

// stderrTail is a bounded ring of the last lines a server wrote to stderr.
type stderrTail struct {
	mu    sync.Mutex
	lines []string
	done  chan struct{}
}

// watchStderr drains r in the background for the life of the process. A nil
// reader (no pipe, e.g. an in-process transport) yields a tail that is simply
// always empty, so callers need no nil checks.
func watchStderr(r io.Reader) *stderrTail {
	t := &stderrTail{done: make(chan struct{})}
	if r == nil {
		close(t.done)
		return t
	}
	go func() {
		defer close(t.done)
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 0, 8*1024), 1024*1024)
		for sc.Scan() {
			line := strings.TrimRight(sc.Text(), "\r")
			if strings.TrimSpace(line) == "" {
				continue
			}
			if len(line) > stderrLineMax {
				line = line[:stderrLineMax] + "…"
			}
			t.mu.Lock()
			t.lines = append(t.lines, line)
			if len(t.lines) > stderrTailLines {
				t.lines = t.lines[len(t.lines)-stderrTailLines:]
			}
			t.mu.Unlock()
		}
	}()
	return t
}

// settle waits briefly for the pipe to reach EOF, so a tail read right after a
// failed handshake includes what the exiting process wrote. Returns as soon as
// the drain finishes; never blocks longer than settleWait.
func (t *stderrTail) settle() {
	select {
	case <-t.done:
	case <-time.After(settleWait):
	}
}

// suffix renders the tail for appending to an error, or "" when the server said
// nothing. Multi-line by design: the message that explains the failure is
// usually the whole point, and truncating it to one line would reintroduce the
// problem this exists to solve.
func (t *stderrTail) suffix() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.lines) == 0 {
		return ""
	}
	return " — server stderr:\n" + strings.Join(t.lines, "\n")
}
