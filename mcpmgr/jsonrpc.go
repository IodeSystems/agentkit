package mcpmgr

// The stdio JSON-RPC transport: spawn an MCP server as a subprocess, frame
// newline-delimited JSON-RPC 2.0 over its stdin/stdout, correlate responses to
// requests, and tear the process down when we're done with it.
//
// This replaced github.com/mark3labs/mcp-go, which was the only non-stdlib
// dependency in this package and cost ~27k LOC plus five transitive modules
// (a JSON Schema validator, a URI-template engine, golang.org/x/text) to
// provide the ~350 lines below. We used seven of its methods. The trade is
// deliberate and the boundary is narrow: three request methods and one
// notification, all of which have been stable across every published revision
// of the MCP spec.
//
// What the swap bought beyond the dependency count: tool schemas now survive
// verbatim (see protocol.go), because nothing decodes them into a fixed struct
// on the way through.
//
// Layering: this file knows about JSON-RPC and processes, not about MCP.
// protocol.go sits on top and knows about initialize/tools/call.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const jsonrpcVersion = "2.0"

// The teardown ladder's rungs. Closing stdin is a request; a server that
// ignores it (mid-index, wedged on a lock, or simply not written to notice)
// still has to go, or every restart leaks a process holding the worktree or
// port the next one needs.
const (
	gracefulShutdownTimeout = 2 * time.Second
	forceKillTimeout        = 3 * time.Second
)

var (
	// errTransportClosed means the subprocess is gone — either we closed it or
	// it died on its own. Distinct from a context cancellation, which leaves
	// the connection usable.
	errTransportClosed = errors.New("mcp: transport closed")
	// errChildShutdownTimeout means the process survived SIGKILL, which in
	// practice means it is stuck in uninterruptible I/O.
	errChildShutdownTimeout = errors.New("mcp: child did not exit after forced shutdown")
)

// rpcError is a JSON-RPC error object. Servers put the reason a call was
// rejected here, and it is usually the only useful thing in the exchange, so
// it is surfaced verbatim rather than flattened to a status.
type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *rpcError) Error() string {
	if len(e.Data) > 0 {
		return fmt.Sprintf("%s (code %d): %s", e.Message, e.Code, string(e.Data))
	}
	return fmt.Sprintf("%s (code %d)", e.Message, e.Code)
}

type rpcResponse struct {
	ID     json.RawMessage `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *rpcError       `json:"error"`
}

// stdioClient is one connection to one MCP server subprocess.
type stdioClient struct {
	command string
	args    []string
	env     []string

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr io.ReadCloser

	// writeMu serializes frame writes. Concurrent callers must not interleave
	// halves of two JSON lines on the subprocess's stdin.
	writeMu sync.Mutex

	mu      sync.Mutex
	pending map[string]chan *rpcResponse
	nextID  atomic.Int64

	notifyMu sync.RWMutex
	onNotify func(method string, params map[string]any)

	// done is closed when the connection is finished, by close() or by the
	// read loop noticing the process died. Closing it unblocks every in-flight
	// call, which is the difference between "the server crashed" and "the
	// caller hangs forever".
	done        chan struct{}
	closeOnce   sync.Once
	cleanupOnce sync.Once
}

func newStdioClient(command string, env, args []string) *stdioClient {
	return &stdioClient{
		command: command,
		args:    args,
		env:     env,
		pending: make(map[string]chan *rpcResponse),
		done:    make(chan struct{}),
	}
}

// setNotificationHandler registers the sink for server→client notifications.
// The handler is invoked on the read goroutine, so it must hand off and
// return: blocking it stalls every response on this connection.
func (c *stdioClient) setNotificationHandler(fn func(method string, params map[string]any)) {
	c.notifyMu.Lock()
	defer c.notifyMu.Unlock()
	c.onNotify = fn
}

// start spawns the subprocess and begins reading its stdout.
//
// Deliberately exec.Command, NOT exec.CommandContext: the subprocess must
// outlive whatever context the caller happened to be holding when it started
// the server. Binding the two was a live footgun under the previous
// implementation — a `defer cancel()` in the caller killed the MCP server the
// moment the starting function returned. Process lifetime is owned by close()
// and nothing else.
func (c *stdioClient) start() error {
	cmd := exec.Command(c.command, c.args...)
	// Inherit the parent environment and let the config override entries,
	// which is what MCP server configs assume (they set one or two variables,
	// not a whole environment).
	cmd.Env = append(os.Environ(), c.env...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start %s: %w", c.command, err)
	}

	c.cmd = cmd
	c.stdin = stdin
	c.stderr = stderr
	// No size cap on the reader: a tool result is a single line and can
	// legitimately be megabytes. bufio.Reader grows; bufio.Scanner would not.
	c.stdout = bufio.NewReader(stdout)

	go c.readLoop()
	return nil
}

// stderrPipe exposes the process's stderr so it can be drained and tailed.
// See stderr.go — nobody else reads this pipe, and a server that refuses to
// run explains itself there and nowhere else.
func (c *stdioClient) stderrPipe() io.ReadCloser { return c.stderr }

func (c *stdioClient) closeDone() {
	c.closeOnce.Do(func() { close(c.done) })
}

// readLoop consumes the server's stdout until it dies or we close it.
func (c *stdioClient) readLoop() {
	// Whatever ends the loop, in-flight callers must stop waiting.
	defer c.closeDone()

	for {
		line, err := c.stdout.ReadString('\n')
		if err != nil {
			return
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}

		var head struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.Unmarshal([]byte(line), &head); err != nil {
			// Not our frame. Servers that log to stdout instead of stderr are
			// common enough that this must not be fatal.
			continue
		}
		hasID := len(head.ID) > 0 && string(head.ID) != "null"

		switch {
		case head.Method != "" && !hasID:
			c.dispatchNotification(line)
		case head.Method != "" && hasID:
			// A server→client request (sampling, roots, elicitation). We
			// implement none of them, but it must be ANSWERED: a server that
			// blocks on the reply would otherwise wedge for good.
			c.rejectRequest(head.ID, head.Method)
		default:
			c.deliverResponse(line)
		}
	}
}

func (c *stdioClient) dispatchNotification(line string) {
	c.notifyMu.RLock()
	fn := c.onNotify
	c.notifyMu.RUnlock()
	if fn == nil {
		return
	}
	var n struct {
		Method string         `json:"method"`
		Params map[string]any `json:"params"`
	}
	if err := json.Unmarshal([]byte(line), &n); err != nil {
		return
	}
	fn(n.Method, n.Params)
}

func (c *stdioClient) rejectRequest(id json.RawMessage, method string) {
	_ = c.writeFrame(map[string]any{
		"jsonrpc": jsonrpcVersion,
		"id":      json.RawMessage(id),
		"error": map[string]any{
			"code":    -32601,
			"message": "method not found: " + method + " (client implements no server-initiated requests)",
		},
	})
}

func (c *stdioClient) deliverResponse(line string) {
	var resp rpcResponse
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		return
	}
	key := idKey(resp.ID)

	c.mu.Lock()
	ch, ok := c.pending[key]
	delete(c.pending, key)
	c.mu.Unlock()

	if ok {
		ch <- &resp
	}
}

// idKey normalizes a JSON-RPC id to a map key. We always send numbers, but the
// spec permits strings and a server is free to echo `"3"` for our `3`. Keying
// on the raw bytes would then never match, and the call would hang until its
// context expired — a failure that looks exactly like a slow server.
func idKey(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if strings.HasPrefix(s, `"`) {
		var unq string
		if json.Unmarshal(raw, &unq) == nil {
			return unq
		}
	}
	return s
}

// writeFrame marshals one message and writes it as a single line.
func (c *stdioClient) writeFrame(msg any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	b = append(b, '\n')

	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if c.stdin == nil {
		return errTransportClosed
	}
	if _, err := c.stdin.Write(b); err != nil {
		return fmt.Errorf("write request: %w", err)
	}
	return nil
}

// call sends a request and waits for its response. out, when non-nil, receives
// the unmarshalled result.
func (c *stdioClient) call(ctx context.Context, method string, params, out any) error {
	select {
	case <-c.done:
		return errTransportClosed
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	id := c.nextID.Add(1)
	key := fmt.Sprint(id)

	ch := make(chan *rpcResponse, 1)
	c.mu.Lock()
	c.pending[key] = ch
	c.mu.Unlock()
	abandon := func() {
		c.mu.Lock()
		delete(c.pending, key)
		c.mu.Unlock()
	}

	if err := c.writeFrame(map[string]any{
		"jsonrpc": jsonrpcVersion,
		"id":      id,
		"method":  method,
		"params":  params,
	}); err != nil {
		abandon()
		return err
	}

	var resp *rpcResponse
	select {
	case resp = <-ch:
	case <-ctx.Done():
		abandon()
		return ctx.Err()
	case <-c.done:
		// Drain first: a valid response can land in the same instant the read
		// loop notices EOF, and reporting "transport closed" for a call that
		// actually succeeded is worse than the crash it is reporting.
		select {
		case resp = <-ch:
		default:
			abandon()
			return errTransportClosed
		}
	}

	if resp.Error != nil {
		return resp.Error
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(resp.Result, out); err != nil {
		return fmt.Errorf("decode %s result: %w", method, err)
	}
	return nil
}

// notify sends a fire-and-forget notification. No id, no reply.
func (c *stdioClient) notify(method string, params any) error {
	select {
	case <-c.done:
		return errTransportClosed
	default:
	}
	return c.writeFrame(map[string]any{
		"jsonrpc": jsonrpcVersion,
		"method":  method,
		"params":  params,
	})
}

// close tears the subprocess down, escalating until it is actually gone:
// close stdin → wait → SIGTERM → wait → SIGKILL → wait.
//
// Safe to call repeatedly and concurrently with the read loop closing done.
// The cleanup runs exactly once regardless of which path got there first —
// skipping it when the read loop won the race would leak the file descriptors
// and leave a zombie.
func (c *stdioClient) close() error {
	c.closeDone()

	var closeErr error
	c.cleanupOnce.Do(func() {
		if c.stdin != nil {
			if err := c.stdin.Close(); err != nil {
				closeErr = fmt.Errorf("close stdin: %w", err)
			}
		}
		if c.stderr != nil {
			c.stderr.Close()
		}
		if c.cmd == nil {
			return
		}

		waitCh := make(chan error, 1)
		go func() { waitCh <- c.cmd.Wait() }()

		if _, exited := waitExit(waitCh, gracefulShutdownTimeout); exited {
			return
		}
		if c.cmd.Process != nil {
			_ = c.cmd.Process.Signal(syscall.SIGTERM)
		}
		if _, exited := waitExit(waitCh, forceKillTimeout); exited {
			return
		}
		if c.cmd.Process != nil {
			if err := c.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) && closeErr == nil {
				closeErr = fmt.Errorf("kill process: %w", err)
			}
		}
		if _, exited := waitExit(waitCh, forceKillTimeout); !exited && closeErr == nil {
			closeErr = errChildShutdownTimeout
		}
	})
	return closeErr
}

// waitExit reports whether the process exited within timeout. The exit status
// itself is discarded: we are killing a long-running server, so "signal:
// killed" and a nonzero exit are the expected outcomes, not failures.
func waitExit(waitCh <-chan error, timeout time.Duration) (error, bool) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case err := <-waitCh:
		return err, true
	case <-timer.C:
		return nil, false
	}
}
