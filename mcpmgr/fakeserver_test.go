package mcpmgr

// An in-repo MCP server, so the spawn/handshake/call path has coverage that
// does not depend on a cross-repo binary. Every pre-existing test in this
// package skips unless `poly-lsp-mcp` is on PATH, which meant the protocol
// layer was effectively untested — a bad place to be while swapping the
// client implementation underneath it.
//
// The server is THIS TEST BINARY re-executed with MCPMGR_FAKE_SERVER set (the
// os/exec TestHelperProcess pattern). That buys hermeticity: no `go build`
// step, no toolchain at test time, no testdata binary to keep in sync — the
// server is compiled by the same `go test` invocation that exercises it.
//
// Scenarios are named by the env var's value; see fakeServerMain.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

const (
	fakeEnvVar     = "MCPMGR_FAKE_SERVER"
	fakePidFileVar = "MCPMGR_FAKE_PIDFILE"

	// fakeProtocolVersion is what the fake negotiates. Deliberately NOT the
	// newest version we know about: a client that only accepts its own latest
	// would pass a test pinned to that latest and still fail against every
	// real server in the wild.
	fakeProtocolVersion = "2025-06-18"
)

func TestMain(m *testing.M) {
	if s := os.Getenv(fakeEnvVar); s != "" {
		fakeServerMain(s)
		return
	}
	os.Exit(m.Run())
}

// fakeServer returns the (command, env) pair that spawns this test binary as
// an MCP server running the named scenario. pidFile, when non-empty, is where
// the child writes its PID — the only way a test can assert the process
// actually died, since the manager does not expose it.
func fakeServer(t *testing.T, scenario, pidFile string) (command string, env []string) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("locating test binary: %v", err)
	}
	env = []string{fakeEnvVar + "=" + scenario}
	if pidFile != "" {
		env = append(env, fakePidFileVar+"="+pidFile)
	}
	return exe, env
}

// ---------------------------------------------------------------------------
// server side
// ---------------------------------------------------------------------------

type fakeConn struct {
	mu  sync.Mutex
	out *bufio.Writer
}

func (c *fakeConn) send(msg map[string]any) {
	b, err := json.Marshal(msg)
	if err != nil {
		panic(err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.out.Write(b)
	c.out.WriteByte('\n')
	c.out.Flush()
}

func (c *fakeConn) result(id any, result any) {
	c.send(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
}

func (c *fakeConn) fail(id any, code int, message string) {
	c.send(map[string]any{
		"jsonrpc": "2.0", "id": id,
		"error": map[string]any{"code": code, "message": message},
	})
}

func (c *fakeConn) notify(method string, params map[string]any) {
	c.send(map[string]any{"jsonrpc": "2.0", "method": method, "params": params})
}

// fakeServerMain is the entry point of the re-executed child. It speaks
// newline-delimited JSON-RPC on stdin/stdout and never returns.
func fakeServerMain(scenario string) {
	if p := os.Getenv(fakePidFileVar); p != "" {
		os.WriteFile(p, []byte(strconv.Itoa(os.Getpid())), 0o644)
	}

	switch scenario {
	case "stderr-die":
		// The case stderrTail exists for: the process explains itself on
		// stderr and exits before the handshake can complete.
		fmt.Fprintln(os.Stderr, "fatal: MYSERVER_TOKEN is not set")
		fmt.Fprintln(os.Stderr, "fatal: refusing to start")
		os.Exit(3)
	case "hang":
		// Swallow SIGTERM so Close() has to escalate all the way to SIGKILL.
		signal.Notify(make(chan os.Signal, 1), syscall.SIGTERM)
	}

	in := bufio.NewReader(os.Stdin)
	conn := &fakeConn{out: bufio.NewWriter(os.Stdout)}

	for {
		line, err := in.ReadString('\n')
		if err != nil {
			if scenario == "hang" {
				// Ignore the closed stdin — a wedged server is exactly what
				// the teardown ladder is for. select{} would trip the
				// deadlock detector, so park on a live timer instead.
				for {
					time.Sleep(time.Second)
				}
			}
			return
		}

		var msg struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			continue
		}

		// A message with a method and no id is a notification from the client.
		if len(msg.ID) == 0 || string(msg.ID) == "null" {
			if msg.Method == "notifications/initialized" && scenario == "notify" {
				conn.notify("notifications/message", map[string]any{
					"level": "info",
					"data":  "late: after initialized",
				})
			}
			continue
		}

		var id any
		json.Unmarshal(msg.ID, &id)

		switch msg.Method {
		case "initialize":
			if scenario == "bad-init" {
				conn.fail(id, -32603, "server is not configured")
				continue
			}
			if scenario == "notify" {
				// Emitted BEFORE the initialize response, to prove the
				// notification sink is wired before the handshake rather
				// than after it.
				conn.notify("notifications/message", map[string]any{
					"level": "warning",
					"data":  "early: during initialize",
				})
			}
			conn.result(id, map[string]any{
				"protocolVersion": fakeProtocolVersion,
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "fake", "version": "1"},
			})

		case "tools/list":
			if scenario == "slow-tools" {
				time.Sleep(30 * time.Second)
			}
			conn.result(id, map[string]any{"tools": fakeTools()})

		case "tools/call":
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			json.Unmarshal(msg.Params, &p)
			conn.result(id, fakeCallResult(p.Name, p.Arguments))

		default:
			conn.fail(id, -32601, "method not found: "+msg.Method)
		}
	}
}

// fakeTools is the advertised tool set. `rich` carries a schema whose keywords
// go beyond {type, properties, required} on purpose — $defs, oneOf,
// additionalProperties and a nested description are exactly what a client that
// decodes inputSchema into a fixed struct silently discards.
func fakeTools() []map[string]any {
	return []map[string]any{
		{
			"name":        "echo",
			"description": "echo the input back",
			"inputSchema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		{
			// No required, no properties: the shape that made llama.cpp
			// reject the whole request when it marshalled to null.
			"name":        "ping",
			"description": "no arguments at all",
			"inputSchema": map[string]any{"type": "object"},
		},
		{
			"name":        "rich",
			"description": "a tool with a schema worth preserving",
			"inputSchema": map[string]any{
				"$schema": "https://json-schema.org/draft/2020-12/schema",
				"title":   "RichArgs",
				"type":    "object",
				"properties": map[string]any{
					"target": map[string]any{
						"$ref":        "#/$defs/target",
						"description": "what to operate on",
					},
					"mode": map[string]any{
						"oneOf": []any{
							map[string]any{"const": "fast"},
							map[string]any{"const": "thorough"},
						},
					},
				},
				"required":             []string{"target"},
				"additionalProperties": false,
				"$defs": map[string]any{
					"target": map[string]any{
						"type":    "string",
						"pattern": "^[a-z]+$",
					},
				},
			},
		},
	}
}

// fakeCallResult covers each content branch CallTool has to render.
func fakeCallResult(name string, args map[string]any) map[string]any {
	switch name {
	case "boom":
		return map[string]any{
			"isError": true,
			"content": []any{
				map[string]any{"type": "text", "text": "it exploded"},
			},
		}
	case "picture":
		return map[string]any{
			"content": []any{
				map[string]any{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
			},
		}
	case "audible":
		// Neither text nor image: must fall through to the JSON-encoded
		// branch rather than being dropped.
		return map[string]any{
			"content": []any{
				map[string]any{"type": "audio", "data": "c291bmQ=", "mimeType": "audio/wav"},
			},
		}
	case "big":
		// A single JSON-RPC frame far past any default buffer. Real tool
		// results (a file read, a search dump) hit this routinely.
		return map[string]any{
			"content": []any{
				map[string]any{"type": "text", "text": strings.Repeat("x", 4<<20)},
			},
		}
	case "multi":
		return map[string]any{
			"content": []any{
				map[string]any{"type": "text", "text": "first"},
				map[string]any{"type": "text", "text": "second"},
			},
		}
	default: // echo
		return map[string]any{
			"content": []any{
				map[string]any{"type": "text", "text": fmt.Sprintf("%v", args["text"])},
			},
		}
	}
}
