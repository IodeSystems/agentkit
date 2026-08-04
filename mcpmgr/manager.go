package mcpmgr

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// secretFilePlaceholder is the literal token substituted in an MCP
// config's Args/Env with the path of the rendered 0600 secret file.
const secretFilePlaceholder = "{{secret_file}}"

// How we identify in the initialize handshake. Servers log this and a few
// gate behaviour on it. It names the LIBRARY, not the host: this used to say
// "autowork3", which meant every consumer of the package announced itself as
// somebody else's daemon.
const (
	clientName    = "agentkit-mcpmgr"
	clientVersion = "0"
)

// MCPConfig holds the configuration for an MCP server connection.
//
// Scope:
//   - ""        — backwards-compat default, treated as "project"
//   - "project" — one shared instance per integration, spawned at
//     daemon boot. Use when the MCP server is workspace-
//     agnostic OR indexes the project's canonical
//     repo_path.
//   - "thread"  — one instance per (integration, thread) pair,
//     spawned lazily when the thread's first dev session
//     activates. The Args list may contain
//     "{{worktree_root}}" placeholders that the
//     scheduler substitutes with the thread's worktree
//     directory at spawn time. Tears down on thread
//     close. Use for per-workspace indexers (poly-lsp-mcp
//     and friends) where each thread has its own
//     worktree.
type MCPConfig struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Command string   `json:"command"`
	Args    []string `json:"args"`
	Env     []string `json:"env"`
	Timeout int      `json:"timeout"` // seconds, default 30
	Scope   string   `json:"scope"`

	// SecretRef names a row in the secrets table whose plaintext is a
	// JSON object (a flat KV map). When set, spawn() resolves it,
	// renders it to SecretFormat, writes it to a per-server 0600 temp
	// file, and substitutes the literal token {{secret_file}} in every
	// Args and Env entry with that file's path. No secret value ever
	// lands in Args/Env directly — only the file path. Carried in the
	// integration's JSONB config (type=mcp); no schema column.
	SecretRef string `json:"secret_ref"`
	// SecretFormat selects the on-disk encoding of the rendered secret
	// file: one of env|properties|json. Only consulted when SecretRef is
	// set. A server wanting YAML takes json — YAML 1.2 is a superset of
	// JSON.
	SecretFormat string `json:"secret_format"`
}

// MCPTool represents a tool discovered from an MCP server.
type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
	ServerID    string                 `json:"server_id"`
}

// MCPServer wraps a connected MCP client.
type MCPServer struct {
	Config   MCPConfig
	client   *stdioClient
	Tools    []MCPTool
	ready    chan struct{}
	readyErr error
	// secretDir is the per-server temp directory holding the rendered
	// 0600 secret file (empty when SecretRef is unset). Teardown
	// (Close / StopServer / StopThreadServers + the delete+replace
	// branches in StartServer / StartThreadServer) os.RemoveAll's it so
	// app-password files don't leak across restarts.
	secretDir string
	// stderr is the bounded tail of what the process wrote to stderr — the
	// only place a server that refuses to run explains why. See stderr.go.
	stderr *stderrTail
}

// Manager manages MCP server connections, both project-scoped
// (one shared instance per integration) and thread-scoped (one
// instance per (integration, thread) pair). Thread-scoped servers
// are what makes a per-thread editor MCP work: each thread's
// poly-lsp-mcp indexes only that thread's worktree, so file
// operations stay single-threaded within a thread (the dev-lane
// lock already serializes Turns) while threads make progress
// independently.
//
// Key conventions:
//   - servers[cfg.ID]                  — project-scoped, started at boot
//   - threadServers[integID][threadID] — thread-scoped, started lazily
//     when the first dev session for the thread activates
type Manager struct {
	mu            sync.RWMutex
	servers       map[string]*MCPServer
	threadServers map[string]map[string]*MCPServer
	// secrets resolves an MCPConfig.SecretRef into a KV map at spawn
	// time. Nil when the daemon has no secrets store (e.g. config.json
	// absent); spawn() errors clearly if a config sets SecretRef in
	// that mode. Set via NewManagerWithSecrets / SetSecretResolver.
	secrets SecretResolver
	// onNotify receives server→client notifications. Until this existed the
	// manager was request/response only — CallTool and nothing else — so a
	// server could observe something (a file changing under it, a merge
	// conflict appearing) and have no way to say so until the next call.
	// Consumers routed around it instead: dun made sub-agents in-process
	// PURELY so a child could notify its parent.
	//
	// Nil is the safe default and stays the common case: a manager that
	// registers no handler simply drops notifications, exactly as before.
	onNotify func(Notification)
}

// Notification is one server→client message, flattened to what a consumer
// acts on. The raw params are kept because MCP's notification methods are
// open-ended and this package should not become a bottleneck on which ones
// mean something.
type Notification struct {
	ServerID string
	Method   string
	Params   map[string]any
}

// Level is the notifications/message severity, when the method carries one.
// Empty for methods that do not.
func (n Notification) Level() string {
	s, _ := n.Params["level"].(string)
	return s
}

// Text renders the human-readable payload of a notifications/message, which
// is where servers put anything they want a person or a model to read. `data`
// is by spec any JSON value; a bare string is the common case and is returned
// as-is rather than re-encoded with quotes around it.
func (n Notification) Text() string {
	switch v := n.Params["data"].(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

// SetNotificationHandler registers the sink for server→client notifications.
// Call before StartServer: handlers are attached at spawn, so a server
// already running keeps whatever sink it was spawned with.
func (m *Manager) SetNotificationHandler(fn func(Notification)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onNotify = fn
}

func NewManager() *Manager {
	return &Manager{
		servers:       make(map[string]*MCPServer),
		threadServers: make(map[string]map[string]*MCPServer),
	}
}

// StartServer spawns a project-scoped MCP server process and
// connects to it. Idempotent: calling twice for the same cfg.ID
// closes the previous instance.
func (m *Manager) StartServer(ctx context.Context, cfg MCPConfig) error {
	m.mu.Lock()
	if existing, ok := m.servers[cfg.ID]; ok {
		m.mu.Unlock()
		existing.close()
		m.mu.Lock()
		delete(m.servers, cfg.ID)
	}
	m.mu.Unlock()

	srv, err := m.spawn(ctx, cfg)
	if err != nil {
		return err
	}

	m.mu.Lock()
	m.servers[cfg.ID] = srv
	m.mu.Unlock()

	log.Printf("mcp: started %s (%s), discovering tools...", cfg.Name, cfg.ID)
	return nil
}

// StartThreadServer spawns a thread-scoped MCP server. integrationID
// identifies the project_integration the config came from;
// (integrationID, threadID) is the spawn key — a second call for the
// same pair restarts the server. Tool dispatch uses cfg.ID for
// routing, so callers should set cfg.ID = threadServerID(integID,
// threadID) before calling.
func (m *Manager) StartThreadServer(ctx context.Context, integrationID, threadID string, cfg MCPConfig) error {
	m.mu.Lock()
	if byThread, ok := m.threadServers[integrationID]; ok {
		if existing, ok := byThread[threadID]; ok {
			m.mu.Unlock()
			existing.close()
			m.mu.Lock()
			delete(byThread, threadID)
		}
	} else {
		m.threadServers[integrationID] = map[string]*MCPServer{}
	}
	m.mu.Unlock()

	srv, err := m.spawn(ctx, cfg)
	if err != nil {
		return err
	}

	m.mu.Lock()
	m.threadServers[integrationID][threadID] = srv
	m.mu.Unlock()

	log.Printf("mcp: started thread-scoped %s (integration=%s thread=%s)",
		cfg.Name, integrationID, threadID)
	return nil
}

// ServerStarted reports whether a project-scoped MCP server is already
// running for id (== cfg.ID). The Phase 3 decision executor uses this
// to spawn the write server (gmail-exec) on demand WITHOUT the
// idempotent-restart of StartServer tearing down a live process on
// every apply (which would kill in-flight IMAP/SMTP calls).
func (m *Manager) ServerStarted(id string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.servers[id]
	return ok
}

// ThreadServerStarted reports whether a thread-scoped MCP server is
// already running for (integrationID, threadID). The scheduler's
// per-Turn collectTools uses this to avoid re-spawning on every tool
// list.
func (m *Manager) ThreadServerStarted(integrationID, threadID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if byThread, ok := m.threadServers[integrationID]; ok {
		_, ok := byThread[threadID]
		return ok
	}
	return false
}

// StopThreadServers tears down every thread-scoped MCP server bound
// to threadID, across all integrations. Called on thread close.
func (m *Manager) StopThreadServers(threadID string) {
	m.mu.Lock()
	type kill struct {
		integrationID string
		srv           *MCPServer
	}
	var toClose []kill
	for integID, byThread := range m.threadServers {
		if srv, ok := byThread[threadID]; ok {
			toClose = append(toClose, kill{integID, srv})
			delete(byThread, threadID)
		}
	}
	m.mu.Unlock()
	for _, k := range toClose {
		k.srv.close()
		log.Printf("mcp: stopped thread-scoped %s (integration=%s thread=%s)",
			k.srv.Config.Name, k.integrationID, threadID)
	}
}

// GetThreadTools returns the tool set the scheduler should expose to
// developers on threadID: project-scoped tools (unchanged behaviour)
// PLUS any thread-scoped tools bound to this thread. Callers that
// don't care about thread scoping can continue using GetTools.
func (m *Manager) GetThreadTools(threadID string) []MCPTool {
	all := m.GetTools()
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, byThread := range m.threadServers {
		srv, ok := byThread[threadID]
		if !ok {
			continue
		}
		select {
		case <-srv.ready:
			if srv.readyErr != nil {
				log.Printf("mcp: thread-scoped %s ready error: %v", srv.Config.Name, srv.readyErr)
				continue
			}
			all = append(all, srv.Tools...)
		default:
			continue
		}
	}
	return all
}

// ThreadServerID is the convention for cfg.ID on thread-scoped
// servers — composite so the scheduler's resolveTool can recover the
// (integration, thread) pair from a single ServerID field and so tool
// names stay collision-free across threads.
func ThreadServerID(integrationID, threadID string) string {
	return integrationID + ":" + threadID
}

// spawn does the actual subprocess + handshake dance. Shared by
// StartServer and StartThreadServer.
func (m *Manager) spawn(ctx context.Context, cfg MCPConfig) (*MCPServer, error) {
	timeout := time.Duration(cfg.Timeout) * time.Second
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	args := cfg.Args
	env := cfg.Env
	var secretDir string
	if cfg.SecretRef != "" {
		var err error
		args, env, secretDir, err = m.materializeSecret(ctx, cfg)
		if err != nil {
			return nil, err
		}
	}

	mc := newStdioClient(cfg.Command, env, args)

	// Server→client push. Registered before start() so nothing a server emits
	// during initialization is missed. The handler runs on the client's read
	// goroutine, so it hands off and returns rather than doing work here —
	// blocking it would stall every response on this connection.
	m.mu.RLock()
	onNotify := m.onNotify
	m.mu.RUnlock()
	if onNotify != nil {
		id := cfg.ID
		mc.setNotificationHandler(func(method string, params map[string]any) {
			onNotify(Notification{ServerID: id, Method: method, Params: params})
		})
	}

	// The subprocess is NOT tied to ctx — see stdioClient.start. It lives as
	// long as the Manager does; Manager.Close() and StopThreadServers tear it
	// down. ctx bounds the handshake below, nothing more.
	if err := mc.start(); err != nil {
		if secretDir != "" {
			os.RemoveAll(secretDir)
		}
		tail := watchStderr(mc.stderrPipe())
		tail.settle()
		return nil, fmt.Errorf("mcp: start %s: %w%s", cfg.Name, err, tail.suffix())
	}

	// Drain stderr for the life of the process: it is where a server explains
	// itself when it refuses to start, and nobody else reads the pipe.
	tail := watchStderr(mc.stderrPipe())

	handshakeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	if _, err := mc.initialize(handshakeCtx, clientName, clientVersion); err != nil {
		mc.close()
		if secretDir != "" {
			os.RemoveAll(secretDir)
		}
		tail.settle()
		return nil, fmt.Errorf("mcp: initialize %s: %w%s", cfg.Name, err, tail.suffix())
	}

	srv := &MCPServer{
		Config:    cfg,
		client:    mc,
		ready:     make(chan struct{}),
		secretDir: secretDir,
		stderr:    tail,
	}

	go func() {
		toolsCtx, toolsCancel := context.WithTimeout(ctx, timeout)
		defer toolsCancel()

		tools, err := mc.listTools(toolsCtx, cfg.Name)
		if err != nil {
			tail.settle()
			srv.readyErr = fmt.Errorf("mcp: list tools %s: %w%s", cfg.Name, err, tail.suffix())
			close(srv.ready)
			return
		}

		for _, t := range tools {
			srv.Tools = append(srv.Tools, MCPTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: normalizeSchema(t.InputSchema),
				ServerID:    cfg.ID,
			})
		}

		srv.readyErr = nil
		close(srv.ready)
	}()

	return srv, nil
}

// materializeSecret resolves cfg.SecretRef, renders the KV map to
// cfg.SecretFormat, writes it to a 0600 file inside a fresh per-server
// temp dir, and returns Args/Env copies with every {{secret_file}}
// token replaced by that file's path. The caller is responsible for
// os.RemoveAll(secretDir) on teardown. Returns a clear error when no
// resolver is wired (e.g. the daemon has no secrets store) so a
// secret-bearing config fails loudly rather than spawning with an
// empty or literal-placeholder file.
func (m *Manager) materializeSecret(ctx context.Context, cfg MCPConfig) (args, env []string, secretDir string, err error) {
	m.mu.RLock()
	resolver := m.secrets
	m.mu.RUnlock()
	if resolver == nil {
		return nil, nil, "", fmt.Errorf("mcp: %s sets secret_ref=%q but no secret resolver is configured", cfg.Name, cfg.SecretRef)
	}

	kv, err := resolver.Resolve(ctx, cfg.SecretRef)
	if err != nil {
		return nil, nil, "", fmt.Errorf("mcp: %s resolve secret_ref=%q: %w", cfg.Name, cfg.SecretRef, err)
	}

	rendered, err := RenderSecret(cfg.SecretFormat, kv)
	if err != nil {
		return nil, nil, "", fmt.Errorf("mcp: %s: %w", cfg.Name, err)
	}

	dir, err := os.MkdirTemp("", "aw3-mcp-secret-")
	if err != nil {
		return nil, nil, "", fmt.Errorf("mcp: %s secret tmp dir: %w", cfg.Name, err)
	}
	path := filepath.Join(dir, "secret")
	if err := os.WriteFile(path, rendered, 0o600); err != nil {
		os.RemoveAll(dir)
		return nil, nil, "", fmt.Errorf("mcp: %s write secret file: %w", cfg.Name, err)
	}

	args = substitutePlaceholder(cfg.Args, secretFilePlaceholder, path)
	env = substitutePlaceholder(cfg.Env, secretFilePlaceholder, path)
	return args, env, dir, nil
}

// substitutePlaceholder returns a copy of in with every occurrence of
// token replaced by val. Returns nil for a nil/empty slice so the
// transport branch on len()==0 stays intact.
func substitutePlaceholder(in []string, token, val string) []string {
	if len(in) == 0 {
		return in
	}
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = strings.ReplaceAll(s, token, val)
	}
	return out
}

// StopServer disconnects and kills an MCP server.
func (m *Manager) StopServer(id string) {
	m.mu.Lock()
	srv, ok := m.servers[id]
	if !ok {
		m.mu.Unlock()
		return
	}
	delete(m.servers, id)
	m.mu.Unlock()

	srv.close()
	log.Printf("mcp: stopped %s", srv.Config.Name)
}

// ServerReadyErr reports why a project-scoped server discovered no tools, once
// discovery has finished. Returns nil while still discovering, when discovery
// succeeded, or when no such server is running — a caller that wants to
// distinguish those cases should pair it with ServerStarted.
//
// This exists because tool discovery is async: StartServer returning nil only
// means the handshake worked, and a server that then fails ListTools otherwise
// shows up as a silent absence of tools.
func (m *Manager) ServerReadyErr(id string) error {
	m.mu.RLock()
	srv, ok := m.servers[id]
	m.mu.RUnlock()
	if !ok {
		return nil
	}
	select {
	case <-srv.ready:
		return srv.readyErr
	default:
		return nil
	}
}

// ServerStderr returns the tail of what a running server wrote to stderr, or ""
// when it has said nothing (or is not running). Useful for reporting a server
// that started but is misbehaving, where no error was ever returned.
func (m *Manager) ServerStderr(id string) string {
	m.mu.RLock()
	srv, ok := m.servers[id]
	m.mu.RUnlock()
	if !ok || srv.stderr == nil {
		return ""
	}
	return strings.TrimPrefix(srv.stderr.suffix(), " — server stderr:\n")
}

// GetTools returns all tools from all connected MCP servers.
func (m *Manager) GetTools() []MCPTool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var all []MCPTool
	for _, srv := range m.servers {
		// Wait briefly for tool discovery
		select {
		case <-srv.ready:
			if srv.readyErr != nil {
				log.Printf("mcp: %s ready error: %v", srv.Config.Name, srv.readyErr)
				continue
			}
			all = append(all, srv.Tools...)
		default:
			// Still discovering, skip for now
			continue
		}
	}
	return all
}

// CallTool invokes a tool on the appropriate MCP server. ServerID
// may be a project-scoped id (one of m.servers' keys) OR a
// thread-scoped id produced by ThreadServerID(integID, threadID).
// Routes to whichever map holds it.
func (m *Manager) CallTool(ctx context.Context, serverID string, toolName string, args map[string]interface{}) (string, error) {
	m.mu.RLock()
	srv, ok := m.servers[serverID]
	if !ok {
		// Try thread-scoped lookup: serverID = "<integID>:<threadID>".
		// Iterate rather than parse — the integration id is opaque
		// and may itself contain colons.
		for _, byThread := range m.threadServers {
			for _, s := range byThread {
				if s.Config.ID == serverID {
					srv = s
					ok = true
					break
				}
			}
			if ok {
				break
			}
		}
	}
	m.mu.RUnlock()

	if !ok {
		return "", fmt.Errorf("mcp server %s not connected", serverID)
	}

	result, err := srv.client.callTool(ctx, toolName, args)
	if err != nil {
		return "", fmt.Errorf("mcp call %s: %w", toolName, err)
	}

	if result.IsError {
		return "", fmt.Errorf("mcp tool error: %v", contentText(result.Content))
	}

	return renderContent(result.Content), nil
}

// Close disconnects all MCP servers (project + thread scope).
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, srv := range m.servers {
		srv.close()
	}
	for _, byThread := range m.threadServers {
		for _, srv := range byThread {
			srv.close()
		}
	}
}

// close tears down the MCP client and removes any per-server secret
// temp dir. Safe to call when secretDir is empty.
func (s *MCPServer) close() {
	if err := s.client.close(); err != nil {
		log.Printf("mcp: closing %s: %v", s.Config.Name, err)
	}
	if s.secretDir != "" {
		os.RemoveAll(s.secretDir)
	}
}
