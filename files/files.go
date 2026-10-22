// Package files is a configurable file toolset an agent host mounts: read,
// edit, write, grep and list (delete and move opt-in), rooted at one directory.
//
// Each enabled tool is an llm.ToolDef plus a branch of one dispatcher. The host
// adds Defs() to its tool list and wraps its dispatcher with Wrap — the same
// decorator shape as every other agentkit dispatcher.
//
// Lessons carried over from aw4's file tools, where each one was measured:
//
//   - Reads are capped, and a read that names no range is capped harder. The
//     bound is for the context window, not memory: one call must not be able to
//     fill a window the agent has no way to know is full. A capped result says
//     the file's length and the exact range to ask for next.
//   - Line numbers are the shared coordinate: read prints them, grep reports
//     them, offset takes them, an edit miss quotes them.
//   - A refusal names what WOULD work: a missing path lists the nearest
//     directory and same-named files elsewhere; an edit miss shows the closest
//     region and the lines that differ; a bad regex is searched literally and
//     the result says so.
//   - Writes never land inside .git (a hook written there runs on the next
//     commit), and generated files ("Code generated … DO NOT EDIT") are refused
//     because the next generator run overwrites the edit.
//
// stdlib only, like the rest of the module. `rg` and `git` are used when they
// are on PATH and the package works without either.
package files

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/iodesystems/agentkit/llm"
)

// Tool names, before the prefix.
const (
	Read   = "read"
	Edit   = "edit"
	Write  = "write"
	Grep   = "grep"
	List   = "list"
	Delete = "delete"
	Move   = "move"
)

// DefaultTools is the set enabled when Config.Tools is nil. Delete and move are
// opt-in: neither is needed to make a change, and both lose work when wrong.
var DefaultTools = []string{Read, Edit, Write, Grep, List}

// readOnlyTools are what a ReadOnly config keeps.
var readOnlyTools = map[string]bool{Read: true, Grep: true, List: true}

// allTools fixes the order Defs reports.
var allTools = []string{Read, Edit, Write, Grep, List, Delete, Move}

// DefaultPrefix makes `file_read`, `file_edit`, …
//
// Not `file.`: OpenAI and Anthropic both require function names to match
// ^[a-zA-Z0-9_-]+$, and a dot is refused by the provider, not by agentkit.
const DefaultPrefix = "file_"

// Default caps. See read.go for the measurements behind the read caps.
const (
	DefaultReadMax     = 64 * 1024
	DefaultUnrangedMax = 8 * 1024
	DefaultGrepMax     = 100
	DefaultListMax     = 200
)

// Errors a Refusal hook can test with errors.Is.
var (
	// ErrEscape: the path resolves outside Root (by `..`, an absolute path, or
	// a symlink whose target is outside).
	ErrEscape = errors.New("outside the root")
	// ErrNotFound: the path does not exist.
	ErrNotFound = errors.New("does not exist")
)

// Config is one mounted toolset.
type Config struct {
	// Root is the directory every path is relative to. Required.
	Root string
	// Tools enables tools by bare name (Read, Edit, …). nil → DefaultTools.
	Tools []string
	// ReadOnly keeps only read, grep and list, whatever Tools says.
	ReadOnly bool
	// NamePrefix is prepended to each tool name. "" → DefaultPrefix. It must
	// keep names inside [a-zA-Z0-9_-].
	NamePrefix string

	// Caps. Zero → the Default* value.
	ReadMax     int // bytes one ranged read returns
	UnrangedMax int // bytes a read with no offset/limit returns
	GrepMax     int // matches one grep returns
	ListMax     int // entries one list returns

	// ExtraIgnore are gitignore-syntax patterns, relative to Root, hidden from
	// grep and list on top of .gitignore.
	ExtraIgnore []string

	// AllowGenerated lets edit/write change files whose header says they are
	// generated. Off: such an edit is overwritten by the next generator run.
	AllowGenerated bool
	// OverwriteUnread lets write replace a file this toolset has not read,
	// without the model passing overwrite:true.
	OverwriteUnread bool
	// NoRipgrep makes grep use the Go engine even when rg is on PATH.
	NoRipgrep bool

	// Resolve replaces path resolution for path arguments: rel in, absolute
	// path out. A host that reads another tree (a branch, a dependency) plugs
	// in here. It owns its own safety; Under is exported to fall back on.
	Resolve func(rel string) (abs string, err error)
	// Refusal builds the text for a path refusal (reason wraps ErrEscape or
	// ErrNotFound and carries the default text). nil → reason.Error().
	Refusal func(path string, reason error) string
	// AfterWrite runs after edit/write/delete/move with the root-relative
	// paths changed (both sides of a move).
	AfterWrite func(paths []string)
}

// Toolset is a mounted Config. One per agent session: it tracks which files
// that session has read, which is what write's overwrite guard consults.
type Toolset struct {
	cfg     Config
	root    string // absolute, symlinks resolved
	enabled map[string]bool
	byName  map[string]string // model-facing name → bare name
	ignore  *matcher          // ExtraIgnore
	rg      string            // rg binary path, "" = Go engine
	// rgFailed counts rg runs that failed and fell back to the Go engine.
	rgFailed atomic.Int64

	mu   sync.Mutex
	seen map[string]bool // real paths read or written by this toolset
}

var toolNameRE = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// New mounts cfg.
func New(cfg Config) (*Toolset, error) {
	if cfg.Root == "" {
		return nil, errors.New("files: Config.Root is required")
	}
	root, err := realRoot(cfg.Root)
	if err != nil {
		return nil, fmt.Errorf("files: root: %w", err)
	}
	if cfg.NamePrefix == "" {
		cfg.NamePrefix = DefaultPrefix
	}
	if cfg.ReadMax <= 0 {
		cfg.ReadMax = DefaultReadMax
	}
	if cfg.UnrangedMax <= 0 {
		cfg.UnrangedMax = DefaultUnrangedMax
	}
	if cfg.UnrangedMax > cfg.ReadMax {
		cfg.UnrangedMax = cfg.ReadMax
	}
	if cfg.GrepMax <= 0 {
		cfg.GrepMax = DefaultGrepMax
	}
	if cfg.ListMax <= 0 {
		cfg.ListMax = DefaultListMax
	}
	tools := cfg.Tools
	if tools == nil {
		tools = DefaultTools
	}
	t := &Toolset{cfg: cfg, root: root, enabled: map[string]bool{}, byName: map[string]string{}, seen: map[string]bool{}}
	for _, n := range tools {
		if !known(n) {
			return nil, fmt.Errorf("files: unknown tool %q (have %s)", n, strings.Join(allTools, ", "))
		}
		if cfg.ReadOnly && !readOnlyTools[n] {
			continue
		}
		name := cfg.NamePrefix + n
		if !toolNameRE.MatchString(name) {
			return nil, fmt.Errorf("files: tool name %q must match %s — providers refuse anything else", name, toolNameRE)
		}
		t.enabled[n] = true
		t.byName[name] = n
	}
	t.ignore = newMatcher()
	t.ignore.add("", cfg.ExtraIgnore)
	if !cfg.NoRipgrep {
		t.rg = findRipgrep()
	}
	return t, nil
}

func known(n string) bool {
	for _, a := range allTools {
		if a == n {
			return true
		}
	}
	return false
}

// Root is the absolute root, symlinks resolved.
func (t *Toolset) Root() string { return t.root }

// Name is the model-facing name of a bare tool name ("read" → "file_read").
func (t *Toolset) Name(bare string) string { return t.cfg.NamePrefix + bare }

// Names are the model-facing names of the enabled tools, in a fixed order.
func (t *Toolset) Names() []string {
	var out []string
	for _, n := range allTools {
		if t.enabled[n] {
			out = append(out, t.Name(n))
		}
	}
	return out
}

// Handles reports whether name is one of this toolset's enabled tools.
func (t *Toolset) Handles(name string) bool { _, ok := t.byName[name]; return ok }

// Bare maps a model-facing name back to its bare tool name, "" if not ours.
func (t *Toolset) Bare(name string) string { return t.byName[name] }

// Defs are the tool definitions for the enabled tools.
func (t *Toolset) Defs() []llm.ToolDef {
	var out []llm.ToolDef
	for _, n := range allTools {
		if t.enabled[n] {
			out = append(out, t.def(n))
		}
	}
	return out
}

// Dispatcher is agent.ToolDispatcher's shape, restated so this package needs
// only llm. A value of either type converts to the other.
type Dispatcher = func(ctx context.Context, tc llm.ToolCall) (string, error)

// Wrap routes this toolset's calls and passes every other call to inner.
func (t *Toolset) Wrap(inner Dispatcher) Dispatcher {
	return func(ctx context.Context, tc llm.ToolCall) (string, error) {
		if !t.Handles(tc.Function.Name) {
			return inner(ctx, tc)
		}
		return t.Dispatch(ctx, tc.Function.Name, tc.Function.Arguments), nil
	}
}

// Dispatch runs one call by its model-facing name with JSON arguments. The
// result is always for the model: a refusal starts with "ERROR: ".
func (t *Toolset) Dispatch(ctx context.Context, name, argsJSON string) string {
	bare, ok := t.byName[name]
	if !ok {
		return fmt.Sprintf("ERROR: unknown tool %q", name)
	}
	var a args
	if strings.TrimSpace(argsJSON) != "" {
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return fmt.Sprintf("ERROR: %s: arguments are not a JSON object: %v", name, err)
		}
	}
	out, err := t.run(ctx, bare, a)
	if err != nil {
		return "ERROR: " + err.Error()
	}
	return out
}

// Run is Dispatch with decoded arguments, returning a model-facing refusal as
// the error. bare is the tool's bare name.
func (t *Toolset) Run(ctx context.Context, bare string, a map[string]any) (string, error) {
	if !t.enabled[bare] {
		return "", fmt.Errorf("tool %q is not enabled", bare)
	}
	b, err := json.Marshal(a)
	if err != nil {
		return "", err
	}
	var parsed args
	if err := json.Unmarshal(b, &parsed); err != nil {
		return "", err
	}
	return t.run(ctx, bare, parsed)
}

func (t *Toolset) run(ctx context.Context, bare string, a args) (string, error) {
	switch bare {
	case Read:
		return t.read(ctx, a)
	case Edit:
		return t.edit(a)
	case Write:
		return t.write(a)
	case Grep:
		return t.grep(ctx, a)
	case List:
		return t.list(ctx, a)
	case Delete:
		return t.delete(a)
	case Move:
		return t.move(a)
	}
	return "", fmt.Errorf("tool %q is not enabled", bare)
}

// args is every tool's arguments in one struct; each tool reads its own.
type args struct {
	Path       string     `json:"path"`
	Offset     flexInt    `json:"offset"`
	Limit      flexInt    `json:"limit"`
	Old        *string    `json:"old"`
	New        *string    `json:"new"`
	Edits      []editPair `json:"edits"`
	Content    *string    `json:"content"`
	Overwrite  flexBool   `json:"overwrite"`
	Pattern    string     `json:"pattern"`
	Glob       string     `json:"glob"`
	Files      flexBool   `json:"files"`
	Context    flexInt    `json:"context"`
	Max        flexInt    `json:"max"`
	IgnoreCase flexBool   `json:"ignore_case"`
	Sizes      flexBool   `json:"sizes"`
	To         string     `json:"to"`
}

type editPair struct {
	Old string `json:"old"`
	New string `json:"new"`
}

// flexInt accepts 200, "200" and 200.0: a model writing limit:"200" means the
// same as limit:200, and refusing one teaches nothing (aw4's intArg).
type flexInt int

func (f *flexInt) UnmarshalJSON(b []byte) error {
	s := strings.Trim(strings.TrimSpace(string(b)), `"`)
	if s == "" || s == "null" {
		*f = 0
		return nil
	}
	var v float64
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return fmt.Errorf("%s is not a number", b)
	}
	*f = flexInt(int(v))
	return nil
}

// flexBool accepts true and "true".
type flexBool bool

func (f *flexBool) UnmarshalJSON(b []byte) error {
	s := strings.ToLower(strings.Trim(strings.TrimSpace(string(b)), `"`))
	*f = flexBool(s == "true" || s == "1" || s == "yes")
	return nil
}

func (t *Toolset) markSeen(real string) {
	t.mu.Lock()
	t.seen[real] = true
	t.mu.Unlock()
}

func (t *Toolset) wasSeen(real string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.seen[real]
}

func (t *Toolset) afterWrite(paths ...string) {
	if t.cfg.AfterWrite == nil {
		return
	}
	sort.Strings(paths)
	t.cfg.AfterWrite(paths)
}
