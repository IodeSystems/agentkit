package files

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// grepLineMax clips each output line: one minified line must not fill the
// result on its own.
const grepLineMax = 300

// grepCountFactor bounds counting past the shown matches: the total is
// exact up to GrepMax×this, then reported as "N+". Counting is what makes
// "how many places do this" answerable without reading them all, but an
// unbounded count over a huge tree is an unbounded call.
const grepCountFactor = 20

func findRipgrep() string {
	p, err := exec.LookPath("rg")
	if err != nil {
		return ""
	}
	return p
}

type grepQuery struct {
	re         *regexp.Regexp
	pattern    string // as given
	literal    bool   // the pattern did not compile and is searched as text
	reErr      error
	ignoreCase bool
	target     string // root-relative slash path, "" = the root
	targetFile bool
	glob       *globMatcher
	globRaw    string
	ctx        int
	filesOnly  bool
}

func (t *Toolset) grep(ctx context.Context, a args) (string, error) {
	if a.Pattern == "" {
		return "", errors.New("grep needs a pattern")
	}
	q := grepQuery{pattern: a.Pattern, ignoreCase: bool(a.IgnoreCase), ctx: min(max(int(a.Context), 0), 10), filesOnly: bool(a.Files), globRaw: a.Glob}
	// A pattern that is not a valid regexp is searched LITERALLY, and the
	// result says so. `Cents(` is someone looking for a call, not a mistake
	// about regular expressions (aw4, 2026-09-04: 26 of 28 grep refusals were
	// that one pattern). Told rather than silent: a wrong search with no
	// warning is worse than a refusal.
	expr := a.Pattern
	if _, err := regexp.Compile(expr); err != nil {
		q.literal, q.reErr = true, err
		expr = regexp.QuoteMeta(a.Pattern)
	}
	if q.ignoreCase {
		expr = "(?i)" + expr
	}
	q.re = regexp.MustCompile(expr)
	g, err := newGlob(a.Glob)
	if err != nil {
		return "", err
	}
	q.glob = g
	if strings.TrimSpace(a.Path) != "" {
		abs, display, err := t.resolve(a.Path)
		if err != nil {
			return "", err
		}
		st, err := os.Stat(abs)
		if err != nil {
			return "", t.notFound(abs, display)
		}
		q.target, q.targetFile = t.rel(abs), !st.IsDir()
		if q.target == "." {
			q.target = ""
		}
	}
	limit := t.cfg.GrepMax
	if a.Max > 0 {
		limit = min(int(a.Max), 1000)
	}
	out := newGrepOut(limit, q.ctx, t.cfg.ReadMax, q.filesOnly)
	if t.rg != "" {
		if err := t.grepRG(ctx, q, out); err == nil {
			return out.render(q, t), nil
		}
		t.rgFailed.Add(1)
		out = newGrepOut(limit, q.ctx, t.cfg.ReadMax, q.filesOnly) // rg failed: start over in Go
	}
	if err := t.grepGo(ctx, q, out); err != nil {
		return "", err
	}
	return out.render(q, t), nil
}

// ── output, shared by both engines so their results are identical ──

type grepOut struct {
	max, ctx, byteCap int
	filesOnly         bool

	b                     strings.Builder
	matches, files, shown int
	capped                bool // counting stopped: totals are lower bounds
	curFile               string
	lastPath              string
	lastLine              int // last line emitted in lastPath
	lastShownLine         int // last MATCH line shown in lastPath
	showing               bool
}

func newGrepOut(max, ctx, byteCap int, filesOnly bool) *grepOut {
	return &grepOut{max: max, ctx: ctx, byteCap: byteCap, filesOnly: filesOnly, showing: true}
}

// done reports whether the engine can stop: everything shown, and counting
// has reached its bound.
func (o *grepOut) done() bool { return o.capped }

func (o *grepOut) file(path string) {
	if path != o.curFile {
		o.curFile = path
		o.files++
		if o.filesOnly {
			o.matches++
			if o.showing && o.shown < o.max {
				o.b.WriteString(path + "\n")
				o.shown++
			}
			if o.files >= o.max*grepCountFactor {
				o.capped = true
			}
		}
	}
}

func (o *grepOut) match(path string, line int, text string) {
	o.file(path)
	o.matches++
	if o.showing && (o.shown >= o.max || o.b.Len() >= o.byteCap) {
		o.showing = false
	}
	if o.showing {
		o.emit(path, line, ':', text)
		o.shown++
		o.lastShownLine = line
	}
	if o.matches >= o.max*grepCountFactor {
		o.capped = true
	}
}

func (o *grepOut) context(path string, line int, text string) {
	if !o.showing {
		// After the cap, only the trailing context of the last shown match.
		if path != o.lastPath || line > o.lastShownLine+o.ctx {
			return
		}
	}
	if path == o.lastPath && line <= o.lastLine {
		return
	}
	o.emit(path, line, '-', text)
}

func (o *grepOut) emit(path string, line int, sep byte, text string) {
	if o.ctx > 0 && o.lastPath != "" && (path != o.lastPath || line > o.lastLine+1) {
		o.b.WriteString("--\n")
	}
	fmt.Fprintf(&o.b, "%s%c%d%c%s\n", path, sep, line, sep, clipLine(strings.TrimRight(text, "\r\n"), grepLineMax))
	o.lastPath, o.lastLine = path, line
}

func (o *grepOut) render(q grepQuery, t *Toolset) string {
	var h strings.Builder
	if q.literal {
		fmt.Fprintf(&h, "searched for %q literally: it is not a valid regular expression (%v)\n", q.pattern, q.reErr)
	}
	plus := ""
	if o.capped {
		plus = "+"
	}
	scope := ""
	if q.target != "" {
		scope += " under " + q.target
	}
	if q.globRaw != "" {
		scope += " matching " + q.globRaw
	}
	if o.matches == 0 {
		fmt.Fprintf(&h, "no matches for /%s/%s", q.pattern, scope)
		return h.String()
	}
	if o.filesOnly {
		fmt.Fprintf(&h, "%s /%s/%s", count(o.files, plus, "file matches", "files match"), q.pattern, scope)
	} else {
		fmt.Fprintf(&h, "%s in %s%s", count(o.matches, plus, "match", "matches"), count(o.files, plus, "file", "files"), scope)
	}
	if o.shown < o.matches {
		fmt.Fprintf(&h, " — showing the first %d; narrow with path or glob", o.shown)
		if !o.filesOnly {
			fmt.Fprintf(&h, ", or files:true for names only")
		}
	}
	return h.String() + "\n" + strings.TrimRight(o.b.String(), "\n")
}

// ── ripgrep ─────────────────────────────────────────────────────────

func (t *Toolset) grepRG(ctx context.Context, q grepQuery, o *grepOut) error {
	argv := []string{"--no-config", "--hidden", "--sort", "path", "--glob", "!.git", "--no-messages"}
	if gitWorkTree(ctx, t.root) {
		// git decides, as in walkGit; rg's .ignore files are not git's.
		argv = append(argv, "--no-ignore-dot")
	} else {
		argv = append(argv, "--no-require-git", "--no-ignore-parent", "--no-ignore-dot", "--no-ignore-global")
		for d := range skipDirs {
			argv = append(argv, "--glob", "!"+d+"/")
		}
	}
	if len(t.cfg.ExtraIgnore) > 0 {
		f, err := os.CreateTemp("", "agentkit-files-ignore-*")
		if err != nil {
			return err
		}
		defer func() { _ = os.Remove(f.Name()) }()
		_, _ = f.WriteString(strings.Join(t.cfg.ExtraIgnore, "\n") + "\n")
		_ = f.Close()
		argv = append(argv, "--ignore-file", f.Name())
	}
	if q.ignoreCase {
		argv = append(argv, "--ignore-case")
	}
	if q.literal {
		argv = append(argv, "--fixed-strings")
	}
	if q.globRaw != "" {
		argv = append(argv, "--glob", q.globRaw)
	}
	if q.filesOnly {
		argv = append(argv, "--files-with-matches", "--null")
	} else {
		argv = append(argv, "--json")
		if q.ctx > 0 {
			argv = append(argv, "--context", fmt.Sprint(q.ctx))
		}
	}
	argv = append(argv, "--regexp", q.pattern, "--")
	if q.target != "" {
		argv = append(argv, q.target)
	} else {
		argv = append(argv, ".")
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(ctx, t.rg, argv...)
	cmd.Dir = t.root
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	r := bufio.NewReaderSize(stdout, 256*1024)
	for !o.done() {
		var line []byte
		var err error
		if q.filesOnly {
			line, err = r.ReadBytes(0)
			line = bytes.TrimSuffix(line, []byte{0})
		} else {
			line, err = r.ReadBytes('\n')
		}
		if len(line) > 0 {
			if q.filesOnly {
				o.file(cleanRGPath(string(line)))
			} else {
				rgEvent(line, o)
			}
		}
		if err != nil {
			break
		}
	}
	stopped := o.done()
	cancel()
	_, _ = io.Copy(io.Discard, stdout)
	werr := cmd.Wait()
	if stopped {
		return nil
	}
	var ee *exec.ExitError
	if errors.As(werr, &ee) && ee.ExitCode() == 1 {
		return nil // no matches
	}
	if werr != nil && o.matches == 0 {
		return fmt.Errorf("rg: %v: %s", werr, strings.TrimSpace(stderr.String()))
	}
	return nil
}

type rgText struct {
	Text  *string `json:"text"`
	Bytes *string `json:"bytes"`
}

func (r rgText) String() string {
	if r.Text != nil {
		return *r.Text
	}
	if r.Bytes != nil {
		b, _ := base64.StdEncoding.DecodeString(*r.Bytes)
		return strings.ToValidUTF8(string(b), "�")
	}
	return ""
}

func rgEvent(line []byte, o *grepOut) {
	var ev struct {
		Type string `json:"type"`
		Data struct {
			Path       rgText `json:"path"`
			Lines      rgText `json:"lines"`
			LineNumber int    `json:"line_number"`
		} `json:"data"`
	}
	if json.Unmarshal(line, &ev) != nil {
		return
	}
	p := cleanRGPath(ev.Data.Path.String())
	switch ev.Type {
	case "match":
		o.match(p, ev.Data.LineNumber, ev.Data.Lines.String())
	case "context":
		o.context(p, ev.Data.LineNumber, ev.Data.Lines.String())
	}
}

func cleanRGPath(p string) string {
	return filepath.ToSlash(strings.TrimPrefix(p, "./"))
}

// ── Go engine ───────────────────────────────────────────────────────

func (t *Toolset) grepGo(ctx context.Context, q grepQuery, o *grepOut) error {
	search := func(rel string) bool {
		if ctx.Err() != nil {
			return false
		}
		t.grepFile(q, rel, o)
		return !o.done()
	}
	if q.targetFile {
		search(q.target)
		return nil
	}
	return t.walk(ctx, q.target, func(rel string) bool {
		if !q.glob.match(rel) {
			return true
		}
		return search(rel)
	})
}

// grepFile searches one file, streaming, and reports matches and the same
// context lines rg would.
func (t *Toolset) grepFile(q grepQuery, rel string, o *grepOut) {
	abs := filepath.Join(t.root, filepath.FromSlash(rel))
	st, err := os.Lstat(abs)
	if err != nil || !st.Mode().IsRegular() {
		return // rg does not follow symlinks either
	}
	f, err := os.Open(abs)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()
	br := bufio.NewReaderSize(f, 64*1024)
	if head, _ := br.Peek(binaryScan); bytes.IndexByte(head, 0) >= 0 {
		return
	}
	type held struct {
		n    int
		text string
	}
	var before []held
	after := 0
	for n := 1; ; n++ {
		raw, _, err := readLine(br, true, 1<<20)
		if raw == nil && err != nil {
			return
		}
		text := string(raw)
		if q.re.MatchString(strings.TrimSuffix(text, "\r")) {
			if q.filesOnly {
				o.file(rel)
				return
			}
			for _, h := range before {
				o.context(rel, h.n, h.text)
			}
			before = before[:0]
			o.match(rel, n, text)
			if o.done() {
				return
			}
			after = q.ctx
		} else if after > 0 {
			o.context(rel, n, text)
			after--
		} else if q.ctx > 0 {
			before = append(before, held{n, text})
			if len(before) > q.ctx {
				before = before[1:]
			}
		}
		if err != nil {
			return
		}
	}
}

// count is "1 file" / "3 files" / "100+ files".
func count(n int, plus, one, many string) string {
	if n == 1 && plus == "" {
		return "1 " + one
	}
	return fmt.Sprintf("%d%s %s", n, plus, many)
}
