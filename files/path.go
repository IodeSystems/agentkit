package files

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
)

var syscallENOTDIR = syscall.ENOTDIR

// realRoot is root made absolute with its own symlinks resolved, so a root
// under a symlinked /tmp compares equal to the paths EvalSymlinks returns.
func realRoot(root string) (string, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(real)
	if err != nil {
		return "", err
	}
	if !st.IsDir() {
		return "", fmt.Errorf("%s is not a directory", root)
	}
	return real, nil
}

// Under resolves rel inside root and refuses anything that escapes it — by
// `..`, by an absolute path elsewhere, or by a symlink whose target is outside.
// The returned path is the REAL one (symlinks resolved), which is what a write
// must target: renaming a temp file over a symlink would replace the link.
//
// Symlinks inside root that point inside root are followed. Ones that point
// out are refused rather than followed (aw4 checked the path lexically only,
// which a link defeats): an agent's file tool is the one place nobody looks
// when a write lands outside the tree, and a dependency's source has its own
// way in through Config.Resolve.
//
// An absolute rel that already lies inside root is accepted: the model is
// often told the root's absolute path, and refusing the path it was given
// teaches nothing.
func Under(root, rel string) (string, error) {
	realRootPath, err := realRoot(root)
	if err != nil {
		return "", err
	}
	return under(realRootPath, root, rel)
}

// under is Under with root already real. lexRoot is the root as configured,
// for absolute paths the model built from the unresolved form.
func under(root, lexRoot, rel string) (string, error) {
	rel = strings.TrimSpace(rel)
	var full string
	if filepath.IsAbs(rel) {
		full = filepath.Clean(rel)
		if !within(root, full) {
			if la, err := filepath.Abs(lexRoot); err == nil && within(la, full) {
				r, _ := filepath.Rel(la, full)
				full = filepath.Join(root, r)
			} else {
				return "", fmt.Errorf("%s: %w — it is absolute and not inside the root; paths are relative to the root", rel, ErrEscape)
			}
		}
	} else {
		full = filepath.Clean(filepath.Join(root, rel))
		if !within(root, full) {
			return "", fmt.Errorf("%s: %w — it climbs out with `..`; paths are relative to the root", rel, ErrEscape)
		}
	}
	real, err := evalExisting(full)
	if err != nil {
		return "", err
	}
	if !within(root, real) {
		return "", fmt.Errorf("%s: %w — it is a symlink (or under one) whose target is outside the root, and such links are not followed", rel, ErrEscape)
	}
	return real, nil
}

// within compares against root + separator, not root alone: /w/app-evil has
// /w/app as a string prefix and is a different directory.
func within(root, p string) bool {
	return p == root || strings.HasPrefix(p, root+string(filepath.Separator))
}

// evalExisting resolves symlinks in the longest existing prefix of p and
// appends the rest, so a path that does not exist yet (a write) is checked
// through whatever links its parents pass through.
func evalExisting(p string) (string, error) {
	tail := ""
	cur := p
	for {
		real, err := filepath.EvalSymlinks(cur)
		if err == nil {
			if tail == "" {
				return real, nil
			}
			return filepath.Join(real, tail), nil
		}
		if !errors.Is(err, fs.ErrNotExist) && !errors.Is(err, syscallENOTDIR) {
			return "", err
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return p, nil
		}
		tail = filepath.Join(filepath.Base(cur), tail)
		cur = parent
	}
}

// resolve is the toolset's path resolution: the Resolve hook if set, else
// Under. It returns the real absolute path and the root-relative display form.
func (t *Toolset) resolve(rel string) (abs, display string, err error) {
	if t.cfg.Resolve != nil {
		abs, err = t.cfg.Resolve(rel)
		if err != nil {
			return "", rel, t.refuse(rel, err)
		}
		return abs, rel, nil
	}
	abs, err = under(t.root, t.cfg.Root, rel)
	if err != nil {
		return "", rel, t.refuse(rel, err)
	}
	return abs, t.rel(abs), nil
}

// resolveLink is resolve for an operation on the path ITSELF (delete, move):
// its directory is resolved, its last element is not, so deleting a symlink
// removes the link and not what it points at.
func (t *Toolset) resolveLink(rel string) (abs, display string, err error) {
	if t.cfg.Resolve != nil {
		return t.resolve(rel)
	}
	clean := filepath.Clean(filepath.FromSlash(strings.TrimSpace(rel)))
	base := filepath.Base(clean)
	if base == "." || base == ".." || base == string(filepath.Separator) {
		return "", rel, t.refuse(rel, fmt.Errorf("%s: names a directory, not a file", rel))
	}
	dir, _, err := t.resolve(filepath.Dir(clean))
	if err != nil {
		return "", rel, err
	}
	abs = filepath.Join(dir, base)
	if abs == t.root {
		return "", rel, fmt.Errorf("%s is the root", rel)
	}
	return abs, t.rel(abs), nil
}

// rel is abs relative to the root, slash-separated, "." for the root.
func (t *Toolset) rel(abs string) string {
	r, err := filepath.Rel(t.root, abs)
	if err != nil || strings.HasPrefix(r, "..") {
		return abs
	}
	return filepath.ToSlash(r)
}

// refuse passes a path refusal through the Refusal hook.
func (t *Toolset) refuse(path string, reason error) error {
	if t.cfg.Refusal == nil {
		return reason
	}
	return refusal{text: t.cfg.Refusal(path, reason), cause: reason}
}

type refusal struct {
	text  string
	cause error
}

func (r refusal) Error() string { return r.text }
func (r refusal) Unwrap() error { return r.cause }

// writable refuses the one place inside the root a tool must not write.
//
// .git is the repository, not the work: .git/hooks/pre-commit runs on the next
// commit, .git/config reaches remotes and core.hooksPath. Reading it stays
// allowed — that is sometimes the honest answer to "what branch am I on". (aw4,
// 2026-09-07: an edit into .git failed only by accident, because in a worktree
// .git is a file; in a plain clone the write would have landed.)
func (t *Toolset) writable(abs, display string) error {
	for _, seg := range strings.Split(filepath.ToSlash(t.rel(abs)), "/") {
		if seg == ".git" {
			return fmt.Errorf("%s is inside .git, which is the repository rather than the work: a hook "+
				"written there runs on the next commit. Edit the source; use git for the repository", display)
		}
	}
	return nil
}

// generatedHeader matches the conventional generated-file marker ("Code
// generated by X. DO NOT EDIT." in Go; most generators copy the wording) in a
// file's first lines.
var generatedHeader = regexp.MustCompile(`(?m)^\W{0,4}\s*(?:Code )?[Gg]enerated\b.*\bDO NOT EDIT\b`)

// refuseGenerated refuses to change a generated file: the next generator run
// overwrites the edit and a generated-file check fails on it (aw4: a
// hand-edited api_gen.ts cost a 16-20 minute run).
func (t *Toolset) refuseGenerated(head []byte, display string) error {
	if t.cfg.AllowGenerated {
		return nil
	}
	if len(head) > 1024 {
		head = head[:1024]
	}
	m := generatedHeader.Find(head)
	if m == nil {
		return nil
	}
	return fmt.Errorf("%s is generated (its header says %q), so an edit is overwritten by the next "+
		"generator run. Change the generator's input and regenerate", display, strings.TrimSpace(string(m)))
}

// ── missing paths ───────────────────────────────────────────────────

// maxNear bounds a refusal's listing: enough to place yourself in a
// directory, short enough to read.
const maxNear = 20

// notFound is the refusal for a path that does not exist. It names what IS
// there, because "no such file" makes the next guess as blind as the last.
// (aw4, 2026-09-04: one session read one nonexistent path 39 times; the file
// existed one directory over.)
func (t *Toolset) notFound(abs, display string) error {
	msg := display + ": " + ErrNotFound.Error() + "."
	if same := t.sameName(filepath.Base(abs), 5); len(same) > 0 {
		msg += " Files with that name: " + strings.Join(same, ", ") + "."
	}
	dir := filepath.Dir(abs)
	for !isDir(dir) && within(t.root, filepath.Dir(dir)) && dir != t.root {
		dir = filepath.Dir(dir)
	}
	if isDir(dir) {
		msg += near(t.rel(dir), dir, filepath.Base(abs))
	}
	return t.refuse(display, notFoundErr{msg})
}

type notFoundErr struct{ msg string }

func (e notFoundErr) Error() string { return e.msg }
func (e notFoundErr) Unwrap() error { return ErrNotFound }

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// sameName finds up to n files named base (case-insensitively) anywhere under
// the root, ignore rules applied, giving up after a bounded walk.
func (t *Toolset) sameName(base string, n int) []string {
	var out []string
	visited := 0
	_ = t.walk(context.Background(), "", func(rel string) bool {
		visited++
		if strings.EqualFold(pathBase(rel), base) {
			out = append(out, rel)
		}
		return len(out) < n && visited < 50000
	})
	return out
}

func pathBase(rel string) string {
	if i := strings.LastIndexByte(rel, '/'); i >= 0 {
		return rel[i+1:]
	}
	return rel
}

// near lists a directory, NEAREST names first rather than alphabetical.
//
// The cap hides the tail, so the order decides what is seen. Alphabetical put
// 000001…000010 first when 000012_ was wanted (aw4, 2026-09-05: 30 failed
// reads in one directory). Names that share the longest prefix with the wanted
// one come first; numbered names are ordered by numeric distance, because
// zero-padding defeats a prefix comparison (000029 shares "00002" with both
// 000020 and 000028).
func near(relDir, dir, want string) string {
	ents, err := os.ReadDir(dir)
	if err != nil || len(ents) == 0 {
		return ""
	}
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() {
			n += "/"
		}
		names = append(names, n)
	}
	wn, wok := leadingNum(want)
	sort.Slice(names, func(i, j int) bool {
		if wok {
			ni, iok := leadingNum(names[i])
			nj, jok := leadingNum(names[j])
			if iok && jok && ni != nj {
				return absInt(ni-wn) < absInt(nj-wn)
			}
			if iok != jok {
				return iok
			}
		}
		if a, b := sharedPrefix(names[i], want), sharedPrefix(names[j], want); a != b {
			return a > b
		}
		return names[i] < names[j]
	})
	more := ""
	if len(names) > maxNear {
		more = fmt.Sprintf(" (+%d more)", len(names)-maxNear)
		names = names[:maxNear]
	}
	return fmt.Sprintf(" %s/ holds: %s%s", strings.TrimSuffix(relDir, "/"), strings.Join(names, ", "), more)
}

func leadingNum(s string) (int, bool) {
	n := 0
	for n < len(s) && s[n] >= '0' && s[n] <= '9' {
		n++
	}
	if n == 0 {
		return 0, false
	}
	v, err := strconv.Atoi(s[:n])
	return v, err == nil
}

func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func sharedPrefix(a, b string) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}
