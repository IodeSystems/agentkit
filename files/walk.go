package files

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// skipDirs are skipped by the walk used when the root is not a git work tree,
// where there is no better signal of what is source. Inside a work tree git's
// own ignore rules decide, so a committed vendor/ is searched like anything
// else.
var skipDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true, "node_modules": true, "vendor": true,
	"dist": true, "build": true, "target": true, ".venv": true, "venv": true,
	"__pycache__": true, ".next": true, ".cache": true, ".gradle": true, ".tox": true,
}

// gitWorkTree reports whether root is inside a git work tree with git on PATH.
func gitWorkTree(ctx context.Context, root string) bool {
	if _, err := exec.LookPath("git"); err != nil {
		return false
	}
	out, err := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "--is-inside-work-tree").Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// walk calls fn with each file under sub (root-relative slash path, "" for the
// root) that the ignore rules keep, until fn returns false. Directories are not
// reported. Symlinks are reported but never descended.
//
// Inside a git work tree the list is `git ls-files --cached --others
// --exclude-standard` — exact .gitignore semantics (nested files, negation,
// .git/info/exclude, the global excludes file) at no cost of our own, and
// streamed, so a large tree is never held in memory. Elsewhere it is a walk
// that honours .gitignore files through matcher and skips skipDirs.
// ExtraIgnore applies to both.
func (t *Toolset) walk(ctx context.Context, sub string, fn func(rel string) bool) error {
	if gitWorkTree(ctx, t.root) {
		return t.walkGit(ctx, sub, fn)
	}
	return t.walkFS(sub, fn)
}

func (t *Toolset) walkGit(ctx context.Context, sub string, fn func(rel string) bool) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	argv := []string{"-C", t.root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"}
	if sub != "" && sub != "." {
		argv = append(argv, "--", sub)
	}
	cmd := exec.CommandContext(ctx, "git", argv...)
	out, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	r := bufio.NewReaderSize(out, 64*1024)
	prev := ""
	stopped := false
	for {
		rel, err := r.ReadString(0)
		if err != nil {
			break
		}
		rel = strings.TrimSuffix(rel, "\x00")
		// An unmerged file is listed once per stage.
		if rel == prev {
			continue
		}
		prev = rel
		if t.ignore.ignored(rel, false) {
			continue
		}
		// --cached lists files deleted from the work tree and submodules
		// (as one path); neither is a file to read.
		st, err := os.Lstat(filepath.Join(t.root, filepath.FromSlash(rel)))
		if err != nil || st.IsDir() {
			continue
		}
		if !fn(rel) {
			stopped = true
			break
		}
	}
	if stopped {
		cancel()
		_, _ = io.Copy(io.Discard, out)
		_ = cmd.Wait()
		return nil
	}
	return cmd.Wait()
}

func (t *Toolset) walkFS(sub string, fn func(rel string) bool) error {
	m := newMatcher()
	// .gitignore files between the root and sub apply to sub too.
	start := t.root
	m.addFile("", filepath.Join(t.root, ".gitignore"))
	if sub != "" && sub != "." {
		parts := strings.Split(sub, "/")
		for i := 1; i < len(parts); i++ {
			base := strings.Join(parts[:i], "/")
			m.addFile(base, filepath.Join(t.root, filepath.FromSlash(base), ".gitignore"))
		}
		start = filepath.Join(t.root, filepath.FromSlash(sub))
	}
	errStop := errors.New("stop")
	err := filepath.WalkDir(start, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil //nolint:nilerr // an unreadable entry is skipped, not fatal
		}
		rel := t.rel(p)
		if d.IsDir() {
			if p == start {
				if p != t.root {
					m.addFile(rel, filepath.Join(p, ".gitignore"))
				}
				return nil
			}
			if skipDirs[d.Name()] || m.ignored(rel, true) || t.ignore.ignored(rel, true) {
				return filepath.SkipDir
			}
			m.addFile(rel, filepath.Join(p, ".gitignore"))
			return nil
		}
		if d.Name() == ".git" { // a worktree's .git is a file
			return nil
		}
		if m.ignored(rel, false) || t.ignore.ignored(rel, false) {
			return nil
		}
		if !fn(rel) {
			return errStop
		}
		return nil
	})
	if errors.Is(err, errStop) {
		return nil
	}
	return err
}

// childIgnored filters one directory's children (root-relative slash paths)
// down to the ones the ignore rules hide.
func (t *Toolset) childIgnored(ctx context.Context, relDir string, names []string, dirs []bool) []bool {
	out := make([]bool, len(names))
	full := func(i int) string {
		if relDir == "" || relDir == "." {
			return names[i]
		}
		return relDir + "/" + names[i]
	}
	if gitWorkTree(ctx, t.root) {
		var in bytes.Buffer
		idx := map[string]int{}
		for i := range names {
			if names[i] == ".git" {
				out[i] = true
				continue
			}
			p := full(i)
			idx[p] = i
			in.WriteString(p)
			in.WriteByte(0)
		}
		cmd := exec.CommandContext(ctx, "git", "-C", t.root, "check-ignore", "-z", "--stdin")
		cmd.Stdin = &in
		b, _ := cmd.Output() // exit 1 = nothing ignored
		for _, p := range strings.Split(string(b), "\x00") {
			if i, ok := idx[p]; ok {
				out[i] = true
			}
		}
	} else {
		m := newMatcher()
		m.addFile("", filepath.Join(t.root, ".gitignore"))
		if relDir != "" && relDir != "." {
			parts := strings.Split(relDir, "/")
			for i := 1; i <= len(parts); i++ {
				base := strings.Join(parts[:i], "/")
				m.addFile(base, filepath.Join(t.root, filepath.FromSlash(base), ".gitignore"))
			}
		}
		for i := range names {
			out[i] = (dirs[i] && skipDirs[names[i]]) || names[i] == ".git" || m.ignored(full(i), dirs[i])
		}
	}
	for i := range names {
		if !out[i] && t.ignore.ignored(full(i), dirs[i]) {
			out[i] = true
		}
	}
	return out
}
