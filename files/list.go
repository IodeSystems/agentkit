package files

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// list shows one directory level, or with glob every matching file beneath
// path. Both honour the ignore rules, both stop at ListMax.
func (t *Toolset) list(ctx context.Context, a args) (string, error) {
	abs, display, err := t.resolve(a.Path)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "", t.notFound(abs, display)
	case err != nil:
		return "", err
	case !st.IsDir():
		return fmt.Sprintf("%s is a file (%s); %s reads it", display, sizeOf(st.Size()), t.Name(Read)), nil
	}
	limit := t.cfg.ListMax
	if a.Max > 0 {
		limit = min(int(a.Max), 5000)
	}
	rel := t.rel(abs)
	if rel == "." {
		rel = ""
	}
	if strings.TrimSpace(a.Glob) != "" {
		return t.listGlob(ctx, rel, display, a, limit)
	}
	return t.listDir(ctx, abs, rel, display, bool(a.Sizes), limit)
}

func (t *Toolset) listDir(ctx context.Context, abs, rel, display string, sizes bool, limit int) (string, error) {
	ents, err := os.ReadDir(abs)
	if err != nil {
		return "", err
	}
	names := make([]string, len(ents))
	dirs := make([]bool, len(ents))
	for i, e := range ents {
		names[i] = e.Name()
		dirs[i] = e.IsDir()
	}
	hidden := t.childIgnored(ctx, rel, names, dirs)
	type row struct {
		name string
		dir  bool
		link bool
		size int64
	}
	var rows []row
	nDirs, nFiles, nHidden := 0, 0, 0
	for i, e := range ents {
		if hidden[i] {
			nHidden++
			continue
		}
		r := row{name: e.Name(), dir: e.IsDir(), link: e.Type()&fs.ModeSymlink != 0}
		if r.link {
			if st, err := os.Stat(filepath.Join(abs, e.Name())); err == nil && st.IsDir() {
				r.dir = true
			}
		}
		if sizes && !r.dir {
			if info, err := e.Info(); err == nil {
				r.size = info.Size()
			}
		}
		if r.dir {
			nDirs++
		} else {
			nFiles++
		}
		rows = append(rows, r)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].dir != rows[j].dir {
			return rows[i].dir
		}
		return rows[i].name < rows[j].name
	})
	var b strings.Builder
	head := display
	if head == "." || head == "" {
		head = "."
	}
	fmt.Fprintf(&b, "%s/: %s, %s", strings.TrimSuffix(head, "/"), plural(nDirs, "dir"), plural(nFiles, "file"))
	if nHidden > 0 {
		fmt.Fprintf(&b, " (%d ignored not shown)", nHidden)
	}
	for i, r := range rows {
		if i >= limit {
			fmt.Fprintf(&b, "\n… %d more — narrow with a glob", len(rows)-limit)
			break
		}
		name := r.name
		if r.dir {
			name += "/"
		}
		if r.link {
			name += " @"
		}
		if sizes && !r.dir {
			name += "  " + sizeOf(r.size)
		}
		b.WriteString("\n" + name)
	}
	return b.String(), nil
}

func (t *Toolset) listGlob(ctx context.Context, rel, display string, a args, limit int) (string, error) {
	g, err := newGlob(a.Glob)
	if err != nil {
		return "", err
	}
	var rows []string
	n := 0
	capped := false
	err = t.walk(ctx, rel, func(p string) bool {
		if !g.match(p) {
			return true
		}
		n++
		if len(rows) < limit {
			row := p
			if a.Sizes {
				if st, err := os.Lstat(filepath.Join(t.root, filepath.FromSlash(p))); err == nil {
					row += "  " + sizeOf(st.Size())
				}
			}
			rows = append(rows, row)
		}
		if n >= limit*grepCountFactor {
			capped = true
			return false
		}
		return true
	})
	if err != nil {
		return "", err
	}
	scope := ""
	if rel != "" {
		scope = " under " + display
	}
	plus := ""
	if capped {
		plus = "+"
	}
	var b strings.Builder
	if n == 0 {
		return fmt.Sprintf("no files match %s%s", a.Glob, scope), nil
	}
	fmt.Fprintf(&b, "%s %s%s", count(n, plus, "file matches", "files match"), a.Glob, scope)
	if len(rows) < n {
		fmt.Fprintf(&b, " — showing the first %d; narrow with path or glob", len(rows))
	}
	for _, r := range rows {
		b.WriteString("\n" + r)
	}
	return b.String(), nil
}
