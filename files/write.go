package files

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// write creates or replaces a whole file.
//
// Replacing a file this session has not read is refused unless overwrite is
// set: a whole-file write composed without reading the file is how a model
// erases what it never saw. The tracking is this Toolset's (one per session),
// in memory, and covers read, edit and write.
func (t *Toolset) write(a args) (string, error) {
	if strings.TrimSpace(a.Path) == "" {
		return "", errors.New("write needs a path")
	}
	if a.Content == nil {
		return "", errors.New("write needs content (the whole file)")
	}
	abs, display, err := t.resolve(a.Path)
	if err != nil {
		return "", err
	}
	if err := t.writable(abs, display); err != nil {
		return "", err
	}
	mode := fs.FileMode(0o644)
	prevLines := -1
	st, err := os.Stat(abs)
	switch {
	case err == nil && st.IsDir():
		return "", fmt.Errorf("%s is a directory", display)
	case err == nil:
		if !bool(a.Overwrite) && !t.cfg.OverwriteUnread && !t.wasSeen(abs) {
			return "", fmt.Errorf("%s exists (%s) and has not been read in this session, so writing it would replace "+
				"text you have not seen. Read it first, use %s for a change, or pass overwrite:true to replace it anyway",
				display, sizeOf(st.Size()), t.Name(Edit))
		}
		old, err := os.ReadFile(abs)
		if err != nil {
			return "", err
		}
		if err := t.refuseGenerated(old, display); err != nil {
			return "", err
		}
		mode = st.Mode()
		prevLines = len(splitLines(string(old)))
	case errors.Is(err, fs.ErrNotExist):
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			return "", err
		}
	default:
		return "", err
	}
	if err := writeAtomic(abs, []byte(*a.Content), mode); err != nil {
		return "", err
	}
	t.markSeen(abs)
	t.afterWrite(t.rel(abs))
	n := len(splitLines(*a.Content))
	if prevLines < 0 {
		return fmt.Sprintf("wrote %s (%s, new)", display, plural(n, "line")), nil
	}
	return fmt.Sprintf("wrote %s (%s, overwritten; was %s)", display, plural(n, "line"), plural(prevLines, "line")), nil
}

// delete removes one file. A directory is refused rather than walked: a
// recursive delete is a different blast radius from "remove this file".
func (t *Toolset) delete(a args) (string, error) {
	if strings.TrimSpace(a.Path) == "" {
		return "", errors.New("delete needs a path")
	}
	abs, display, err := t.resolveLink(a.Path)
	if err != nil {
		return "", err
	}
	if err := t.writable(abs, display); err != nil {
		return "", err
	}
	st, err := os.Lstat(abs)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "", t.notFound(abs, display)
	case err != nil:
		return "", err
	case st.IsDir():
		return "", fmt.Errorf("%s is a directory; delete removes one file. Remove its files one at a time", display)
	}
	if err := os.Remove(abs); err != nil {
		return "", err
	}
	t.afterWrite(t.rel(abs))
	return fmt.Sprintf("deleted %s (%s)", display, sizeOf(st.Size())), nil
}

// move renames a file or directory and refuses to replace anything.
func (t *Toolset) move(a args) (string, error) {
	if strings.TrimSpace(a.Path) == "" || strings.TrimSpace(a.To) == "" {
		return "", errors.New("move needs path and to")
	}
	from, fromD, err := t.resolveLink(a.Path)
	if err != nil {
		return "", err
	}
	to, toD, err := t.resolveLink(a.To)
	if err != nil {
		return "", err
	}
	for _, p := range [][2]string{{from, fromD}, {to, toD}} {
		if err := t.writable(p[0], p[1]); err != nil {
			return "", err
		}
	}
	if _, err := os.Lstat(from); errors.Is(err, fs.ErrNotExist) {
		return "", t.notFound(from, fromD)
	} else if err != nil {
		return "", err
	}
	if _, err := os.Lstat(to); err == nil {
		return "", fmt.Errorf("%s already exists; move does not replace files. Delete it first or pick another name", toD)
	}
	if within(from, to) {
		return "", fmt.Errorf("cannot move %s into itself", fromD)
	}
	if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(from, to); err != nil {
		return "", err
	}
	if t.wasSeen(from) {
		t.markSeen(to)
	}
	t.afterWrite(t.rel(from), t.rel(to))
	return fmt.Sprintf("moved %s → %s", fromD, toD), nil
}

func plural(n int, w string) string {
	if n == 1 {
		return fmt.Sprintf("1 %s", w)
	}
	return fmt.Sprintf("%d %ss", n, w)
}
