package files

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// at pages a file with no usable lines by bytes: each page names the next,
// and walking them returns the file exactly.
func TestReadAtPagesBytes(t *testing.T) {
	blob := strings.Repeat("abcdefghij", 20000) // 200,000 bytes, one line
	root := tree(t, map[string]string{"min.json": blob})
	ts := mount(t, Config{Root: root})
	next := regexp.MustCompile(`next page at=(\d+)`)
	var got strings.Builder
	at, pages := 0, 0
	for {
		out := mustOK(t, call(t, ts, Read, map[string]any{"path": "min.json", "at": at}))
		head, body, _ := strings.Cut(out, "\n")
		if !strings.HasPrefix(head, fmt.Sprintf("min.json: bytes %d-", at)) {
			t.Fatalf("page header: %q", head)
		}
		got.WriteString(body)
		pages++
		m := next.FindStringSubmatch(head)
		if m == nil {
			break
		}
		at, _ = strconv.Atoi(m[1])
	}
	if got.String() != blob || pages != 4 {
		t.Fatalf("pages=%d, reassembled %d bytes of %d", pages, got.Len(), len(blob))
	}
	mustErr(t, call(t, ts, Read, map[string]any{"path": "min.json", "at": 200000}), "past the end")
	mustErr(t, call(t, ts, Read, map[string]any{"path": "min.json", "at": 0, "offset": 1}), "one or the other")
	// Negative counts from the end, as offset does.
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "min.json", "at": -10})), "bytes 199990-200000 of 200000")
}

// A page never splits a rune: a start inside one moves forward, an end
// inside one moves back.
func TestReadAtRuneSafe(t *testing.T) {
	root := tree(t, map[string]string{"u.txt": "aé" + strings.Repeat("ü", 10)})
	ts := mount(t, Config{Root: root})
	out := mustOK(t, call(t, ts, Read, map[string]any{"path": "u.txt", "at": 2})) // inside é
	contains(t, out, "bytes 3-")
	if _, body, _ := strings.Cut(out, "\n"); !strings.HasPrefix(body, "ü") {
		t.Errorf("did not start on a rune: %q", body)
	}
}

// A line longer than the read cap is cut, and the cut names the at that
// continues it exactly.
func TestReadCutLineNamesAt(t *testing.T) {
	long := strings.Repeat("0123456789", 10000) // 100,000 bytes: past the 64KB ranged cap
	root := tree(t, map[string]string{"f.txt": "first\n" + long + "\nlast\n"})
	ts := mount(t, Config{Root: root})
	out := mustOK(t, call(t, ts, Read, map[string]any{"path": "f.txt", "offset": 2}))
	m := regexp.MustCompile(`Read on with at=(\d+)\]`).FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no at in the cut: %s", out)
	}
	at, _ := strconv.Atoi(m[1])
	shown := strings.SplitN(strings.SplitN(out, "2\t", 2)[1], "…[", 2)[0]
	if at != len("first\n")+len(shown) {
		t.Fatalf("at=%d, want %d", at, len("first\n")+len(shown))
	}
	page := mustOK(t, call(t, ts, Read, map[string]any{"path": "f.txt", "at": at}))
	if _, body, _ := strings.Cut(page, "\n"); !strings.HasPrefix(long[len(shown):], body[:50]) {
		t.Errorf("at did not continue the line: %q", body[:50])
	}
}

// A match deep in a long line comes back as a window around it, and its at
// is the file offset of the window: reading there shows the match. Both
// engines, in a file where the long line is not the first.
func TestGrepLongLineWindowAt(t *testing.T) {
	long := strings.Repeat("x", 50000) + "NEEDLE" + strings.Repeat("y", 50000)
	content := "short line\n" + long + "\n"
	for _, noRG := range []bool{true, false} {
		t.Run(fmt.Sprintf("noRipgrep=%v", noRG), func(t *testing.T) {
			if !noRG && findRipgrep() == "" {
				t.Skip("rg not on PATH")
			}
			root := tree(t, map[string]string{"bundle.js": content})
			ts := mount(t, Config{Root: root, NoRipgrep: noRG})
			out := mustOK(t, call(t, ts, Grep, map[string]any{"pattern": "NEEDLE"}))
			contains(t, out, "bundle.js:2:…", "NEEDLE", "match at byte 50000 of it")
			m := regexp.MustCompile(`read on with at=(\d+)\]`).FindStringSubmatch(out)
			if m == nil {
				t.Fatalf("no at: %s", out)
			}
			at, _ := strconv.Atoi(m[1])
			if want := len("short line\n") + 50000 - grepLineMax/4; at != want {
				t.Errorf("at=%d, want %d", at, want)
			}
			contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "bundle.js", "at": at})), "NEEDLE")
			for _, l := range strings.Split(out, "\n") {
				if len(l) > grepLineMax+40 {
					t.Fatalf("line not clipped: %d bytes", len(l))
				}
			}
		})
	}
}
