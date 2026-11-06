package files

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRelKeepsDotDotPrefixedNameInsideRoot(t *testing.T) {
	ts := mount(t, Config{Root: tree(t, map[string]string{"..cache/a": "x"})})
	if got := ts.rel(filepath.Join(ts.root, "..cache", "a")); got != "..cache/a" {
		t.Errorf("rel = %q, want ..cache/a", got)
	}
}

func TestWalkGitTreatsSubdirectoryAsLiteralPath(t *testing.T) {
	root := tree(t, map[string]string{"a*/f.txt": "one", "ab/g.txt": "two"})
	gitInit(t, root)
	ts := mount(t, Config{Root: root})
	var got []string
	if err := ts.walk(context.Background(), "a*", func(rel string) bool { got = append(got, rel); return true }); err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, ",") != "a*/f.txt" {
		t.Errorf("walk(a*) = %v, want only a*/f.txt", got)
	}
}

func TestGrepCancelledContextReturnsErrorNotNoMatches(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "needle\n"})
	for _, noRG := range []bool{false, true} {
		ts := mount(t, Config{Root: root, NoRipgrep: noRG})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		out, err := ts.grep(ctx, args{Pattern: "needle"})
		if err == nil {
			t.Errorf("noRG=%v: cancelled grep returned %q and no error", noRG, out)
		}
	}
}

func TestSafeCutKeepsInvalidBytesBeforeTheEnd(t *testing.T) {
	if got := safeCut("a\xffb"); got != "a\xffb" {
		t.Errorf("safeCut cut mid-string: %q", got)
	}
	if got := safeCut("a" + "é"[:1]); got != "a" {
		t.Errorf("partial trailing rune kept: %q", got)
	}
	if got := safeCut("aé"); got != "aé" {
		t.Errorf("whole rune cut: %q", got)
	}
	big := "\xff" + strings.Repeat("x", 1<<20)
	if got := safeCut(big); len(got) != len(big) || !utf8.ValidString(got[1:]) {
		t.Errorf("long string with early invalid byte cut to %d", len(got))
	}
}

func TestReadDescriptionMatchesLargeFileHead(t *testing.T) {
	ts := mount(t, Config{Root: t.TempDir()})
	for _, d := range ts.Defs() {
		if d.Function.Name == ts.Name(Read) {
			if !strings.Contains(d.Function.Description, sizeOf(largeHead)) {
				t.Errorf("read description does not state the head size of a large file:\n%s", d.Function.Description)
			}
		}
	}
}

func TestEditReportsWhitespaceOnlyChange(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "one\ntwo\n", "b.txt": "one\r\ntwo\r\n"})
	ts := mount(t, Config{Root: root})
	call(t, ts, Read, map[string]any{"path": "a.txt"})
	out := mustOK(t, call(t, ts, Edit, map[string]any{"path": "a.txt", "old": "two\n", "new": "two"}))
	if strings.Contains(out, "+0 -0") && !strings.Contains(out, "newline") {
		t.Errorf("trailing-newline edit reported as +0 -0:\n%s", out)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "a.txt")); string(b) != "one\ntwo" {
		t.Fatalf("file = %q", b)
	}
	if got := diffSummary("a\nb\n", "a\nb"); strings.HasPrefix(got, "+0 -0") && !strings.Contains(got, "newline") {
		t.Errorf("diffSummary = %q", got)
	}
}

func TestEditReportsLineEndingOnlyChange(t *testing.T) {
	root := tree(t, map[string]string{"m.txt": "one\r\ntwo\nthree\n"})
	ts := mount(t, Config{Root: root})
	call(t, ts, Read, map[string]any{"path": "m.txt"})
	out := mustOK(t, call(t, ts, Edit, map[string]any{"path": "m.txt", "old": "one\r\n", "new": "one\n"}))
	if !strings.Contains(out, "line endings") {
		t.Errorf("line-ending edit not explained:\n%s", out)
	}
}
