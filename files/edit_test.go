package files

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const goSrc = `package main

import "fmt"

func main() {
	fmt.Println("hello")
	fmt.Println("world")
}

func helper() int {
	return 1
}
`

func TestEdit(t *testing.T) {
	cases := []struct {
		name  string
		args  map[string]any
		want  string   // the file after; "" = unchanged
		out   []string // in the result
		err   []string // a refusal containing these
		files map[string]string
	}{
		{
			name: "single pair",
			args: map[string]any{"old": `fmt.Println("hello")`, "new": `fmt.Println("hi")`},
			want: strings.Replace(goSrc, `"hello"`, `"hi"`, 1),
			out:  []string{"edited main.go: +1 -1", "@@ -4,5 +4,5 @@", `-	fmt.Println("hello")`, `+	fmt.Println("hi")`},
		},
		{
			name: "multi in order",
			args: map[string]any{"edits": []map[string]any{
				{"old": `"hello"`, "new": `"one"`},
				{"old": `"one")`, "new": `"two")`}, // sees the first edit's result
				{"old": "return 1", "new": "return 2\n\t// done"},
			}},
			want: strings.Replace(strings.Replace(goSrc, `"hello"`, `"two"`, 1), "return 1", "return 2\n\t// done", 1),
			out:  []string{"edited main.go: +3 -2\n@@ -4,9 +4,10 @@", `+	fmt.Println("two")`, "+	// done"},
		},
		{
			name: "multi is atomic",
			args: map[string]any{"edits": []map[string]any{
				{"old": `"hello"`, "new": `"one"`},
				{"old": "not in the file", "new": "x"},
			}},
			err: []string{"edit 2 of 2", "does not appear", "No edits were applied"},
		},
		{
			name: "multiple matches",
			args: map[string]any{"old": "fmt.Println", "new": "log.Println"},
			err:  []string{"appears 2 times", "lines 6, 7", "more of the surrounding lines"},
		},
		{
			name: "whitespace miss shows the region",
			args: map[string]any{"old": "func helper() int {\n    return 1\n}", "new": "x"},
			err:  []string{"different whitespace", "at line 10", "10\tfunc helper() int {", "11\t\treturn 1", `anchor: \s\s\s\sreturn 1`, `file:   \treturn 1`},
		},
		{
			name: "near miss shows the closest window",
			args: map[string]any{"old": "func main() {\n\tfmt.Println(\"hello\")\n\tfmt.Println(\"there\")\n}", "new": "x"},
			err:  []string{"Closest: lines 5-8, 3 of 4 lines equal", "7\t\tfmt.Println(\"world\")", "line 7"},
		},
		{
			name: "fuzzy best line",
			args: map[string]any{"old": "func helpr() int {", "new": "x"},
			err:  []string{"closest single line", "line 10", "10\tfunc helper() int {"},
		},
		{
			name: "absent entirely",
			args: map[string]any{"old": "completely different\ntext here", "new": "x"},
			err:  []string{"does not appear in main.go at all", "Only 0 of its 2 lines"},
		},
		{
			name: "line-number prefixes stripped",
			args: map[string]any{"old": "10\tfunc helper() int {\n11\t\treturn 1", "new": "10\tfunc helper() int {\n11\t\treturn 3"},
			want: strings.Replace(goSrc, "return 1", "return 3", 1),
			out:  []string{"line-number prefixes copied from read were removed"},
		},
		{name: "empty old", args: map[string]any{"old": "", "new": "x"}, err: []string{"old is empty"}},
		{name: "same text", args: map[string]any{"old": "return 1", "new": "return 1"}, err: []string{"the same text"}},
		{name: "old without new", args: map[string]any{"old": "return 1"}, err: []string{"old needs a new"}},
		{name: "both shapes", args: map[string]any{"old": "a", "new": "b", "edits": []map[string]any{{"old": "c", "new": "d"}}}, err: []string{"not both"}},
		{name: "neither", args: map[string]any{}, err: []string{"edit needs edits", "file_write"}},
		{name: "path only", args: map[string]any{"path": "a.txt"}, err: []string{"sent only path", "nothing was changed"}},
		{
			name: "cancelling edits write nothing",
			args: map[string]any{"edits": []map[string]any{{"old": "return 1", "new": "return 9"}, {"old": "return 9", "new": "return 1"}}},
			out:  []string{"no change"},
		},
		{
			name: "delete text",
			args: map[string]any{"old": "\tfmt.Println(\"world\")\n", "new": ""},
			want: strings.Replace(goSrc, "\tfmt.Println(\"world\")\n", "", 1),
			out:  []string{"+0 -1"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := tree(t, map[string]string{"main.go": goSrc})
			ts := mount(t, Config{Root: root})
			a := map[string]any{"path": "main.go"}
			for k, v := range c.args {
				a[k] = v
			}
			out := call(t, ts, Edit, a)
			got := readFile(t, filepath.Join(root, "main.go"))
			if c.err != nil {
				mustErr(t, out, c.err...)
				if got != goSrc {
					t.Fatalf("a refused edit changed the file:\n%s", got)
				}
				return
			}
			contains(t, mustOK(t, out), c.out...)
			want := c.want
			if want == "" {
				want = goSrc
			}
			if got != want {
				t.Fatalf("file after:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

func TestEditPreservesCRLFAndMode(t *testing.T) {
	root := tree(t, map[string]string{"win.txt": "alpha\r\nbeta\r\ngamma\r\n", "run.sh": "#!/bin/sh\necho hi\n"})
	must(t, os.Chmod(filepath.Join(root, "run.sh"), 0o755))
	ts := mount(t, Config{Root: root})

	// The anchor comes from read, which shows LF lines.
	mustOK(t, call(t, ts, Edit, map[string]any{"path": "win.txt", "old": "beta\ngamma", "new": "BETA\nnew\ngamma"}))
	if got := readFile(t, filepath.Join(root, "win.txt")); got != "alpha\r\nBETA\r\nnew\r\ngamma\r\n" {
		t.Fatalf("CRLF not preserved: %q", got)
	}
	mustOK(t, call(t, ts, Edit, map[string]any{"path": "run.sh", "old": "echo hi", "new": "echo bye"}))
	st, err := os.Stat(filepath.Join(root, "run.sh"))
	must(t, err)
	if st.Mode().Perm() != 0o755 {
		t.Fatalf("mode = %v, want 0755", st.Mode().Perm())
	}
	// No temp files left behind.
	ents, _ := os.ReadDir(root)
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temp file left: %s", e.Name())
		}
	}
}

func TestEditRefuses(t *testing.T) {
	root := tree(t, map[string]string{
		"gen.go":    "// Code generated by stringer. DO NOT EDIT.\n\npackage x\n",
		"bin.dat":   "a\x00b",
		".git/HEAD": "ref: refs/heads/main\n",
		"dir/x.txt": "x\n",
		"ok.txt":    "fine\n",
	})
	ts := mount(t, Config{Root: root})
	cases := []struct {
		name string
		args map[string]any
		err  []string
	}{
		{"generated", map[string]any{"path": "gen.go", "old": "package x", "new": "package y"}, []string{"is generated", "DO NOT EDIT", "regenerate"}},
		{"binary", map[string]any{"path": "bin.dat", "old": "a", "new": "b"}, []string{"binary"}},
		{"inside .git", map[string]any{"path": ".git/HEAD", "old": "main", "new": "evil"}, []string{"inside .git"}},
		{"directory", map[string]any{"path": "dir", "old": "a", "new": "b"}, []string{"is a directory"}},
		{"missing", map[string]any{"path": "nope.txt", "old": "a", "new": "b"}, []string{"does not exist", "to create it, use file_write"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { mustErr(t, call(t, ts, Edit, c.args), c.err...) })
	}
	allowed := mount(t, Config{Root: root, AllowGenerated: true})
	mustOK(t, call(t, allowed, Edit, map[string]any{"path": "gen.go", "old": "package x", "new": "package y"}))
}

func TestEditThroughInternalSymlinkKeepsLink(t *testing.T) {
	root := tree(t, map[string]string{"real.txt": "one\n"})
	must(t, os.Symlink("real.txt", filepath.Join(root, "alias.txt")))
	ts := mount(t, Config{Root: root})
	mustOK(t, call(t, ts, Edit, map[string]any{"path": "alias.txt", "old": "one", "new": "two"}))
	if got := readFile(t, filepath.Join(root, "real.txt")); got != "two\n" {
		t.Fatalf("target = %q", got)
	}
	if st, err := os.Lstat(filepath.Join(root, "alias.txt")); err != nil || st.Mode()&os.ModeSymlink == 0 {
		t.Fatal("the symlink was replaced by a file")
	}
}

func TestWrite(t *testing.T) {
	root := tree(t, map[string]string{"exists.txt": "old\ncontent\n"})
	ts := mount(t, Config{Root: root})

	out := mustOK(t, call(t, ts, Write, map[string]any{"path": "a/b/new.txt", "content": "one\ntwo\n"}))
	contains(t, out, "wrote a/b/new.txt (2 lines, new)")
	if readFile(t, filepath.Join(root, "a/b/new.txt")) != "one\ntwo\n" {
		t.Fatal("content wrong")
	}

	mustErr(t, call(t, ts, Write, map[string]any{"path": "exists.txt", "content": "x"}),
		"has not been read in this session", "overwrite:true", "file_edit")
	if readFile(t, filepath.Join(root, "exists.txt")) != "old\ncontent\n" {
		t.Fatal("refused write changed the file")
	}

	mustOK(t, call(t, ts, Read, map[string]any{"path": "exists.txt", "offset": 1, "limit": 1}))
	contains(t, mustOK(t, call(t, ts, Write, map[string]any{"path": "exists.txt", "content": "new\n"})),
		"wrote exists.txt (1 line, overwritten; was 2 lines)")

	// A file this toolset wrote counts as seen.
	mustOK(t, call(t, ts, Write, map[string]any{"path": "a/b/new.txt", "content": "again\n"}))

	// overwrite:true and the config switch both bypass the guard.
	must(t, os.WriteFile(filepath.Join(root, "other.txt"), []byte("o\n"), 0o600))
	mustOK(t, call(t, ts, Write, map[string]any{"path": "other.txt", "content": "p\n", "overwrite": true}))
	if st, _ := os.Stat(filepath.Join(root, "other.txt")); st.Mode().Perm() != 0o600 {
		t.Fatalf("overwrite changed mode to %v", st.Mode().Perm())
	}
	must(t, os.WriteFile(filepath.Join(root, "third.txt"), []byte("t\n"), 0o644))
	loose := mount(t, Config{Root: root, OverwriteUnread: true})
	mustOK(t, call(t, loose, Write, map[string]any{"path": "third.txt", "content": "u\n"}))

	// Read-tracking is per toolset: a second session has read nothing.
	fresh := mount(t, Config{Root: root})
	mustErr(t, call(t, fresh, Write, map[string]any{"path": "exists.txt", "content": "x"}), "has not been read")

	mustErr(t, call(t, ts, Write, map[string]any{"path": "a", "content": "x"}), "is a directory")
	mustErr(t, call(t, ts, Write, map[string]any{"path": "x.txt"}), "needs content")
	mustErr(t, call(t, ts, Write, map[string]any{"path": ".git/hooks/pre-commit", "content": "x"}), "inside .git")
	contains(t, mustOK(t, call(t, ts, Write, map[string]any{"path": "empty.txt", "content": ""})), "(0 lines, new)")
}

func TestDeleteAndMove(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "a\n", "b.txt": "b\n", "d/x.txt": "x\n", "target.txt": "t\n"})
	must(t, os.Symlink("target.txt", filepath.Join(root, "link.txt")))

	off := mount(t, Config{Root: root})
	mustErr(t, call(t, off, Delete, map[string]any{"path": "a.txt"}), "unknown tool")

	ts := mount(t, Config{Root: root, Tools: allTools})
	contains(t, mustOK(t, call(t, ts, Delete, map[string]any{"path": "a.txt"})), "deleted a.txt")
	if _, err := os.Stat(filepath.Join(root, "a.txt")); err == nil {
		t.Fatal("not deleted")
	}
	mustErr(t, call(t, ts, Delete, map[string]any{"path": "d"}), "is a directory")
	mustErr(t, call(t, ts, Delete, map[string]any{"path": "a.txt"}), "does not exist")
	mustErr(t, call(t, ts, Delete, map[string]any{"path": "."}), "names a directory")

	// Deleting a link removes the link, not its target.
	mustOK(t, call(t, ts, Delete, map[string]any{"path": "link.txt"}))
	if readFile(t, filepath.Join(root, "target.txt")) != "t\n" {
		t.Fatal("deleting a link removed its target")
	}

	mustErr(t, call(t, ts, Move, map[string]any{"path": "b.txt", "to": "d/x.txt"}), "already exists")
	contains(t, mustOK(t, call(t, ts, Move, map[string]any{"path": "b.txt", "to": "e/f/b.txt"})), "moved b.txt → e/f/b.txt")
	if readFile(t, filepath.Join(root, "e/f/b.txt")) != "b\n" {
		t.Fatal("move lost content")
	}
	mustOK(t, call(t, ts, Move, map[string]any{"path": "d", "to": "d2"}))
	mustErr(t, call(t, ts, Move, map[string]any{"path": "d2", "to": "d2/inner"}), "into itself")
	mustErr(t, call(t, ts, Move, map[string]any{"path": "nope", "to": "x"}), "does not exist")
}

func TestDiffSummary(t *testing.T) {
	a := lines(30, func(i int) string { return "l" + itoa(i) })
	b := strings.Replace(strings.Replace(a, "l5\n", "five\n", 1), "l25\n", "l25\nextra\n", 1)
	got := diffSummary(a, b)
	contains(t, got, "+2 -1 in 2 hunks", "@@ -3,5 +3,5 @@\n l3\n l4\n-l5\n+five\n l6\n l7", "@@ -24,4 +24,5 @@\n l24\n l25\n+extra\n l26\n l27")

	// Close changes merge into one hunk.
	c := strings.Replace(strings.Replace(a, "l5\n", "X\n", 1), "l8\n", "Y\n", 1)
	contains(t, diffSummary(a, c), "+2 -2\n@@ -3,8 +3,8 @@")

	// A rewrite larger than the search bound is still a correct diff.
	big := lines(3000, func(i int) string { return "a" + itoa(i) })
	other := lines(3000, func(i int) string { return "b" + itoa(i) })
	contains(t, diffSummary(big, other), "+3000 -3000", "more line(s)")
}

func itoa(i int) string { return strconv.Itoa(i) }
