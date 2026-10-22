package files

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestList(t *testing.T) {
	for _, git := range []bool{false, true} {
		t.Run(fmt.Sprintf("git=%v", git), func(t *testing.T) {
			root := tree(t, map[string]string{
				".gitignore":            "*.log\nbuild/\n",
				"README.md":             "r\n",
				"main.go":               "package main\n",
				"debug.log":             "x\n",
				"build/out.bin":         "x",
				"internal/a/a.go":       "package a\n",
				"internal/a/a_test.go":  "package a\n",
				"internal/b/b.go":       "package b\n",
				"internal/b/.gitignore": "gen.go\n",
				"internal/b/gen.go":     "package b\n",
				"docs/x.md":             "x\n",
			})
			if git {
				gitInit(t, root)
			}
			must(t, os.Symlink("internal", filepath.Join(root, "src")))
			ts := mount(t, Config{Root: root})

			out := mustOK(t, call(t, ts, List, map[string]any{}))
			contains(t, out, "./: 3 dirs, 3 files", "ignored not shown", "docs/\ninternal/\nsrc/ @\n", "README.md", "main.go", ".gitignore")
			lacks(t, out, "debug.log", "build/")

			out = mustOK(t, call(t, ts, List, map[string]any{"path": "internal/b", "sizes": true}))
			contains(t, out, "internal/b/: 0 dirs, 2 files", "b.go  10 bytes")
			lacks(t, out, "gen.go")

			out = mustOK(t, call(t, ts, List, map[string]any{"glob": "**/*.go"}))
			contains(t, out, "4 files match **/*.go", "main.go", "internal/a/a.go", "internal/a/a_test.go", "internal/b/b.go")
			lacks(t, out, "gen.go", "src/")

			out = mustOK(t, call(t, ts, List, map[string]any{"path": "internal", "glob": "*_test.go"}))
			contains(t, out, "1 file matches *_test.go under internal\ninternal/a/a_test.go")

			out = mustOK(t, call(t, ts, List, map[string]any{"glob": "**/*.go", "max": 2}))
			contains(t, out, "4 files match **/*.go — showing the first 2")
			if strings.Count(out, "\n") != 2 {
				t.Fatalf("cap not applied:\n%s", out)
			}
			out = mustOK(t, call(t, ts, List, map[string]any{"max": 2}))
			contains(t, out, "… 4 more — narrow with a glob")

			contains(t, mustOK(t, call(t, ts, List, map[string]any{"glob": "*.rs"})), "no files match *.rs")
			contains(t, mustOK(t, call(t, ts, List, map[string]any{"path": "main.go"})), "main.go is a file", "file_read")
			mustErr(t, call(t, ts, List, map[string]any{"path": "nope"}), "does not exist")
		})
	}
}

func TestMatcher(t *testing.T) {
	m := newMatcher()
	m.add("", []string{
		"# comment", "", "*.log", "!keep.log", "/root-only.txt", "build/", "docs/**/*.tmp",
		"**/cache", "a/**/z", `\!bang`, "trailing   ", "foo/bar/",
	})
	m.add("sub", []string{"*.gen", "/anch"})
	cases := []struct {
		path string
		dir  bool
		want bool
	}{
		{"x.log", false, true},
		{"deep/x.log", false, true},
		{"keep.log", false, false},
		{"deep/keep.log", false, false},
		{"root-only.txt", false, true},
		{"sub/root-only.txt", false, false},
		{"build", true, true},
		{"build", false, false}, // dir-only rule, a file named build
		{"x/build", true, true},
		{"build/out.o", false, true}, // under an ignored dir
		{"docs/a/b/c.tmp", false, true},
		{"docs/c.tmp", false, true},
		{"other/c.tmp", false, false},
		{"cache", true, true},
		{"x/y/cache", true, true},
		{"a/z", false, true},
		{"a/b/c/z", false, true},
		{"!bang", false, true},
		{"trailing", false, true},
		{"foo/bar", true, true},
		{"x/foo/bar", true, false}, // has a slash: anchored
		{"sub/x.gen", false, true},
		{"sub/deep/x.gen", false, true},
		{"x.gen", false, false}, // rule is scoped to sub/
		{"sub/anch", false, true},
		{"sub/deep/anch", false, false},
		{"main.go", false, false},
	}
	for _, c := range cases {
		if got := m.ignored(c.path, c.dir); got != c.want {
			t.Errorf("ignored(%q, dir=%v) = %v, want %v", c.path, c.dir, got, c.want)
		}
	}
}

func TestGlob(t *testing.T) {
	cases := []struct {
		glob, path string
		want       bool
	}{
		{"*.go", "a.go", true},
		{"*.go", "x/y/a.go", true},
		{"*.go", "a.gox", false},
		{"**/*.go", "a.go", true},
		{"**/*.go", "x/a.go", true},
		{"internal/*.go", "internal/a.go", true},
		{"internal/*.go", "internal/x/a.go", false},
		{"internal/**", "internal/x/a.go", true},
		{"a?.txt", "ab.txt", true},
		{"[ab].txt", "b.txt", true},
		{"[!ab].txt", "b.txt", false},
	}
	for _, c := range cases {
		g, err := newGlob(c.glob)
		if err != nil {
			t.Fatal(err)
		}
		if got := g.match(c.path); got != c.want {
			t.Errorf("glob %q on %q = %v, want %v", c.glob, c.path, got, c.want)
		}
	}
}
