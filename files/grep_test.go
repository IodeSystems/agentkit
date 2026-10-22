package files

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// searchTree has nested .gitignore files with negation, a dir rule, a
// skip-list dir, a binary and a hidden file.
func searchTree(t *testing.T) string {
	return tree(t, map[string]string{
		".gitignore":          "*.log\n/out/\n!keep.log\n",
		"main.go":             "package main\n\nfunc main() {\n\tneedle()\n}\n",
		"util.go":             "package main\n\n// needle helper\nfunc needle() {}\n",
		"debug.log":           "needle in a log\n",
		"keep.log":            "needle kept\n",
		"out/gen.go":          "needle in out\n",
		"sub/out/real.go":     "needle in sub/out (not the root out/)\n",
		"sub/.gitignore":      "secret*\n!secret_ok.txt\n",
		"sub/secret.txt":      "needle secret\n",
		"sub/secret_ok.txt":   "needle allowed\n",
		"sub/deep/a_test.go":  "package deep\n// needle in a test\n",
		"sub/deep/notes.md":   "Needle uppercase\n",
		"node_modules/m/i.js": "needle in node_modules\n",
		".hidden/conf":        "needle hidden\n",
		"bin.dat":             "needle\x00binary",
		"ctx.txt":             "a\nb\nc\nneedle 1\nd\ne\nf\ng\nh\nneedle 2\ni\nneedle 3\nj\n",
	})
}

func grepCases() []struct {
	name string
	args map[string]any
	want []string
	not  []string
} {
	return []struct {
		name string
		args map[string]any
		want []string
		not  []string
	}{
		{"gitignore with nested negation",
			map[string]any{"pattern": "needle"},
			[]string{"main.go:4:\tneedle()", "util.go:3:// needle helper", "keep.log:1:needle kept", "sub/secret_ok.txt:1:", "sub/out/real.go:1:", ".hidden/conf:1:", "sub/deep/a_test.go:2:"},
			[]string{"debug.log", "out/gen.go:", "sub/secret.txt", "bin.dat", "node_modules"}},
		{"glob by name", map[string]any{"pattern": "needle", "glob": "*.go"},
			[]string{"main.go:4:", "sub/deep/a_test.go:2:"}, []string{"keep.log", "ctx.txt"}},
		{"glob by path", map[string]any{"pattern": "needle", "glob": "sub/**/*_test.go"},
			[]string{"1 match in 1 file", "sub/deep/a_test.go:2:"}, []string{"main.go"}},
		{"path subdir", map[string]any{"pattern": "needle", "path": "sub"},
			[]string{"under sub", "sub/secret_ok.txt:1:"}, []string{"main.go", "sub/secret.txt"}},
		{"path file", map[string]any{"pattern": "needle", "path": "util.go"},
			[]string{"2 matches in 1 file", "util.go:4:func needle() {}"}, []string{"main.go"}},
		{"files only", map[string]any{"pattern": "needle", "files": true, "glob": "*.go"},
			[]string{"files match /needle/", "main.go\n", "util.go"}, []string{":4:"}},
		{"ignore case", map[string]any{"pattern": "NEEDLE UPPER", "ignore_case": true},
			[]string{"sub/deep/notes.md:1:Needle uppercase"}, nil},
		{"case sensitive by default", map[string]any{"pattern": "Needle"},
			[]string{"sub/deep/notes.md:1:"}, []string{"main.go"}},
		{"context", map[string]any{"pattern": "needle", "path": "ctx.txt", "context": 2},
			[]string{"3 matches in 1 file", "ctx.txt-2-b\nctx.txt-3-c\nctx.txt:4:needle 1\nctx.txt-5-d\nctx.txt-6-e\n--\nctx.txt-8-g\nctx.txt-9-h\nctx.txt:10:needle 2\nctx.txt-11-i\nctx.txt:12:needle 3\nctx.txt-13-j"},
			[]string{"ctx.txt-7-"}},
		{"cap with tail", map[string]any{"pattern": "needle", "path": "ctx.txt", "max": 2},
			[]string{"3 matches in 1 file under ctx.txt — showing the first 2; narrow with path or glob", "ctx.txt:4:", "ctx.txt:10:"}, []string{"ctx.txt:12:"}},
		{"invalid regex searched literally", map[string]any{"pattern": "needle("},
			[]string{"searched for \"needle(\" literally", "main.go:4:\tneedle()"}, nil},
		{"no matches", map[string]any{"pattern": "zzzqqq", "glob": "*.go"},
			[]string{"no matches for /zzzqqq/ matching *.go"}, nil},
	}
}

func runGrepCases(t *testing.T, root string, cfg Config) map[string]string {
	cfg.Root = root
	ts := mount(t, cfg)
	outs := map[string]string{}
	for _, c := range grepCases() {
		t.Run(c.name, func(t *testing.T) {
			out := mustOK(t, call(t, ts, Grep, c.args))
			contains(t, out, c.want...)
			lacks(t, out, c.not...)
			outs[c.name] = out
		})
	}
	return outs
}

// The Go engine, walking a tree that is not a git work tree: our own
// .gitignore matcher and the skip list decide.
func TestGrepGoWalk(t *testing.T) {
	runGrepCases(t, searchTree(t), Config{NoRipgrep: true})
}

// The Go engine inside a git work tree: git ls-files decides.
func TestGrepGoGit(t *testing.T) {
	root := searchTree(t)
	gitInit(t, root)
	// node_modules is not special to git; ignore it the way a project would.
	must(t, os.WriteFile(filepath.Join(root, ".gitignore"), []byte("*.log\n/out/\n!keep.log\nnode_modules/\n"), 0o644))
	runGrepCases(t, root, Config{NoRipgrep: true})
}

// rg and the Go engine must say the same thing, byte for byte.
func TestGrepRipgrepParity(t *testing.T) {
	if findRipgrep() == "" {
		t.Skip("rg not on PATH")
	}
	for _, git := range []bool{false, true} {
		t.Run(fmt.Sprintf("git=%v", git), func(t *testing.T) {
			root := searchTree(t)
			if git {
				gitInit(t, root)
				must(t, os.WriteFile(filepath.Join(root, ".gitignore"), []byte("*.log\n/out/\n!keep.log\nnode_modules/\n"), 0o644))
			}
			rg := mount(t, Config{Root: root})
			if rg.rg == "" {
				t.Fatal("rg found but not used")
			}
			goEng := mount(t, Config{Root: root, NoRipgrep: true})
			for _, c := range grepCases() {
				t.Run(c.name, func(t *testing.T) {
					a, b := call(t, rg, Grep, c.args), call(t, goEng, Grep, c.args)
					if a != b {
						t.Fatalf("rg and Go differ\n--- rg:\n%s\n--- go:\n%s", a, b)
					}
					if n := rg.rgFailed.Load(); n != 0 {
						t.Fatalf("rg failed %d time(s) and the Go engine answered instead", n)
					}
				})
			}
		})
	}
}

func TestGrepCapsAndCounts(t *testing.T) {
	files := map[string]string{}
	for i := 0; i < 30; i++ {
		files[fmt.Sprintf("f%02d.txt", i)] = lines(10, func(j int) string { return "hit" })
	}
	root := tree(t, files)
	for _, noRG := range []bool{true, false} {
		t.Run(fmt.Sprintf("noRipgrep=%v", noRG), func(t *testing.T) {
			if !noRG && findRipgrep() == "" {
				t.Skip("rg not on PATH")
			}
			ts := mount(t, Config{Root: root, GrepMax: 5, NoRipgrep: noRG})
			// 300 matches; counting stops at 5×20 = 100.
			out := mustOK(t, call(t, ts, Grep, map[string]any{"pattern": "hit"}))
			contains(t, out, "100+ matches in 10+ files — showing the first 5")
			if n := strings.Count(out, ":hit"); n != 5 {
				t.Fatalf("%d matches shown, want 5", n)
			}
			// Exact below the counting bound.
			out = mustOK(t, call(t, ts, Grep, map[string]any{"pattern": "hit", "glob": "f0[0-3].txt"}))
			contains(t, out, "40 matches in 4 files matching f0[0-3].txt — showing the first 5")
			out = mustOK(t, call(t, ts, Grep, map[string]any{"pattern": "hit", "files": true, "max": 3}))
			contains(t, out, "30 files match /hit/ — showing the first 3")
		})
	}
}

func TestGrepExtraIgnoreAndLongLines(t *testing.T) {
	root := tree(t, map[string]string{
		"a.go":        "needle\n",
		"gen/b.go":    "needle\n",
		"gen/keep.go": "needle\n",
		"long.txt":    strings.Repeat("x", 1000) + "needle\n",
	})
	for _, noRG := range []bool{true, false} {
		t.Run(fmt.Sprintf("noRipgrep=%v", noRG), func(t *testing.T) {
			if !noRG && findRipgrep() == "" {
				t.Skip("rg not on PATH")
			}
			ts := mount(t, Config{Root: root, ExtraIgnore: []string{"gen/*", "!gen/keep.go"}, NoRipgrep: noRG})
			out := mustOK(t, call(t, ts, Grep, map[string]any{"pattern": "needle"}))
			contains(t, out, "a.go:1:", "gen/keep.go:1:", "long.txt:1:xxx")
			lacks(t, out, "gen/b.go")
			for _, l := range strings.Split(out, "\n") {
				if len(l) > grepLineMax+40 {
					t.Fatalf("line not clipped: %d bytes", len(l))
				}
			}
		})
	}
}

func TestGrepRefuses(t *testing.T) {
	root := tree(t, map[string]string{"a.go": "x\n"})
	ts := mount(t, Config{Root: root})
	mustErr(t, call(t, ts, Grep, map[string]any{}), "needs a pattern")
	mustErr(t, call(t, ts, Grep, map[string]any{"pattern": "x", "path": "nope"}), "does not exist")
	mustErr(t, call(t, ts, Grep, map[string]any{"pattern": "x", "glob": "[z-a]"}), "malformed")
}
