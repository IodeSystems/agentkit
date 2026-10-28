package files

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadRanges(t *testing.T) {
	ten := lines(10, func(i int) string { return fmt.Sprint("line ", i) })
	root := tree(t, map[string]string{
		"ten.txt":   ten,
		"nonl.txt":  "a\nb\nc",
		"crlf.txt":  "one\r\ntwo\r\n",
		"empty.txt": "",
		"blank.txt": "\n\n",
	})
	ts := mount(t, Config{Root: root})
	cases := []struct {
		name   string
		path   string
		offset any
		limit  any
		want   []string
		not    []string
		err    []string
	}{
		{name: "whole", path: "ten.txt", want: []string{"ten.txt: 10 lines\n1\tline 1\n", "10\tline 10"}},
		{name: "range", path: "ten.txt", offset: 3, limit: 2, want: []string{"ten.txt: lines 3-4 of 10\n3\tline 3\n4\tline 4"}, not: []string{"line 5", "line 2\n"}},
		{name: "offset only", path: "ten.txt", offset: 9, want: []string{"lines 9-10 of 10", "9\tline 9", "10\tline 10"}},
		{name: "limit only", path: "ten.txt", limit: 2, want: []string{"lines 1-2 of 10"}, not: []string{"line 3"}},
		{name: "negative", path: "ten.txt", offset: -3, want: []string{"lines 8-10 of 10", "8\tline 8", "10\tline 10"}, not: []string{"line 7"}},
		{name: "negative with limit", path: "ten.txt", offset: -3, limit: 1, want: []string{"lines 8-8 of 10"}, not: []string{"line 9"}},
		{name: "negative past start", path: "ten.txt", offset: -100, want: []string{"ten.txt: 10 lines", "1\tline 1"}},
		{name: "limit past end", path: "ten.txt", offset: 9, limit: 50, want: []string{"lines 9-10 of 10"}},
		{name: "offset past end", path: "ten.txt", offset: 11, err: []string{"past the end", "has 10 lines"}},
		{name: "negative limit", path: "ten.txt", limit: -5, err: []string{"cannot be negative", "offset=-50"}},
		{name: "no final newline", path: "nonl.txt", want: []string{"nonl.txt: 3 lines", "3\tc"}},
		{name: "crlf shown without CR", path: "crlf.txt", want: []string{"1\tone\n2\ttwo"}, not: []string{"\r"}},
		{name: "empty", path: "empty.txt", want: []string{"empty.txt: empty (0 bytes)"}},
		{name: "blank lines", path: "blank.txt", want: []string{"blank.txt: 2 lines\n1\t\n2\t"}},
		{name: "no path", path: "", err: []string{"needs a path"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := map[string]any{"path": c.path}
			if c.offset != nil {
				a["offset"] = c.offset
			}
			if c.limit != nil {
				a["limit"] = c.limit
			}
			out := call(t, ts, Read, a)
			if c.err != nil {
				mustErr(t, out, c.err...)
				return
			}
			contains(t, mustOK(t, out), c.want...)
			lacks(t, out, c.not...)
		})
	}
}

func TestReadCaps(t *testing.T) {
	big := lines(1000, func(i int) string { return fmt.Sprintf("row %04d %s", i, strings.Repeat("x", 40)) })
	root := tree(t, map[string]string{"big.txt": big, "small.txt": "tiny\n", "oneline.txt": strings.Repeat("y", 5000) + "\n"})
	ts := mount(t, Config{Root: root, UnrangedMax: 1000, ReadMax: 4000})

	t.Run("unranged gets a head and a paging hint", func(t *testing.T) {
		out := mustOK(t, call(t, ts, Read, map[string]any{"path": "big.txt"}))
		contains(t, out, "big.txt: lines 1-", "of 1000", "this read named no range", "1000 lines", "offset=-200", "up to 3.9KB")
		if n := strings.Count(out, "\trow "); n < 15 || n > 25 {
			t.Errorf("unranged returned %d rows for a 1000-byte cap", n)
		}
		// The hint's next offset is the line after the last one shown.
		last := strings.Count(out, "\t")
		contains(t, out, fmt.Sprintf("offset=%d limit=200", last+1))
	})
	t.Run("ranged gets the larger cap", func(t *testing.T) {
		out := mustOK(t, call(t, ts, Read, map[string]any{"path": "big.txt", "offset": 1, "limit": 1000}))
		contains(t, out, "cut at 3.9KB", "Continue with offset=")
		if n := strings.Count(out, "\trow "); n < 60 {
			t.Errorf("ranged returned only %d rows", n)
		}
		if len(out) > 4000+500 {
			t.Errorf("ranged result is %d bytes for a 4000-byte cap", len(out))
		}
	})
	t.Run("a small unranged read is whole", func(t *testing.T) {
		out := mustOK(t, call(t, ts, Read, map[string]any{"path": "small.txt"}))
		lacks(t, out, "cut at")
	})
	t.Run("one line longer than the cap", func(t *testing.T) {
		out := mustOK(t, call(t, ts, Read, map[string]any{"path": "oneline.txt"}))
		contains(t, out, "1\tyyyy", "line is 4.9KB; cut")
		if len(out) > 1200 {
			t.Errorf("one-line read is %d bytes", len(out))
		}
	})
	t.Run("defaults", func(t *testing.T) {
		d := mount(t, Config{Root: root})
		if d.cfg.ReadMax != 64*1024 || d.cfg.UnrangedMax != 8*1024 {
			t.Fatalf("default caps %d/%d", d.cfg.ReadMax, d.cfg.UnrangedMax)
		}
	})
}

func TestReadBinary(t *testing.T) {
	root := tree(t, map[string]string{
		"img.png": "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
		"blob":    "abc\x00def",
	})
	ts := mount(t, Config{Root: root})
	mustErr(t, call(t, ts, Read, map[string]any{"path": "img.png"}), "binary file", "image/png", "16 bytes")
	mustErr(t, call(t, ts, Read, map[string]any{"path": "blob"}), "binary file", "7 bytes", "offset 3")
}

func TestReadMissingSuggests(t *testing.T) {
	root := tree(t, map[string]string{
		"internal/store/user.go":          "package store\n",
		"internal/api/handler.go":         "package api\n",
		"migrations/000011_a.sql":         "",
		"migrations/000020_b.sql":         "",
		"migrations/000028_c.sql":         "",
		"node_modules/x/user.go":          "",
		"internal/store/user_test.go":     "",
		"internal/store/zzz_unrelated.go": "",
	})
	ts := mount(t, Config{Root: root})
	cases := []struct {
		name string
		path string
		want []string
		not  []string
	}{
		{"same name elsewhere", "pkg/user.go", []string{"pkg/user.go: does not exist", "internal/store/user.go"}, []string{"node_modules/x"}},
		{"nearest dir listed, nearest name first", "internal/store/usr.go", []string{"internal/store/ holds: user.go, user_test.go"}, nil},
		{"numbered names by distance", "migrations/000029_d.sql", []string{"migrations/ holds: 000028_c.sql, 000020_b.sql, 000011_a.sql"}, nil},
		{"missing dirs climb to an existing one", "internal/nope/deeper/x.go", []string{"internal/ holds: api/, store/"}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := call(t, ts, Read, map[string]any{"path": c.path})
			mustErr(t, out, c.want...)
			lacks(t, out, c.not...)
		})
	}
	// A directory is listed, not refused.
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "internal"})), "internal/: 2 dirs, 0 files", "api/\nstore/")
}

// read on a directory is list: the same output for the same arguments,
// with read alone mounted.
func TestReadDirectoryIsList(t *testing.T) {
	root := tree(t, map[string]string{
		".gitignore":           "*.log\n",
		"README.md":            "r\n",
		"debug.log":            "x\n",
		"internal/a/a.go":      "package a\n",
		"internal/a/a_test.go": "package a\n",
		"internal/b/b.go":      "package b\n",
	})
	both := mount(t, Config{Root: root})
	only := mount(t, Config{Root: root, Tools: []string{Read}})
	for _, a := range []map[string]any{
		{"path": "."},
		{"path": "internal/a", "sizes": true},
		{"path": ".", "glob": "**/*.go"},
		{"path": "internal", "glob": "*_test.go", "sizes": true},
		{"path": ".", "glob": "**/*.go", "max": 1},
		{"path": ".", "max": 1},
	} {
		want := mustOK(t, call(t, both, List, a))
		if got := mustOK(t, call(t, only, Read, a)); got != want {
			t.Errorf("read %v:\n%s\nlist:\n%s", a, got, want)
		}
	}
	lacks(t, mustOK(t, call(t, only, Read, map[string]any{"path": "."})), "debug.log")
}

// read with grep is grep: the same output for every grep case, on a file
// and on a tree, under the grep name and under grep's own "pattern".
func TestReadGrepIsGrep(t *testing.T) {
	root := searchTree(t)
	both := mount(t, Config{Root: root})
	only := mount(t, Config{Root: root, Tools: []string{Read}})
	for _, c := range grepCases() {
		t.Run(c.name, func(t *testing.T) {
			want := mustOK(t, call(t, both, Grep, c.args))
			a := map[string]any{}
			for k, v := range c.args {
				a[k] = v
			}
			a["grep"] = a["pattern"]
			delete(a, "pattern")
			if got := mustOK(t, call(t, only, Read, a)); got != want {
				t.Errorf("read %v:\n%s\ngrep:\n%s", a, got, want)
			}
			if got := mustOK(t, call(t, only, Read, c.args)); got != want {
				t.Errorf("read with pattern %v:\n%s\ngrep:\n%s", c.args, got, want)
			}
		})
	}
	// On a file: only the matching lines, numbered, with context.
	out := mustOK(t, call(t, only, Read, map[string]any{"path": "ctx.txt", "grep": "needle 2", "context": 1}))
	contains(t, out, "1 match in 1 file under ctx.txt\nctx.txt-9-h\nctx.txt:10:needle 2\nctx.txt-11-i")
	lacks(t, out, "needle 1", "needle 3")
}

func TestReadLargeFileNote(t *testing.T) {
	log := lines(3000, func(i int) string { return fmt.Sprintf("step %04d ok %s", i, strings.Repeat("-", 20)) })
	root := tree(t, map[string]string{"ci.log": log, "small.txt": "a\nb\n"})
	ts := mount(t, Config{Root: root})
	note := fmt.Sprintf("ci.log is %s, 3000 lines: search it with grep", sizeOf(int64(len(log))))
	// No range on a large file: a short head (largeHead), the note, and no
	// paging advice — searching is the next step, not reading on.
	head := mustOK(t, call(t, ts, Read, map[string]any{"path": "ci.log"}))
	contains(t, head, "lines 1-", note, `grep:"ERROR|FAIL"`)
	lacks(t, head, "cut at", "Continue with offset")
	if len(head) > largeHead+400 {
		t.Errorf("the head of a large file is %d bytes, want about %d", len(head), largeHead)
	}
	out := mustOK(t, call(t, ts, Read, map[string]any{"path": "ci.log", "offset": -120}))
	contains(t, out, "lines 2881-3000 of 3000", note)
	if !strings.HasSuffix(out, "or read a range]") {
		t.Errorf("the note is not the last line:\n%s", out[len(out)-200:])
	}
	lacks(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "ci.log", "offset": 100, "limit": 5})), note)
	lacks(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "ci.log", "grep": "step 0042"})), note)
	lacks(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "small.txt"})), "search it with grep")
}

func TestReadDef(t *testing.T) {
	ts := mount(t, Config{Root: t.TempDir(), Tools: []string{Read}})
	d := ts.Defs()[0]
	contains(t, d.Function.Description, "list a directory", "with grep search a file or a directory tree", "large file or a log")
	props := d.Function.Parameters.(map[string]any)["properties"].(map[string]any)
	for _, p := range []string{"path", "offset", "limit", "grep", "context", "glob", "sizes"} {
		if props[p] == nil {
			t.Errorf("read does not advertise %s", p)
		}
	}
}

func TestReadStreamsLargeFiles(t *testing.T) {
	// 20MB file: a ranged read near the end must not need the whole file in
	// memory to answer, and must answer correctly.
	root := t.TempDir()
	f, err := os.Create(filepath.Join(root, "huge.log"))
	must(t, err)
	for i := 1; i <= 200000; i++ {
		fmt.Fprintf(f, "entry %06d %s\n", i, strings.Repeat("z", 90))
	}
	must(t, f.Close())
	ts := mount(t, Config{Root: root})
	out := mustOK(t, call(t, ts, Read, map[string]any{"path": "huge.log", "offset": -2}))
	contains(t, out, "lines 199999-200000 of 200000", "200000\tentry 200000")
}
