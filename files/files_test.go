package files

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func TestNamesAndDefs(t *testing.T) {
	root := tree(t, nil)
	cases := []struct {
		name string
		cfg  Config
		want []string
	}{
		{"defaults", Config{Root: root}, []string{"file_read", "file_edit", "file_write", "file_grep", "file_list"}},
		{"all", Config{Root: root, Tools: allTools}, []string{"file_read", "file_edit", "file_write", "file_grep", "file_list", "file_delete", "file_move"}},
		{"read-only drops writers", Config{Root: root, Tools: allTools, ReadOnly: true}, []string{"file_read", "file_grep", "file_list"}},
		{"prefix", Config{Root: root, Tools: []string{Read}, NamePrefix: "fs-"}, []string{"fs-read"}},
		{"empty set", Config{Root: root, Tools: []string{}}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ts := mount(t, c.cfg)
			if got := ts.Names(); !reflect.DeepEqual(got, c.want) {
				t.Fatalf("Names = %v, want %v", got, c.want)
			}
			defs := ts.Defs()
			if len(defs) != len(c.want) {
				t.Fatalf("%d defs for %d names", len(defs), len(c.want))
			}
			for i, d := range defs {
				if d.Function.Name != c.want[i] || d.Type != "function" || d.Function.Description == "" || d.Function.Parameters == nil {
					t.Errorf("def %d malformed: %+v", i, d)
				}
			}
		})
	}
}

func TestNewRefuses(t *testing.T) {
	root := tree(t, map[string]string{"f": "x"})
	cases := []struct {
		name string
		cfg  Config
		want string
	}{
		{"no root", Config{}, "Root is required"},
		{"root is a file", Config{Root: filepath.Join(root, "f")}, "not a directory"},
		{"missing root", Config{Root: filepath.Join(root, "nope")}, "root"},
		{"dotted prefix", Config{Root: root, NamePrefix: "file."}, "must match"},
		{"unknown tool", Config{Root: root, Tools: []string{"chmod"}}, "unknown tool"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := New(c.cfg)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want %q", err, c.want)
			}
		})
	}
}

func TestReadOnlyRefusesWriters(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "hello\n"})
	ts := mount(t, Config{Root: root, ReadOnly: true})
	for _, n := range []string{"file_edit", "file_write", "file_delete", "file_move"} {
		out := ts.Dispatch(context.Background(), n, `{"path":"a.txt","old":"hello","new":"bye","content":"x"}`)
		mustErr(t, out, "unknown tool")
	}
	if readFile(t, filepath.Join(root, "a.txt")) != "hello\n" {
		t.Fatal("read-only toolset changed a file")
	}
	mustOK(t, call(t, ts, Read, map[string]any{"path": "a.txt"}))
	if _, err := ts.Run(context.Background(), Edit, map[string]any{"path": "a.txt"}); err == nil {
		t.Fatal("Run edit on a read-only toolset succeeded")
	}
}

func TestWrapRoutesOnlyItsOwn(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "hello\n"})
	ts := mount(t, Config{Root: root})
	var passed []string
	inner := func(_ context.Context, tc llm.ToolCall) (string, error) {
		passed = append(passed, tc.Function.Name)
		return "inner", nil
	}
	d := ts.Wrap(inner)
	tc := func(name, args string) llm.ToolCall {
		var c llm.ToolCall
		c.Function.Name, c.Function.Arguments = name, args
		return c
	}
	if out, _ := d(context.Background(), tc("exec", `{}`)); out != "inner" {
		t.Fatalf("exec not passed through: %q", out)
	}
	if out, _ := d(context.Background(), tc("file_read", `{"path":"a.txt"}`)); !strings.Contains(out, "1\thello") {
		t.Fatalf("file_read not handled: %q", out)
	}
	if out, _ := d(context.Background(), tc("file_read", `{"path":`)); !strings.HasPrefix(out, "ERROR") {
		t.Fatalf("bad JSON not refused: %q", out)
	}
	if !reflect.DeepEqual(passed, []string{"exec"}) {
		t.Fatalf("inner saw %v", passed)
	}
}

func TestFlexibleArgs(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": lines(10, func(i int) string { return fmt.Sprint("l", i) })})
	ts := mount(t, Config{Root: root})
	out := ts.Dispatch(context.Background(), "file_read", `{"path":"a.txt","offset":"3","limit":2.0}`)
	contains(t, mustOK(t, out), "lines 3-4 of 10", "3\tl3", "4\tl4")
}

func TestEscapes(t *testing.T) {
	outside := tree(t, map[string]string{"secret.txt": "secret\n"})
	root := tree(t, map[string]string{"in.txt": "inside\n", "dir/f.txt": "f\n"})
	must(t, os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "out-file")))
	must(t, os.Symlink(outside, filepath.Join(root, "out-dir")))
	must(t, os.Symlink(filepath.Join(root, "in.txt"), filepath.Join(root, "in-link")))
	must(t, os.Symlink("dir", filepath.Join(root, "dir-link")))
	ts := mount(t, Config{Root: root, Tools: allTools})

	refused := []struct {
		name string
		tool string
		args map[string]any
	}{
		{"dotdot read", Read, map[string]any{"path": "../secret.txt"}},
		{"nested dotdot", Read, map[string]any{"path": "dir/../../x"}},
		{"absolute outside", Read, map[string]any{"path": filepath.Join(outside, "secret.txt")}},
		{"etc", Read, map[string]any{"path": "/etc/passwd"}},
		{"symlink file out", Read, map[string]any{"path": "out-file"}},
		{"through symlink dir", Read, map[string]any{"path": "out-dir/secret.txt"}},
		{"write through symlink dir", Write, map[string]any{"path": "out-dir/new.txt", "content": "x"}},
		{"write over out link", Write, map[string]any{"path": "out-file", "content": "x", "overwrite": true}},
		{"edit through link", Edit, map[string]any{"path": "out-file", "old": "secret", "new": "x"}},
		{"grep path out", Grep, map[string]any{"pattern": "secret", "path": "out-dir"}},
		{"grep path dotdot", Grep, map[string]any{"pattern": "secret", "path": ".."}},
		{"list out", List, map[string]any{"path": "out-dir"}},
		{"list dotdot", List, map[string]any{"path": "../"}},
		{"move out", Move, map[string]any{"path": "in.txt", "to": "../stolen.txt"}},
		{"move into out dir", Move, map[string]any{"path": "in.txt", "to": "out-dir/stolen.txt"}},
		{"delete through link dir", Delete, map[string]any{"path": "out-dir/secret.txt"}},
	}
	for _, c := range refused {
		t.Run(c.name, func(t *testing.T) {
			mustErr(t, call(t, ts, c.tool, c.args), "outside the root")
		})
	}
	if readFile(t, filepath.Join(outside, "secret.txt")) != "secret\n" {
		t.Fatal("outside file changed")
	}
	if _, err := os.Stat(filepath.Join(outside, "new.txt")); err == nil {
		t.Fatal("write landed outside")
	}

	allowed := []struct {
		name string
		args map[string]any
		want string
	}{
		{"link inside", map[string]any{"path": "in-link"}, "1\tinside"},
		{"dir link inside", map[string]any{"path": "dir-link/f.txt"}, "1\tf"},
		{"absolute inside", map[string]any{"path": filepath.Join(root, "in.txt")}, "1\tinside"},
		{"dot segments inside", map[string]any{"path": "dir/../in.txt"}, "1\tinside"},
	}
	for _, c := range allowed {
		t.Run(c.name, func(t *testing.T) {
			contains(t, mustOK(t, call(t, ts, Read, c.args)), c.want)
		})
	}

	// Under, exported for Resolve hooks, applies the same rule.
	if _, err := Under(root, "out-file"); !errors.Is(err, ErrEscape) {
		t.Fatalf("Under(out-file) = %v", err)
	}
	if p, err := Under(root, "in-link"); err != nil || filepath.Base(p) != "in.txt" {
		t.Fatalf("Under(in-link) = %q, %v", p, err)
	}
}

func TestSymlinkedRoot(t *testing.T) {
	real := tree(t, map[string]string{"a.txt": "a\n"})
	link := filepath.Join(t.TempDir(), "root-link")
	must(t, os.Symlink(real, link))
	ts := mount(t, Config{Root: link})
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "a.txt"})), "1\ta")
	// The model may be told the unresolved absolute root.
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": filepath.Join(link, "a.txt")})), "1\ta")
}

func TestHooks(t *testing.T) {
	root := tree(t, map[string]string{"a.txt": "hello\n", "other/b.txt": "b\n"})
	var resolved []string
	var refused []error
	var written [][]string
	ts := mount(t, Config{
		Root:  root,
		Tools: allTools,
		Resolve: func(rel string) (string, error) {
			resolved = append(resolved, rel)
			if strings.HasPrefix(rel, "dep:") {
				return filepath.Join(root, "other", strings.TrimPrefix(rel, "dep:")), nil
			}
			return Under(root, rel)
		},
		Refusal: func(path string, reason error) string {
			refused = append(refused, reason)
			return "custom refusal for " + path + " (use dep.read)"
		},
		AfterWrite: func(paths []string) { written = append(written, paths) },
	})
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "dep:b.txt"})), "1\tb")
	mustErr(t, call(t, ts, Read, map[string]any{"path": "../x"}), "custom refusal for ../x (use dep.read)")
	if len(refused) != 1 || !errors.Is(refused[0], ErrEscape) {
		t.Fatalf("Refusal saw %v", refused)
	}
	// The default resolver's not-found refusal goes through the hook too.
	ts2 := mount(t, Config{Root: root, Refusal: func(path string, reason error) string {
		refused = append(refused, reason)
		return "hooked: " + reason.Error()
	}})
	mustErr(t, call(t, ts2, Read, map[string]any{"path": "nope.txt"}), "hooked: nope.txt: does not exist")
	if !errors.Is(refused[len(refused)-1], ErrNotFound) {
		t.Fatalf("not-found reason = %v", refused[len(refused)-1])
	}

	mustOK(t, call(t, ts, Read, map[string]any{"path": "a.txt"}))
	mustOK(t, call(t, ts, Edit, map[string]any{"path": "a.txt", "old": "hello", "new": "bye"}))
	mustOK(t, call(t, ts, Write, map[string]any{"path": "n/new.txt", "content": "x\n"}))
	mustOK(t, call(t, ts, Move, map[string]any{"path": "n/new.txt", "to": "n/moved.txt"}))
	mustOK(t, call(t, ts, Delete, map[string]any{"path": "n/moved.txt"}))
	want := [][]string{{"a.txt"}, {"n/new.txt"}, {"n/moved.txt", "n/new.txt"}, {"n/moved.txt"}}
	if !reflect.DeepEqual(written, want) {
		t.Fatalf("AfterWrite = %v, want %v", written, want)
	}
	if len(resolved) == 0 {
		t.Fatal("Resolve never called")
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
