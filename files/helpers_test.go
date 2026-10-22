package files

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// tree writes files (slash paths → content) under a new temp dir.
func tree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for p, c := range files {
		full := filepath.Join(root, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func gitInit(t *testing.T, root string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	cmd := exec.Command("git", "init", "-q", root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
}

func mount(t *testing.T, cfg Config) *Toolset {
	t.Helper()
	ts, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return ts
}

// call runs a bare tool through Dispatch, the path a model takes.
func call(t *testing.T, ts *Toolset, bare string, a map[string]any) string {
	t.Helper()
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	return ts.Dispatch(context.Background(), ts.Name(bare), string(b))
}

func mustOK(t *testing.T, out string) string {
	t.Helper()
	if strings.HasPrefix(out, "ERROR") {
		t.Fatalf("unexpected refusal: %s", out)
	}
	return out
}

func mustErr(t *testing.T, out string, wants ...string) {
	t.Helper()
	if !strings.HasPrefix(out, "ERROR: ") {
		t.Fatalf("want a refusal, got: %s", out)
	}
	for _, w := range wants {
		if !strings.Contains(out, w) {
			t.Errorf("refusal missing %q:\n%s", w, out)
		}
	}
}

func contains(t *testing.T, out string, wants ...string) {
	t.Helper()
	for _, w := range wants {
		if !strings.Contains(out, w) {
			t.Errorf("missing %q in:\n%s", w, out)
		}
	}
}

func lacks(t *testing.T, out string, nots ...string) {
	t.Helper()
	for _, w := range nots {
		if strings.Contains(out, w) {
			t.Errorf("unexpected %q in:\n%s", w, out)
		}
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func lines(n int, f func(i int) string) string {
	var b strings.Builder
	for i := 1; i <= n; i++ {
		b.WriteString(f(i) + "\n")
	}
	return b.String()
}

func str(s string) *string { return &s }
