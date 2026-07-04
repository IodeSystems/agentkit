package mcpmgr

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeResolver is an in-memory SecretResolver for tests — no DB, no
// real crypto.
type fakeResolver struct {
	kv  map[string]string
	err error
}

func (f fakeResolver) Resolve(ctx context.Context, ref string) (map[string]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.kv, nil
}

func TestMaterializeSecret_PlaceholderAndPerms(t *testing.T) {
	m := NewManagerWithSecrets(fakeResolver{kv: map[string]string{
		"GMAIL_USER":         "a@b.com",
		"GMAIL_APP_PASSWORD": "pw",
	}})

	cfg := MCPConfig{
		Name:         "gmcpail",
		SecretRef:    "gmail-creds",
		SecretFormat: "env",
		Args:         []string{"--config", secretFilePlaceholder, "--verbose"},
		Env:          []string{"CONFIG_PATH=" + secretFilePlaceholder, "OTHER=1"},
	}

	args, env, dir, err := m.materializeSecret(context.Background(), cfg)
	if err != nil {
		t.Fatalf("materializeSecret: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })

	// Placeholder gone from both Args and Env; a real path took its place.
	path := args[1]
	if path == secretFilePlaceholder || strings.Contains(path, "{{") {
		t.Fatalf("Args placeholder not substituted: %q", args)
	}
	if env[0] != "CONFIG_PATH="+path {
		t.Fatalf("Env placeholder not substituted: %q", env)
	}
	// Untouched entries pass through.
	if args[0] != "--config" || args[2] != "--verbose" || env[1] != "OTHER=1" {
		t.Fatalf("non-placeholder entries mutated: args=%q env=%q", args, env)
	}
	// No secret VALUE leaked into Args/Env — only the path.
	for _, e := range append(append([]string{}, args...), env...) {
		if strings.Contains(e, "pw") || strings.Contains(e, "a@b.com") {
			t.Fatalf("secret value leaked into args/env: %q", e)
		}
	}

	// File exists, mode 0600, content is the rendered env.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat secret file: %v", err)
	}
	if runtime.GOOS != "windows" {
		if got := info.Mode().Perm(); got != 0o600 {
			t.Fatalf("secret file perm = %o, want 0600", got)
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read secret file: %v", err)
	}
	want := "GMAIL_APP_PASSWORD=pw\nGMAIL_USER=a@b.com\n"
	if string(body) != want {
		t.Fatalf("secret file content = %q, want %q", body, want)
	}

	// Cleanup contract: removing the dir clears the file (mirrors
	// MCPServer.close / teardown).
	if filepath.Dir(path) != dir {
		t.Fatalf("secret file not under returned dir: %q vs %q", path, dir)
	}
	os.RemoveAll(dir)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("secret file survived dir removal: %v", err)
	}
}

func TestMaterializeSecret_NoResolver(t *testing.T) {
	m := NewManager() // no resolver wired
	cfg := MCPConfig{Name: "x", SecretRef: "ref", SecretFormat: "env"}
	if _, _, _, err := m.materializeSecret(context.Background(), cfg); err == nil {
		t.Fatal("expected error when no resolver is configured")
	}
}

func TestMaterializeSecret_ResolverError(t *testing.T) {
	m := NewManagerWithSecrets(fakeResolver{err: errors.New("boom")})
	cfg := MCPConfig{Name: "x", SecretRef: "ref", SecretFormat: "env"}
	_, _, dir, err := m.materializeSecret(context.Background(), cfg)
	if err == nil {
		t.Fatal("expected error from resolver")
	}
	if dir != "" {
		t.Fatalf("no temp dir should be created on resolver error, got %q", dir)
	}
}
