package llm

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

// dumpFiles returns the dump files in dir, sorted by name (= by time).
func dumpFiles(t *testing.T, dir string) []string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dump dir: %v", err)
	}
	var out []string
	for _, e := range ents {
		out = append(out, filepath.Join(dir, e.Name()))
	}
	sort.Strings(out)
	return out
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func drainDump(t *testing.T, ch <-chan StreamChunk) {
	t.Helper()
	for c := range ch {
		if err := c.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
	}
}

// The dump exists for what the parser hides: a comment line, the [DONE]
// sentinel, the exact request body with its tools. Losing any of them, or
// leaking the key, makes the capture useless or dangerous.
func TestWireDump_StreamKeepsEveryLineAndRedactsKeys(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "dump") // missing: must be created
	t.Setenv(WireDumpEnv, dir)

	var mu sync.Mutex
	var sent []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		sent = b
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Session-Token", "tok-from-server")
		io.WriteString(w, ": keep-alive\n\n")
		flush(w)
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "sk-secret-123", "m")
	c.Header = http.Header{"X-Api-Key": {"xk-secret-456"}}
	var td ToolDef
	td.Type = "function"
	td.Function.Name = "look_up"
	td.Function.Parameters = map[string]any{"type": "object"}
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, []ToolDef{td}, &ChatOpts{TraceID: "thread/1"})
	if err != nil {
		t.Fatal(err)
	}
	drainDump(t, ch)

	files := dumpFiles(t, dir)
	if len(files) != 1 {
		t.Fatalf("want 1 dump file, got %v", files)
	}
	if fi, _ := os.Stat(files[0]); fi.Mode().Perm() != 0o600 {
		t.Errorf("file mode %v, want 0600", fi.Mode().Perm())
	}
	if fi, _ := os.Stat(dir); fi.Mode().Perm() != 0o700 {
		t.Errorf("dir mode %v, want 0700", fi.Mode().Perm())
	}
	if base := filepath.Base(files[0]); !strings.Contains(base, "-thread_1-a1.txt") {
		t.Errorf("name %q lacks trace id and attempt", base)
	}
	got := readFile(t, files[0])
	for _, leak := range []string{"sk-secret-123", "xk-secret-456", "tok-from-server"} {
		if strings.Contains(got, leak) {
			t.Errorf("dump leaks %q:\n%s", leak, got)
		}
	}
	for _, want := range []string{
		"> POST " + srv.URL + "/v1/chat/completions\n",
		"> Authorization: [redacted]\n",
		"> X-Api-Key: [redacted]\n",
		"> X-Trace-Id: thread/1\n",
		"< 200 OK ",
		"< X-Session-Token: [redacted]\n",
		" : keep-alive\n",
		" data: [DONE]\n",
		"# eof ",
		"# closed ",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("dump lacks %q:\n%s", want, got)
		}
	}
	mu.Lock()
	body := sent
	mu.Unlock()
	if !bytes.Contains([]byte(got), append(body, '\n')) || !bytes.Contains(body, []byte(`"look_up"`)) {
		t.Errorf("request body not in dump byte for byte (tools included):\nsent %s\ndump:\n%s", body, got)
	}
	// Every SSE line, blank separators included, in order.
	var lines []string
	for _, l := range strings.Split(got, "\n") {
		if strings.HasPrefix(l, "+") {
			_, rest, _ := strings.Cut(l, " ")
			lines = append(lines, rest)
		}
	}
	want := []string{": keep-alive", "", `data: {"choices":[{"delta":{"content":"hi"}}]}`, "",
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`, "", "data: [DONE]", ""}
	if strings.Join(lines, "|") != strings.Join(want, "|") {
		t.Errorf("body lines\n got %q\nwant %q", lines, want)
	}
}

// Off must mean off: no directory, no file.
func TestWireDump_OffWritesNothing(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "dump")
	t.Setenv(WireDumpEnv, "")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	ch, err := NewClient(srv.URL, "k", "m").ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	drainDump(t, ch)
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("dump dir exists with the env empty: %v", err)
	}
}

// A key in the URL query is a credential too (Gemini's ?key=). The other
// parameters stay as sent, in order.
func TestWireDump_RedactsKeyInURLQuery(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(WireDumpEnv, dir)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTeapot)
		io.WriteString(w, `{"error":"no"}`)
	}))
	defer srv.Close()
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/models?b=2&key=AIza-secret&a=1&access_token=t0", nil)
	resp, err := NewClient(srv.URL, "", "m").do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.ReadAll(resp.Body)
	resp.Body.Close()

	files := dumpFiles(t, dir)
	if len(files) != 1 {
		t.Fatalf("want 1 dump file, got %v", files)
	}
	got := readFile(t, files[0])
	if strings.Contains(got, "AIza-secret") || strings.Contains(got, "=t0") {
		t.Errorf("query key leaked:\n%s", got)
	}
	for _, want := range []string{
		"/models?b=2&key=[redacted]&a=1&access_token=[redacted]\n",
		"< 418 I'm a teapot ",
		` {"error":"no"} # no trailing newline` + "\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("dump lacks %q:\n%s", want, got)
		}
	}
}

// A retry is a second HTTP attempt and gets its own file, named with its
// attempt number, so the rejected attempt and the one that answered can be told
// apart.
func TestWireDump_RetryIsSeparateAttempt(t *testing.T) {
	fastBackoff(t)
	dir := t.TempDir()
	t.Setenv(WireDumpEnv, dir)
	var mu sync.Mutex
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hits++
		n := hits
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			io.WriteString(w, "busy\n")
			return
		}
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	ch, err := NewClient(srv.URL, "", "m").ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, &ChatOpts{TraceID: "t1"})
	if err != nil {
		t.Fatal(err)
	}
	drainDump(t, ch)

	files := dumpFiles(t, dir)
	if len(files) != 2 {
		t.Fatalf("want 2 dump files, got %v", files)
	}
	if !strings.HasSuffix(files[0], "-t1-a1.txt") || !strings.HasSuffix(files[1], "-t1-a2.txt") {
		t.Errorf("attempts not named in order: %v", files)
	}
	if first := readFile(t, files[0]); !strings.Contains(first, "< 503 ") || !strings.Contains(first, " busy\n") {
		t.Errorf("first attempt lacks the 503 body:\n%s", first)
	}
	if second := readFile(t, files[1]); !strings.Contains(second, " data: [DONE]\n") {
		t.Errorf("second attempt lacks the stream:\n%s", second)
	}
}
