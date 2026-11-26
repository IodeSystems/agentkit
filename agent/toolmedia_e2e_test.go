package agent_test

// End to end: a real Session, a real *llm.Client against httptest servers, and
// files.Toolset reading a PNG. Guards the whole media route: tool result Parts →
// Entry.Parts → OpenAI two-phase form / Messages API tool_result image block,
// and that a stored history replays across a model switch.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/iodesystems/agentkit/agent"
	"github.com/iodesystems/agentkit/files"
	"github.com/iodesystems/agentkit/llm"
)

type e2eStore struct {
	mu      sync.Mutex
	entries []agent.Entry
	pending int
}

func (m *e2eStore) ClaimPending(context.Context, string, int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := m.pending
	m.pending = 0
	return n, nil
}

func (m *e2eStore) Append(_ context.Context, _ string, e agent.Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, e)
	return nil
}

func (m *e2eStore) Context(context.Context, string) ([]agent.Entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]agent.Entry(nil), m.entries...), nil
}

func (m *e2eStore) Compact(context.Context, string, agent.Compaction) error { return nil }

func (m *e2eStore) queueUser(id, text string, at int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, agent.Entry{ID: id, Kind: agent.KindUser, Content: text, CreatedAt: at})
	m.pending++
}

// e2eServer answers round i with replies[i] and keeps every request body.
type e2eServer struct {
	URL    string
	mu     sync.Mutex
	bodies []map[string]any
}

func (s *e2eServer) body(i int) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	if i >= len(s.bodies) {
		return map[string]any{}
	}
	return s.bodies[i]
}

func newE2EServer(t *testing.T, replies []string) *e2eServer {
	t.Helper()
	s := &e2eServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		s.mu.Lock()
		n := len(s.bodies)
		s.bodies = append(s.bodies, m)
		s.mu.Unlock()
		if n >= len(replies) {
			t.Errorf("unexpected request %d", n+1)
			http.Error(w, "no script", 500)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, replies[n])
	}))
	t.Cleanup(srv.Close)
	s.URL = srv.URL
	return s
}

func sse(events ...string) string {
	var sb strings.Builder
	for _, e := range events {
		var h struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(e), &h)
		fmt.Fprintf(&sb, "event: %s\ndata: %s\n\n", h.Type, e)
	}
	return sb.String()
}

func oaiCall(id, name, args string) string {
	a, _ := json.Marshal(args)
	return fmt.Sprintf(`data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":%q,"type":"function","function":{"name":%q,"arguments":%s}}]}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`, id, name, a)
}

func oaiText(text string) string {
	t, _ := json.Marshal(text)
	return fmt.Sprintf(`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":%s}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`, t)
}

const msgStart = `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5","usage":{"input_tokens":10,"output_tokens":1}}}`

func antCall(id, name, args string) string {
	a, _ := json.Marshal(args)
	return sse(msgStart,
		fmt.Sprintf(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":%q,"name":%q,"input":{}}}`, id, name),
		fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":%s}}`, a),
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`)
}

func antText(text string) string {
	t, _ := json.Marshal(text)
	return sse(msgStart,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%s}}`, t),
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`)
}

func makePNG(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for x := 0; x < 8; x++ {
		for y := 0; y < 8; y++ {
			img.Set(x, y, color.RGBA{uint8(x * 30), uint8(y * 30), 200, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pic.png"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func e2eSession(t *testing.T, root string, c *llm.Client, store agent.Store) *agent.Session {
	t.Helper()
	ts, err := files.New(files.Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	return &agent.Session{
		SessionID: "s1", System: "sys", Store: store, Runner: c,
		Tools:    ts.Defs(),
		Dispatch: ts.Wrap(func(context.Context, llm.ToolCall) (llm.ToolResult, error) { return llm.TextResult("unknown"), nil }),
	}
}

func msgsOf(b map[string]any) []map[string]any {
	raw, _ := b["messages"].([]any)
	var out []map[string]any
	for _, m := range raw {
		out = append(out, m.(map[string]any))
	}
	return out
}

func contentStr(m map[string]any) string {
	if s, ok := m["content"].(string); ok {
		return s
	}
	b, _ := json.Marshal(m["content"])
	return string(b)
}

// assertTwoPhase checks the OpenAI render: a tool message whose text names the
// attachment, then a user message with an image_url data: URI right after it.
// It returns the index of the tool message.
func assertTwoPhase(t *testing.T, msgs []map[string]any) int {
	t.Helper()
	for i, m := range msgs {
		if m["role"] != "tool" {
			continue
		}
		if s := contentStr(m); !strings.Contains(s, "image/png is attached in the next user message") {
			t.Fatalf("tool message lacks the attachment note: %s", s)
		}
		if i+1 >= len(msgs) || msgs[i+1]["role"] != "user" {
			t.Fatalf("no user message straight after the tool message: %v", msgs[i:])
		}
		parts, _ := msgs[i+1]["content"].([]any)
		for _, p := range parts {
			pm, _ := p.(map[string]any)
			if pm["type"] == "image_url" {
				iu, _ := pm["image_url"].(map[string]any)
				if u, _ := iu["url"].(string); strings.HasPrefix(u, "data:image/png;base64,") {
					return i
				}
			}
		}
		t.Fatalf("user message after the tool message has no image_url data: URI: %v", msgs[i+1])
	}
	t.Fatal("no tool message in request")
	return -1
}

func toolEntry(t *testing.T, st *e2eStore) agent.Entry {
	t.Helper()
	es, _ := st.Context(context.Background(), "s1")
	for _, e := range es {
		if e.Kind == agent.KindToolResult {
			return e
		}
	}
	t.Fatal("no tool result entry")
	return agent.Entry{}
}

func hasImagePart(e agent.Entry) bool {
	for _, p := range e.Parts {
		if p.Type == "image_url" && p.ImageURL != nil && strings.HasPrefix(p.ImageURL.URL, "data:image/png;base64,") {
			return true
		}
	}
	return false
}

func TestE2E_ToolMediaOpenAIFormat(t *testing.T) {
	root := makePNG(t)
	srv := newE2EServer(t, []string{
		oaiCall("c1", "file_read", `{"path":"pic.png"}`),
		oaiText("a gradient"),
		oaiText("still a gradient"),
	})
	c := llm.NewClient(srv.URL+"/v1", "k", "gpt-test")
	store := &e2eStore{}
	s := e2eSession(t, root, c, store)

	store.queueUser("u1", "look at pic.png", 1)
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e := toolEntry(t, store); !hasImagePart(e) {
		t.Fatalf("stored tool result lost the image: %+v", e)
	}
	r2 := msgsOf(srv.body(1))
	end1 := assertTwoPhase(t, r2) + 2 // the tool message and its image user message

	store.queueUser("u2", "and again?", time.Now().UnixNano())
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	r3 := msgsOf(srv.body(2))
	assertTwoPhase(t, r3)
	if len(r3) <= len(r2) {
		t.Fatalf("request 3 (%d msgs) not longer than request 2 (%d)", len(r3), len(r2))
	}
	if !reflect.DeepEqual(r2[:end1], r3[:end1]) {
		a, _ := json.Marshal(r2[:end1])
		b, _ := json.Marshal(r3[:end1])
		t.Fatalf("prefix not stable across rounds:\nreq2 %s\nreq3 %s", a, b)
	}
}

func TestE2E_ToolMediaMessagesAPIAndModelSwitch(t *testing.T) {
	root := makePNG(t)
	srv := newE2EServer(t, []string{
		antCall("c1", "file_read", `{"path":"pic.png"}`),
		antText("a gradient"),
	})
	c := llm.NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	store := &e2eStore{}
	s := e2eSession(t, root, c, store)

	store.queueUser("u1", "look at pic.png", 1)
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if e := toolEntry(t, store); !hasImagePart(e) {
		t.Fatalf("stored tool result lost the image: %+v", e)
	}

	found := false
	for _, m := range msgsOf(srv.body(1)) {
		blocks, _ := m["content"].([]any)
		for _, b := range blocks {
			bm, _ := b.(map[string]any)
			if bm["type"] != "tool_result" {
				continue
			}
			inner, _ := bm["content"].([]any)
			for _, ib := range inner {
				if ibm, _ := ib.(map[string]any); ibm["type"] == "image" {
					found = true
				}
			}
			if !found {
				raw, _ := json.Marshal(bm)
				t.Fatalf("tool_result has no image block: %s", raw)
			}
		}
	}
	if !found {
		raw, _ := json.Marshal(srv.body(1))
		t.Fatalf("request 2 has no tool_result: %s", raw)
	}

	// Model switch: the same stored history, an OpenAI-format client.
	oai := newE2EServer(t, []string{oaiText("ok")})
	c2 := llm.NewClient(oai.URL+"/v1", "k", "gpt-test")
	s2 := e2eSession(t, root, c2, store)
	store.queueUser("u2", "what was it?", time.Now().UnixNano())
	if _, err := s2.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	assertTwoPhase(t, msgsOf(oai.body(0)))
}
