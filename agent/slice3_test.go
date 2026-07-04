package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// ── lifting: wire protocol ──────────────────────────────────────────

func TestParseLiftRequest(t *testing.T) {
	cases := []struct {
		name   string
		result string
		wantOK bool
		wantID string
	}{
		{"happy", `{"pending":true,"correlation_id":"abc","ttl_s":30}`, true, "abc"},
		{"no ttl", `{"pending":true,"correlation_id":"x"}`, true, "x"},
		{"not pending", `{"pending":false,"correlation_id":"x"}`, false, ""},
		{"missing corr", `{"pending":true}`, false, ""},
		{"plain text", `done`, false, ""},
		{"other json", `{"result":"ok"}`, false, ""},
		{"empty", ``, false, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			lr, ok := ParseLiftRequest(c.result)
			if ok != c.wantOK {
				t.Fatalf("ok=%v want %v", ok, c.wantOK)
			}
			if ok && lr.CorrelationID != c.wantID {
				t.Fatalf("corr=%q want %q", lr.CorrelationID, c.wantID)
			}
		})
	}
}

func TestPendingResult(t *testing.T) {
	msg := PendingResult("abc", "call_1", 30)
	for _, want := range []string{"PENDING", "correlation_id=abc", "tool_call_id=call_1", "Deadline: 30s", "Do not retry"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message missing %q: %s", want, msg)
		}
	}
	if strings.Contains(PendingResult("a", "b", 0), "Deadline") {
		t.Error("ttl=0 should omit deadline")
	}
}

func TestParseClearRequest(t *testing.T) {
	cases := []struct {
		name    string
		result  string
		wantOK  bool
		wantKey string
	}{
		{"happy", `{"clear":true,"group_by":"file","key":"a.go"}`, true, "a.go"},
		{"not clear", `{"clear":false,"group_by":"file","key":"a.go"}`, false, ""},
		{"missing group_by", `{"clear":true,"key":"a.go"}`, false, ""},
		{"missing key", `{"clear":true,"group_by":"file"}`, false, ""},
		{"plain", `resolved`, false, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cr, ok := ParseClearRequest(c.result)
			if ok != c.wantOK {
				t.Fatalf("ok=%v want %v", ok, c.wantOK)
			}
			if ok && cr.Key != c.wantKey {
				t.Fatalf("key=%q want %q", cr.Key, c.wantKey)
			}
		})
	}
}

// ── validation: schema fix loop ─────────────────────────────────────

func toolDef(name string, params any) llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = name
	td.Function.Parameters = params
	return td
}

func TestSchemaValidator_RequiredAndTypes(t *testing.T) {
	tools := []llm.ToolDef{toolDef("submit", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"reply":  map[string]any{"type": "string"},
			"count":  map[string]any{"type": "integer"},
			"issues": map[string]any{"type": "array"},
		},
		"required": []string{"reply", "issues"},
	})}
	v := NewSchemaValidator(tools)

	cases := []struct {
		name    string
		args    string
		wantErr string // substring; "" = accept
	}{
		{"valid", `{"reply":"hi","issues":[]}`, ""},
		{"valid all", `{"reply":"hi","issues":[1],"count":3}`, ""},
		{"missing required", `{"reply":"hi"}`, "missing required field(s): issues"},
		{"two missing", `{}`, "issues, reply"},
		{"null required", `{"reply":"hi","issues":null}`, "missing required field(s): issues"},
		{"wrong type", `{"reply":123,"issues":[]}`, "reply should be string, got number"},
		{"integer ok as number", `{"reply":"h","issues":[],"count":5}`, ""},
		{"bad json", `not json`, "not a JSON object"},
		{"unknown tool ignored", `garbage`, ""}, // only when tool unknown; see below
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			name := "submit"
			if c.name == "unknown tool ignored" {
				name = "nope"
			}
			err := v.ValidateArgs(name, c.args)
			if c.wantErr == "" {
				if err != nil {
					t.Fatalf("want accept, got %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Fatalf("want err containing %q, got %v", c.wantErr, err)
			}
		})
	}
}

func TestValidatingDispatcher_FixLoop(t *testing.T) {
	tools := []llm.ToolDef{toolDef("submit", map[string]any{
		"type":     "object",
		"required": []string{"reply"},
	})}
	var dispatched int
	inner := func(_ context.Context, tc llm.ToolCall) (string, error) {
		dispatched++
		return "ok", nil
	}
	d := ValidatingDispatcher(inner, NewSchemaValidator(tools))

	bad := llm.ToolCall{}
	bad.Function.Name = "submit"
	bad.Function.Arguments = `{}`
	res, err := d(context.Background(), bad)
	if err != nil {
		t.Fatalf("validation failure must be a soft result, got err %v", err)
	}
	if dispatched != 0 {
		t.Fatal("inner dispatched despite invalid args")
	}
	if !strings.Contains(res, "INVALID") || !strings.Contains(res, "reply") {
		t.Fatalf("fix instruction malformed: %s", res)
	}

	good := llm.ToolCall{}
	good.Function.Name = "submit"
	good.Function.Arguments = `{"reply":"hi"}`
	if res, err := d(context.Background(), good); err != nil || res != "ok" || dispatched != 1 {
		t.Fatalf("valid call not passed through: res=%q err=%v dispatched=%d", res, err, dispatched)
	}
}

// ── batching: N queued messages → ONE turn ──────────────────────────

func TestTurn_BatchesQueuedMessages(t *testing.T) {
	store := &memStore{}
	// Three messages queued between activations.
	store.queue(Entry{ID: "m1", Kind: KindUser, Content: "first", CreatedAt: 1})
	store.queue(Entry{ID: "m2", Kind: KindUser, Content: "second", CreatedAt: 2})
	store.queue(Entry{ID: "m3", Kind: KindUser, Content: "third", CreatedAt: 3})

	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "acknowledged all three"}}, // one idle reply, no tool calls
	}}
	s := &Session{
		SessionID: "s1",
		System:    "sys",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 100 },
	}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply != "acknowledged all three" {
		t.Fatalf("reply=%q", res.Reply)
	}
	// Exactly ONE chat round-trip consumed all three queued messages.
	if len(runner.seen) != 1 {
		t.Fatalf("expected 1 chat call (batched), got %d", len(runner.seen))
	}
	first := runner.seen[0]
	var got []string
	for _, m := range first {
		got = append(got, m.Content)
	}
	joined := strings.Join(got, "|")
	for _, want := range []string{"first", "second", "third"} {
		if !strings.Contains(joined, want) {
			t.Errorf("batched turn missing %q; saw %s", want, joined)
		}
	}
}

// ── preparer: revalidate/clear stale notices before send ────────────

func TestPreparer_RunsBeforeBuild(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "n1", Kind: KindNotification, Tag: "build", Content: "stale bad build", CreatedAt: 1})
	store.queue(Entry{ID: "m1", Kind: KindUser, Content: "hi", CreatedAt: 2})

	runner := &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "ok"}}}}
	called := false
	s := &Session{
		SessionID: "s1",
		Store:     store,
		Runner:    runner,
		Now:       func() int64 { return 9 },
		Preparer: PreparerFunc(func(_ context.Context, sid string) error {
			called = true
			if sid != "s1" {
				t.Errorf("preparer got sessionID %q; want s1", sid)
			}
			// Revalidation decided n1 is stale — drop it before send.
			kept := store.entries[:0:0]
			for _, e := range store.entries {
				if e.ID != "n1" {
					kept = append(kept, e)
				}
			}
			store.entries = kept
			return nil
		}),
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("preparer never ran")
	}
	if len(runner.seen) != 1 {
		t.Fatalf("want 1 chat call, got %d", len(runner.seen))
	}
	for _, m := range runner.seen[0] {
		if strings.Contains(m.Content, "stale bad build") {
			t.Error("stale notice reached the model despite preparer clearing it")
		}
	}
}

// ── injection: Inject shows up in the next build ────────────────────

func TestInject_VisibleNextContext(t *testing.T) {
	store := &memStore{}
	s := &Session{SessionID: "s1", Store: store, Now: func() int64 { return 7 }}
	if err := s.Inject(context.Background(), Entry{Kind: KindNotification, Tag: "nudge", Content: "resolve your task"}); err != nil {
		t.Fatal(err)
	}
	msgs, err := DefaultContextBuilder(context.Background(), store, "s1", "")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || !strings.Contains(msgs[0].Content, "[nudge] resolve your task") {
		t.Fatalf("injected entry not rendered: %+v", msgs)
	}
}
