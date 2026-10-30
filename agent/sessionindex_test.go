package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

type memIndex struct {
	docs   []SessionDoc
	putErr error
	closed int
}

func (m *memIndex) Put(_ context.Context, d SessionDoc) error {
	if m.putErr != nil {
		return m.putErr
	}
	m.docs = append(m.docs, d)
	return nil
}

func (m *memIndex) Search(_ context.Context, q string, n int) ([]SessionHit, error) {
	var out []SessionHit
	for _, d := range m.docs {
		if strings.Contains(d.Body, q) && len(out) < n {
			out = append(out, SessionHit{Key: d.Key, Name: d.Name, Snippet: q})
		}
	}
	return out, nil
}

func (m *memIndex) Close(context.Context) error { m.closed++; return nil }

func foldStore() *memStore {
	s := &memStore{}
	for i, k := range []EntryKind{KindUser, KindToolCall, KindToolResult, KindAssistant, KindUser} {
		s.entries = append(s.entries, Entry{ID: fmt.Sprintf("e%d", i), Kind: k, ToolCallID: "c1",
			Content: fmt.Sprintf("event-%d ", i) + strings.Repeat("body ", 200), CreatedAt: int64(i + 1)})
	}
	return s
}

// A fold deletes what it summarizes; the index keeps it, described by the
// summary that replaced it.
func TestFold_PutsTheSubsumedEntries(t *testing.T) {
	idx := &memIndex{}
	sh := &Shaper{Store: foldStore(), Index: idx,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "THE SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1}}
	info, ok, err := sh.Compact(context.Background(), "s")
	if err != nil || !ok || info.IndexErr != nil {
		t.Fatalf("Compact: ok=%v err=%v indexErr=%v", ok, err, info.IndexErr)
	}
	if len(idx.docs) != 1 {
		t.Fatalf("%d documents put", len(idx.docs))
	}
	d := idx.docs[0]
	if d.Kind != SessionDocFold || !strings.HasPrefix(d.Key, "fold-") || d.Summary != "THE SUMMARY" {
		t.Errorf("doc %q %q %q", d.Kind, d.Key, d.Summary)
	}
	for _, want := range []string{"event-0", "event-3", "[tool_result]"} {
		if !strings.Contains(d.Body, want) {
			t.Errorf("body lacks %q", want)
		}
	}
	if strings.Contains(d.Body, "event-4") {
		t.Error("the pristine tail was put")
	}
	if !strings.HasPrefix(d.Name, "event-0") {
		t.Errorf("name %q, want the span's first request", d.Name)
	}
}

// An index that fails does not undo the fold: the summary is already in the
// transcript.
func TestFold_IndexErrorIsReportedNotFatal(t *testing.T) {
	idx := &memIndex{putErr: errors.New("daemon down")}
	s := foldStore()
	sh := &Shaper{Store: s, Index: idx,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "S"}}}},
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1}}
	info, ok, err := sh.Compact(context.Background(), "s")
	if err != nil || !ok {
		t.Fatalf("Compact: ok=%v err=%v", ok, err)
	}
	if info.IndexErr == nil || !strings.Contains(info.IndexErr.Error(), "daemon down") {
		t.Errorf("IndexErr %v", info.IndexErr)
	}
	if len(s.entries) >= 5 {
		t.Errorf("the fold did not apply: %d entries", len(s.entries))
	}
}

func TestSessionSearchTool(t *testing.T) {
	idx := &memIndex{}
	if err := PutSpan(context.Background(), idx, SessionDocRecap, "recap-1", "", "sum",
		[]Entry{{Kind: KindUser, Content: "find the port"}, {Kind: KindToolResult, Content: "listening on 7420"}}); err != nil {
		t.Fatal(err)
	}
	td, run := SessionSearchTool(idx)
	if td.Function.Name != SessionSearchName {
		t.Errorf("name %q", td.Function.Name)
	}
	if out := run(context.Background(), `{"query":"7420"}`); !strings.Contains(out, "## recap-1 — find the port") {
		t.Errorf("hit %q", out)
	}
	if out := run(context.Background(), `{"query":"nothing"}`); !strings.Contains(out, "No earlier history") {
		t.Errorf("miss %q", out)
	}
	if out := run(context.Background(), `{}`); !strings.HasPrefix(out, "ERROR:") {
		t.Errorf("no query %q", out)
	}
	if err := PutSpan(context.Background(), nil, SessionDocFold, "k", "", "", []Entry{{Content: "x"}}); err != nil {
		t.Errorf("no index is not an error: %v", err)
	}
}
