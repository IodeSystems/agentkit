package agent

import (
	"strings"
	"testing"

	"github.com/google/uuid"
)

// ── classifier ──────────────────────────────────────────────────────

func TestClassifyPristineCount(t *testing.T) {
	cases := []struct {
		name     string
		kinds    []EntryKind
		maxMsgs  int
		maxTools int
		want     int
	}{
		{"empty", nil, 5, 5, 0},
		{
			"all messages under budget",
			[]EntryKind{KindUser, KindAssistant, KindUser, KindAssistant},
			5, 5, 4,
		},
		{
			"messages over budget — stop at budget edge",
			[]EntryKind{
				KindUser, KindAssistant,
				KindUser, KindAssistant,
				KindUser, KindAssistant,
			},
			3, 0, 3,
		},
		{
			"mixed messages + tools — both budgets respected",
			[]EntryKind{KindToolCall, KindAssistant, KindToolResult, KindUser},
			2, 1, 3,
		},
		{
			"compaction markers always pass through",
			[]EntryKind{KindCompaction, KindUser},
			1, 0, 2,
		},
		{
			"return short-circuits on first over-budget walking back",
			[]EntryKind{KindNotification, KindUser, KindAssistant},
			1, 0, 1,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			entries := make([]Entry, len(c.kinds))
			for i, k := range c.kinds {
				entries[i] = Entry{ID: uuid.New().String(), Kind: k}
			}
			got := classifyPristineCount(entries, c.maxMsgs, c.maxTools)
			if got != c.want {
				t.Errorf("classifyPristineCount(%v, msgs=%d, tools=%d) = %d, want %d",
					c.kinds, c.maxMsgs, c.maxTools, got, c.want)
			}
		})
	}
}

// ── LOD stub format ─────────────────────────────────────────────────

// TestLodStub_BackPointsToEvent — the stub must embed the source entry id so
// a future read-truncated-field tool call can recover the original payload.
// This is the "toolcall lifting" mechanism.
func TestLodStub_BackPointsToEvent(t *testing.T) {
	e := Entry{
		ID:      "abc-123",
		Kind:    KindToolResult,
		Content: strings.Repeat("x", 10_000),
	}
	stub := lodStub(e, 4000)

	if !strings.Contains(stub, "event_id=abc-123") {
		t.Errorf("stub missing event_id back-pointer: %q", stub)
	}
	if !strings.Contains(stub, "len=10000") {
		t.Errorf("stub missing original length: %q", stub)
	}
	if len(stub) >= len(e.Content) {
		t.Errorf("stub did not shrink content: %d >= %d", len(stub), len(e.Content))
	}
	if !strings.Contains(stub, "…") {
		t.Errorf("stub missing ellipsis sentinel: %q", stub)
	}
}
