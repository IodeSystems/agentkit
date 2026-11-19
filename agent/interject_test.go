package agent

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// roles lists message roles, with "assistant+calls" for a tool-calling turn.
func roles(msgs []llm.Message) []string {
	var out []string
	for _, m := range msgs {
		r := m.Role
		if len(m.ToolCalls) > 0 {
			r += "+calls"
		}
		out = append(out, r)
	}
	return out
}

func renderEntries(entries []Entry, ro RenderOptions) []llm.Message {
	sortEntries(entries)
	return groupMessages("", entries, func(_ int, e Entry) string { return e.Content }, ro)
}

func callEnt(id string, at int64) Entry {
	return Entry{ID: "call-" + id, Kind: KindToolCall, ToolCallID: id, ToolName: "t", Content: "{}", CreatedAt: at}
}
func resultEnt(id, out string, at int64) Entry {
	return Entry{ID: "res-" + id, Kind: KindToolResult, ToolCallID: id, ToolName: "t", Content: out, CreatedAt: at}
}

// Words typed while a tool ran are older than its result. Sorted by time alone
// they land between assistant(tool_calls) and tool, which strict endpoints refuse.
func TestRender_UserTypedMidToolDoesNotSplitCallFromResult(t *testing.T) {
	msgs := renderEntries([]Entry{
		{ID: "u0", Kind: KindUser, Content: "go", CreatedAt: 1},
		callEnt("c1", 2),
		{ID: "u1", Kind: KindUser, Content: "also b", CreatedAt: 3},
		resultEnt("c1", "out", 4),
	}, RenderOptions{})
	want := []string{"user", "assistant+calls", "tool", "user"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
	if msgs[3].Content != "also b" {
		t.Fatalf("user content %q", msgs[3].Content)
	}
}

// A user message must wait for the LAST open result, not the first: between two
// results of one parallel batch it would split the batch.
func TestRender_UserBetweenParallelResultsWaitsForTheLastResult(t *testing.T) {
	msgs := renderEntries([]Entry{
		callEnt("c1", 1), callEnt("c2", 2),
		resultEnt("c1", "o1", 3),
		{ID: "u", Kind: KindUser, Content: "hey", CreatedAt: 4},
		{ID: "n", Kind: KindNotification, Tag: "tag", Content: "note", CreatedAt: 5},
		resultEnt("c2", "o2", 6),
	}, RenderOptions{})
	want := []string{"assistant+calls", "tool", "tool", "user", "user"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
	if msgs[3].Content != "hey" || !strings.HasPrefix(msgs[4].Content, "[tag]") {
		t.Fatalf("held entries out of order: %+v", msgs[3:])
	}
}

// A call that never got a result must not swallow a later user message: it is
// released at the next assistant entry.
func TestRender_HeldEntryIsReleasedByTheNextAssistant(t *testing.T) {
	msgs := renderEntries([]Entry{
		callEnt("c1", 1),
		{ID: "u", Kind: KindUser, Content: "hey", CreatedAt: 2},
		{ID: "a", Kind: KindAssistant, Content: "ok", CreatedAt: 3},
	}, RenderOptions{})
	want := []string{"assistant+calls", "user", "assistant"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
}

// The Shaper's LOD keys on the entry's index; a held entry keeps its own.
func TestRender_HeldEntryKeepsItsOriginalIndexForContentOf(t *testing.T) {
	entries := []Entry{
		callEnt("c1", 1),
		{ID: "u", Kind: KindUser, Content: "hey", CreatedAt: 2},
		resultEnt("c1", "out", 3),
	}
	sortEntries(entries)
	idx := map[string]int{}
	groupMessages("", entries, func(i int, e Entry) string { idx[e.ID] = i; return e.Content }, RenderOptions{})
	if idx["u"] != 1 || idx["res-c1"] != 2 {
		t.Fatalf("indices %v", idx)
	}
}

func TestRender_HeredocHoldsUserUntilTheResult(t *testing.T) {
	msgs := renderEntries([]Entry{
		callEnt("c1", 1),
		{ID: "u", Kind: KindUser, Content: "hey", CreatedAt: 2},
		resultEnt("c1", "out", 3),
	}, RenderOptions{ToolFormat: ToolFormatHeredoc})
	want := []string{"assistant", "tool", "user"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
}

func TestRender_FoldUserIntoToolResultMergesTextOnly(t *testing.T) {
	msgs := renderEntries([]Entry{
		callEnt("c1", 1),
		{ID: "u", Kind: KindUser, Content: "also b", CreatedAt: 2},
		resultEnt("c1", "out", 3),
	}, RenderOptions{FoldUserIntoToolResult: true})
	want := []string{"assistant+calls", "tool"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
	if got, w := msgs[1].Content, "out\n\n[the user wrote, while this ran] also b"; got != w {
		t.Fatalf("content %q, want %q", got, w)
	}
}

// A tool result is text only: folding an attachment would drop it.
func TestRender_FoldKeepsAUserEntryWithPartsAsAUserMessage(t *testing.T) {
	parts := []llm.ContentPart{llm.TextPart("look")}
	msgs := renderEntries([]Entry{
		callEnt("c1", 1),
		{ID: "u", Kind: KindUser, Content: "look", Parts: parts, CreatedAt: 2},
		resultEnt("c1", "out", 3),
	}, RenderOptions{FoldUserIntoToolResult: true})
	want := []string{"assistant+calls", "tool", "user"}
	if got := roles(msgs); !reflect.DeepEqual(got, want) {
		t.Fatalf("roles %v, want %v", got, want)
	}
	if len(msgs[2].Parts) != 1 || strings.Contains(msgs[1].Content, "the user wrote") {
		t.Fatalf("parts lost or folded: %+v", msgs)
	}
}

// End to end: a host saves a pending user entry while a tool runs. The next
// request carries it after the result and the Turn goes on to a reply.
func TestTurn_UserTypedDuringToolLandsAfterTheResultAndTurnContinues(t *testing.T) {
	store := &memStore{}
	store.queue(Entry{ID: "u0", Kind: KindUser, Content: "go", CreatedAt: 1})
	tc := &llm.ToolCall{ID: "c1", Type: "function"}
	tc.Function.Name = "t"
	tc.Function.Arguments = `{}`
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{ToolCall: tc}},
		{{Content: "done"}},
	}}
	clock := int64(100)
	now := func() int64 { clock++; return clock }
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, Now: now,
		Dispatch: func(context.Context, llm.ToolCall) (string, error) {
			store.queue(Entry{ID: "u1", Kind: KindUser, Content: "also b", CreatedAt: now()})
			return "out", nil
		},
	}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(runner.seen) != 2 {
		t.Fatalf("rounds %d, want 2", len(runner.seen))
	}
	r1, r2 := runner.seen[0], runner.seen[1]
	want := []string{"user", "assistant+calls", "tool", "user"}
	if got := roles(r2); !reflect.DeepEqual(got, want) {
		t.Fatalf("round 2 roles %v, want %v", got, want)
	}
	// Cache: round 1 is a prefix of round 2.
	if !reflect.DeepEqual(r1, r2[:len(r1)]) {
		t.Fatalf("round 1 is not a prefix of round 2:\n%+v\n%+v", r1, r2)
	}
}
