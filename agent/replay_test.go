package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// A two-turn history: an old turn with a reasoned reply, then the current
// tool loop: a reasoned call round and a reasoned reply.
func replayEntries() []Entry {
	op := func(s string) json.RawMessage {
		return json.RawMessage(`[{"type":"reasoning.text","text":"` + s + `","signature":"S","index":0}]`)
	}
	return []Entry{
		{ID: "u1", Kind: KindUser, Content: "first", CreatedAt: 1},
		{ID: "a1", Kind: KindAssistant, Content: "old answer", Reasoning: "old thought", ReasoningOpaque: op("old thought"), CreatedAt: 2},
		{ID: "u2", Kind: KindUser, Content: "second", CreatedAt: 3},
		{ID: "c1", Kind: KindToolCall, ToolCallID: "t1", ToolName: "f", Content: "{}", Reasoning: "call f", ReasoningOpaque: op("call f"), CreatedAt: 4},
		{ID: "r1", Kind: KindToolResult, ToolCallID: "t1", ToolName: "f", Content: "ok", CreatedAt: 5},
		{ID: "a2", Kind: KindAssistant, Content: "new answer", Reasoning: "wrap up", CreatedAt: 6},
	}
}

func replayed(msgs []llm.Message) []string {
	var out []string
	for _, m := range msgs {
		if m.Reasoning != "" || len(m.ReasoningOpaque) > 0 {
			out = append(out, m.Reasoning)
		}
	}
	return out
}

func renderWith(entries []Entry, mode ReasoningReplay) []llm.Message {
	return groupMessages("sys", entries, func(_ int, e Entry) string { return e.Content }, RenderOptions{ReplayReasoning: mode})
}

func TestReplayNoneRendersNoReasoning(t *testing.T) {
	if got := replayed(renderWith(replayEntries(), ReplayNone)); len(got) != 0 {
		t.Errorf("default policy replayed %v", got)
	}
	// And the messages equal those of entries that never had reasoning.
	bare := replayEntries()
	for i := range bare {
		bare[i].Reasoning, bare[i].ReasoningOpaque = "", nil
	}
	if !reflect.DeepEqual(renderWith(replayEntries(), ReplayNone), renderWith(bare, ReplayNone)) {
		t.Error("ReplayNone render differs from a render of entries with no reasoning")
	}
}

func TestReplayTurnSendsOnlyTheCurrentToolLoop(t *testing.T) {
	msgs := renderWith(replayEntries(), ReplayTurn)
	if got := replayed(msgs); !reflect.DeepEqual(got, []string{"call f", "wrap up"}) {
		t.Errorf("turn replayed %v", got)
	}
	// The call-only round's reasoning rides on its assistant(tool_calls) message.
	for _, m := range msgs {
		if len(m.ToolCalls) > 0 && (m.Reasoning != "call f" || len(m.ReasoningOpaque) == 0) {
			t.Errorf("tool-call message = %+v", m)
		}
	}
}

func TestReplayAllSendsEveryAssistant(t *testing.T) {
	if got := replayed(renderWith(replayEntries(), ReplayAll)); !reflect.DeepEqual(got, []string{"old thought", "call f", "wrap up"}) {
		t.Errorf("all replayed %v", got)
	}
}

// Within a turn, round N's messages are a prefix of round N+1's: the cache
// keeps hitting. A new user message is the one documented break.
func TestReplayTurnKeepsThePrefixWithinATurn(t *testing.T) {
	es := replayEntries()
	before := renderWith(es[:5], ReplayTurn)
	after := renderWith(es, ReplayTurn)
	if !reflect.DeepEqual(before, after[:len(before)]) {
		t.Errorf("round N is not a prefix of round N+1 under ReplayTurn")
	}
	next := append(append([]Entry(nil), es...), Entry{ID: "u3", Kind: KindUser, Content: "third", CreatedAt: 7})
	if got := replayed(renderWith(next, ReplayTurn)); len(got) != 0 {
		t.Errorf("a new user message should drop earlier reasoning, replayed %v", got)
	}
	allBefore, allAfter := renderWith(es, ReplayAll), renderWith(next, ReplayAll)
	if !reflect.DeepEqual(allBefore, allAfter[:len(allBefore)]) {
		t.Error("ReplayAll broke the prefix on a new user message")
	}
}

func TestReplayStubbedEntryLosesItsReasoning(t *testing.T) {
	msgs := groupMessages("", replayEntries(), func(_ int, e Entry) string {
		if e.ID == "a2" {
			return "[stub]"
		}
		return e.Content
	}, RenderOptions{ReplayReasoning: ReplayAll})
	for _, m := range msgs {
		if m.Content == "[stub]" && m.Reasoning != "" {
			t.Errorf("a stubbed entry kept its reasoning: %+v", m)
		}
	}
}

func TestReplayShaperPolicyReachesTheRender(t *testing.T) {
	store := &memStore{entries: replayEntries()}
	sh := &Shaper{Store: store, Policy: ShaperPolicy{ReplayReasoning: ReplayTurn}}
	msgs, err := sh.Build(context.Background(), "s", "sys")
	if err != nil {
		t.Fatal(err)
	}
	if got := replayed(msgs); !reflect.DeepEqual(got, []string{"call f", "wrap up"}) {
		t.Errorf("shaper replayed %v", got)
	}
}

func TestReasoningOpaqueIsPersistedWhereReasoningIs(t *testing.T) {
	op := json.RawMessage(`[{"type":"reasoning.text","text":"r","signature":"S","index":0}]`)
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Reasoning: "r"}, {ToolCall: call("1", "look")}, {ToolCall: call("2", "look")}, {Done: true, ReasoningOpaque: op}, {Usage: &llm.Usage{}}},
		{{Reasoning: "q"}, {Content: "answer"}, {Done: true, ReasoningOpaque: json.RawMessage(`[{"x":1}]`)}, {Usage: &llm.Usage{}}},
	}}
	s := &Session{SessionID: "s", Store: store, MaxTurns: 5, Runner: runner,
		Dispatch: func(context.Context, llm.ToolCall) (llm.ToolResult, error) { return llm.TextResult("ok"), nil }}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	var calls, replies []Entry
	for _, e := range store.entries {
		switch e.Kind {
		case KindToolCall:
			calls = append(calls, e)
		case KindAssistant:
			replies = append(replies, e)
		}
	}
	if string(calls[0].ReasoningOpaque) != string(op) || calls[1].ReasoningOpaque != nil {
		t.Errorf("calls = %s / %s", calls[0].ReasoningOpaque, calls[1].ReasoningOpaque)
	}
	if string(replies[0].ReasoningOpaque) != `[{"x":1}]` {
		t.Errorf("reply opaque = %s", replies[0].ReasoningOpaque)
	}
}

func TestEntryReasoningRoundTripsAndOldRecordsLoad(t *testing.T) {
	var old Entry
	if err := json.Unmarshal([]byte(`{"ID":"a","Kind":"assistant","Content":"x","CreatedAt":1}`), &old); err != nil {
		t.Fatal(err)
	}
	if old.Reasoning != "" || old.ReasoningOpaque != nil {
		t.Errorf("old record = %+v", old)
	}
	e := Entry{ID: "a", Kind: KindAssistant, Content: "x", Reasoning: "r",
		ReasoningOpaque: json.RawMessage(`[{"type":"reasoning.text","signature":"S+/=","text":"r"}]`)}
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var back Entry
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Reasoning != "r" || string(back.ReasoningOpaque) != string(e.ReasoningOpaque) {
		t.Errorf("round trip = %+v (%s)", back, back.ReasoningOpaque)
	}
}

func TestRepetitionCutRoundStoresNoReasoning(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Reasoning: "loop loop loop"}, {Content: "partial"}, {Done: true, StopReason: llm.StopReasonRepetition, Repetition: &llm.RepetitionInfo{}}},
		{{Content: "fine"}, {Usage: &llm.Usage{}}},
	}}
	s := &Session{SessionID: "s", Store: store, MaxTurns: 5, Runner: runner}
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, e := range store.entries {
		if e.Kind == KindAssistant && e.Reasoning != "" {
			t.Errorf("a looping round's reasoning was stored for replay: %+v", e)
		}
	}
}

// End to end through the llm client: under ReplayTurn a DeepSeek-shaped
// endpoint gets reasoning_content on the current loop's assistant message,
// and under ReplayNone the body is the same bytes as with no reasoning stored.
func TestReplayReachesTheWireAndNoneIsByteIdentical(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, b)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	send := func(entries []Entry, mode ReasoningReplay) []byte {
		c := llm.NewClient(srv.URL, "k", "deepseek-v4")
		ch, err := c.ChatStream(context.Background(), renderWith(entries, mode), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for range ch {
		}
		mu.Lock()
		defer mu.Unlock()
		return bodies[len(bodies)-1]
	}
	bare := replayEntries()
	for i := range bare {
		bare[i].Reasoning, bare[i].ReasoningOpaque = "", nil
	}
	if a, b := send(replayEntries(), ReplayNone), send(bare, ReplayNone); string(a) != string(b) {
		t.Errorf("ReplayNone body differs:\n%s\n%s", a, b)
	}
	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(send(replayEntries(), ReplayTurn), &body); err != nil {
		t.Fatal(err)
	}
	var got []any
	for _, m := range body.Messages {
		if v, ok := m["reasoning_content"]; ok {
			got = append(got, v)
		}
	}
	if !reflect.DeepEqual(got, []any{"call f", "wrap up"}) {
		t.Errorf("reasoning_content on the wire = %v", got)
	}
}

// The prefix fold renders the folded history as the session's rounds did: the
// turn boundary is the newest user entry of the whole context, not of the
// folded part.
func TestReplayPrefixFoldUsesTheSessionsTurnBoundary(t *testing.T) {
	big := func(s string) string { return s + " " + strings.Repeat("x", 3000) }
	for _, tc := range []struct {
		name    string
		entries []Entry
		want    []string
	}{
		{"newest user in the tail", []Entry{
			{ID: "u1", Kind: KindUser, Content: big("one"), CreatedAt: 1},
			{ID: "a1", Kind: KindAssistant, Content: big("a"), Reasoning: "folded thought", CreatedAt: 2},
			{ID: "u2", Kind: KindUser, Content: "now", CreatedAt: 3},
		}, nil},
		{"newest user folded", []Entry{
			{ID: "u1", Kind: KindUser, Content: big("one"), CreatedAt: 1},
			{ID: "a1", Kind: KindAssistant, Content: big("a"), Reasoning: "loop thought", CreatedAt: 2},
			{ID: "n1", Kind: KindNotification, Content: big("note"), CreatedAt: 3},
			{ID: "a2", Kind: KindAssistant, Content: "last", Reasoning: "tail thought", CreatedAt: 4},
		}, []string{"loop thought"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := &memStore{entries: tc.entries}
			r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
			sh := &Shaper{Store: st, Runner: r,
				Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1, ReplayReasoning: ReplayTurn},
				SessionPrefix: SessionPrefix{System: "sys"}}
			if _, did, err := sh.Compact(context.Background(), "t"); err != nil || !did {
				t.Fatalf("fold: did=%v err=%v", did, err)
			}
			if got := replayed(r.msgs[0]); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("fold prefix replayed %v, want %v", got, tc.want)
			}
		})
	}
}
