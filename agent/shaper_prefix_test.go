package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// ── F6: the single-stage fold reuses the session's provider cache prefix ──
//
// The fold used to send {worker-system, numbered-text-blob} with nil tools,
// so the whole session history was re-read uncached on every fold. With
// SessionPrefix set, a fold that fits in one request instead sends the
// session's own system prompt and tool definitions, the session history
// rendered as messages, and the summary instruction as the final user
// message, with tool_choice none — so the provider cache prefix stays warm
// and the round's uncached input is the summary instruction only. A staged
// fold (history larger than the window) keeps the worker's own prompt.

// prefixTools is a non-trivial tool set whose byte identity can be checked.
func prefixTools() []llm.ToolDef {
	a := llm.ToolDef{Type: "function"}
	a.Function.Name = "read_file"
	a.Function.Description = "Read a file"
	a.Function.Parameters = map[string]any{"type": "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}}}
	b := llm.ToolDef{Type: "function"}
	b.Function.Name = "run_tests"
	b.Function.Description = "Run the test suite"
	b.Function.Parameters = map[string]any{"type": "object",
		"properties": map[string]any{"filter": map[string]any{"type": "string"}}}
	return []llm.ToolDef{a, b}
}

// smallHistory is a foldable backlog of ~10k estimated tokens.
func smallHistory() *memStore {
	st := &memStore{}
	for i := 0; i < 10; i++ {
		st.entries = append(st.entries, Entry{ID: fmt.Sprintf("e%d", i), Kind: KindUser,
			Content: fmt.Sprintf("event %d %s", i, strings.Repeat("x", 3000)), CreatedAt: int64(i + 1)})
	}
	// pristine tail
	st.entries = append(st.entries, Entry{ID: "tail", Kind: KindAssistant, Content: "last", CreatedAt: 100})
	return st
}

// recordingMultiRunner records EVERY request it answers, one canned summary
// per call.
type recordingMultiRunner struct {
	turns [][]llm.StreamChunk
	msgs  [][]llm.Message
	tools [][]llm.ToolDef
	opts  []*llm.ChatOpts
}

func (r *recordingMultiRunner) ChatStream(_ context.Context, msgs []llm.Message, tools []llm.ToolDef, o *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	cp := make([]llm.Message, len(msgs))
	copy(cp, msgs)
	r.msgs = append(r.msgs, cp)
	r.tools = append(r.tools, tools)
	r.opts = append(r.opts, o)
	var chunks []llm.StreamChunk
	if len(r.turns) > 0 {
		chunks = r.turns[0]
		r.turns = r.turns[1:]
	}
	ch := make(chan llm.StreamChunk, len(chunks)+1)
	for _, c := range chunks {
		ch <- c
	}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

func toolsByteEqual(a, b []llm.ToolDef) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Type != b[i].Type ||
			a[i].Function.Name != b[i].Function.Name ||
			a[i].Function.Description != b[i].Function.Description {
			return false
		}
		aj, aerr := json.Marshal(a[i].Function.Parameters)
		bj, berr := json.Marshal(b[i].Function.Parameters)
		if aerr != nil || berr != nil || !bytes.Equal(aj, bj) {
			return false
		}
	}
	return true
}

// Scenario: the fold request's system prompt is byte-identical to the
// session's system prompt.
func TestFoldCarriesSessionPrefix(t *testing.T) {
	const sessionSystem = "You are the assistant. Be brief.\nSlot note: session s."
	st := smallHistory()
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: sessionSystem, Tools: prefixTools()},
	}
	if _, did, err := sh.Compact(context.Background(), "the task"); err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	if len(r.msgs) != 1 {
		t.Fatalf("a fold that fits must be one request, got %d", len(r.msgs))
	}
	got := r.msgs[0][0]
	if got.Role != "system" || got.Content != sessionSystem {
		t.Errorf("system prompt = %q, want byte-identical %q", got.Content, sessionSystem)
	}
}

// Scenario: the fold request's tool definitions are byte-identical to the
// session's tool definitions.
func TestFoldToolDefinitionsByteIdentical(t *testing.T) {
	tools := prefixTools()
	st := smallHistory()
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "sys", Tools: tools},
	}
	if _, did, err := sh.Compact(context.Background(), "t"); err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	if len(r.tools) != 1 {
		t.Fatalf("no tools recorded on the fold request")
	}
	if !toolsByteEqual(r.tools[0], tools) {
		t.Errorf("fold request's tool definitions are not byte-identical to the session's")
	}
}

// Scenario: the fold request's messages are the session history with the
// summary instruction as the final user message, sent with tool_choice none.
func TestFoldRequestShape(t *testing.T) {
	const sessionSystem = "session system"
	st := smallHistory()
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: sessionSystem, Tools: prefixTools()},
	}
	if _, did, err := sh.Compact(context.Background(), "the task"); err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	msgs := r.msgs[0]
	if len(msgs) < 3 {
		t.Fatalf("want system + history + instruction, got %d message(s)", len(msgs))
	}
	// First message: the session's system prompt.
	if msgs[0].Role != "system" || msgs[0].Content != sessionSystem {
		t.Errorf("first message = (%s, %q), want the session's system prompt", msgs[0].Role, msgs[0].Content)
	}
	// The rendered history carries the foldable events, one message per entry.
	body := ""
	for _, m := range msgs[1:] {
		body += m.Content
	}
	for i := 0; i < 10; i++ {
		if !strings.Contains(body, fmt.Sprintf("event %d", i)) {
			t.Errorf("the fold request is missing foldable event %d", i)
		}
	}
	// The pristine tail must not be folded.
	for _, m := range msgs {
		if m.Content == "last" {
			t.Error("the pristine tail was folded; it must stay out of the request")
		}
	}
	// Final message: the summary instruction, as a user message.
	last := msgs[len(msgs)-1]
	if last.Role != "user" {
		t.Errorf("final message role = %q, want user", last.Role)
	}
	if !strings.Contains(last.Content, "Summarize the following events") {
		t.Errorf("final message is not the summary instruction: %q", last.Content[:min(len(last.Content), 80)])
	}
	// tool_choice none.
	if r.opts[0] == nil || r.opts[0].ToolChoice != "none" {
		tc := ""
		if r.opts[0] != nil {
			tc = r.opts[0].ToolChoice
		}
		t.Errorf("tool_choice = %q, want \"none\"", tc)
	}
}

// Scenario: a fold too large for one request sends its FIRST request as the
// session's prefix (system, tools, key, tool_choice none), covering the head
// of the history that fits; every later stage uses the worker prompt with no
// tools and no key, carrying the running summary. A fold is triggered near the
// budget, so this split is the common case: in v0.33.1 the split fold sent no
// prefix at all and most real folds read the history uncached (dun-bf).
func TestSplitFoldFirstRequestCarriesPrefixRestStage(t *testing.T) {
	st := smallHistory()
	// 5k window against ~10k of events: the fold must split.
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{
		{{Content: "S0"}}, {{Content: "S1"}}, {{Content: "S2"}}, {{Content: "S3"}},
		{{Content: "S4"}}, {{Content: "S5"}}, {{Content: "S6"}}, {{Content: "S7"}},
		{{Content: "S8"}}, {{Content: "S9"}}, {{Content: "S10"}},
	}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 5000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "session system", Tools: prefixTools()},
		Estimate:      Default(),
	}
	summary, err := sh.summarize(context.Background(), st.entries[:10], "t", 100, 5000, "k")
	if err != nil {
		t.Fatal(err)
	}
	if summary == "" {
		t.Fatal("empty summary")
	}
	if len(r.msgs) < 2 {
		t.Fatalf("a fold that exceeds the window must split, got %d request(s)", len(r.msgs))
	}
	first := r.msgs[0]
	if first[0].Content != "session system" || !toolsByteEqual(r.tools[0], prefixTools()) ||
		r.opts[0].ToolChoice != "none" || r.opts[0].PromptCacheKey != "k" {
		t.Fatalf("first request is not the session prefix: system=%q tools=%d opts=%+v",
			first[0].Content, len(r.tools[0]), r.opts[0])
	}
	if first[1].Content != st.entries[0].Content {
		t.Errorf("first request must start at the first event, got %q", first[1].Content)
	}
	for i := 1; i < len(r.msgs); i++ {
		call := r.msgs[i]
		if call[0].Role != "system" || call[0].Content != DefaultSummarySystem {
			t.Errorf("stage %d system = %q, want the worker prompt", i, call[0].Content)
		}
		if len(r.tools[i]) != 0 || r.opts[i].ToolChoice != "" || r.opts[i].PromptCacheKey != "" {
			t.Errorf("stage %d carries tools=%d opts=%+v; a later stage is not the session prefix", i, len(r.tools[i]), r.opts[i])
		}
	}
	if !strings.Contains(r.msgs[1][1].Content, "S0") {
		t.Errorf("stage 1 must carry the prefix request's summary, got %q", r.msgs[1][1].Content)
	}
}

// Scenario: fold round reads from cache.
//
// A warm session folds over its own history. The fold request carries the
// session's prefix (part 1), so the provider reports the session prompt as a
// cache read; its usage must land in the session's ledger (s.usageCached)
// with the cached count at 90% or more of the session's prompt size.
func TestFoldRoundUsageSurfacesIntoSessionLedger(t *testing.T) {
	const sessionPrompt = 120_000 // the warm prefix the fold request reuses

	st := smallHistory() // ~10k estimated tokens, over the budget below
	shaper := &Shaper{
		Store: st,
		Policy: ShaperPolicy{BudgetTokens: 10_000, PreserveLastMessages: 1},
		Estimate:      Default(),
		SessionPrefix: SessionPrefix{System: "session system", Tools: prefixTools()},
	}
	// Warm mock: the prefix fold request reports the whole session prompt as
	// a cache read; the second fold stage and the final round report nothing.
	r := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "SUMMARY"},
			{Usage: &llm.Usage{PromptTokens: sessionPrompt + 200,
				PromptTokensDetails: &llm.PromptTokensDetails{CachedTokens: sessionPrompt},
				CompletionTokens: 800}}},
		{{Content: "STAGE2"}},
		{{Content: "done"}},
	}}
	shaper.Runner = r
	s := &Session{
		SessionID: "s",
		Store:     st,
		Runner:    r,
		MaxTurns:  5,
		Build:     shaper.Build,
	}
	shaper.OnFoldUsage = s.AddFoldUsage

	// The turn's prompt arrives in the inbox; Build sees it and the backlog
	// is over budget, so the round forces a fold first.
	st.queue(Entry{ID: "ask", Kind: KindUser, Content: "keep going", CreatedAt: 200})
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("turn: %v", err)
	}

	// The fold round's usage reached the session ledger: the cached count is
	// the session prompt size, and the ledger says so.
	if got := res.Usage.Cached; got < sessionPrompt {
		t.Fatalf("ledger cached = %d, want >= %d (the fold round's cache read must land in the ledger)",
			got, sessionPrompt)
	}
	ratio := float64(res.Usage.Cached) / float64(sessionPrompt)
	if ratio < 0.90 {
		t.Errorf("fold round's cached count is %.1f%% of the session prompt, want >=90%%", ratio*100)
	}
	// The fold's generated tokens are in the ledger too, not only the cache.
	if res.Usage.Generated < 800 {
		t.Errorf("ledger generated = %d, want >= 800 (the fold's summary)", res.Usage.Generated)
	}
	// The fold request was the warm one: prefix system, tools, choice none.
	fold := r.seen[0]
	if fold[0].Content != "session system" {
		t.Errorf("fold request system = %q, want the session's", fold[0].Content)
	}
}

// Scenario: the split never ends the prefix request on a tool call whose
// result it leaves out: a strict endpoint refuses a call with no result.
func TestPrefixCutKeepsToolExchangesWhole(t *testing.T) {
	sh := &Shaper{Estimate: Default(), SessionPrefix: SessionPrefix{System: "s"}}
	big := strings.Repeat("x", 4000)
	entries := []Entry{
		{Kind: KindUser, Content: "u"},
		{Kind: KindToolCall, ToolCallID: "c1", Content: "{}"},
		{Kind: KindToolResult, ToolCallID: "c1", Content: big},
		{Kind: KindAssistant, Content: "a"},
	}
	lines := make([]string, len(entries))
	for i, e := range entries {
		lines[i] = e.Content
	}
	// Room for the user and the call, not for the result.
	window := sh.Estimate.Estimate("s") + sh.Estimate.Estimate("h") + SummaryCap(100) +
		sh.Estimate.Estimate("u") + sh.Estimate.Estimate("{}") + 10
	if cut := sh.prefixCut(entries, lines, "h", 100, window); cut != 1 {
		t.Fatalf("cut = %d, want 1 (before the call, not between call and result)", cut)
	}
}

// Scenario: the prefix request's size check counts the SESSION's fixed part,
// tools included. Counting the worker prompt instead let a request pass the
// check and be refused as too large, a wasted round.
func TestPrefixCutCountsSessionTools(t *testing.T) {
	tools := prefixTools()
	tools[0].Function.Description = strings.Repeat("d", 8000)
	with := &Shaper{Estimate: Default(), SessionPrefix: SessionPrefix{System: "s", Tools: tools}}
	without := &Shaper{Estimate: Default(), SessionPrefix: SessionPrefix{System: "s"}}
	var entries []Entry
	var lines []string
	for i := 0; i < 20; i++ {
		c := strings.Repeat("y", 1000)
		entries = append(entries, Entry{Kind: KindUser, Content: c})
		lines = append(lines, c)
	}
	a := with.prefixCut(entries, lines, "h", 100, 6000)
	b := without.prefixCut(entries, lines, "h", 100, 6000)
	if a >= b {
		t.Fatalf("cut with big tools = %d, without = %d; tools must take room", a, b)
	}
}

// Scenario: a host's runner can tell a fold request from a session round by
// the ctx, not by matching the system prompt (which a prefix fold shares with
// the session).
func TestIsFoldMarksEveryFoldRequest(t *testing.T) {
	st := smallHistory()
	var marks []bool
	r := &ctxRecordingRunner{inner: &recordingMultiRunner{turns: [][]llm.StreamChunk{
		{{Content: "S0"}}, {{Content: "S1"}}, {{Content: "S2"}}, {{Content: "S3"}},
		{{Content: "S4"}}, {{Content: "S5"}}, {{Content: "S6"}}, {{Content: "S7"}},
	}}, marks: &marks}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 5000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "session system", Tools: prefixTools()},
		Estimate:      Default(),
	}
	if _, err := sh.summarize(context.Background(), st.entries[:10], "t", 100, 5000, "k"); err != nil {
		t.Fatal(err)
	}
	if len(marks) < 2 {
		t.Fatalf("want a split fold, got %d request(s)", len(marks))
	}
	for i, m := range marks {
		if !m {
			t.Errorf("request %d: IsFold = false", i)
		}
	}
	if IsFold(context.Background()) {
		t.Error("IsFold true on a plain ctx")
	}
}

type ctxRecordingRunner struct {
	inner *recordingMultiRunner
	marks *[]bool
}

func (r *ctxRecordingRunner) ChatStream(ctx context.Context, m []llm.Message, td []llm.ToolDef, o *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	*r.marks = append(*r.marks, IsFold(ctx))
	return r.inner.ChatStream(ctx, m, td, o)
}

// On an InputTooLargeError the prefix path falls back to the staged
// worker-prompt loop and the fold still gets through.
func TestFoldInputTooLargeFallsBackToStaged(t *testing.T) {
	st := smallHistory()
	inner := &recordingMultiRunner{turns: [][]llm.StreamChunk{
		{{Content: "S0"}}, {{Content: "S1"}}, {{Content: "S2"}},
		{{Content: "S3"}}, {{Content: "S4"}}, {{Content: "S5"}},
	}}
	r := &refusePrefix{recordingMultiRunner: inner}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "session system", Tools: prefixTools()},
	}
	if _, did, err := sh.Compact(context.Background(), "t"); err != nil || !did {
		t.Fatalf("fold after fallback: did=%v err=%v", did, err)
	}
	if len(r.msgs) < 2 {
		t.Fatalf("want a prefix attempt + a staged call, got %d", len(r.msgs))
	}
	// The staged call must open with the worker prompt and carry no tools.
	if r.msgs[1][0].Content != DefaultSummarySystem {
		t.Errorf("fallback staged system = %q, want %q", r.msgs[1][0].Content, DefaultSummarySystem)
	}
	if len(r.tools[1]) != 0 {
		t.Errorf("fallback staged call carries %d tool(s), want none", len(r.tools[1]))
	}
}

// refusePrefix records like recordingMultiRunner and refuses requests that
// carry tools — i.e. the prefix path — with a typed InputTooLargeError.
type refusePrefix struct {
	*recordingMultiRunner
}

func (r *refusePrefix) ChatStream(ctx context.Context, msgs []llm.Message, tools []llm.ToolDef, o *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	if len(tools) > 0 {
		cp := make([]llm.Message, len(msgs))
		copy(cp, msgs)
		r.msgs = append(r.msgs, cp)
		r.tools = append(r.tools, tools)
		r.opts = append(r.opts, o)
		return nil, &llm.InputTooLargeError{Status: 400, PromptTokens: 200000, ContextTokens: 128000}
	}
	return r.recordingMultiRunner.ChatStream(ctx, msgs, tools, o)
}

// Scenario: the prefix fold must send the key the session's own rounds send
// (session id + newest marker before this fold). The key routes the request on
// OpenAI-shaped endpoints and corrallm; without it the warm prefix can sit on a
// slot the fold never reaches, and SessionPrefix buys nothing. v0.33.0 sent no
// key.
func TestPrefixFoldSendsTheSessionsCacheKey(t *testing.T) {
	st := smallHistory()
	// An earlier fold's marker: the session's current key names it.
	st.entries = append([]Entry{{ID: "m-old", Kind: KindCompaction, Content: "older", CreatedAt: 0}}, st.entries...)
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "sys", Tools: prefixTools()},
	}
	if _, did, err := sh.Compact(context.Background(), "s1"); err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	want := promptCacheKey("s1", "m-old")
	if len(r.opts) != 1 || r.opts[0] == nil || r.opts[0].PromptCacheKey != want {
		t.Fatalf("fold key = %+v, want %q (the session's current key)", r.opts, want)
	}
}

// Scenario: a host that sets its own key in SummaryOpts keeps it, as a host
// key in Session.ChatOpts is kept on the session's rounds.
func TestPrefixFoldKeepsHostSummaryKey(t *testing.T) {
	st := smallHistory()
	r := &recordingMultiRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{
		Store: st, Runner: r,
		Policy:        ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1},
		SessionPrefix: SessionPrefix{System: "sys", Tools: prefixTools()},
		SummaryOpts:   &llm.ChatOpts{PromptCacheKey: "host-key"},
	}
	if _, did, err := sh.Compact(context.Background(), "s1"); err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	if got := r.opts[0].PromptCacheKey; got != "host-key" {
		t.Fatalf("fold key = %q, want the host's", got)
	}
}
