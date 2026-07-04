package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/iodesystems/agentkit/agent"
	"github.com/iodesystems/agentkit/llm"
)

// clock is a monotonic counter for deterministic CreatedAt ordering across the
// entries a demo appends and the ones the loop appends (share it with
// Session.Now so everything interleaves correctly).
type clock struct{ n int64 }

func (c *clock) next() int64 { c.n++; return c.n }

// ── chat: streaming + 429 backpressure ─────────────────────────────

func runChat(ctx context.Context, cfg config) error {
	p := cfg.prompt
	if p == "" {
		p = "In one sentence, what is an agent tool-call loop?"
	}
	fmt.Printf("prompt: %s\n\n", p)
	// ChatStream retries 429 INTERNALLY, honoring the server's retry_after
	// (header or corrallm JSON body) until this ctx's deadline. A returned
	// error means the deadline elapsed or a non-retryable failure.
	ch, err := cfg.client().ChatStream(ctx, []llm.Message{{Role: "user", Content: p}}, nil, nil)
	if err != nil {
		return err
	}
	var usage *llm.Usage
	for chunk := range ch {
		if chunk.Error != "" {
			return fmt.Errorf("stream: %s", chunk.Error)
		}
		fmt.Print(chunk.Content)
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	fmt.Println()
	if usage != nil {
		fmt.Printf("\n[tokens: prompt=%d completion=%d total=%d]\n",
			usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
	}
	return nil
}

// ── tools: local Go tool-call loop ─────────────────────────────────

func runTools(ctx context.Context, cfg config) error {
	clk := &clock{}
	store := newDemoStore()
	p := cfg.prompt
	if p == "" {
		p = "What's the weather in Denver in Fahrenheit, and what's 21 plus 21? Use your tools."
	}
	store.publish(entry(agent.KindUser, p, clk.next()))
	fmt.Printf("prompt: %s\n", p)

	sess := &agent.Session{
		SessionID:        "demo",
		System:           "You are a concise assistant. Use the provided tools when they help; then answer in one line.",
		Store:            store,
		Runner:           cfg.client(),
		Tools:            localTools,
		Dispatch:         verbose(dispatchLocal),
		Now:              clk.next,
		MaxTurns:         8,
		Tracer:           cfg.tracer(),
		OnAssistantToken: func(s string) { fmt.Print(s) },
	}
	fmt.Print("\nassistant: ")
	reply, err := sess.Turn(ctx)
	fmt.Println()
	if err != nil {
		return err
	}
	if strings.TrimSpace(reply) == "" {
		fmt.Println("(model ended on tool calls with no final text)")
	}
	return nil
}

// ── schema: client-side validation + fix loop ──────────────────────

func runSchema(ctx context.Context, cfg config) error {
	tools := []llm.ToolDef{
		toolDef("create_task", "Create a task.", obj([]string{"title", "priority"}, map[string]any{
			"title":    prop("string", "short task title"),
			"priority": prop("string", "one of low|medium|high"),
		})),
	}
	v := agent.NewSchemaValidator(tools)

	// The real dispatcher never runs on an invalid call — the wrapper short-
	// circuits with a fix instruction, keeping the session alive so the model
	// retries. This is the same rejection pattern hosts hand-roll per tool,
	// made generic from the tool's own JSON Schema.
	real := agent.ToolDispatcher(func(_ context.Context, tc llm.ToolCall) (string, error) {
		return "OK: task created (" + tc.Function.Arguments + ")", nil
	})
	disp := agent.ValidatingDispatcher(real, v)

	fmt.Println("The tool requires {title:string, priority:string}. Watch the gate:")
	cases := []string{
		`{"title": 123, "priority": "high"}`,          // wrong type for title
		`{"title": "write docs"}`,                     // missing priority
		`{"title": "write docs", "priority": "high"}`, // valid
	}
	for _, args := range cases {
		call := llm.ToolCall{ID: "c1", Type: "function"}
		call.Function.Name = "create_task"
		call.Function.Arguments = args
		res, _ := disp(ctx, call)
		fmt.Printf("\n  args:   %s\n  result: %s\n", args, oneLine(res))
	}
	fmt.Println("\nIn a live loop the fix instruction becomes the tool result, the model")
	fmt.Println("corrects the call, and the turn continues — no host code per tool.")
	return nil
}

// ── structured: reliable typed extraction ──────────────────────────
//
// Combines four features into the "get typed data out of an LLM, reliably"
// story: tool_choice forces the model to CALL the tool, the SchemaValidator
// fix loop gates the ARGUMENTS (retrying on a bad one), ForcedTerminalTool
// makes it the session's only exit, and ErrSessionClosed ends the loop the
// instant valid data arrives.
func runStructured(ctx context.Context, cfg config) error {
	tools := []llm.ToolDef{
		toolDef("record_person", "Record the person extracted from the text.",
			obj([]string{"name", "age", "email"}, map[string]any{
				"name":  prop("string", "full name"),
				"age":   prop("integer", "age in years"),
				"email": prop("string", "email address"),
			})),
	}

	var captured string
	// Terminal dispatcher: capture the (already-validated) args and close.
	terminal := agent.ToolDispatcher(func(_ context.Context, tc llm.ToolCall) (string, error) {
		captured = tc.Function.Arguments
		return "recorded", agent.ErrSessionClosed
	})
	disp := agent.ValidatingDispatcher(terminal, agent.NewSchemaValidator(tools))

	clk := &clock{}
	store := newDemoStore()
	text := cfg.prompt
	if text == "" {
		text = "Jane Doe is 34 years old; reach her at jane@example.com."
	}
	store.publish(entry(agent.KindUser, "Extract the person from this text: "+text, clk.next()))
	fmt.Printf("input: %s\n\n", text)

	sess := &agent.Session{
		SessionID: "demo",
		System:    "You extract structured data. Call record_person with the fields from the text.",
		Store:     store,
		Runner:    cfg.client(),
		Tools:     tools,
		Dispatch:  verbose(disp),
		// Force the model to call exactly this tool — its only legitimate exit.
		ChatOpts:           &llm.ChatOpts{ToolChoice: `{"type":"function","function":{"name":"record_person"}}`},
		ForcedTerminalTool: "record_person",
		Now:                clk.next,
		MaxTurns:           5,
		Tracer:             cfg.tracer(),
	}
	if _, err := sess.Turn(ctx); err != nil {
		return err
	}
	fmt.Printf("\nvalidated structured output:\n%s\n", prettyJSON(captured))
	fmt.Println("\ntool_choice forced the call; the SchemaValidator gated the arguments;")
	fmt.Println("ErrSessionClosed ended the loop the moment valid data arrived. For a HARD")
	fmt.Println("guarantee instead of a fix loop, add ChatOpts.Grammar / ResponseFormat (see `grammar`).")
	return nil
}

// ── grammar: server-side constrained decoding ──────────────────────

func runGrammar(ctx context.Context, cfg config) error {
	client := cfg.client()

	// 1) GBNF grammar: the server constrains token sampling so the output MUST
	//    match. Impossible to produce anything but one of the three words.
	grammar := `root ::= "red" | "green" | "blue"`
	fmt.Printf("GBNF: %s\n", grammar)
	fmt.Print("constrained output: ")
	out, err := completeConstrained(ctx, client,
		[]llm.Message{{Role: "user", Content: "Name a primary color."}},
		&llm.ChatOpts{Grammar: grammar})
	if err != nil {
		return err
	}
	fmt.Printf("%q\n", out)

	// 2) response_format: json_object mode — the server guarantees valid JSON.
	fmt.Println("\nresponse_format: {\"type\":\"json_object\"}")
	fmt.Print("json output: ")
	out, err = completeConstrained(ctx, client, []llm.Message{
		{Role: "system", Content: `Reply with a JSON object: {"city": string, "temp_c": number}.`},
		{Role: "user", Content: "Weather in Denver?"},
	}, &llm.ChatOpts{ResponseFormat: map[string]any{"type": "json_object"}})
	if err != nil {
		return err
	}
	fmt.Println(out)
	fmt.Println("\nNote: constrained decoding is a SERVER guarantee (hard). The Validator")
	fmt.Println("(see `schema`) is a CLIENT fix loop (soft). Use both as needed.")
	return nil
}

// completeConstrained runs one non-empty streaming completion with the given
// constraint opts and returns the accumulated text.
func completeConstrained(ctx context.Context, c *llm.Client, msgs []llm.Message, opts *llm.ChatOpts) (string, error) {
	ch, err := c.ChatStream(ctx, msgs, nil, opts)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for chunk := range ch {
		if chunk.Error != "" {
			return "", fmt.Errorf("stream: %s", chunk.Error)
		}
		b.WriteString(chunk.Content)
	}
	return strings.TrimSpace(b.String()), nil
}

// ── inject: notification injection + batching (offline proof) ──────

func runInject(ctx context.Context, cfg config) error {
	clk := &clock{}
	store := newDemoStore()
	sess := &agent.Session{SessionID: "demo", System: "You are an assistant.", Store: store, Runner: cfg.client(), Now: clk.next}

	// Two user messages queue while the agent is idle...
	store.publish(entry(agent.KindUser, "Remind me to review the PR.", clk.next()))
	store.publish(entry(agent.KindUser, "Also: what time is standup?", clk.next()))
	// ...and a system notification is injected into the same session.
	_ = sess.Inject(ctx, agent.Entry{Kind: agent.KindNotification, Tag: "deploy", Content: "Deploy #42 succeeded on prod.", CreatedAt: clk.next()})

	// BATCHING: ClaimPending marks ALL pending arrivals shown in one shot — the
	// two queued messages reach the model in ONE turn, not one turn each.
	n, _ := store.ClaimPending(ctx, "demo", clk.next())
	fmt.Printf("pending inbox arrivals claimed in one turn (batching): %d\n\n", n)

	// PROOF (no model needed): the injected notification + both messages all
	// render into the single context the next Turn would send.
	msgs, err := agent.DefaultContextBuilder(ctx, store, "demo", sess.System)
	if err != nil {
		return err
	}
	fmt.Println("context the next Turn sends to the model:")
	for _, m := range msgs {
		fmt.Printf("  %-9s %s\n", m.Role, oneLine(m.Content))
	}
	fmt.Println("\nThe [deploy] line is the injected notification — the model sees it inline")
	fmt.Println("with the conversation. A host injects mid-flight; the loop delivers it at")
	fmt.Println("the next turn boundary.")
	return nil
}

// ── lift: async tool results ───────────────────────────────────────

func runLift(ctx context.Context, cfg config) error {
	// A tool that can't answer inline (kicked off a long job) returns the LIFT
	// wire shape instead of a result:
	toolCallID := "call_run_job_1"
	liftWire := `{"pending": true, "correlation_id": "job-7", "ttl_s": 30}`
	fmt.Printf("tool returned: %s\n", liftWire)

	lr, ok := agent.ParseLiftRequest(liftWire)
	if !ok {
		return fmt.Errorf("expected a lift request")
	}
	// The dispatcher substitutes a clear "pending, wrap up" message so the model
	// stops acting on the placeholder. The turn ends; the SESSION STAYS ACTIVE.
	pending := agent.PendingResult(lr.CorrelationID, toolCallID, lr.TTLSeconds)
	fmt.Printf("\nmodel sees instead:\n  %s\n", oneLine(pending))

	// Later, when the upstream completes, the HOST injects the real payload as a
	// tool_result keyed by the SAME tool_call_id. The next Turn reconciles it.
	clk := &clock{}
	store := newDemoStore()
	store.publish(agent.Entry{
		ID: "r1", Kind: agent.KindToolResult, ToolCallID: toolCallID,
		ToolName: "run_job", Content: "Job job-7 finished: 3 files changed, tests green.",
		CreatedAt: clk.next(),
	})
	fmt.Printf("\nupstream completes → host injects (keyed by tool_call_id=%s):\n", toolCallID)
	msgs, _ := agent.DefaultContextBuilder(ctx, store, "demo", "")
	for _, m := range msgs {
		fmt.Printf("  %-9s %s\n", m.Role, oneLine(m.Content))
	}
	fmt.Println("\nagentkit owns only the wire shape + wording. Storage, the completion")
	fmt.Println("endpoint, and deadline GC stay with the host — lifting is event-driven,")
	fmt.Println("never a blocked goroutine.")
	return nil
}

// ── notify: supersede / clear / revalidator preparer ───────────────

func runNotify(ctx context.Context, cfg config) error {
	clk := &clock{}

	// SUPERSEDE: three re-emits for the same (group=file, key=main.go) collapse
	// to one live notice — the newest truth, not a stack of stale ones.
	store := newDemoStore()
	store.publishNotice(entry(agent.KindNotification, "build failed: main.go:10 undefined x", clk.next()), "file", "main.go", true)
	store.publishNotice(entry(agent.KindNotification, "build failed: main.go:12 undefined y", clk.next()), "file", "main.go", true)
	store.publishNotice(entry(agent.KindNotification, "build failed: main.go:14 undefined z", clk.next()), "file", "main.go", true)
	fmt.Printf("after 3 superseding emits, live notices: %d (newest wins)\n", store.liveNoticeCount())

	// CLEAR: a tool result / integration callback carries the clear wire shape;
	// the host retracts the notice. Zero model cost.
	cr, ok := agent.ParseClearRequest(`{"clear": true, "group_by": "file", "key": "main.go"}`)
	if !ok {
		return fmt.Errorf("expected a clear request")
	}
	_ = store.Clear(ctx, "demo", cr.GroupBy, cr.Key)
	fmt.Printf("after clear(%s=%s), live notices: %d\n", cr.GroupBy, cr.Key, store.liveNoticeCount())

	// PREPARER (MCP-revalidator convention): before each Turn, revalidate every
	// pending notice against the integration's truth. Empty truth = resolved →
	// clear. This is the prepareNotificationsBeforeSend hook.
	store2 := newDemoStore()
	store2.publishNotice(entry(agent.KindNotification, "lint error in a.go", clk.next()), "file", "a.go", false)
	store2.publishNotice(entry(agent.KindNotification, "lint error in b.go", clk.next()), "file", "b.go", false)
	rv := fakeRevalidator{truth: map[string]string{
		"a.go": "",                 // resolved → will be cleared
		"b.go": "lint: b.go:3 ...", // still broken → kept
	}}
	prep := agent.MCPPreparer(store2, rv)
	fmt.Printf("\npreparer: before-turn revalidation of 2 notices...\n")
	if err := prep.PrepareNotifications(ctx, "demo"); err != nil {
		return err
	}
	fmt.Printf("after revalidation, live notices: %d (a.go resolved → cleared; b.go kept)\n", store2.liveNoticeCount())
	fmt.Println("\nStale notices never reach the model, so it never pays to re-validate a")
	fmt.Println("condition that already resolved.")
	return nil
}

// fakeRevalidator stands in for an MCP integration's masked revalidator tool.
type fakeRevalidator struct{ truth map[string]string }

func (f fakeRevalidator) Revalidate(_ context.Context, _, key string) (string, bool, error) {
	v, ok := f.truth[key]
	return v, ok, nil
}

// ── compact: LOD truncation + compaction ───────────────────────────

func runCompact(ctx context.Context, cfg config) error {
	clk := &clock{}
	store := newDemoStore()
	big := strings.Repeat("lorem ipsum dolor sit amet, ", 40) // ~1100 chars
	for i := range 4 {
		store.publish(entry(agent.KindAssistant, fmt.Sprintf("event %d: %s", i, big), clk.next()))
	}
	store.publish(entry(agent.KindUser, "What did we decide?", clk.next()))

	// LOD: older large entries render as truncated stubs (an event-id pointer +
	// head) while the pristine tail stays verbatim — pure render-time, no model.
	lodShaper := &agent.Shaper{
		Store: store, Runner: cfg.client(),
		Policy: agent.ShaperPolicy{
			// Budget sits between the full render (~1100 tokens) and the
			// LOD-stubbed render (~150), so phase-0 overflows and LOD fires.
			BudgetTokens: 600, PreserveLastMessages: 1, LODTruncateAboveChars: 200,
		},
	}
	msgs, err := lodShaper.Build(ctx, "demo", "You are an assistant.")
	if err != nil {
		return err
	}
	fmt.Println("LOD-shaped context (older entries truncated, last message pristine):")
	for _, m := range msgs {
		fmt.Printf("  %-9s %s\n", m.Role, oneLine(m.Content))
	}

	// COMPACTION: with a budget too small for even the LOD stubs, the Shaper
	// summarizes the oldest prefix into a compaction marker via Store.Compact.
	// Summarize calls the Runner — here a stub so the demo is deterministic; in
	// production it's the real client.
	compStore := newDemoStore()
	for i := range 4 {
		compStore.publish(entry(agent.KindAssistant, fmt.Sprintf("event %d: %s", i, big), clk.next()))
	}
	compStore.publish(entry(agent.KindUser, "What did we decide?", clk.next()))
	compShaper := &agent.Shaper{
		Store: compStore, Runner: stubRunner{summary: "Summary: 4 setup events occurred; the open question is what we decided."},
		Policy: agent.ShaperPolicy{BudgetTokens: 120, PreserveLastMessages: 1, LODTruncateAboveChars: 100},
	}
	msgs, err = compShaper.Build(ctx, "demo", "You are an assistant.")
	if err != nil {
		return err
	}
	fmt.Println("\nCompacted context (oldest prefix folded into a [compacted] marker):")
	for _, m := range msgs {
		fmt.Printf("  %-9s %s\n", m.Role, oneLine(m.Content))
	}
	return nil
}

// stubRunner is a canned agent.LLMRunner returning a fixed summary, so the
// compaction demo is deterministic without depending on the (busy) model.
type stubRunner struct{ summary string }

func (s stubRunner) ChatStream(_ context.Context, _ []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	ch := make(chan llm.StreamChunk, 2)
	ch <- llm.StreamChunk{Content: s.summary}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

// ── shared helpers ─────────────────────────────────────────────────

// verbose wraps a dispatcher to print each tool call + result.
func verbose(inner agent.ToolDispatcher) agent.ToolDispatcher {
	return func(ctx context.Context, tc llm.ToolCall) (string, error) {
		fmt.Printf("\n  ↳ %s(%s)\n", tc.Function.Name, strings.TrimSpace(tc.Function.Arguments))
		res, err := inner(ctx, tc)
		fmt.Printf("    = %s\n  ", oneLine(res))
		return res, err
	}
}

// prettyJSON indents a JSON string; returns it unchanged if it isn't JSON.
func prettyJSON(s string) string {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return s
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return s
	}
	return string(b)
}

// oneLine collapses whitespace and truncates for tidy transcript printing.
func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 100 {
		return s[:97] + "..."
	}
	return s
}
