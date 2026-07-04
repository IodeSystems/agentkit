# Features

Each tablestake, the code that uses it, and the demo that shows it. Run any
demo with `go run ./examples/agentkit-demo <name>`.

## 1. Streaming chat + fair-share 429 retry — `chat`

`llm.Client.ChatStream` returns a channel of `StreamChunk` (content deltas,
then tool calls, then a final chunk carrying `Usage`). Content streams token by
token; tool-call fragments are reassembled internally before emission.

Rate limits are handled **inside** the client. On 429 it waits and retries,
honoring the server's `retry_after` — from the `Retry-After` header **or** a
JSON backpressure body like corrallm's:

```json
{"error":{"reason":"queue-timeout","retry_after":10,"type":"backpressure"}}
```

429 retries until a slot frees, bounded by `Client.RetryBudget` (default 5m) and
your `ctx` deadline, whichever is shorter — on a contended endpoint another wait
is another shot at a slot, but the budget stops it hanging forever. 5xx retries
a bounded number of times then fails. A retry delay is always clamped to a
safety ceiling.

```go
client := llm.NewClient(baseURL, apiKey, model)
client.RetryBudget = 5 * time.Minute // cap total time spent retrying one request
```

```go
ch, err := client.ChatStream(ctx, msgs, tools, &llm.ChatOpts{TraceID: "thread-42"})
for chunk := range ch {
    fmt.Print(chunk.Content)
    if chunk.Usage != nil { /* prompt/completion/total (+ cache fields if reported) */ }
}
```

> The API key is your **scheduling identity** (`Authorization: Bearer …` → a
> priority on the fair-share proxy), not just auth. `TraceID` is correlation
> only (an `X-Trace-Id` header for server logs).

## 2. The tool-call loop — `tools`

`Session.Turn` runs chat → tool_calls → dispatch → feed back → repeat. You
supply `Tools []llm.ToolDef` and a `Dispatch`:

```go
func dispatch(ctx context.Context, tc llm.ToolCall) (string, error) {
    switch tc.Function.Name {
    case "get_weather":
        // parse tc.Function.Arguments (JSON), do the work, return a string
        return result, nil
    default:
        return fmt.Sprintf("ERROR: unknown tool %q", tc.Function.Name), nil // note: not a Go error
    }
}
```

Errors meant for the **model** (unknown tool, bad args) go into the result
string so the loop stays alive and the model can recover. A returned Go error
aborts the Turn. A terminal tool returns `agent.ErrSessionClosed` to stop the
loop after its result is persisted (used with `tool_choice=required` roles).

## 3. Schema validation + fix loop — `schema`

`tool_choice=required` forces the model to *call* a tool, not to fill it
correctly. `ValidatingDispatcher` closes that gap: a call whose arguments fail
validation is rejected with a **fix instruction as its tool result** — the
session stays active and the model retries. The inner dispatcher never runs on
a bad call.

```go
v := agent.NewSchemaValidator(tools)                 // required-keys + JSON-kind checks from the tools' own schemas
dispatch := agent.ValidatingDispatcher(realDispatch, v)
```

`SchemaValidator` is dependency-free and conservative (object shape, required
fields present + non-null, primitive type match). For stricter guarantees, plug
your own `agent.Validator` (e.g. a full JSON-Schema library) — the interface is
the seam.

## 4. Server-side constrained decoding — `grammar`

The **hard** counterpart to the (soft) validator: the server constrains token
sampling so the output *cannot* violate the shape. Two knobs on `ChatOpts`,
forwarded raw to the endpoint (llama.cpp / corrallm honor them; others ignore):

```go
// GBNF grammar — output must match the grammar.
&llm.ChatOpts{Grammar: `root ::= "red" | "green" | "blue"`}

// JSON mode — output is guaranteed valid JSON.
&llm.ChatOpts{ResponseFormat: map[string]any{"type": "json_object"}}

// or a full json_schema response format:
&llm.ChatOpts{ResponseFormat: map[string]any{
    "type": "json_schema",
    "json_schema": map[string]any{"name": "weather", "schema": schemaObj},
}}
```

Use constrained decoding when the structure must be guaranteed; use the
Validator when you want the model to self-correct with feedback. They compose.

## 5. Notification injection + batching — `inject`

Inject a message or system notification into a session; the next `Turn`
renders it inline:

```go
sess.Inject(ctx, agent.Entry{Kind: agent.KindNotification, Tag: "deploy",
    Content: "Deploy #42 succeeded on prod."})
```

**Batching is inherent:** `ClaimPending` marks *all* pending arrivals shown at
the top of a turn, and the context builder renders every non-subsumed entry —
so several messages that queued between activations reach the model in **one**
turn, not one turn each.

## 6. Async tool results (lifting) — `lift`

A tool that can't answer inline (kicked off a job, an approval, a long shell)
returns the lift wire shape instead of a result:

```json
{"pending": true, "correlation_id": "job-7", "ttl_s": 30}
```

The dispatcher recognizes it and substitutes a "pending, wrap up" message:

```go
if lr, ok := agent.ParseLiftRequest(result); ok {
    result = agent.PendingResult(lr.CorrelationID, tc.ID, lr.TTLSeconds)
    // record the pending call host-side, keyed by tc.ID
}
```

The turn ends normally; the **session stays active**. When the upstream
completes, the host injects the real payload as a `KindToolResult` Entry keyed
by the same `ToolCallID` (`sess.Inject`), and the **next Turn reconciles** it.
The `lift` demo runs this end to end: Turn 1 parks on the async tool and the
model acknowledges; the host injects the finished result; Turn 2 resumes and
answers — the turn never blocked on the job. agentkit owns only the wire shape +
wording; storage, the completion endpoint, and deadline GC stay with you.
Lifting is event-driven, never a blocked goroutine.

## SSE streaming to a UI — `serve`

`llm.StreamChunkToSSE` formats a `StreamChunk` as a Server-Sent Events frame —
content deltas as `data: {"type":"content","text":"…"}`, tool calls as
`{"type":"tool_call",…}`, plus `[DONE]` and errors. Relaying a live completion
to a browser is then a ~10-line handler:

```go
ch, _ := client.ChatStream(r.Context(), msgs, tools, nil)
w.Header().Set("Content-Type", "text/event-stream")
for chunk := range ch {
    io.WriteString(w, llm.StreamChunkToSSE(chunk))
    w.(http.Flusher).Flush()
}
```

The `serve` demo stands up this endpoint and self-requests it, printing the raw
frames a browser's `EventSource` would receive.

## 7. Notification lifecycle — `notify`

Notifications are cheap to publish but expensive to *consume*: a stale in-flight
notice (an MCP-LSP "bad build" for a file that now compiles) is re-validated by
the model every turn until it's cleared. Three primitives, keyed on a notice's
`(groupBy, key)` partition:

- **supersede** — a re-emit for the same key replaces the prior unshown notice
  (newest wins) instead of stacking. The inbox holds ≤1 live notice per key.
- **clear** — retract the notice when its condition resolves. A tool result or
  integration callback carries `agent.ClearRequest`
  (`{"clear":true,"group_by":"file","key":"main.go"}`); `ParseClearRequest`
  reads it. Zero model cost.
- **preparer** — the `NotificationPreparer` hook runs at the top of every turn
  (after claim, before build) so a resolved notice is gone before it renders.
  `agent.MCPPreparer` is the ready-made preparer for the MCP-revalidator
  convention: for each pending notice, call the integration's masked
  revalidator tool for its group key; an empty "current truth" means resolved →
  clear (`agent.IsResolvedTruth`). Fail-open on a flaky tool.

This is where an *active* "check back" belongs — an integrator-owned hook that
runs whatever it trusts, not daemon-side shell.

## 8. Context shaping — `compact`

`Shaper.Build` fits history to the window (see [concepts.md](concepts.md#shaper--fitting-the-context-window)):
pristine tail → LOD truncation → compaction. LOD is pure render-time (the
stored entry keeps its full content behind an `event_id` pointer); compaction
folds the oldest prefix into a summary marker via `Store.Compact`.

## 9. MCP tools — `mcp`

`mcpmgr.Manager` spawns stdio MCP servers, discovers their tools, and calls
them, with project-scoped (shared) and thread-scoped (per-workspace) instances,
plus 0600 secret-file materialization for servers that need credentials. The
entire integration is two bridges:

```go
mgr := mcpmgr.NewManager()
mgr.StartServer(ctx, mcpmgr.MCPConfig{ID: "everything", Command: "npx",
    Args: []string{"-y", "@modelcontextprotocol/server-everything"}})
tools := mgr.GetTools()                       // discovered MCPTools

defs := mcpToolDefs(tools)                     // MCPTool → llm.ToolDef (advertise)
dispatch := mcpDispatcher(mgr, tools)          // route calls → Manager.CallTool
```

`mcpToolDefs` copies name/description and drops the MCP `InputSchema` straight
into `ToolDef.Parameters` (it's already JSON Schema); `mcpDispatcher` maps each
tool name to its owning server and calls `Manager.CallTool`. See
`examples/agentkit-demo/mcp.go`. mcpmgr is independent of `agent` — use it
wherever you build tool defs.

## 10. Reliable typed extraction — `structured`

The features compose. To get *guaranteed* typed data out of a model, the
`structured` demo stacks four of them: `tool_choice` forces the model to call a
specific tool, the `SchemaValidator` fix loop gates its arguments,
`ForcedTerminalTool` makes it the session's only exit, and `ErrSessionClosed`
ends the loop the instant valid data arrives. Add `ChatOpts.Grammar` /
`ResponseFormat` for a hard server-side guarantee instead of the soft fix loop.

## 11. Observability — `--trace`

Every `Session` and `Shaper` takes an optional `agent.Tracer` (nil = zero
overhead). It captures spans for `Turn`, `streamChat`, and `Shaper.Build` with
attributes (message count, tool calls, token usage). A host adapts its own
tracing onto the two-method interface; `examples/agentkit-demo/trace.go` is a
~40-line stdout implementation you get with `--trace`:

```
┌ agent.Turn
│ ┌ agent.streamChat
│ └ agent.streamChat (16.9s)  n_messages=2 n_tool_calls=1 total_tokens=410
└ agent.Turn (16.9s)
```
