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
a bounded number of times then fails. A billing refusal (corrallm's 402, or a provider's own 400/403/429 saying the credit balance, quota or workspace limit is spent) is never retried: it returns `*llm.BudgetSpentError`, typed `upstream_funds_exhausted`, so a host can fall back. A retry delay is always clamped to a
safety ceiling.

**Finish reasons.** `finish_reason` and OpenRouter's `native_finish_reason` (e.g. `MALFORMED_FUNCTION_CALL`) are kept verbatim: `StreamChunk.FinishReason`/`NativeFinishReason` on the Done chunk, the same names on `llm.Usage` (streaming and non-streaming), and `agent.TurnResult.FinishReason`/`NativeFinishReason` for the Turn's last round.

**OpenRouter `reasoning_details`.** Its `reasoning.text` and `reasoning.summary` entries stream as `StreamChunk.Reasoning`, only when `delta.reasoning` and `delta.reasoning_content` are empty (OpenRouter often sends both; the text is never doubled). `reasoning.encrypted` is ignored. The capability record's `stream.reasoning_field` can read `reasoning_details`.

A provider can also fail with HTTP 200 and then an `{"error":{...}}` SSE event. The client keeps the whole object as `*llm.StreamError` (message, code as number or string, type, param, raw `metadata`, `retry_after`), on `StreamChunk.StreamErr` and as the Turn's error (`errors.As`). If such an error arrives before any content, reasoning, tool call or usage was forwarded and looks transient (code 429 or 5xx, or "overloaded", "rate limit", "unavailable" in code, type or message), the request is re-sent inside the same `RetryBudget`, jittered, honoring the error's `retry_after`; the caller sees one stream. An event with `"reason":"queue-timeout"` (corrallm, after it queued the request; `StreamError.Reason`, and `Status` counts when `code` is absent) retries immediately like a queue-timeout 429, at most `maxQueueTimeoutStreak` (5) times in a row, then backs off. After the first forwarded chunk it is not retried.

A silent link is bounded too. After a network change a socket can stay open and deliver nothing, and no retry starts because no attempt ends. Two limits end such an attempt: `Client.FirstByteTimeout` (request sent until the first response byte, default 15m, long because llama.cpp sends nothing while it processes a long prompt and corrallm can hold a queued request up to 12m with no bytes) and `Client.StallTimeout` (gap between bytes once the first arrived, default 2m). 0 means the default, negative turns the limit off. Any bytes count, SSE comment lines such as `: keep-alive` included. The failure is `*llm.StallError{Phase: "first_byte" | "between_chunks", After}`, transient for `llm.TransientUpstream`. Before anything reached the caller it is retried inside `RetryBudget`, jittered, reported as `RetryStall`; after output was forwarded it is returned (`StreamChunk.Stall`, the Turn's error) and not resent. A caller cancel stays `ctx.Err()`. The client also owns its HTTP transport (a clone of the default, one per client) with HTTP/2 pings: a connection silent for 30s is pinged and closed if no answer in 15s.

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

Correlation ids. `llm.WithTraceID(ctx, id)` sets the `X-Trace-Id` for calls that have no `ChatOpts` (embed, transcribe, `DiscoverContext`, probes); a chat call uses `ChatOpts.TraceID` first, then the ctx id. Every `Usage` carries `TraceID` (what was sent) and `ResponseID` (the reply body's `id`, first non-empty chunk id on a stream; the key for a provider's generation lookup). `ChatOpts.Headers` adds per-request headers (for example OpenRouter `x-openrouter-trace-<key>`); `Authorization`, `Content-Type` and `X-Api-Key` are ignored.

Cost currency. `Usage.Cost.Currency` for a `usage.cost` in the body, first match wins: `cost_estimated` true gives `"USD estimate"`; else `usage.cost_currency` (trimmed; `usd` in any case is `USD`, other codes kept as sent); else a `usage.cost_source` field (corrallm only; stopgap until it sends `cost_currency`) gives `USD`; else the LiteLLM header gives `USD`; else an `openrouter.ai` base URL gives `USD` (OpenRouter credits are dollars, per its FAQ); else `""`. OpenRouter key budgets are USD too.

## 2. The tool-call loop — `tools`

`Session.Turn` runs chat → tool_calls → dispatch → feed back → repeat. You
supply `Tools []llm.ToolDef` and a `Dispatch`:

```go
func dispatch(ctx context.Context, tc llm.ToolCall) (llm.ToolResult, error) {
    switch tc.Function.Name {
    case "get_weather":
        // parse tc.Function.Arguments (JSON), do the work, return the text
        return llm.TextResult(result), nil
    case "screenshot":
        // media rides in Parts: any llm.ContentPart kind
        return llm.ToolResult{Text: "the screen", Parts: []llm.ContentPart{llm.ImagePart(dataURI)}}, nil
    default:
        return llm.TextResult(fmt.Sprintf("ERROR: unknown tool %q", tc.Function.Name)), nil // note: not a Go error
    }
}
```

Errors meant for the **model** (unknown tool, bad args) go into the result's
Text so the loop stays alive and the model can recover. A result's Parts are
persisted on the tool-result Entry (`Entry.Parts`; the Store decides how media
bytes are kept) and rendered as one tool message: a text part, then the media.
An LOD stub or summary drops them. `Session.PrepareToolResult` sees every
dispatched result before it is persisted, so a host can keep a part, replace it
with a text description, or drop it (a model that cannot take that kind); an
error from it is recorded as an `ERROR: ...` result. A returned Go error
is recorded as an `ERROR: ...` result and the loop continues. A terminal tool returns `agent.ErrSessionClosed` to stop the
loop after its result is persisted (used with `tool_choice=required` roles).

### Claude Messages API

A client pointed at `api.anthropic.com` (or with `MessagesAPI: true`) talks to
`/v1/messages`. Callers change nothing: messages, tools, options and stream chunks
stay in the OpenAI shape. The body is built as for any provider, translated by
[openai-claude-api-shim](https://github.com/iodesystems/openai-claude-api-shim), and
sent through the same retries; the reply is transcoded back into what `readStream`
reads. The decision is per client, so a parent on one provider can run a child on
another. The reason is cost: Anthropic's OpenAI-compatible endpoint reads nothing from
the prompt cache, and the shim places breakpoints.

`ChatOpts.Effort` becomes `output_config.effort`; thinking cannot be switched off on
every model, so `none` is `low`. A refusal is a stream error. The client needs a key:
Workload Identity Federation went with the SDK. What the shape cannot carry is listed
in the shim's README.

### Effort levels

`ChatOpts.Effort` is the one setting for how hard the model thinks. Levels, lowest
first: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. The empty value
(`EffortUnset`) sends nothing and the server decides. `llm.Efforts` lists the levels;
`llm.ParseEffort` reads a name; `llm.EffortAbout` has one line per level.

The client maps the level to the endpoint's fields. In brief:

| level | openai | openrouter | llama.cpp | anthropic compat | Messages API |
|---|---|---|---|---|---|
| none | `none` | thinking off | thinking off, budget 0 | no thinking | `low` |
| minimal | `minimal` | `minimal` | budget 512 | budget 1024 (`low`) | `low` |
| low, medium | as is | as is | budget 1024, 4096 | budget 1024, 4096 | as is |
| high | `high` | `high` | thinking on, no budget | budget 16k | `high` |
| xhigh | `xhigh` | `xhigh` | `high` | budget 32k | `xhigh` |
| max | `xhigh` | `xhigh` | `high` | budget 64k | `max` |

A Messages API model that predates the effort field gets nothing. `ReasoningBudgetTokens`
is an explicit cap: with a level it sends min(level budget, explicit); with no level, the explicit budget alone; at `none`, nothing. Anthropic-compat keeps the budget below `max_tokens` (or sends no thinking block if under 1024). The record's refusals and `reasoning.off` bind only openai, openrouter and the Messages API; llama.cpp and anthropic-compat ignore them, while `thinking_only` binds all.

**Clamp.** If the capability record refuses a level, the client sends the nearest
lower level that is allowed. If none is lower, it sends the nearest higher one. It never
steps down to `none`: that is a different request. A `none` request that the record
refuses (`thinking_only`) goes to the lowest level allowed. If no level is left, the
client sends nothing and says why.

**Read the result.** `Client.Effective(level)` returns an `EffortChoice`: `Wanted`,
`Effective`, `Native` (false when a token budget stands in for the level), `Reason`
(empty when `Effective` equals `Wanted`) and `Cause`. `Client.OnTranslate` reports each
request whose level changed, with `Key` `"effort"` and `Reason` set. A refusal that
teaches a fact rebuilds the request, so the new level is reported in the same call.

**Pickers.** `Client.EffortSupport()` returns one `EffortChoice` per level. A row whose
`Effective` differs from `Wanted` is not available as asked; `Reason` says why. The
pure form, `llm.EffortSupport(dialect, messagesAPI, model, caps)`, needs no Client.

### Runaway-loop detection (on by default)

Two different failures, two guards.

**A generation that collapses into repetition.** Measured: ~30 minutes spent
re-emitting an 85-character fragment, to the server's context limit. `Stop`
can't catch it (the text isn't known in advance) and a grammar can't (the
output is well-formed) — only something watching the bytes arrive can, so
`llm.Client` does. It watches content, reasoning **and** tool-call arguments (the
measured loop was inside `arguments`), and on a trip it closes the stream, which
disconnects and makes the server abandon the generation.

The caller sees a final chunk carrying the reason; `Session.Turn` handles it for
you — it trims the redundant copies off the reply *before* persisting it (so the
retry isn't built from a context primed to continue the loop), tells the model
what it did, and re-prompts. Two consecutive collapses are retried; a third
fails the Turn rather than paying the same cost in installments.

```go
client.Repetition = llm.RepetitionGuard{Off: true}   // disable entirely
client.Repetition = llm.RepetitionGuard{MinSpan: 2048} // or just loosen it
session.MaxRepetitionRetries = -1                    // never give up (MaxTurns still bounds)
```

Thresholds are set so that ordinary repetition — blank lines, markdown table
rules, indentation, recurring boilerplate — never trips it. A cycle shorter than
`MinPeriod` (24 bytes) is deliberately *not* reported at all; `ChatOpts.MaxTokens`
is the backstop for those.

**A reasoning-only round.** A reply with reasoning and no text and no tool call
(a thinking budget ate the room, or the model stopped after a failed call) used
to persist nothing and end the Turn as `TurnReply`. Now the loop appends a
notification tagged `agent.ReasoningCutTag` holding the last ~2000 bytes of the
reasoning, and runs one repair round with reasoning off (`Effort`
`none` and a reasoning budget of 0; the capability record rewrites the level per
model). One repair per Turn. If that round is empty too, the Turn
ends `TurnReasoningCut` with `*agent.ReasoningCutError`. `Session.OnReasoningCut`
reports the cut (`Repaired` false) and the recovery (`Repaired` true);
`Session.NoReasoningCutRepair` turns the repair off.

**An empty round.** A round with no text, no tool call and no reasoning (seen:
Gemini through OpenRouter; `MALFORMED_FUNCTION_CALL` now has its own repair,
below) used to end
the Turn as `TurnReply` with `Reply` "" and a nil error. Now it gets the same
single repair, shared with the reasoning cut (one per Turn in total): a
notification tagged `agent.EmptyReplyTag`, then one round with the host's opts
unchanged. If the next round is empty too, the Turn ends `TurnEmptyReply` with
`*agent.EmptyReplyError`, which holds `FinishReason` and `NativeFinishReason`
and matches `errors.Is(err, agent.ErrEmptyReply)`. An empty round with finish
`length` is not this: it follows the overflow path. With
`Session.NoReasoningCutRepair` set there is no repair round and an empty round
returns the error at once. Hosts that treated `Reply == ""` with a nil error as
"nothing to say" must now handle the error.

**A provider failure dressed as a stop.** OpenRouter reports an upstream failure
it does not recognize as finish `stop` and puts the real reason in
`native_finish_reason` (seen: Gemini streams reasoning, then fails with
`MALFORMED_RESPONSE`). An empty round (no visible text, no tool call; reasoning
does not count) with finish `stop` and a native reason that is not a clean stop
(`stop`, `end_turn`, `stop_sequence`, `eos`, `complete`, `completed`,
`pause_turn`, `compaction`, `end_of_stream`, any case) is not sent to the
repairs above: it is discarded and the same request is resent, with no notice
and the opts unchanged. Resends back off from 1s, doubling, each sleep capped at
5 minutes, plus additive jitter; at most `Session.MaxProviderFailRetries` per
Turn (0 means 10, negative turns it off). They do not count against `MaxTurns`
or the one repair. `Session.OnProviderFailRetry` gets each resend (attempt,
native reason, the wait) before the sleep. A reason that repeats stops the
resends sooner: after 2 resends in a row that came back with the same native
reason (any case), the round is not resent again; a different reason starts the
count again. When the resends stop, the round goes to the repairs. A round with
visible text is kept as the reply whatever its native reason says.

**A malformed tool call.** Gemini ends a round with native reason
`MALFORMED_FUNCTION_CALL` (OpenRouter reports it with finish `stop` or `error`)
when the model wrote a tool call the provider could not parse. Resending the
same request gets the same bad call, so it is not resent. An empty round with
this native reason, whatever the finish, gets a notification tagged
`agent.MalformedCallTag` that names the cause and the usual fixes (call the
tool by name with one JSON object, not `default_api.fn(...)`; escape newlines
and quotes; keep long arguments short), then one round with the host's opts
unchanged. At most `Session.MaxMalformedCallRepairs` per Turn (0 means 3,
negative turns it off and the round is resent as above). This budget is apart
from the one shared repair and from the resends, and `NoReasoningCutRepair`
does not turn it off. When it is spent the Turn ends `TurnEmptyReply` with
`*agent.EmptyReplyError` holding the native reason.

**A turn-level loop:** the same call returning the same result, round after
round. Every round is well-formed, so only the repetition across rounds shows
it. `MaxRepeatedExchanges` (default 5) warns the model at 5 identical
call+result exchanges and stops the Turn at 10.

The **result** is part of that comparison on purpose: polling is a legitimate
reason to repeat a call, so a guard keyed on the call alone would kill a job
that is correctly waiting. A poll whose result advances never trips this.

```go
session.MaxRepeatedExchanges = -1 // off
```

**A loop with no period: `loopdetector`.** Opt-in. The guards above need identical
bytes back to back or identical rounds. A model can also go round many variations,
in any order, with other output between the copies: one production reply was 291
sentences of which 40 were distinct, each said about 21 times, and no exact guard
saw it. `loopdetector.LoopDetector` finds those.

Its input is the session plus the stream: `Begin(history)` takes the request's
messages, `Watch(channel, delta)` the reply as it arrives, thinking included. There
is no cache; nothing outlives the reply. Its output is a count, and `Tolerance` says
at what count the stream is cut:

```go
d := loopdetector.New(loopdetector.Config{
	Tolerance: 3, // cut at 3 loops. Negative: count, never cut
})
d.Install(client) // replaces client.Repetition for this client's streams

// Or feed it yourself, which is what a proxy does:
s := d.Begin(messages)
if info := s.Watch(llm.ChannelReasoning, delta); info != nil { /* cut */ }
loops, kind := s.Loops()
d.Counters() // totals, for metrics
```

It finds four kinds: `exact` (the guard), `cycle` (the same sentences recurring),
`phrase` (the same, a little reworded, matched by BM25) and `tool` (the same call
with the same result; no tool is exempt, so polling for an unchanged result counts,
and a host should wait on an event instead). A window of recent sentences has to be mostly repeats, so a
sentence the model says often is not a loop. History is read from the last message by a
person on, so a person asking again is not a loop, and a loop that goes on only
across their messages is not seen. Host notices (`[tag] text`) do not end a turn. History is counted but cannot cut: only a reply
that continues the loop is interrupted. `ToolTolerance` counts tools while text is
cut. A finding carries `First` and `Last`, the earliest repeat in view and the one
that tripped it, capped at `SampleBytes`. `Finding()` returns it whether or not the
stream was cut. A proxy that found a loop and let the stream pass sends it on the
final event, where it arrives as `StreamChunk.Repetition` with no `StopReason`:
nothing was cut, and what it means is the caller's to judge. Random data inside every copy
of a line, and real churn that is new text each time, are not found.

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

### Interjecting while a turn runs

A person's words typed while a turn is in flight are delivered as a real
`role: user` message, not text inside a tool result (some models, Gemini via
OpenRouter among them, ignore text inside a functionResponse). The host saves a
pending `KindUser` entry to the store; `Turn` claims pending entries at the top
of every loop iteration, so the next request carries it and the loop goes on.

Where it lands: entries sort by `CreatedAt`, and a tool result is stamped when
the tool finishes, so words typed while a tool ran are older than its result.
The renderer holds any user, notification or compaction entry back while a
round's tool calls are unanswered and emits it right after the last result:
`assistant(tool_calls) -> tool -> user`. On the Claude Messages API the shim
puts the text in the same user turn as the tool_result blocks, after them.

Some chat templates refuse a user message right after tool results. Set
`ShaperPolicy.FoldUserIntoToolResult` (or `RenderOptions.FoldUserIntoToolResult`
through `DefaultContextBuilderWith`, as the Session's `Build`) and a text user
entry that follows tool results is appended to the last result as
`[the user wrote, while this ran] <text>`. An entry with multimodal `Parts`
stays a user message. Off by default.

Cache: the order depends only on the stored entries, so each round's messages
are a prefix of the next round's.

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

To protect the KV cache, reshaping is deferred until the context would cross
`BudgetTokens − LODHeadroomTokens` (default ~10k), then done in one decisive
pass with runway — not eagerly every turn. A compaction is **surfaced**
(`OnCompaction` + `TurnResult.Compactions`) with its summary + before/after
tokens, and the same turn continues to the reply. Every Turn also reports
`TurnResult.Usage` — **Total** billed vs **Active** window. The `compact` demo
shows a compaction folding `1150→28` tokens, the summary emitted as a hidden
field, and the `total`/`active` tally.

**A fold can keep more than the pristine tail (`ShaperPolicy.FoldKeepPct`, opt-in).**
By default a fold keeps only the `PreserveLast*` tail, so the window is nearly
empty after it and the next fold comes soon. With `FoldKeepPct` set (say 55) and
a target, the fold keeps the largest newest suffix whose rebuilt prompt (the
caller's prefix, a summary at its cap, pinned entries, the suffix; no LOD) fits
that percent of the target. The pristine tail is the minimum, the suffix is
widened so no result loses its call, and when the band holds nothing more the
fold is the default one.

**What a fold reports (`CompactionInfo`).** All after-numbers are measured on
the store after the fold, from the render the next build sends, and are
conversation tokens: the system prompt and tool definitions are not counted.

| field | meaning |
|---|---|
| `TokensBefore` | the render that decided the fold, less the system prompt |
| `TokensAfter` | what the next build sends, less the system prompt |
| `SummaryTokens` | the marker this fold wrote |
| `PinnedTokens` | pinned entries older than the kept part (a fold skips them) |
| `KeptTokens`, `KeptCount` | everything else: normally the newest entries kept verbatim |

`SummaryTokens + PinnedTokens + KeptTokens == TokensAfter`. The fold decision
itself still counts the system prompt: `CompactionStart.TokensBefore` is that
number, compared with `Target`.

**Old results can be stubbed under target after a fold (`ShaperPolicy.FoldMidLOD`, opt-in).**
Under target a build normally sends every entry verbatim, so a wide kept band
refills the window with results that were already sent whole. With
`FoldMidLOD`, once the context holds a compaction marker, a build under target
stubs oversized entries older than the LOD boundary, stepped by `LODStep`: the
prompt prefix changes once per step, not on every round. The pristine tail, the
pins and the markers stay verbatim, and every call keeps its result. The fold
sizes its kept band with the same render, so `FoldKeepPct` keeps more entries,
and `TokensAfter` is that render. Off, the render is the old one.

**A fold can reuse the session's cache (`Shaper.SessionPrefix`, opt-in).** Set
`SessionPrefix{System, Tools}` to the session's EFFECTIVE system prompt and the
exact tool defs it sends. The fold's FIRST request is then sent as the session's
own prefix: that system, the history as messages, the session's tools with
`tool_choice: "none"`, the summary instruction last, and the session's current
`PromptCacheKey` (session id + newest marker before the fold; a key in
`SummaryOpts` wins), because that key routes the request to the slot holding the
prefix. When the whole fold fits, that request is the fold. When it does not
(the common case: a fold is triggered near the budget), it covers the longest
head of the history that fits beside the session's system and tools, never
splitting a tool call from its result; the staged worker-prompt loop folds the
rest on top of its summary, with no session tools and no key. A prefix request
refused with `InputTooLargeError` falls back to the staged loop from the start.
Limit: pinned entries (`PinFirstUser`, `PinLatestUser`) are not in the fold
request, so the cached prefix matches only up to the first pinned entry.

**Telling a fold from a round: `agent.IsFold(ctx)`.** Every model request the
Shaper sends to fold carries a ctx mark. A host that wraps the `LLMRunner` (to
count fold usage apart, or route it) reads `IsFold(ctx)` instead of matching the
system prompt, which a prefix fold shares with the session.

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

**Roots and other server requests.** By default the client sends `capabilities:
{}` and answers every server-initiated request -32601. To serve `roots/list`,
set `MCPConfig.ClientCapabilities` (`{"roots":{"listChanged":true}}`) and
`Manager.SetRequestHandler` (before `StartServer`; the handler gets the server
id, runs in its own goroutine, and its ctx ends with the connection; return
`*RequestError` for a specific JSON-RPC code, `ErrMethodNotFound` to decline).
`Manager.Notify(serverID, "notifications/roots/list_changed", nil)` tells the
server the list changed. Works on stdio and HTTP.

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

### Raw wire dump — `AGENTKIT_WIRE_DUMP`

For a provider bug the parser hides (a malformed event, a comment line, a
stream that ends oddly), set a directory and reproduce:

```
AGENTKIT_WIRE_DUMP=/tmp/agentkit-wire ./your-host
```

Every HTTP request the `llm.Client` makes writes one file there (created 0700,
files 0600), named `<UTC time>-<seq>-<trace id>-a<attempt>.txt`; a retry is a
new file with the next attempt number. Each file holds the request line,
headers and body byte for byte, the response status and headers, every response
line as received with a time offset (`: keep-alive`, `event:`, `data:`,
`[DONE]`), and how the body ended (eof, read error such as a stall, or closed
early). The variable is read per request, so a host can turn it on and off
without a restart; unset or empty is off.

Redacted: `Authorization`, `Proxy-Authorization`, `X-Api-Key`, `Api-Key`,
`Cookie`, `Set-Cookie`, any header or URL query parameter whose name contains
`key`, `token` or `secret`, and a URL password. **The bodies are not redacted**:
a dump holds the full prompt, the tools and every tool result. Treat it as
sensitive. A failure to write the dump never fails the request; that dump just
stops.

## 12. Putting it together — `converge`

The features above are not independent — they **coalesce**. A lifted tool
result, user messages that queued, and live notifications all converge into one
merged context at the next turn (see
[concepts.md → the coalescing turn boundary](concepts.md#the-coalescing-turn-boundary)).
The `converge` demo runs it end to end: a tool call parks, then a resolved
result + a user message + a notification accumulate while the model is idle, and
the next Turn delivers all three together — the model addresses them in one
reply. This convergence is the point of the event-driven turn model.
