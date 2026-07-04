# Concepts

agentkit's `agent` package has a small, deliberately host-neutral core. Learn
these five types and the rest is mechanics.

## The neutral seam

`agent` reasons over exactly two shapes:

- **`llm.Message`** — the wire format sent to the model (`role` + `content`).
- **`agent.Entry`** — one durable conversation record, the shape your storage
  persists and the shaper reasons over.

It never imports your storage or event model. You map your own rows onto
`Entry` and back. Fields agentkit doesn't interpret (`Tag`, `Origin`) are
round-tripped verbatim, so you can smuggle host provenance through without
agentkit knowing what it means.

## `Entry` and `EntryKind`

```go
type Entry struct {
    ID         string
    Kind       EntryKind  // user | assistant | tool_call | tool_result | compaction | notification
    Content    string
    ToolCallID string     // correlates a tool_call with its tool_result
    ToolName   string
    Tag        string     // opaque display label for a notification (e.g. "deploy")
    Origin     string     // opaque host provenance; agentkit never reads it
    CreatedAt  int64      // ns; the ordering key
}
```

`EntryKind` drives how an entry renders into an `llm.Message`:

| Kind | renders as |
|------|-----------|
| `KindUser` | `user` message |
| `KindAssistant` | `assistant` message |
| `KindToolResult` | `tool` message (named by entry ID) |
| `KindCompaction` | `user` message prefixed `[compacted]` |
| `KindNotification` (and anything unrecognized) | `user` message prefixed `[<Tag>]` |

## `Store` — the persistence you implement

Six methods. No host event types leak across it.

```go
type Store interface {
    ClaimPending(ctx, sessionID, at int64) (int, error) // mark pending inbox arrivals shown; return how many
    Append(ctx, sessionID, Entry) error                 // persist one entry
    Context(ctx, sessionID) ([]Entry, error)            // all non-subsumed entries (any order)
    Compact(ctx, sessionID, Compaction) error           // write a summary marker + flag subsumed rows, atomically
}
```

`ClaimPending` returning a count is how the loop knows something arrived while
it was idle — the entries themselves surface through `Context`. Your inbox /
publish helpers (how a message *becomes* pending) are yours; they aren't part
of the interface. See `examples/agentkit-demo/store.go` for a complete
in-memory implementation you can copy.

## `Session` — what you drive

You construct a `Session` per unit of work and call `Turn`:

```go
sess := &agent.Session{
    SessionID: "s1",
    System:    "…",
    Store:     store,
    Runner:    client,          // agent.LLMRunner; *llm.Client satisfies it
    Tools:     tools,           // []llm.ToolDef advertised to the model
    Dispatch:  dispatch,        // agent.ToolDispatcher
    ChatOpts:  &llm.ChatOpts{}, // optional: tool_choice, grammar, response_format
    // optional seams:
    Build:     shaper.Build,    // ContextBuilder; nil → DefaultContextBuilder (verbatim)
    Preparer:  preparer,        // pre-turn notification revalidation
    OnAssistantToken: onToken,  // streamed content for SSE / live UI
    ForcedTerminalTool: "",     // name the one tool that's the session's only exit
    MaxTurns:  100,
    Tracer:    tracer,          // optional spans
}
reply, err := sess.Turn(ctx)
```

`Turn` is the loop: claim the inbox → prepare notifications → build context →
stream a completion → persist the reply → dispatch tool calls → feed results
back → repeat, until the model stops calling tools (or a terminal tool fires,
or `MaxTurns`).

`Session.Inject(ctx, Entry)` appends to this session's own log so the next
`Turn` renders it — the self-inbox injection primitive.

## `Shaper` — fitting the context window

A `Session.Build` is any `ContextBuilder`. The default renders history
verbatim. `Shaper.Build` adds three phases on top:

1. **pristine tail** — the last N messages + M tool exchanges are always kept
   verbatim, regardless of size.
2. **LOD truncation** — older oversized entries render as a short stub (an
   `event_id` pointer + head). Pure render-time; the stored entry is untouched.
3. **compaction** — if LOD-truncated context still overflows, summarize the
   oldest contiguous prefix into a `KindCompaction` marker via `Store.Compact`,
   then re-check.

Policy is per-model:

```go
type ShaperPolicy struct {
    BudgetTokens          int
    PreserveLastMessages  int
    PreserveLastToolCalls int
    LODTruncateAboveChars int
}
```

Token estimation is a pluggable `TokenEstimator` (default: a conservative
chars/4 heuristic). `agent.Budget(contextTokens, reservePct)` computes a
budget that leaves room for the response.

## Where orchestration stops

agentkit gives you a `Session` and the loop. It does not decide *which* session
runs, *when*, or *why*, nor does it model roles or a task graph. That's your
harness. The line is intentional: a client that couldn't run a tool loop would
force every host to re-wrap one; a client that owned scheduling would force
every host to adopt its orchestration model.
