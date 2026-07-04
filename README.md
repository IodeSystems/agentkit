# agentkit

A **batteries-included** Go client for OpenAI-compatible endpoints.

Most "LLM clients" hand you a `chat()` call and stop. agentkit owns the
**tablestakes** every real agent client re-implements anyway — so you don't
have to wrap them yourself:

- the tool-call **loop** — chat → tool_calls → execute → feed back → repeat
- context **compaction + LOD truncation** to fit the window (the Shaper)
- message / notification **injection** into an in-flight conversation
- queued-message **batching** — N arrivals coalesce into one turn
- **lifting** — async tool results (a slow tool parks the turn, resumes later)
- schema **validation** with a fix loop, and server-side **constrained decoding**
- notification **lifecycle** — supersede, clear, and pre-turn revalidation
- a fair-share **429/backpressure** retry that honors the server's `retry_after`

What agentkit does **not** own is **orchestration** — roles, a task DAG,
scheduling, when/why to run. That's your harness's job. agentkit gives it a
`Session` to drive and a few small interfaces to implement.

```
go get github.com/iodesystems/agentkit
```

Requires Go 1.26+.

## Packages

| package  | what it is | dependencies |
|----------|------------|--------------|
| `llm`    | the streaming OpenAI-compatible chat client (tools, tool_choice, grammar, response_format, 429 retry) | stdlib only |
| `mcpmgr` | MCP server manager — spawn stdio MCP servers, discover tools, call them, per-thread scoping, secret files | `mark3labs/mcp-go` |
| `agent`  | the tablestakes: `Session.Turn` loop, `Shaper`, injection, lifting, validation, notification lifecycle | `llm` only |

`llm` and `mcpmgr` are deliberately zero-internal-dep — use them standalone.
`agent` imports only `llm` + stdlib; it never sees your storage model.

## Quickstart — a tool-call loop in ~20 lines

```go
client := llm.NewClient("https://llm.iodesystems.com", os.Getenv("AGENTKIT_API_KEY"), "Qwen3-6-27B-MPT")

store := newMyStore()                 // you implement agent.Store (6 methods)
store.Append(ctx, "s1", agent.Entry{Kind: agent.KindUser, Content: "Weather in Denver?"})

sess := &agent.Session{
    SessionID: "s1",
    System:    "You are a helpful assistant. Use tools when they help.",
    Store:     store,
    Runner:    client,                // *llm.Client satisfies agent.LLMRunner
    Tools:     myTools,               // []llm.ToolDef
    Dispatch:  myDispatch,            // func(ctx, llm.ToolCall) (string, error)
    OnAssistantToken: func(s string) { fmt.Print(s) },
}

res, err := sess.Turn(ctx)            // res.Reply + res.Compactions + res.Usage{Total, Active}
```

That `Turn` call streams the completion, dispatches every tool the model
requests, feeds the results back, re-prompts, and returns when the model stops
calling tools — batching any messages that queued in the meantime, shaping the
context to fit the window, and reporting any compaction + the token tally.

## Runnable example

[`examples/agentkit-demo`](examples/agentkit-demo) is a CLI with one subcommand
per feature, wired to iode's **corrallm** server at `llm.iodesystems.com`:

```
go run ./examples/agentkit-demo chat       # streaming + 429 backpressure retry
go run ./examples/agentkit-demo tools      # local Go tool-call loop
go run ./examples/agentkit-demo schema     # client-side validation + fix loop
go run ./examples/agentkit-demo grammar    # server-side constrained decoding
go run ./examples/agentkit-demo inject     # notification injection + batching
go run ./examples/agentkit-demo lift       # async tool results
go run ./examples/agentkit-demo notify     # supersede / clear / preparer
go run ./examples/agentkit-demo compact    # LOD truncation + compaction
```

The `schema`, `inject`, `lift`, `notify`, and `compact` demos are fully offline
(they exercise the mechanics without the model). `chat`, `tools`, and `grammar`
hit the live model — which is small and **often busy**. A 429 there is a
feature demo, not a bug: the client honors the server's `retry_after` and keeps
trying until `--timeout`.

## Docs

- [docs/concepts.md](docs/concepts.md) — the core model: `Store`, `Entry`, `Session`, `Shaper`, the neutral seam.
- [docs/features.md](docs/features.md) — every tablestake, with code and the demo that shows it.
- [docs/provider.md](docs/provider.md) — connecting to corrallm / any OpenAI-compatible endpoint; 429, grammar, `response_format`, api-key-as-priority.

## Status

Pre-release. Module path `github.com/iodesystems/agentkit` is final. Extracted
from autowork3 (its first consumer); the interface is deliberately host-neutral
so a second consumer can implement `Store` over its own storage.

## License

[MIT](LICENSE) © IodeSystems
