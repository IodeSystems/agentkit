# agentkit-demo

A CLI showcasing every major agentkit feature against an OpenAI-compatible
endpoint (default: iode's corrallm at `llm.iodesystems.com`).

```
go run ./examples/agentkit-demo <command> [flags]
```

| command | shows | needs the model? |
|---------|-------|------------------|
| `chat`       | streaming chat + fair-share 429 backpressure retry | yes |
| `tools`      | local Go tool-call loop (`get_weather`, `add`) | yes |
| `mcp`        | MCP server integration: spawn, discover tools, call them | yes + npx |
| `schema`     | client-side schema validation + fix loop | no |
| `structured` | reliable typed extraction: tool_choice + validator + terminal tool | yes |
| `grammar`    | server-side constrained decoding (GBNF + json) | yes |
| `inject`     | notification injection + queued-message batching | no |
| `lift`       | async tool results (lifting) | no |
| `notify`     | notification lifecycle: supersede / clear / preparer | no |
| `compact`    | context shaping: LOD truncation + compaction | no |

Offline demos exercise the mechanics deterministically (no network). The live
demos hit the shared, capacity-limited model — a **429 is a feature demo**: the
client honors the server's `retry_after` and retries until `--timeout`.

Add `--trace` to any Session-driven demo (`tools`, `mcp`, `structured`) to print
the loop's spans (Turn → streamChat, Shaper.Build) with timings — the same
`agent.Tracer` seam a host wires to OpenTelemetry.

`mcp` spawns the MCP reference "everything" server via `npx -y
@modelcontextprotocol/server-everything` (first run downloads it). Point it at a
different server with `--prompt "<command> <args…>"`.

## Flags / env

```
--base-url   OpenAI-compatible base URL   ($AGENTKIT_BASE_URL, default https://llm.iodesystems.com)
--model      model id                     ($AGENTKIT_MODEL, default Qwen3-6-27B-MPT)
--api-key    API key = scheduling priority ($AGENTKIT_API_KEY)
--timeout    deadline for a live call (429 retries count against it; default 120s)
--prompt     override the command's default prompt
```

## Files worth reading

- `store.go` — a complete in-memory `agent.Store` (+ inbox/notice helpers). Copy
  it as the starting point for a real host's store.
- `tools.go` — hand-written local tools + a dispatcher, and how errors-for-the-model
  differ from errors-that-abort.
- `mcp.go` — the entire MCP integration: `mcpToolDefs` (advertise) + `mcpDispatcher`
  (route to `Manager.CallTool`). mcpmgr owns spawn/discovery.
- `trace.go` — a ~40-line `agent.Tracer` implementation (`--trace`).
- `demos.go` — each feature end to end.
