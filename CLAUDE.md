# CLAUDE.md — agentkit

Guidance for agents working in this repo. (User/global conventions still apply.)

## What this is

`github.com/iodesystems/agentkit` — a batteries-included Go client for
OpenAI-compatible endpoints. It owns the tablestakes every agent client needs
(tool loop, compaction, injection, lifting, batching, validation, notification
lifecycle) but **NOT** orchestration (roles, task DAG, scheduling). A host
(autowork3 is the first consumer) drives an `agent.Session` and implements the
small interfaces. See `README.md` + `docs/` for the external story.

## Layout

```
llm/      OpenAI-compatible streaming chat client. stdlib only.
mcpmgr/   MCP server manager (spawn/discover/call, thread scoping, secrets).
          Own JSON-RPC stdio client (jsonrpc.go + protocol.go). stdlib only.
agent/    the tablestakes engine: Session.Turn loop + Shaper + primitives. imports llm + stdlib only.
examples/agentkit-demo/  runnable CLI, one subcommand per feature.
plan/     living plan. Only plan.md exists today; it carries its own Icebox
          section rather than the separate done.md / icebox.md the global
          convention describes.
```

## Invariants — do not break

0. **The module has NO third-party dependencies.** `go.mod` declares a module
   path and a Go version, nothing else. Verify with `go list -m all` — one
   line. Adding a `require` is a decision, not a convenience: this library sits
   under a host's tree where every module it pulls in is one the host cannot
   refuse, and it runs a loop that executes tool calls. Reach for stdlib, or
   write the ~50 lines. (What that bought so far: mark3labs/mcp-go → `mcpmgr`'s
   own JSON-RPC client; google/uuid → `agent.NewID`; gopkg.in/yaml.v3 → deleted
   the unused `yaml` secret format.)
1. **`llm` and `mcpmgr` stay zero-internal-dep.** They must be usable
   standalone. Verify with `go list -deps ./llm ./mcpmgr` — no `agentkit/agent`.
2. **The neutral seam.** `agent` imports ONLY `agentkit/llm`, `agentkit/mcpmgr`,
   and stdlib. It must NEVER import a host's storage/event model. If a host type
   feels necessary, add a field to `Entry` (round-tripped verbatim, like `Tag`
   and `Origin`) or a method to `Store` instead — never a concrete host import.
3. **`agent` works on two shapes only:** `llm.Message` on the wire, `Entry` for
   persistence/shaping. The host maps its own rows onto `Entry` and back.
4. **Pre-release: no compat shims.** Edit in place, delete freely, no deprecation
   dance. The module path is final; nothing else is.

## The contract (what a host implements)

- `agent.Store` — 4 methods: `ClaimPending`, `Append`, `Context`, `Compact`
  (+ the host's own inbox/publish helpers, not part of the interface).
- `agent.ToolDispatcher` — `func(ctx, llm.ToolCall) (string, error)`. Errors
  meant for the MODEL go INTO the result string; a returned Go error aborts the
  whole Turn. Return `agent.ErrSessionClosed` from a terminal tool to stop the loop.
  It is a func type, so DECORATING it is the extension seam — `ValidatingDispatcher`
  is one, and dun's `withLiftedQueue` (fold buffered news into the in-flight
  tool's result) is another. Reach for a wrapper before a new primitive.
- `agent.LLMRunner` — one streaming round-trip; `*llm.Client` already satisfies it.
- `Session.OnToolCalls func([]llm.ToolCall) []llm.ToolCall` — optional hook to
  rewrite the model's tool calls before they are persisted and dispatched. Hosts
  use it to INJECT calls of their own so they land as a proper
  assistant(tool_calls) → tool(result) pair rather than a disembodied notice.
- `agent.NewID()` — canonical RFC 4122 v4, for hosts that don't mint their own
  Entry IDs. Format is the compatibility surface; see `agent/id.go`.
- optional: `agent.Validator`, `agent.NotificationPreparer`, `agent.Tracer`,
  `agent.RevalidateStore` + `agent.Revalidator` (the MCP-revalidator convention).

## Key mechanics (so you don't re-derive them)

- **`mcpmgr` owns its MCP client.** `jsonrpc.go` = subprocess + newline-framed
  JSON-RPC 2.0 + id correlation + the teardown ladder (close stdin → 2s →
  SIGTERM → 3s → SIGKILL → 3s). `protocol.go` = the four methods we speak
  (initialize, notifications/initialized, tools/list, tools/call) + schema
  normalization + content rendering. This replaced mark3labs/mcp-go, of which
  we used seven methods: 22.4k LOC of its packages were compiled in, 68k with
  the transitive modules that came along (a JSON Schema validator, a
  URI-template engine, ~20k of `golang.org/x/text` to translate errors from a
  validator we never called). Three things are load-bearing and easy to undo
  by accident: (1) `exec.Command`, NOT
  `CommandContext` — binding the child to a caller's ctx meant a `defer
  cancel()` killed the server; (2) `bufio.Reader.ReadString`, never a
  `Scanner` — one frame is one line and a tool result is routinely megabytes;
  (3) `close()`'s cleanup runs under `cleanupOnce` regardless of who closed
  `done` first, or a server that died on its own leaks FDs and a zombie.
- **Tool `inputSchema` is round-tripped VERBATIM** as `map[string]any`.
  `normalizeSchema` only fills nil `type`/`properties`/`required` (a null
  `required` makes llama.cpp reject the whole chat request) and copies
  everything else through untouched. Decoding it into a struct — what mcp-go
  did — silently deleted `$defs`, `$schema`, `title`, `additionalProperties`
  before any caller saw them. `TestToolSchemaKeepsFullFidelity` guards this.
- **`mcpmgr` tests need no external binary.** `fakeserver_test.go` re-execs the
  TEST BINARY as an MCP server (`MCPMGR_FAKE_SERVER=<scenario>`), so the
  protocol layer has real coverage without a cross-repo build. Scenarios:
  happy / notify / stderr-die / hang / bad-init / slow-tools. The older
  `poly-lsp-mcp` tests still run when that binary is on PATH and are the
  real-server check.
- **Batching is inherent in `Turn`:** `ClaimPending` marks ALL pending arrivals
  shown at the top of an iteration, and `build()` renders every non-subsumed
  entry — so N queued messages are seen in ONE turn.
- **Lifting is NOT a goroutine park.** A tool returns `{"pending":true,...}`;
  the dispatcher rewrites it to `PendingResult`, the turn ends, the session
  stays active, and the host re-injects the real result as a `KindToolResult`
  Entry keyed by the same `ToolCallID`. Storage + deadline GC stay host-side.
- **Validation is a soft fix loop.** `ValidatingDispatcher` returns the fix
  instruction as a (non-error) tool result → session stays active → model
  retries. It never calls the inner dispatcher on a bad call.
- **Constrained decoding is server-side + separate.** `ChatOpts.Grammar`
  (GBNF) and `ChatOpts.ResponseFormat` are forwarded raw to the server (hard
  guarantee). The Validator is the client-side (soft) counterpart. Different
  tools; document both.
- **Loop guards are ON by default, at two levels.** `llm.RepetitionGuard`
  watches each stream (content AND tool-call arguments — the measured 30-minute
  loop was inside `arguments`) for a periodic suffix and CLOSES THE BODY; the
  disconnect is what makes the server abandon the generation. `MinPeriod` is
  tested against the block's MINIMAL period — skip that reduction and it is
  decorative, since every multiple of a period is also a period. `agent` treats
  a cut as recoverable: trim the redundant copies BEFORE persisting (or the
  retry is primed to continue the loop), notify, re-prompt, bounded by
  `MaxRepetitionRetries`. Separately `MaxRepeatedExchanges` catches the
  turn-level loop; its signature includes the RESULTS so legitimate polling
  never trips it.
- **429 retry honors `retry_after` from header OR body, bounded by a budget.**
  corrallm's fair-share proxy returns the hint in a JSON backpressure body
  (`{"error":{"reason":"queue-timeout","retry_after":10}}`), not the
  `Retry-After` header. `retryAfterFrom` checks header first, then body. Retry
  is NOT infinite: `Client.RetryBudget` (default `defaultRetryBudget` = 5m)
  caps total wall-clock spent retrying 429/5xx; ctx deadline still wins if
  shorter. The api key is the scheduling IDENTITY (Authorization Bearer →
  priority), not the trace header.
- **Every retry sleep is jittered, additively.** `jittered()` adds a random
  `[0, retryJitterFraction)` of the sleep — never subtracts. corrallm hands
  every caller rejected in the same moment the IDENTICAL `Retry-After`, and an
  exact sleep turns the schedule into a synchronization primitive: N callers
  wake together and each re-POSTs a FULL payload (`postWithRetry` resends the
  same bytes every attempt) at the worst possible instant. Subtractive jitter is
  wrong here — a draw below `Retry-After` is a guaranteed second 429. The slack
  is drawn from the CLEAN `backoff` (so it never compounds with the doubling)
  and BEFORE the deadline check (so the budget measures the real sleep). An
  outer retrier built on `RetryPolicy()` gets the unjittered numbers and should
  jitter them itself.

## Build / test guardrail (run after every change)

```
go build ./... && go vet ./... && go test ./...
```

The example is part of the module — `go build ./...` covers it. Offline demos
(`schema`, `inject`, `lift`, `notify`, `compact`) run without the model and are
the quickest smoke test of the primitives:

```
go run ./examples/agentkit-demo notify
```

## Conventions

- Match the existing house style: heavy doc comments explaining WHY (the
  tradeoff), not just what. Test names describe the failure mode they guard.
- New primitives go in a focused file (`lift.go`, `validate.go`, `clear.go`,
  `prepare.go`) with a top-of-file doc block framing the feature.
- Keep the consumers in mind — **autowork3** (first; drives `Session` over
  Postgres event streams) and **dun** (a coding-agent harness; `../dun`, uses a
  `replace` directive). Additive changes only unless the plan says otherwise.
  Behavior changes to shared paths need a call-out. dun is the useful second
  opinion on the contract: where it had to write a wrapper, the seam was
  probably in the right place; where it had to reach around one, it wasn't.
