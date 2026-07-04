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
mcpmgr/   MCP server manager (spawn/discover/call, thread scoping, secrets). dep: mark3labs/mcp-go.
agent/    the tablestakes engine: Session.Turn loop + Shaper + primitives. imports llm + stdlib only.
examples/agentkit-demo/  runnable CLI, one subcommand per feature.
plan/     living plan (plan.md = active, done.md = archive, icebox.md = deferred).
```

## Invariants — do not break

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

- `agent.Store` — 6 methods: `ClaimPending`, `Append`, `Context`, `Compact`
  (+ the host's own inbox/publish helpers, not part of the interface).
- `agent.ToolDispatcher` — `func(ctx, llm.ToolCall) (string, error)`. Errors
  meant for the MODEL go INTO the result string; a returned Go error aborts the
  whole Turn. Return `agent.ErrSessionClosed` from a terminal tool to stop the loop.
- `agent.LLMRunner` — one streaming round-trip; `*llm.Client` already satisfies it.
- optional: `agent.Validator`, `agent.NotificationPreparer`, `agent.Tracer`,
  `agent.RevalidateStore` + `agent.Revalidator` (the MCP-revalidator convention).

## Key mechanics (so you don't re-derive them)

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
- **429 retry honors `retry_after` from header OR body, bounded by a budget.**
  corrallm's fair-share proxy returns the hint in a JSON backpressure body
  (`{"error":{"reason":"queue-timeout","retry_after":10}}`), not the
  `Retry-After` header. `retryAfterFrom` checks header first, then body. Retry
  is NOT infinite: `Client.RetryBudget` (default `defaultRetryBudget` = 5m)
  caps total wall-clock spent retrying 429/5xx; ctx deadline still wins if
  shorter. The api key is the scheduling IDENTITY (Authorization Bearer →
  priority), not the trace header.

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
- Keep autowork3 (the consumer) in mind: additive changes only unless the plan
  says otherwise. Behavior changes to shared paths need a call-out.
