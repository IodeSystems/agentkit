# CLAUDE.md — agentkit

Guidance for agents working in this repo. (User/global conventions still apply.)

## What this is

`github.com/iodesystems/agentkit` — a batteries-included Go client for
OpenAI-compatible endpoints. It owns the tablestakes every agent client needs
(tool loop, compaction, injection, lifting, batching, validation, notification
lifecycle) but **NOT** orchestration (roles, the host's task DAG, scheduling:
which session runs, when, why). A plan the MODEL keeps for itself is not
orchestration and lives in the optional `taskgraph` (Carl, 2026-11-03). A host
(autowork3 is the first consumer) drives an `agent.Session` and implements the
small interfaces. See `README.md` + `docs/` for the external story.

## Layout

```
llm/      OpenAI-compatible streaming chat client; the Claude Messages API goes
          through openai-claude-api-shim (ours, stdlib only). Multimodal user
          parts: text, image, audio, video, file.
mcpmgr/   MCP server manager (spawn/discover/call, thread scoping, secrets).
          Own JSON-RPC stdio client (jsonrpc.go + protocol.go). stdlib only.
agent/    the tablestakes engine: Session.Turn loop + Shaper + primitives. imports llm + stdlib only.
files/    mountable file toolset (read/edit/write/grep/list, opt-in delete/move) for a
          host's tool list: Defs() + Wrap(dispatcher). imports llm + stdlib only;
          shells out to rg/git when on PATH, works without either.
loopdetector/  opt-in loop detection from session + stream: a count and a tolerance.
          imports llm + stdlib only. Hosts install it; agent does not import it.
taskgraph/  opt-in model-kept plan: DAG of nodes, the four `plan_*` tools (`taskgraph.New`: Defs/Wrap/Dispatch), and Diff/Apply
          (what a write changed, which owners to tell). Shared by a main agent and
          its sub-agents through ONE host-provided Store (no impl shipped). Owner is
          opaque; permission, approval, delegation and delivery are host hooks.
          imports llm + stdlib only; agent does not import it. Ported from dun.
mcpmgr/mcpauth/  OAuth discovery for MCP servers (RFC 9728 / 8414). Used by dun.
agent/toolfmt/   token-lean re-encoding of tool results (EncodeTightC).
cmd/toolfmt/     stdin→stdout CLI over agent/toolfmt (tightc's conformance test).
ragnotify/       parses a retrieval MCP server's hits into []agent.DocHit (ParseHits).
anthropicadmin/  an Anthropic org's real spend, by Admin key or `ant` login. Used by dun.
examples/agentkit-demo/  runnable CLI, one subcommand per feature.
plan/     living plan. plan.md = current state + active work + decisions;
          done.md = archive of finished work; icebox.md = deferred, opt-in
          steps (each with a resume condition); claude-api-plan.md = gaps
          between the OpenAI shape and the Claude Messages API.
```

## Invariants — do not break

0. **The module has NO third-party dependencies.** `go.mod` declares a module
   path, a Go version and one `require`: openai-claude-api-shim, which is ours and
   standard-library only (Carl, 2026-11-01). Verify with `go list -m all` — two
   lines. Adding a `require` is a decision, not a convenience: this library sits
   under a host's tree where every module it pulls in is one the host cannot
   refuse, and it runs a loop that executes tool calls. Reach for stdlib, or
   write the ~50 lines. (What that bought so far: mark3labs/mcp-go → `mcpmgr`'s
   own JSON-RPC client; google/uuid → `agent.NewID`; gopkg.in/yaml.v3 → deleted
   the unused `yaml` secret format.)
1. **`llm` and `mcpmgr` stay zero-internal-dep.** They must be usable
   standalone. Verify with `go list -deps ./llm ./mcpmgr` — no `agentkit/agent`.
2. **The neutral seam.** `agent` imports ONLY `agentkit/llm` and stdlib.
   It must NEVER import a host's storage/event model. If a host type
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
  meant for the MODEL go INTO the result string; any other returned Go error is
  recorded as an `ERROR: ...` tool result and the loop continues (it does not
  abort the Turn). Return `agent.ErrSessionClosed` from a terminal tool to stop the loop.
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
- **One effort path.** `ChatOpts.Effort` (none..max) → `EffortFor` clamp (record
  refusals, lower level first, never down to `none`) → the dialect's fields.
  `ReasoningBudgetTokens` is a cap: min(level budget, explicit); with no Effort it is sent as before. Refusals (`reasoning.efforts.refused`, `reasoning.off`) bind only dialects that send an effort field; `thinking_only` binds all. Anthropic-compat budgets stay below max_tokens.
  The change is reported via `OnTranslate` with `Key` "effort" and a `Reason`.
  `ReasoningEffort`/`EnableThinking` no longer exist.
- **Mid-turn user words render AFTER the round's tool results.** A host saves a
  pending `KindUser` while a tool runs; `ClaimPending` picks it up next
  iteration. It sorts before the result (`CreatedAt`), so `groupMessages` holds
  every non-assistant/call/result entry while calls are unanswered and emits it
  after the last open result (original index kept for `contentOf`). Order is a
  pure function of the entries, so round N is a prefix of round N+1.
  `RenderOptions.FoldUserIntoToolResult` (`ShaperPolicy` field;
  `DefaultContextBuilderWith` without a Shaper) folds text user entries into
  the last tool result for templates that refuse user-after-tool. Parts stay.
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
- **Cancel never leaves an orphan tool call.** Once ctx is done the dispatch
  loop stops and answers the current and remaining calls with `ERROR: canceled
  before this call ran`; the call and result appends use `context.WithoutCancel` (the host
  canceled the turn, not the history). Strict endpoints refuse a tool call
  with no result. Never use WithoutCancel for dispatch. `Session.UnrunResult`
  (gets the canceled ctx) lets a host word that result, e.g. who stopped the turn.
- **A prefix fold must look like a session round to the cache.**
  `Shaper.SessionPrefix` sends the session's system + tools byte-identically AND
  the session's current `PromptCacheKey` (`promptCacheKey(sessionID, newest
  marker)`, the same derivation as `streamChat`). Drop the key and the fold
  can be routed off the slot that holds the prefix. A fold too big for one
  request sends its FIRST request this way (`prefixCut`: the head that fits
  beside the session's system AND tools, never ending on a tool call); later
  stages carry a running summary, so they use the worker prompt, no tools, no
  key. Every fold request's ctx is marked: `IsFold(ctx)`.
- **A fold keeps the pristine tail, or more with `ShaperPolicy.FoldKeepPct`.**
  `foldKeep` (shaper.go) picks the largest newest suffix whose rebuilt prompt
  (prefix from `tokensBefore` + `SummaryCap(SummaryMaxTokens)` + pinned + suffix,
  rendered WITHOUT LOD) fits that percent of target, widened by
  `keepCallsWithResults`; it falls back to the pristine tail when the band is no
  wider or nothing new would fold. 0 is the old behavior exactly;
  `CompactionInfo.TokensAfter` is the non-LOD render: what the next build sends.
- **Constrained decoding is server-side + separate.** `ChatOpts.Grammar`
  (GBNF) and `ChatOpts.ResponseFormat` are forwarded raw to the server (hard
  guarantee). The Validator is the client-side (soft) counterpart. Different
  tools; document both.
- **A reasoning-only round is repaired once per Turn.** No text + no tool call +
  reasoning (any finish) is checked BEFORE the overflow path (a length cut inside
  a tool call is not empty and stays there). `reasoningcut.go`: a
  `ReasoningCutTag` notification with the reasoning tail, one round with a COPY
  of the opts set off (`reasoningOff`; `s.ChatOpts` is never mutated), no empty
  assistant entry persisted. A second empty round ends `TurnReasoningCut` +
  `*ReasoningCutError`. A round with NO reasoning either (nothing at all, not a
  length cut) shares the same one-repair budget: `EmptyReplyTag` notice, one
  round with the host's opts UNCHANGED, then `TurnEmptyReply` +
  `*EmptyReplyError` (`errors.Is(err, ErrEmptyReply)`; carries finish_reason and
  native_finish_reason). Never `TurnReply` with Reply "". An empty length cut
  stays with the overflow path. `NoReasoningCutRepair` turns off both repairs;
  an empty round then returns the error with no repair round.
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
  priority), not the trace header. A transient `{"error":{...}}`
  SSE event inside a 200 stream is retried by `pumpStream` on the same budget and
  jitter ONLY while nothing was forwarded to the caller (`streamAttempt`'s
  `forwarded` flag; the failed attempt skips OnUsage/observeStream); it is kept
  as `*llm.StreamError` (`StreamChunk.StreamErr`, `chunk.Err()`).
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
