# agentkit — plan

> How this plan works: current state + active work + decisions ONLY.
> Completed trees move to `plan/done.md` (one-line pointer left behind).
> Deferred/opt-in next-steps move to `plan/icebox.md`. Status marks:
> ◻ todo · ◐ in progress · ✅ done · ⏸ parked · ❓ blocked.

## What this is

`github.com/iodesystems/agentkit` — a **batteries-included** Go client for
OpenAI-compatible endpoints, extracted from autowork3. It owns the
TABLESTAKES every real agent client needs, so consumers don't re-wrap them:

- the tool-call **loop** (chat → tool_calls → exec → feed back → repeat)
- context **compaction** + **LOD** truncation (the Shaper)
- message/notification **injection** into an in-flight conversation
- **lifting**: async tool results (a tool parks the turn, resumes on resolve)
- queued-message **batching** (coalesce inbox into one turn)
- grammar / JSON-Schema **validation** with a retry/fix loop

It does NOT own **orchestration** (roles, task DAG, scheduling, when/why to
run). That's the harness's job. autowork3 is one consumer; the user's
separate **claude-openai project** is the intended second consumer (validate
the interface against it — path currently unknown, find it).

Design rationale (user's words): "a client without a tool loop is dumb,
because then everyone would have to wrap that... these are tablestakes for
any client implementation. the harness is the orchestration, not the basic
trousers of going out in public."

## Consumer / layout

```
agentkit/
  llm/      ✅ moved from autowork3/internal/llm  — zero internal deps
  mcpmgr/   ✅ moved from autowork3/internal/mcpmgr — zero internal deps
  agent/    ✅ tool loop + shaper (Turn/Shaper/Store/Entry/Tracer) — slice 2 done
            ◐ slice 3: lifting + injection/batching + Validator still to wire
```

autowork3 consumes via `replace github.com/iodesystems/agentkit => ../../agentkit`
in `autowork3/go.mod`. NOT yet a git repo, NOT yet published — module path is
final though.

## Active work

### ✅ Slice 1 — extraction + neutral contract
- `llm` + `mcpmgr` moved into agentkit; all 22 autowork3 import sites
  rewritten to `github.com/iodesystems/agentkit/{llm,mcpmgr}`; both modules
  build + test green (autowork3 server suite ~44s).
- `agent/agent.go` = the CONTRACT (types + method sigs + doc comments; the
  two methods `panic("not yet implemented")`). Compiles.
- **Contract surface** (`agent/agent.go`) — the 5 load-bearing decisions:
  1. Neutral `Store`/`Entry` model — agent works on `llm.Message` (wire) +
     `Entry` (persist/shape); never a host's event types. Host maps its rows
     onto `Entry`. `EntryKind` ∈ {user, assistant, tool_call, tool_result,
     compaction}.
  2. `ToolExecutor func(ctx, llm.ToolCall) (result string, lift *Lift, err)`
     — inline result, OR non-nil `*Lift` to park the turn (async lifting),
     resumed when host feeds a tool_result Entry keyed by `ToolCallID`.
     `Lift.Deadline` (ns) so a stuck tool can't park forever.
  3. `Store.Claim(ctx, sessionID) []Entry` = the inbox → BATCHED into one
     Turn (injection + queued-batching, not one-turn-each).
  4. `Validator interface { Validate(reply string) error }` — grammar/schema
     gate; error message is the fix instruction fed back (the fix loop).
  5. `Session{ID,Client,Store,Exec,Policy,System,Tools,ChatOpts,Validator,
     MaxToolRounds}` with `Turn(ctx)(reply,err)` (the loop) + `Inject(ctx,
     content)` (append to inbox). `ShaperPolicy{BudgetTokens,
     PreserveLastMessages, PreserveLastToolCalls, LODTruncateAboveChars}`.
- **BLOCKING DECISION — RESOLVED.** User green-lit ("lets go — we compacted,
  so lets hit it"). Contract shipped as-is; `Lift`/`Validator`/`Inject`/
  `ToolExecutor` deferred to slice 3 (not in code yet — the working tool seam
  is `ToolDispatcher`, matching the ported impl).

### ✅ Slice 2 — Turn loop + Shaper moved into agentkit
- Ported `harness/session.go` (Turn loop, streamChat, DefaultContextBuilder)
  + `harness/shaper.go` (pristine-tail + LOD + compaction) into `agent/`,
  retargeted from `events` types onto neutral `Entry`/`Store`/`Compaction`.
  Files: `agent/{agent,session,shaper,render,tokens,tracer}.go`.
- **Neutral seam realized:**
  - `Entry` gained `Tag` (opaque display label for notification render) +
    `Origin` (opaque host provenance — round-tripped through `Compaction.
    Subsumes` so the host routes subsumed rows). agent interprets neither.
  - `Store` = `{ClaimPending(→count), Append, Context, Compact}`. The host's
    two-stream merge (deliveries vs private log) collapses into the adapter's
    `Context()`; `Compact` re-splits by `Origin`.
  - `Tracer`/`Span` = tiny nil-safe interface; `SpanPrefix` field (default
    "agent") lets the host keep span labels — autowork3 sets "harness" so
    `harness.Turn`/`.streamChat`/`.Shaper.Build` stay stable for aw log / aw
    cost / api.go LastContext.
- **autowork3 = thin adapter now:** `harness/session.go` holds
  `eventStoreAdapter` (events⇄Entry, incl. marker metadata reconstruction) +
  `traceAdapter`/`spanAdapter`, plus `Session`/`Shaper` wrapper structs
  (native fields) that delegate Turn/Build. `EventStore`/`ShaperCompactor`
  interfaces stay; `LLMRunner`/`ToolDispatcher`/`ContextBuilder`/
  `TokenEstimator`/`ShaperPolicy` are now aliases to `agent.*`; `Default`/
  `Budget` delegate. **Zero churn at the 3 call sites** (scheduler/reviewer/
  api) — same struct literals.
- **Tests:** `TestClassifyPristineCount` + `TestLodStub` moved to
  `agent/shaper_test.go` (rewritten vs `Entry`). All other harness tests
  unchanged — they exercise the wrapper end-to-end (the adapter guardrail).
- **Green:** `agentkit go test ./...` ✅; autowork3 `go build ./...` ✅,
  `go test -count=1 ./internal/harness/... ./internal/server/...` ✅ (server
  42.7s), full `go test ./...` ✅.

### ✅ Slice 3 — remaining tablestakes wired
- **Key reframing:** autowork3's lifting is NOT a goroutine-park — the
  dispatcher rewrites a `{"pending":true,...}` result to "wrap up", the
  session stays active, and `EventLiftResult` re-wakes it via the inbox. And
  BATCHING is already inherent in the Turn loop (claim-all → build → one
  turn). So slice 3 was NOT Turn surgery — it was making these reusable +
  adding the one genuinely-missing piece (schema validation).
- **Added to `agent` (pure additions, zero autowork3 behavior change):**
  - `lift.go` — `LiftRequest` + `ParseLiftRequest` + `PendingResult`: the
    async-tool wire shape + canonical "pending, wrap up" message as a shared
    primitive. Host keeps the pending table + redelivery + deadline GC.
  - `validate.go` — `Validator` iface + `SchemaValidator` (dependency-free:
    required-keys + JSON-kind checks from the tool's own schema) +
    `ValidatingDispatcher` (validates args pre-dispatch; on failure returns
    the fix instruction as a soft result → session stays active → model
    retries = the fix loop). Drops into the existing loop, no Turn surgery.
  - `Session.Inject(ctx, Entry)` — self-inbox injection primitive; +doc that
    batching is inherent.
  - Tests: `slice3_test.go` (lift/validate/batching/inject) + `mem_test.go`
    (in-memory Store + scripted runner).
- **Dogfooded in autowork3:** `server/pending_lifts.go::maybeOpenLift` now
  uses `agent.ParseLiftRequest` + `agent.PendingResult` (wording identical →
  behavior-preserving). Proves the primitive against the real consumer.
- **Green:** agentkit `go test ./...` ✅; autowork3 `go build` ✅,
  `go test -count=1 ./internal/server/...` ✅ (41.6s).
- **NOT adopted by autowork3 (deliberate):** the generic `SchemaValidator` /
  `ValidatingDispatcher` — autowork3 keeps its hand-rolled per-tool rejection
  (works; adopting would be behavior-changing). Primitives exist for the 2nd
  consumer + optional future adoption.

### ✅ Slice 4 — notification lifecycle: supersede + clear + stale-waiter
- **Problem (user, real pain):** MCP LSP emits many notifications; stale
  in-flight ones cost the model a re-validation every Turn. Merge folds
  repeats but NEVER removes and can't catch staleness. User picked **both
  tiers + both clear signals**.
- **Delivered (all keyed on the existing `(type, groupByKey)` partition):**
  - **Supersede** — `MergeSpec.Strategy="replace"`: a re-emitted notice for
    the same key overwrites the prior unshown one wholesale (content AND
    metadata — so TTL/correlation refresh). Inbox holds ≤1 live notice/key.
  - **Explicit clear** — `Store.ClearDeliveries(thread, session, type,
    groupBy, key)` deletes unshown matching notices (session-scoped or
    thread-wide). Integration emits it on resolve. Zero model cost.
  - **Stale-waiter (TTL)** — `ttl_s` on a notice → `stale_after` in metadata;
    scheduler tick pass `Store.ClearStaleDeliveries(now)` retracts expired
    unshown notices (integration went quiet/died). Mirrors the pending-lift
    deadline pass.
  - **Endpoint:** `IngestNotification` gained `group_by` (supersede),
    `clear` (retract, target="clear", returns count), `ttl_s`.
  - **agentkit:** `agent/clear.go` — `ClearRequest` + `ParseClearRequest`
    wire primitive (a tool result / integration callback can retract).
- **Files:** `internal/events/{store,postgres,memory,merge}.go`,
  `internal/server/{notifications,scheduler}.go`, `agentkit/agent/clear.go`.
  No schema migration needed — `stale_after` rides in existing JSONB
  metadata (sentinel-style, per the codebase ethos).
- **Tests:** `events/lifecycle_test.go` (supersede, clear, thread-scope,
  stale + refresh-on-supersede), `server/notifications_test.go` (endpoint
  supersede+clear), `agentkit` clear parse. **Full autowork3 suite +
  agentkit green.**
### ✅ Slice 5a — active-revalidation seam (agentkit side)
- **Reframing (user):** the deferred active re-check should NOT be daemon
  shell (privilege-escalation surface). It belongs at a
  `prepareNotificationsBeforeSend` HOOK on the integrator side. And since MCP
  has no notification API, wire MCP revalidation via a masked-tool CONVENTION.
  User: "a mcp revalidator convention is nice, but so is the hook" → ship
  BOTH: the hook is the base seam, the convention is one preparer on it.
- **Delivered in `agent/prepare.go` (+ `session.go` wiring):**
  - `NotificationPreparer` hook + `Session.Preparer`, invoked at the top of
    every Turn iteration (after ClaimPending, before build) so a resolved
    notice is cleared before it's rendered. Any integrator plugs in a Go
    preparer (HTTP, direct query, in-process shell — whatever it trusts).
  - `MCPPreparer(RevalidateStore, Revalidator)` — the ready-made preparer for
    the masked-MCP convention: per pending notice, call the integration's
    designated tool for the group key; **empty "current truth" → clear**
    (ratified). Fail-open on a flaky tool.
  - `IsResolvedTruth` (empty/null/{}/[]/"" = resolved). Two host interfaces
    (`RevalidateStore`, `Revalidator`) keep the convention logic tested in
    agentkit; the host implements the store + tool-call halves.
  - `harness.Session.Preparer` passthrough added (nil today).
- **Tests:** `prepare_test.go` (IsResolvedTruth, MCPPreparer clears-resolved/
  keeps-live/keeps-unconfigured) + `slice3_test.go` (hook runs before send &
  drops a stale notice). agentkit green; autowork3 builds.
- **Ratified decisions (for 5b):** designation = explicit `revalidators`
  map `{"<groupBy>": "<toolName>"}` on the MCP integration config; result
  shape = return current truth, empty = clear.

### ◻ Slice 5b — autowork3 MCP-convention wiring (NEXT)
- **next:** implement the concrete consumer:
  1. store `group_by` in notification metadata on publish (preparer needs each
     notice's key field) — small `IngestNotification` edit.
  2. `RevalidateStore` impl: `PendingNotices` = unshown notification
     deliveries for the session, parse metadata.group_by + content[group_by]
     → key; `Clear` → `events.ClearDeliveries`. (needs a per-session unshown-
     notification read — extend events or reuse ListContextDeliveries filtered
     to unshown notices.)
  3. `Revalidator` impl: parse `revalidators` from the type=mcp integration
     config (a raw-JSON extra key alongside the mcpmgr.MCPConfig fields — do
     NOT push notification concepts into agentkit/mcpmgr); resolve the
     thread's MCP server id; `s.mcpMgr.CallTool(serverID, tool, {key})` (use
     the `toolCaller` seam for tests).
  4. mask revalidator tool names out of `availableTools`/`collectTools`.
  5. wire `Preparer: agent.MCPPreparer(store, rv)` into the scheduler's
     `harness.Session` construction (scheduler.go ~1378).
- **risks:** per-Turn MCP calls add latency — only iterate notices that carry
  a group_by + have a configured revalidator; fail-open. Keep mcpmgr
  notification-agnostic (revalidators is autowork3 config, parsed host-side).
- **DEFERRED still:** daemon-side arbitrary-shell recheck stays out (the hook
  + MCP convention replace the need). → icebox if ever revisited.

### ◻ Slice 3 — wire the remaining tablestakes
- **lifting:** autowork3's `pending_lifts` mechanism → `agent.Lift` (park +
  resume on tool_result Entry).
- **injection/batching:** autowork3's inbox (`ListContextDeliveries` /
  session deliveries) → `Store.Claim`.
- **grammar/schema Validator:** the existing dispatcher-enforcement retry
  (protocol-bound roles under `tool_choice=required`) → `agent.Validator`
  fix loop. JSON-Schema validation + fix loop for structured tool outputs.

### ◻ Slice 4 — collapse the harness to a thin adapter
- Delete the now-moved loop/shaper code from autowork3; `harness` package
  becomes orchestration-only (roles, scheduler, provider lanes stay in
  autowork3). Verify full suite green.
- Then: point the **claude-openai project** + a future **openai-session-
  source** at `agentkit/agent` to prove the interface against a 2nd consumer.

## Decisions / conventions
- Module path `github.com/iodesystems/agentkit` is FINAL.
- `llm` + `mcpmgr` MUST stay zero-internal-dep (verified via `go list -deps`).
- Neutral seam rule: `agent` imports only `agentkit/llm`, `agentkit/mcpmgr`,
  stdlib. It must NEVER import an `events`/host model. If a host type is
  tempting, add a field to `Entry` or a method to `Store` instead.
- Pre-release everywhere: no compat shims, edit in place, delete freely.

## How to re-pick-up
1. Read this file. Slices 1–2 done; **Slice 3 is next** (lifting +
   injection/batching + Validator).
2. Engine lives in `agent/{agent,session,shaper,render,tokens,tracer}.go`;
   autowork3 adapter in `autowork3/internal/harness/{session,shaper,tokens}.go`.
3. For slice-3 lifting, study autowork3's `EventLiftResult` / `job_output`
   flow + `pending_lifts` + scheduler timeout synthesis before designing the
   park/resume seam. Keep park/resume inside `agent`; host only feeds results.
4. Guardrail after every step: `cd agentkit && go build ./... && go test ./...`
   AND `cd autowork3 && go build ./... && go test -count=1 ./internal/harness/...
   ./internal/server/...`.
5. Related autowork3 memory: [[migration-in-place-db-drift]] (unrelated to
   agentkit, but the live daemon runs against the drifted dev DB).
