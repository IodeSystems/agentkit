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

### ✅ Slice 5b — autowork3 MCP-convention wiring
- `internal/server/notify_revalidate.go`: `parseRevalidators` (a
  `revalidators` {groupBy:toolName} map on the MCP integration config, parsed
  host-side — mcpmgr stays notification-agnostic); `threadRevalidators` walk;
  `mcpRevalidator` (agent.Revalidator over `mcpMgr.CallTool`); `noticeStore`
  (agent.RevalidateStore over the events substrate — `PendingNotices` filters
  `ListContextDeliveries` to notices with metadata.group_by, `Clear` →
  `ClearDeliveries`); `notificationPreparer` → `agent.MCPPreparer` or nil.
- scheduler: `Session.Preparer` set per session; `attachMCP` masks revalidator
  tool names from the model's surface.
- `IngestNotification` stores `group_by` in metadata.
- **Slice-4 semantics fixed:** `ClearDeliveries`/`ClearStaleDeliveries` now
  retract non-compacted deliveries regardless of `shown_at` — the dominant
  waste is a notice already shown once then re-read every Turn (build renders
  all non-compacted deliveries); the old `shown_at IS NULL` guard only caught
  not-yet-delivered notices.
- Tests: `notify_revalidate_test.go` (parse + end-to-end glue) + flipped
  lifecycle assertion. Committed `646d348`. Full suite + agentkit green.

### ✅ Slice 6 — public-facing docs + runnable example + grammar passthrough
- **Goal (user):** show integrators the value — README, CLAUDE.md, docs, and a
  CLI example against corrallm (`llm.iodesystems.com`) demonstrating every
  major feature (tool loop, injection, local tools, schemas, grammar, …).
- **Verified provider facts:** endpoint live, OpenAI-compatible; chat model
  `Qwen3-6-27B-MPT` (~220k ctx); no auth for reads; api-key = scheduling
  priority. 429 backpressure returns `retry_after` in a JSON BODY
  (`{"error":{"reason":"queue-timeout","retry_after":10}}`), NOT the
  `Retry-After` header. Backend is single-capacity + usually busy.
- **Library changes (additive, tested):**
  - `llm.ChatOpts` gained `Grammar` (GBNF) + `ResponseFormat` (any) → forwarded
    raw into the request body when set; omitted when unset. This is the ONLY
    reason server-side constrained decoding didn't work before — the client
    never sent the fields (backend/config were fine). `tool_choice` already
    shipped.
  - 429/5xx retry now honors `retry_after` from the JSON body too, not just the
    `Retry-After` header (`retryAfterFrom` helper; header wins, body falls
    back, clamped to ceiling). Directly addresses corrallm's backpressure shape.
  - **Retry is now BOUNDED (was infinite-until-ctx).** `Client.RetryBudget`
    (default `defaultRetryBudget` = 5m) caps total wall-clock spent retrying;
    ctx deadline still wins if shorter. Reverses the prior deliberate
    infinite-429 policy per user ("super busy box, exponential retry to a
    limit, timeouts configurable 5m"). Demo `--timeout` (default 5m) drives
    both the ctx deadline and `client.RetryBudget`.
  - Tests: `TestRetryAfterFrom`, `TestPostChatWithRetry_RespectsBodyRetryAfter`,
    `TestPostChatWithRetry_GivesUpAfterRetryBudget`,
    `TestChatStream_ForwardsConstrainedDecoding`.
- **corrallm side (ml-kit/corrallm.yaml):** added a random-minted interactive
  key `sk-agentkit-…25` for the demo. corrallm loads keys once at boot →
  **needs a restart to activate** (not done: can't fully reconstruct the live
  launch env — ADDR etc. — and it'd bounce the 2.5-day prod server + evict
  Qwen + interrupt aw3). Staged only; user owns the restart.
- **Example `examples/agentkit-demo/`** (same module, package main): subcommands
  chat/tools/schema/grammar/inject/lift/notify/compact. `store.go` = a complete
  in-mem `agent.Store` (+ RevalidateStore) integrators can copy. Offline demos
  (schema/inject/lift/notify/compact) are deterministic (compact uses a
  stubRunner for the summarize step); live demos (chat/tools/grammar) hit the
  model and degrade gracefully on 429 with an explanation. All offline demos
  run-verified; live path confirmed to honor backpressure to `--timeout`.
- **Docs:** `README.md`, repo `CLAUDE.md` (invariants + guardrails), `docs/
  {concepts,features,provider}.md`, `examples/agentkit-demo/README.md`.
- **Kitchen-sink continuation (round 2):**
  - `mcp` demo — real mcpmgr integration (`mcp.go`: `mcpToolDefs` +
    `mcpDispatcher` bridges); spawns `npx @modelcontextprotocol/server-everything`.
    **Live-confirmed** (model called `echo` through MCP).
  - `structured` demo — reliable typed extraction stacking tool_choice +
    SchemaValidator + ForcedTerminalTool + ErrSessionClosed. **Live-confirmed.**
  - `--trace` flag + `trace.go` — a ~40-line stdout `agent.Tracer` (nested
    Turn/streamChat spans + timings). **Live-confirmed.**
  - **llm.Client now surfaces the 4xx/5xx BODY** (`statusError`) instead of an
    opaque "status N" — this is how we found the MCP bug. +test.
  - **mcpmgr bug fixed:** a tool with no required fields emitted
    `"required": null`, which llama.cpp rejects ("type must be array, but is
    null") → the whole chat 400s. `normalizeSchema` defaults nil→[]/{} and
    empty type→object. +test. Benefits autowork3 too.
  - `serve` demo — SSE relay via `llm.StreamChunkToSSE`. **Live-confirmed.**
  - `lift` demo upgraded to a live end-to-end park+resume across two Turns.
  - `converge` capstone demo + `docs/concepts.md` "coalescing turn boundary"
    section — names the composite pattern (a lifted result + queued messages +
    notifications merge into ONE next-turn context). This was the underdocumented
    headline of the event-driven turn model. **Live-confirmed.**
  - `llm.Client` surfaces the 4xx/5xx BODY (`statusError`) not opaque status.
- **Green:** `go build ./... && go vet ./... && go test ./...` ✅ (all live
  demos except MCP/structured/grammar are offline-deterministic; those three +
  chat/tools confirmed live via the dedicated `agentkit` key).
- **NOT done / next:** not committed yet; live grammar/tool_choice not yet
  confirmed against corrallm end-to-end (model was saturated — 429 the whole
  session). Confirm when the backend is idle.

### ✅ Slice 7 — v0.2.0: cache-aware shaping + surfacing + token accounting
- **User spec:** LOD should trigger on ~10k tokens *remaining* (not eager —
  each LOD/compaction rewrites the prompt prefix + blows the KV cache), LOD
  before compaction; emit the compaction summary as a hidden field + continue
  the same turn with meta; emit token usage total + active(current window).
- **Delivered (breaking — Turn signature changed):**
  - `ShaperPolicy.LODHeadroomTokens` (0→10k default; <0 disables). `shapeTarget()`
    = Budget−headroom; Build reshapes to the target so a restructure leaves
    runway. Phase 0 leaves the prefix untouched (cache intact) until crossed.
  - `CompactionInfo{Summary,SubsumedCount,TokensBefore,TokensAfter}` surfaced
    via a **ctx compaction sink** (Shaper→Session, no ContextBuilder sig change)
    → `TurnResult.Compactions` + `Session.OnCompaction`. Same-turn continuance
    already inherent (compaction is inside Build).
  - `Turn` now returns `TurnResult{Reply, Compactions, Usage}` (was `(string,
    error)`). `TokenUsage{Total,Active}`: Total=cumulative billed this session,
    Active=current live (compacted+LOD) window. `Session.OnUsage` callback.
    `streamChat` now returns usage; Session accumulates `usageTotal`.
  - Surfacing = BOTH (callbacks + TurnResult), model-ctx = host-provides (per
    user).
- **Tests:** `shape_v2_test.go` (shapeTarget cases + Turn surfaces compaction &
  usage). Example: `compact` demo rebuilt to run a Session.Turn showing
  OnCompaction (summary as hidden field, 1150→28 tokens) + total/active tally;
  `tools`/`converge` print usage. Docs (concepts/features/README) updated.
- **Green:** `go build/vet/test ./...` ✅. autowork3 INSULATED on v0.1.0 (not
  bumped — its 3 Turn call sites would need updating for the new signature).
- **Next:** tag v0.2.0; optionally bump+adapt autowork3 to it.

## Status: the whole arc (slices 1–5b) is SHIPPED + committed
- agentkit: initial commit `e152268` (not pushed; not yet its own remote).
- autowork3: branch `agentkit-extraction`, commits `840d37c` (extraction +
  notification lifecycle) + `646d348` (5b). Not pushed.
- **Next natural steps (unstarted):** push/publish agentkit as its own repo +
  drop the `replace` directive; point the user's separate claude-openai
  project at `agentkit/agent` as the 2nd consumer to validate the seam.

## Icebox (deferred, opt-in)
- **Daemon-side active recheck** — a notice carrying a shell/MCP command the
  daemon RUNS. NOT built: arbitrary shell from a notification payload is a
  privilege-escalation surface. Superseded by the hook + masked-MCP
  convention (integrator owns the check). Only revisit with an explicit
  execution-context + auth decision.
- **Supersede vs. shown stacking** — supersede (merge replace) folds only
  UNSHOWN accumulators, so a new emit while an old same-key notice is already
  shown can briefly leave two in context until the preparer/TTL/clear catches
  it. Acceptable today; tighten to "delete prior non-compacted + insert" if it
  bites.

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
