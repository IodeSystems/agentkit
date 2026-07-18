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

### ✅ Slice 8 — proactive retrieval (RAG-notify finder)
- **Ask (user):** a "listen to the conversation and ping relevant docs" add-on
  over their **ragtag** service (ingests + indexes + RAG-queries docs). Wants
  BOTH an explicit MCP search tool AND background pings off user/agent messages
  even when the model never searches.
- **Design decisions (user-chosen):** new sibling pkg + neutral seam · ragtag is
  already an MCP server · notices are **pointer-only** (id+title+score+line, not
  the body).
- **Validated against industry practice** (research, 2024–2026): the seam
  placement (retrieval on a pre-model hook — LangGraph `pre_model_hook`, Claude
  Code `UserPromptSubmit`, Zep `get_user_context`), pointer-not-payload
  (claude-mem's index+pull), dual-channel (tool-call vs always-on is a
  latency/UX call), and MCP `tools/call` pull-transport (nobody uses MCP push
  for retrieval) are all mainstream. Two additions the research forced:
  `Timeout` (hot-path latency — folded in) and the written assumption that
  **RERANKING lives server-side in ragtag**, not in the finder (the #1 quality
  lever; a raw-vector-top-k finder is noisy regardless of MinScore).
- **Delivered:**
  - `agent/finder.go` — `DocFinder` neutral interface + `DocHit` + `FinderOpts`
    (`MinScore`/`MaxHits`/`Tag`/`Kinds`/`Timeout`/`Render`/`Now`) +
    `FinderPreparer`: a `NotificationPreparer` that observes entries past a
    per-session watermark, queries the finder, injects one pointer
    `KindNotification` per fresh (unseen, above-threshold) hit, capped best-first
    at MaxHits. Fail-open (Find error/timeout → skip, **don't advance
    watermark** → retry next pass). Dedup by DocID per session (in-mem closure
    state, like MCPPreparer leaves state host-side). No new Store method, no
    Entry field, invariant #2 intact (`go list -deps` clean).
  - `ragnotify/` sibling pkg (imports agent + mcpmgr only) — `MCPFinder` over an
    MCP search tool + `ParseHits` (lenient: bare array or `hits`/`results`/
    `documents` wrapper; `doc_id`|`id`, `snippet`|`text`|`content`; id-less hits
    dropped). Same MCP server feeds both channels.
  - `examples/agentkit-demo ragnotify` — offline keyword-stub finder; shows
    pointer notice + MinScore threshold + per-doc dedup across 3 passes.
- **Tests:** `agent/finder_test.go` (threshold, MaxHits best-first, dedup across
  passes, watermark gates re-query, unobserved kinds skipped, fail-open retry) +
  `ragnotify/ragtag_test.go` (parse shapes, id-less/empty drop, clip). Green:
  `go build/vet/test ./...` ✅; ragnotify demo run-verified.
- **Timing (documented, honest):** a USER message → pointer lands in the SAME
  iteration (preparer runs pre-build). An ASSISTANT reply is observed on the
  NEXT pass/Turn (the loop only re-enters a preparer when the turn continues or
  a new Turn starts) — the inherent one-turn lag of the event-driven model.
- **BLOCKING (user owns):** paste one real ragtag search-result JSON so
  `ParseHits` matches it exactly instead of the assumed shapes. Confirm ragtag
  reranks server-side.
- **NOT committed yet.**

### ◐ Slice 9 — local composable RAG tool (NEW separate module) + multimodal enabler
- **Ask (user):** build a local, composable doc-RAG tool ON agentkit — BM25 +
  PDF pagify/OCR via a media-capable LLM + sqlite `document:page:fragment`
  index (+ a custom NSW/HNSW sidecar only IF cosine similarity underperforms).
  Strategy: "worm into workflows via simple composable tools that scale" —
  candidate to **supersede ragtag**.
- **Decisions (user-chosen):**
  - **Lives in a NEW separate module** that depends on agentkit (keeps
    agentkit's dep hygiene — sqlite/PDF deps stay out of llm/mcpmgr/agent).
    Module name/path: **UNDECIDED — blocking the scaffold.**
  - **Multimodal `llm` FIRST** (reusable beyond this tool). ✅ DONE — see below.
  - **PDF→image: pure-Go, assume image-PDFs** (extract embedded page images
    via pdfcpu; born-digital vector-text PDFs out of scope for v1). No native
    dep, no CGo.
- **Design ratified for the new module:**
  - **SQLite FTS5 = BM25 + the index in ONE pure-Go dep** (`modernc.org/sqlite`,
    CGo-free → single static binary). Schema: documents / pages / fragments
    (+ `fragments_fts` fts5, `bm25()` ranking). One `.sqlite` file IS the index.
  - Composable CLIs, Unix-pipeable, each a lib too: `pagify` (pdfcpu image
    extract) · `ocr` (multimodal llm → vision model) · `index` · `search`
    (BM25) · `serve` (MCP tool + **`agent.DocFinder`**). The DocFinder seam
    (slice 8) makes local↔remote-ragtag a swap, not a rewrite.
  - Vectors deferred: add sqlite-vec or a custom NSW file only if FTS5 lexical
    proves insufficient (ragtag already deleted its file-vector store `pkg/zvec`
    in favor of PG+HNSW — don't rebuild it speculatively).
- **✅ Multimodal enabler (in agentkit `llm`, this repo):** `Message.Parts
  []ContentPart` + `ContentPart`/`ImageURL` + `TextPart`/`ImagePart`/`ImageData`
  helpers + custom `Message.MarshalJSON` — Parts (when set) owns `content` and
  marshals OpenAI's array-of-parts shape; the plain-string path is byte-for-byte
  unchanged (test-pinned). Send-only (replies are text). stdlib-only
  (`encoding/base64`) → invariant #1 intact (`llm` still zero third-party dep).
  Tests: `llm/multimodal_test.go` (string-unchanged, multimodal array, ImageData
  URI, tool-fields-unaffected). Green.
- **Sequencing:** A = lexical core (FTS5 + index/search + DocFinder + MCP, fully
  offline) · B = multimodal llm ✅ · C = PDF pagify+OCR · D (opt-in) = vectors.
- **✅ Module scaffolded: `github.com/iodesystems/raglit`** at
  `~/local/src/iodesystems/raglit` (replace → ../agentkit during dev).
  - `store.go` — SQLite FTS5 store: `documents`/`fragments` + `fragments_fts`
    (external-content, trigger-synced) + `bm25()` ranking. `Open`/`Ingest`
    (idempotent replace-on-reingest)/`Search`. `ftsQuery` quotes tokens so
    user punctuation/operators can't break MATCH; `Hit.Score = -bm25` (higher =
    better, agentkit convention). Pure-Go `modernc.org/sqlite@v1.51.0`.
  - `finder.go` — `Finder` = `agent.DocFinder` over the Store; collapses
    fragments → one DocHit per document (best passage as the pointer line).
  - `cmd/raglit` — `index` (walk dir, paragraph-split text/md) + `search` CLIs.
  - `README.md`. Tests: BM25 ranking, idempotent reingest, punctuation-safe
    query, Finder best-per-doc. **Green** (`go build/vet/test ./...`).
  - **End-to-end run-verified:** indexed a 2-doc corpus, "why does my refresh
    token expire" → correct auth fragment top (score 1.882); "rollback deploy"
    → deploy fragment top. One portable `.sqlite` file.
- **✅ Slice A complete — `raglit serve` (MCP):** `cmd/raglit/serve.go` — stdio
  MCP server (mark3labs/mcp-go server side) with one `search` tool. `hitsJSON`
  emits `{hits:[{doc_id,title,page,score,snippet}]}` — the exact shape
  `ragnotify.ParseHits` consumes, so ONE raglit server drives BOTH channels
  (explicit tool + proactive DocFinder). **Verified two ways:** (1) driven raw
  over stdio (initialize→tools/call) → correct ranked JSON; (2) `serve_test.go`
  loop-closure — raglit `hitsJSON` output → `ragnotify.ParseHits` → DocHits, in
  code. raglit now requires agentkit at test scope (imports ragnotify).
- **✅ Slice C — PDF pagify + vision-LLM OCR** (raglit commit `383cd9e`):
  `pagify.go` (pdfcpu embedded-image extraction, image/scanned PDFs only;
  `ErrNoPageImages` for born-digital), `ocr.go` (`OCR.Page` via multimodal
  `llm.Message`; `Chatter` iface for stubbing), `Store.IngestPDF` (pagify→OCR→
  index with real page nums; images persist under `<home>/pages/`), CLI `pagify`
  + `ocr` + `index` PDF routing. **LIVE-VERIFIED** end-to-end: rendered text →
  PDF → OCR → index → search hit.
- **bonsai reality (resolved):** the vision model is **`ternary-bonsai-27b`**
  (llm.iodesystems.com, key `raglit`) — live, accepts images, confirmed OCR.
  `gemma-4-12b` and `Qwen3-6-27B-MPT` were "no backend available" (down) at test
  time. CLI `--llm-model` defaults to `ternary-bonsai-27b`. Also available:
  `nomic-embed-text` (embeddings) → unlocks Slice D vectors when wanted.
- **✅ Slice D — vectors** (raglit `c6c9419`; agentkit `cd6ea06`): agentkit
  `llm.Client.Embed` (OpenAI `/v1/embeddings`, shared retry via `postWithRetry`).
  raglit `embed.go` (Embedder + nomic prefixes + float32 BLOB codec + cosine),
  `fragment_vectors` table (FK-cascade), `Store.VecSearch` (brute-force cosine —
  pure-Go, modernc can't load sqlite-vec) + `HybridSearch` (BM25 ⊕ vec RRF);
  `Ingest` takes ctx + embeds when `SetEmbedder`; Finder goes hybrid when an
  embedder is set. CLI `index --embed` / `search --mode bm25|vec|hybrid`.
  **LIVE-VERIFIED** vs `nomic-embed-text`: zero-lexical-overlap query hit the
  right fragment. Custom NSW sidecar deferred — only if a linear scan gets slow.
- **✅ init wizard + config** (raglit `7c882b0`): raglit is unusable until
  configured (per user — "not everyone is as good as corrallm"). `raglit init`
  wizard prompts OpenAI-compatible base URL + token, queries `/v1/models`, picks
  a vision model + an embedding model → `<home>/config.json` (0600). No-arg
  `raglit` runs the wizard when uninited. Flags default from config→env→OpenAI
  fallback; `requireVision`/`requireEmbed` emit a "run raglit init" hint.
  **LIVE-VERIFIED** wizard + config-driven commands vs bonsai.
- **✅ Slice E — lazy ingest queue + URL fetch + status + demo** (raglit
  `f092fc5`): `ingest_jobs` table + `Enqueue`/`claimNextJob`/`IndexStatus`
  (doc/fragment counts, per-state job counts, recent rate jobs/min, per-item ETA
  = queue-pos × avg duration); `Fetch` (file://, http(s)://, bare path; PDF by
  ext/content-type; 64 MiB cap); `Worker` (ProcessOne/Drain/Run — per-URL fail
  recorded, never fatal). serve gained a background worker + `ingest` +
  `index_status` MCP tools (alongside `search`). CLI `ingest [--now]` / `work` /
  `status`; `demo` = self-contained offline tour (embedded corpus → lazy queue →
  drain-with-status → search). **VERIFIED:** `raglit demo` end-to-end + serve
  MCP loop (ingest → bg worker → index_status done → search).
- **raglit status: A–E shipped + committed** on `main` (`85e2533` core+home,
  `383cd9e` OCR, `c6c9419` vectors, `7c882b0` init, `f092fc5` lazy-ingest+demo).
  None pushed. Full spine: init → index/ingest (local + URL, lazy, text+PDF,
  ±embed) → status → search (bm25/vec/hybrid) / serve (MCP: search+ingest+
  index_status, both channels). Single static binary, pure-Go.

### ✅ Slice F — LLM-segmented ingest engine (raglit) — SHIPPED
- Unified segmentation: VISION model for page images, TEXT model for on-disk
  text/code — same schema, continuation, pipeline. Per unit the LLM emits
  schema-validated `{continues_previous, fragments:[{text}]}` (an `emit_fragments`
  ToolDef + `agent.SchemaValidator` fix-loop → fallback to whole-unit fragment).
  Fragments are model-judged coherent chunks (bind small related units — several
  short funcs, list clusters; no pathological atoms).
- **Continuation + deferred embed:** one OPEN fragment carried across units;
  a unit's first fragment extends it when continues_previous (keeps start
  page/ord) else closes it; the OPEN fragment is NOT embedded until the next
  unit/EOF resolves it.
- **Two concurrent in-process pipelines:** segment goroutine (sequential units)
  → channel → embed goroutine, at once.
- **Context discovery (not compaction):** text/code windows sized to the model
  context, found by BLOWING the limit (probe until overflow, cache per
  endpoint+model). `DiscoverContext` → agentkit `llm` (reusable).
- **✅ ALL PHASES SHIPPED** (raglit `9502004` p1, `47b5113` p2, agentkit
  `e77afef` DiscoverContext, raglit `65ecb1d` p3+4):
  - **p1 segment.go** — `Segmenter` (SegmentImage/SegmentText, SchemaValidator
    fix-loop + fallback) + `Assembler` (deferred open fragment, cross-unit merge).
    Live: bonsai bound two short funcs into one fragment.
  - **p2 pipeline.go** — `ingestUnits`: segment → insert (BM25) + CONCURRENT
    embed goroutine; `IngestPDF` rewritten to segment page images. Live PDF OK.
  - **p3 window.go + llm.DiscoverContext** — text windows sized to the probed
    context (÷2 for the echoed output), cached in `config.ContextTokens`, lazy
    on first text job. `ingestText` windows + segments code/text.
  - **p4** — `index` now enqueues+drains through the one pipeline; duplicate
    `readDoc` deleted.
  - **Live-verified end-to-end:** `index auth.go` → bonsai segmented it into 4
    function-level fragments (each func + doc comment bound); hybrid search for
    "what happens if a refresh token is reused" returned the rotateRefresh
    fragment. The whole "very good at code on filesystem" goal, working.
  - Risks that materialized as designed: small-model JSON absorbed by fix-loop +
    fallback; continuation keeps OCR sequential, embed runs concurrently.
  - **Context handling — final shape** (agentkit `1d82331`, raglit `2a3f989`,
    `f5fc108`): FOUND corrallm/bonsai accepts 60k-token prompts at HTTP 200 (no
    overflow error, no advertised n_ctx) — so blow-the-limit has no boundary.
    bonsai's real context is **256k** (user). Resolution: **config + smart
    default, NOT an ingest-path probe.** `WindowCharsForHome` = configured
    `ContextTokens` or `defaultContextTokens` (131072); window is
    output-reliability-capped (`maxWindowChars` ~16k tok / 64k chars), so any
    context ≥ ~40k gives the same window → the exact number rarely matters.
    `DiscoverContext` bounded at 32k (`contextProbeCeiling`, safe lower bound on
    tolerant servers; real boundary on rejecting servers) and kept as the wizard
    "probe" option. `--context-tokens` flag overrides. Live: config-only home
    (no flags) segments via bonsai; DiscoverContext → 32768 in 15s.

## What's next (open, none blocking)
- **Deferred/opt-in:** runtime `select_indexes` MCP tool; eager summaries for
  oversized fragments (currently pointer-notify covers it).
- **Segmentation quality pass:** verified it WORKS, but no systematic eval across
  diverse docs / prompt tuning. Would need a small eval corpus.
- **End-to-end proactive demo:** wire raglit's DocFinder over a real index into an
  agentkit Session + FinderPreparer — the original "listen + ping relevant docs"
  loop, shown working against raglit. Validates the whole arc.
- **Repos/publish:** agentkit rides `fix/cached-token-accounting` with a stack of
  feature commits (multimodal, ragnotify, Embed, DiscoverContext) — wants its own
  branch/PR. raglit has NO remote yet. Neither pushed.
- **Carried from slice 8:** real ragtag search-result JSON for `ragnotify.ParseHits`
  (only matters when pointing ragnotify at actual ragtag; raglit's serve output
  already matches).

### ✅ Slice G — multi-index + selection (raglit `400c8a0`) — SHIPPED
- `OpenIndex(home,name)` — "default"=index.sqlite, others=index-<name>.sqlite,
  sharing originals/pages; name sanitized to [a-z0-9_-] (no traversal).
  `Registry` (Get caches/creates, Names scans disk, SetEmbedder-all).
- serve hosts the registry: `search` defaults to ALL (RRF-merged, hits tagged by
  `index`) or a comma-set; `ingest` targets an index (created on demand);
  `index_status` per-index/aggregate; `list_indexes`. One background loop drains
  every index round-robin (per-index workers cached). `--index` on single-index
  CLI (default "default" = existing file, back-compat).
- **Live-watch selection: no code needed** — `ragnotify.MCPFinder` passes
  `Opts.ExtraArgs {"index":...}` to search (default all = omit). Runtime
  `select_indexes` tool DEFERRED (static/initial selection covers the ask).
- **Verified over MCP stdio:** ingest to two indexes, list_indexes, search-all
  (both, tagged) + scoped search. Registry unit tests green.

### Fragment sizing (raglit `6f4a78e`) — decided + shipped
- ~500-word floor: below it a hit can't concept-chain. Assembler greedily
  absorbs sub-floor sibling fragments (MinChars ~3000) up to a ceiling
  (MaxChars ~9000). Oversized fragments → pointer notifications (fetch on
  demand), NOT summaries (deferred; extra cost + staleness). Prompt asks ~400-800
  words.
- **Committed:** agentkit `9471217` (multimodal llm) + `f7af638` (ragnotify),
  on `fix/cached-token-accounting`. raglit new repo `main`: `85e2533` (core +
  home) + `383cd9e` (OCR). None pushed.
- **Still open (user):** ragtag search-result JSON shape for
  `ragnotify.ParseHits` (carried from slice 8; raglit's serve output already
  matches it, so this only matters for pointing ragnotify at real ragtag).

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
- **Async RAG-notify (zero hot-path latency)** — Slice 8's `FinderPreparer`
  calls `Find` synchronously before build, so a user-message turn eats a ragtag
  round-trip (bounded by `Timeout`, fail-open). Optional: kick `Find` in a
  goroutine, inject via the inbox, hit lands next Turn (same lag assistant hits
  already have) — Copilot-style hidden latency. Only build if the sync round-trip
  measurably hurts turn latency.
- **RAG-notify watermark/dedup persistence** — Slice 8 keeps watermark + seen-
  DocID sets in an in-mem closure (one preparer per live session). A host that
  recreates the Session/preparer per Turn re-observes history + re-pings. If that
  host shape appears, make the watermark/seen store pluggable (a small interface,
  like RevalidateStore) instead of in-mem.
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
