# agentkit — plan

> How this plan works: current state + active work + decisions ONLY. Finished work lives in `plan/done.md`, deferred opt-in work in `plan/icebox.md`, the Claude Messages API gaps in `plan/claude-api-plan.md`. Marks: ◻ todo · ◐ in progress · ✅ done · ⏸ parked · ❓ blocked. Every active item carries next / risks / blocking decisions / optional extensions.

## What this is

`github.com/iodesystems/agentkit`: a batteries-included Go client for OpenAI-compatible endpoints (and Claude's Messages API through the shim). It owns the tablestakes every agent client needs: the tool loop, compaction and LOD (the Shaper), injection, lifting, batching, validation, the notification lifecycle, turn and compaction hooks, loop guards. It does NOT own orchestration (roles, a host's task DAG, scheduling). A plan the model keeps for itself is not orchestration: optional `taskgraph` (Carl, 2026-11-03).

Packages: `llm`, `mcpmgr` (stdlib only, standalone), `agent` (imports `llm` + stdlib), `files`, `loopdetector`, `taskgraph`, `ragnotify`, `anthropicadmin`, `cmd/toolfmt`, `examples/agentkit-demo`. Zero third-party dependencies; the one `require` is `openai-claude-api-shim`, ours and stdlib-only. Layout and invariants: `CLAUDE.md`. External story: `README.md`, `docs/`.

## Current state (verified 2026-11-06)

- Release: agentkit v0.25.0 (overflow fold/retry, files byte-offset paging), shim v0.5.0. v0.17.0 is retracted (carried the Anthropic SDK).
- Consumers, by their `go.mod`: corrallm at v0.24.0 / shim v0.5.0; dun v0.24.0 (checked 2026-11-06); autowork3 v0.6.0; aw4 uses `replace => ../agentkit`.
- Guardrail: `go build ./... && go vet ./... && go test ./...`.

## Active work

### ◐ `files` read: byte-offset paging, grep windows in long lines (2026-11-06)
For dun retiring its `read_saved` tool (dun aw4 issue i_1m6t8yrva6w1app5mjdwv5): saved output becomes a file under /tmp that `read` must handle as well as `read_saved` did. Owner's rule from dun (docs/decisions.md): line-based readers are useless on text with no lines. `read` takes `at` (byte offset; pages up to ReadMax, rune-safe both ends, header names the next `at`; negative counts from the end; not with offset/limit). A line cut for exceeding the cap names the `at` that continues it. `grep` shows a match past a long line's first 300 bytes as a window around it (was: the head, without the match), with the `at` of the window (rg: `absolute_offset`; Go engine: counted).
- Tests: `files/page_test.go`; both engines verified with rg 14.1.0 (rg is not on PATH on the dev box; the rg test variant skips without it).
- **next:** released in v0.25.0; dun bumps and retires `read_saved`.
- **risks:** grep output changes for every host (aw4 builds from source via replace): a long match line now starts with `…`. `at` adds ~170 chars to the read schema on every request. A multi-match line shows only the window of its first match.
- **blocking decisions:** none.
- **optional extensions:** a window per match on a line with several.

### ◻ Per-(endpoint, model) capability record (2026-11-07)
Requested by dun-bb, owner direction (aw4 issue i_1m6v7hgpwcwrk5pbswp0w6, project agentkit, 4 attachments). agentkit keeps a capability record per (endpoint, model) and translates requests by it, so dun works on any OpenAI-compatible API plus Anthropic. Learning is client-side and costs no extra requests:
- from normal traffic: the reasoning tokens reported for the control that was sent;
- from refusals: a 400 that names a field updates the record and the request is sent again, once;
- from declared metadata when the endpoint has it (corrallm `thinking_*`, OpenRouter `supported_parameters`);
- an explicit probe only on dun's `/model probe`. Extend `llm.ProbeToolSurface`; do not start a second prober.

Scope: reasoning (off, efforts, budget field and its minimum), real context and output limits, tool calls (parallel, tool_choice forcing, argument fidelity), streaming (usage in the stream, reasoning field shape), sampling (logit_bias, banned strings, DRY), constrained decoding (GBNF, response_format json_schema, vLLM guided_*, strict tools), modalities. corrallm should use these helpers and not build its own.
- Evidence (dun-bb, Fireworks GLM-5.3, 2026-11-07): `reasoning_effort: "none"` → 400 (thinking-only); unset → 300 max_tokens all thinking, empty content; "low", "minimal" and an integer (64) accepted; `thinking.budget_tokens` must be ≥ 1024. Median reasoning: 791 tokens at the default, 103 at low.
- Decided (Carl, 2026-11-07, "use the most open standard"): the openai dialect sends the OpenAI Chat Completions values as given (none, minimal, low, medium, high, xhigh); off is "none". This replaces dun-bb's stopgap `bbead4f` (off → "low"). GLM-5.3's 400 on "none" is a per-model gap for the record, not a global mapping; until the record exists, dun sets "minimal" for that provider.
- Design: `plan/capabilities-design.md` (2026-11-07). Slices: 1 core, 2 learn (+ reasoning translation), 3 declared (corrallm/OpenRouter `/models`), 4 probe, 5 hosts.
- Slices 1–4 done 2026-11-07 (not pushed, not tagged): core `2944095`, declared `e209053`+`c9ef602`, probe `ef35608`, learn+translate `840ff6c` (SHARED PATH: `chatBody`/`sendChat`/stream reader), integration fix `a5d91b1` (probe requests are raw: no translation, no traffic learning). API: `SetCapabilities`, `Capabilities()`, `OnLearn`, `OnTranslate`, `ResendOnRefusal` (false), `LoadDeclared` (uses `/v1/models` when the base lacks `/v1`), `ProbeCapabilities(ctx, c, ProbeOpts{Context})`.
- Slice 5, dun: done on dun branch `capabilities` (dun-bb, 2026-11-07, built via GOWORK against `a479e71`). `TrackCaps` in `tuneClient` covers main and child clients; record in `$DUN_HOME/capabilities/<sha(endpoint\nmodel)>.json`; `effort_off` pinned as SourceHost; `/model caps`, `/model probe`. dun main keeps `d41e9a1` until a tag.
- Passthrough for proxies (Carl, 2026-11-07): `Client.TranslateBody(map)` and `Client.LearnResponse(sent, status, body)` for a body the Client did not build (corrallm forwards raw bodies). Also fixed: a JSON-decoded `max_tokens` (float64) read as unset, so the output cap RAISED it.
- **next:** tag a release; dun merges `capabilities` and bumps; corrallm (slice 5b) wires the passthrough pair into its per-candidate rewrite sequence. corrallm-2c's placement (2026-11-07, no diff yet): TranslateBody after each `withSampling` (chat completions; /v1/messages turned into chat), not on nativeMessages; LearnResponse in the deferred finish with the pre-shim body, Anthropic targets included; a record-only `llm.Client` per (Target URL, upstreamID), stored in SQLite; `clampMaxTokens` stays; remote OpenAI-API backends only. Waits on: a tag; Carl on translation for every key vs opt-in; Carl on a single transform chain.
- dun stores the whole `Capabilities()` snapshot under its store lock on each OnLearn (no out-of-order writes) and drops SourceHost facts before storing, so a pin lives only as long as its config line.
- **gaps:** traffic learning picks `reasoning.off` by ladder order; only the probe picks by observed reasoning tokens (dun: GLM `low` 107 chars < `minimal` 220, one sample). The probe detects a 4xx by matching `llm: status NNN` in the error text (no typed status error). `think_tags` is never learned (no inline-think detection to hook).
- **risks:** a 400-then-resend is a second retry rung beside the overflow resend; both must stay bounded to one. A record learned from one bad response pins wrong behavior (the same trap as `ShapeHint.Scale`), and now persists in the host's store too: the report must carry its evidence (what was sent, what came back) so a host can expire or override it.
- Decided (Carl, 2026-11-07): agentkit does not store the record. What it learns is reported to the holder (a callback on the Client), and corrallm and dun each store it themselves; the host seeds the Client from its store at start. agentkit keeps only the in-memory copy for the Client's life.
- **blocking decisions (Carl):** whether a refusal-driven resend is wanted at all in `llm`, a path every host shares. Until then it ships as `Client.ResendOnRefusal`, default false.
- **optional extensions:** expose the record in `/v1/models`-style form for corrallm's dashboard.

### ◐ corrallm upgrade to agentkit v0.24.0 + shim v0.5.0 (2026-11-06)
corrallm committed the bump as `a18fe07` (go.mod/go.sum only, no code changes), then the PDF rule as `1fc9519` (replaced the unpushed `5cd7b16`). By its own report, build, vet and `go test ./...` pass with GOWORK=off. Deployed 2026-11-06 as v0.1.0-97-g1fc9519 (pushed af31a95..1fc9519). Live checks by corrallm-12: Swift alias → text conversion OK; chandra → vision OK.
- **remaining:** a live PDF to Claude, not verified. The anthropic daily budget was spent and the claude sub-* path needs a caller login. Covered by integration tests only. Waits on: budget reset (23:59 UTC) or a sub login.
- **next:** fold corrallm's translator use of the shim into `claude-api-plan.md` "Future: corrallm owns Claude".
- **risks:** conversion runs once, before the lane walk, using the lane's first member's strategy, so in lane [claude-sub-*, Swift] the Swift fallback gets the default text, not its own `pdf: vision` (behavior that already existed). No live check yet that Claude receives a `document` block. Shim errors (incl. `ErrUnsupportedContent`) surface as 400 `invalid_request_error`.
- **decided (Carl, 2026-11-06, revised):** native first, else convert. A model that reads PDFs natively (Claude) gets the part unchanged. Otherwise the configured `pdf:` strategy is used, else the built-in `text` default. This replaces corrallm `5cd7b16` (strategy-only, which cost the fireworks models PDF reading). Implemented in `1fc9519`: the native signal is the backend's `api: anthropic`. A native backend ignores an explicit strategy; no Claude model sets one today. One test per branch, `go test ./...` green by corrallm-12's report. Swift's own strategy is `pdf: text` (live config rev 239, after it OOM-crashed on `pdf: vision`; chandra2 is the PDF/image model, Carl). ConvertFor now resolves aliases (`b7d2be9`, pushed, not deployed).
- **optional extensions:** `count_tokens` in corrallm (see icebox: shim `EstimateTokens` counts base64).

### ◐ Context overflow: bounded fold + one shrink-retry (2026-11-06)
Cause: aw4's 400s at 16:59. `Shaper.summarize` sent every foldable entry raw in one request: 89,653 tokens against an 86,016 window. A 4xx overflow was untyped, so nothing recovered. Written by corrallm-12 on Carl's ask; reviewed here. `68c81e8` types a 400/413 overflow (llama.cpp, OpenAI, vLLM, Anthropic bodies) as `*llm.InputTooLargeError` with `PromptTokens`/`ContextTokens`; still not retried by `postWithRetry`. `4a8dd0c` (SHARED PATH: dun, aw4): fold in stages that fit the budget, one shrink-retry per fold; `Session.Turn` learns a `ShapeHint` (window + estimator scale) from a refusal, rebuilds, resends once, else `TurnOverflow` + typed error. Review fix: a refused send no longer spends a `MaxTurns` round (it lost the typed error on the last round).
- **next:** released in v0.25.0. dun bumped (`77ced8d`, not pushed; 19 packages green on a clean export). aw4 builds and its 6 agentkit-importing packages pass against v0.25.0 (aw4-3b, not deployed; the daemon is still the 21:43 build). The 16:59 session cannot be replayed (aw4 replay re-runs from the brief, not the accumulated history), so the case is pinned here instead: `TestFold_AW4ScaleBacklogGetsThrough`, ~90k estimated tokens plus one 100k event against 86,016: 3 stages, 0 refusals at a 1.15× undercount; 4 stages, 1 refusal at 1.5×. Live confirmation waits on Carl approving an aw4 daemon rebuild; aw4-3b then counts 400 "exceeds the available context size" (3–13/day since 10-29) and TurnOverflow.
- **dun stacks a second rung:** `rescue.go` `runTurnWithRescue` catches the typed error (plus text fallbacks), runs up to 2 LLM folds that re-root the session, and re-runs the Turn. Worst case per turn: 3 Turns, each with one internal resend, plus 2 folds. The ShapeHint.Scale learned before a re-root still shrinks later builds.
- **risks:** `ShapeHint.Scale` only grows and lives for the Session: one refusal with a bad estimate (e.g. image-heavy) shrinks every later build. A tight window with a large carried summary makes one fold call per entry (bounded by entry count, not by a cap). Only a Shaper honours the hint; another ContextBuilder ends in `TurnOverflow` at once. `ShapeHint.Window` uses `DefaultOutputReservePct`, not the host's reserve. The per-entry fold line format is a copy of `eventLines`. Not run against a live endpoint.
- **blocking decisions (Carl):** dun keeps its rescue as a second rung, or drops it now that Turn resends.
- **aw4 side:** treat TurnOverflow as final for the session so it does not resume into the same overflow (aw4 issue i_1m6sy2eb).
- **optional extensions:** log the fold stage count (aw4 asked; nothing logs it today); decay or reset `Scale` after a successful round; a cap on fold stages.

### ◐ `taskgraph` — a plan the model keeps, shared by sub-agents (2026-11-03)
Ported from dun's `internal/taskgraph` (dun `098c15b`), on Carl's go. Graph, views and `plan` tool as dun had them, plus: `Store` is interface-only (the integration provides it; dun's `MemStore` stays in dun), `Diff`/`Apply` compute per-write `[]Change` with `Tell` = owners to notify (node owner before/after; dependents' owners on a status move; never the actor), `WithOnChange` hands the tool's kept changes to the host. Owner label in views is `(owner X)`, was `(agent X)`.
- Decided: boundary text narrowed in CLAUDE.md / README / concepts.md. Moved before dun's delegation shipped and with one consumer, Carl's call.
- Evidence 2026-11-06: dun already imports `agentkit/taskgraph` (`plan.go`; commits `b555223` v0.22.0, `da7d351` owner on a node). dun's `plan.go` has uncommitted edits.
- **next:** dun finishes and commits its `plan.go` work and bumps past v0.23.0; confirm `Tell` against a real delegation.
- **risks:** `Tell` rules untested against real delegation; the delegator is NOT told (host adds it, the graph has no delegator record). `List()` ends with `ReadyLine("")`, the main agent's ready set, even for a sub-agent's `show`. `Node.Files` is carried but unenforced.
- **blocking decisions:** none open.
- **optional extensions:** a per-owner view (subtree + what it waits on, read-only) for a sub-agent's `show`; a `delegate` action.

### ◐ `loopdetector` — loops with no period (2026-10-31)
Owner: centralize loop detection here, input is session + stream chunk (no cache), output is a loop count, the cut is a configurable tolerance. Package and `llm` seam done (`Client.Watch`, `StreamWatcher`, `ToolCallWatcher`); the default guard watches reasoning too; a proxy's `content_filter` + `loop` reads as StopReasonRepetition. Reviewed by aw4 and dun 10-31. Measured on 1,315 sessions of corrallm traffic: fires on 4, including a real 40-sentence reply cycle the exact guard missed; 1 of 2,172 replies.
- Evidence 2026-11-06: dun installs it count-only (`cmd/dun/loopscmd.go`, mode `count`, keeps the guard because `Install` replaces it).
- **next:** read dun's counters; the numbers decide when text cutting turns on.
- **risks:** defaults (Window 48, MinShare 0.7, Similarity 0.85) tuned on real traces: 0 cuts at Tolerance 3 in aw4's 13,454 thinking blocks (48 h); the real 40-sentence cycle is still found, at 19; dun's tool loops still trip. Lost: dun's idle sub-agent, too few sentences for the window. No real runaway in thinking seen yet, so that side is unproven. aw4's tool sequences were not replayed.
- **decisions (owner 10-31):** no polling exemption: an unchanged result counts, clients wait on an event. dun keeps `loopguard.go` beside this; aw4 keeps `MaxRepeatedExchanges`.
- **assumptions:** `agent` must not import it (invariant 2), so a host installs it. Tool arguments get the exact detector only: writing a file again is not a loop. History is read from the last message by a person: a person asking again is not a loop, and a loop that goes on only across their messages is not seen. Host notices (`[tag] text`) do not end a turn. Edit then re-run with the same failure stays under MinShare, because edits are distinct calls.
- **blocking decisions:** when to turn cutting on (Carl), after the counters.
- **optional extensions:** digit-insensitive matching for random data inside a line.

### ◐ Claude Messages API (`llm/anthropic.go`, `docs/features.md`)
Live against api.anthropic.com through dun. Gaps, proposals, next steps and blocking decisions: `plan/claude-api-plan.md`.

## Decisions / conventions
- Module path `github.com/iodesystems/agentkit` is final.
- `llm` and `mcpmgr` stay zero-internal-dep (`go list -deps ./llm ./mcpmgr`).
- Neutral seam: `agent` imports only `agentkit/llm` + stdlib (checked 2026-11-06 with `go list -f '{{.Imports}}' ./agent`). It must never import a host's `events`/storage model. If a host type is tempting, add a field to `Entry` or a method to `Store`.
- Zero third-party dependencies (invariant 0); adding a `require` is a decision.
- Pre-release: no compat shims, edit in place, delete freely.
- Unsupported media fails loudly; agentkit never strips a part (Carl, 2026-11-06).

## Done pointers
Everything shipped, slices 1–N and the lifecycle, files, money and OAuth work: `plan/done.md`.
