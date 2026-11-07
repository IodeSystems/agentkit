# agentkit — plan

> How this plan works: current state + active work + decisions ONLY. Finished work lives in `plan/done.md`, deferred opt-in work in `plan/icebox.md`, the Claude Messages API gaps in `plan/claude-api-plan.md`. Marks: ◻ todo · ◐ in progress · ✅ done · ⏸ parked · ❓ blocked. Every active item carries next / risks / blocking decisions / optional extensions.

## What this is

`github.com/iodesystems/agentkit`: a batteries-included Go client for OpenAI-compatible endpoints (and Claude's Messages API through the shim). It owns the tablestakes every agent client needs: the tool loop, compaction and LOD (the Shaper), injection, lifting, batching, validation, the notification lifecycle, turn and compaction hooks, loop guards. It does NOT own orchestration (roles, a host's task DAG, scheduling). A plan the model keeps for itself is not orchestration: optional `taskgraph` (Carl, 2026-11-03).

Packages: `llm`, `mcpmgr` (stdlib only, standalone), `agent` (imports `llm` + stdlib), `files`, `loopdetector`, `taskgraph`, `ragnotify`, `anthropicadmin`, `cmd/toolfmt`, `examples/agentkit-demo`. Zero third-party dependencies; the one `require` is `openai-claude-api-shim`, ours and stdlib-only. Layout and invariants: `CLAUDE.md`. External story: `README.md`, `docs/`.

## Current state (verified 2026-11-06)

- Release: agentkit v0.24.0, shim v0.5.0. v0.17.0 is retracted (carried the Anthropic SDK).
- Consumers, by their `go.mod`: corrallm working tree at v0.24.0 / shim v0.5.0 (uncommitted); dun v0.23.0 / shim v0.4.0; autowork3 v0.6.0; aw4 uses `replace => ../agentkit`.
- Guardrail: `go build ./... && go vet ./... && go test ./...`.

## Active work

### ◐ corrallm upgrade to agentkit v0.24.0 + shim v0.5.0 (2026-11-06)
corrallm committed the bump as `a18fe07` (go.mod/go.sum only, no code changes), then the PDF rule as `1fc9519` (replaced the unpushed `5cd7b16`). By its own report, build, vet and `go test ./...` pass with GOWORK=off. Deployed 2026-11-06 as v0.1.0-97-g1fc9519 (pushed af31a95..1fc9519). Live checks by corrallm-12: Swift alias → text conversion OK; chandra → vision OK.
- **remaining:** a live PDF to Claude, not verified. The anthropic daily budget was spent and the claude sub-* path needs a caller login. Covered by integration tests only. Waits on: budget reset (23:59 UTC) or a sub login.
- **next:** fold corrallm's translator use of the shim into `claude-api-plan.md` "Future: corrallm owns Claude".
- **risks:** conversion runs once, before the lane walk, using the lane's first member's strategy, so in lane [claude-sub-*, Swift] the Swift fallback gets the default text, not its own `pdf: vision` (behavior that already existed). No live check yet that Claude receives a `document` block. Shim errors (incl. `ErrUnsupportedContent`) surface as 400 `invalid_request_error`.
- **decided (Carl, 2026-11-06, revised):** native first, else convert. A model that reads PDFs natively (Claude) gets the part unchanged. Otherwise the configured `pdf:` strategy is used, else the built-in `text` default. This replaces corrallm `5cd7b16` (strategy-only, which cost the fireworks models PDF reading). Implemented in `1fc9519`: the native signal is the backend's `api: anthropic`. A native backend ignores an explicit strategy; no Claude model sets one today. One test per branch, `go test ./...` green by corrallm-12's report. Swift's own strategy is `pdf: text` (live config rev 239, after it OOM-crashed on `pdf: vision`; chandra2 is the PDF/image model, Carl). ConvertFor now resolves aliases (`b7d2be9`, pushed, not deployed).
- **optional extensions:** `count_tokens` in corrallm (see icebox: shim `EstimateTokens` counts base64).

### ◐ Context overflow: bounded fold + one shrink-retry (2026-11-06)
Cause: aw4's 400s at 16:59. `Shaper.summarize` sent every foldable entry raw in one request: 89,653 tokens against an 86,016 window. A 4xx overflow was untyped, so nothing recovered. Written by corrallm-12 on Carl's ask; reviewed here. `68c81e8` types a 400/413 overflow (llama.cpp, OpenAI, vLLM, Anthropic bodies) as `*llm.InputTooLargeError` with `PromptTokens`/`ContextTokens`; still not retried by `postWithRetry`. `4a8dd0c` (SHARED PATH: dun, aw4): fold in stages that fit the budget, one shrink-retry per fold; `Session.Turn` learns a `ShapeHint` (window + estimator scale) from a refusal, rebuilds, resends once, else `TurnOverflow` + typed error. Review fix: a refused send no longer spends a `MaxTurns` round (it lost the typed error on the last round).
- **next:** tag a release; aw4 and dun bump; confirm against the session that failed at 16:59.
- **risks:** `ShapeHint.Scale` only grows and lives for the Session: one refusal with a bad estimate (e.g. image-heavy) shrinks every later build. A tight window with a large carried summary makes one fold call per entry (bounded by entry count, not by a cap). Only a Shaper honours the hint; another ContextBuilder ends in `TurnOverflow` at once. `ShapeHint.Window` uses `DefaultOutputReservePct`, not the host's reserve. The per-entry fold line format is a copy of `eventLines`. Not run against a live endpoint.
- **blocking decisions:** none.
- **optional extensions:** decay or reset `Scale` after a successful round; a cap on fold stages.

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
