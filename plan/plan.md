# agentkit — plan

> How this plan works: current state + active work + decisions ONLY. Finished work lives in `plan/done.md`, deferred opt-in work in `plan/icebox.md`, the Claude Messages API gaps in `plan/claude-api-plan.md`. Marks: ◻ todo · ◐ in progress · ✅ done · ⏸ parked · ❓ blocked. Every active item carries next / risks / blocking decisions / optional extensions.

## What this is

`github.com/iodesystems/agentkit`: a batteries-included Go client for OpenAI-compatible endpoints (and Claude's Messages API through the shim). It owns the tablestakes every agent client needs: the tool loop, compaction and LOD (the Shaper), injection, lifting, batching, validation, the notification lifecycle, turn and compaction hooks, loop guards. It does NOT own orchestration (roles, a host's task DAG, scheduling). A plan the model keeps for itself is not orchestration: optional `taskgraph` (Carl, 2026-11-03).

Packages: `llm`, `mcpmgr` (stdlib only, standalone), `agent` (imports `llm` + stdlib), `files`, `loopdetector`, `taskgraph`, `ragnotify`, `anthropicadmin`, `cmd/toolfmt`, `examples/agentkit-demo`. Zero third-party dependencies; the one `require` is `openai-claude-api-shim`, ours and stdlib-only. Layout and invariants: `CLAUDE.md`. External story: `README.md`, `docs/`.

## Current state (2026-11-11)

- Release: agentkit v0.30.0 (`ChatOpts.PromptCacheKey`, derived by Session from SessionID + newest compaction marker, for corrallm's slot affinity; billing refusals sent as 400/403/429 are `BudgetSpentError`; neither checked live). Before it, v0.29.0 (taskgraph split into plan_add/update/close/show, `ERROR: plan: ` refusals; breaks `taskgraph.Tool` callers). Before it, v0.28.0 (Usage.ServedModel/RequestID from corrallm, OpenAI, Anthropic, LiteLLM headers and the reply's model; not yet checked live, dun-bf asked 2026-11-11), shim v0.6.0. v0.17.0 is retracted (carried the Anthropic SDK). Release history: `done.md` "Releases".
- Consumers, by their peers' reports: dun v0.4.1 on agentkit v0.27.0 / shim v0.6.0 (2026-11-09); corrallm v0.2.2 on v0.26.0 / shim v0.6.0, deployed on box1 (2026-11-10); autowork3 v0.6.0; aw4 uses `replace => ../agentkit`.
- Guardrail: `go build ./... && go vet ./... && go test ./...`.

## Active work

### ◐ Capability record: open gaps (2026-11-07)
Shipped (slices 1–5, passthrough, corrallm publishing, LoadDeclared reader): `done.md` "Per-(endpoint, model) capability record". Design: `plan/capabilities-design.md`.
- **next:** check `LoadDeclared` against a live corrallm `/v1/models` body with `capabilities` (fixture-tested only, c8ccbbf). Waits on corrallm's first learned record (2026-11-09: 5,995 aw4 requests taught nothing, as designed; Fireworks/GLM traffic from dun is the likely first source). corrallm-2c sends the body.
- **gaps:** traffic learning picks `reasoning.off` by ladder order; only the probe picks by observed reasoning tokens. The probe detects a 4xx by matching `llm: status NNN` in the error text (no typed status error). `think_tags` is never learned. Refusals are matched by error text (now gated on 400/422, 994d7b5). Passthrough has no stream observation (icebox).
- **risks:** a record learned from one bad response pins wrong behavior and persists in the host's store; the evidence travels with each fact so a host can expire or override it.
- **blocking decisions:** none open.

### ◻ Effort selector rooted in agentkit (2026-11-08)
Carl (2026-11-08): "capabilities and effort selector are things that should be rooted in agentkit and integrated by dun." Today dun's `effort.go` owns the levels (default/off/low/medium/high), their descriptions and Level → ChatOpts. agentkit owns the record and the per-dialect translation.
- **next:** dun-bb sends effort.go's semantics (levels, defaults, what "default" means per dialect) when dun is ready to move it; design note before code.
- **sketch:** agentkit owns the levels, their descriptions, and Level → ChatOpts through the record (off → `reasoning.off`, refused → nearest accepted). dun keeps `/effort`, its config line, and the UI.
- **risks:** "default" differs per dialect (send nothing vs. the server's own default); moving it changes dun's wire output unless the mapping is ported exactly.
- **blocking decisions:** none yet.

### ◻ Voice conversation: barge-in + phrase splitter (2026-11-14)
Carl: "io listen events that have stt stream input audio that feeds a model", "Conversation." Design only; no code before Carl's OK.
- **split:** the host owns audio, the STT/TTS sockets (WebSocket client is host-side; no provider needs extensions or subprotocols server-side), endpointing, playback position, echo cancel. agentkit owns the turn: a final transcript is a pending user Entry, end-of-turn starts `Turn` (batching collects all pending), barge-in cancels it.
- **found (2026-11-14, code trace):** a canceled stream persists nothing: `streamChat` returns partial text + error, the caller returns at `agent/session.go:449` before the `Append` at :504; reasoning and usage are lost too. `OnAssistantToken` (:42, :986) already gives the host each text delta. No Entry field marks a cut.
- **piece 1, barge-in:** host cancels with `context.WithCancelCause(ctx)` → `cancel(agent.Interrupted{Heard: n})`. `n` = bytes of reply text (as streamed through `OnAssistantToken`) the user heard. Session persists `content[:n]` as the assistant Entry (cut back to a rune boundary), drops reasoning-only and partial tool calls (same rule as truncated arguments), ends `TurnInterrupted`. A plain cancel keeps today's behaviour. How the host gets `n`: ElevenLabs gives per-char `alignment` (`charStartTimesMs`); OpenAI Realtime, Deepgram Aura and Kokoro give none, so the host counts played audio per phrase (whole phrases heard + estimate inside the current one). OpenAI's own `conversation.item.truncate{audio_end_ms}` is the same idea server-side.
- **piece 2, phrase splitter:** stdlib-only, text in, speakable phrases out: split at sentence end, then clause marks past a min length, hard cap; skip fenced code, strip markdown. Needed for whole-text TTS (Kokoro `/v1/audio/speech`, Aura `Speak`+`Flush`). ElevenLabs buffers itself (`chunk_length_schedule` [120,160,250,290]), so there it only sets flush points. Each phrase carries its byte range in the reply, which is what makes `Heard` countable.
- **optional, speculative turn:** Deepgram Flux only (`EagerEndOfTurn` / `TurnResumed` / `EndOfTurn`, needs `eager_eot_threshold`; +50–70% LLM calls per Deepgram). Host starts `Turn` on eager, plain-cancels on `TurnResumed`. Problem: `ClaimPending` already marked the eager transcript shown; the host must replace that user Entry with the final one. Not designed.
- **risks:** `Heard` is an estimate for every provider but ElevenLabs; too high and the model believes it said something unheard. Echo of our own TTS read as user speech triggers false barge-in (host). Split phrases break mid-number or abbreviation ("e.g.", "3.5").
- **blocking decisions (Carl):** (a) where "io" lives (aw4 events, `/para io`, new), which names the first host; (b) the marker for a cut reply: a new `Entry` field (e.g. `Interrupted bool`, the model sees nothing) vs. a following `KindNotification` "user interrupted after: …" the model sees; (c) splitter location: new package `speech/` vs. `llm`; (d) `Heard` in bytes of text (proposed) vs. runes vs. a string prefix.
- **optional extensions:** speculative turn; a reasoning-delta hook beside `OnAssistantToken`; stdlib WebSocket client in agentkit (icebox unless two hosts need it).

### ◻ Reasoning-cut repair (2026-11-16)
Proposed by corrallm-a3 as Carl's; Carl's own OK not yet given here. aw4 sends `reasoning_budget_tokens=2500`; empties come mostly from max_tokens running out right after llama-server's forced `</think>`. aw4 retries length-cut empties itself (aw4 007de172a).
- **found (2026-11-16, code trace):** an empty reply (no text, no tool call) persists nothing and drops its reasoning (`agent/session.go:508`, `:517`); ends `TurnReply`. finish `length` → `*Overflow` → `OnOverflow`, whose Retry is a plain resend. No prefill anywhere; the shim turns a trailing assistant message into a user "Continue." (shim.go:733). `Usage.CompletionTokensDetails.ReasoningTokens` already parsed.
- **proposed:** detect no text + no tool call + reasoning present, any finish reason (this also covers the open stall re-prompt question). Repair once per turn: hidden notification with the reasoning tail (~2000 chars) + reasoning off for that round via the capability record. Same shape as the repetition re-prompt. `OnReasoningCut(ReasoningCut{Tokens, Finish, Repaired})`; a second empty ends `TurnReasoningCut` (failed).
- **deferred:** prefill continuation (needs `llm.Message` reasoning, per-dialect prefill, a shim change); only after a live check that llama-server continues a prefilled assistant turn with reasoning under Swift's template.
- **risks:** the tail repeats reasoning as visible context; where the record cannot turn reasoning off the repair round thinks again into the same wall.
- **blocking decisions (Carl):** go; default on (proposed, with an off switch) vs opt-in.
- **corrallm side (live 2026-11-16, v0.3.0-106-g389f3ee):** `completion_tokens_details.reasoning_tokens` on /v1/chat/completions (stream + non-stream; /v1/messages unchanged). Answer-room guard only lowers corrallm's own injected budget (`reasoning-budget-clamped from=N to=M`); a client budget leaving < 2048 for the answer gets `reasoning-budget-short answer_room=N` (life-raglit: 800). No "budget forcing triggered" signal exists anywhere (OpenRouter, vLLM, LiteLLM; llama-server logs it at trace only), so detection is empty text + no tool call + reasoning present.

### ◻ Cancel during tool dispatch leaves orphan tool calls (2026-11-14)
Found in the voice trace. A round's `KindToolCall` entries are persisted first (`agent/session.go:674-686`); the dispatch loop (:690-745) never checks `ctx.Err()` and writes no result for calls that never ran; a ctx-honouring `Store.Append` then fails at :741 with the canceled ctx. History ends with tool calls without results, which strict endpoints refuse on the next request.
- **proposed:** stop dispatch at the first `ctx.Err()`; append `ERROR: canceled before this call ran` for each remaining call under `context.WithoutCancel(ctx)`. Own commit, independent of voice.
- **risks:** a tool that already ran side effects but returned after cancel still gets its real result (correct); `WithoutCancel` writes after the host canceled, which a host may not expect.
- **blocking decisions:** Carl's OK (behaviour change on a shared path).

### ◐ Context overflow: bounded fold + one shrink-retry (2026-11-06)
Shipped in v0.25.0 (`68c81e8` typed overflow, `4a8dd0c` staged fold + one shrink-retry + `ShapeHint`). The aw4 16:59 case is pinned as `TestFold_AW4ScaleBacklogGetsThrough`. dun stacks its own `rescue.go` as a second rung (kept, 2026-11-08).
- **next:** live confirmation. Waits on Carl approving an aw4 daemon rebuild; aw4-3b then counts 400 "exceeds the available context size" (3–13/day since 10-29) and TurnOverflow.
- **risks:** `ShapeHint.Scale` only grows and lives for the Session: one bad estimate (e.g. image-heavy) shrinks every later build. A tight window with a large carried summary makes one fold call per entry (no cap). Only a Shaper honours the hint. `ShapeHint.Window` uses `DefaultOutputReservePct`, not the host's reserve. Not run against a live endpoint.
- **blocking decisions:** the aw4 daemon rebuild (Carl).
- **aw4 side:** treat TurnOverflow as final for the session (aw4 issue i_1m6sy2eb).
- **optional extensions:** log the fold stage count (aw4 asked); decay or reset `Scale` after a successful round; a cap on fold stages.

### ◐ `taskgraph` — a plan the model keeps, shared by sub-agents (2026-11-03)
Ported from dun's `internal/taskgraph`. `Store` is interface-only (host provides it); `Diff`/`Apply` compute per-write `[]Change` with `Tell` = owners to notify (node owner before/after; dependents' owners on a status move; never the actor); `WithOnChange` hands kept changes to the host.
- Decided 2026-11-09 (Carl): node shape is title, details, done_when, needs; `files` removed; `show` lists title, status, done_when, `show <id>` gives details; unknown fields are refused with a hint (v0.27.0). dun v0.4.1 uses it and migrates its own stored plans.
- Decided 2026-11-12 (Carl): four tools with flat schemas and real `required` lists, not one tool with an `action` enum and not oneOf (uneven provider support): `plan_add` (title, done_when), `plan_update` (id), `plan_close` (id, outcome done|block|drop + account/reason), `plan_show` (optional id; folds in `ready`). API like `files/`: `New`, `Defs`, `Dispatch`, `Wrap`, `Has`. Every refusal starts with `ERROR: plan: ` and names the call to retry. Cause: a model (dun, 2026-11-12) sent add without done_when, got a bare refusal, reasoned, and ended the turn; raw traffic was not recorded, so the backend (llama.cpp via corrallm, or OpenRouter/LiteLLM) is not known.
- **next:** confirm `Tell` against a real delegation (needs dun to delegate).
- **risks:** the delegator is NOT told (host adds it; the graph has no delegator record).
- **blocking decisions:** none open.
- **optional extensions:** a per-owner subtree view for a sub-agent's `plan_show`; a `delegate` action.

### ◐ `loopdetector` — loops with no period (2026-10-31)
Owner: centralize loop detection here, input is session + stream chunk (no cache), output is a loop count, the cut is a configurable tolerance. Package and `llm` seam done (`Client.Watch`, `StreamWatcher`, `ToolCallWatcher`); the default guard watches reasoning too; a proxy's `content_filter` + `loop` reads as StopReasonRepetition. Measured on 1,315 sessions of corrallm traffic: fires on 4, including a real 40-sentence reply cycle the exact guard missed.
- Evidence 2026-11-06: dun installs it count-only (`cmd/dun/loopscmd.go`, mode `count`).
- **next:** read dun's counters; the numbers decide when text cutting turns on.
- **risks:** defaults (Window 48, MinShare 0.7, Similarity 0.85): 0 cuts at Tolerance 3 in aw4's 13,454 thinking blocks (48 h); no real runaway in thinking seen yet, so that side is unproven. aw4's tool sequences were not replayed.
- **decisions (owner 10-31):** no polling exemption. dun keeps `loopguard.go` beside this; aw4 keeps `MaxRepeatedExchanges`.
- **assumptions:** `agent` must not import it (invariant 2). Tool arguments get the exact detector only: arguments can rightly hold text from earlier in the conversation. A person asking again is not a loop. Host notices do not end a turn.
- **blocking decisions:** when to turn cutting on (Carl), after the counters.
- **optional extensions:** digit-insensitive matching for random data inside a line.

### ◐ Claude Messages API (`llm/anthropic.go`, `docs/features.md`)
Live against api.anthropic.com through dun. Gaps, proposals, next steps and blocking decisions: `plan/claude-api-plan.md`.

## Decisions / conventions
- Module path `github.com/iodesystems/agentkit` is final.
- `llm` and `mcpmgr` stay zero-internal-dep (`go list -deps ./llm ./mcpmgr`).
- Neutral seam: `agent` imports only `agentkit/llm` + stdlib. It must never import a host's `events`/storage model. If a host type is tempting, add a field to `Entry` or a method to `Store`.
- Zero third-party dependencies (invariant 0); adding a `require` is a decision.
- Pre-release: no compat shims, edit in place, delete freely.
- Unsupported media fails loudly; agentkit never strips a part (Carl, 2026-11-06).
- Capability record (Carl, 2026-11-07/08): agentkit does not store it (the host does, via OnLearn); limits only narrow downstream (smallest wins, each lowering reported); corrallm publishes it on `/v1/models`, read as `declared`; `ResendOnRefusal` off by default.
- No caller ever gets a tool call it could run with truncated arguments (Carl, 2026-11-08): with loop detection on, corrallm holds tool calls to the end of the reply and drops them on a cut; the shim ends a dropped stream with an `error` event.

## Done pointers
Everything shipped, slices 1–N and the lifecycle, files, money and OAuth work, the capability record slices, the corrallm v0.24.0 upgrade: `plan/done.md`.
