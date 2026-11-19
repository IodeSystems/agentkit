# Effort scale in agentkit — design note (2026-11-18)

Asked by dun-bf (Carl's request). Status: decided (Carl, 2026-11-18, all recommendations), in build. Planner draft, claims checked: dun `applyEffort` (`../dun/internal/harness/effort.go:117-133`), clamp goes ABOVE first today (`llm/capabilities_learn.go:183-188`, `plan/capabilities-design.md:116`).

## Ask
1. Levels, ordered: none, minimal, low, medium, high, xhigh, max; plus unset (send nothing).
2. One level → wire table per dialect in agentkit; dun drops `applyEffort`.
3. Clamp an unsupported level to the nearest supported one from the record (learned + declared). Never fall to unset without saying so.
4. Report the effective level + reason per request; see a mid-turn change.
5. Supported set per model, with a reason per unavailable level (dun's /effort picker).
dun keeps the saved level and shows saved vs effective.

## Exists today
- Dialects: llamacpp, anthropic, openrouter, openai (`llm/reasoning.go:28-34`). No Gemini dialect.
- Messages API forwards only `ReasoningEffort`; the shim maps none/minimal → low, passes low..max, drops effort for older models (shim `shim.go:663-676`; read from the local checkout, go.mod pins v0.5.0).
- dun's wire today: Anthropic compat off → nothing (honest off), **high → nothing (no thinking: a dun bug)**; OpenRouter low/medium → `reasoning.max_tokens` 1024/4096, high → effort high; llama.cpp off → enable_thinking false, low/medium → true + 1024/4096, high → true, no budget.

## API (new `llm/effort.go`)
- `type Effort string`: `EffortUnset ""`, none … max. `Efforts` (ordered), `EffortAbout` (descriptions), `ParseEffort`.
- `EffortChoice{Wanted, Effective Effort; Native bool; Reason, Cause string}`. Native false = approximated by a budget. Cause = fact key or `dialect:<name>`.
- `EffortFor(dialect, messagesAPI, model, caps, e) EffortChoice` (pure) + `(*Client).Effective(e)`.
- `EffortSupport(...) []EffortChoice`, one per level, + `(*Client).EffortSupport()`.

## Composition (one path)
- `ChatOpts.Effort` replaces `ReasoningEffort` and `EnableThinking`. `ReasoningBudgetTokens`/`ReasoningBudgetMessage` stay as an explicit cap that overrides the level's budget (aw4 sends 2500).
- `chatBody` → `applyReasoning` resolves `EffortFor`, writes the dialect's fields. `translateEffort` becomes a wrapper on the same clamp core (passthrough keeps working). Probe stays raw.
- `agent` `reasoningOff` → `Effort=none` + budget 0.

## Table
| level | openai | openrouter | llama.cpp | anthropic compat | Messages API |
|---|---|---|---|---|---|
| unset | — | — | — | — | — |
| none | "none" | enabled:false | enable_thinking false + budget 0 | no thinking block | → low ("thinking cannot be off") |
| minimal | minimal | effort minimal | true + B_min | budget 1024 (= low) | → low |
| low/medium | as is | effort (decision 2) | true + 1024/4096 | 1024/4096 | as is |
| high | high | high | true, no budget | B_high (fixes dun bug) | high |
| xhigh | xhigh | xhigh (unverified) | → high | B_xh | xhigh (unverified) |
| max | → xhigh | → xhigh | → high | B_max | max |
Old Claude models on Messages API: effective unset, reported "model predates effort".

## Clamp
Available = what the dialect expresses minus what the record refuses (`efforts.refused`, `thinking_only` refuses none, `sampling.reasoning_effort=refused` → nothing sendable). none uses `reasoning.off` if known. Never step DOWN to none unless none was asked. Nothing expressible → unset with a reason. Reason built from fact key + Source + clipped Evidence, or the static row.

## Mid-turn event
Reuse `OnTranslate`: add `Translation.Reason`, emit `Key:"effort"` per request when Effective ≠ Wanted. The resend rebuilds the body after a learned fact (`llm/client.go:1728`), so the new level is reported in the same call, after `OnLearn`. No new hook.

## DAG
N1 `llm/effort.go` + table tests (sonnet) → N2 ChatOpts fields + `applyReasoning` rewrite + golden bodies (sonnet) → {N3 `translateEffort` on the core, `Translation.Reason`, probe bypass (sonnet); N4 `reasoningOff` (haiku); N5 fix callers/examples (haiku)} → {N6 mid-turn test (sonnet); N7 docs (sonnet)} → N8 guardrail + audit (haiku).

## Risks
- dun wire changes: OpenRouter low/medium (if decision 2 = effort), llama.cpp off gains budget 0, Anthropic-compat high gains a budget.
- Removing fields breaks autowork3/corrallm/aw4 at compile time.
- Unverified: xhigh on OpenRouter/Anthropic, max with shim v0.5.0.
- unset cannot say what the server does (corrallm `thinking_default` is not in the record).

## Decisions (Carl, 2026-11-18)
1. Clamp: LOWER first for a refused non-none level (reverses today's above-first; aw4/corrallm see it). none still never reached by stepping down.
2. OpenRouter low/medium: native `reasoning.effort`; a budget only when the caller sets one.
3. Budgets: llama.cpp minimal 512; anthropic compat minimal 1024 (= low), high 16k, xhigh 32k, max 64k.
4. Remove `ReasoningEffort`/`EnableThinking`; `ChatOpts.Effort` is the one path.
5. No native Gemini dialect.
Also (dun-bf, 2026-11-18): `thinkingOnlyRe` misses "Reasoning is mandatory" / "reasoning is required" alone.

## Optional extensions
corrallm `thinking_*` from `/models` into declared facts (so unset can say what the server does); `reasoning_effort` via chat_template_kwargs for gpt-oss on llama.cpp.
