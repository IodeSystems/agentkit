# Capability record — design note (2026-11-07)

> Status: design, for Carl. Plan item: `plan.md` "Per-(endpoint, model) capability record". Requested by dun (aw4 issue i_1m6v7hgpwcwrk5pbswp0w6); corrallm will consume it and not build its own.

## Problem

Two endpoints that both speak "the OpenAI API" differ on almost every optional field, and they differ silently. Measured cases:

- Fireworks GLM-5.3 answers 400 to `reasoning_effort: "none"`. Sending nothing makes it think at full strength (median 791 reasoning tokens against 103 at `low`), and a fold with a small `max_tokens` came back empty.
- A llama.cpp template answered 500 to `"high"` (Qwen3.8). For this reason the `openai` dialect is named, never detected.
- The window is unknown on Fireworks. dun's `windowTokens` is 0 there, so it cannot tell a crowded prompt from a reply that hit the output cap.
- llama.cpp refuses a grammar combined with tools. `ProbeToolSurface` measures this, but only on demand, and nothing keeps the answer.

Today each host learns these by hand (dun's `effort_off`, `DUN_REASONING`) or not at all.

## Decisions already made (Carl)

1. Open standard first: the `openai` dialect sends the OpenAI Chat Completions values as given. A model that refuses a standard value is a per-model gap that the record handles, not a global mapping (`6e9c828`).
2. agentkit does not store the record. What it learns is reported to the holder through a callback. dun and corrallm each store it and pass it back when they create a Client. agentkit keeps only the in-memory copy for the Client's life.

## Shape

New file `llm/capabilities.go`. It uses stdlib only, and `llm` stays standalone.

```go
// Capabilities is what one (endpoint, model) is known to do.
type Capabilities struct {
    Endpoint string          // normalized base URL (scheme+host+path, no trailing slash)
    Model    string          // the model name the Client was given, not a resolved alias
    Facts    map[string]Fact // keyed by the Cap* constants below
}

type Fact struct {
    Value    string    // canonical text form; typed getters parse it
    Source   Source
    Evidence string    // what was sent and what came back, short: "sent reasoning_effort=none → 400 'thinking cannot be disabled'"
    At       time.Time
}

type Source string
const (
    SourceHost     Source = "host"     // pinned by the host's config; never overwritten by learning
    SourceRefused  Source = "refused"  // a 4xx naming the field
    SourceObserved Source = "observed" // read off normal traffic
    SourceProbed   Source = "probed"   // an explicit ProbeCapabilities run
    SourceDeclared Source = "declared" // the endpoint's own /models metadata
)
```

The facts are a keyed map of strings, not a struct with typed fields. Three reasons:

- dun's `/model` command shows every fact with its provenance.
- A host stores the record without knowing its keys.
- A key added later does not break a stored record.

Typed getters (`c.Caps().ContextWindow()`, `.OffEffort()`, `.ToolChoiceHonored()`) parse the values and return `(value, known bool)`.

Keys. The first slice implements the ones marked ★.

| key | value | learned from |
|---|---|---|
| ★ `reasoning.off` | the effort that turns thinking off or lowest: `none`, `minimal`, `low` | refusal, observed, declared, probe |
| ★ `reasoning.efforts.accepted` / `.refused` | comma list | refusal, observed |
| ★ `reasoning.thinking_only` | bool | refusal ("cannot be disabled"), declared (`thinking_modes` without `off`) |
| `reasoning.budget_field`, `reasoning.budget_min` | name, int | refusal (GLM: `thinking.budget_tokens` ≥ 1024), probe |
| ★ `limits.context` | int tokens | `InputTooLargeError.ContextTokens`, declared `context_length`, `DiscoverContext` |
| `limits.output` | int tokens | refusal ("max_tokens must be ≤ N"), declared (`modalities.text.maxTokens`, OpenRouter `top_provider.max_completion_tokens`) |
| ★ `stream.usage` | bool | observed (a usage chunk arrived) |
| ★ `stream.reasoning_field` | `reasoning_content`, `reasoning`, `think_tags` | observed (which delta field carried it) |
| ★ `tools.choice_forced` | bool | observed (`tool_choice` required/named, and the reply had no call → false), probe |
| `tools.parallel`, `tools.arg_fidelity` | bool, `exact`/`lossy` | observed, probe (`ProbeToolSurface` echo) |
| `sampling.<param>` | `accepted`/`refused` (logit_bias, banned strings, DRY, …) | refusal |
| `decode.grammar`, `decode.grammar_with_tools`, `decode.json_schema`, `decode.guided`, `decode.strict_tools` | bool | refusal, probe |
| `modality.<image\|audio\|video\|file>` | bool | declared (`modalities`), refusal, `ErrUnsupportedContent` |

## Precedence

When a new fact arrives for a key that already has one:

1. `host` is never replaced by learning. It is the host's override (dun's `effort_off`).
2. Otherwise, a fact backed by evidence (`refused`, `observed`, `probed`) replaces any `declared` fact, and it replaces an older evidence fact of the same key. The latest evidence wins, so a model that is fixed upstream gets learned again.
3. `declared` replaces only `declared`.

A seeded record keeps the sources it was stored with, so a stored `refused` still beats a fresh `declared` after restart. That is the rule from the last conversation: the stored record wins only where the endpoint was seen to refuse or accept, and the endpoint's own metadata wins elsewhere.

There is a trap: a fact learned from one bad response would otherwise persist forever, and now in the host's store as well. This is the same trap as `ShapeHint.Scale`. Every fact therefore carries `Evidence` and `At`, so a host can expire or override it. An observed fact only flips on a clear signal, never on a count of zero alone. The thresholds per key are fixed in the code, with the measurement that set them.

## API

```go
func (c *Client) SetCapabilities(Capabilities)       // seed from the host's store; call before first use
func (c *Client) Capabilities() Capabilities         // snapshot (copy), for /model and for storing
OnLearn func(Learned)                                // Client field, like OnRetry/OnUsage
type Learned struct { Endpoint, Model, Key string; Fact Fact; Previous *Fact }
func (c *Client) LoadDeclared(ctx) error             // one GET {base}/models, opt-in
func ProbeCapabilities(ctx, c) (Capabilities, error) // explicit; dun's /model probe
```

- `OnLearn` is called synchronously after the record's lock is released, from the goroutine that made the request, as `OnRetry` is. It must not block, and it must not re-enter the Client. It fires only when a value or source changes, not on every confirmation.
- Child clients: each Client has its own record keyed by its own (endpoint, model). dun seeds each of its `childClients` from its store separately. agentkit does not share records between Clients.
- `LoadDeclared` is opt-in because it is an extra request. It reads two shapes and ignores fields it does not know:
  - corrallm: `context_length` where > 0, `modalities` keys, and `thinking_default`/`thinking_modes`/`thinking_effort` when present. Only canonical ids are listed, so it matches a lane or alias by id only. A miss is not an error.
  - OpenRouter: `supported_parameters`, `context_length`, `top_provider.max_completion_tokens`.
  - corrallm's `thinking_budget` is corrallm's own cap, not a model limit. It is not read as `reasoning.budget_min`.
- `ProbeCapabilities` wraps `ProbeToolSurface` (echo fidelity, grammar, grammar+tools) and adds:
  - one short request per effort in the standard ladder, which also learns `reasoning.off`
  - one `tool_choice: required` request
  - the window from `DiscoverContext` where the endpoint has a boundary

  It reports each result as a `probed` fact through `OnLearn`, and returns the whole record. It is the only path that spends requests for learning.

## Translation (what the record changes on the wire)

There is one hook, in `applyReasoning` and in `chatBody` (shared by `Chat`, `ChatStream` and `ChatWithOpts`). The first slice changes only reasoning:

- An off request (`ReasoningEffort: "none"`) sends `reasoning.off` when the record knows it. Otherwise it sends the standard `"none"`.
- An effort in `reasoning.efforts.refused` is replaced by the nearest accepted effort above it, and the substitution is noted in the RetryEvent-style hook (see below) so it is never silent.
- A known `limits.output` caps `max_tokens`.

Everything else in the table is read-only for hosts at first (dun asked for `limits.context`, `tools.choice_forced` and `stream.reasoning_field`). The wire changes for those come one key at a time, each with its own call-out.

## Learning from refusals, and the open decision

A 400 whose body names a request field is parsed by `refusalFacts(body, sent)`. This is a table of patterns, each with the server that produced it. Examples:

- `"'none' is not one of"`
- `"Unrecognized request argument supplied: X"`
- `"Unsupported parameter: 'X'"`
- `"Extra inputs are not permitted"` with `loc`
- `"thinking cannot be disabled"`
- `"budget_tokens must be >= N"`

It records the facts and the request still fails as today.

**Open (Carl):** whether `llm` then sends the request again, once, translated by the new fact. Two arguments:

- **For:** this is the only way the first request to a new model succeeds without a probe, and dun asked for it.
- **Against:** it would be a second automatic resend in a path every host shares, beside the overflow resend in `Session.Turn`.

Until Carl decides, it ships behind `Client.ResendOnRefusal bool`, false by default. When true, it is bounded to one resend per call, only for a refusal whose fact changed the request bytes. A refusal that teaches nothing new is returned as is.

## Slices

1. **core**: types, precedence, `SetCapabilities`/`Capabilities`/`OnLearn`, typed getters, tests. Independent of the wire.
2. **learn**: observed facts (stream usage, reasoning field, reasoning tokens against the effort sent, `InputTooLargeError` counts, `tool_choice` honored), the refusal parser, and the reasoning translation. `ResendOnRefusal` is off.
3. **declared**: `LoadDeclared` for the corrallm and OpenRouter shapes, from fixtures of the live bodies.
4. **probe**: `ProbeCapabilities` over `ProbeToolSurface`.
5. Hosts: dun replaces `effort_off` (`d41e9a1`) with the record, stores records in `DUN_HOME`, and shows them on `/model`. corrallm stores per (backend, model) for its dashboard. Both run in their own sessions, and both route changes through agentkit-f2.

Slice 1 lands first. Slices 2, 3 and 4 touch different files (`capabilities_learn.go`, `capabilities_declared.go`, `capabilities_probe.go`), so they can run in parallel. The two shared hooks, in `applyReasoning`/`chatBody` and in the stream reader, belong to slice 2 only.

## Not in scope

- Any storage in agentkit.
- A shared registry across Clients.
- Cost or price facts (`shim.Prices`).
- Learning sampling quality, such as whether DRY helps. The record says what a model accepts, not what works well on it.
