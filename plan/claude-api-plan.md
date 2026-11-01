# agentkit — Claude Messages API plan

> How this plan works: the gaps between the OpenAI shape agentkit speaks and the
> Claude Messages API, with a proposal for each. Rules are in `plan/plan.md`.
> Status marks: ◻ todo · ◐ in progress · ✅ done · ⏸ parked · ❓ blocked.
> Move a finished slice to `plan/done.md` and delete it here.

## State

✅ Built: `llm/anthropic.go` (2ed3442, local, no tag). A client pointed at
`api.anthropic.com`, or with `MessagesAPI: true`, talks to `/v1/messages`. Callers
keep `Message`, `ToolDef`, `ChatOpts`, `StreamChunk`. The request is translated; the
stream is transcoded into the OpenAI SSE that `readStream` reads, so the loop
watcher, usage and truncated-call handling are one code path. Other providers are
unchanged.

Measured live through dun, 2026-11-01:

| Check | Result |
|---|---|
| Opus 5.5, 2 user turns, 4 rounds, tools | Works; 20,058 tokens read from cache |
| Raw `/v1/messages`, Haiku, Sonnet, Opus 5.5 | Second round read 11,484 from cache |
| The same on the OpenAI-compatible endpoint, through dun | 0 read from cache |
| No key on the client, `ANTHROPIC_API_KEY` in the environment | Works |
| Workload Identity Federation | Not tested: needs an identity token |

Decided (Carl, 2026-11-01): keep the SDK dependency for now. **This breaks invariant
0 in `CLAUDE.md`, which still states the old rule**: `go list -m all` went from 1
module to 73, and `llm` is no longer standalone. The invariant was not read when the
SDK was added, and Carl was told the cost in lines, not that it broke a stated rule.
A separate shim project (below) would restore it. agentkit had none before; `anthropic-sdk-go` adds 22 lines to `go.mod` and 137 to `go.sum`, and every
consumer inherits them. Thinking text is not expected back from Claude, so an empty
thinking stream is not a fault.

## Gaps: OpenAI shape to Claude

The caller asks for something the Messages API does not take.

| # | Status | Gap | Now | Proposal |
|---|---|---|---|---|
| 1 | ✅ | Forced `ToolChoice` (object or `required`) | `required` is `{type:any}`, a named function `{type:tool,name:X}` | Done. On Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1, which return a 400, it is `auto`; a named function is then the only tool sent, and `required` stays `auto` over every tool (`noForcedToolChoice`) |
| 2 | ◻ | `ResponseFormat` | Ignored | Map to `output_config.format` |
| 3 | ⏸ | `Grammar` | Ignored | None. It belongs to the heredoc transport, which this path does not use |
| 4 | ⏸ | `Temperature`, `Seed`, penalties | Not sent | None. Current models reject them |

Only slice 1 mattered to dun today, and it is done:

- `fork.go` forces one tool on a side call. With `auto` and every tool, the model
  could answer in text or call another tool, and the fork returned nothing. On the
  models that reject forced choice it can still answer in text, but cannot call
  another tool.
- `suggest.go` is not affected. It passes `tool_choice` `none`, which already worked,
  and uses `ResponseFormat` only in heredoc mode, which this path does not use. So
  gap 2 has no consumer yet.

**next:** gap 2 only when a host needs `ResponseFormat` on this path. **risks:** which
models reject forced choice changes by release; `noForcedToolChoice` is a list in
code, and asking `/v1/models` would be better if it ever reports this. **blocking
decisions:** none.

Known cost of gap 1: tools lead the cached prefix, so a forced named tool on Opus
5.5, Sonnet 5.5, Fable 5.1 or Mythos 5.1 narrows `tools` and that call misses the
cache and rewrites it. Kept: a side call that can only call that tool is worth one
miss, and dun's one such caller, `/remember`, is rare and reports a text reply as an
error.

## Gaps: Claude features the conversion flattens

| # | Status | Gap | Now | Proposal | Size |
|---|---|---|---|---|---|
| 5 | ◻ | Tool errors | Text, `is_error` false | Carry an error flag on the tool `Message`; dun's results start `ERROR:` | Small |
| 6 | ◻ | Refusals | A stream error; the turn fails | `fallbacks: "default"` on the beta endpoint for Opus and Sonnet 5.5 | Small |
| 7 | ◻ | Rate limits and cost | `responseMeta{}` is empty | Read the response headers through SDK middleware | Small |
| 8 | ◻ | Retries | SDK retries; `RetryBudget`, `OnRetry` and `/retry budget` do nothing | Report and bound them in SDK middleware | Medium |
| 9 | ◻ | Host notices | User messages | Mid-conversation system messages, which keep the cached prefix | Medium |
| 10 | ◻ | Cache lifetime | 5 minutes; an idle session misses | Option for the 1-hour TTL | Small |
| 11 | ◻ | Thinking blocks | Not replayed | See below | Large |
| 12 | ◻ | Context measuring | `tokenize.go` routes do not know this API | `count_tokens` | Small |
| 13 | ◻ | Spend | dun's meter showed no cost for this provider on the compatible path; not re-checked here | Check first, then price from usage | Small |

**Thinking replay (11).** `Message` has no place for a signed block, so nothing is
sent back, and no request has failed for it. That is 4 rounds on Opus 5.5, not a long
session. Replay needs a field on `llm.Message`, on `agent.Entry` and in the session
file, and a rule for folds: a block is valid only for the conversation that produced
it, so a fold or an edited tool result invalidates what follows.

**next:** run one long tool session on Opus 5.5 and compare work done with and without
replay. Do not build replay on a guess. **risks:** storing blocks grows session files;
a fold that keeps a block can be a 400 on an enforced account. **blocking decisions:**
whether replay is worth a change to `Entry` (Carl).

## Not exposed

Nothing depends on these; a host folds its own history. Each needs a field on
`ChatOpts` or a new surface.

- Server compaction and context editing
- Server tools: web search, web fetch, code execution
- Citations and document blocks
- Task budgets
- Strict tools
- Batches
- Thinking summaries (`display`)

## Credentials

◐ With no key on the client the SDK resolves credentials from the environment. That
is the path Workload Identity Federation takes: `ANTHROPIC_FEDERATION_RULE_ID`,
`ANTHROPIC_ORGANIZATION_ID`, `ANTHROPIC_SERVICE_ACCOUNT_ID`,
`ANTHROPIC_IDENTITY_TOKEN_FILE`. Any of `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or
`ANTHROPIC_PROFILE`, set at all, shadows it.

**next:** test against a real federation rule. **risks:** a laptop has no identity
token, so this fits CI and server runs, not interactive dun. **blocking decisions:**
which workload (Carl). He is an admin of his personal org, not of the corp one.

## Release

◻ Tag agentkit, then dun bumps it. dun needs no code change.

**blocking decisions:** the version (Carl). Slice 1 is in; slice 2 has no consumer
and need not block it.

## Future: corrallm owns Claude

⏸ Direction (Carl, 2026-11-01): the key lives in corrallm, and corrallm does this
translation for every client. A client speaks OpenAI to corrallm and Claude works:
aw4 and dun manage no Claude pathway, and no client holds a Claude key.

**corrallm already has a translator.** `internal/proxy/anthropic.go` (51142ad, 1,055
lines) is a shim for a backend with `api: anthropic`: OpenAI in, `/v1/messages`, OpenAI
JSON or SSE back, with its own cache breakpoints. This file was written without
knowing that. There are two translators now, and they should become one, in agentkit,
which corrallm already imports.

| | corrallm shim | `llm/anthropic.go` |
|---|---|---|
| Cache breakpoints | `placeCacheBreakpoints`, TTL option, cap of 4 | System and end of conversation, 5 minutes |
| Non-streaming replies | Yes | Collected from the stream |
| Unsupported content | Refused with a 400 | Dropped or passed as text |
| Forced `tool_choice` | Sent; the provider's 400 is passed back | Sent, or `auto` on models that reject it |
| Effort | Dropped | `output_config.effort` |
| Broken transcripts | Not read | Repaired (missing results, orphans) |
| Credentials | Key, caller auth | SDK: key or federation |
| Transport | Raw HTTP | SDK |

Also in scope, from corrallm-f2's review of this table (it read the code and ran
nothing):

- `internal/proxy/messages.go` (1,017 lines) is a reverse translator: Anthropic in,
  OpenAI backends out. Claude Code depends on it. "One translator" must say whether
  it stays.
- `callerAuth` is a subscription passthrough: the caller's own sign-in goes to
  Anthropic. "The key lives in corrallm" does not cover that path.
- Caller-placed `cache_control` markers are kept verbatim and count against the cap
  of 4.
- Cache writes bill at 1.25× (2× for 1h); corrallm's cost model counts them at 1×.
- `loop`: the detector sits behind the translator and reads OpenAI events, so a shared
  translator must keep emitting OpenAI SSE ahead of it. `/v1/messages` is not watched.

Proposed (Carl, undecided): a separate Go project, `openai-claude-api-shim`, pure
bytes, no HTTP, no SDK, no agentkit import. agentkit and corrallm both call it.

What it takes:

- Export the two halves from agentkit. The proxy has an OpenAI request and must write
  OpenAI SSE, so it needs them as public functions on bytes, not only through
  `Client.ChatStream`.
- Take the better half of each row above, then delete corrallm's copy.
- Pass through what corrallm already sends a client: usage, and `loop` from its own
  detector.
- Gaps 1 to 13 are then corrallm's too. Each one fixed here is fixed there.

**next:** nothing until the direct path has run in daily use. **risks:** a client that
sends OpenAI `tools` gains nothing it lacks today, but thinking replay (gap 11) would
need a field the OpenAI shape does not have. **blocking decisions:** when; and whether
the direct path stays for a host with no corrallm in front of it. Recommended: keep it,
because `llm/anthropic.go` is the same code either way.

## Assumptions

- `api.anthropic.com` in the base URL is the right trigger. A proxy in front of Claude
  stays on the OpenAI path unless the host sets `MessagesAPI`.
- Effort `none` is `low`, because thinking cannot be off on every model. Sonnet 5.5's
  `between_tools` is not used.
- `anthropicEffort` lists the models that reject effort by name.
- `max_tokens` defaults to 64,000.
