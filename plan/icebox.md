# agentkit — icebox

> Deferred, opt-in next steps. Each entry: RESUME CONDITION (checkable), WAITS ON
> (a decision / a long agent run / a system), EVIDENCE. No checkable condition means
> a wish, and a wish is deleted, not parked. Active work: `plan/plan.md`. Finished
> work: `plan/done.md`.
> Slice-level "Not done" and "Known limits" lists (slices I, J, K) stay in their
> `done.md` entries; they have no resume condition.

## Media and token counting

### Images in tool results
Today only a USER entry carries `Parts`; `renderEntry` drops them on `KindToolResult`,
and `mcpmgr/protocol.go` renders an MCP `image` block as `[image: <full base64>]`
text. The shim already maps image parts inside a `tool` message to a Claude
`tool_result` image; OpenAI-compatible servers reject images in a `tool` message.
- **Resume condition:** a host needs a tool to show the model a picture.
- **Waits on:** a decision (Carl): tool_result image on Claude + follow-up user
  message elsewhere, vs always a follow-up user message.
- **Evidence:** dun (`images.go`, user-side only, x/image scaling) and aw4
  (`toolpicture`, newest-two-shown aging) each built their own; image aging is the
  part that may belong here. The base64-as-text render is a bug to fix regardless.

### aw4 modelgate counts media parts as free
`aw4/internal/agent/modelgate.go` `promptTokenEstimate` counts `image_url` as 85 and
every other part by `len(p.Text)`, so an `input_audio`, `video_url` or `file` part
counts 0.
- **Resume condition:** aw4 sends a non-image media part to a gated model.
- **Waits on:** aw4 (call a shared estimator or copy the per-kind rules).
- **Evidence:** read 2026-11-06 (lines ~494–520); agentkit's own `messageTokens`
  counts them (`agent/mediatokens.go`).

### shim `EstimateTokens` counts base64 as text
It adds `len(content JSON)/4`, so a 1 MB PDF or image counts ~350k tokens where
Claude bills a few thousand.
- **Resume condition:** a Messages-API front end (corrallm `count_tokens`) answers
  for requests carrying media.
- **Waits on:** a system (corrallm `count_tokens`).
- **Evidence:** `reverse.go` `EstimateTokens`, read 2026-11-06; it errs high, so it
  is a cost of folding early, not a refusal.

## Notifications and RAG

### Daemon-side active recheck
A notice carrying a shell/MCP command the daemon RUNS. NOT built: arbitrary shell
from a notification payload is a privilege-escalation surface. Superseded by the
hook + masked-MCP convention (integrator owns the check, slices 5a/5b).
- **Resume condition:** an explicit execution-context + auth design exists.
- **Waits on:** a decision (Carl).
- **Evidence:** slice 5a reframing, `done.md`; no consumer asked for it since.

### Async RAG-notify (zero hot-path latency)
Slice 8's `FinderPreparer` calls `Find` synchronously before build, so a user-message
turn eats a ragtag round-trip (bounded by `Timeout`, fail-open). Option: kick `Find`
in a goroutine, inject via the inbox, hit lands next Turn (same lag assistant hits
already have).
- **Resume condition:** a measured sync `Find` round-trip hurts turn latency.
- **Waits on:** a system (a latency measurement from a host).
- **Evidence:** none measured; `Timeout` bounds the cost.

### RAG-notify watermark/dedup persistence
Slice 8 keeps watermark + seen-DocID sets in an in-mem closure (one preparer per live
session). A host that recreates the Session/preparer per Turn re-observes history and
re-pings. Fix: make the watermark/seen store pluggable (a small interface, like
`RevalidateStore`).
- **Resume condition:** a host builds a new Session/preparer per Turn.
- **Waits on:** a host (dun and aw4 keep one per live session today; not re-checked
  2026-11-06).
- **Evidence:** `agent/finder.go` closure state.

### Supersede vs. shown stacking
Supersede (merge replace) folds only UNSHOWN accumulators, so a new emit while an old
same-key notice is already shown can briefly leave two in context until the
preparer/TTL/clear catches it. Tighten to "delete prior non-compacted + insert".
- **Resume condition:** two same-key notices seen together in a model context.
- **Waits on:** a host report.
- **Evidence:** acceptable today; reasoning only, never observed.

### `select_indexes` MCP tool (runtime index selection)
- **Resume condition:** a host must change the searched index set mid-session.
- **Waits on:** a host request.
- **Evidence:** slice G: static selection through `ragnotify.MCPFinder`
  `Opts.ExtraArgs {"index":...}` covers the ask (verified over MCP stdio).

### Eager summaries for oversized fragments
- **Resume condition:** pointer-notify (fetch on demand) proves insufficient for
  oversized fragments in real use.
- **Waits on:** a host report; costs extra tokens and goes stale.
- **Evidence:** fragment sizing decision (raglit `6f4a78e`), `done.md`.

### Segmentation quality eval
Segmentation was verified to work (`auth.go` → 4 function-level fragments, live) but
there is no systematic eval across diverse documents or prompt tuning.
- **Resume condition:** a small eval corpus exists, or a real corpus shows bad
  fragment boundaries.
- **Waits on:** a decision (Carl) to build the corpus; the run itself is an agent run.
- **Evidence:** slice F live verification only.

### Real ragtag search-result JSON for `ragnotify.ParseHits`
Only matters when ragnotify points at actual ragtag; raglit's serve output already
matches the parsed shape (loop-closure test in raglit `serve_test.go`). Also confirm
ragtag reranks server-side.
- **Resume condition:** ragnotify is pointed at a real ragtag.
- **Waits on:** Carl, to paste one result.
- **Evidence:** slice 8 and slice 9 entries, `done.md`.

## autowork3 and dun follow-ups

### Slice 3 leftover: the Validator in autowork3
No `ValidatingDispatcher` / `agent.Validator` reference in autowork3; the
dispatcher-enforcement retry for protocol-bound roles under `tool_choice=required`
has not moved onto the primitive. Lifting, injection/batching and the notification
lifecycle are wired (`done.md`).
- **Resume condition:** autowork3 bumps agentkit past v0.6.0 and wants the primitive
  instead of its hand-rolled per-tool rejection (adopting changes behavior).
- **Waits on:** a decision (Carl) and the autowork3 bump.
- **Evidence:** grep of autowork3 at the time of the slice 3 re-check; not re-run
  2026-11-06 (autowork3 go.mod pins v0.6.0).

### Slice 4 leftover: reread dun for reach-arounds
Walk dun for places it reaches AROUND the contract rather than wrapping it; those mark
the contract being wrong. Known so far: func-type seams took wrappers cleanly
(`withLiftedQueue`); the one gap, server→client MCP push, is fixed.
- **Resume condition:** before the next breaking change to `agent.Store`,
  `ToolDispatcher` or `Session` (the reread would shape it).
- **Waits on:** an agent run (read-only review of `../dun`).
- **Evidence:** dun is on agentkit v0.23.0 (read 2026-11-06); no reread has been done.
