// Package agent is agentkit's batteries-included agent client over an
// OpenAI-compatible endpoint. It owns the TABLESTAKES every real client
// needs — so a consumer doesn't re-wrap them:
//
//   - the tool-call LOOP (chat → tool_calls → execute → feed back → repeat
//     until the model stops calling tools or a terminal tool fires)
//   - context COMPACTION + LOD truncation (the Shaper) to fit the window
//
// Slice 3 (not yet wired here; see plan/plan.md) adds message/notification
// INJECTION, LIFTING (async tool results), queued-message BATCHING, and
// grammar / JSON-Schema VALIDATION with a fix loop.
//
// What agent does NOT own is ORCHESTRATION — roles, a task DAG, scheduling,
// when/why to run. That's the harness's job (e.g. autowork3), which drives
// an agent.Session and implements the small interfaces below. The claude-
// openai project is another consumer with its own Store impl.
//
// The neutral seam: agent works in terms of llm.Message on the wire and
// Entry for persistence/shaping. It never imports a host's event model —
// the host maps its own rows onto Entry and back.
package agent

import (
	"context"
	"errors"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// EntryKind classifies a conversation entry for shaping + rendering. The
// host maps its own event types onto these; anything that isn't one of the
// first five maps to KindNotification.
type EntryKind string

const (
	KindUser         EntryKind = "user"         // a human/injected message
	KindAssistant    EntryKind = "assistant"    // an LLM reply
	KindToolCall     EntryKind = "tool_call"    // the model asked to call a tool
	KindToolResult   EntryKind = "tool_result"  // a tool's result
	KindCompaction   EntryKind = "compaction"   // a marker that subsumes older entries
	KindNotification EntryKind = "notification" // system push (nudge, friction, …); tail-kept, budget-neutral
)

// Entry is one durable conversation record — the neutral shape the Store
// persists and the Shaper reasons over. A host maps its own rows onto this
// and back; fields agent doesn't interpret (Tag, Origin) are round-tripped
// verbatim.
type Entry struct {
	ID      string
	Kind    EntryKind
	Content string
	// Parts, when non-empty, carries MULTIMODAL content (text + image parts)
	// for this entry and replaces Content when the entry is rendered to an
	// llm.Message. Content should still be set to a plain-text summary of the
	// parts: it is what LOD truncation, compaction summaries, and any
	// text-only consumer will see, so an entry with Parts but an empty Content
	// goes blank the moment the shaper substitutes a stub.
	//
	// Empty Parts (the overwhelming majority) renders exactly as before, so
	// existing hosts and stores are unaffected. A Store that does not persist
	// Parts simply loses the attachment across a reload, degrading to the text
	// Content — acceptable, and the reason Content stays authoritative.
	//
	// Only KindUser is rendered multimodally today; provider APIs accept image
	// parts on user messages, not on tool results or assistant turns.
	Parts      []llm.ContentPart
	ToolCallID string // correlates KindToolCall / KindToolResult

	// Usage records what the round-trip that PRODUCED this entry cost, stamped
	// on KindAssistant entries. It lives on the entry — and therefore in the
	// session log — rather than only in a trace sidecar, because the log is the
	// durable record: a replay can then answer how long each turn took and what
	// it spent without a second file that may not exist.
	//
	// Nil on every other kind, and omitted when marshalled, so stores and
	// consumers that predate it are unaffected.
	Usage    *EntryUsage `json:"Usage,omitempty"`
	ToolName string      // the tool for KindToolCall / KindToolResult
	// Tag is an opaque display label used only when rendering a
	// KindNotification (e.g. the host's raw event-type "nudge"). Empty →
	// the Kind string is used.
	Tag string
	// Origin is an opaque host provenance tag. agent never reads it; it is
	// carried on entries returned by Store.Context and handed back inside
	// Compaction.Subsumes so the host can route subsumed rows to the right
	// storage (e.g. autowork3's public-delivery vs private-log streams).
	Origin    string
	CreatedAt int64 // ns; ordering key
}

// Compaction is what the Shaper hands Store.Compact: a summary marker plus
// the entries it subsumes. The host writes the marker + flags every
// subsumed entry (routing by Entry.Origin) in one transaction.
type Compaction struct {
	Marker   Entry   // Kind=KindCompaction, Content=summary, CreatedAt already placed
	Subsumes []Entry // the entries folded into Marker (Origin intact)
}

// Store is the minimal conversation persistence the loop needs. A host
// (autowork3's event tables; the claude-openai project's own storage; an
// in-memory slice in tests) implements it. No host event types leak here.
type Store interface {
	// ClaimPending marks the session's pending inbox arrivals as shown and
	// returns how many there were. The loop uses the count to decide whether
	// a re-prompt is warranted; the entries themselves surface via Context.
	ClaimPending(ctx context.Context, sessionID string, at int64) (int, error)
	// Append persists one entry (an assistant reply, a tool result, …).
	Append(ctx context.Context, sessionID string, e Entry) error
	// Context returns the session's non-subsumed entries (any order; the
	// engine sorts). The host merges whatever internal streams it has into
	// this single list.
	Context(ctx context.Context, sessionID string) ([]Entry, error)
	// Compact records a compaction marker + flags the subsumed entries, in
	// one transaction.
	Compact(ctx context.Context, sessionID string, c Compaction) error
}

// LLMRunner does one streaming chat round-trip. opts is optional — nil means
// default behavior. It carries ToolChoice for forcing a tool call (used by
// protocol-bound roles whose only exit is a structured terminal tool).
type LLMRunner interface {
	ChatStream(ctx context.Context, messages []llm.Message, tools []llm.ToolDef, opts *llm.ChatOpts) (<-chan llm.StreamChunk, error)
}

// ToolDispatcher executes a tool call and returns the string the model will
// see as the tool result. Errors meant to reach the model (unknown tool,
// bad args) should be formatted into the result; return a non-nil error only
// to abort the whole turn. Returning ErrSessionClosed tells the loop to stop
// after the result is persisted — used by terminal tools so the loop doesn't
// keep prompting an already-closed session under tool_choice=required.
type ToolDispatcher func(ctx context.Context, tc llm.ToolCall) (string, error)

// ContextBuilder materializes the message list shown to the LLM from the
// session's history. DefaultContextBuilder is the plain merge; a Shaper adds
// pristine-tail / LOD / compaction on top.
type ContextBuilder func(ctx context.Context, sessionID string, system string) ([]llm.Message, error)

// ErrSessionClosed is the sentinel a terminal-tool dispatcher returns to tell
// Turn the session was closed and no further chat rounds should fire.
var ErrSessionClosed = errors.New("agent: session closed by terminal tool")

// CompactionInfo describes a compaction the Shaper performed during a Turn. It
// is surfaced (via Turn's result and the OnCompactionComplete callback) so the host can
// persist the summary as a hidden field on the turn and show meta about what
// was folded. Token counts are the active estimator's numbers.
type CompactionInfo struct {
	Summary       string // the tight summary that replaced the folded prefix
	SubsumedCount int    // how many entries were folded into the summary
	TokensBefore  int    // estimated active-window tokens before this compaction
	TokensAfter   int    // estimated active-window tokens after

	// Duration is how long the fold took, dominated by the summarizer's own
	// round-trip. Reported because compaction is an unasked-for LLM call in the
	// middle of somebody's turn: it is the one part of a slow turn that is not
	// the work, and a host that cannot separate it will read its own latency
	// wrong. It is also the number that says whether compacting is cheaper than
	// the context it saves.
	Duration time.Duration
}

// CompactionStart is delivered to Session.OnCompactionStart before the fold
// begins.
//
// It exists because compaction is the one thing that happens to a turn without
// anybody asking for it, and it is slow. Reported only on completion, a host has
// no way to say "this is compacting" while it is happening — the turn simply
// stalls, and the reason arrives after the wait is over.
type CompactionStart struct {
	SessionID string
	// TokensBefore is the active window that triggered the fold.
	TokensBefore int
	// Target is the size the shaper is folding down to. Together with
	// TokensBefore it says how much has to go.
	Target int
	// Subsuming is how many entries are about to be folded.
	Subsuming int
	At        time.Time
}

// ToolFormat selects how tool calls travel between the model and the loop.
type ToolFormat string

const (
	// ToolFormatNative uses the provider's own tool_calls field. Default, and
	// correct for any provider whose template round-trips arguments faithfully.
	ToolFormatNative ToolFormat = ""

	// ToolFormatHeredoc carries calls as json-loose-heredoc in ordinary content,
	// grammar-constrained, parsed client-side.
	//
	// Use it where the native format loses data. Measured on Qwen3: a value
	// containing `</parameter>` comes back truncated at the delimiter, and the
	// truncation is the MODEL's — visible with the server's parser bypassed — so
	// no provider or template fix reaches it. The heredoc body has no such
	// delimiter and carries the value verbatim.
	//
	// It also costs less: 42% fewer prompt tokens and 8% fewer generated on a
	// four-tool set, because the tool list is a signature line rather than a JSON
	// Schema dump.
	//
	// Requires a provider that accepts `grammar`. llama.cpp refuses a grammar
	// together with `tools`, which is why this format leaves the native path
	// rather than extending it. Probe with llm.ProbeToolSurface.
	ToolFormatHeredoc ToolFormat = "heredoc"
)

// TokenUsage is the session-level token accounting surfaced every Turn.
type TokenUsage struct {
	// Total is cumulative prompt+completion tokens billed across every chat
	// round of this Session's lifetime (what you paid). A host that reuses one
	// Session across a whole conversation sees the running total; one that makes
	// a fresh Session per Turn should persist + sum these itself.
	Total int
	// Active is the token count of the CURRENT live window — the built
	// (compacted + LOD) context the model sees now, INCLUDING any compaction
	// summary. This is the underlying-session size, distinct from Total.
	Active int

	// Cached, Processed and Generated split Total into the three channels that
	// actually price differently. Total alone is close to useless for judging
	// cost: it is dominated by the re-sent prefix, which is the CHEAPEST part.
	// Measured on one agent run, Total was 97% Cached — so ranking two runs by
	// Total ranks them by how much cache they read.
	//
	//   Cached    prompt tokens served from the provider's cache. Nearly free
	//             (llama.cpp reuses the KV slot; ~17x faster than generation).
	//   Processed prompt tokens the provider actually had to evaluate. In an
	//             agent loop this is mostly TOOL RESULTS — the honest per-turn
	//             input cost, and the thing a tool surface controls.
	//   Generated completion tokens. The most expensive per token, and usually
	//             the binding constraint, since output caps are hit long before
	//             context limits.
	//
	// A provider that reports no cache breakdown leaves Cached at 0 and puts the
	// whole prompt in Processed, which is the correct reading for that provider.
	Cached    int
	Processed int
	Generated int

	// Turns counts chat round-trips accumulated into this tally — the divisor
	// for any per-turn figure.
	Turns int
}

// EntryUsage is the per-call cost of one round-trip, persisted with the entry
// it produced. Deliberately the SPLIT rather than a single total: cached,
// processed and generated tokens price differently by more than an order of
// magnitude, and a lone total is dominated by the cheapest of the three.
type EntryUsage struct {
	LatencyMS int64 `json:"latency_ms,omitempty"`
	Cached    int   `json:"cached,omitempty"`
	Processed int   `json:"processed,omitempty"`
	Generated int   `json:"generated,omitempty"`
}

// TurnResult is what Turn returns: the model's final reply, whatever the loop
// did to the context to keep it in budget, and the running token tally.
type TurnResult struct {
	Reply       string
	Compactions []CompactionInfo // usually empty or one; more if a huge context folds in stages
	Usage       TokenUsage
}

// TurnEndReason names WHY a Turn stopped.
//
// It exists because the loop's six exits collapse into two observable states —
// err nil or not — and a host that wants to tell "the model answered" from "a
// terminal tool closed the session", or "we ran out of rounds" from "the store
// failed", has nothing to key on but the text of an error string. Error strings
// are for people; this is for code.
//
// The clean exits are TurnReply and TurnTerminalTool, and both carry a nil Err.
// Everything else is a failure and carries the error the Turn returned.
type TurnEndReason string

const (
	// TurnReply — the model answered and asked for no further tool calls, with
	// nothing new pending. The ordinary end of a turn.
	TurnReply TurnEndReason = "reply"
	// TurnTerminalTool — a terminal tool fired and closed the session, so the
	// loop stopped rather than running another chat round.
	TurnTerminalTool TurnEndReason = "terminal_tool"
	// TurnMaxTurns — the loop hit MaxTurns without the model settling. Note this
	// counts ROUNDS inside one Turn, not turns in a conversation.
	TurnMaxTurns TurnEndReason = "max_turns"
	// TurnOverflow — the provider cut generation for room MaxOverflowRetries
	// rounds running, and shedding more context stopped helping.
	TurnOverflow TurnEndReason = "overflow"
	// TurnRepetition — the model's generation kept collapsing into repetition
	// past MaxRepetitionRetries, so the loop stopped re-prompting it.
	TurnRepetition TurnEndReason = "repetition"
	// TurnCanceled — the caller's context was canceled or its deadline passed.
	// Separated from TurnError because it is usually not a fault: it is a host
	// shutting a session down, and a host that logs it as a failure cries wolf.
	TurnCanceled TurnEndReason = "canceled"
	// TurnError — anything else: the store, the context builder, or the provider
	// failed. Err carries it.
	TurnError TurnEndReason = "error"
)

// Failed reports whether the reason describes a Turn that did not complete.
// TurnCanceled counts as failed — the turn did not finish — even though it is
// usually deliberate.
func (r TurnEndReason) Failed() bool { return r != TurnReply && r != TurnTerminalTool }

// TurnStart is delivered to Session.OnTurnStart once, before the loop's first
// chat round.
type TurnStart struct {
	SessionID string
	ThreadID  string
	At        time.Time
}

// TurnEnd is delivered to Session.OnTurnEnd exactly once per Turn, on every
// exit path including a panic-free error return.
//
// Duration is wall clock and is NOT time the model spent working. It includes
// tool execution, client-side 429 backoff, and any time the proxy held the
// request in its admission queue. Anything charging a budget against elapsed
// time must decompose it or it prices the same work differently depending on
// how busy the box was — which is what a budget exists to factor out.
//
// What each part needs:
//
//   - proxy queueing and cold loads → ServerOverheadMS, below.
//   - client-side 429 backoff → Client.OnRetry (RetryEvent.Delay/Elapsed).
//   - tool execution → the host's own dispatcher timing; agent does not measure
//     it, because only the host knows which tools are its own work and which
//     are another wait.
type TurnEnd struct {
	SessionID string
	ThreadID  string
	// Reason is why the loop stopped. Always set.
	Reason TurnEndReason
	// Err is the error Turn returned, nil on the clean exits.
	Err error
	// Rounds is how many chat rounds the loop ran, 1-based.
	Rounds int
	// Usage is the running tally as of the end of the turn.
	Usage TokenUsage
	// Duration is wall clock from OnTurnStart to here. See the type comment
	// before charging anyone for it.
	Duration time.Duration

	// ServerOverheadMS is what the proxy reported spending NOT running the
	// model — admission queueing plus cold loads — summed over the turn's
	// rounds. Subtract it from Duration to stop billing a caller for the queue
	// it waited in.
	ServerOverheadMS int64
	// ServerTimingSeen distinguishes "the proxy reported no overhead" from
	// "nothing reported". A plain OpenAI endpoint says nothing, and a host that
	// read a confident zero there would subtract an overhead it never measured.
	// False means ServerOverheadMS is unknown, not zero.
	ServerTimingSeen bool
}

// compactionSinkKey carries a compaction reporter through the ctx, so the Shaper
// surfaces compactions to the Session WITHOUT changing the ContextBuilder
// signature. A plain builder simply never reports; a Session installs the sink
// before it calls build().
type compactionSinkKey struct{}

func withCompactionSink(ctx context.Context, fn func(CompactionInfo)) context.Context {
	return context.WithValue(ctx, compactionSinkKey{}, fn)
}

// compactionStartSinkKey carries the START reporter the same way, for the same
// reason: the Shaper is reached through ContextBuilder, whose signature the
// Session does not own.
type compactionStartSinkKey struct{}

func withCompactionStartSink(ctx context.Context, fn func(CompactionStart)) context.Context {
	return context.WithValue(ctx, compactionStartSinkKey{}, fn)
}

// reportCompactionStart delivers cs to the sink installed in ctx, if any. The
// Shaper calls it once it knows a fold is happening and what it will cost —
// after the decision, before the summarizer.
func reportCompactionStart(ctx context.Context, cs CompactionStart) {
	if fn, ok := ctx.Value(compactionStartSinkKey{}).(func(CompactionStart)); ok && fn != nil {
		fn(cs)
	}
}

// reportCompaction delivers ci to the sink installed in ctx, if any. The Shaper
// (or any custom builder that folds history) calls this after it compacts.
func reportCompaction(ctx context.Context, ci CompactionInfo) {
	if fn, ok := ctx.Value(compactionSinkKey{}).(func(CompactionInfo)); ok && fn != nil {
		fn(ci)
	}
}
