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
	ID         string
	Kind       EntryKind
	Content    string
	ToolCallID string // correlates KindToolCall / KindToolResult
	ToolName   string // the tool for KindToolCall / KindToolResult
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
// is surfaced (via Turn's result and the OnCompaction callback) so the host can
// persist the summary as a hidden field on the turn and show meta about what
// was folded. Token counts are the active estimator's numbers.
type CompactionInfo struct {
	Summary       string // the tight summary that replaced the folded prefix
	SubsumedCount int    // how many entries were folded into the summary
	TokensBefore  int    // estimated active-window tokens before this compaction
	TokensAfter   int    // estimated active-window tokens after
}

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
}

// TurnResult is what Turn returns: the model's final reply, whatever the loop
// did to the context to keep it in budget, and the running token tally.
type TurnResult struct {
	Reply       string
	Compactions []CompactionInfo // usually empty or one; more if a huge context folds in stages
	Usage       TokenUsage
}

// compactionSinkKey carries a compaction reporter through the ctx, so the Shaper
// surfaces compactions to the Session WITHOUT changing the ContextBuilder
// signature. A plain builder simply never reports; a Session installs the sink
// before it calls build().
type compactionSinkKey struct{}

func withCompactionSink(ctx context.Context, fn func(CompactionInfo)) context.Context {
	return context.WithValue(ctx, compactionSinkKey{}, fn)
}

// reportCompaction delivers ci to the sink installed in ctx, if any. The Shaper
// (or any custom builder that folds history) calls this after it compacts.
func reportCompaction(ctx context.Context, ci CompactionInfo) {
	if fn, ok := ctx.Value(compactionSinkKey{}).(func(CompactionInfo)); ok && fn != nil {
		fn(ci)
	}
}
