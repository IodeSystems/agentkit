package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/iodesystems/agentkit/llm"
)

// LODKeep says which end of a truncated tool result carries its answer.
//
// A file read is answered at the top and a test run at the bottom: its first
// failures come first and its verdict comes last, so a head-only stub of a
// suite drops the line that says whether it passed. Which end matters is the
// tool's business, and the tool is the only thing that knows it.
type LODKeep string

const (
	// LODHead keeps the beginning. The default, and right for a file, a
	// listing, or a search whose matches are ordered.
	LODHead LODKeep = "head"
	// LODTail keeps the end — output whose conclusion is its last line.
	LODTail LODKeep = "tail"
	// LODEnds keeps both, with the elision counted between them. For output
	// that opens with what went wrong and closes with the verdict on it.
	LODEnds LODKeep = "ends"
)

// ShaperPolicy captures per-model context-window policy. A host builds it
// from its model row at session-resolve time.
type ShaperPolicy struct {
	// BudgetTokens is the model's usable context window in tokens.
	//
	// ZERO (or negative) means UNBUDGETED: no LOD, no compaction, the
	// conversation is rendered whole. It cannot mean "a budget of nothing" —
	// that reading cost a real session 45 compactions in 29 minutes, one per
	// turn, each spending a summarization call to fold history that had just
	// been summarized, leaving 5 surviving entries out of 50. A host that has
	// not been told the window size must not have its history destroyed on
	// that basis.
	BudgetTokens          int
	PreserveLastMessages  int
	PreserveLastToolCalls int

	// VerbatimToolResults passes tool-result content to the model byte for byte,
	// INCLUDING chat-template control tokens. Off by default.
	//
	// Set it only when the server's chat template neutralizes those tokens
	// itself, which is the right place for the fix: it costs nothing there and
	// covers every client. Left off against a stock template, `<|im_start|>` in a
	// file — a log, a README, a document about the template — tokenizes to the
	// real control token and opens a turn inside the prompt. Verified end to end:
	// the rendered turn sequence gains a SYSTEM message that came from disk.
	//
	// Probe a server before turning this on: ml-kit/templates/probe.py render
	// grades a template's tier-1 (frame-breaking) behavior, and llm.ProbeToolSurface
	// checks a live endpoint's tool-call round trip.
	VerbatimToolResults bool

	// ToolFormat must match the Session's, so replayed history is shown in the
	// same dialect the model writes.
	ToolFormat            ToolFormat
	LODTruncateAboveChars int
	// MaxToolResultChars bounds a single tool result REGARDLESS of age.
	//
	// LODTruncateAboveChars only reaches entries older than the preserved tail.
	// That exemption protects the result a session is about to act on, and it
	// means one oversized result rides in the window whole for as many turns as
	// the tail is long. This is the floor under that: the freshest result may be
	// large, but not unbounded. 0 → no limit.
	MaxToolResultChars int
	// LODHeadroomTokens is the runway kept below BudgetTokens. The Shaper
	// restructures (LOD, then compaction) once the estimated context would
	// exceed BudgetTokens-LODHeadroomTokens — NOT at the hard budget. Keeping
	// headroom means the prompt prefix is rewritten (and the KV cache
	// invalidated) in one decisive pass with room to spare, instead of eagerly
	// re-truncating a little more every turn. 0 → defaultLODHeadroom (10k);
	// negative → no headroom (restructure only at BudgetTokens).
	LODHeadroomTokens int
	// AlwaysLOD forces LOD (level-of-detail) truncation of oversized older
	// entries on EVERY Build, from Phase 0 on — skipping the "pristine, no-LOD"
	// fast path that otherwise leaves the context untouched while under budget.
	// Use it to force LOD every turn (e.g. a benchmark measuring recall over
	// truncated history). LOD is render-time and stateless — there is no
	// persisted-LOD op — so a policy flag, not a stored operation, is the right
	// shape for this knob.
	AlwaysLOD bool
}

// defaultLODHeadroom is the runway kept below budget when LODHeadroomTokens is
// unset — a rule of thumb: start reshaping with ~10k tokens to spare.
const defaultLODHeadroom = 10_000

// shapeTarget is the token ceiling the Shaper actually shapes to: the hard
// budget minus the headroom, so a restructure leaves runway. Falls back to the
// hard budget when the headroom would meet or exceed it (small budgets/tests).
func (p ShaperPolicy) shapeTarget() int {
	headroom := p.LODHeadroomTokens
	if headroom == 0 {
		headroom = defaultLODHeadroom
	}
	if headroom < 0 {
		headroom = 0
	}
	target := p.BudgetTokens - headroom
	if target <= 0 {
		return p.BudgetTokens
	}
	return target
}

// Shaper builds the LLM message list with a three-step algorithm:
//
//  0. pristine tail: the last N text messages + M tool-call exchanges are
//     always included verbatim regardless of size.
//  1. LOD render: older entries whose content exceeds the policy threshold
//     get truncated stubs (entry-id pointer + head). No writes — pure
//     render-time transformation; the source entry stays intact.
//  2. Compaction: if LOD-truncated context still exceeds budget, summarize
//     the oldest contiguous prefix of "older" entries into a compaction
//     marker via Store.Compact. Build re-runs with the marker substituted.
type Shaper struct {
	Store    Store
	Runner   LLMRunner
	Estimate TokenEstimator
	Policy   ShaperPolicy
	// SpanPrefix names the span namespace (default "agent"): the build span
	// is "<prefix>.Shaper.Build". A host sets this to keep its labels stable.
	SpanPrefix string
	Tracer     Tracer // optional; nil = no spans
	// ToolLOD says which end of a truncated result each tool answers at.
	//
	// Keyed by tool name; absent means LODHead, which is right for the majority
	// (a file, a listing, a search). It is a property of the TOOL, known where
	// the tool is registered, so the shaper is told rather than guessing — and
	// it cannot ride on `llm.ToolDef`, which is the wire shape sent to the
	// provider.
	ToolLOD map[string]LODKeep

	// Pause, if set, is called before the compaction LLM call and returns the
	// resume func. A host uses it to STOP the turn's budget clock for the
	// duration of the summarize call: compaction is bookkeeping, not model
	// work, so it must not burn the turn's time budget. Without this, a slow
	// endpoint + a long compaction can run the turn past its budget MID-fold,
	// cancelling the compaction worker with a bare "context canceled" that
	// reads as a crash. nil = no pause (the call runs on the caller's context
	// and budget as-is).
	Pause func() (resume func())

	// Task is what the session is FOR, shown to the summarizer on every fold.
	//
	// A fold sends only the oldest entries, and the summarizer was told nothing
	// else — so it judged what mattered by what was there, and one large early
	// tool result dominated the summary while the task it was read for went
	// unmentioned. The task is the measure of what to keep.
	//
	// Empty → Build falls back to the system prompt it was given, which is
	// where a host normally states the task. Compact has no system prompt, so
	// a host that forces folds sets this.
	Task string

	// Rehead, if set, is the host's own way to shrink the context, tried once
	// per Build just before a compaction fold — after the window has been
	// found over target even with LOD.
	//
	// It exists because a fold is the one moment the prompt prefix is
	// rewritten anyway: any other rewrite of earlier context breaks the
	// provider's prompt cache from that entry on, once per rewrite. A host that
	// knows some entries are redundant (a file read that a later, wider read
	// covers; a read made stale by an edit) replaces them here, through its
	// Store, and the cache is broken once for both. Rehead reports whether it
	// changed the context; when it did, Build re-reads the context and checks
	// the budget again, and the fold runs only if the window is still over.
	//
	// The host owns its errors: a re-heading that fails reports false and the
	// fold proceeds as it would have without one. nil = no re-heading, which is
	// the behavior before this field existed.
	Rehead func(ctx context.Context, sessionID string) (changed bool)
}

func (sh *Shaper) spanName(n string) string {
	p := sh.SpanPrefix
	if p == "" {
		p = "agent"
	}
	return p + "." + n
}

// Build is the ContextBuilder entry point. It reads the session's entries,
// then applies the pristine-tail + LOD + compaction phases.
func (sh *Shaper) Build(ctx context.Context, sessionID, system string) (msgs []llm.Message, err error) {
	sp, ctx := startSpan(sh.Tracer, ctx, sh.spanName("Shaper.Build"))
	sp.Set("session_id", sessionID).Set("budget_tokens", sh.Policy.BudgetTokens)
	defer func() {
		sp.Set("n_messages", len(msgs))
		if err != nil {
			sp.EndError(err)
		} else {
			sp.End()
		}
	}()
	if sh.Estimate == nil {
		sh.Estimate = Default()
	}

	// Unbudgeted: render and return. Every phase below compares against a token
	// ceiling, and with no ceiling there is nothing to compare — treating 0 as
	// the ceiling makes every build "over budget" and compacts on every turn.
	if sh.Policy.BudgetTokens <= 0 {
		entries, err := sh.Store.Context(ctx, sessionID)
		if err != nil {
			return nil, err
		}
		sortEntries(entries)
		pristineCount := classifyPristineCount(entries, sh.Policy.PreserveLastMessages, sh.Policy.PreserveLastToolCalls)
		return sh.render(entries, pristineCount, system, sh.Policy.AlwaysLOD), nil
	}

	reheaded := false
	for attempt := 0; attempt < 4; attempt++ {
		entries, err := sh.Store.Context(ctx, sessionID)
		if err != nil {
			return nil, err
		}
		sortEntries(entries)

		pristineCount := classifyPristineCount(entries, sh.Policy.PreserveLastMessages, sh.Policy.PreserveLastToolCalls)

		// Shape to BudgetTokens-headroom, not the hard budget, so a restructure
		// leaves runway and doesn't re-fire (and re-invalidate the KV cache)
		// next turn.
		target := sh.Policy.shapeTarget()

		// Phase 0: pristine tail, normally rendered WITHOUT LOD. If we're under
		// target, DON'T touch the prefix — appending to the tail keeps the cached
		// prefix intact. AlwaysLOD forces LOD truncation from this phase on, so
		// oversized older entries are truncated every turn even under budget.
		messages := sh.render(entries, pristineCount, system, sh.Policy.AlwaysLOD)
		before := sh.tokens(messages)
		if before <= target {
			return messages, nil
		}
		// Phase 1: LOD. Truncate older oversized entries. If that fits the
		// target, we're done — LOD is cheaper + lossless-ish vs compaction.
		messages = sh.render(entries, pristineCount, system, true)
		if sh.tokens(messages) <= target {
			return messages, nil
		}

		// Phase 1b: the host's re-heading, once, before any fold. It does not
		// spend an attempt: the attempts bound folds, and this is not one.
		if sh.Rehead != nil && !reheaded {
			reheaded = true
			if sh.Rehead(ctx, sessionID) {
				attempt--
				continue
			}
		}

		// Phase 2: compaction. Fold the oldest contiguous prefix of "older"
		// entries into a summary marker. If nothing summarizable remains, return
		// what we have.
		info, didCompact, err := sh.compactOldest(ctx, sessionID, before, target, sh.taskOr(system))
		if err != nil {
			return nil, err
		}
		if !didCompact {
			return messages, nil
		}
		// Surface the compaction (summary + meta) to whoever installed a sink —
		// the Session records it into TurnResult + fires OnCompactionComplete.
		reportCompaction(ctx, info)

		// Loop: Context picks up the marker in place of the subsumed rows and
		// we re-check budget.
	}
	return nil, fmt.Errorf("agent/shaper: compaction did not converge in 4 attempts")
}

// compactOldest folds the oldest contiguous prefix of non-pristine entries into
// a single compaction marker: it loads + sorts the session's entries, classifies
// the pristine tail under the policy, collects the summarizable older entries
// (skipping any existing KindCompaction markers so a marker is never
// re-summarized), summarizes them, and records the marker via Store.Compact at
// the latest CreatedAt among the subsumed rows (so the marker sits chrono-
// logically where those events were, not at wall-clock now).
//
// Returns (info, didCompact, err). didCompact is false with a zero
// CompactionInfo — NOT an error — when there is nothing summarizable (only the
// pristine tail remains, or every older entry is already a marker). This is the
// single implementation shared by Build's Phase 2 and the public Compact.
// TokensBefore/After mirror Build: a full (no-LOD) render before, an LOD render
// of the post-compaction window after.
// tokensBefore/target are what the CALLER measured, passed in rather than
// recomputed: Build already rendered the window to decide a fold was needed, and
// re-measuring here would report a different number than the one that triggered
// it. Zero from a caller that has no measurement (the public Compact) means the
// start report carries what it knows and no more.
func (sh *Shaper) compactOldest(ctx context.Context, sessionID string, tokensBefore, target int, task string) (CompactionInfo, bool, error) {
	if sh.Estimate == nil {
		sh.Estimate = Default()
	}
	entries, err := sh.Store.Context(ctx, sessionID)
	if err != nil {
		return CompactionInfo{}, false, err
	}
	sortEntries(entries)
	pristineCount := classifyPristineCount(entries, sh.Policy.PreserveLastMessages, sh.Policy.PreserveLastToolCalls)

	olderEnd := len(entries) - pristineCount
	if olderEnd <= 0 {
		return CompactionInfo{}, false, nil
	}

	var subsumed []Entry
	for i := 0; i < olderEnd; i++ {
		if entries[i].Kind == KindCompaction {
			// Already-compacted region — don't re-summarize a marker.
			continue
		}
		subsumed = append(subsumed, entries[i])
	}
	if len(subsumed) == 0 {
		return CompactionInfo{}, false, nil
	}

	before := sh.tokens(sh.render(entries, pristineCount, "", false))

	// Announced only now: everything above can still decide there is nothing to
	// fold, and a start with no matching completion is worse than no start.
	startedAt := time.Now()
	reportCompactionStart(ctx, CompactionStart{
		SessionID:    sessionID,
		TokensBefore: tokensBefore,
		Target:       target,
		Subsuming:    len(subsumed),
		At:           startedAt,
	})

	summary, err := sh.summarize(ctx, entries[:olderEnd], task)
	if err != nil {
		return CompactionInfo{}, false, fmt.Errorf("agent/shaper: summarize: %w", err)
	}

	// Place the marker at the latest CreatedAt among the subsumed rows so it
	// sits chronologically where those events were — not at wall-clock now.
	// Otherwise the marker (older history) floats to the end and the pristine
	// tail reorders behind it.
	markerCreatedAt := int64(0)
	for _, e := range entries[:olderEnd] {
		if e.CreatedAt > markerCreatedAt {
			markerCreatedAt = e.CreatedAt
		}
	}
	if err := sh.Store.Compact(ctx, sessionID, Compaction{
		Marker: Entry{
			ID:        NewID(),
			Kind:      KindCompaction,
			Content:   summary,
			CreatedAt: markerCreatedAt,
		},
		Subsumes: subsumed,
	}); err != nil {
		return CompactionInfo{}, false, fmt.Errorf("agent/shaper: compact: %w", err)
	}

	// Measure the new active window by re-rendering the post-compaction entries.
	afterEntries, aerr := sh.Store.Context(ctx, sessionID)
	after := 0
	if aerr == nil {
		sortEntries(afterEntries)
		ap := classifyPristineCount(afterEntries, sh.Policy.PreserveLastMessages, sh.Policy.PreserveLastToolCalls)
		after = sh.tokens(sh.render(afterEntries, ap, "", true))
	}

	return CompactionInfo{
		Summary:       summary,
		SubsumedCount: len(subsumed),
		TokensBefore:  before,
		TokensAfter:   after,
		Duration:      time.Since(startedAt),
	}, true, nil
}

// Compact forces a compaction fold NOW, regardless of budget — the manual
// counterpart to the implicit folds Build performs under budget pressure. It
// summarizes the oldest contiguous prefix of non-pristine entries into a marker
// (via compactOldest) and returns (info, didCompact, err). When only pristine
// entries remain (nothing summarizable), it returns (zero, false, nil): no
// marker written, no error — so calling it repeatedly is safe and idempotent-ish
// once history is fully folded. Hosts use this to force a deterministic fold
// (e.g. a benchmark that must exercise history compaction, then measure recall).
func (sh *Shaper) Compact(ctx context.Context, sessionID string) (CompactionInfo, bool, error) {
	// No measured trigger: this fold was asked for, not forced by a budget, so
	// there is no "tokens before" that caused it and no target it is aiming at.
	return sh.compactOldest(ctx, sessionID, 0, 0, sh.Task)
}

// taskOr is the task a fold is judged against: the host's, else the system
// prompt the turn was built with.
func (sh *Shaper) taskOr(system string) string {
	if sh.Task != "" {
		return sh.Task
	}
	return system
}

// classifyPristineCount returns the number of trailing entries that qualify
// as the pristine tail under the policy.
func classifyPristineCount(entries []Entry, maxMessages, maxToolCalls int) int {
	if maxMessages < 0 {
		maxMessages = 0
	}
	if maxToolCalls < 0 {
		maxToolCalls = 0
	}
	msgs, tools := 0, 0
	count := 0
	for i := len(entries) - 1; i >= 0; i-- {
		switch entries[i].Kind {
		case KindUser, KindAssistant:
			if msgs < maxMessages {
				msgs++
				count++
				continue
			}
			return count
		case KindToolCall, KindToolResult:
			if tools < maxToolCalls {
				tools++
				count++
				continue
			}
			return count
		case KindCompaction:
			// Markers are already compressed; always pass through.
			count++
		default:
			// Notifications / system entries — count toward neither budget
			// but stay in the tail.
			count++
		}
	}
	return count
}

// render builds the llm.Message list. When lod is true, entries outside the
// pristine tail get LOD-truncated content.
func (sh *Shaper) render(entries []Entry, pristineCount int, system string, lod bool) []llm.Message {
	olderEnd := len(entries) - pristineCount
	// What each result was ASKED. The arguments live on the call and the
	// truncation happens on the result, so a stub cannot see them without this.
	//
	// A stub says what it replaced — "[truncated event_id=… len=191623]" — and
	// the head that follows is the top of whatever came back, which for a file
	// is its first lines and for a listing is an arbitrary few. What it never
	// said is what the session was LOOKING for, which is the thing it needs
	// forty rounds later. The arguments answer that for free: the model already
	// wrote them, they are already in the log, and `{"pattern":"func TestD1"}`
	// is the purpose of that call in compressed form.
	askedBy := map[string]string{}
	for _, e := range entries {
		if e.Kind == KindToolCall && e.ToolCallID != "" {
			askedBy[e.ToolCallID] = e.Content
		}
	}
	return groupMessages(system, entries, func(i int, e Entry) string {
		content := e.Content
		if lod && i < olderEnd && len(content) > sh.Policy.LODTruncateAboveChars && sh.Policy.LODTruncateAboveChars > 0 {
			content = sh.stub(e, sh.Policy.LODTruncateAboveChars, askedBy[e.ToolCallID])
		} else if n := sh.Policy.MaxToolResultChars; n > 0 && len(content) > n {
			// Not aged out and not under LOD — the preserved tail. Bounded
			// anyway, because "recent" is not a reason to be unbounded.
			content = sh.stub(e, n, askedBy[e.ToolCallID])
		}
		return content
	}, sh.Policy.VerbatimToolResults, sh.Policy.ToolFormat)
}

// stub truncates one entry, keeping the end of it this tool's answers live at.
func (sh *Shaper) stub(e Entry, maxChars int, asked string) string {
	keep := LODHead
	if k, ok := sh.ToolLOD[e.ToolName]; ok {
		keep = k
	}
	return lodStub(e, maxChars, keep, e.ToolName, asked)
}

// lodStub replaces content with a short excerpt + a pointer back to the source
// entry so a future read-truncated-field tool call can fetch the full version.
//
// WHICH excerpt is the tool's business, not the shaper's. A file read is
// answered at the top; a test run puts its first failures at the top and its
// verdict at the BOTTOM, so a head-only stub of a suite drops the one line that
// says whether it passed — a known defect since 2026-09-06, deliberately not
// gold-plated then and closed here.
func lodStub(e Entry, maxChars int, keep LODKeep, tool, asked string) string {
	body, n := e.Content, maxChars/2
	if n < 100 {
		n = 100
	}
	var excerpt string
	switch {
	case len(body) <= n:
		excerpt = body
	case keep == LODTail:
		excerpt = "…\n" + cutRunes(body[len(body)-n:], true)
	case keep == LODEnds:
		// Half each end. The elision says how much went, so a model can tell a
		// gap from a join.
		h := n / 2
		excerpt = cutRunes(body[:h], false) +
			fmt.Sprintf("\n… %d characters …\n", len(body)-2*h) +
			cutRunes(body[len(body)-h:], true)
	default:
		excerpt = cutRunes(body[:n], false) + "\n…"
	}
	head := fmt.Sprintf("[truncated event_id=%s len=%d", e.ID, len(e.Content))
	if tool != "" {
		head += " · " + tool
	}
	if asked != "" {
		head += " " + oneLine(asked, 200)
	}
	return head + "]\n" + excerpt
}

// cutRunes trims a byte slice back to a rune boundary — from the front when the
// cut was at the front, from the back otherwise. A cut mid-rune is invalid
// UTF-8, which some providers reject outright.
func cutRunes(s string, fromFront bool) string {
	for len(s) > 0 && !utf8.ValidString(s) {
		if fromFront {
			s = s[1:]
		} else {
			s = s[:len(s)-1]
		}
	}
	return s
}

// oneLine flattens a call's arguments onto the stub's header line.
func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > max {
		s = cutRunes(s[:max], false) + "…"
	}
	return s
}

func ifThen(cond bool, then string) string {
	if cond {
		return then
	}
	return ""
}

// summarizeErr wraps a compaction failure so the jsonl says WHY the fold died,
// not just "context canceled". A cancelled context is inherited from the turn
// clock, so the same two words cover a real budget expiry, a deferred Stop()
// from a turn that already ended, and a user ctrl-C — all three read as an
// unexplained crash at the error site. The context's CAUSE distinguishes them
// when the clock set one (errTurnBudget); we carry it out so turnErr and the
// event log can name the decision rather than let "context canceled" masquerade
// as a transient fault.
func summarizeErr(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil && errors.Is(err, ctxErr) {
		cause := context.Cause(ctx)
		if cause != nil && !errors.Is(cause, context.Canceled) {
			return fmt.Errorf("agent/shaper: summarize: compaction aborted — %v (the turn's budget clock cancelled it)", cause)
		}
		return fmt.Errorf("agent/shaper: summarize: compaction aborted — %v (no turn-budget cause: ctrl-C, or the parent context went down)", ctxErr)
	}
	return err
}

// summarize calls the model on the oldest contiguous prefix of entries and
// asks for a concise summary. The returned text becomes the marker content.
//
// The task goes first, so the summarizer keeps what serves it rather than what
// was largest. The session after the fold still has its system prompt; what it
// loses is the history, so the summary must carry what the task needs from it.
func (sh *Shaper) summarize(ctx context.Context, entries []Entry, task string) (string, error) {
	var b strings.Builder
	if task = strings.TrimSpace(task); task != "" {
		b.WriteString("The session these events belong to has this task:\n<task>\n")
		b.WriteString(task)
		b.WriteString("\n</task>\n\nKeep what serves that task: files and identifiers it touched, what was changed or committed, results that decide it, and what remains. Drop output that does not bear on it.\n\n")
	}
	b.WriteString("Summarize the following events in <= 500 tokens. Preserve user requests, agent identifiers, decisions, and any open questions. Output the summary only — no preamble.\n\n")
	for i, e := range entries {
		label := string(e.Kind)
		if e.Tag != "" {
			label = e.Tag
		}
		fmt.Fprintf(&b, "%d. [%s] %s\n", i+1, label, e.Content)
	}

	// Stop the turn's budget clock for the fold: compaction is bookkeeping, not
	// model work, so it must not burn the turn's time budget. See Pause.
	resume := func() {}
	if sh.Pause != nil {
		resume = sh.Pause()
	}
	defer resume()
	ch, err := sh.Runner.ChatStream(ctx, []llm.Message{
		{Role: "system", Content: "You are a compaction worker. Summarize event logs concisely."},
		{Role: "user", Content: b.String()},
	}, nil, nil)
	if err != nil {
		return "", summarizeErr(ctx, err)
	}
	var summary strings.Builder
	for chunk := range ch {
		if chunk.Error != "" {
			return "", summarizeErr(ctx, fmt.Errorf("%s", chunk.Error))
		}
		summary.WriteString(chunk.Content)
		if chunk.Done {
			go func() {
				for range ch {
				}
			}()
			break
		}
	}
	out := strings.TrimSpace(summary.String())
	if out == "" {
		return "", fmt.Errorf("agent/shaper: model returned empty summary")
	}
	return out, nil
}

// tokens estimates the token cost of a fully built message list.
func (sh *Shaper) tokens(msgs []llm.Message) int {
	t := 0
	for _, m := range msgs {
		t += sh.Estimate.Estimate(m.Content)
		t += sh.Estimate.Estimate(m.Role) + sh.Estimate.Estimate(m.Name)
	}
	return t
}
