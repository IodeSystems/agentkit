package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/google/uuid"

	"github.com/iodesystems/agentkit/llm"
)

// ShaperPolicy captures per-model context-window policy. A host builds it
// from its model row at session-resolve time.
type ShaperPolicy struct {
	BudgetTokens          int
	PreserveLastMessages  int
	PreserveLastToolCalls int
	LODTruncateAboveChars int
}

// Shaper builds the LLM message list with a three-step algorithm:
//
//	0. pristine tail: the last N text messages + M tool-call exchanges are
//	   always included verbatim regardless of size.
//	1. LOD render: older entries whose content exceeds the policy threshold
//	   get truncated stubs (entry-id pointer + head). No writes — pure
//	   render-time transformation; the source entry stays intact.
//	2. Compaction: if LOD-truncated context still exceeds budget, summarize
//	   the oldest contiguous prefix of "older" entries into a compaction
//	   marker via Store.Compact. Build re-runs with the marker substituted.
type Shaper struct {
	Store    Store
	Runner   LLMRunner
	Estimate TokenEstimator
	Policy   ShaperPolicy
	// SpanPrefix names the span namespace (default "agent"): the build span
	// is "<prefix>.Shaper.Build". A host sets this to keep its labels stable.
	SpanPrefix string
	Tracer     Tracer // optional; nil = no spans
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

	for attempt := 0; attempt < 4; attempt++ {
		entries, err := sh.Store.Context(ctx, sessionID)
		if err != nil {
			return nil, err
		}
		sortEntries(entries)

		pristineCount := classifyPristineCount(entries, sh.Policy.PreserveLastMessages, sh.Policy.PreserveLastToolCalls)

		// Phase 0+1: render with pristine tail + LOD on older.
		messages := sh.render(entries, pristineCount, system, false)
		if sh.tokens(messages) <= sh.Policy.BudgetTokens {
			return messages, nil
		}
		messages = sh.render(entries, pristineCount, system, true)
		if sh.tokens(messages) <= sh.Policy.BudgetTokens {
			return messages, nil
		}

		// Phase 2: compaction. Summarize the oldest contiguous prefix of
		// "older" entries. If nothing summarizable remains, return what we
		// have.
		olderEnd := len(entries) - pristineCount
		if olderEnd <= 0 {
			return messages, nil
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
			return messages, nil
		}

		summary, err := sh.summarize(ctx, entries[:olderEnd])
		if err != nil {
			return nil, fmt.Errorf("agent/shaper: summarize: %w", err)
		}

		// Place the marker at the latest CreatedAt among the subsumed rows so
		// it sits chronologically where those events were — not at wall-clock
		// now. Otherwise the marker (older history) floats to the end and the
		// pristine tail reorders behind it.
		markerCreatedAt := int64(0)
		for _, e := range entries[:olderEnd] {
			if e.CreatedAt > markerCreatedAt {
				markerCreatedAt = e.CreatedAt
			}
		}
		if err := sh.Store.Compact(ctx, sessionID, Compaction{
			Marker: Entry{
				ID:        uuid.New().String(),
				Kind:      KindCompaction,
				Content:   summary,
				CreatedAt: markerCreatedAt,
			},
			Subsumes: subsumed,
		}); err != nil {
			return nil, fmt.Errorf("agent/shaper: compact: %w", err)
		}
		// Loop: Context picks up the marker in place of the subsumed rows and
		// we re-check budget.
	}
	return nil, fmt.Errorf("agent/shaper: compaction did not converge in 4 attempts")
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
	out := make([]llm.Message, 0, len(entries)+1)
	if system != "" {
		out = append(out, llm.Message{Role: "system", Content: system})
	}
	olderEnd := len(entries) - pristineCount
	for i, e := range entries {
		content := e.Content
		if lod && i < olderEnd && len(content) > sh.Policy.LODTruncateAboveChars && sh.Policy.LODTruncateAboveChars > 0 {
			content = lodStub(e, sh.Policy.LODTruncateAboveChars)
		}
		out = append(out, renderEntry(e, content))
	}
	return out
}

// lodStub replaces content with a short head + a pointer back to the source
// entry so a future read-truncated-field tool call can fetch the full version.
func lodStub(e Entry, maxChars int) string {
	head := e.Content
	keep := maxChars / 2
	if keep < 100 {
		keep = 100
	}
	if keep > len(head) {
		keep = len(head)
	}
	return fmt.Sprintf("[truncated event_id=%s len=%d]\n%s%s",
		e.ID, len(e.Content), head[:keep],
		ifThen(len(head) > keep, "\n…"),
	)
}

func ifThen(cond bool, then string) string {
	if cond {
		return then
	}
	return ""
}

// summarize calls the model on the oldest contiguous prefix of entries and
// asks for a concise summary. The returned text becomes the marker content.
func (sh *Shaper) summarize(ctx context.Context, entries []Entry) (string, error) {
	var b strings.Builder
	b.WriteString("Summarize the following events in <= 500 tokens. Preserve user requests, agent identifiers, decisions, and any open questions. Output the summary only — no preamble.\n\n")
	for i, e := range entries {
		label := string(e.Kind)
		if e.Tag != "" {
			label = e.Tag
		}
		fmt.Fprintf(&b, "%d. [%s] %s\n", i+1, label, e.Content)
	}

	ch, err := sh.Runner.ChatStream(ctx, []llm.Message{
		{Role: "system", Content: "You are a compaction worker. Summarize event logs concisely."},
		{Role: "user", Content: b.String()},
	}, nil, nil)
	if err != nil {
		return "", err
	}
	var summary strings.Builder
	for chunk := range ch {
		if chunk.Error != "" {
			return "", fmt.Errorf("%s", chunk.Error)
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
