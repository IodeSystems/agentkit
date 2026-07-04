package agent

import (
	"context"
	"fmt"
	"sort"

	"github.com/iodesystems/agentkit/llm"
)

// sortEntries orders entries by CreatedAt ascending. Ties (same nanosecond)
// break by ID lexicographically — deterministic across restarts.
func sortEntries(es []Entry) {
	sort.SliceStable(es, func(i, j int) bool {
		if es[i].CreatedAt != es[j].CreatedAt {
			return es[i].CreatedAt < es[j].CreatedAt
		}
		return es[i].ID < es[j].ID
	})
}

// renderEntry maps an Entry to an llm.Message. content is passed separately
// so the shaper can substitute an LOD stub without touching the entry.
func renderEntry(e Entry, content string) llm.Message {
	switch e.Kind {
	case KindUser:
		return llm.Message{Role: "user", Content: content}
	case KindAssistant:
		return llm.Message{Role: "assistant", Content: content}
	case KindToolResult:
		return llm.Message{Role: "tool", Content: content, Name: e.ID}
	case KindCompaction:
		return llm.Message{Role: "user", Content: "[compacted]\n" + content}
	default:
		// Notifications / tool_call / system-flavored entries feed in as
		// user-role context with a type tag so the model can distinguish
		// them from a real user turn.
		tag := e.Tag
		if tag == "" {
			tag = string(e.Kind)
		}
		return llm.Message{Role: "user", Content: fmt.Sprintf("[%s] %s", tag, content)}
	}
}

// DefaultContextBuilder renders the session's history verbatim in
// chronological order. The budget-aware Shaper does the same but adds the
// pristine-tail / LOD / compaction phases on top.
func DefaultContextBuilder(ctx context.Context, store Store, sessionID, system string) ([]llm.Message, error) {
	entries, err := store.Context(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	sortEntries(entries)
	out := make([]llm.Message, 0, len(entries)+1)
	if system != "" {
		out = append(out, llm.Message{Role: "system", Content: system})
	}
	for _, e := range entries {
		out = append(out, renderEntry(e, e.Content))
	}
	return out, nil
}
