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

// renderEntry maps a NON-assistant/NON-tool_call Entry to an llm.Message.
// content is passed separately so the shaper can substitute an LOD stub. The
// assistant + tool_call kinds are handled by groupMessages (they must merge
// into one assistant message with tool_calls) and never reach here.
func renderEntry(e Entry, content string) llm.Message {
	switch e.Kind {
	case KindUser:
		// Multimodal user turn: hand the parts through so the provider gets an
		// OpenAI content ARRAY. Guarded on content == e.Content — when the
		// shaper passes a substituted LOD stub or a compaction summary, the
		// original attachment is deliberately no longer what we want to send,
		// and re-attaching the image would defeat the truncation that decided
		// to drop it.
		if len(e.Parts) > 0 && content == e.Content {
			return llm.Message{Role: "user", Parts: e.Parts}
		}
		return llm.Message{Role: "user", Content: content}
	case KindAssistant:
		return llm.Message{Role: "assistant", Content: content}
	case KindToolResult:
		// Link the result back to its call. Fall back to the entry id for
		// legacy rows that predate tool_call_id persistence.
		id := e.ToolCallID
		if id == "" {
			id = e.ID
		}
		return llm.Message{Role: "tool", Content: content, ToolCallID: id}
	case KindCompaction:
		return llm.Message{Role: "user", Content: "[compacted]\n" + content}
	default:
		// Notifications / system-flavored entries feed in as user-role context
		// with a type tag so the model can distinguish them from a real user
		// turn.
		tag := e.Tag
		if tag == "" {
			tag = string(e.Kind)
		}
		return llm.Message{Role: "user", Content: fmt.Sprintf("[%s] %s", tag, content)}
	}
}

// toToolCall reconstructs an llm.ToolCall from a persisted KindToolCall entry.
func toToolCall(e Entry) llm.ToolCall {
	var tc llm.ToolCall
	tc.ID = e.ToolCallID
	tc.Type = "function"
	tc.Function.Name = e.ToolName
	tc.Function.Arguments = e.Content
	return tc
}

// groupMessages renders sorted entries into OpenAI-format messages, collapsing
// an assistant text + its following KindToolCall entries into ONE assistant
// message carrying tool_calls — so the model sees a valid
// assistant(tool_calls) → tool(tool_call_id) exchange, not orphan tool
// messages. contentOf yields the (possibly LOD-truncated) content per entry.
func groupMessages(system string, entries []Entry, contentOf func(i int, e Entry) string) []llm.Message {
	out := make([]llm.Message, 0, len(entries)+1)
	if system != "" {
		out = append(out, llm.Message{Role: "system", Content: system})
	}
	var pending *llm.Message // an assistant message accumulating tool_calls
	flush := func() {
		if pending != nil {
			out = append(out, *pending)
			pending = nil
		}
	}
	for i, e := range entries {
		switch e.Kind {
		case KindAssistant:
			flush()
			m := llm.Message{Role: "assistant", Content: contentOf(i, e)}
			pending = &m
		case KindToolCall:
			if pending == nil {
				m := llm.Message{Role: "assistant"}
				pending = &m
			}
			pending.ToolCalls = append(pending.ToolCalls, toToolCall(e))
		default:
			flush()
			out = append(out, renderEntry(e, contentOf(i, e)))
		}
	}
	flush()
	return out
}

// DefaultContextBuilder renders the session's history in chronological order.
// The budget-aware Shaper does the same but adds pristine-tail / LOD /
// compaction on top.
func DefaultContextBuilder(ctx context.Context, store Store, sessionID, system string) ([]llm.Message, error) {
	entries, err := store.Context(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	sortEntries(entries)
	return groupMessages(system, entries, func(_ int, e Entry) string { return e.Content }), nil
}
