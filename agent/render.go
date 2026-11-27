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
func renderEntry(e Entry, content string, verbatimToolResults bool, format ToolFormat) llm.Message {
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
		// A tool result is not data to the tokenizer, it is prompt. Unless the
		// server's chat template already neutralizes them (see
		// ShaperPolicy.VerbatimToolResults), escape control tokens here so file
		// bytes cannot open a turn. Byte-identical for any content without `<|`.
		//
		// Media parts render only when content is the entry's own text, the
		// same guard KindUser uses: a stub or summary means the shaper chose to
		// drop the bytes, and re-attaching them would undo that. Checked before
		// neutralizing, which changes content that holds `<|`.
		withParts := len(e.Parts) > 0 && content == e.Content
		if !verbatimToolResults {
			content = NeutralizeSpecialTokens(content)
		}
		m := llm.Message{Role: "tool", Content: content, ToolCallID: id}
		if format == ToolFormatHeredoc {
			// No assistant tool_calls exist to correlate against, so a tool_call_id
			// would dangle. The provider renders a bare tool role fine (Qwen wraps
			// it in <tool_response>), and the model sees the result where it
			// expects one.
			m.ToolCallID = ""
		}
		if withParts {
			m.Content = ""
			m.Parts = toolResultParts(content, e.Parts, verbatimToolResults)
		}
		return m
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

// toolResultParts is a tool message's parts: the result's text (already
// neutralized by the caller) first, then the entry's parts. A fresh slice, so
// folding user words into it (FoldUserIntoToolResult) never writes through to
// the stored entry. Text parts among the entry's parts are prompt like the
// text, and are neutralized the same way. An empty text is left out: the
// Messages API refuses an empty text block.
func toolResultParts(text string, parts []llm.ContentPart, verbatim bool) []llm.ContentPart {
	out := make([]llm.ContentPart, 0, len(parts)+1)
	if text != "" {
		out = append(out, llm.TextPart(text))
	}
	for _, p := range parts {
		if p.Type == "text" && !verbatim {
			p.Text = NeutralizeSpecialTokens(p.Text)
		}
		out = append(out, p)
	}
	return out
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

// RenderOptions are the knobs of the entries → messages rendering that live
// outside the entries themselves. A Shaper carries them in its ShaperPolicy; a
// host with no Shaper passes them to DefaultContextBuilderWith.
type RenderOptions struct {
	// ToolFormat must match the Session's.
	ToolFormat ToolFormat
	// VerbatimToolResults: see ShaperPolicy.VerbatimToolResults.
	VerbatimToolResults bool
	// FoldUserIntoToolResult: see ShaperPolicy.FoldUserIntoToolResult.
	FoldUserIntoToolResult bool
	// ReplayReasoning: see ShaperPolicy.ReplayReasoning.
	ReplayReasoning ReasoningReplay
}

// userFrame prefixes a user's words folded into a tool result
// (FoldUserIntoToolResult). The frame says whose words they are and that they
// arrived while the tool ran, so the model does not read them as tool output.
const userFrame = "\n\n[the user wrote, while this ran] "

// foldIntoTool appends s to a tool message's text. A tool message with media
// Parts has its text in its first text part (toolResultParts puts it there),
// and Content is ignored beside Parts, so the words go into that part; with
// no text part (an empty result) one is put first.
func foldIntoTool(m *llm.Message, s string) {
	if len(m.Parts) == 0 {
		m.Content += s
		return
	}
	if m.Parts[0].Type == "text" {
		m.Parts[0].Text += s
		return
	}
	m.Parts = append([]llm.ContentPart{llm.TextPart(s)}, m.Parts...)
}

// groupMessages renders sorted entries into OpenAI-format messages, collapsing
// an assistant text + its following KindToolCall entries into ONE assistant
// message carrying tool_calls — so the model sees a valid
// assistant(tool_calls) → tool(tool_call_id) exchange, not orphan tool
// messages. contentOf yields the (possibly LOD-truncated) content per entry.
//
// Entries sort by CreatedAt, and a tool result is stamped when the tool
// FINISHES. Words a person typed while the tool ran (a pending KindUser saved
// mid-dispatch) are therefore older than the result and would sort between the
// assistant(tool_calls) and its tool message: a sequence strict endpoints
// refuse. While calls are still unanswered, every other entry (user,
// notification, compaction) is held back and emitted right after the last open
// result, or when the next assistant entry or the end arrives. Each held entry
// keeps its ORIGINAL index for contentOf (the Shaper's LOD keys on it). The
// order depends only on the entries, so round N's messages stay a prefix of
// round N+1's and the provider's prompt cache keeps hitting.
func groupMessages(system string, entries []Entry, contentOf func(i int, e Entry) string, ro RenderOptions) []llm.Message {
	return groupMessagesFrom(system, entries, contentOf, ro, replayFrom(entries, ro.ReplayReasoning))
}

// groupMessagesFrom is groupMessages with the reasoning boundary given:
// assistant entries at index >= from render their reasoning (replay.go). A
// render of a prefix of the context passes the whole context's boundary.
func groupMessagesFrom(system string, entries []Entry, contentOf func(i int, e Entry) string, ro RenderOptions, from int) []llm.Message {
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
	open := 0 // tool calls rendered whose result has not been
	type held struct {
		i int
		e Entry
	}
	var buf []held
	emit := func(i int, e Entry) {
		content := contentOf(i, e)
		// A user entry's Parts stay a user message: folding moves only words,
		// and would drop the attachment. (A stubbed Parts entry has lost them
		// already and folds as text.)
		textOnly := len(e.Parts) == 0 || content != e.Content
		if ro.FoldUserIntoToolResult && e.Kind == KindUser && textOnly &&
			len(out) > 0 && out[len(out)-1].Role == "tool" {
			foldIntoTool(&out[len(out)-1], userFrame+content)
			return
		}
		out = append(out, renderEntry(e, content, ro.VerbatimToolResults, ro.ToolFormat))
	}
	release := func() {
		for _, h := range buf {
			emit(h.i, h.e)
		}
		buf = nil
	}
	for i, e := range entries {
		switch e.Kind {
		case KindAssistant:
			flush()
			release()
			open = 0
			content := contentOf(i, e)
			m := llm.Message{Role: "assistant", Content: content}
			if i >= from {
				withReasoning(&m, e, content)
			}
			pending = &m
		case KindToolCall:
			if pending == nil {
				m := llm.Message{Role: "assistant"}
				pending = &m
			}
			if i >= from {
				withReasoning(pending, e, contentOf(i, e))
			}
			open++
			if ro.ToolFormat == ToolFormatHeredoc {
				// Replay the call as the TEXT the model wrote. Sending it as
				// native tool_calls would re-render it through the provider's XML
				// path — the one this format exists to avoid — and would need a
				// `tools` array heredoc mode does not send.
				if pending.Content != "" {
					pending.Content += "\n"
				}
				pending.Content += llm.RenderToolCall(e.ToolName, contentOf(i, e))
				continue
			}
			pending.ToolCalls = append(pending.ToolCalls, toToolCall(e))
		case KindToolResult:
			flush()
			out = append(out, renderEntry(e, contentOf(i, e), ro.VerbatimToolResults, ro.ToolFormat))
			if open > 0 {
				open--
			}
			if open == 0 {
				release()
			}
		default:
			flush()
			if open > 0 {
				buf = append(buf, held{i, e})
				continue
			}
			emit(i, e)
		}
	}
	flush()
	release()
	return out
}

// DefaultContextBuilder renders the session's history in chronological order.
// The budget-aware Shaper does the same but adds pristine-tail / LOD /
// compaction on top.
func DefaultContextBuilder(ctx context.Context, store Store, sessionID, system string) ([]llm.Message, error) {
	// No policy here: default to the SAFE behavior.
	return DefaultContextBuilderWith(ctx, store, sessionID, system, RenderOptions{})
}

// DefaultContextBuilderFormat is DefaultContextBuilder for a specific tool
// transport. History has to be replayed in the dialect the model writes, so a
// host running the heredoc transport without a Shaper needs this rather than the
// plain builder.
func DefaultContextBuilderFormat(ctx context.Context, store Store, sessionID, system string,
	format ToolFormat) ([]llm.Message, error) {
	return DefaultContextBuilderWith(ctx, store, sessionID, system, RenderOptions{ToolFormat: format})
}

// DefaultContextBuilderWith is DefaultContextBuilder with every rendering knob.
// A host without a Shaper that needs FoldUserIntoToolResult passes this as the
// Session's Build.
func DefaultContextBuilderWith(ctx context.Context, store Store, sessionID, system string,
	ro RenderOptions) ([]llm.Message, error) {
	entries, err := store.Context(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	sortEntries(entries)
	return groupMessages(system, entries, func(_ int, e Entry) string { return e.Content }, ro), nil
}
