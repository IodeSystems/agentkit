package agent

// Reasoning replay.
//
// A round's reasoning is stored on the entry that carries the round's Usage
// (Entry.Reasoning, Entry.ReasoningOpaque): the KindAssistant reply, or the
// first KindToolCall of a round with no prose. ShaperPolicy.ReplayReasoning
// (RenderOptions.ReplayReasoning without a Shaper) puts it back on the
// rendered assistant message (llm.Message.Reasoning, ReasoningOpaque), and the
// llm client sends it in the endpoint's own field (llm/replay.go). Thinking
// models re-derive their plan each request without it: dun measured 4.6x pi's
// cost on DeepSeek, and replaying cut reasoning tokens by about 40%. With
// thinking on, Anthropic requires the signed thinking blocks on the assistant
// turn that continues a tool loop.
//
// Prompt-cache stability: an entry's rendering is a function of the entries
// alone, so within one policy round N's messages stay a prefix of round N+1's.
// The one break: under ReplayTurn a new KindUser entry moves the boundary, and
// the assistant messages before it lose their reasoning, once, on the request
// that carries the new user message. An entry the Shaper stubs (LOD) loses its
// reasoning with its content, at the same step.

import "github.com/iodesystems/agentkit/llm"

// ReasoningReplay picks which assistant entries render with their reasoning.
type ReasoningReplay string

const (
	// ReplayNone sends no reasoning: the requests are byte-identical to the
	// ones built before replay existed. The default.
	ReplayNone ReasoningReplay = ""
	// ReplayTurn sends the reasoning of assistant entries after the newest
	// KindUser entry: the current tool loop. What DeepSeek and Anthropic ask
	// for; earlier turns' reasoning is spent context.
	ReplayTurn ReasoningReplay = "turn"
	// ReplayAll sends every assistant entry's reasoning.
	ReplayAll ReasoningReplay = "all"
)

// replayFrom is the index of the first entry whose reasoning renders, or
// len(entries) when none does. Computed on the WHOLE sorted context, so a
// render of a prefix (the Shaper's prefix fold) uses the session's boundary.
func replayFrom(entries []Entry, mode ReasoningReplay) int {
	switch mode {
	case ReplayAll:
		return 0
	case ReplayTurn:
		for i := len(entries) - 1; i >= 0; i-- {
			if entries[i].Kind == KindUser {
				return i + 1
			}
		}
		return 0
	}
	return len(entries)
}

// withReasoning puts e's reasoning on the assistant message m when m has none
// yet and the entry renders as stored (content is e's own: a stub dropped it).
func withReasoning(m *llm.Message, e Entry, content string) {
	if content != e.Content || m.Reasoning != "" || len(m.ReasoningOpaque) > 0 {
		return
	}
	m.Reasoning, m.ReasoningOpaque = e.Reasoning, e.ReasoningOpaque
}

// roundReasoningOf is the reasoning a round stores: none for a round the
// transport cut for repetition, since the loop may be in the reasoning and a
// replay would prime the next round to continue it.
func roundReasoningOf(reasoning string, rep *llm.RepetitionInfo) string {
	if rep != nil {
		return ""
	}
	return reasoning
}
