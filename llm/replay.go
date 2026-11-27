package llm

// Reasoning replay: an assistant message's earlier reasoning, sent back.
//
// A thinking model that is not shown its own reasoning re-derives its plan on
// every request of a tool loop. dun measured 4.6x pi's cost on DeepSeek from
// that; sending the reasoning back cut reasoning tokens by about 40%. Anthropic
// goes further: with thinking on, the assistant turn that continues a tool loop
// must carry the thinking blocks it was given, signed.
//
// Two fields on Message carry it, set by the caller (agent.Session does it from
// the stored Entry, under ShaperPolicy.ReplayReasoning):
//
//	Reasoning        the text
//	ReasoningOpaque  the provider's record, verbatim: a JSON array in
//	                 OpenRouter's reasoning_details shape
//	                 ({type: reasoning.text|reasoning.summary|reasoning.encrypted,
//	                   text, summary, signature, data, format, index, id})
//
// The reader fills both: the stream merges every delta.reasoning_details into
// StreamChunk.ReasoningOpaque on the Done chunk; ChatResult has both. The shim
// sends an Anthropic thinking block as one reasoning.text item with its
// signature (format anthropic-claude-v1) and a redacted_thinking block as a
// reasoning.encrypted item, so one record covers both providers.
//
// On the wire, assistant messages only, only when set:
//
//	Messages API (shim)    reasoning_details = opaque; the shim makes thinking and
//	                       redacted_thinking blocks of the signed items and drops
//	                       the rest. Text alone is never sent: Anthropic refuses
//	                       thinking it did not sign.
//	openrouter             reasoning_details = opaque, else reasoning = text
//	                       (openrouter.ai/docs/use-cases/reasoning-tokens,
//	                       "preserving reasoning")
//	llamacpp, openai       reasoning_content = text (DeepSeek, llama.cpp, vLLM)
//	anthropic (compat)     nothing
//
// An endpoint that 400s naming the field it was sent teaches
// CapReasoningReplay=refused for its endpoint+model+provider, the request is
// sent once more without it (capabilities_learn.go), and later requests skip
// it; each skip is reported through OnTranslate with Key "messages.reasoning".

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
)

// replayWire is what one assistant message carries on the wire for its
// reasoning: at most one field is set.
type replayWire struct {
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Reasoning        string          `json:"reasoning,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
}

func (w *replayWire) field() string {
	switch {
	case w.ReasoningContent != "":
		return "reasoning_content"
	case w.Reasoning != "":
		return "reasoning"
	case len(w.ReasoningDetails) > 0:
		return "reasoning_details"
	}
	return ""
}

// replayFor is the wire form of m's reasoning on this endpoint, nil when it
// sends none.
func (c *Client) replayFor(m Message) *replayWire {
	opaque := len(bytes.TrimSpace(m.ReasoningOpaque)) > 0
	if c.usesMessagesAPI() {
		if opaque {
			return &replayWire{ReasoningDetails: m.ReasoningOpaque}
		}
		return nil
	}
	switch c.dialect() {
	case DialectOpenRouter:
		if opaque {
			return &replayWire{ReasoningDetails: m.ReasoningOpaque}
		}
		if m.Reasoning != "" {
			return &replayWire{Reasoning: m.Reasoning}
		}
	case DialectLlamaCpp, DialectOpenAI:
		if m.Reasoning != "" {
			return &replayWire{ReasoningContent: m.Reasoning}
		}
	}
	return nil
}

// replayReasoning returns msgs with each assistant message's reasoning set for
// the wire, on a copy (the caller's slice is never written). With no assistant
// reasoning it returns msgs itself, so the body is the one built before replay
// existed. When the record says the endpoint refuses the field, nothing is set
// and the skip is returned as a Translation.
func (c *Client) replayReasoning(msgs []Message, caps Capabilities) ([]Message, *Translation) {
	refused := false
	if v, _ := caps.fact(CapReasoningReplay); v == "refused" {
		refused = true
	}
	var out []Message
	var skipped string
	for i, m := range msgs {
		if m.Role != "assistant" || (m.Reasoning == "" && len(m.ReasoningOpaque) == 0) {
			continue
		}
		w := c.replayFor(m)
		if w == nil {
			continue
		}
		if refused {
			skipped = w.field()
			continue
		}
		if out == nil {
			out = append([]Message(nil), msgs...)
		}
		out[i].replay = w
	}
	var tr *Translation
	if skipped != "" {
		tr = &Translation{Key: "messages.reasoning", Wanted: skipped}
	}
	if out == nil {
		return msgs, tr
	}
	return out, tr
}

// sentReplayField is the reasoning field the sent body put on an assistant
// message, "" when none.
func sentReplayField(sent map[string]any) string {
	msgs, _ := sent["messages"].([]Message)
	for _, m := range msgs {
		if m.replay != nil {
			return m.replay.field()
		}
	}
	return ""
}

var (
	// The plain reasoning field is also the top-level effort object on
	// OpenRouter, so only a refusal that names it inside a message counts:
	// messages[1].reasoning, messages.1.reasoning, ('body', 'messages', 1, 'reasoning').
	reMessageReasoning = regexp.MustCompile(`messages\W{1,4}\d+\W{1,4}reasoning\b`)
	// Anthropic, on a thinking block whose signature it does not accept:
	// "messages.1.content.0: Invalid `signature` in `thinking` block".
	reBadSignature = regexp.MustCompile(`(?i)signature`)
	// A body that asks for MORE reasoning is not a refusal of the field.
	// DeepSeek with thinking and tools: "Missing `reasoning_content` field in
	// the assistant message at message index 2".
	reReplayWanted = regexp.MustCompile(`(?i)(missing|required|must (include|contain|be passed))\W[^.]{0,60}reasoning|reasoning\w*\W[^.]{0,40}(is required|is missing|must be passed)`)
)

// replayRefusal is the fact a 400/422 teaches about the reasoning field the
// request carried on its assistant messages, nil when it says nothing about it.
//
// Matched wordings, none captured from a live endpoint: DeepSeek's older
// thinking models ("Please remove the reasoning_content from your message"), a
// strict OpenAI-schema server ("Unrecognized request argument supplied:
// reasoning_content", "Additional properties are not allowed
// ('reasoning_content' was unexpected)"), pydantic extra_forbidden on
// messages.N.reasoning_content, and Anthropic's invalid signature.
func replayRefusal(r refusalCtx) []factUpdate {
	field := sentReplayField(r.sent)
	if field == "" || reReplayWanted.MatchString(r.text) {
		return nil
	}
	var match string
	switch field {
	case "reasoning":
		match = reMessageReasoning.FindString(r.text)
	case "reasoning_details":
		if strings.Contains(r.text, field) {
			match = field
		} else {
			// The Messages API: the shim turned the details into thinking blocks.
			match = reBadSignature.FindString(r.text)
		}
	default:
		if strings.Contains(r.text, field) {
			match = field
		}
	}
	if match == "" {
		return nil
	}
	return []factUpdate{{Key: CapReasoningReplay, Value: "refused",
		Evidence: r.evidence("messages[].assistant."+field, firstLineOfBody(r.text))}}
}

// --- reasoning_details -------------------------------------------------------

// reasoningItem is one reasoning_details element, every field kept as sent.
type reasoningItem map[string]json.RawMessage

func (it reasoningItem) str(key string) string {
	var s string
	if raw, ok := it[key]; ok {
		_ = json.Unmarshal(raw, &s)
	}
	return s
}

// parseReasoningDetails reads one delta's (or one message's) reasoning_details.
// Anything that is not an array of objects is nil.
func parseReasoningDetails(raw json.RawMessage) []reasoningItem {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '[' {
		return nil
	}
	var items []reasoningItem
	if json.Unmarshal(raw, &items) != nil {
		return nil
	}
	out := items[:0]
	for _, it := range items {
		if it != nil {
			out = append(out, it)
		}
	}
	return out
}

// mergeReasoningDetails appends a delta's items to acc. OpenRouter streams one
// item in fragments, each carrying the item's index and type: a fragment with
// the same index and type as the last item merges into it (text and summary
// concatenate; any other field takes the newest non-empty value: the
// signature arrives last, alone). An item with no index is appended whole.
// The docs say only "concatenating all chunks in order"; merging by index is
// what the Anthropic-backed stream needs, since one thinking block must stay
// one block with one signature.
func mergeReasoningDetails(acc []reasoningItem, ds []reasoningItem) []reasoningItem {
	for _, d := range ds {
		n := len(acc)
		idx, hasIdx := d["index"]
		if n == 0 || !hasIdx || !bytes.Equal(acc[n-1]["index"], idx) || acc[n-1].str("type") != d.str("type") {
			cp := reasoningItem{}
			for k, v := range d {
				cp[k] = v
			}
			acc = append(acc, cp)
			continue
		}
		last := acc[n-1]
		for k, v := range d {
			switch k {
			case "text", "summary":
				if s := d.str(k); s != "" {
					joined, _ := json.Marshal(last.str(k) + s)
					last[k] = joined
				}
			default:
				t := bytes.TrimSpace(v)
				if len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte(`""`)) {
					continue
				}
				last[k] = v
			}
		}
	}
	return acc
}

// marshalReasoningDetails is the merged record as Message.ReasoningOpaque takes
// it, nil when empty.
func marshalReasoningDetails(acc []reasoningItem) json.RawMessage {
	if len(acc) == 0 {
		return nil
	}
	b, err := json.Marshal(acc)
	if err != nil {
		return nil
	}
	return b
}

// reasoningFromDetails joins the text and summary entries of reasoning_details.
// The stream reads it ONLY while no delta has carried reasoning_content or
// reasoning: OpenRouter often sends the same text in both, and the shim sends a
// thinking block's whole text again with its signature, so reading both would
// double it.
func reasoningFromDetails(ds []reasoningItem) string {
	var b strings.Builder
	for _, d := range ds {
		switch d.str("type") {
		case "reasoning.text":
			b.WriteString(d.str("text"))
		case "reasoning.summary":
			b.WriteString(d.str("summary"))
		}
	}
	return b.String()
}
