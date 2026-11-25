package llm

// toolmedia.go — where a tool result's media parts go on the wire.
//
// A role "tool" Message may carry Parts: what a tool returned beside its text
// (a screenshot, a recording, a PDF). The two wire formats this package speaks
// disagree on where media may sit:
//
//   - OpenAI chat format (every endpoint but the Messages API): a tool message
//     is text only. OpenAI documents it so, and llama.cpp and vLLM templates
//     render a tool message's content as a string. Media is accepted in a USER
//     message. So each media part MOVES: the tool message keeps its text plus a
//     label per part, and ONE user message straight after the run of tool
//     messages carries the parts. A run, not each message: the run is the
//     answer to one assistant turn's parallel calls, and a strict endpoint
//     refuses anything between an assistant(tool_calls) and its last result.
//   - Claude Messages API (through the shim): tool_result content takes image
//     and document blocks natively, so image parts and PDF/text files stay in
//     the tool message. Audio and video it takes nowhere, and the shim refuses
//     them with ErrUnsupportedContent, which would fail the whole request
//     because of one tool's output. Those are OMITTED with a label.
//
// Omission is the exception to ContentPart's "no kind is ever stripped". That
// rule exists because a model that answers about media it never saw is worse
// than a refusal; the label in place of the part tells the model it did not see
// it, which is the thing the rule protects. A request the endpoint refuses
// outright, on the other hand, takes the session down over a tool result the
// model could have worked around. Never silent: every omitted part leaves a
// label.
//
// The capability record gates too: when it KNOWS a model does not take a
// modality (a fact "modality.<kind>" = false), a part of that kind is omitted
// with a label on either format. Unknown means send — the endpoint answers for
// itself. Declared metadata (GET /models) only ever records "true", so in
// practice the gate fires on a host-pinned or host-stored "false".
//
// Prompt caching: this is a pure function of the message list (and the record),
// so round N renders identically as the prefix of round N+1. A run is complete
// once a request is sent (every call has its result, or the endpoint refuses
// it), so its numbering never changes as history grows. A message with no media
// is passed through untouched, and a list with none is returned as is: the bytes
// every existing caller sends do not change.

import (
	"mime"
	"strconv"
	"strings"
)

// wireFormat is the request shape a client sends.
type wireFormat int

const (
	wireChat     wireFormat = iota // OpenAI chat completions
	wireMessages                   // Claude Messages API, via the shim
)

// mediaRoute is what happens to one content part of a tool message.
type mediaRoute int

const (
	routeNative mediaRoute = iota // stays in the tool message
	routeMove                     // moves to the user message after the run
	routeOmit                     // replaced by a label
)

// toolPartRoute is the routing table: (wire format, part kind) → route. A kind
// it does not name (one added later) moves on the chat format, where the user
// message is the place media goes, and is omitted on the Messages API, whose
// shim refuses any type it does not know.
func toolPartRoute(f wireFormat, p ContentPart) mediaRoute {
	if p.Type == "text" {
		return routeNative
	}
	if f == wireChat {
		return routeMove
	}
	switch p.Type {
	case "image_url":
		return routeNative
	case "file":
		if messagesTakesFile(p.File) {
			return routeNative
		}
	}
	return routeOmit
}

// messagesTakesFile mirrors the shim's documentBlock: a data: URI holding a PDF
// or text. Anything else the shim refuses, so it is omitted here instead.
func messagesTakesFile(f *FileRef) bool {
	if f == nil {
		return false
	}
	media, ok := dataURIMedia(f.FileData)
	if !ok {
		return false
	}
	switch media {
	case "application/pdf", "application/json", "application/xml", "application/x-yaml", "application/yaml":
		return true
	}
	return strings.HasPrefix(media, "text/")
}

// wireMessageList is the message list as this client sends it. caps is the
// record chatBody already read (empty on a raw probe, so no gate). The caller's
// slice is never mutated.
func (c *Client) wireMessageList(messages []Message, caps Capabilities) []Message {
	f := wireChat
	if c.usesMessagesAPI() {
		f = wireMessages
	}
	return routeToolMedia(messages, f, caps)
}

// routeToolMedia applies the routing table to every tool message. It returns
// messages itself when no tool message carries a part to move or omit.
func routeToolMedia(messages []Message, f wireFormat, caps Capabilities) []Message {
	if !anyToolMedia(messages) {
		return messages
	}
	out := make([]Message, 0, len(messages)+1)
	for i := 0; i < len(messages); {
		if messages[i].Role != "tool" {
			out = append(out, messages[i])
			i++
			continue
		}
		j := i
		for j < len(messages) && messages[j].Role == "tool" {
			j++
		}
		out = appendToolRun(out, messages[i:j], f, caps)
		i = j
	}
	return out
}

func anyToolMedia(messages []Message) bool {
	for _, m := range messages {
		if m.Role == "tool" && hasMedia(m) {
			return true
		}
	}
	return false
}

// appendToolRun appends one run of consecutive tool messages, rewritten, and on
// the chat format the user message that carries what moved.
func appendToolRun(out []Message, run []Message, f wireFormat, caps Capabilities) []Message {
	// Count first: one moved part is labeled plainly, several are numbered.
	moved := 0
	for _, m := range run {
		for _, p := range m.Parts {
			if gateRefuses(caps, p) == "" && toolPartRoute(f, p) == routeMove {
				moved++
			}
		}
	}
	numbered := moved > 1
	var carry []ContentPart
	n := 0
	for _, m := range run {
		if !hasMedia(m) {
			out = append(out, m)
			continue
		}
		var text []string // the chat format's text-only body
		var parts []ContentPart
		hasText := false
		for _, p := range m.Parts {
			if p.Type == "text" {
				hasText = true
				break
			}
		}
		// Parts owns content, so Content is otherwise ignored. A tool message
		// whose Parts are media only keeps Content as its text, so a host that
		// put the words in Content and the media in Parts loses neither.
		if !hasText && m.Content != "" {
			text = append(text, m.Content)
			parts = append(parts, TextPart(m.Content))
		}
		for _, p := range m.Parts {
			if why := gateRefuses(caps, p); why != "" {
				label := omitLabel(p, why)
				text = append(text, label)
				parts = append(parts, TextPart(label))
				continue
			}
			switch toolPartRoute(f, p) {
			case routeNative:
				if p.Type == "text" {
					text = append(text, p.Text)
				}
				parts = append(parts, p)
			case routeMove:
				n++
				label := partMime(p) + " is attached in the next user message"
				if numbered {
					tag := partKind(p) + " " + strconv.Itoa(n) + ":"
					label = tag + " " + label
					carry = append(carry, TextPart(tag))
				}
				carry = append(carry, p)
				text = append(text, label)
			case routeOmit:
				label := omitLabel(p, "this endpoint does not accept "+kindNoun(p)+" in tool results")
				text = append(text, label)
				parts = append(parts, TextPart(label))
			}
		}
		r := m
		if f == wireChat {
			r.Parts = nil
			r.Content = strings.Join(text, "\n\n")
		} else {
			r.Parts = parts
		}
		out = append(out, r)
	}
	if len(carry) > 0 {
		out = append(out, Message{Role: "user", Parts: carry})
	}
	return out
}

func hasMedia(m Message) bool {
	for _, p := range m.Parts {
		if p.Type != "text" {
			return true
		}
	}
	return false
}

// gateRefuses names why the capability record rules a part out, or "".
func gateRefuses(caps Capabilities, p ContentPart) string {
	if p.Type == "text" {
		return ""
	}
	if ok, known := caps.boolFact(CapModalityPrefix + partKind(p)); known && !ok {
		return "this model does not accept " + partKind(p) + " input"
	}
	return ""
}

func omitLabel(p ContentPart, why string) string {
	return "[" + partMime(p) + " not sent: " + why + "]"
}

// partKind is the word for a part's kind: the modality key of the capability
// record, and the word in a numbered label. An unknown type is its own word.
func partKind(p ContentPart) string {
	switch p.Type {
	case "image_url":
		return "image"
	case "input_audio":
		return "audio"
	case "video_url":
		return "video"
	case "file":
		return "file"
	}
	return p.Type
}

// kindNoun is the kind as the object of "does not accept".
func kindNoun(p ContentPart) string {
	switch k := partKind(p); k {
	case "image", "file":
		return k + "s"
	default:
		return k
	}
}

// partMime is the part's media type, from its data: URI, an audio format or a
// file name; the kind word when none says.
func partMime(p ContentPart) string {
	switch {
	case p.ImageURL != nil:
		if m, ok := dataURIMedia(p.ImageURL.URL); ok {
			return m
		}
	case p.VideoURL != nil:
		if m, ok := dataURIMedia(p.VideoURL.URL); ok {
			return m
		}
	case p.InputAudio != nil:
		if f := strings.ToLower(strings.TrimSpace(p.InputAudio.Format)); f != "" {
			return "audio/" + f
		}
	case p.File != nil:
		if m, ok := dataURIMedia(p.File.FileData); ok {
			return m
		}
		if i := strings.LastIndexByte(p.File.Filename, '.'); i >= 0 {
			if m, _, err := mime.ParseMediaType(mime.TypeByExtension(p.File.Filename[i:])); err == nil {
				return m
			}
		}
	}
	return partKind(p)
}

// dataURIMedia is the media type of a "data:<type>[;…],…" URI, lowercased.
func dataURIMedia(u string) (string, bool) {
	if !strings.HasPrefix(u, "data:") {
		return "", false
	}
	meta, _, ok := strings.Cut(u[len("data:"):], ",")
	if !ok {
		return "", false
	}
	media, _, _ := strings.Cut(meta, ";")
	media = strings.ToLower(strings.TrimSpace(media))
	return media, media != ""
}
