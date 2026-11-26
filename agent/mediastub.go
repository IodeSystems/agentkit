package agent

// mediastub.go — old media stops riding every request, and stays recoverable.
//
// A media part in a tool result or a user entry is in the history, so every
// later request resends it: a screenshot read in round 3 is still 400 KB of
// base64 in round 60. LOD does not reach it, because LOD stubs an entry by the
// length of its TEXT and a read's text is a one-line label. So
// ShaperPolicy.StubMediaOlderThan stubs the PARTS: an entry that at least that
// many model rounds have followed renders with each media part replaced by a
// text label, and its text unchanged.
//
// The rule (Carl, 2026-11-25): media stays recoverable while a reference to it
// is in the active session. The label names the part by llm.MediaRef, a hash of
// its bytes, and tells the model to read that "media:<hex>" path; a files
// toolset whose Config.MediaRef is Session.MediaRef answers it from the stored
// entry. No new tool: read is the one path media takes into the model. Once a
// fold summarizes the entry away, Store.Context no longer returns it and the
// reference no longer resolves; the read says so.
//
// Nothing stored changes. The stub is a render of a COPY of the entry, so the
// Store keeps the parts and a later policy (or a host UI) still has them.
//
// Prompt caching: the boundary is the entry that starts the
// StubMediaOlderThan-th newest round, capped at the pristine tail edge and
// rounded DOWN to a multiple of LODStep, the same stepping as FoldMidLOD's
// lodBoundary. Between steps the boundary does not move, so round N's request
// is a byte-exact prefix of round N+1's; at a step one batch of entries loses
// its media together.

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/iodesystems/agentkit/llm"
)

// mediaBoundary is the index before which entries render without media, or
// 0. A round is one model reply: an assistant entry, or a run of tool calls
// that no assistant entry starts.
func (sh *Shaper) mediaBoundary(entries []Entry, olderEnd int) int {
	n := sh.Policy.StubMediaOlderThan
	if n <= 0 {
		return 0
	}
	var starts []int
	for i, e := range entries {
		switch e.Kind {
		case KindAssistant:
			starts = append(starts, i)
		case KindToolCall:
			if i == 0 || (entries[i-1].Kind != KindAssistant && entries[i-1].Kind != KindToolCall) {
				starts = append(starts, i)
			}
		}
	}
	if len(starts) < n {
		return 0
	}
	// Every entry before starts[len-n] has at least n rounds after it.
	return sh.Policy.lodBoundary(min(starts[len(starts)-n], olderEnd))
}

// stubMedia returns entries with the media parts of every KindToolResult and
// KindUser entry before the media boundary replaced by labels. Pinned entries
// keep theirs. The input slice and the entries' Parts are never written: an
// entry that changes gets a fresh Parts slice in a copy of the slice.
func (sh *Shaper) stubMedia(entries []Entry, olderEnd int, pinned pinSet) []Entry {
	b := sh.mediaBoundary(entries, olderEnd)
	if b == 0 {
		return entries
	}
	calls := map[string]Entry{}
	for _, e := range entries[:b] {
		if e.Kind == KindToolCall && e.ToolCallID != "" {
			calls[e.ToolCallID] = e
		}
	}
	out := entries
	copied := false
	for i := 0; i < b; i++ {
		e := entries[i]
		if pinned.has(i) || (e.Kind != KindToolResult && e.Kind != KindUser) || !hasMediaPart(e.Parts) {
			continue
		}
		from := ""
		if e.Kind == KindUser {
			from = "an earlier message"
		} else {
			from = sourceOf(calls[e.ToolCallID], e.ToolName)
		}
		parts := make([]llm.ContentPart, len(e.Parts))
		for j, p := range e.Parts {
			if p.Type == "text" {
				parts[j] = p
				continue
			}
			parts[j] = llm.TextPart(mediaLabel(p, from, sh.Policy.MediaReadTool))
		}
		if !copied {
			out = append([]Entry(nil), entries...)
			copied = true
		}
		e.Parts = parts
		out[i] = e
	}
	return out
}

func hasMediaPart(parts []llm.ContentPart) bool {
	for _, p := range parts {
		if p.Type != "text" {
			return true
		}
	}
	return false
}

// sourceOf is the "from" clause of a tool result's label: the tool and the
// path it was called with, from the matching call. "" when neither is known.
func sourceOf(call Entry, resultTool string) string {
	tool := call.ToolName
	if tool == "" {
		tool = resultTool
	}
	if tool == "" {
		return ""
	}
	var a struct {
		Path string `json:"path"`
	}
	if json.Unmarshal([]byte(call.Content), &a) == nil && a.Path != "" {
		return fmt.Sprintf("%s %q", tool, a.Path)
	}
	return tool
}

// mediaLabel is the text a stubbed media part renders as:
//
//	[image/png media:3f9a… from file_read "shots/home.png", not resent; call file_read with path "media:3f9a…" to see it again]
func mediaLabel(p llm.ContentPart, from, readTool string) string {
	ref := llm.MediaRef(p)
	s := "[" + llm.MediaMIME(p) + " " + ref
	if from != "" {
		s += " from " + from
	}
	call := "the read file tool"
	if readTool != "" {
		call = readTool
	}
	return s + fmt.Sprintf(", not resent; call %s with path %q to see it again]", call, ref)
}

// MediaRef finds a media part by its llm.MediaRef in the entries the Store's
// Context returns for this session now: a tool result's or a user entry's
// Parts, any Kind that has them; the oldest match wins. false when no entry
// holds it, as after a fold summarized the entry away. A files toolset's
// Config.MediaRef is this method, so a stubbed label's "media:<hex>" path reads
// the stored part back.
func (s *Session) MediaRef(ctx context.Context, ref string) (llm.ContentPart, bool) {
	if ref == "" || s.Store == nil {
		return llm.ContentPart{}, false
	}
	entries, err := s.Store.Context(ctx, s.SessionID)
	if err != nil {
		return llm.ContentPart{}, false
	}
	sortEntries(entries)
	for _, e := range entries {
		for _, p := range e.Parts {
			if p.Type != "text" && llm.MediaRef(p) == ref {
				return p, true
			}
		}
	}
	return llm.ContentPart{}, false
}
