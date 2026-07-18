package agent

import (
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/iodesystems/agentkit/llm"
)

// Transclusion — the model REFERENCES large tool output instead of retyping it.
//
// To show a tool's output verbatim to the user, the model normally has to copy
// it into its assistant reply. That copy is OUTPUT tokens — serial decode, the
// slow and expensive half of a round-trip, and drift-prone — spent on bytes the
// model already holds verbatim in its INPUT context. Transclusion splits the
// two audiences of one payload: the model writes a short PLACEHOLDER, and the
// engine splices the real bytes in at the user-render boundary. The model's
// output shrinks to the placeholder; the user still sees the full content; the
// model keeps the full bytes in context to reason over. It is the inverse of
// lifting — lifting defers a result, transclusion forwards one verbatim.
//
// Two placeholder forms, both sourced from a tool RESULT:
//
//	{OUTPUT}   the entire most-recent tool result (any <name> wrappers unwrapped)
//	{name}     the inner content of a <name>...</name> section the tool emitted
//
// A tool that wants named sections wraps them itself, e.g. a shell tool emitting
//
//	<stdout>...</stdout><stderr>...</stderr>
//
// makes {stdout} and {stderr} available; {OUTPUT} is always available with no
// cooperation from the tool.
//
// Expansion is deliberately CONSERVATIVE: only {OUTPUT} and known captured
// {name} placeholders expand. An unknown {token} — ordinary braces in JSON,
// code, or prose — passes through untouched, so transclusion never mangles
// content it did not itself put there. The section grammar is likewise narrow
// (a leading letter/underscore then word chars) so stray angle brackets
// (</div>, generics) are not mistaken for slots.
//
// agent owns the capture + expand + the model-facing convention wording
// (SlotSystemNote); the host owns display. On the live path the engine expands
// TurnResult.Reply; to re-render stored history a host calls ExpandEntries. The
// assistant Entry is PERSISTED RAW (placeholder intact) so model replay stays
// token-lean — expansion is a render-time transform, not a stored one.

const slotOutput = "OUTPUT" // reserved slot name: the whole result

var (
	// sectionRe matches a <name>...</name> capture section. RE2 has no
	// backreferences, so the closing name is captured separately and compared
	// in code; mismatched pairs are skipped. Non-greedy + DOTALL so a section
	// spans newlines but stops at its own close tag.
	sectionRe = regexp.MustCompile(`(?s)<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</([A-Za-z_][A-Za-z0-9_]*)>`)
	// placeholderRe matches a {name} reference in an assistant reply. Narrow on
	// purpose: {"k":1} and other brace content never match.
	placeholderRe = regexp.MustCompile(`\{([A-Za-z_][A-Za-z0-9_]*)\}`)
)

// captureSlots scans one tool RESULT and returns its transclusion slots: each
// well-formed <name>...</name> section by name (inner content, tags stripped),
// plus OUTPUT — the whole result with every such wrapper unwrapped (content
// kept, tags removed) so a spliced {OUTPUT} reads clean. Always returns a
// non-nil map carrying at least OUTPUT.
func captureSlots(result string) map[string]string {
	slots := map[string]string{}
	for _, m := range sectionRe.FindAllStringSubmatch(result, -1) {
		if m[1] != m[3] {
			continue // mismatched open/close tag — not a section
		}
		slots[m[1]] = m[2]
	}
	slots[slotOutput] = sectionRe.ReplaceAllStringFunc(result, func(s string) string {
		m := sectionRe.FindStringSubmatch(s)
		if m[1] != m[3] {
			return s
		}
		return m[2]
	})
	return slots
}

// expandSlots replaces {OUTPUT}/{name} placeholders in text with slot content.
// Only names present in slots expand; every other {token} is left verbatim, so
// prose/JSON/code braces survive untouched. A nil/empty slot map is a no-op.
func expandSlots(text string, slots map[string]string) string {
	if len(slots) == 0 {
		return text
	}
	return placeholderRe.ReplaceAllStringFunc(text, func(s string) string {
		if v, ok := slots[s[1:len(s)-1]]; ok {
			return v
		}
		return s
	})
}

// ExpandEntries returns a display copy of entries with each KindAssistant
// Content expanded against the slots captured from the KindToolResult entries
// in the same set — the seam a host uses to re-render STORED history to a user.
// (The live path already returns an expanded TurnResult.Reply.) The input is
// not mutated; non-assistant entries pass through unchanged.
//
// Slots merge across all results with last-writer-wins, so {OUTPUT} binds to
// the last result in `entries` — reference distinct results by named section
// when several are in play. Pass entries in the order you display them.
func ExpandEntries(entries []Entry) []Entry {
	slots := map[string]string{}
	for _, e := range entries {
		if e.Kind == KindToolResult {
			maps.Copy(slots, captureSlots(e.Content))
		}
	}
	out := make([]Entry, len(entries))
	for i, e := range entries {
		if e.Kind == KindAssistant {
			e.Content = expandSlots(e.Content, slots)
		}
		out[i] = e
	}
	return out
}

// truncateToolMessages caps every tool-role message's content to limit chars
// (see truncateForModel). Mutates in place; the caller passes a fresh per-round
// []llm.Message, so storage is untouched.
func truncateToolMessages(messages []llm.Message, limit int) {
	for i := range messages {
		if messages[i].Role == "tool" {
			messages[i].Content = truncateForModel(messages[i].Content, limit)
		}
	}
}

// truncateForModel shrinks an over-limit tool result for the MODEL's view,
// keeping a head and a tail around a marker that (a) reports how much was cut
// and (b) reminds the model it can write {OUTPUT} (or a named section) to
// surface the full bytes to the user. Cuts land on rune boundaries so the
// preview stays valid UTF-8. Sections present anywhere in the full content are
// named in the marker so the model can address them even when the tag itself
// fell outside the kept window.
func truncateForModel(content string, limit int) string {
	if limit <= 0 || len(content) <= limit {
		return content
	}
	marker := "\n…[truncated " + fmt.Sprintf("%d", len(content)-limit) +
		" of " + fmt.Sprintf("%d", len(content)) +
		" chars — write {OUTPUT} for the full result"
	if names := sectionNames(content); len(names) > 0 {
		marker += ", or a section: " + strings.Join(names, " ")
	}
	marker += "]…\n"

	// Split the budget head-heavy: errors/tails matter, but the opening usually
	// carries the most signal (a command, a header, the first rows).
	head := limit * 2 / 3
	tail := limit - head
	head = backToRuneStart(content, head)
	tail = len(content) - backToRuneStart(content, len(content)-tail)
	return content[:head] + marker + content[len(content)-tail:]
}

// sectionNames returns the {name} placeholders for the well-formed sections in
// content, deduped + sorted — for the truncation marker's hint.
func sectionNames(content string) []string {
	seen := map[string]bool{}
	for _, m := range sectionRe.FindAllStringSubmatch(content, -1) {
		if m[1] == m[3] {
			seen[m[1]] = true
		}
	}
	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for n := range seen {
		out = append(out, "{"+n+"}")
	}
	sort.Strings(out)
	return out
}

// backToRuneStart backs i down to the nearest rune boundary at or below it, so
// slicing content[:i] never splits a multi-byte rune.
func backToRuneStart(content string, i int) int {
	if i >= len(content) {
		return len(content)
	}
	for i > 0 && !utf8.RuneStart(content[i]) {
		i--
	}
	return i
}

// SlotSystemNote is the canonical system-prompt fragment that teaches the model
// the transclusion convention. A host appends it to Session.System when it
// wires transclusion — shipped here (like PendingResult's wording) so consumers
// don't each reinvent it.
const SlotSystemNote = "When a tool's output is large and the user needs to see it verbatim, do NOT retype it. " +
	"Reference it by placeholder instead: write {OUTPUT} for the entire most-recent tool result, " +
	"or {name} for a section the tool wrapped as <name>...</name>. The user sees the real content " +
	"spliced in where you wrote the placeholder. A tool result may be shown to you TRUNCATED (a " +
	"'[truncated …]' marker says so and lists any sections) — {OUTPUT}/{name} still surface the " +
	"complete bytes to the user even when your view was cut, so you never need the full text in your " +
	"reply to show it. Only placeholders that name real tool output expand; ordinary braces are left alone."
