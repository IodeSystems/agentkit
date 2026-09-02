package agent

// recap.go — the model rewriting its own recent history.
//
// Compaction is the automatic answer to a full context window: a summarizer
// folds the oldest entries and the model has no say in it. Recap is the
// deliberate one. When a stretch of work went badly — a tool result nobody
// needed to keep, six attempts at the same fix, a misunderstanding corrected
// three turns later — the churn stays in the window forever, and every later
// turn pays for it AND reads the wrong account of what happened.
//
// So the model replaces that stretch with what it now knows to be true. Cheaper
// than compaction (no summarizer call), more accurate (the model writing it has
// the whole span in view), and it can fix the RECORD rather than merely shorten
// it — a misunderstanding removed here stops misleading every later turn.
//
// Ported from dun, 2026-09-02, where it worked and where every other host had to
// go without. The comments below record failures observed in live runs; they are
// the expensive part of this file and are kept verbatim in substance.
//
// Three rules make it safe:
//
//   - It is VISIBLE, not confirmed. Asking a human to approve a rewrite of a
//     span they would have to re-read to judge, mid-turn, has one realistic
//     answer, and a question whose answer is never no is a delay rather than a
//     safeguard. So it applies and REPORTS.
//   - Nothing is DESTROYED. Subsumed entries go to a RecapSink. They leave the
//     context; they do not leave the record, because churn is the evidence you
//     need to fix the tooling that produced it.
//   - Tool calls keep their PAIRS. A call whose result is dropped — or the
//     reverse — is the one shape a provider rejects outright, so a span that
//     would orphan one is refused before anything is applied.

import (
	"fmt"
	"strings"
)

// KindRecap is the citation left in the transcript: a record that a span was
// replaced and where it went. Hosts should filter it out of the rendered
// context — it costs the model nothing and exists for whoever reads the log.
const KindRecap EntryKind = "recap"

// RecapSink is where subsumed entries go when a recap is applied.
//
// An interface because "next to the session file" is a host's answer, not the
// mechanism's: dun writes a numbered sidecar, a host with a database writes a
// row, and a host with neither can decline. Store returns a NAME for the
// citation — a filename, a key, whatever a person would use to find it again.
//
// Returning an error REFUSES the recap. Losing the churn silently is the one
// outcome this design exists to prevent, so a sink that cannot save is a sink
// that stops the operation.
type RecapSink interface {
	Store(entries []Entry) (name string, err error)
}

// RecapSpan is what a recap would do, computed before anything is applied.
type RecapSpan struct {
	// Subsumes are the entries that would leave the context.
	Subsumes []Entry
	// Kept are entries preserved inside the span: tool pairs named by `keep`,
	// and the live call that must survive.
	Kept []Entry
	// Anchor is the entry the phrase matched; nil means the start of the
	// session.
	Anchor *Entry
	// Chars is how much would leave.
	Chars int
	// Unmatched are keep terms that named nothing. Reported rather than
	// ignored: a model that asked to keep a result and silently lost it has been
	// told the opposite of the truth.
	Unmatched []string
	// KeptCalls names the tools kept, for the report.
	KeptCalls []string
}

// PlanRecap works out which entries a recap would replace, and refuses rather
// than guessing.
//
// `from` is matched against entry content from the END backwards, so the phrase
// a model remembers most recently is the one it gets. Everything after the match
// is in the span, except what `keep` names and the tool call currently
// executing — which must survive, or its own result is orphaned the moment it
// returns.
func PlanRecap(entries []Entry, from string, keep []string) (RecapSpan, error) {
	var sp RecapSpan
	if strings.TrimSpace(from) == "" {
		return sp, fmt.Errorf("recap needs `from`: a phrase from the message the churn started after")
	}

	// Search BELOW the live call. The recap call's own arguments contain `from`
	// verbatim, so a plain backward search always matches the recap itself,
	// making every span empty. Found by the first end-to-end test in dun.
	limit := len(entries)
	live := LiveCallID(entries)
	if live != "" {
		for i, e := range entries {
			if e.ToolCallID == live && e.Kind == KindToolCall {
				limit = i
				break
			}
		}
	}

	// Prefer a USER message. A nudge hands the model a phrase from the last user
	// message, the model echoes that phrase in its own assistant turns, and "the
	// last entry containing it" is then an echo three lines back — the span
	// collapses to nothing. Observed live: 1 entry of 85 characters recapped
	// while a 255,720-character result sat untouched.
	start := -1
	for i := limit - 1; i >= 0; i-- {
		if entries[i].Kind == KindUser && strings.Contains(entries[i].Content, from) {
			start = i
			break
		}
	}
	if start < 0 {
		for i := limit - 1; i >= 0; i-- {
			if strings.Contains(entries[i].Content, from) {
				start = i
				break
			}
		}
	}
	if start < 0 {
		return sp, fmt.Errorf("no entry contains %q — quote a phrase exactly as it appears", from)
	}
	sp.Anchor = &entries[start]

	// A keep term may be a tool_call id, a tool NAME, or a substring of the
	// call's arguments — because the model cannot see call ids. They are
	// protocol-level and never appear in its context, so a keep list keyed only
	// on them is unusable by the only caller there is. Observed live: the model
	// invented an id, matched nothing, and silently lost the one result it had
	// asked to keep.
	keeping, matched := map[string]bool{}, map[string]bool{}
	for _, term := range keep {
		if term == "" {
			continue
		}
		// A bare tool NAME keeps only that tool's MOST RECENT call. The model
		// reaches for the name it knows, and keeping every call of it preserved
		// the 255,720-character result the recap existed to remove — measured,
		// on the first run where a model recapped unprompted. A phrase from the
		// arguments is specific, so it keeps every match.
		byName := false
		for _, e := range entries[start+1:] {
			if e.Kind == KindToolCall && e.ToolName == term && e.ToolCallID != live {
				byName = true
			}
		}
		for i := len(entries) - 1; i > start; i-- {
			e := entries[i]
			if e.Kind != KindToolCall || e.ToolCallID == live {
				continue
			}
			hit := e.ToolCallID == term || strings.Contains(e.Content, term) ||
				(byName && e.ToolName == term)
			if !hit {
				continue
			}
			keeping[e.ToolCallID] = true
			matched[term] = true
			sp.KeptCalls = append(sp.KeptCalls, e.ToolName)
			if byName && e.ToolName == term && !strings.Contains(e.Content, term) {
				break // the most recent one only
			}
		}
	}
	for _, term := range keep {
		if term != "" && !matched[term] {
			sp.Unmatched = append(sp.Unmatched, term)
		}
	}

	for _, e := range entries[start+1:] {
		switch {
		case e.Kind == KindRecap:
			continue // a previous citation; it is not conversation
		case e.ToolCallID != "" && (keeping[e.ToolCallID] || e.ToolCallID == live):
			sp.Kept = append(sp.Kept, e)
		case e.Kind == KindAssistant && hasPendingCall(entries, e):
			// The assistant turn that ISSUED the live call: dropping it would
			// leave the call with no message to belong to.
			sp.Kept = append(sp.Kept, e)
		default:
			sp.Subsumes = append(sp.Subsumes, e)
			sp.Chars += len(e.Content)
		}
	}
	if len(sp.Subsumes) == 0 {
		return sp, fmt.Errorf("nothing to recap after %q — the span is already clean", from)
	}
	return sp, nil
}

// LiveCallID is the tool call still awaiting a result — during a recap, the
// recap call itself.
func LiveCallID(entries []Entry) string {
	results := map[string]bool{}
	for _, e := range entries {
		if e.Kind == KindToolResult {
			results[e.ToolCallID] = true
		}
	}
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Kind == KindToolCall && !results[entries[i].ToolCallID] {
			return entries[i].ToolCallID
		}
	}
	return ""
}

// hasPendingCall reports whether e is the assistant entry immediately preceding
// an unanswered tool call.
func hasPendingCall(entries []Entry, e Entry) bool {
	live := LiveCallID(entries)
	if live == "" {
		return false
	}
	for i, x := range entries {
		if x.ToolCallID == live && x.Kind == KindToolCall && i > 0 {
			return entries[i-1].ID == e.ID
		}
	}
	return false
}

// Report is what a person reads AFTER the fact: what went, what stayed, what
// replaced it.
//
// Nothing here is a question — the recap has already applied — so it is written
// as an account, and it carries the replacement in full, because "9 entries
// removed" alone is a number rather than visibility.
func (sp RecapSpan) Report(summary, userEdit, where string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Removed %d entries (~%d characters) from the model's context.\n",
		len(sp.Subsumes), sp.Chars)
	if n := len(sp.KeptCalls); n > 0 {
		s := ""
		if n > 1 {
			s = "s"
		}
		fmt.Fprintf(&b, "Kept the %s call%s and its result.\n", strings.Join(sp.KeptCalls, ", "), s)
	}
	if len(sp.Unmatched) > 0 {
		fmt.Fprintf(&b, "NOTE: %q matched nothing and was NOT kept.\n", strings.Join(sp.Unmatched, ", "))
	}
	b.WriteString("\nReplaced with:\n" + indentRecap(summary))
	if userEdit != "" && sp.Anchor != nil {
		b.WriteString("\n\nAnd your message now reads:\n" + indentRecap(userEdit))
	}
	if where != "" {
		fmt.Fprintf(&b, "\n\nWhat was removed is in %s; it left the model's context, not the record.", where)
	}
	return b.String()
}

func indentRecap(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = "  │ " + l
	}
	return strings.Join(lines, "\n")
}
