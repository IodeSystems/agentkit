package llm

import (
	"fmt"
	"strings"
)

// Effort: one scale for "how hard should the model think", resolved against the
// endpoint's dialect and the capability record.
//
// Before this file a caller (dun) kept its own table from level to wire fields
// per dialect, and each copy drifted (Anthropic-compat "high" sent nothing, so
// it meant no thinking at all). The table lives here once. A caller picks a
// level; EffortFor says what will be sent, whether that is exact (Native) or an
// approximation by a token budget, and WHY when the level it got is not the
// level it asked for. Never a silent fall to "send nothing": a refusal that
// leaves nothing expressible returns EffortUnset with a Reason.
//
// Pure on purpose (dialect, API shape, model and record in; choice out): a host
// can render a picker, with a reason per greyed-out level, without a Client or
// a request. The Client wrappers only fill in its own state.

// Effort is a reasoning level, ordered from least to most.
type Effort string

const (
	EffortUnset   Effort = ""        // send nothing: the server's default decides
	EffortNone    Effort = "none"    // thinking off, where the endpoint can
	EffortMinimal Effort = "minimal" // as little as the endpoint allows
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// Efforts is every set level, lowest first. EffortUnset is not in it.
var Efforts = []Effort{EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}

// EffortAbout is a one-line description of each level, for a picker.
var EffortAbout = map[Effort]string{
	EffortUnset:   "send nothing; the server's default decides",
	EffortNone:    "no thinking",
	EffortMinimal: "the least thinking the endpoint allows",
	EffortLow:     "light thinking",
	EffortMedium:  "moderate thinking",
	EffortHigh:    "deep thinking",
	EffortXHigh:   "deeper than high",
	EffortMax:     "the most thinking the endpoint allows",
}

// ParseEffort reads a canonical level name, or "" for unset. Aliases ("off")
// are the caller's: a second spelling here would be a second table to keep.
func ParseEffort(s string) (Effort, error) {
	e := Effort(strings.ToLower(strings.TrimSpace(s)))
	if e == EffortUnset {
		return EffortUnset, nil
	}
	if effortIndex(e) < 0 {
		return EffortUnset, fmt.Errorf("llm: unknown effort %q; want one of %s or empty", s, effortNames())
	}
	return e, nil
}

func effortNames() string {
	n := make([]string, len(Efforts))
	for i, e := range Efforts {
		n[i] = string(e)
	}
	return strings.Join(n, ", ")
}

func effortIndex(e Effort) int {
	for i, l := range Efforts {
		if l == e {
			return i
		}
	}
	return -1
}

// EffortChoice is what one wanted level becomes.
type EffortChoice struct {
	Wanted    Effort
	Effective Effort // what is sent; EffortUnset when nothing is
	// Native is false when Effective is approximated by a token budget (llama.cpp
	// minimal/low/medium, every Anthropic-compat level but none).
	Native bool
	// Reason says why Effective differs from Wanted; "" when it does not.
	// "minimal (none not supported: reasoning.thinking_only=true ...)".
	Reason string
	// Cause is the fact key behind the change, or "dialect:<name>" for the
	// static table; "" when Effective == Wanted.
	Cause string
}

// effortRow is how one dialect expresses one wanted level.
type effortRow struct {
	eff    Effort // the level the dialect actually expresses for it
	native bool
	note   string // why eff differs from the wanted level; "" when equal
	ok     bool   // false: not expressible at all (note says why)
}

// Dialect names in Cause/Reason for the Messages API, which is not a Dialect*.
const dialectMessages = "messages"

// Claude models before the effort field. The shim drops the field for them
// (openai-claude-api-shim v0.5.0, shim.go effort()); the list is copied
// because the shim keeps it unexported. Update both together.
var predatesEffort = []string{"claude-haiku-4-5", "claude-sonnet-4-5", "claude-3"}

func rowFor(dialect string, messagesAPI bool, model string, e Effort) effortRow {
	same := effortRow{eff: e, native: true, ok: true}
	to := func(l Effort, why string) effortRow {
		return effortRow{eff: l, native: true, note: why, ok: true}
	}
	if messagesAPI {
		for _, old := range predatesEffort {
			if strings.HasPrefix(model, old) {
				return effortRow{note: "model predates effort"}
			}
		}
		switch e {
		case EffortNone:
			return to(EffortLow, "thinking cannot be turned off on the Messages API")
		case EffortMinimal:
			return to(EffortLow, "the Messages API has no minimal")
		}
		return same
	}
	switch dialect {
	case DialectOpenAI:
		if e == EffortMax {
			return to(EffortXHigh, "the openai dialect has no max")
		}
		return same
	case DialectOpenRouter:
		// xhigh is unverified against OpenRouter's own docs: sent as given.
		if e == EffortMax {
			return to(EffortXHigh, "openrouter has no max")
		}
		return same
	case DialectLlamaCpp:
		switch e {
		case EffortXHigh, EffortMax:
			return to(EffortHigh, "llama.cpp has no "+string(e)+": thinking with no budget is the most it expresses")
		case EffortMinimal, EffortLow, EffortMedium:
			return effortRow{eff: e, native: false, ok: true}
		}
		return same
	case DialectAnthropic:
		switch e {
		case EffortMinimal:
			return effortRow{eff: EffortLow, note: "anthropic budgets start at 1024, which is low", ok: true}
		case EffortNone:
			return same
		}
		return effortRow{eff: e, native: false, ok: true}
	}
	// Unknown dialect, or a raw body (passthrough, probe): every level passes.
	return same
}

// sendsEffortField: the dialects whose body carries reasoning_effort, so the
// record's "sampling.reasoning_effort refused" applies.
func sendsEffortField(dialect string, messagesAPI bool) bool {
	switch dialect {
	case DialectLlamaCpp, DialectAnthropic, DialectOpenRouter:
		return messagesAPI
	}
	return true
}

// Level budgets for the dialects that approximate a level with tokens (decision
// 3, Carl 2026-11-18). ok=false: no budget for that level in that dialect.
func effortBudget(dialect string, e Effort) (int, bool) {
	switch dialect {
	case DialectLlamaCpp:
		switch e {
		case EffortNone:
			return 0, true
		case EffortMinimal:
			return 512, true
		case EffortLow:
			return 1024, true
		case EffortMedium:
			return 4096, true
		}
	case DialectAnthropic:
		switch e {
		case EffortMinimal, EffortLow:
			return 1024, true
		case EffortMedium:
			return 4096, true
		case EffortHigh:
			return 16384, true
		case EffortXHigh:
			return 32768, true
		case EffortMax:
			return 65536, true
		}
	}
	return 0, false
}

// refusal is why the record rules a level out.
type refusal struct{ key, detail string }

// levelFactsApply: whether reasoning.efforts.refused and reasoning.off bind this
// dialect. They are facts about the reasoning_effort / reasoning.effort FIELD
// (what the endpoint answered to a value of it), so only a dialect that sends
// that field is bound: openai, openrouter, the Messages API, and a raw body
// that carries the field (dialect ""). llama.cpp and anthropic-compat express a
// level as enable_thinking / a token budget, which the endpoint never judged:
// applying the facts there would clamp a level that was never refused.
// reasoning.thinking_only is about the model, so it binds every dialect.
func levelFactsApply(dialect string, messagesAPI bool) bool {
	switch dialect {
	case DialectLlamaCpp, DialectAnthropic:
		return messagesAPI
	}
	return true
}

// effortRefusals reads the record: levels it refuses, with the fact behind each.
// field is levelFactsApply for the dialect.
func effortRefusals(caps Capabilities, field bool) map[Effort]refusal {
	out := map[Effort]refusal{}
	if list, _ := caps.RefusedEfforts(); field && len(list) > 0 {
		for _, e := range list {
			out[Effort(e)] = refusal{CapReasoningEffortsRefused, factDetail(caps, CapReasoningEffortsRefused)}
		}
	}
	if only, ok := caps.ThinkingOnly(); ok && only {
		out[EffortNone] = refusal{CapReasoningThinkingOnly, factDetail(caps, CapReasoningThinkingOnly)}
	}
	return out
}

// factDetail: "key=value (source: evidence)", evidence clipped.
func factDetail(caps Capabilities, key string) string {
	f, ok := caps.Facts[key]
	if !ok {
		return key
	}
	s := key + "=" + f.Value + " (" + string(f.Source)
	if f.Evidence != "" {
		s += ": " + clip(f.Evidence, 100)
	}
	return s + ")"
}

// EffortFor resolves a wanted level for a dialect, API shape, model and
// capability record. The clamp, in order:
//
//   - The dialect's row maps the level to one it expresses (max to xhigh on
//     openai); nothing expressible (an old Claude model) is unset.
//   - A level the record does not refuse is used as is.
//   - A refused level (not none) goes to the nearest LOWER level not refused,
//     then the nearest higher: the caller asked for that much thinking or less
//     is the safer reading of a refusal (Carl, 2026-11-18). It never steps down
//     to none: that is a different request.
//   - none uses the record's reasoning.off when known, else none if not
//     refused, else the lowest level the record allows, from minimal up.
//   - Nothing left: EffortUnset, with the reason.
//
// Only levels the dialect expresses as themselves are candidates for a clamp.
func EffortFor(dialect string, messagesAPI bool, model string, caps Capabilities, e Effort) EffortChoice {
	ch := EffortChoice{Wanted: e}
	if e == EffortUnset {
		return ch
	}
	cause := "dialect:" + dialect
	if messagesAPI {
		cause = "dialect:" + dialectMessages
	}
	fail := func(cause, why string) EffortChoice {
		ch.Cause, ch.Reason = cause, fmt.Sprintf("unset (%s not supported: %s)", e, why)
		return ch
	}
	if effortIndex(e) < 0 {
		return fail(cause, "not an effort level")
	}
	row := rowFor(dialect, messagesAPI, model, e)
	if !row.ok {
		return fail(cause, row.note)
	}
	if sendsEffortField(dialect, messagesAPI) {
		if v, _ := caps.fact(CapSamplingPrefix + "reasoning_effort"); v == "refused" {
			return fail(CapSamplingPrefix+"reasoning_effort", factDetail(caps, CapSamplingPrefix+"reasoning_effort"))
		}
	}
	field := levelFactsApply(dialect, messagesAPI)
	refused := effortRefusals(caps, field)
	// reasoning.off is the same kind of fact: a value of the effort field.
	offEffort := func() (string, bool) {
		if !field {
			return "", false
		}
		return caps.OffEffort()
	}
	avail := func(l Effort) bool {
		r := rowFor(dialect, messagesAPI, model, l)
		return r.ok && r.eff == l
	}
	free := func(l Effort) bool { _, bad := refused[l]; return !bad }
	var parts []string
	if row.eff != e {
		parts = append(parts, fmt.Sprintf("%s not supported: %s", e, row.note))
	}
	eff, why := row.eff, refusal{}
	if rf, bad := refused[row.eff]; bad {
		why = rf
		eff = ""
		if row.eff == EffortNone {
			if off, ok := offEffort(); ok && avail(Effort(off)) && free(Effort(off)) {
				eff = Effort(off)
			}
			for i := 1; eff == "" && i < len(Efforts); i++ {
				if l := Efforts[i]; avail(l) && free(l) {
					eff = l
				}
			}
		} else {
			i := effortIndex(row.eff)
			for j := i - 1; eff == "" && j >= 1; j-- {
				if l := Efforts[j]; avail(l) && free(l) {
					eff = l
				}
			}
			for j := i + 1; eff == "" && j < len(Efforts); j++ {
				if l := Efforts[j]; avail(l) && free(l) {
					eff = l
				}
			}
		}
		parts = append(parts, fmt.Sprintf("%s not supported: %s", row.eff, why.detail))
		cause = why.key
	} else if row.eff == EffortNone {
		// Not refused. reasoning.off, when the record knows it, is the pinned or
		// learned way to turn thinking off on this model.
		if off, ok := offEffort(); ok && avail(Effort(off)) && free(Effort(off)) && Effort(off) != EffortNone {
			eff = Effort(off)
			parts = append(parts, fmt.Sprintf("none is sent as %s: %s", off, factDetail(caps, CapReasoningOff)))
			cause = CapReasoningOff
		}
	}
	ch.Effective = eff
	if eff != "" {
		ch.Native = rowFor(dialect, messagesAPI, model, eff).native
	}
	if eff == e {
		return ch
	}
	if len(parts) == 0 {
		parts = append(parts, "nothing expressible")
	}
	shown := string(eff)
	if eff == "" {
		shown = "unset"
		parts = append(parts, "no level left that the record allows")
	}
	ch.Cause = cause
	ch.Reason = fmt.Sprintf("%s (%s)", shown, strings.Join(parts, "; "))
	return ch
}

// EffortSupport is EffortFor for every level, lowest first: a picker's rows. A
// row whose Effective differs from Wanted is unavailable as asked, with Reason.
func EffortSupport(dialect string, messagesAPI bool, model string, caps Capabilities) []EffortChoice {
	out := make([]EffortChoice, len(Efforts))
	for i, e := range Efforts {
		out[i] = EffortFor(dialect, messagesAPI, model, caps, e)
	}
	return out
}

// Effective is what this Client would send for e now, by its dialect, API, model
// and current record.
func (c *Client) Effective(e Effort) EffortChoice {
	return EffortFor(c.dialect(), c.usesMessagesAPI(), c.model, c.Capabilities(), e)
}

// EffortSupport is the per-level support list for this Client, as it is now.
func (c *Client) EffortSupport() []EffortChoice {
	return EffortSupport(c.dialect(), c.usesMessagesAPI(), c.model, c.Capabilities())
}
