// Learning the capability record from traffic, and translating requests by it.
//
// Slice 2 of plan/capabilities-design.md. Three things live here:
//
//   - Observation: facts read off normal traffic (a usage chunk arrived, which
//     delta field carried reasoning, the window an overflow named, whether a
//     forced tool_choice produced a call, which efforts were served). Source
//     observed. Each flips only on a clear signal, never on a count of zero.
//   - Refusals: refusalFacts reads a 4xx chat body that names a request field,
//     through a table of patterns, each with the server it came from. Source
//     refused. The request still fails exactly as before; a body no pattern
//     knows teaches nothing.
//   - Translation: translate rewrites the outgoing body by the record (an off
//     request becomes the effort the model accepts; a refused effort becomes
//     the nearest one above it; a known output limit caps max_tokens). Every
//     change is reported through OnTranslate, so it is never silent.
//
// ResendOnRefusal (Client field) decides whether a refusal that changed the
// translated bytes is sent once more. It is off by default: an open decision.
package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// Translation is one request field the capability record changed on the way
// out. Reported through Client.OnTranslate just before the request is sent.
type Translation struct {
	Key    string // the request field that changed: reasoning_effort, max_tokens
	Sent   string // what went on the wire; "" when the field was dropped
	Wanted string // what the caller asked for; "" when it asked for nothing
}

// effortLadder is the OpenAI Chat Completions reasoning_effort values, lowest
// first. "Nearest above" and "lowest not refused" are positions on it.
var effortLadder = []string{"none", "minimal", "low", "medium", "high", "xhigh"}

func effortRank(e string) int {
	for i, l := range effortLadder {
		if l == e {
			return i
		}
	}
	return len(effortLadder)
}

// Values of stream.reasoning_field.
const (
	reasoningFieldContent   = "reasoning_content" // llama.cpp, DeepSeek
	reasoningFieldReasoning = "reasoning"         // vLLM, OpenRouter
	// "think_tags" (inline <think> in content) is a legal value, but nothing in
	// this package detects inline tags today, so nothing learns it.
)

// --- translation -----------------------------------------------------------

// translate rewrites body by the capability record and returns what it
// changed. Called at the end of chatBody, after every dialect has written its
// fields, so one pass covers the openai dialect and the Messages API (both put
// reasoning_effort on the body as a string) and a max_tokens that opts==nil
// never set. An empty record changes nothing: a host with no record sends
// exactly the bytes it sent before this file existed.
func (c *Client) translate(body map[string]any, opts *ChatOpts) []Translation {
	c.caps.mu.Lock()
	if len(c.caps.facts) == 0 {
		c.caps.mu.Unlock()
		return nil
	}
	caps := Capabilities{Facts: copyFacts(c.caps.facts)}
	c.caps.mu.Unlock()

	var out []Translation
	if want, ok := body["reasoning_effort"].(string); ok {
		if v, _ := caps.fact(CapSamplingPrefix + "reasoning_effort"); v == "refused" {
			// The endpoint refused the field itself, whatever its value.
			delete(body, "reasoning_effort")
			out = append(out, Translation{Key: "reasoning_effort", Wanted: want})
		} else if send := translateEffort(want, caps); send != want {
			body["reasoning_effort"] = send
			out = append(out, Translation{Key: "reasoning_effort", Sent: send, Wanted: want})
		}
	}
	if lim, ok := caps.OutputLimit(); ok {
		want, _ := body["max_tokens"].(int)
		if want == 0 || want > lim {
			body["max_tokens"] = lim
			w := ""
			if want > 0 {
				w = strconv.Itoa(want)
			}
			out = append(out, Translation{Key: "max_tokens", Sent: strconv.Itoa(lim), Wanted: w})
		}
	}
	return out
}

// translateEffort is the effort to send for want under caps.
//
// Off ("none") sends reasoning.off when the record knows it: that is the host's
// pin or what was learned to work. Otherwise, when "none" is refused (or the
// model is thinking-only), it sends the LOWEST effort not refused — the closest
// honest reading of "think as little as you can". Sending nothing instead would
// leave the model's default, which on GLM-5.3 is the most thinking (median 791
// reasoning tokens against 103 at low).
//
// Any other refused effort becomes the nearest one ABOVE it that is not
// refused (accepted or unknown): the caller asked for at least that much
// thinking. Only when nothing above is left does it go below, never down to
// "none" — a known refusal sent again is a certain failure, and "none" would
// change what was asked for into off. A value off the ladder is left alone.
func translateEffort(want string, caps Capabilities) string {
	refused := map[string]bool{}
	list, _ := caps.RefusedEfforts()
	for _, e := range list {
		refused[e] = true
	}
	if only, ok := caps.ThinkingOnly(); ok && only {
		refused["none"] = true
	}
	if want == "none" {
		if off, ok := caps.OffEffort(); ok {
			return off
		}
		if !refused["none"] {
			return "none"
		}
		for _, e := range effortLadder[1:] {
			if !refused[e] {
				return e
			}
		}
		return want
	}
	i := effortRank(want)
	if !refused[want] || i == len(effortLadder) {
		return want
	}
	for j := i + 1; j < len(effortLadder); j++ {
		if !refused[effortLadder[j]] {
			return effortLadder[j]
		}
	}
	for j := i - 1; j >= 1; j-- {
		if !refused[effortLadder[j]] {
			return effortLadder[j]
		}
	}
	return want
}

func (c *Client) reportTranslations(trs []Translation) {
	if c.OnTranslate == nil {
		return
	}
	for _, t := range trs {
		c.OnTranslate(t)
	}
}

// --- record updates ----------------------------------------------------------

// update applies next to the current value of key under the record's lock and
// the same precedence rule as learn (replaces), and reports the change through
// OnLearn after the lock is released. next returns ok=false for "nothing to
// do". Returns whether the value or source changed.
//
// Not learn itself because a comma list is a read-modify-write: two requests
// finishing together would each read the old list and the second write would
// drop the first one's value.
func (c *Client) update(key string, src Source, evidence string, next func(old string) (string, bool)) bool {
	c.caps.mu.Lock()
	old, had := c.caps.facts[key]
	if had && !replaces(src, old.Source) {
		c.caps.mu.Unlock()
		return false
	}
	val, ok := next(old.Value)
	if !ok {
		c.caps.mu.Unlock()
		return false
	}
	f := Fact{Value: val, Source: src, Evidence: evidence, At: time.Now()}
	if c.caps.facts == nil {
		c.caps.facts = map[string]Fact{}
	}
	c.caps.facts[key] = f
	changed := !had || old.Value != val || old.Source != src
	c.caps.mu.Unlock()

	if !changed || c.OnLearn == nil {
		return changed
	}
	l := Learned{Endpoint: normalizeEndpoint(c.baseURL), Model: c.model, Key: key, Fact: f}
	if had {
		p := old
		l.Previous = &p
	}
	c.OnLearn(l)
	return changed
}

// editList adds add to and removes remove from a comma list, deduped, in ladder
// order (values off the ladder last, sorted). ok=false when nothing changed.
func editList(cur, add, remove string) (string, bool) {
	var items []string
	seen := map[string]bool{}
	changed := false
	for _, p := range strings.Split(cur, ",") {
		p = strings.TrimSpace(p)
		if p == "" || seen[p] {
			continue
		}
		if p == remove {
			changed = true
			continue
		}
		seen[p] = true
		items = append(items, p)
	}
	if add != "" && !seen[add] {
		items = append(items, add)
		changed = true
	}
	if !changed {
		return cur, false
	}
	sort.SliceStable(items, func(i, j int) bool {
		ri, rj := effortRank(items[i]), effortRank(items[j])
		if ri != rj {
			return ri < rj
		}
		return items[i] < items[j]
	})
	return strings.Join(items, ","), true
}

// factUpdate is one fact a refusal (or an observation) teaches.
type factUpdate struct {
	Key   string
	Value string
	// List: Value is added to the comma list at Key and removed from the list
	// at Opposite, so an effort is never both accepted and refused. The latest
	// evidence wins per effort, as it does per key.
	List     bool
	Opposite string
	Evidence string
}

func (c *Client) apply(u factUpdate, src Source) bool {
	if !u.List {
		return c.update(u.Key, src, u.Evidence, func(string) (string, bool) { return u.Value, true })
	}
	changed := c.update(u.Key, src, u.Evidence, func(old string) (string, bool) { return editList(old, u.Value, "") })
	if u.Opposite != "" {
		if c.update(u.Opposite, src, u.Evidence, func(old string) (string, bool) { return editList(old, "", u.Value) }) {
			changed = true
		}
	}
	return changed
}

// --- observation -------------------------------------------------------------

// sentRequest is what a chat request asked for that the response can confirm
// or deny. Read off the body as sent (after translation), not off opts.
type sentRequest struct {
	effort string // reasoning_effort on the wire; "" when none was sent
	forced string // tool_choice when it forces a call: "required" or "function:<name>"
	raw    bool   // a probe request: observe nothing (see withRawProbe)
}

// rawProbeKey marks a request made by ProbeCapabilities. A probe must see the
// endpoint's answer to exactly what it chose to send, so its requests skip
// translation (probing "none" after "none" was refused would otherwise send
// "minimal" and record that answer as the answer to "none"), and they teach
// the record nothing as traffic: the probe records its own findings, as
// probed, with evidence.
type rawProbeKey struct{}

func withRawProbe(ctx context.Context) context.Context {
	return context.WithValue(ctx, rawProbeKey{}, true)
}

func isRawProbe(ctx context.Context) bool {
	v, _ := ctx.Value(rawProbeKey{}).(bool)
	return v
}

func sentFrom(body map[string]any) sentRequest {
	var s sentRequest
	s.effort, _ = body["reasoning_effort"].(string)
	if _, ok := body["tools"]; !ok {
		return s
	}
	switch tc := body["tool_choice"].(type) {
	case string:
		if tc == "required" {
			s.forced = "required"
		}
	case map[string]any:
		if fn, ok := tc["function"].(map[string]any); ok {
			if n, _ := fn["name"].(string); n != "" {
				s.forced = "function:" + n
			}
		}
	}
	return s
}

// observeAccepted records the effort a 200 was given. A 200 is the clear signal
// here: an endpoint that does not take a value refuses the request before it
// generates (GLM-5.3: 400; Qwen3.8's template: 500), so a stream that later
// dies says nothing about the field.
//
// TODO(capabilities): accepted is not honored. "none" accepted with as many
// reasoning tokens as the default would mean off is not honored, and only
// Usage.ReportedReasoningTokens can show it. Needed first: the reasoning-token
// count at "none" against the count at the endpoint's default, on the same
// prompt, per model — a threshold set from that measurement, not guessed.
func (c *Client) observeAccepted(s sentRequest) {
	if s.raw {
		return
	}
	if s.effort == "" {
		return
	}
	c.apply(factUpdate{
		Key: CapReasoningEffortsAccepted, Value: s.effort, List: true, Opposite: CapReasoningEffortsRefused,
		Evidence: "sent reasoning_effort=" + s.effort + " → 200",
	}, SourceObserved)
}

// observeToolChoice records whether a forced tool_choice produced a call, from
// a reply that finished normally. A reply cut by length or a filter has no
// verdict: the call may simply not have fit.
func (c *Client) observeToolChoice(s sentRequest, finish string, calls int) {
	if s.raw {
		return
	}
	if s.forced == "" {
		return
	}
	switch finish {
	case "stop", "tool_calls", "function_call":
	default:
		return
	}
	v, ev := "true", fmt.Sprintf("sent tool_choice=%s → %d tool call(s)", s.forced, calls)
	if calls == 0 {
		v, ev = "false", fmt.Sprintf("sent tool_choice=%s → finish_reason=%s with no tool call", s.forced, finish)
	}
	c.apply(factUpdate{Key: CapToolsChoiceForced, Value: v, Evidence: ev}, SourceObserved)
}

// streamObservation is what one stream showed, collected by readStream.
type streamObservation struct {
	usage          bool   // a usage chunk arrived
	done           bool   // the stream ended with [DONE]
	failed         bool   // an error event, or a loop cut (ours or a proxy's)
	reasoningField string // the delta field that first carried reasoning
	finish         string // finish_reason, "" when none arrived
	toolCalls      int    // calls flushed
}

func (c *Client) observeStream(s sentRequest, o streamObservation) {
	if s.raw {
		return
	}
	switch {
	case o.usage:
		// A usage chunk on an errored stream is still a usage chunk.
		c.apply(factUpdate{Key: CapStreamUsage, Value: "true",
			Evidence: "sent stream_options.include_usage → usage chunk"}, SourceObserved)
	case o.done && !o.failed:
		// Every stream asks for include_usage, and the spec puts the usage
		// chunk just before [DONE]. A stream that reached [DONE] without one
		// is the clear signal; one that ended any other way is not, since it
		// may simply have stopped before the usage chunk.
		c.apply(factUpdate{Key: CapStreamUsage, Value: "false",
			Evidence: "sent stream_options.include_usage → [DONE] with no usage chunk"}, SourceObserved)
	}
	if o.reasoningField != "" {
		c.apply(factUpdate{Key: CapStreamReasoningField, Value: o.reasoningField,
			Evidence: "streamed reasoning in delta." + o.reasoningField}, SourceObserved)
	}
	if !o.failed {
		c.observeToolChoice(s, o.finish, o.toolCalls)
	}
}

// learnFromChatError records the window an overflow named. Chat paths only: an
// embedding model's window is not the chat model's. Both the 4xx (statusError)
// and the 5xx (postWithRetry) overflow arrive here as *InputTooLargeError.
func (c *Client) learnFromChatError(err error) bool {
	var big *InputTooLargeError
	if !errors.As(err, &big) || big.ContextTokens <= 0 {
		return false
	}
	ev := fmt.Sprintf("sent a prompt → %d %q", big.Status, clip(firstLineOfBody(big.Body), 120))
	if big.PromptTokens > 0 {
		ev = fmt.Sprintf("sent %d prompt tokens → %d %q", big.PromptTokens, big.Status, clip(firstLineOfBody(big.Body), 120))
	}
	return c.apply(factUpdate{Key: CapLimitsContext, Value: strconv.Itoa(big.ContextTokens), Evidence: ev}, SourceObserved)
}

// chatRefusal turns a non-200 chat response into the same error statusError
// returns, and learns from its body on the way. taught reports whether any
// fact was added or changed.
func (c *Client) chatRefusal(resp *http.Response, sent map[string]any) (taught bool, err error) {
	status, msg := resp.StatusCode, readErrorBody(resp)
	err = statusErrorBody(status, msg)
	for _, u := range refusalFacts(status, msg, sent) {
		if c.apply(u, SourceRefused) {
			taught = true
		}
	}
	if c.learnFromChatError(err) {
		taught = true
	}
	return taught, err
}

// --- refusals ----------------------------------------------------------------

// refusalCtx is what a pattern sees besides its match.
type refusalCtx struct {
	status int
	sent   map[string]any
	effort string // reasoning_effort as sent, "" when none was
}

func (r refusalCtx) evidence(what, match string) string {
	return fmt.Sprintf("sent %s → %d %q", what, r.status, clip(strings.TrimSpace(match), 120))
}

func (r refusalCtx) effortRefused(match string) factUpdate {
	return factUpdate{
		Key: CapReasoningEffortsRefused, Value: r.effort, List: true, Opposite: CapReasoningEffortsAccepted,
		Evidence: r.evidence("reasoning_effort="+r.effort, match),
	}
}

// paramRefused is the fact for "the endpoint does not take field X". Only for a
// field that was sent: on the Messages API the shim writes fields of its own,
// and a refusal of one of those says nothing about the OpenAI-shaped request.
func (r refusalCtx) paramRefused(field, match string) []factUpdate {
	top, _, _ := strings.Cut(field, ".")
	if _, ok := r.sent[top]; !ok {
		return nil
	}
	ev := r.evidence(field, match)
	if field == "grammar" {
		return []factUpdate{{Key: CapDecodeGrammar, Value: "false", Evidence: ev}}
	}
	return []factUpdate{{Key: CapSamplingPrefix + field, Value: "refused", Evidence: ev}}
}

type refusalPattern struct {
	re    *regexp.Regexp
	facts func(m []string, r refusalCtx) []factUpdate
}

// refusalPatterns: one entry per wording, each naming the server it came from.
// A pattern that matches but whose field or value was not sent teaches nothing.
var refusalPatterns = []refusalPattern{
	// OpenAI: "Unsupported parameter: 'temperature' is not supported with this
	// model." (o-series and gpt-5 on temperature, top_p, max_tokens).
	{regexp.MustCompile(`Unsupported parameter: '([\w.\[\]-]+)'`), func(m []string, r refusalCtx) []factUpdate {
		return r.paramRefused(m[1], m[0])
	}},
	// OpenAI: "Unsupported value: 'reasoning_effort' does not support 'none'
	// with this model. Supported values are: 'low', 'medium', and 'high'."
	{regexp.MustCompile(`Unsupported value: '([\w.]+)' does not support '([^']*)'`), func(m []string, r refusalCtx) []factUpdate {
		if m[1] == "reasoning_effort" && r.effort != "" && m[2] == r.effort {
			return []factUpdate{r.effortRefused(m[0])}
		}
		return nil
	}},
	// Azure OpenAI: "Unrecognized request argument supplied: reasoning_effort".
	{regexp.MustCompile(`Unrecognized request argument supplied: ([\w.]+)`), func(m []string, r refusalCtx) []factUpdate {
		return r.paramRefused(m[1], m[0])
	}},
	// Python jsonschema's enum message, from a server that validates the body
	// against the OpenAI request schema: "'none' is not one of ['low',
	// 'medium', 'high']". Only the effort that was sent is refused: the
	// message rarely names the field, so the value is the anchor.
	{regexp.MustCompile(`'([^'\s]+)' is not one of`), func(m []string, r refusalCtx) []factUpdate {
		if r.effort != "" && m[1] == r.effort {
			return []factUpdate{r.effortRefused(m[0])}
		}
		return nil
	}},
	// A thinking-only model refusing off. Fireworks GLM-5.3 answers 400 to
	// reasoning_effort "none" (dun-bb, 2026-11-07); its exact sentence was not
	// kept, so these are the wordings of that refusal seen across providers
	// ("Thinking cannot be disabled for this model", "thinking is required",
	// "does not support disabling thinking"). UNVERIFIED against a captured
	// Fireworks body: replace with it when one is recorded.
	{regexp.MustCompile(`(?i)(?:thinking|reasoning)[\w\s]{0,40}?(?:cannot|can ?not|can't) be (?:disabled|turned off)|(?:thinking|reasoning) (?:is|must be) (?:required|enabled)|(?:cannot|can ?not|can't|does not support|doesn't support) (?:disabl(?:e|ing)|turn(?:ing)? off) (?:thinking|reasoning)`),
		func(m []string, r refusalCtx) []factUpdate {
			out := []factUpdate{{Key: CapReasoningThinkingOnly, Value: "true", Evidence: r.evidence("thinking off", m[0])}}
			if r.effort == "none" {
				out = append(out, r.effortRefused(m[0]))
			}
			return out
		}},
	// A minimum thinking budget. Anthropic: "thinking.budget_tokens: Input
	// should be greater than or equal to 1024"; GLM-5.3 on Fireworks measured
	// the same floor (thinking.budget_tokens ≥ 1024, dun-bb). The field name is
	// kept as written, dotted path included.
	{regexp.MustCompile(`((?:[A-Za-z_]+\.)*budget_tokens)\W[^0-9\n]{0,80}?(?:>=|≥|greater than or equal to|at least)\s*(\d+)`),
		func(m []string, r refusalCtx) []factUpdate {
			ev := r.evidence(m[1], m[0])
			return []factUpdate{
				{Key: CapReasoningBudgetMin, Value: m[2], Evidence: ev},
				{Key: CapReasoningBudgetField, Value: m[1], Evidence: ev},
			}
		}},
	// OpenAI: "max_tokens is too large: 50000. This model supports at most
	// 16384 completion tokens, whereas you provided 50000."
	{regexp.MustCompile(`supports at most (\d+) completion tokens`), func(m []string, r refusalCtx) []factUpdate {
		return []factUpdate{{Key: CapLimitsOutput, Value: m[1], Evidence: r.evidence("max_tokens", m[0])}}
	}},
	// Anthropic: "max_tokens: 100000 > 64000, which is the maximum allowed
	// number of output tokens for claude-...".
	{regexp.MustCompile(`max_tokens: \d+ > (\d+), which is the maximum`), func(m []string, r refusalCtx) []factUpdate {
		return []factUpdate{{Key: CapLimitsOutput, Value: m[1], Evidence: r.evidence("max_tokens", m[0])}}
	}},
}

var (
	// One pydantic error entry starts at a dict whose first key is type (v2)
	// or loc (v1). Splitting there keeps one entry's msg from being read as
	// another's.
	rePydanticEntry = regexp.MustCompile(`\{\s*['"](?:type|loc)['"]\s*:`)
	// loc: ["body", "x"] in JSON (FastAPI), ('body', 'x') in a Python repr
	// (vLLM puts the error list's repr in its message).
	rePydanticLoc = regexp.MustCompile(`['"]loc['"]\s*:\s*[\[(]\s*['"]body['"]\s*,\s*['"]([\w.]+)['"]`)
)

// pydanticFacts reads FastAPI/pydantic validation errors (vLLM, SGLang, any
// FastAPI server with extra="forbid"):
//
//	{"type":"extra_forbidden","loc":["body","x"],"msg":"Extra inputs are not permitted"}
//	{"type":"literal_error","loc":["body","reasoning_effort"],"msg":"Input should be 'low', 'medium' or 'high'"}
func pydanticFacts(body string, r refusalCtx) []factUpdate {
	var out []factUpdate
	idx := rePydanticEntry.FindAllStringIndex(body, -1)
	for i, at := range idx {
		end := len(body)
		if i+1 < len(idx) {
			end = idx[i+1][0]
		}
		entry := body[at[0]:end]
		m := rePydanticLoc.FindStringSubmatch(entry)
		if m == nil {
			continue
		}
		switch {
		case strings.Contains(entry, "Extra inputs are not permitted") ||
			strings.Contains(entry, "extra fields not permitted") ||
			strings.Contains(entry, "extra_forbidden"):
			out = append(out, r.paramRefused(m[1], entry)...)
		case m[1] == "reasoning_effort" && r.effort != "" &&
			(strings.Contains(entry, "literal_error") || strings.Contains(entry, "Input should be")):
			out = append(out, r.effortRefused(entry))
		}
	}
	return out
}

// refusalFacts reads a 4xx chat body that names a request field and returns
// the facts it teaches. sent is the request body as sent. A 429 is not a
// refusal of a field, and a 5xx is not a statement about the request (a
// llama.cpp template's 500 on "high" is real, but so is every crash). An
// unknown body returns nil.
func refusalFacts(status int, body string, sent map[string]any) []factUpdate {
	if status < 400 || status >= 500 || status == 429 || body == "" {
		return nil
	}
	r := refusalCtx{status: status, sent: sent}
	r.effort, _ = sent["reasoning_effort"].(string)
	// A message carried as a JSON string inside a JSON body has its quotes
	// escaped; the patterns are written against the plain text.
	text := strings.ReplaceAll(body, `\"`, `"`)

	var out []factUpdate
	seen := map[string]bool{}
	add := func(us []factUpdate) {
		for _, u := range us {
			if k := u.Key + "\x00" + u.Value; !seen[k] {
				seen[k] = true
				out = append(out, u)
			}
		}
	}
	for _, p := range refusalPatterns {
		for _, m := range p.re.FindAllStringSubmatch(text, -1) {
			add(p.facts(m, r))
		}
	}
	add(pydanticFacts(text, r))
	return out
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "…"
}
