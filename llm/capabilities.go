// Capabilities: what one (endpoint, model) is known to do, kept in memory for
// a Client's life and reported to the host as it changes.
//
// Two endpoints that both speak "the OpenAI API" differ on almost every
// optional field, and they differ silently (Fireworks answers 400 to
// reasoning_effort "none"; a llama.cpp template answers 500 to "high"; the
// window is unknown on some hosts). The record is where those answers live.
//
// agentkit does NOT store the record. The host seeds it with SetCapabilities
// and receives every change through OnLearn, then keeps it wherever it keeps
// state. See plan/capabilities-design.md.
//
// The facts are a keyed map of strings, not a struct of typed fields, so a host
// can show every fact with its provenance, store the record without knowing the
// keys, and keep a stored record that holds a key added later. Typed getters
// parse the strings and say "not known" for a value they cannot read.
package llm

import (
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Source says where a Fact came from. It decides which fact wins (see learn).
type Source string

const (
	// SourceHost: pinned by the host's config. Learning never replaces it.
	SourceHost Source = "host"
	// SourceRefused: a 4xx that named the field.
	SourceRefused Source = "refused"
	// SourceObserved: read off normal traffic.
	SourceObserved Source = "observed"
	// SourceProbed: an explicit probe run.
	SourceProbed Source = "probed"
	// SourceDeclared: the endpoint's own /models metadata.
	SourceDeclared Source = "declared"
)

// isEvidence reports whether s is backed by something that was seen on the
// wire, as opposed to something the endpoint claims about itself.
func (s Source) isEvidence() bool {
	return s == SourceRefused || s == SourceObserved || s == SourceProbed
}

// Keys of Capabilities.Facts. A key not listed here is legal: the map is open.
const (
	CapReasoningOff             = "reasoning.off"              // effort that turns thinking off, or lowest: none, minimal, low
	CapReasoningEffortsAccepted = "reasoning.efforts.accepted" // comma list
	CapReasoningEffortsRefused  = "reasoning.efforts.refused"  // comma list
	CapReasoningThinkingOnly    = "reasoning.thinking_only"    // bool
	CapReasoningBudgetField     = "reasoning.budget_field"     // field name
	CapReasoningBudgetMin       = "reasoning.budget_min"       // int tokens
	CapLimitsContext            = "limits.context"             // int tokens
	CapLimitsOutput             = "limits.output"              // int tokens
	CapStreamUsage              = "stream.usage"               // bool
	CapStreamReasoningField     = "stream.reasoning_field"     // reasoning_content, reasoning, think_tags
	CapToolsChoiceForced        = "tools.choice_forced"        // bool
	CapToolsParallel            = "tools.parallel"             // bool
	CapToolsArgFidelity         = "tools.arg_fidelity"         // exact, lossy
	CapDecodeGrammar            = "decode.grammar"             // bool
	CapDecodeGrammarWithTools   = "decode.grammar_with_tools"  // bool
	CapDecodeJSONSchema         = "decode.json_schema"         // bool
	CapDecodeGuided             = "decode.guided"              // bool
	CapDecodeStrictTools        = "decode.strict_tools"        // bool
	CapSamplingPrefix           = "sampling."                  // + param; accepted, refused
	CapModalityPrefix           = "modality."                  // + image, audio, video, file; bool
)

// Fact is one thing known about a key, with enough provenance for a host to
// show it, expire it, or override it. A fact learned from one bad response
// would otherwise persist forever, and now in the host's store as well.
type Fact struct {
	Value    string    // canonical text form; the typed getters parse it
	Source   Source    //
	Evidence string    // what was sent and what came back, short
	At       time.Time // when it was learned
}

// Capabilities is what one (endpoint, model) is known to do.
type Capabilities struct {
	Endpoint string          // normalized base URL: scheme+host+path, no trailing slash
	Model    string          // the model name the Client was given, not a resolved alias
	Facts    map[string]Fact // keyed by the Cap* constants
}

// Learned is what OnLearn receives: one fact that was added or changed.
type Learned struct {
	Endpoint, Model, Key string
	Fact                 Fact
	Previous             *Fact // nil when the key had no fact before
}

// capRecord is the Client's in-memory record. Its own mutex, not a Client-wide
// one: learning happens on the request goroutine while the host snapshots from
// another, and neither may wait on the HTTP path.
type capRecord struct {
	mu    sync.Mutex
	facts map[string]Fact
}

// normalizeEndpoint reduces a base URL to scheme+host+path with no trailing
// slash, so the same endpoint written two ways keys the same stored record.
// Query, fragment and userinfo are dropped: an api key in the URL must not end
// up in a host's store. A string that does not parse as a URL is only trimmed.
func normalizeEndpoint(raw string) string {
	raw = strings.TrimSpace(raw)
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return strings.TrimRight(raw, "/")
	}
	return strings.ToLower(u.Scheme) + "://" + strings.ToLower(u.Host) + strings.TrimRight(u.EscapedPath(), "/")
}

func copyFacts(in map[string]Fact) map[string]Fact {
	out := make(map[string]Fact, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// SetCapabilities seeds the record from the host's store. Call it before first
// use. The seeded facts keep the sources they were stored with, so a stored
// "refused" still beats a fresh "declared" after a restart. It replaces the
// whole record; c.Endpoint and c.Model of the argument are ignored, because the
// record always describes this Client's own endpoint and model. OnLearn is not
// called: the host already holds what it just passed in.
func (c *Client) SetCapabilities(in Capabilities) {
	c.caps.mu.Lock()
	defer c.caps.mu.Unlock()
	c.caps.facts = copyFacts(in.Facts)
}

// Capabilities returns a snapshot copy of the record, for display and for
// storing. Changing the result does not change the Client.
func (c *Client) Capabilities() Capabilities {
	c.caps.mu.Lock()
	facts := copyFacts(c.caps.facts)
	c.caps.mu.Unlock()
	return Capabilities{Endpoint: normalizeEndpoint(c.baseURL), Model: c.model, Facts: facts}
}

// learn applies one fact under the precedence rule and reports it.
//
//  1. A host fact is never replaced by learning.
//  2. Evidence (refused, observed, probed) replaces a declared fact and an
//     older evidence fact of the same key: the latest evidence wins, so a model
//     fixed upstream is learned again.
//  3. Declared replaces only declared.
//
// OnLearn runs AFTER the lock is released and only when the value or the source
// changed. Same value and source with newer evidence is stored quietly, so a
// snapshot carries the latest At without the host hearing about every request.
// A fact with source host is not learned: only SetCapabilities pins.
func (c *Client) learn(key string, f Fact) {
	if f.Source == SourceHost || f.Source == "" {
		return
	}
	if f.At.IsZero() {
		f.At = time.Now()
	}
	c.caps.mu.Lock()
	old, had := c.caps.facts[key]
	if had && !replaces(f.Source, old.Source) {
		c.caps.mu.Unlock()
		return
	}
	if c.caps.facts == nil {
		c.caps.facts = map[string]Fact{}
	}
	c.caps.facts[key] = f
	changed := !had || old.Value != f.Value || old.Source != f.Source
	c.caps.mu.Unlock()

	if !changed || c.OnLearn == nil {
		return
	}
	l := Learned{Endpoint: normalizeEndpoint(c.baseURL), Model: c.model, Key: key, Fact: f}
	if had {
		p := old
		l.Previous = &p
	}
	c.OnLearn(l)
}

// replaces reports whether a new fact from src may replace an existing fact
// from old.
func replaces(src, old Source) bool {
	switch {
	case old == SourceHost:
		return false
	case src.isEvidence():
		return true
	case src == SourceDeclared:
		return old == SourceDeclared
	}
	return false
}

// --- typed getters ---------------------------------------------------------
//
// Each returns (value, known). A missing key, an empty value, or a value that
// does not parse is "not known": a stored record from another version must not
// make the Client act on garbage.

func (c Capabilities) fact(key string) (string, bool) {
	f, ok := c.Facts[key]
	if !ok {
		return "", false
	}
	v := strings.TrimSpace(f.Value)
	return v, v != ""
}

func (c Capabilities) intFact(key string) (int, bool) {
	v, ok := c.fact(key)
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, false
	}
	return n, true
}

func (c Capabilities) boolFact(key string) (bool, bool) {
	v, ok := c.fact(key)
	if !ok {
		return false, false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false, false
	}
	return b, true
}

func (c Capabilities) listFact(key string) ([]string, bool) {
	v, ok := c.fact(key)
	if !ok {
		return nil, false
	}
	var out []string
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out, len(out) > 0
}

// ContextWindow is the model's window in tokens (limits.context).
func (c Capabilities) ContextWindow() (int, bool) { return c.intFact(CapLimitsContext) }

// OutputLimit is the most tokens one reply may have (limits.output).
func (c Capabilities) OutputLimit() (int, bool) { return c.intFact(CapLimitsOutput) }

// OffEffort is the effort that turns thinking off, or lowest (reasoning.off).
func (c Capabilities) OffEffort() (string, bool) { return c.fact(CapReasoningOff) }

// RefusedEfforts are efforts the endpoint answered with an error.
func (c Capabilities) RefusedEfforts() ([]string, bool) {
	return c.listFact(CapReasoningEffortsRefused)
}

// AcceptedEfforts are efforts the endpoint served.
func (c Capabilities) AcceptedEfforts() ([]string, bool) {
	return c.listFact(CapReasoningEffortsAccepted)
}

// ThinkingOnly is true when thinking cannot be turned off.
func (c Capabilities) ThinkingOnly() (bool, bool) { return c.boolFact(CapReasoningThinkingOnly) }

// ToolChoiceHonored is true when a forced tool_choice produces a call.
func (c Capabilities) ToolChoiceHonored() (bool, bool) { return c.boolFact(CapToolsChoiceForced) }

// StreamUsage is true when a stream carries a usage chunk.
func (c Capabilities) StreamUsage() (bool, bool) { return c.boolFact(CapStreamUsage) }

// ReasoningField names the delta field that carries reasoning text.
func (c Capabilities) ReasoningField() (string, bool) { return c.fact(CapStreamReasoningField) }

// Keys returns the fact keys in sorted order, for stable display.
func (c Capabilities) Keys() []string {
	ks := make([]string, 0, len(c.Facts))
	for k := range c.Facts {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
