package llm

// Provider records: the capability record is keyed endpoint+model+PROVIDER.
//
// A gateway serves one model id from several upstreams. OpenRouter routes
// "some/model" to whichever provider is up and cheap; corrallm fails over
// between lanes. They differ on the fields agentkit learns about (one upstream
// takes effort max, the next refuses it), so a fact is kept with the route that
// showed it, and a request uses:
//
//   - no pin: every route's record merged, narrowest wins. The next request
//     may land on any route, so anything one route refused or limited applies
//     (the same rule as "limits only narrow, smallest wins", Carl 2026-11-07).
//   - a pin (ChatOpts.Provider with one route and no fallback): the "" record
//     plus the pinned route's record. A route the request cannot reach does
//     not narrow it.
//
// The route is read off the response (providerOf), never guessed from the
// request: OpenRouter falls back to another provider unless told not to.

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

// ProviderRouting is OpenRouter's request `provider` object, the subset that
// decides which route serves the request. Sent as given; an endpoint that is not
// OpenRouter ignores or refuses it, so a host sets it only for OpenRouter.
// https://openrouter.ai/docs/features/provider-routing
type ProviderRouting struct {
	Order          []string `json:"order,omitempty"`           // try these first, in order
	Only           []string `json:"only,omitempty"`            // allow only these
	AllowFallbacks *bool    `json:"allow_fallbacks,omitempty"` // false: never leave Order
}

// pinned is the routes this request can be served by, nil when any route can.
// Only wins; Order pins only with AllowFallbacks false (with fallbacks on,
// OpenRouter may still serve from a route not listed).
func (p *ProviderRouting) pinned() []string {
	switch {
	case p == nil:
		return nil
	case len(p.Only) > 0:
		return p.Only
	case len(p.Order) > 0 && p.AllowFallbacks != nil && !*p.AllowFallbacks:
		return p.Order
	}
	return nil
}

// pinnedFromBody is pinned for a body the Client did not build (TranslateBody):
// a caller's own `provider` object.
func pinnedFromBody(body map[string]any) []string {
	v, ok := body["provider"]
	if !ok {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var p ProviderRouting
	if json.Unmarshal(raw, &p) != nil {
		return nil
	}
	return p.pinned()
}

// providerMatches: a stored route key ("served/provider" or either alone)
// names one of pin. OpenRouter compares provider names without case
// (request slug "deepinfra", response name "DeepInfra"); the provider is the
// key's last segment, and the corrallm model before it may itself hold a "/".
// UNVERIFIED: that every OpenRouter slug equals its response name lowercased.
func providerMatches(key string, pin []string) bool {
	k := strings.ToLower(key)
	for _, p := range pin {
		p = strings.ToLower(strings.TrimSpace(p))
		if p != "" && (k == p || strings.HasSuffix(k, "/"+p)) {
			return true
		}
	}
	return false
}

// joinProvider is the route key: "served/provider" when both are known,
// either alone, "" for a direct endpoint.
func joinProvider(served, provider string) string {
	served, provider = strings.TrimSpace(served), strings.TrimSpace(provider)
	switch {
	case served != "" && provider != "":
		return served + "/" + provider
	case served != "":
		return served
	}
	return provider
}

// providerOf is the route that served a response: X-Corrallm-Served from the
// headers, joined with the provider the body names (bodyProvider).
func providerOf(h http.Header, body []byte) string {
	served := ""
	if h != nil {
		served = h.Get(HeaderServed)
	}
	return joinProvider(served, bodyProvider(body))
}

// bodyProvider is OpenRouter's top-level `provider` (a reply body, a stream
// chunk), else error.metadata.provider_name (an error body or error chunk).
func bodyProvider(body []byte) string {
	var v struct {
		Provider json.RawMessage `json:"provider"`
		Error    *struct {
			Metadata json.RawMessage `json:"metadata"`
		} `json:"error"`
	}
	if len(body) == 0 || json.Unmarshal(body, &v) != nil {
		return ""
	}
	var s string
	if json.Unmarshal(v.Provider, &s) == nil && strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s)
	}
	if v.Error != nil {
		return metadataProvider(v.Error.Metadata)
	}
	return ""
}

// metadataProvider is provider_name from an error's metadata object.
func metadataProvider(md json.RawMessage) string {
	var m struct {
		ProviderName string `json:"provider_name"`
	}
	if len(md) == 0 || json.Unmarshal(md, &m) != nil {
		return ""
	}
	return strings.TrimSpace(m.ProviderName)
}

// lateServed reads corrallm's late header from an SSE comment line,
// ": x-corrallm-served: <model>". corrallm sends it as a comment because the
// served lane is known only after the response headers went out.
func lateServed(line string) (string, bool) {
	rest, ok := strings.CutPrefix(line, ":")
	if !ok {
		return "", false
	}
	name, val, ok := strings.Cut(strings.TrimSpace(rest), ":")
	if !ok || !strings.EqualFold(strings.TrimSpace(name), HeaderServed) {
		return "", false
	}
	val = strings.TrimSpace(val)
	return val, val != ""
}

// --- merge -------------------------------------------------------------------

// mergeNarrowest folds several provider records into the one a request uses.
// Per key, the narrowest value wins:
//
//   - a host fact wins over learned ones (a pin is a pin on every route); among
//     several facts of the same class, the rules below.
//   - reasoning.efforts.refused: the union. reasoning.efforts.accepted: the
//     union, minus anything refused.
//   - limits.*: the smallest. reasoning.budget_min: the largest (a floor).
//   - sampling.<field>: "refused" if any route refused it.
//   - reasoning.thinking_only: true if any. tools.arg_fidelity: lossy if any.
//   - any other boolean (decode.*, tools.*, stream.usage, modality.*,
//     reasoning.supported): false if any, since it says what a route can do.
//   - anything else (reasoning.off, a field name): the newest fact.
//
// The merged fact keeps the Source, Evidence and At of the fact that decided
// it (the newest one for a union). One record is returned as a copy, unchanged.
func mergeNarrowest(recs []map[string]Fact) map[string]Fact {
	switch len(recs) {
	case 0:
		return map[string]Fact{}
	case 1:
		return copyFacts(recs[0])
	}
	byKey := map[string][]Fact{}
	for _, r := range recs {
		for k, f := range r {
			byKey[k] = append(byKey[k], f)
		}
	}
	out := make(map[string]Fact, len(byKey))
	for k, fs := range byKey {
		var host []Fact
		for _, f := range fs {
			if f.Source == SourceHost {
				host = append(host, f)
			}
		}
		if len(host) > 0 {
			fs = host
		}
		out[k] = mergeKey(k, fs)
	}
	if acc, ok := out[CapReasoningEffortsAccepted]; ok {
		if ref, ok := out[CapReasoningEffortsRefused]; ok {
			v := acc.Value
			for _, e := range strings.Split(ref.Value, ",") {
				v, _ = editList(v, "", strings.TrimSpace(e))
			}
			acc.Value = v
			out[CapReasoningEffortsAccepted] = acc
		}
	}
	return out
}

func newest(fs []Fact) Fact {
	best := fs[0]
	for _, f := range fs[1:] {
		if f.At.After(best.At) {
			best = f
		}
	}
	return best
}

// pick is the first fact whose value satisfies want, else the newest.
func pick(fs []Fact, want func(string) bool) Fact {
	var hit []Fact
	for _, f := range fs {
		if want(strings.TrimSpace(f.Value)) {
			hit = append(hit, f)
		}
	}
	if len(hit) > 0 {
		return newest(hit)
	}
	return newest(fs)
}

// pickInt is the fact with the smallest (or largest) valid int; the newest
// when none parses.
func pickInt(fs []Fact, smallest bool) Fact {
	var best *Fact
	bestN := 0
	for i := range fs {
		n, err := strconv.Atoi(strings.TrimSpace(fs[i].Value))
		if err != nil || n <= 0 {
			continue
		}
		if best == nil || (smallest && n < bestN) || (!smallest && n > bestN) {
			best, bestN = &fs[i], n
		}
	}
	if best == nil {
		return newest(fs)
	}
	return *best
}

func mergeKey(key string, fs []Fact) Fact {
	if len(fs) == 1 {
		return fs[0]
	}
	switch {
	case key == CapReasoningEffortsRefused || key == CapReasoningEffortsAccepted:
		f := newest(fs)
		v := ""
		for _, g := range fs {
			for _, e := range strings.Split(g.Value, ",") {
				v, _ = editList(v, strings.TrimSpace(e), "")
			}
		}
		f.Value = v
		return f
	case strings.HasPrefix(key, "limits."):
		return pickInt(fs, true)
	case key == CapReasoningBudgetMin:
		return pickInt(fs, false)
	case strings.HasPrefix(key, CapSamplingPrefix):
		return pick(fs, func(v string) bool { return v == "refused" })
	case key == CapReasoningThinkingOnly:
		return pick(fs, func(v string) bool { b, err := strconv.ParseBool(v); return err == nil && b })
	case key == CapToolsArgFidelity:
		return pick(fs, func(v string) bool { return v == "lossy" })
	}
	for _, f := range fs {
		if _, err := strconv.ParseBool(strings.TrimSpace(f.Value)); err == nil {
			return pick(fs, func(v string) bool { b, err := strconv.ParseBool(v); return err == nil && !b })
		}
	}
	return newest(fs)
}
