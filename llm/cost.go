package llm

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Money and limits, as the endpoint reports them.
//
// Token counts say how much work a call was; they do not say what it cost or
// how close the caller is to being refused. Gateways that bill do say so, each
// in its own place:
//
//   - LiteLLM proxy stamps the response cost (USD) on non-streaming responses
//     (x-litellm-response-cost), the key's spend and budget on every response
//     (x-litellm-key-spend, x-litellm-key-max-budget), and the rate-limit window
//     in the OpenAI shape (x-ratelimit-*). On a stream the cost header is absent
//     — headers go out before the cost is known — and the proxy puts it in the
//     final usage chunk instead (usage.cost), when the operator turned on
//     include_cost_in_streaming_usage.
//   - OpenRouter always puts usage.cost (credits) in the usage object, on the
//     final chunk of a stream.
//   - OpenAI sends x-ratelimit-{limit,remaining,reset}-{requests,tokens}.
//   - Anthropic sends anthropic-ratelimit-{requests,tokens,input-tokens,
//     output-tokens}-{limit,remaining,reset}. LiteLLM forwards a provider's own
//     headers with an llm_provider- prefix, so the Anthropic family is also read
//     under that prefix.
//
// Everything here is stamped client-side from the response, like ServerTiming:
// zero-valued and Reported=false against an endpoint that says nothing, because
// "costs nothing" and "did not say" are different facts.

// Currencies a cost or budget can be in. Empty means the source did not say.
const (
	CurrencyUSD = "USD"
	// CurrencyCredits is OpenRouter's unit for usage.cost and its key limits.
	// Its docs do not state the exchange rate, so it is not relabelled USD.
	CurrencyCredits = "credits"
)

// LiteLLM's per-response headers. Values are decimal strings; a header whose
// value is None on the proxy side is omitted.
const (
	HeaderLiteLLMResponseCost = "X-Litellm-Response-Cost"
	HeaderLiteLLMKeySpend     = "X-Litellm-Key-Spend"
	HeaderLiteLLMKeyMaxBudget = "X-Litellm-Key-Max-Budget"
)

// Cost is the money one response cost.
type Cost struct {
	// Amount in Currency. A reported 0 is a real zero (a cache hit, a free
	// model).
	Amount float64
	// Currency is CurrencyUSD, CurrencyCredits, or "" when the source names
	// none (usage.cost from an endpoint that is neither LiteLLM nor
	// OpenRouter).
	Currency string
	// Reported is true when the response carried a cost at all.
	Reported bool
}

// RateWindow is one rate-limit window: requests, or tokens of some kind.
type RateWindow struct {
	// Limit is the window's size. 0 = not reported (a limit of zero is not a
	// window anyone is sent).
	Limit int64
	// Remaining is what is left in the window. Meaningful only when
	// HasRemaining: 0 left is the most important value this can carry.
	Remaining    int64
	HasRemaining bool
	// ResetAt is when the window is full again, absolute — a relative reset
	// ("6m0s") is added to the time the response arrived. Zero = not reported.
	ResetAt time.Time
	// Reported is true when any of the window's headers was present.
	Reported bool
}

// Fraction is Remaining/Limit, and ok=false when either is not reported.
func (w RateWindow) Fraction() (f float64, ok bool) {
	if !w.HasRemaining || w.Limit <= 0 {
		return 0, false
	}
	return float64(w.Remaining) / float64(w.Limit), true
}

// RateLimits is the rate-limit snapshot one response carried.
type RateLimits struct {
	Requests RateWindow
	// Tokens is the combined token window (OpenAI, LiteLLM), or on Anthropic
	// the most restrictive token limit in effect.
	Tokens RateWindow
	// InputTokens and OutputTokens are Anthropic's split windows.
	InputTokens  RateWindow
	OutputTokens RateWindow
	// Reported is true when any window was.
	Reported bool
}

// Windows lists the reported windows with a short name each, in a fixed order.
func (l RateLimits) Windows() []NamedWindow {
	var out []NamedWindow
	for _, w := range []NamedWindow{
		{"requests", l.Requests}, {"tokens", l.Tokens},
		{"input tokens", l.InputTokens}, {"output tokens", l.OutputTokens},
	} {
		if w.Window.Reported {
			out = append(out, w)
		}
	}
	return out
}

// NamedWindow is a RateWindow with its name, for display.
type NamedWindow struct {
	Name   string
	Window RateWindow
}

// rate-limit header families, in precedence order. The gateway's own headers
// come first: behind LiteLLM, x-ratelimit-* are the KEY's limits when it has
// any, which bind before the provider's.
var rateFamilies = []struct {
	prefix string
	// name builds a header from a window ("requests") and a field ("limit").
	name func(prefix, window, field string) string
}{
	{"x-ratelimit-", openAIHeader},
	{"anthropic-ratelimit-", anthropicHeader},
	{"llm_provider-anthropic-ratelimit-", anthropicHeader},
	{"llm_provider-x-ratelimit-", openAIHeader},
}

// x-ratelimit-remaining-requests: field before window.
func openAIHeader(prefix, window, field string) string { return prefix + field + "-" + window }

// anthropic-ratelimit-requests-remaining: window before field.
func anthropicHeader(prefix, window, field string) string { return prefix + window + "-" + field }

// rateLimitsFrom reads every family it knows. Per window, the first family that
// reports it wins; families are not mixed within one window, so a limit and a
// remaining from two different sources never form a fraction.
func rateLimitsFrom(h http.Header, now time.Time) RateLimits {
	var l RateLimits
	if h == nil {
		return l
	}
	for _, w := range []struct {
		name   string
		dst    *RateWindow
		openAI bool // the OpenAI shape has only requests and tokens
	}{
		{"requests", &l.Requests, true},
		{"tokens", &l.Tokens, true},
		{"input-tokens", &l.InputTokens, false},
		{"output-tokens", &l.OutputTokens, false},
	} {
		for _, f := range rateFamilies {
			isOpenAI := strings.HasSuffix(f.prefix, "x-ratelimit-")
			if isOpenAI && !w.openAI {
				continue
			}
			if win := windowFrom(h, now, f.prefix, w.name, f.name); win.Reported {
				*w.dst = win
				l.Reported = true
				break
			}
		}
	}
	return l
}

func windowFrom(h http.Header, now time.Time, prefix, window string, name func(string, string, string) string) RateWindow {
	var w RateWindow
	if n, ok := parseCount(h.Get(name(prefix, window, "limit"))); ok && n > 0 {
		w.Limit, w.Reported = n, true
	}
	if n, ok := parseCount(h.Get(name(prefix, window, "remaining"))); ok {
		w.Remaining, w.HasRemaining, w.Reported = n, true, true
	}
	if t, ok := parseReset(h.Get(name(prefix, window, "reset")), now); ok {
		w.ResetAt, w.Reported = t, true
	}
	return w
}

// parseCount reads a non-negative count. A float ("59.0") is accepted, since a
// gateway re-serializing through a dynamic language may write one.
func parseCount(v string) (int64, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if n, err := strconv.ParseInt(v, 10, 64); err == nil {
		return n, n >= 0
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
		return int64(f), true
	}
	return 0, false
}

// parseReset reads a reset in any of the shapes seen: a Go-style duration
// (OpenAI: "1s", "6m0s", "20ms"), plain seconds (an integer or float), or an
// RFC 3339 time (Anthropic). Relative shapes are taken from now.
func parseReset(v string, now time.Time) (time.Time, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return time.Time{}, false
	}
	if t, err := time.Parse(time.RFC3339, v); err == nil {
		return t, true
	}
	if d, err := time.ParseDuration(v); err == nil && d >= 0 {
		return now.Add(d), true
	}
	if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
		return now.Add(time.Duration(f * float64(time.Second))), true
	}
	return time.Time{}, false
}

// parseMoney reads a decimal amount. Negative, NaN and the proxy's "None" are
// not amounts.
func parseMoney(v string) (float64, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f != f || f < 0 {
		return 0, false
	}
	return f, true
}

// isLiteLLM reports whether a response came through a LiteLLM proxy: it stamps
// x-litellm-* on everything it serves.
func isLiteLLM(h http.Header) bool {
	for k := range h {
		if strings.HasPrefix(strings.ToLower(k), "x-litellm-") {
			return true
		}
	}
	return false
}

// costFromHeader reads LiteLLM's response-cost header (USD).
func costFromHeader(h http.Header) Cost {
	if h == nil {
		return Cost{}
	}
	if f, ok := parseMoney(h.Get(HeaderLiteLLMResponseCost)); ok {
		return Cost{Amount: f, Currency: CurrencyUSD, Reported: true}
	}
	return Cost{}
}

// keyBudgetFromHeader reads LiteLLM's per-response key spend and budget. It is
// the same number /key/info would return, delivered for free on every call —
// but without the reset time, which only /key/info has.
func keyBudgetFromHeader(h http.Header) KeyBudget {
	if h == nil {
		return KeyBudget{}
	}
	spend, ok := parseMoney(h.Get(HeaderLiteLLMKeySpend))
	if !ok {
		return KeyBudget{}
	}
	b := KeyBudget{Spend: spend, Currency: CurrencyUSD, Source: BudgetSourceLiteLLMHeaders, Reported: true}
	if max, ok := parseMoney(h.Get(HeaderLiteLLMKeyMaxBudget)); ok {
		b.setLimit(max)
	}
	return b
}

// responseMeta is everything a response's headers said, read once when they
// arrive and stamped on every Usage the body produces.
type responseMeta struct {
	timing ServerTiming
	cost   Cost
	limits RateLimits
	budget KeyBudget
	// bodyCurrency is the currency a usage.cost in the body is in, when the
	// response or the endpoint says which.
	bodyCurrency string
}

func (c *Client) metaFrom(h http.Header) responseMeta {
	m := responseMeta{
		timing: serverTimingFrom(h),
		cost:   costFromHeader(h),
		limits: rateLimitsFrom(h, time.Now()),
		budget: keyBudgetFromHeader(h),
	}
	switch {
	case isLiteLLM(h):
		m.bodyCurrency = CurrencyUSD // LiteLLM prices from its own USD cost map
	case strings.Contains(c.baseURL, "openrouter.ai"):
		m.bodyCurrency = CurrencyCredits
	}
	return m
}

// stamp copies the response's header facts onto a Usage, and the body's
// usage.cost when it had one. A cost in the body wins over the header: on a
// stream only the body knows it, and where both exist they are one number.
func (m responseMeta) stamp(u *Usage, bodyCost *float64) {
	u.Server = m.timing
	u.Cost = m.cost
	if bodyCost != nil && *bodyCost >= 0 {
		u.Cost = Cost{Amount: *bodyCost, Currency: m.bodyCurrency, Reported: true}
	}
	u.Limits = m.limits
	u.KeyBudget = m.budget
}

// reported is whether the headers said anything worth a Usage of their own.
func (m responseMeta) reported() bool {
	return m.cost.Reported || m.limits.Reported || m.budget.Reported || m.timing.Reported
}

// usageCost pulls usage.cost out of a raw event or response. A second, narrow
// decode rather than a field on Usage: the wire carries a bare number, and
// Usage.Cost is a struct that also needs a currency the body does not state.
func usageCost(data []byte) *float64 {
	var v struct {
		Usage struct {
			Cost *float64 `json:"cost"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &v) != nil {
		return nil
	}
	return v.Usage.Cost
}
