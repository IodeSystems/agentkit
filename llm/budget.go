package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Reading a key's spend budget, where the gateway exposes one.
//
// Two gateways document an endpoint a key may call about itself:
//
//   - LiteLLM proxy: GET /key/info (no ?key= → the caller's own key) answers
//     {"key": ..., "info": {"spend", "max_budget", "budget_duration",
//     "budget_reset_at", ...}} in USD. A non-admin key may read its own row.
//   - OpenRouter: GET /api/v1/key answers {"data": {"usage", "limit",
//     "limit_remaining", "limit_reset", "usage_daily", ..., "effective_budget"}}.
//     Key fields are credits; the budget objects are USD.
//
// Anything else has no such endpoint that this client knows of, and gets
// ErrBudgetUnsupported rather than a guess. Detection probes the documented
// paths relative to the base URL, once per base URL and key, and remembers what
// it found — including nothing.

// ErrBudgetUnsupported is returned by KeyBudget for an endpoint that exposes no
// key budget this client knows how to read.
var ErrBudgetUnsupported = errors.New("llm: endpoint exposes no key budget")

// Where a KeyBudget came from.
const (
	BudgetSourceLiteLLM        = "litellm"         // GET /key/info
	BudgetSourceLiteLLMHeaders = "litellm-headers" // x-litellm-key-spend on a response
	BudgetSourceOpenRouter     = "openrouter"      // GET /api/v1/key
)

// budgetProbeTimeout bounds each probe request. The endpoints are database
// reads on the gateway; a slow one is not worth holding a caller for.
const budgetProbeTimeout = 5 * time.Second

// KeyBudget is a key's spend against its budget.
type KeyBudget struct {
	// Spend is what the key has spent in the current budget period (or all
	// time when the budget does not reset).
	Spend float64
	// Limit is the budget. Meaningful only when HasLimit; false means no limit
	// is set or none was reported.
	Limit    float64
	HasLimit bool
	// Remaining is what is left. Meaningful only when HasRemaining.
	Remaining    float64
	HasRemaining bool
	// ResetAt is when the budget resets. Zero = not reported or never.
	ResetAt time.Time
	// Interval is the reset period as the gateway names it ("30d", "monthly").
	// Empty = not reported or never.
	Interval string
	// Currency of every amount above.
	Currency string
	// Source is one of the BudgetSource* constants.
	Source string
	// Reported is true when any of this was reported.
	Reported bool
}

// Fraction is Remaining/Limit, and ok=false when either is unknown.
func (b KeyBudget) Fraction() (f float64, ok bool) {
	if !b.HasLimit || !b.HasRemaining || b.Limit <= 0 {
		return 0, false
	}
	return b.Remaining / b.Limit, true
}

// setLimit records a limit and derives what remains of it.
func (b *KeyBudget) setLimit(limit float64) {
	b.Limit, b.HasLimit = limit, true
	b.Remaining, b.HasRemaining = limit-b.Spend, true
	if b.Remaining < 0 {
		b.Remaining = 0
	}
}

// budgetRoute is a detected budget endpoint. url=="" after a probe means none.
type budgetRoute struct {
	url    string
	source string
}

var budgetRoutes sync.Map // baseURL+apiKey -> *budgetRoute

// KeyBudget reads this client's key budget from the gateway.
//
// The first call probes for a supported endpoint (each probe bounded by a short
// timeout) and remembers the answer for this base URL and key; later calls make
// one request. Returns ErrBudgetUnsupported when no supported endpoint
// answered. A transport failure is returned as itself and is NOT remembered, so
// a gateway that was down is probed again next time.
func (c *Client) KeyBudget(ctx context.Context) (KeyBudget, error) {
	key := c.baseURL + "\x00" + c.apiKey
	if v, ok := budgetRoutes.Load(key); ok {
		r := v.(*budgetRoute)
		if r.url == "" {
			return KeyBudget{}, ErrBudgetUnsupported
		}
		b, _, err := c.fetchBudget(ctx, *r)
		return b, err
	}
	var transportErr error
	for _, cand := range c.budgetCandidates() {
		b, definitive, err := c.fetchBudget(ctx, cand)
		if err == nil {
			budgetRoutes.Store(key, &cand)
			return b, nil
		}
		if !definitive && transportErr == nil {
			transportErr = err
		}
	}
	if transportErr != nil {
		return KeyBudget{}, transportErr
	}
	budgetRoutes.Store(key, &budgetRoute{})
	return KeyBudget{}, ErrBudgetUnsupported
}

// budgetCandidates are the documented endpoints, in the order worth trying.
// LiteLLM's sits at the root, beside /v1; OpenRouter's inside it
// (https://openrouter.ai/api/v1/key).
func (c *Client) budgetCandidates() []budgetRoute {
	root := strings.TrimSuffix(c.baseURL, "/v1")
	return []budgetRoute{
		{url: root + "/key/info", source: BudgetSourceLiteLLM},
		{url: root + "/v1/key", source: BudgetSourceOpenRouter},
	}
}

// fetchBudget asks one endpoint. definitive reports whether the endpoint
// ANSWERED (any HTTP status, any body): a definitive failure means "not this
// kind of gateway", a non-definitive one means "could not ask".
func (c *Client) fetchBudget(ctx context.Context, r budgetRoute) (b KeyBudget, definitive bool, err error) {
	ctx, cancel := context.WithTimeout(ctx, budgetProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.url, nil)
	if err != nil {
		return KeyBudget{}, true, err
	}
	for k, vs := range c.Header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return KeyBudget{}, false, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return KeyBudget{}, false, err
	}
	if resp.StatusCode != http.StatusOK {
		return KeyBudget{}, true, fmt.Errorf("llm: key budget %s: status %d", r.url, resp.StatusCode)
	}
	switch r.source {
	case BudgetSourceLiteLLM:
		b, err = parseLiteLLMKeyInfo(raw)
	case BudgetSourceOpenRouter:
		b, err = parseOpenRouterKey(raw)
	default:
		err = ErrBudgetUnsupported
	}
	return b, true, err
}

// parseLiteLLMKeyInfo reads GET /key/info. A key whose budget comes from a
// budget table (budget_id) carries max_budget there instead of on its row.
func parseLiteLLMKeyInfo(raw []byte) (KeyBudget, error) {
	var v struct {
		Info *struct {
			Spend          *float64 `json:"spend"`
			MaxBudget      *float64 `json:"max_budget"`
			BudgetDuration *string  `json:"budget_duration"`
			BudgetResetAt  *string  `json:"budget_reset_at"`
			BudgetTable    *struct {
				MaxBudget      *float64 `json:"max_budget"`
				BudgetDuration *string  `json:"budget_duration"`
			} `json:"litellm_budget_table"`
		} `json:"info"`
	}
	if err := json.Unmarshal(raw, &v); err != nil || v.Info == nil || v.Info.Spend == nil {
		return KeyBudget{}, fmt.Errorf("llm: not a LiteLLM /key/info response")
	}
	in := v.Info
	b := KeyBudget{Spend: *in.Spend, Currency: CurrencyUSD, Source: BudgetSourceLiteLLM, Reported: true}
	max, dur := in.MaxBudget, in.BudgetDuration
	if t := in.BudgetTable; t != nil {
		if max == nil {
			max = t.MaxBudget
		}
		if dur == nil {
			dur = t.BudgetDuration
		}
	}
	if max != nil {
		b.setLimit(*max)
	}
	if dur != nil {
		b.Interval = *dur
	}
	if in.BudgetResetAt != nil {
		b.ResetAt, _ = parseTimestamp(*in.BudgetResetAt)
	}
	return b, nil
}

// parseOpenRouterKey reads GET /api/v1/key. effective_budget — the budget that
// blocks requests first, in USD — is preferred when present; otherwise the
// key's own credit limit. limit null means unlimited.
func parseOpenRouterKey(raw []byte) (KeyBudget, error) {
	var v struct {
		Data *struct {
			Usage          *float64 `json:"usage"`
			UsageDaily     *float64 `json:"usage_daily"`
			UsageWeekly    *float64 `json:"usage_weekly"`
			UsageMonthly   *float64 `json:"usage_monthly"`
			Limit          *float64 `json:"limit"`
			LimitRemaining *float64 `json:"limit_remaining"`
			LimitReset     *string  `json:"limit_reset"`
			Effective      *struct {
				LimitUSD      *float64 `json:"limit_usd"`
				SpendUSD      *float64 `json:"spend_usd"`
				RemainingUSD  *float64 `json:"remaining_usd"`
				ResetInterval *string  `json:"reset_interval"`
				ResetsAt      *string  `json:"resets_at"`
			} `json:"effective_budget"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &v); err != nil || v.Data == nil || v.Data.Usage == nil {
		return KeyBudget{}, fmt.Errorf("llm: not an OpenRouter /key response")
	}
	d := v.Data
	if e := d.Effective; e != nil && e.LimitUSD != nil {
		b := KeyBudget{Limit: *e.LimitUSD, HasLimit: true, Currency: CurrencyUSD,
			Source: BudgetSourceOpenRouter, Reported: true}
		if e.SpendUSD != nil {
			b.Spend = *e.SpendUSD
		}
		if e.RemainingUSD != nil {
			b.Remaining, b.HasRemaining = *e.RemainingUSD, true
		}
		if e.ResetInterval != nil {
			b.Interval = *e.ResetInterval
		}
		if e.ResetsAt != nil {
			b.ResetAt, _ = parseTimestamp(*e.ResetsAt)
		}
		return b, nil
	}
	b := KeyBudget{Spend: *d.Usage, Currency: CurrencyCredits, Source: BudgetSourceOpenRouter, Reported: true}
	if d.LimitReset != nil {
		b.Interval = *d.LimitReset
		// The spend that counts against a resetting limit is the period's.
		period := map[string]*float64{"daily": d.UsageDaily, "weekly": d.UsageWeekly, "monthly": d.UsageMonthly}
		if p := period[b.Interval]; p != nil {
			b.Spend = *p
		}
	}
	if d.Limit != nil {
		b.Limit, b.HasLimit = *d.Limit, true
	}
	if d.LimitRemaining != nil {
		b.Remaining, b.HasRemaining = *d.LimitRemaining, true
	}
	return b, nil
}

// parseTimestamp reads the ISO 8601 shapes the gateways write: RFC 3339, and
// Python's isoformat without a zone (taken as UTC).
func parseTimestamp(s string) (time.Time, bool) {
	for _, layout := range []string{time.RFC3339Nano, "2006-01-02T15:04:05.999999", "2006-01-02 15:04:05.999999-07:00", "2006-01-02 15:04:05.999999"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}
