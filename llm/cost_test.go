package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// sseUsage is a one-chunk stream whose final event carries usage, plus extra
// usage fields (e.g. `,"cost":0.0123`).
func sseUsage(w http.ResponseWriter, usageExtra string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12"+usageExtra+"}}\n\n")
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func streamUsage(t *testing.T, c *Client) *Usage {
	t.Helper()
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var u *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			u = chunk.Usage
		}
	}
	if u == nil {
		t.Fatal("no usage chunk arrived")
	}
	return u
}

// LiteLLM's non-streaming shape: the cost is a header, in USD.
func TestCostFromLiteLLMHeaderNonStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-litellm-response-cost", "0.00042")
		w.Header().Set("x-litellm-key-spend", "12.5")
		w.Header().Set("x-litellm-key-max-budget", "50.0")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "k", "m")
	var got []Usage
	c.OnUsage = func(u Usage) { got = append(got, u) }
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("OnUsage called %d times, want 1", len(got))
	}
	u := got[0]
	if !u.Cost.Reported || u.Cost.Amount != 0.00042 || u.Cost.Currency != CurrencyUSD {
		t.Errorf("Cost = %+v", u.Cost)
	}
	if u.PromptTokens != 3 {
		t.Errorf("tokens lost: %+v", u)
	}
	b := u.KeyBudget
	if !b.Reported || b.Spend != 12.5 || !b.HasLimit || b.Limit != 50 || b.Remaining != 37.5 || b.Source != BudgetSourceLiteLLMHeaders {
		t.Errorf("KeyBudget = %+v", b)
	}
}

// OpenRouter: usage.cost in the final stream chunk, in credits. Headers arrive
// before the stream; the cost arrives at its end.
func TestCostFromOpenRouterBodyStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sseUsage(w, `,"cost":0.0123,"cost_details":{"upstream_inference_cost":null}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	u := streamUsage(t, c)
	if !u.Cost.Reported || u.Cost.Amount != 0.0123 || u.Cost.Currency != "" {
		t.Errorf("Cost = %+v, want 0.0123 with no currency", u.Cost)
	}

	// The same body from an OpenRouter base URL is in credits.
	m := (&Client{baseURL: "https://openrouter.ai/api/v1"}).metaFrom(http.Header{})
	var u2 Usage
	cost := 0.5
	m.stamp(&u2, &cost)
	if u2.Cost.Currency != CurrencyCredits {
		t.Errorf("openrouter currency = %q", u2.Cost.Currency)
	}
}

// LiteLLM on a stream: no cost header (it is not known when headers go out),
// usage.cost injected into the final chunk, and the x-litellm-* headers say the
// body's number is USD.
func TestCostFromLiteLLMStreamBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-litellm-version", "1.80.0")
		sseUsage(w, `,"cost":0.002`)
	}))
	defer srv.Close()
	var calls atomic.Int32
	c := NewClient(srv.URL, "", "m")
	c.OnUsage = func(Usage) { calls.Add(1) }
	u := streamUsage(t, c)
	if !u.Cost.Reported || u.Cost.Amount != 0.002 || u.Cost.Currency != CurrencyUSD {
		t.Errorf("Cost = %+v", u.Cost)
	}
	if calls.Load() != 1 {
		t.Errorf("OnUsage called %d times, want once per response", calls.Load())
	}
}

// A body cost wins over the header; where both exist they are one number.
func TestBodyCostWinsOverHeader(t *testing.T) {
	h := http.Header{}
	h.Set("x-litellm-response-cost", "1.0")
	m := (&Client{}).metaFrom(h)
	var u Usage
	body := 2.0
	m.stamp(&u, &body)
	if u.Cost.Amount != 2 || u.Cost.Currency != CurrencyUSD {
		t.Errorf("Cost = %+v", u.Cost)
	}
}

func TestPlainEndpointReportsNoCostOrLimits(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sseUsage(w, "")
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	u := streamUsage(t, c)
	if u.Cost.Reported || u.Limits.Reported || u.KeyBudget.Reported {
		t.Errorf("nothing was reported, got cost %+v limits %+v budget %+v", u.Cost, u.Limits, u.KeyBudget)
	}
	if len(u.Limits.Windows()) != 0 {
		t.Error("Windows() must be empty when nothing was reported")
	}
}

func TestRateLimitsOpenAIHeaders(t *testing.T) {
	now := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)
	h := http.Header{}
	h.Set("x-ratelimit-limit-requests", "60")
	h.Set("x-ratelimit-remaining-requests", "59")
	h.Set("x-ratelimit-reset-requests", "1s")
	h.Set("x-ratelimit-limit-tokens", "150000")
	h.Set("x-ratelimit-remaining-tokens", "149984")
	h.Set("x-ratelimit-reset-tokens", "6m0s")
	l := rateLimitsFrom(h, now)
	if !l.Reported {
		t.Fatal("not reported")
	}
	r := l.Requests
	if r.Limit != 60 || !r.HasRemaining || r.Remaining != 59 || !r.ResetAt.Equal(now.Add(time.Second)) {
		t.Errorf("Requests = %+v", r)
	}
	tk := l.Tokens
	if tk.Limit != 150000 || tk.Remaining != 149984 || !tk.ResetAt.Equal(now.Add(6*time.Minute)) {
		t.Errorf("Tokens = %+v", tk)
	}
	if f, ok := r.Fraction(); !ok || f < 0.98 || f > 0.99 {
		t.Errorf("Fraction = %v %v", f, ok)
	}
	if l.InputTokens.Reported || l.OutputTokens.Reported {
		t.Error("the OpenAI shape has no input/output split")
	}
}

func TestRateLimitsAnthropicHeaders(t *testing.T) {
	now := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)
	h := http.Header{}
	h.Set("anthropic-ratelimit-requests-limit", "4000")
	h.Set("anthropic-ratelimit-requests-remaining", "0")
	h.Set("anthropic-ratelimit-requests-reset", "2026-10-22T12:00:30Z")
	h.Set("anthropic-ratelimit-tokens-limit", "400000")
	h.Set("anthropic-ratelimit-tokens-remaining", "399000")
	h.Set("anthropic-ratelimit-input-tokens-limit", "200000")
	h.Set("anthropic-ratelimit-input-tokens-remaining", "199000")
	h.Set("anthropic-ratelimit-input-tokens-reset", "2026-10-22T12:01:00Z")
	h.Set("anthropic-ratelimit-output-tokens-limit", "80000")
	h.Set("anthropic-ratelimit-output-tokens-remaining", "80000")
	l := rateLimitsFrom(h, now)
	r := l.Requests
	// 0 remaining is reported, not absent.
	if r.Limit != 4000 || !r.HasRemaining || r.Remaining != 0 || !r.ResetAt.Equal(now.Add(30*time.Second)) {
		t.Errorf("Requests = %+v", r)
	}
	if l.Tokens.Limit != 400000 || l.InputTokens.Remaining != 199000 || l.OutputTokens.Limit != 80000 {
		t.Errorf("limits = %+v", l)
	}
	if !l.InputTokens.ResetAt.Equal(now.Add(time.Minute)) {
		t.Errorf("input reset = %v", l.InputTokens.ResetAt)
	}
	if n := len(l.Windows()); n != 4 {
		t.Errorf("Windows() = %d, want 4", n)
	}
}

// Behind LiteLLM: its own x-ratelimit-* (the key's limits) and the provider's
// forwarded under llm_provider-. The gateway's own wins per window; a window
// only the provider reports still comes through.
func TestRateLimitsLiteLLMHeaders(t *testing.T) {
	now := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)
	h := http.Header{}
	h.Set("x-ratelimit-limit-requests", "100")
	h.Set("x-ratelimit-remaining-requests", "7")
	h.Set("x-ratelimit-reset-requests", "42") // integer seconds
	h.Set("llm_provider-anthropic-ratelimit-requests-limit", "4000")
	h.Set("llm_provider-anthropic-ratelimit-requests-remaining", "3999")
	h.Set("llm_provider-anthropic-ratelimit-output-tokens-limit", "80000")
	h.Set("llm_provider-anthropic-ratelimit-output-tokens-remaining", "1000")
	l := rateLimitsFrom(h, now)
	if l.Requests.Limit != 100 || l.Requests.Remaining != 7 || !l.Requests.ResetAt.Equal(now.Add(42*time.Second)) {
		t.Errorf("Requests = %+v, want the gateway's own window", l.Requests)
	}
	if l.OutputTokens.Limit != 80000 || l.OutputTokens.Remaining != 1000 {
		t.Errorf("OutputTokens = %+v, want the forwarded provider window", l.OutputTokens)
	}
	if l.Tokens.Reported {
		t.Errorf("Tokens = %+v, nothing reported it", l.Tokens)
	}
}

func TestRateLimitsMalformedIsNotReported(t *testing.T) {
	h := http.Header{}
	h.Set("x-ratelimit-remaining-requests", "lots")
	h.Set("x-ratelimit-reset-requests", "soon")
	if l := rateLimitsFrom(h, time.Now()); l.Reported {
		t.Errorf("malformed headers reported as %+v", l)
	}
}

// Headers arrive before the stream: limits read from them must reach the usage
// at its end, and OnUsage fires before the channel closes.
func TestLimitsReachUsageOverStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-ratelimit-limit-tokens", "1000")
		w.Header().Set("x-ratelimit-remaining-tokens", "50")
		w.Header().Set("x-ratelimit-reset-tokens", "20ms")
		sseUsage(w, "")
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	var mu sync.Mutex
	var seen *Usage
	c.OnUsage = func(u Usage) { mu.Lock(); seen = &u; mu.Unlock() }
	u := streamUsage(t, c)
	if f, ok := u.Limits.Tokens.Fraction(); !ok || f != 0.05 {
		t.Errorf("Tokens fraction = %v %v", f, ok)
	}
	mu.Lock()
	defer mu.Unlock()
	if seen == nil || seen.Limits.Tokens.Remaining != 50 || seen.TotalTokens != 12 {
		t.Errorf("OnUsage saw %+v before the channel closed", seen)
	}
}

// A stream with no usage chunk still tells OnUsage what its headers said.
func TestOnUsageHeaderOnlyStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-ratelimit-remaining-requests", "3")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "m")
	var got []Usage
	c.OnUsage = func(u Usage) { got = append(got, u) }
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	if len(got) != 1 || got[0].Limits.Requests.Remaining != 3 || got[0].TotalTokens != 0 {
		t.Errorf("got %+v", got)
	}
}

func resetBudgetRoutes() { budgetRoutes = sync.Map{} }

// The /key/info shape from LiteLLM's key_management_endpoints.info_key_fn:
// {"key": ..., "info": <key row minus token>}.
func TestKeyBudgetLiteLLM(t *testing.T) {
	resetBudgetRoutes()
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/key/info" {
			http.NotFound(w, r)
			return
		}
		auth = r.Header.Get("Authorization")
		fmt.Fprint(w, `{"key":"sk-test","info":{"key_alias":"me","spend":12.25,"max_budget":50.0,
			"budget_duration":"30d","budget_reset_at":"2026-11-01T00:00:00+00:00","tpm_limit":null,"rpm_limit":100}}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "sk-test", "m")
	b, err := c.KeyBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer sk-test" {
		t.Errorf("auth = %q, want the client's own key", auth)
	}
	want := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
	if b.Source != BudgetSourceLiteLLM || b.Spend != 12.25 || !b.HasLimit || b.Limit != 50 ||
		!b.HasRemaining || b.Remaining != 37.75 || b.Interval != "30d" || !b.ResetAt.Equal(want) || b.Currency != CurrencyUSD {
		t.Errorf("budget = %+v", b)
	}
	if f, ok := b.Fraction(); !ok || f != 37.75/50 {
		t.Errorf("Fraction = %v %v", f, ok)
	}
}

// A LiteLLM key with no max_budget: spend reported, limit not.
func TestKeyBudgetLiteLLMUnlimited(t *testing.T) {
	resetBudgetRoutes()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"key":"x","info":{"spend":1.5,"max_budget":null,"budget_reset_at":null}}`)
	}))
	defer srv.Close()
	b, err := NewClient(srv.URL, "k", "m").KeyBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if b.Spend != 1.5 || b.HasLimit || b.HasRemaining || !b.ResetAt.IsZero() {
		t.Errorf("budget = %+v", b)
	}
	if _, ok := b.Fraction(); ok {
		t.Error("no limit, no fraction")
	}
}

// OpenRouter's GET /api/v1/key, key-level fields (credits).
func TestKeyBudgetOpenRouter(t *testing.T) {
	resetBudgetRoutes()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/key" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":{"label":"sk-or-v1-abc...","limit":20,"limit_remaining":15.5,"limit_reset":"monthly",
			"usage":100,"usage_daily":1,"usage_weekly":2,"usage_monthly":4.5,"is_free_tier":false,"effective_budget":null}}`)
	}))
	defer srv.Close()
	b, err := NewClient(srv.URL+"/api/v1", "k", "m").KeyBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if b.Source != BudgetSourceOpenRouter || b.Spend != 4.5 || b.Limit != 20 || b.Remaining != 15.5 ||
		b.Interval != "monthly" || b.Currency != CurrencyCredits {
		t.Errorf("budget = %+v", b)
	}
}

// OpenRouter with an effective_budget: the USD budget that binds first wins.
func TestKeyBudgetOpenRouterEffectiveBudget(t *testing.T) {
	resetBudgetRoutes()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/key" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":{"limit":null,"limit_remaining":null,"limit_reset":null,"usage":100,
			"effective_budget":{"scope":"api_key","limit_usd":10,"spend_usd":9.5,"remaining_usd":0.5,
			"reset_interval":"daily","resets_at":"2026-10-23T00:00:00Z","include_byok":false}}}`)
	}))
	defer srv.Close()
	b, err := NewClient(srv.URL+"/api/v1", "k", "m").KeyBudget(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if b.Currency != CurrencyUSD || b.Limit != 10 || b.Spend != 9.5 || b.Remaining != 0.5 ||
		b.Interval != "daily" || !b.ResetAt.Equal(time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)) {
		t.Errorf("budget = %+v", b)
	}
}

// A plain OpenAI-compatible server: typed "not supported", and the answer is
// remembered — the next call makes no request.
func TestKeyBudgetUnsupported(t *testing.T) {
	resetBudgetRoutes()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.NotFound(w, r)
	}))
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "k", "m")
	if _, err := c.KeyBudget(context.Background()); !errors.Is(err, ErrBudgetUnsupported) {
		t.Fatalf("err = %v, want ErrBudgetUnsupported", err)
	}
	first := hits.Load()
	if first != 2 {
		t.Errorf("probed %d endpoints, want 2", first)
	}
	if _, err := c.KeyBudget(context.Background()); !errors.Is(err, ErrBudgetUnsupported) {
		t.Fatalf("second err = %v", err)
	}
	if hits.Load() != first {
		t.Error("an unsupported endpoint was probed again")
	}
}

// A 200 that is not a budget (a catch-all route) is not mistaken for one.
func TestKeyBudgetIgnoresUnrelated200(t *testing.T) {
	resetBudgetRoutes()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer srv.Close()
	if _, err := NewClient(srv.URL, "k", "m").KeyBudget(context.Background()); !errors.Is(err, ErrBudgetUnsupported) {
		t.Fatalf("err = %v", err)
	}
}

// A gateway that is down is not remembered as unsupported.
func TestKeyBudgetTransportErrorNotCached(t *testing.T) {
	resetBudgetRoutes()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()
	c := NewClient(url, "k", "m")
	if _, err := c.KeyBudget(context.Background()); err == nil || errors.Is(err, ErrBudgetUnsupported) {
		t.Fatalf("err = %v, want a transport error", err)
	}
	if _, ok := budgetRoutes.Load(url + "\x00k"); ok {
		t.Error("a transport failure was cached")
	}
}
