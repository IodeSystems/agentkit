package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

const (
	anthropicCreditBody = `{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}`
	openaiQuotaBody     = `{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","code":"insufficient_quota"}}`
)

func fundsServer(t *testing.T, status int, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func wantSpent(t *testing.T, err error, status int) *BudgetSpentError {
	t.Helper()
	var spent *BudgetSpentError
	if !errors.As(err, &spent) {
		t.Fatalf("err = %v (%T); want *BudgetSpentError", err, err)
	}
	if spent.Status != status || spent.BP.Type != "upstream_funds_exhausted" || spent.BP.Message == "" {
		t.Errorf("spent = %+v", spent)
	}
	if TransientUpstream(err) {
		t.Errorf("TransientUpstream(%v) = true; a spent account is not the server going away", err)
	}
	return spent
}

// Anthropic sends a workspace spend cap as a 400 naming the regain date. It must
// be a BudgetSpentError whose AskedWait is the time to that date, from one request.
func TestFundsRefusalAnthropicWorkspaceLimit400(t *testing.T) {
	at := time.Now().UTC().Add(48 * time.Hour)
	body := fmt.Sprintf(`{"type":"error","error":{"type":"invalid_request_error","message":"You have reached your specified workspace API usage limits. You will regain access on %s at %s UTC."}}`,
		at.Format("2006-01-02"), at.Format("15:04"))
	srv, hits := fundsServer(t, 400, body)
	var evs []RetryEvent
	c := NewClient(srv.URL, "", "m")
	c.OnRetry = func(ev RetryEvent) { evs = append(evs, ev) }
	_, err := c.postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	spent := wantSpent(t, err, 400)
	if d := spent.BP.AskedWait - 48*time.Hour; d > time.Minute || d < -2*time.Minute {
		t.Errorf("AskedWait = %s; want about 48h", spent.BP.AskedWait)
	}
	if n := hits.Load(); n != 1 {
		t.Errorf("hits = %d; want 1", n)
	}
	if len(evs) != 1 || evs[0].Kind != RetryGiveUp || evs[0].Status != 400 {
		t.Errorf("events = %+v", evs)
	}
}

// A regain date already past must not become a negative or stale wait.
func TestFundsRefusalPastRegainDateLeavesAskedWaitZero(t *testing.T) {
	srv, _ := fundsServer(t, 400, `{"error":{"message":"You have reached your specified workspace API usage limits. You will regain access on 2020-01-01 at 00:00 UTC."}}`)
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	if spent := wantSpent(t, err, 400); spent.BP.AskedWait != 0 {
		t.Errorf("AskedWait = %s; want 0", spent.BP.AskedWait)
	}
}

func TestFundsRefusalAnthropicCreditBalance400(t *testing.T) {
	srv, hits := fundsServer(t, 400, anthropicCreditBody)
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	wantSpent(t, err, 400)
	if n := hits.Load(); n != 1 {
		t.Errorf("hits = %d; want 1", n)
	}
}

// OpenAI's 429 insufficient_quota used to be retried for the whole RetryBudget.
// No fastBackoff here: any sleep would show as a slow test.
func TestFundsRefusalOpenAI429InsufficientQuotaIsNotRetriedOrSleptOn(t *testing.T) {
	srv, hits := fundsServer(t, 429, openaiQuotaBody)
	t0 := time.Now()
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	wantSpent(t, err, 429)
	if n := hits.Load(); n != 1 {
		t.Errorf("hits = %d; want 1", n)
	}
	if d := time.Since(t0); d > 2*time.Second {
		t.Errorf("took %s; must not sleep", d)
	}
}

func TestFundsRefusal403WithMarker(t *testing.T) {
	srv, _ := fundsServer(t, 403, openaiQuotaBody)
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	wantSpent(t, err, 403)
}

// A 429 with no billing marker is still a rate limit: retried, then served.
func TestPlain429WithBodyStillRetries(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(429)
			_, _ = io.WriteString(w, `{"error":{"message":"Rate limit reached for requests","type":"requests"}}`)
			return
		}
		_, _ = io.WriteString(w, `ok`)
	}))
	defer srv.Close()
	resp, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if n := hits.Load(); n != 2 {
		t.Errorf("hits = %d; want 2", n)
	}
}

// A plain 400 keeps the old error, its whole body (the peek puts it back), and
// still teaches the capability record.
func TestPlain400KeepsOldErrorAndStillLearnsItsFact(t *testing.T) {
	srv := newCapServer(t, func(_ int, _ map[string]any) (int, string) { return 400, glmRefusal })
	c := openaiClient(srv.URL)
	var learned int
	c.OnLearn = func(Learned) { learned++ }
	_, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: "none"})
	var spent *BudgetSpentError
	if err == nil || errors.As(err, &spent) {
		t.Fatalf("err = %v; want the plain status error", err)
	}
	if want := "llm: status 400: " + glmRefusal; err.Error() != want {
		t.Errorf("err = %q; want %q", err.Error(), want)
	}
	if learned == 0 {
		t.Error("OnLearn did not fire; a plain refusal must still teach")
	}
}

// A billing 400 names no request field and must teach nothing, even when the
// request carried a field the record learns from.
func TestBillingRefusal400TeachesTheCapabilityRecordNothing(t *testing.T) {
	srv := newCapServer(t, func(_ int, _ map[string]any) (int, string) { return 400, anthropicCreditBody })
	c := openaiClient(srv.URL)
	var learned int
	c.OnLearn = func(Learned) { learned++ }
	before := len(c.Capabilities().Facts)
	_, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: "none"})
	wantSpent(t, err, 400)
	if learned != 0 || len(c.Capabilities().Facts) != before {
		t.Errorf("learned = %d, facts %d -> %d; want no change", learned, before, len(c.Capabilities().Facts))
	}
}

func TestFundsRefusalMessagesAPI400(t *testing.T) {
	srv, hits := fundsServer(t, 400, anthropicCreditBody)
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	_, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	wantSpent(t, err, 400)
	if n := hits.Load(); n != 1 {
		t.Errorf("hits = %d; want 1", n)
	}
}

// corrallm's 402 for Anthropic's workspace limit carries the regain time as
// error.retry_at and Retry-After, never retry_after. AskedWait stayed zero, so
// aw4 held Sonnet spent for the retry loop's clamped 300 s instead of until the
// reset, and spent a call on it every five minutes.
func TestSpent402TakesAskedWaitFromRetryAtNotTheClampedHeader(t *testing.T) {
	at := time.Now().Add(72 * time.Hour).UTC().Truncate(time.Second)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "259200")
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"error":{"message":"You have reached your specified workspace API usage limits. You will regain access on `+
			at.Format("2006-01-02")+` at 00:00 UTC.","type":"upstream_funds_exhausted","provider":"anthropic","upstream_status":400,"retry_at":"`+
			at.Format(time.RFC3339)+`"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	var spent *BudgetSpentError
	if !errors.As(err, &spent) {
		t.Fatalf("err = %v; want *BudgetSpentError", err)
	}
	if d := spent.BP.AskedWait - time.Until(at); d < -2*time.Second || d > 2*time.Second {
		t.Errorf("AskedWait = %s; want ~%s (retry_at)", spent.BP.AskedWait, time.Until(at))
	}
}

// The same 402 with only the provider's message (an older corrallm, or another
// proxy): the regain date inside the message is the wait.
func TestSpent402TakesAskedWaitFromRegainDateInMessage(t *testing.T) {
	day := time.Now().Add(96 * time.Hour).UTC()
	at := time.Date(day.Year(), day.Month(), day.Day(), 0, 0, 0, 0, time.UTC)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"error":{"message":"You have reached your specified workspace API usage limits. You will regain access on `+
			at.Format("2006-01-02")+` at 00:00 UTC.","type":"upstream_funds_exhausted"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	var spent *BudgetSpentError
	if !errors.As(err, &spent) {
		t.Fatalf("err = %v; want *BudgetSpentError", err)
	}
	if d := spent.BP.AskedWait - time.Until(at); d < -2*time.Second || d > 2*time.Second {
		t.Errorf("AskedWait = %s; want ~%s (regain date)", spent.BP.AskedWait, time.Until(at))
	}
}

// Only a Retry-After header: read it unclamped. The retry loop's 300 s ceiling
// is about sleeping, not about how long the budget is gone.
func TestSpent402TakesAskedWaitFromUnclampedRetryAfterHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "86400")
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = io.WriteString(w, `{"error":{"message":"spent","type":"upstream_funds_exhausted"}}`)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL, "", "m").postChatWithRetry(context.Background(), []byte(`{}`), callIdent{})
	var spent *BudgetSpentError
	if !errors.As(err, &spent) || spent.BP.AskedWait != 24*time.Hour {
		t.Fatalf("err = %v; want BudgetSpentError with AskedWait 24h", err)
	}
}
