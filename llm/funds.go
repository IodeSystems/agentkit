package llm

// Funds refusals. corrallm turns a remote provider's billing refusal into a 402
// (see BudgetSpentError). Talking to the provider DIRECTLY, the same refusal
// arrives as a 400, 403 or 429, so a host's fallback keyed on BudgetSpentError
// never fired, and OpenAI's 429 insufficient_quota was retried for the whole
// RetryBudget. fundsBody recognises those by marker text; postTypedWithRetry
// checks it before any retry, capability-learning or InputTooLarge logic so a
// billing 400 is never read as a bad parameter or an oversized prompt.
//
// Mirror of corrallm internal/proxy/fundsrefusal.go: keep the markers and the
// regain-date pattern in step.

import (
	"bytes"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// typeUpstreamFunds is Backpressure.Type of a provider billing refusal, as
// corrallm names it. Set over the provider's own type ("invalid_request_error"
// says nothing about money).
const typeUpstreamFunds = "upstream_funds_exhausted"

// fundsPeekLimit caps how much of an error body is read to classify it.
const fundsPeekLimit = 8 << 10

// fundsMarkers are lower-case phrases marking a billing refusal. A plain rate
// limit or a bad parameter carries none of them.
var fundsMarkers = []string{
	"insufficient_quota",
	"billing_hard_limit_reached",
	"credit balance is too low",
	"exceeded your current quota",
	"workspace api usage limits",
}

// regainRE finds Anthropic's "regain access on YYYY-MM-DD at HH:MM UTC".
var regainRE = regexp.MustCompile(`(?i)regain access on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}) UTC`)

// fundsBody reports whether body names a billing refusal.
func fundsBody(body []byte) bool {
	low := strings.ToLower(string(body))
	for _, m := range fundsMarkers {
		if strings.Contains(low, m) {
			return true
		}
	}
	return false
}

// regainAt is the time the body says access returns, zero when it names none.
func regainAt(text string) time.Time {
	m := regainRE.FindStringSubmatch(text)
	if m == nil {
		return time.Time{}
	}
	t, err := time.Parse("2006-01-02 15:04", m[1]+" "+m[2])
	if err != nil {
		return time.Time{}
	}
	return t.UTC()
}

// peekFundsBody reads the head of a 400, 403 or 429 body and reports whether it
// is a billing refusal. A body that is not one is put back in front of the
// rest, so the caller reads it whole.
func peekFundsBody(resp *http.Response) ([]byte, bool) {
	switch resp.StatusCode {
	case http.StatusBadRequest, http.StatusForbidden, http.StatusTooManyRequests:
	default:
		return nil, false
	}
	if resp.Body == nil {
		return nil, false
	}
	head, _ := io.ReadAll(io.LimitReader(resp.Body, fundsPeekLimit))
	if fundsBody(head) {
		return head, true
	}
	resp.Body = struct {
		io.Reader
		io.Closer
	}{io.MultiReader(bytes.NewReader(head), resp.Body), resp.Body}
	return head, false
}

// fundsBackpressure fills what a provider's own body lacks: the type, the
// message, and AskedWait from a named regain date still in the future.
func fundsBackpressure(bp Backpressure, body []byte) Backpressure {
	bp.Type = typeUpstreamFunds
	if bp.Message == "" {
		bp.Message = strings.TrimSpace(firstLineOfBody(string(body)))
	}
	if at := regainAt(string(body)); !at.IsZero() {
		if d := time.Until(at); d > 0 {
			bp.AskedWait = d
		}
	}
	return bp
}
