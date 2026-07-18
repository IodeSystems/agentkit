package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Context-window discovery.
//
// Not every OpenAI-compatible server advertises its model's context size, and
// the number that matters is the one the server will actually ACCEPT. The
// reliable way to learn it is empirical: send progressively larger prompts until
// the server rejects one as too long, then binary-search the boundary. This
// MEASURES the window by deliberately overflowing it — it is the opposite of
// compaction (which shrinks a prompt to fit); here we grow a throwaway prompt to
// find where "fit" ends.

// contextOverflowHints are substrings that mark a "prompt too long for the
// model's context" rejection across common servers (OpenAI, llama.cpp, vLLM,
// TGI). Matched case-insensitively against the error response body.
var contextOverflowHints = []string{
	"context length", "context window", "maximum context", "context_length_exceeded",
	"n_ctx", "too long", "exceeds", "longer than", "reduce the length",
	"maximum number of tokens", "maximum prompt",
}

// contextProbeCeiling bounds the probe: past this the requests get large and
// slow (a 60k-token prompt is ~25s of prompt processing), and a server that
// hasn't rejected by here won't give us a boundary. 32k is a safe lower bound
// for windowing — bigger real windows just mean we window a touch smaller.
const contextProbeCeiling = 32768

func bodyIsContextOverflow(body string) bool {
	s := strings.ToLower(body)
	for _, h := range contextOverflowHints {
		if strings.Contains(s, h) {
			return true
		}
	}
	return false
}

// DiscoverContext probes the configured model's usable context window: it sends
// prompts that grow exponentially until the server rejects one as too long, then
// binary-searches to ~256-token resolution. It returns an approximate maximum
// INPUT token count (~1 token per probe word) — a safe LOWER BOUND on the real
// window. Costs O(log N) round-trips; cache the result rather than calling it
// per request.
//
// IMPORTANT: this only finds a boundary on servers that REJECT an over-long
// prompt (llama.cpp direct, OpenAI, vLLM, TGI). Some proxies accept and process
// arbitrarily large prompts without error (corrallm was observed accepting 60k
// tokens with HTTP 200) — there is no boundary to find, so the probe stops at
// contextProbeCeiling and returns it as a lower bound. That keeps the probe
// cheap and bounded rather than climbing until requests get enormous.
func (c *Client) DiscoverContext(ctx context.Context) (int, error) {
	// probe reports whether ~nTokens of prompt is accepted (no overflow).
	probe := func(nTokens int) (fits bool, err error) {
		body, _ := json.Marshal(map[string]any{
			"model":      c.model,
			"messages":   []Message{{Role: "user", Content: strings.Repeat("word ", nTokens)}},
			"max_tokens": 1,
			"stream":     false,
		})
		resp, err := c.postWithRetry(ctx, c.chatURL(), body, "")
		if err != nil {
			return false, err
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
			return true, nil
		}
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		if bodyIsContextOverflow(string(b)) {
			return false, nil
		}
		return false, fmt.Errorf("llm: discover-context probe: status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	lo := 0
	hi := 0
	for n := 512; ; n *= 2 {
		fits, err := probe(n)
		if err != nil {
			return 0, err
		}
		if !fits {
			hi = n
			break
		}
		lo = n
		if n >= contextProbeCeiling {
			// No boundary by the ceiling → the server tolerates large prompts (or
			// the window really is this big). Return the ceiling as a safe lower
			// bound instead of climbing into multi-MB requests.
			return lo, nil
		}
	}
	for hi-lo > 256 {
		mid := (lo + hi) / 2
		fits, err := probe(mid)
		if err != nil {
			return 0, err
		}
		if fits {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo, nil
}
