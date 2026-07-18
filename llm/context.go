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
// INPUT token count (~1 token per probe word). Costs O(log N) round-trips —
// cache the result rather than calling it per request.
//
// It returns the largest size that was ACCEPTED (a lower bound on the true
// window), so callers can size inputs to it safely. A model with no discoverable
// ceiling (accepts up to the 1M probe cap) returns that cap.
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

	const ceiling = 1 << 20 // 1M-token ceiling: effectively unbounded
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
		if n >= ceiling {
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
