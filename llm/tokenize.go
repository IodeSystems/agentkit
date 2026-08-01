package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

// Counting tokens exactly, when the endpoint can.
//
// A caller sizing a request against a token limit has to convert its text to
// tokens, and every way of doing that without the model's own tokenizer is a
// guess. Measured on one corpus against this endpoint's model, the same
// "characters per token" runs from 5.54 on prose to 1.11 on a surveyor's metes
// and bounds — `N88°14'32"E 147.03'` is nearly one token per character. A cap
// derived from a single ratio is therefore either five times too tight or
// silently over the limit, depending on which document arrives, and the second
// case is the one that loses documents.
//
// OpenAI has no tokenization endpoint — it ships tiktoken as a client library
// instead. llama.cpp's server does (`POST /tokenize`), and so does vLLM, with a
// different request and response shape. Neither is part of the OpenAI-compatible
// surface, so this probes for one and remembers what it found, including finding
// nothing: an endpoint without a tokenizer must cost one failed request, not one
// per call.

// tokenizeRoute is a discovered tokenizer: where it lives and which dialect it
// speaks.
type tokenizeRoute struct {
	url     string
	vllm    bool // vLLM wants {"model","prompt"} and answers {"count"}
	checked bool
	ok      bool
}

// CountTokens returns the exact number of tokens `text` becomes for this
// client's model, and ok=false when the endpoint offers no tokenizer.
//
// ok=false is a fact about the endpoint, not an error: the caller's job is to
// fall back to estimating, not to fail. A transport failure against a tokenizer
// that DID exist is also reported as ok=false — a count that could not be
// obtained is not a count, and blocking a document on it would trade a wrong
// size for no progress at all.
func (c *Client) CountTokens(ctx context.Context, text string) (int, bool) {
	r := c.tokenizeRoute(ctx)
	if !r.ok {
		return 0, false
	}
	n, err := c.countAt(ctx, r, text)
	if err != nil {
		return 0, false
	}
	return n, true
}

// HasTokenizer reports whether this endpoint can count tokens exactly, probing
// once. Useful for deciding a strategy before there is text to count.
func (c *Client) HasTokenizer(ctx context.Context) bool { return c.tokenizeRoute(ctx).ok }

var tokenizeRoutes sync.Map // baseURL+model -> *tokenizeRoute

func (c *Client) tokenizeRoute(ctx context.Context) *tokenizeRoute {
	key := c.baseURL + "\x00" + c.model
	if v, ok := tokenizeRoutes.Load(key); ok {
		return v.(*tokenizeRoute)
	}
	found := &tokenizeRoute{}
	for _, cand := range c.tokenizeCandidates() {
		if _, err := c.countAt(ctx, &cand, "probe"); err == nil {
			found = &tokenizeRoute{url: cand.url, vllm: cand.vllm, checked: true, ok: true}
			break
		}
	}
	found.checked = true
	tokenizeRoutes.Store(key, found)
	return found
}

// tokenizeCandidates are the places a tokenizer is known to live, in the order
// worth trying.
//
// The root sibling comes first because that is bare llama.cpp, where /tokenize
// sits beside /v1 rather than inside it. The /upstream form is corrallm's
// passthrough: it resolves the model, makes sure the backend is resident and
// forwards the rest of the path, which is how a llama.cpp route survives behind
// a multi-model proxy. The /v1 sibling is vLLM.
func (c *Client) tokenizeCandidates() []tokenizeRoute {
	root := strings.TrimSuffix(c.baseURL, "/v1")
	v1 := root + "/v1"
	return []tokenizeRoute{
		{url: root + "/tokenize"},
		{url: root + "/upstream/" + c.model + "/tokenize"},
		{url: v1 + "/tokenize", vllm: true},
	}
}

func (c *Client) countAt(ctx context.Context, r *tokenizeRoute, text string) (int, error) {
	var payload map[string]any
	if r.vllm {
		payload = map[string]any{"model": c.model, "prompt": text}
	} else {
		// add_special false: the caller is sizing a fragment of a larger prompt,
		// and BOS/EOS belong to the whole request, not to each piece. Counting
		// them per piece overstates by a couple of tokens per call, which is the
		// wrong direction only in that it hides where the real budget went.
		payload = map[string]any{"content": text, "add_special": false}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("llm: tokenize %s: status %d", r.url, resp.StatusCode)
	}
	// A single-page app answering 200 with HTML for every unknown path is the
	// normal shape of a proxy with a UI, and it is why a status check alone
	// cannot decide that a tokenizer exists. corrallm answers /tokenize and
	// /props with its own index.html; only parsing the body tells them apart.
	var out struct {
		Tokens []json.RawMessage `json:"tokens"`
		Count  *int              `json:"count"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, fmt.Errorf("llm: tokenize %s: not a tokenizer response", r.url)
	}
	if out.Count != nil {
		return *out.Count, nil
	}
	if out.Tokens == nil {
		return 0, fmt.Errorf("llm: tokenize %s: no token count in response", r.url)
	}
	return len(out.Tokens), nil
}

// ContextWindow reports the model's context size as the SERVER states it, and
// ok=false when it does not state one.
//
// Preferred over DiscoverContext wherever it answers: the probe costs O(log N)
// round trips and returns a lower bound, while this is one request for the
// number itself. llama.cpp reports it at /props as
// default_generation_settings.n_ctx.
func (c *Client) ContextWindow(ctx context.Context) (int, bool) {
	root := strings.TrimSuffix(c.baseURL, "/v1")
	for _, u := range []string{root + "/props", root + "/upstream/" + c.model + "/props"} {
		n, err := c.propsContext(ctx, u)
		if err == nil && n > 0 {
			return n, true
		}
	}
	return 0, false
}

func (c *Client) propsContext(ctx context.Context, url string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("llm: props %s: status %d", url, resp.StatusCode)
	}
	var out struct {
		NCtx     int `json:"n_ctx"`
		Defaults struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, fmt.Errorf("llm: props %s: not a props response", url)
	}
	if out.Defaults.NCtx > 0 {
		return out.Defaults.NCtx, nil
	}
	return out.NCtx, nil
}
