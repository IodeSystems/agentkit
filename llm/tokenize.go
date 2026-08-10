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
// anthropic marks a route that counts a RENDERED PROMPT rather than a string:
// system + tools + messages in, one number out, including the provider's own
// tool-use preamble. Measured against Anthropic 2026-08-10, that preamble is
// ~497 tokens the moment any tool is declared — which is why a structured
// counter cannot be reached through a string-shaped one.
type tokenizeRoute struct {
	url       string
	vllm      bool // vLLM wants {"model","prompt"} and answers {"count"}
	anthropic bool // counts a rendered prompt; answers {"input_tokens"}
	checked   bool
	ok        bool
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
	// A structured-only endpoint cannot answer "how many tokens is this string".
	// It can only price a whole rendered prompt, envelope included. Returning
	// that number here would be a count of something the caller did not ask
	// about, so this reports no raw-string tokenizer and points at CountPrompt.
	if r.anthropic {
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
			// Copy the CANDIDATE, don't rebuild it field by field. The rebuild
			// silently dropped the dialect flag when one was added, so a
			// discovered structured endpoint came back out of the cache looking
			// like a string tokenizer and answered the wrong question.
			hit := cand
			hit.checked, hit.ok = true, true
			found = &hit
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
		// Anthropic's own shape, reachable directly or through a proxy that
		// forwards the path (corrallm mounts it outside admission). Last because
		// it counts a different thing: a rendered prompt, not a string.
		{url: v1 + "/messages/count_tokens", anthropic: true},
	}
}

// probeMessages is the minimal `messages` an Anthropic count needs. Its cost is
// constant, which is the property CountPrompt relies on: every count carries the
// same envelope, so a DIFFERENCE between two counts cancels it exactly.
var probeMessages = []map[string]any{{"role": "user", "content": "."}}

func (c *Client) countAt(ctx context.Context, r *tokenizeRoute, text string) (int, error) {
	var payload map[string]any
	if r.anthropic {
		return c.countPromptAt(ctx, r, text, nil)
	}
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
		Tokens      []json.RawMessage `json:"tokens"`
		Count       *int              `json:"count"`
		InputTokens *int              `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, fmt.Errorf("llm: tokenize %s: not a tokenizer response", r.url)
	}
	if out.InputTokens != nil {
		return *out.InputTokens, nil
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

// CountPrompt returns the exact token count of a system prompt plus a set of
// tool definitions, as this endpoint's model would actually see them, and
// ok=false when the endpoint cannot count.
//
// This exists because the two kinds of endpoint count different things and the
// difference is not a detail. llama.cpp and vLLM tokenize a STRING: hand them
// text, get its tokens. Anthropic prices a RENDERED PROMPT — and the moment any
// tool is declared it adds its own tool-use preamble, measured at ~497 tokens
// on 2026-08-10 against a preamble-free baseline of 9:
//
//	no tools   9      1 tool   551      2 tools   596
//
// So on Anthropic the parts of a prompt are NOT independently countable and
// summable: counting each MCP server's schemas alone charges that preamble once
// per server. A caller wanting per-part costs has to take DIFFERENCES between
// whole-prompt counts, which is what the constant envelope in probeMessages
// makes exact.
//
// On a string tokenizer this falls back to counting the concatenation, so one
// caller works against both and only the residual differs.
func (c *Client) CountPrompt(ctx context.Context, system string, tools []ToolDef) (int, bool) {
	r := c.tokenizeRoute(ctx)
	if !r.ok {
		return 0, false
	}
	if r.anthropic {
		n, err := c.countPromptAt(ctx, r, system, tools)
		if err != nil {
			return 0, false
		}
		return n, true
	}
	var b strings.Builder
	b.WriteString(system)
	for _, t := range tools {
		raw, err := json.Marshal(t)
		if err != nil {
			// A schema that will not marshal cannot be sent either; skipping it
			// undercounts by one tool rather than failing the whole measurement.
			continue
		}
		b.Write(raw)
	}
	return c.CountTokens(ctx, b.String())
}

// countPromptAt posts Anthropic's count_tokens shape. Tools are sent only when
// present: an empty `tools` array still turns the provider's tool-use preamble
// on, so passing one unconditionally would put ~497 tokens into a measurement
// of a prompt that declares no tools.
func (c *Client) countPromptAt(ctx context.Context, r *tokenizeRoute, system string, tools []ToolDef) (int, error) {
	payload := map[string]any{"model": c.model, "messages": probeMessages}
	if system != "" {
		payload["system"] = system
	}
	if len(tools) > 0 {
		payload["tools"] = anthropicTools(tools)
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
		return 0, fmt.Errorf("llm: count_tokens %s: status %d", r.url, resp.StatusCode)
	}
	var out struct {
		InputTokens *int `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.InputTokens == nil {
		// Same trap as /tokenize: a proxy with a UI answers 200 and HTML for an
		// unknown path, so only the parsed body proves the route exists.
		return 0, fmt.Errorf("llm: count_tokens %s: not a count response", r.url)
	}
	return *out.InputTokens, nil
}

// anthropicTools converts OpenAI-shaped tool definitions to Anthropic's, which
// is what the count has to price — the two spell the same schema differently and
// counting the wrong spelling is a plausible wrong number.
func anthropicTools(tools []ToolDef) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"name":         t.Function.Name,
			"description":  t.Function.Description,
			"input_schema": t.Function.Parameters,
		})
	}
	return out
}
