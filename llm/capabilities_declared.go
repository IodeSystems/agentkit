// LoadDeclared: seed the capability record from the endpoint's own model list.
//
// Some endpoints say what a model does in GET /models. corrallm lists the
// window, the input kinds and the thinking modes; OpenRouter lists the window,
// the output cap and the request parameters a model takes. The facts are
// recorded with source declared: a claim by the endpoint, not something seen
// on the wire. Evidence (refused, observed, probed) replaces them, and a host
// pin is never touched (see Client.learn).
//
// Opt-in: it is an extra request, so nothing calls it. A host calls it once
// after NewClient, or after SetCapabilities if it wants stored evidence to win.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

// maxModelsBody bounds the /models read. OpenRouter lists hundreds of models;
// several MB is real, unbounded is not.
const maxModelsBody = 32 << 20

// samplingParams are the supported_parameters names recorded as
// sampling.<name>=accepted. Other names are either mapped on their own below or
// ignored.
var samplingParams = map[string]bool{
	"temperature": true, "top_p": true, "top_k": true, "min_p": true, "top_a": true,
	"frequency_penalty": true, "presence_penalty": true, "repetition_penalty": true,
	"seed": true, "logit_bias": true,
}

// modelsURL is the model-list endpoint: the same base chatURL uses.
func (c *Client) modelsURL() string {
	u := c.baseURL
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	return u + "/models"
}

// declaredEntry is the union of the two shapes. Unknown fields are ignored.
type declaredEntry struct {
	ID            string                     `json:"id"`
	ContextLength int                        `json:"context_length"`
	Modalities    map[string]json.RawMessage `json:"modalities"` // corrallm
	// corrallm
	ThinkingDefault string   `json:"thinking_default"`
	ThinkingModes   []string `json:"thinking_modes"`
	ThinkingEffort  string   `json:"thinking_effort"`
	// OpenRouter
	TopProvider *struct {
		MaxCompletionTokens int `json:"max_completion_tokens"`
	} `json:"top_provider"`
	SupportedParameters []string `json:"supported_parameters"`
	Architecture        *struct {
		InputModalities []string `json:"input_modalities"`
	} `json:"architecture"`
}

// LoadDeclared reads GET {base}/models once and records what the entry for this
// Client's model declares. The id must equal the model exactly: corrallm lists
// canonical ids and lanes, not aliases, so an alias finds nothing. No entry is
// not an error; nothing is learned.
//
// A Messages API client makes no request and returns nil. Anthropic's model
// list carries none of the fields read here (no window, no modalities, no
// parameter list), so the request would cost a round-trip and teach nothing.
//
// The URL adds /v1 to a base that lacks it, as chatURL does: corrallm answers
// a host-only {base}/models with its HTML page.
//
// It sends the same headers and auth as a chat call, once, without the retry
// loop: this is a hint, and a host that wants it again calls it again.
func (c *Client) LoadDeclared(ctx context.Context) error {
	if c.usesMessagesAPI() {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("llm: models request: %w", err)
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
		return fmt.Errorf("llm: models: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxModelsBody))
	if err != nil {
		return fmt.Errorf("llm: models: read: %w", err)
	}
	// 404/405: the endpoint has no model list. Not a failure of ours; learn
	// nothing. Auth failures and server errors are reported.
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return nil
	}
	if resp.StatusCode/100 != 2 {
		snippet := strings.TrimSpace(string(body))
		if len(snippet) > 200 {
			snippet = snippet[:200]
		}
		return fmt.Errorf("llm: models: HTTP %d: %s", resp.StatusCode, snippet)
	}
	var list struct {
		Data []declaredEntry `json:"data"`
	}
	// A 2xx that is not JSON (an HTML landing page from a gateway that does
	// not serve /models at this path) is "no declared metadata", not an error.
	if err := json.Unmarshal(body, &list); err != nil {
		return nil
	}
	for i := range list.Data {
		if list.Data[i].ID == c.model {
			c.learnDeclared(&list.Data[i])
			return nil
		}
	}
	return nil
}

func (c *Client) learnDeclared(e *declaredEntry) {
	decl := func(key, value, evidence string) {
		c.learn(key, Fact{Value: value, Source: SourceDeclared, Evidence: "GET /models: " + evidence})
	}
	itoa := strconv.Itoa

	if e.ContextLength > 0 {
		decl(CapLimitsContext, itoa(e.ContextLength), "context_length="+itoa(e.ContextLength))
	}

	// corrallm: modalities is an object keyed by kind. Keys, not values, say
	// what the model takes; the values carry formats we do not record.
	keys := make([]string, 0, len(e.Modalities))
	for k := range e.Modalities {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		decl(CapModalityPrefix+k, "true", "modalities."+k)
	}
	// modalities.text.maxTokens: the reply cap, when corrallm sets one.
	if raw, ok := e.Modalities["text"]; ok {
		var t struct {
			MaxTokens int `json:"maxTokens"`
		}
		if json.Unmarshal(raw, &t) == nil && t.MaxTokens > 0 {
			decl(CapLimitsOutput, itoa(t.MaxTokens), "modalities.text.maxTokens="+itoa(t.MaxTokens))
		}
	}

	// corrallm thinking modes. "off" listed means thinking can be turned off.
	// thinking_effort is a launch flag of the backend, not a ladder the caller
	// can pick from, so it goes in the evidence and nowhere else.
	// thinking_budget is corrallm's own cap, not a model limit; it is not read.
	if len(e.ThinkingModes) > 0 {
		off := false
		for _, m := range e.ThinkingModes {
			if m == "off" {
				off = true
			}
		}
		ev := "thinking_modes=[" + strings.Join(e.ThinkingModes, ",") + "]"
		if e.ThinkingDefault != "" {
			ev += " thinking_default=" + e.ThinkingDefault
		}
		if e.ThinkingEffort != "" {
			ev += " thinking_effort=" + e.ThinkingEffort + " (launch flag, not mapped)"
		}
		decl(CapReasoningThinkingOnly, strconv.FormatBool(!off), ev)
	}

	// OpenRouter.
	if e.TopProvider != nil && e.TopProvider.MaxCompletionTokens > 0 {
		n := e.TopProvider.MaxCompletionTokens
		decl(CapLimitsOutput, itoa(n), "top_provider.max_completion_tokens="+itoa(n))
	}
	if e.Architecture != nil {
		for _, m := range e.Architecture.InputModalities {
			if m = strings.TrimSpace(m); m != "" {
				decl(CapModalityPrefix+m, "true", "architecture.input_modalities has "+m)
			}
		}
	}
	for _, p := range e.SupportedParameters {
		ev := "supported_parameters has " + p
		switch {
		case p == "reasoning" || p == "include_reasoning":
			decl("reasoning.supported", "true", ev)
		case p == "structured_outputs":
			decl(CapDecodeJSONSchema, "true", ev)
		case p == "response_format":
			decl("decode.response_format", "true", ev)
		case samplingParams[p]:
			decl(CapSamplingPrefix+p, "accepted", ev)
		}
		// "tools" teaches nothing yet. "tool_choice" is listed support, not
		// proof a forced choice is honored, so tools.choice_forced stays unknown.
	}
}
