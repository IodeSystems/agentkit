// Package llm is the OpenAI-compatible streaming chat client every
// agent role's Turn loop uses to reach the LLM. It supports streaming
// + non-streaming completions, tool calling via ToolDef / ToolCall,
// and per-call options through ChatOpts (notably ToolChoice for
// forcing a typed terminal tool). StreamChunk carries one token at a
// time plus the final Usage report; StreamChunkToSSE formats those
// for SSE relay to UI consumers.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Message represents a chat message in the OpenAI-compatible format.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
	// ToolCalls carries an assistant message's requested tool calls, so a
	// reconstructed conversation replays a valid assistant(tool_calls) →
	// tool(tool_call_id) structure instead of orphan tool messages.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a role="tool" result back to the assistant tool call
	// that produced it (OpenAI requires this correlation).
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool call request from the LLM.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ToolDef describes a tool the LLM can call, matching the OpenAI tool format.
type ToolDef struct {
	Type     string `json:"type"`
	Function struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  any    `json:"parameters"`
	} `json:"function"`
}

// StreamChunk is one token from the streaming response. The final
// chunk carries Usage (when the provider supports include_usage); all
// other fields will be zero on that chunk.
type StreamChunk struct {
	Content  string
	ToolCall *ToolCall
	Done     bool
	Error    string
	Usage    *Usage
}

// Usage carries provider-reported token counts for one chat
// completion. Fields beyond prompt/completion/total are optional and
// only filled when the provider returns them (Anthropic's cache
// fields, OpenAI's reasoning tokens, etc).
type Usage struct {
	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// Client sends requests to an OpenAI-compatible LLM endpoint.
type Client struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client

	// RetryBudget caps the total wall-clock a single request spends retrying
	// 429/5xx before giving up (exponential backoff to retryMaxBackoff,
	// honoring retry_after, but bounded). 0 → defaultRetryBudget (5m). The
	// caller's ctx deadline still wins if shorter. Set it per Client for a
	// busy endpoint.
	RetryBudget time.Duration
}

func NewClient(baseURL, apiKey, model string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		http:    &http.Client{},
	}
}

// retryBudget returns the effective retry budget (the field, or the default).
func (c *Client) retryBudget() time.Duration {
	if c.RetryBudget > 0 {
		return c.RetryBudget
	}
	return defaultRetryBudget
}

// chatURL returns the chat-completions endpoint. Accepts a baseURL with
// or without a trailing "/v1" so both http://host:11434/v1 and
// https://host (the OpenAI-compat convention) work.
func (c *Client) chatURL() string {
	u := c.baseURL
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	return u + "/chat/completions"
}

// 429 backoff schedule. Exp from initial → max, then HOLDS at max until
// the request succeeds, the ctx is canceled, or the RETRY BUDGET is
// exhausted (Client.RetryBudget, default defaultRetryBudget). On a
// contended box every wait is another shot at a slot — so we keep trying
// rather than giving up after a fixed attempt count — but a SUPER-busy box
// shouldn't be pounded forever, so the budget is the wall-clock ceiling.
//
// First few attempts climb 1s → 2s → 4s → 8s → 16s → 30s → 30s → …
// so a transient burst clears fast while a sustained limit doesn't
// flood the provider.
//
// var (not const) so tests can swap to millisecond timings via
// retry_test.go::retryInitialBackoffSet; production never mutates
// these values.
var (
	retryInitialBackoff = 1 * time.Second
	retryMaxBackoff     = 30 * time.Second
)

// defaultRetryBudget bounds the TOTAL wall-clock a single postChatWithRetry
// spends retrying (429 + 5xx) before it gives up with a clear error, when the
// caller hasn't set Client.RetryBudget. 5m matches a "keep trying through a
// busy spell, but don't hang a caller forever" policy; the caller's ctx
// deadline still wins if it's shorter. var so tests can shrink it.
var defaultRetryBudget = 5 * time.Minute

// retryLogEvery throttles the "still retrying" log line. The first
// few retries log every time; once we're holding at retryMaxBackoff
// every line is the same and one-per-call is noise, so we log every
// Nth attempt instead. Set to 1 for verbose debugging.
const retryLogEvery = 10

// retryAfterCeiling caps the Retry-After value the server is allowed
// to ask us to wait. The fair-share proxy is the authority on
// timing — we honor what it sends exactly, not the 30s schedule —
// but a misbehaving / misconfigured server saying "wait 1 hour"
// shouldn't lock the daemon out indefinitely. ctx is still the
// authoritative stop; this is just a sanity guard.
//
// var so retry_test.go can swap to ms scales alongside the backoff
// vars; production never mutates.
var retryAfterCeiling = 5 * time.Minute

// retry5xxMaxAttempts caps how many times we retry on a 5xx response.
// 429 has its own path (retry until a slot frees, bounded only by the
// RetryBudget + ctx — Retry-After is the proxy's authoritative signal).
// 5xx is different — it usually means
// "upstream broke" or "transient gateway error"; retrying a few
// times catches the proxy-blip / cold-start / momentary upstream
// hiccup, but persistent 5xx should fail fast so the operator hears
// about a real outage instead of waiting forever.
//
// Counts ATTEMPTS (so 5 means: first try + up to 4 retries). Set
// via var so tests can override; production never mutates.
var retry5xxMaxAttempts = 5

// postChatWithRetry issues a chat-completions POST and transparently
// retries on HTTP 429 with exponential backoff capped at 30s.
// Returns the live response (caller closes its body) on success or a
// non-retryable failure. The retry delay is taken from the Retry-After
// header OR a corrallm-style JSON backpressure body (retryAfterFrom),
// clamped to retryAfterCeiling; absent both, the next scheduled backoff
// wins.
//
// payload is the marshaled body. We rebuild the request every attempt
// because http.Request.Body is single-use; bytes.NewReader is cheap.
//
// Why bake this into the client (not the caller): every consumer of
// ChatStream/Chat hits 429 the same way (provider rate limit), and
// every consumer wants the same recovery (wait and retry, give up on
// ctx cancel, RetryBudget exhaustion, or repeated 5xx). Pushing this
// into the harness would duplicate the loop in every role.
func (c *Client) postChatWithRetry(ctx context.Context, payload []byte, traceID string) (*http.Response, error) {
	backoff := retryInitialBackoff
	fiveXXAttempts := 0
	budget := c.retryBudget()
	start := time.Now()
	deadline := start.Add(budget)
	for attempt := 0; ; attempt++ {
		// Cheap ctx check before each attempt so a cancellation that
		// arrived between the last sleep and now short-circuits
		// before the next round-trip.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatURL(), bytes.NewReader(payload))
		if err != nil {
			return nil, fmt.Errorf("llm: request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		// X-Trace-Id (thread id, optional) lets the server logs attribute
		// requests back to an autowork3 thread for debugging. Skipped when
		// empty so an ad-hoc probe doesn't emit a header with no value.
		if traceID != "" {
			req.Header.Set("X-Trace-Id", traceID)
		}
		// The API key IS autowork3's scheduling identity to the llama-swap
		// fork: it maps the key to a priority. Configure the key on the
		// provider (api_key_env / secrets store); empty = no auth, default
		// priority.
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}
		resp, err := c.http.Do(req)
		if err != nil {
			return nil, fmt.Errorf("llm: do: %w", err)
		}

		status := resp.StatusCode
		switch {
		case status == http.StatusTooManyRequests:
			// 429 path: drain + close so the connection can be
			// reused for the retry. Retry-After is the fair-share
			// proxy's authoritative signal — honor it exactly
			// (clamped to retryAfterCeiling for safety against a
			// misbehaving server). When absent (other providers,
			// transient overload at the model), fall back to the
			// exp-backoff schedule. Counter NOT incremented for the
			// 5xx cap — 429 is "wait your turn", not "broken".
			sleep := backoff
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if d, ok := retryAfterFrom(resp.Header, bodyBytes); ok {
				sleep = d
			}

			if time.Now().Add(sleep).After(deadline) {
				return nil, fmt.Errorf("llm: retry budget %s exhausted after %s of 429 backpressure",
					budget, time.Since(start).Round(time.Second))
			}
			if attempt < retryLogEvery || attempt%retryLogEvery == 0 {
				log.Printf("llm: provider returned 429 (attempt %d); retrying in %s",
					attempt+1, sleep)
			}
			if !sleepOrCancel(ctx, sleep) {
				return nil, ctx.Err()
			}
		case status >= 500:
			// 5xx path: transient upstream / proxy error. Retry a
			// bounded number of times so the trial doesn't fail
			// on a single bad-gateway blip, but don't loop forever
			// on a real outage — operator hears about it after the
			// cap. Same exp-backoff schedule as 429; Retry-After
			// also honored if the server sends it.
			fiveXXAttempts++
			sleep := backoff
			bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if d, ok := retryAfterFrom(resp.Header, bodyBytes); ok {
				sleep = d
			}

			if fiveXXAttempts >= retry5xxMaxAttempts {
				return nil, fmt.Errorf("llm: status %d (after %d retries)", status, fiveXXAttempts-1)
			}
			if time.Now().Add(sleep).After(deadline) {
				return nil, fmt.Errorf("llm: retry budget %s exhausted after %s (last status %d)",
					budget, time.Since(start).Round(time.Second), status)
			}
			log.Printf("llm: upstream returned %d (5xx attempt %d/%d); retrying in %s",
				status, fiveXXAttempts, retry5xxMaxAttempts, sleep)
			if !sleepOrCancel(ctx, sleep) {
				return nil, ctx.Err()
			}
		default:
			// Anything else (2xx success, 4xx non-429): hand back.
			return resp, nil
		}
		backoff *= 2
		if backoff > retryMaxBackoff {
			backoff = retryMaxBackoff
		}
	}
}

// retryAfterFrom extracts a retry delay from a throttled/failed response.
// The Retry-After HEADER wins (HTTP standard, integer seconds). When absent,
// corrallm-style backpressure BODIES carry the hint as JSON — either a
// top-level "retry_after" or one nested under "error" (the shape corrallm's
// fair-share proxy returns on 429: {"error":{"reason":"queue-timeout",
// "retry_after":10,...}}). Returns ok=false when neither is present, so the
// caller keeps its exponential-backoff schedule. Always clamped to
// retryAfterCeiling so a misbehaving server can't park the daemon.
func retryAfterFrom(h http.Header, body []byte) (time.Duration, bool) {
	if ra := strings.TrimSpace(h.Get("Retry-After")); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
			return clampRetry(secs), true
		}
	}
	if len(body) > 0 {
		var parsed struct {
			RetryAfter int `json:"retry_after"`
			Error      struct {
				RetryAfter int `json:"retry_after"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &parsed) == nil {
			secs := parsed.RetryAfter
			if secs == 0 {
				secs = parsed.Error.RetryAfter
			}
			if secs > 0 {
				return clampRetry(secs), true
			}
		}
	}
	return 0, false
}

func clampRetry(secs int) time.Duration {
	d := time.Duration(secs) * time.Second
	if d > retryAfterCeiling {
		d = retryAfterCeiling
	}
	return d
}

// sleepOrCancel waits sleep duration unless ctx fires first. Returns
// true if the sleep completed, false if ctx canceled — caller returns
// ctx.Err() on the false branch.
func sleepOrCancel(ctx context.Context, sleep time.Duration) bool {
	timer := time.NewTimer(sleep)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// ChatOpts carries per-call switches.
//
// ToolChoice: "", "auto", "required", or a JSON-encoded
//
//	{"type":"function","function":{"name":"foo"}}.
//	OpenAI-compatible providers honor "required" or a specific
//	tool spec.
//
// TraceID: the value sent in the X-Trace-Id header, for correlation
//
//	only — server-side logs can attribute requests back to a thread.
//	Scheduling priority is keyed off the API key (Authorization
//	Bearer), not this header. Harness Sessions set TraceID = thread
//	id; empty is fine (the header just won't be sent).
//
// Grammar: when non-empty, forwarded as the request body's "grammar"
//
//	field — a GBNF grammar the server constrains token sampling to
//	(llama.cpp / corrallm). Raw passthrough; the server owns the
//	syntax. Use for hard structural guarantees the model cannot
//	violate (vs the client-side agent.Validator fix loop, which
//	corrects a bad reply after the fact).
//
// ResponseFormat: when non-nil, forwarded as the request body's
//
//	"response_format" — e.g. map[string]any{"type":"json_object"} or a
//	{"type":"json_schema","json_schema":{...}} object. Marshaled
//	as-is; the server decides support.
//
// Nil opts behaves as the default.
type ChatOpts struct {
	ToolChoice     string
	TraceID        string
	Grammar        string
	ResponseFormat any
}

// ChatStream sends a chat completion request with streaming enabled.
// It returns a channel that emits StreamChunks as they arrive.
func (c *Client) ChatStream(ctx context.Context, messages []Message, tools []ToolDef, opts *ChatOpts) (<-chan StreamChunk, error) {
	body := map[string]any{
		"model":    c.model,
		"messages": messages,
		"stream":   true,
		// Ask the provider for token usage on the final stream chunk.
		// Honored by OpenAI; Anthropic-via-compat provides usage by
		// default; other providers silently ignore.
		"stream_options": map[string]any{"include_usage": true},
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if opts != nil && opts.ToolChoice != "" {
		// Forward raw — "required" / "auto" / "none" pass through as
		// strings; an object-shaped choice is JSON-decoded first so the
		// body marshals it correctly.
		if strings.HasPrefix(opts.ToolChoice, "{") {
			var obj any
			if err := json.Unmarshal([]byte(opts.ToolChoice), &obj); err == nil {
				body["tool_choice"] = obj
			}
		} else {
			body["tool_choice"] = opts.ToolChoice
		}
	}
	// Constrained-decoding passthroughs. grammar (GBNF) and
	// response_format are server-side sampling constraints — llama.cpp /
	// corrallm honor them; providers that don't simply ignore the fields.
	if opts != nil && opts.Grammar != "" {
		body["grammar"] = opts.Grammar
	}
	if opts != nil && opts.ResponseFormat != nil {
		body["response_format"] = opts.ResponseFormat
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal: %w", err)
	}

	traceID := ""
	if opts != nil {
		traceID = opts.TraceID
	}
	resp, err := c.postChatWithRetry(ctx, payload, traceID)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, statusError(resp)
	}

	ch := make(chan StreamChunk, 64)
	go c.readStream(ctx, resp.Body, ch)
	return ch, nil
}

// statusError formats a non-2xx response into an error that INCLUDES a snippet
// of the response body — the provider's error message (e.g. a 400 explaining
// which tool schema it rejected) is the single most useful thing for debugging
// an integration, and dropping it turns every failure into an opaque "status
// 400". Closes the body.
func statusError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
	resp.Body.Close()
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		return fmt.Errorf("llm: status %d", resp.StatusCode)
	}
	return fmt.Errorf("llm: status %d: %s", resp.StatusCode, msg)
}

// Chat sends a non-streaming chat completion request.
func (c *Client) Chat(ctx context.Context, messages []Message, tools []ToolDef) (string, []ToolCall, error) {
	body := map[string]any{
		"model":    c.model,
		"messages": messages,
		"stream":   false,
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", nil, fmt.Errorf("llm: marshal: %w", err)
	}

	// Chat is the non-streaming path; today no caller passes a trace
	// id here, so X-Trace-Id stays unset.
	resp, err := c.postChatWithRetry(ctx, payload, "")
	if err != nil {
		return "", nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return "", nil, statusError(resp)
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Message struct {
				Content   string     `json:"content"`
				ToolCalls []ToolCall `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", nil, fmt.Errorf("llm: decode: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", nil, nil
	}
	return result.Choices[0].Message.Content, result.Choices[0].Message.ToolCalls, nil
}

// readStream parses SSE events from the response body.
//
// Content deltas are forwarded immediately (per token, for streaming
// to the UI). Tool-call deltas are *accumulated* internally — the
// OpenAI streaming protocol fragments each call's arguments across
// many chunks keyed by `index`. Emitting a chunk per fragment would
// dispatch one fake tool call per character of arguments. Tool calls
// are flushed as complete StreamChunks at `finish_reason` time.
//
// Usage lands on a final chunk (often a no-choices one when
// `stream_options.include_usage=true`); forward as soon as seen.
func (c *Client) readStream(ctx context.Context, body io.ReadCloser, ch chan<- StreamChunk) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 65536), 65536)

	// Tool-call accumulator. Keyed by index from the streaming protocol.
	// Order is maintained via toolOrder (deltas might arrive out of
	// numeric order, but typically index 0 streams first).
	toolBuf := map[int]*ToolCall{}
	var toolOrder []int

	flushTools := func() {
		for _, idx := range toolOrder {
			tc := toolBuf[idx]
			if tc == nil {
				continue
			}
			// Default the type — OpenAI omits "function" on later
			// deltas but the dispatcher expects it set.
			if tc.Type == "" {
				tc.Type = "function"
			}
			ch <- StreamChunk{ToolCall: tc}
		}
		toolBuf = map[int]*ToolCall{}
		toolOrder = nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			flushTools()
			ch <- StreamChunk{Done: true}
			return
		}

		// Streaming-delta shape. Note `Index` on each tool_call entry —
		// OpenAI uses it to thread fragments of the same call together.
		var event struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id,omitempty"`
						Type     string `json:"type,omitempty"`
						Function struct {
							Name      string `json:"name,omitempty"`
							Arguments string `json:"arguments,omitempty"`
						} `json:"function"`
					} `json:"tool_calls,omitempty"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *Usage `json:"usage"`
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if event.Error.Message != "" {
			ch <- StreamChunk{Error: event.Error.Message}
			return
		}
		// Usage-only chunk (no choices) — providers using
		// stream_options.include_usage send one of these after content.
		if len(event.Choices) == 0 {
			if event.Usage != nil {
				ch <- StreamChunk{Usage: event.Usage}
			}
			continue
		}
		choice := event.Choices[0]

		// Forward content immediately for token-level streaming.
		if choice.Delta.Content != "" {
			ch <- StreamChunk{Content: choice.Delta.Content}
		}

		// Accumulate tool-call fragments by index.
		for _, td := range choice.Delta.ToolCalls {
			tc, ok := toolBuf[td.Index]
			if !ok {
				tc = &ToolCall{}
				toolBuf[td.Index] = tc
				toolOrder = append(toolOrder, td.Index)
			}
			if td.ID != "" {
				tc.ID = td.ID
			}
			if td.Type != "" {
				tc.Type = td.Type
			}
			if td.Function.Name != "" {
				tc.Function.Name = td.Function.Name
			}
			if td.Function.Arguments != "" {
				tc.Function.Arguments += td.Function.Arguments
			}
		}

		// finish_reason marks the end of the assistant's response. For
		// "tool_calls" it's the trigger to flush the accumulated calls;
		// for "stop" with no tool calls accumulated, just done.
		if choice.FinishReason != "" {
			flushTools()
			done := StreamChunk{Done: true}
			if event.Usage != nil {
				done.Usage = event.Usage
			}
			ch <- done
		} else if event.Usage != nil {
			ch <- StreamChunk{Usage: event.Usage}
		}
	}
	if err := scanner.Err(); err != nil {
		ch <- StreamChunk{Error: err.Error()}
	}
	flushTools()
}

// StreamChunkToSSE formats a StreamChunk as an SSE event string.
func StreamChunkToSSE(chunk StreamChunk) string {
	var buf bytes.Buffer
	if chunk.Content != "" {
		buf.WriteString(fmt.Sprintf("data: %s\n\n", jsonString(map[string]string{"type": "content", "text": chunk.Content})))
	}
	if chunk.ToolCall != nil {
		call, _ := json.Marshal(chunk.ToolCall)
		buf.WriteString(fmt.Sprintf("data: %s\n\n", jsonString(map[string]any{"type": "tool_call", "call": string(call)})))
	}
	if chunk.Done {
		buf.WriteString("data: [DONE]\n\n")
	}
	if chunk.Error != "" {
		buf.WriteString(fmt.Sprintf("data: %s\n\n", jsonString(map[string]string{"type": "error", "text": chunk.Error})))
	}
	return buf.String()
}

func jsonString(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
