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
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Message represents a chat message in the OpenAI-compatible format.
//
// Content is the plain-text body — the overwhelmingly common case, and the wire
// shape is a bare string. For MULTIMODAL input (a vision model reading an
// image — e.g. OCR), set Parts instead: when Parts is non-empty it OWNS the
// "content" field, which marshals as OpenAI's array-of-parts shape
// ([{type:text,...},{type:image_url,...}]) and Content is ignored. The string
// path is left byte-for-byte unchanged so every existing caller is unaffected.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
	// Parts, when non-empty, replaces Content with a multimodal content array
	// (text + image parts). Send-only: responses are always plain text, so this
	// is never populated on a decoded reply. Tagged "-" — MarshalJSON emits it
	// under "content" itself.
	Parts []ContentPart `json:"-"`
	// ToolCalls carries an assistant message's requested tool calls, so a
	// reconstructed conversation replays a valid assistant(tool_calls) →
	// tool(tool_call_id) structure instead of orphan tool messages.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a role="tool" result back to the assistant tool call
	// that produced it (OpenAI requires this correlation).
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ContentPart is one element of a multimodal content array. Type is "text" or
// "image_url"; exactly one of Text / ImageURL is set to match.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL is an image reference: either an https URL or an inline data: URI
// ("data:image/png;base64,…"). Detail ("auto"|"low"|"high") is optional.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// TextPart builds a text content part.
func TextPart(text string) ContentPart { return ContentPart{Type: "text", Text: text} }

// ImagePart builds an image content part from a URL or data: URI.
func ImagePart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url}}
}

// ImageData builds an image content part from raw bytes, encoding them as a
// base64 data: URI with the given MIME type (e.g. "image/png"). This is the
// OCR path: rasterize/extract a page image, hand the bytes straight to a vision
// model without writing a temp file or hosting a URL.
func ImageData(mime string, raw []byte) ContentPart {
	uri := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw)
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: uri}}
}

// MarshalJSON renders "content" as a bare string (Parts empty, the default) or
// as the multimodal array (Parts set). Everything else marshals normally.
func (m Message) MarshalJSON() ([]byte, error) {
	type alias Message // strip the custom marshaler to avoid recursion
	if len(m.Parts) == 0 {
		return json.Marshal(alias(m))
	}
	// Outer Content (array, depth 0) shadows alias.Content (string, depth 1).
	return json.Marshal(struct {
		alias
		Content []ContentPart `json:"content"`
	}{alias: alias(m), Content: m.Parts})
}

// ToolCall represents a tool call request from the LLM.
//
// Arguments is a STRING holding JSON, per the OpenAI spec — and per that same
// spec it is not guaranteed to be valid JSON, so every consumer must parse
// defensively (agent.Session refuses to dispatch a call it cannot parse).
// Marshalling keeps the string form, which is what a request must carry.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// UnmarshalJSON accepts `arguments` as either the spec's JSON string or a bare
// JSON object, which some providers send instead.
//
// Without this, an object made the whole decode fail: the non-streaming path
// returned a decode error for the entire response, and the streaming path — which
// ignores unparseable events by design, since a provider may interleave shapes it
// does not recognise — dropped the chunk SILENTLY. Measured: zero tool calls, zero
// errors, no log. The turn saw a model that had said nothing, which is
// indistinguishable from a model that chose to say nothing.
func (tc *ToolCall) UnmarshalJSON(b []byte) error {
	var shadow struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(b, &shadow); err != nil {
		return err
	}
	tc.ID = shadow.ID
	tc.Type = shadow.Type
	tc.Function.Name = shadow.Function.Name
	tc.Function.Arguments = normalizeArgs(shadow.Function.Arguments)
	return nil
}

// normalizeArgs renders a wire `arguments` value as the JSON string the rest of
// the code works in.
//
// A JSON string is unquoted (its contents ARE the arguments, and in a stream it
// is one fragment of them). Anything else is taken verbatim as its own JSON —
// an off-spec provider sending the object directly. Absent and null both mean
// "no arguments", which is distinct from unparseable and must not be reported
// as a failure: a tool with no parameters legitimately has none.
func normalizeArgs(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return ""
	}
	if s[0] == '"' {
		var out string
		if err := json.Unmarshal(raw, &out); err != nil {
			return "" // unreachable: raw came from a successful parse
		}
		return out
	}
	log.Printf("llm: provider sent tool-call arguments as %s, not the string the "+
		"OpenAI spec defines; accepting it verbatim", jsonShape(s))
	return s
}

// jsonShape names a JSON value's kind from its first byte, for the log above.
func jsonShape(s string) string {
	switch s[0] {
	case '{':
		return "an object"
	case '[':
		return "an array"
	case 't', 'f':
		return "a boolean"
	default:
		return "a number"
	}
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

	// PartialToolCalls carries whatever tool-call arguments had accumulated
	// when an Error chunk was produced. Providers stream tool arguments
	// incrementally, so a call cut off mid-argument (a context window filling
	// mid-write) is already buffered client-side — the caller can salvage the
	// work instead of making the model regenerate it.
	//
	// These are INCOMPLETE and deliberately NOT delivered as ToolCall chunks:
	// dispatching a truncated call would run it with half its arguments.
	// Only ever set alongside Error.
	PartialToolCalls []ToolCall

	// StopReason names a non-model reason the stream ended — set on the Done
	// chunk. Empty on a normal completion, so consumers that ignore it are
	// unaffected. Today the only value is StopReasonRepetition.
	StopReason StopReason

	// Repetition describes the degenerate loop that was cut, when StopReason is
	// StopReasonRepetition. It carries what to log, what to tell the model, and
	// how many trailing bytes are redundant copies. Nil otherwise.
	Repetition *RepetitionInfo
}

// Usage carries provider-reported token counts for one chat
// completion. Fields beyond prompt/completion/total are optional and
// only filled when the provider returns them (Anthropic's cache
// fields, OpenAI's reasoning tokens, etc).
//
// PromptTokens counts the prompt the provider was SENT, not the prompt it
// actually had to process — a cached prefix is billed here in full even
// though it was never re-evaluated. Use CachedPromptTokens/NewPromptTokens
// for the real work; see the note on CachedPromptTokens for why that
// distinction is easy to get catastrophically wrong.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// Anthropic's shape: flat, top-level.
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`

	// OpenAI's shape: NESTED. llama-server (and thus corrallm) speaks this
	// one, and only this one — it was previously unparsed, so every cached
	// token read as zero and prompt_tokens looked like real work forever.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`

	// LatencyMS is wall-clock for this one round-trip, measured client-side.
	// Not reported by any provider — but "how long did that call take" is half
	// of what anyone asks of a trace, and without it the token counts alone
	// cannot distinguish a slow cold load from a large generation.
	LatencyMS int64 `json:"latency_ms,omitempty"`
}

// PromptTokensDetails is OpenAI's nested prompt-token breakdown.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// CachedPromptTokens returns prompt tokens served from cache, normalizing the
// two provider shapes.
//
// This matters more than it looks. A cached prefix costs ~nothing to process
// (llama.cpp reuses the KV slot; measured ~1,860 tok/s prompt eval vs ~110
// tok/s generation), but PromptTokens still reports it at full size on EVERY
// turn. Summing PromptTokens across a conversation therefore charges a stable
// prefix once per turn — which ranks the most-cached region of the prompt (the
// tool schemas, byte-identical every turn) as the dominant cost when it is
// very nearly free. That artifact drove a whole tool-surface redesign against
// a gap that, measured on generated tokens and wall clock, was ~12%.
func (u *Usage) CachedPromptTokens() int {
	if u == nil {
		return 0
	}
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		return u.PromptTokensDetails.CachedTokens
	}
	return u.CacheReadInputTokens
}

// NewPromptTokens returns the prompt tokens the provider actually had to
// evaluate this turn — the honest per-turn prompt cost.
func (u *Usage) NewPromptTokens() int {
	if u == nil {
		return 0
	}
	if n := u.PromptTokens - u.CachedPromptTokens(); n > 0 {
		return n
	}
	return 0
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
	//
	// NEGATIVE means unbounded: ride out backpressure until the ctx says stop.
	// Correct for a single-slot endpoint, where a wait is not a delay but
	// another attempt at the only slot, and a 5-minute default gives up in the
	// middle of an ordinary busy spell.
	//
	// Unbounded is safe because it only extends the 429 path. 5xx is ALSO capped
	// by an attempt count (retry5xxMaxAttempts), so a genuinely broken upstream
	// still stops; and a 500 reporting unparseable tool-call arguments is
	// returned immediately without retrying, since the model produced those bytes
	// deterministically from this context and every retry reproduces them.
	// Nothing can spin forever on an outcome that cannot change.
	RetryBudget time.Duration

	// Retry5xxAttempts caps how many times a 5xx is retried. 0 → default (5).
	//
	// The default is tuned for a chat turn, where five failures means the
	// upstream is genuinely broken and an operator should hear about it. It is
	// wrong for a long BATCH: transcribing a 33-page document is ~33 requests,
	// and a single upstream blip outlasting ~15s of backoff fails the whole
	// document. Measured on a real corpus, every document over 20 pages failed
	// this way and no document over 20 pages ever completed.
	//
	// Raising it is safe because RetryBudget still bounds the wall clock — the
	// attempt cap and the budget are two different guards, and for batch work the
	// budget is the one that should bind.
	Retry5xxAttempts int

	// Repetition configures degenerate-loop detection on streamed output. The
	// zero value is ON with defaults; set Repetition.Off to disable. See
	// RepetitionGuard for why this is not opt-in.
	Repetition RepetitionGuard

	// OnRetry, when set, is called once per retry decision (and once when the
	// loop recovers or gives up), so a caller can SHOW the wait instead of
	// sitting mute through it. See RetryEvent.
	//
	// Called from the goroutine issuing the request, synchronously, while the
	// caller is blocked in Chat/ChatStream/Embed — so it must not block for long
	// and must not re-enter the client.
	OnRetry func(RetryEvent)
}

// retry5xxAttempts is the effective 5xx attempt cap.
func (c *Client) retry5xxAttempts() int {
	if c.Retry5xxAttempts > 0 {
		return c.Retry5xxAttempts
	}
	return retry5xxMaxAttempts
}

func NewClient(baseURL, apiKey, model string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		http:    &http.Client{},
	}
}

// RetryPolicy reports the backoff schedule this client actually uses: the first
// delay, the ceiling it climbs to, and the wall-clock budget for one call.
//
// It exists for a caller that must add an OUTER retry — the client's loop covers
// everything up to the response headers, but a stream that dies MID-generation
// is not resumable here, so only the caller can decide to run the turn again.
// Such a caller should not invent a second schedule out of thin air: the numbers
// (and any operator override of RetryBudget) are policy, and one policy is
// better than two that disagree.
func (c *Client) RetryPolicy() (initial, max, budget time.Duration) {
	return retryInitialBackoff, retryMaxBackoff, c.retryBudget()
}

// retryBudget returns the effective retry budget: the field, the default, or
// effectively-forever when the caller asked for unbounded. The ctx still bounds
// it, so "forever" means "as long as the caller is willing to wait".
func (c *Client) retryBudget() time.Duration {
	if c.RetryBudget < 0 {
		return unboundedRetryBudget
	}
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

// embedURL returns the embeddings endpoint (sibling of chatURL).
func (c *Client) embedURL() string {
	u := c.baseURL
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	return u + "/embeddings"
}

// Embed returns one embedding vector per input string from an OpenAI-compatible
// /v1/embeddings endpoint. model is passed explicitly because embeddings use a
// different model than chat (the Client's configured model is the chat model).
// Output order matches input order. Honors the same auth + retry policy as Chat.
func (c *Client) Embed(ctx context.Context, model string, input []string) ([][]float32, error) {
	if len(input) == 0 {
		return nil, nil
	}
	payload, err := json.Marshal(map[string]any{"model": model, "input": input})
	if err != nil {
		return nil, fmt.Errorf("llm: marshal embed: %w", err)
	}
	resp, err := c.postWithRetry(ctx, c.embedURL(), payload, "")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, statusError(resp)
	}
	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("llm: decode embed: %w", err)
	}
	if len(result.Data) != len(input) {
		return nil, fmt.Errorf("llm: embed returned %d vectors for %d inputs", len(result.Data), len(input))
	}
	// Re-order by Index defensively (spec says input order, but don't trust it).
	vecs := make([][]float32, len(input))
	for _, d := range result.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("llm: embed index %d out of range", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
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

// unboundedRetryBudget stands in for "no ceiling" (RetryBudget < 0). A duration
// rather than a special case throughout the retry loop: every deadline
// comparison keeps working unchanged, and the caller's ctx remains the real
// bound. A century is past any plausible ctx.
var unboundedRetryBudget = 100 * 365 * 24 * time.Hour

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
	return c.postWithRetry(ctx, c.chatURL(), payload, traceID)
}

// postWithRetry POSTs payload to url with the same auth headers and 429/5xx
// retry policy as the chat path — shared by Chat/ChatStream and Embed so every
// endpoint honors the fair-share backpressure + retry budget identically.
func (c *Client) postWithRetry(ctx context.Context, url string, payload []byte, traceID string) (*http.Response, error) {
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
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
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
			// Transport path: the gateway is DOWN, not answering. This never
			// reaches the 429/5xx branches below, because those classify a
			// status code and there is no response to classify — so before
			// this branch retried, a restart of the gateway failed every
			// in-flight and newly-issued call on attempt 1, with the 5m budget
			// and the 5xx cap both untouched.
			//
			// Treated like 429 rather than 5xx: "the server is not here yet"
			// is wait-your-turn, not broken, so no attempt cap — the wall-clock
			// RetryBudget and the caller's ctx are the bounds. A default 5m
			// budget covers an ordinary restart (stop, rebuild, cold-load)
			// with room to spare; a caller that wants to ride out longer sets
			// Client.RetryBudget, or a negative value for no ceiling.
			//
			// Scope: this covers failures up to and including response
			// headers. A stream that dies MID-generation is readStream's
			// problem and is not resumable — the tokens are already emitted,
			// so a retry would regenerate from scratch rather than continue.
			if !transportRetryable(ctx, err) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Err: err,
					Elapsed: time.Since(start), Budget: budget,
					Reason: fmt.Sprintf("cannot reach the provider and retrying will not help: %v", err)})
				return nil, fmt.Errorf("llm: do: %w", err)
			}
			if time.Now().Add(backoff).After(deadline) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Err: err,
					Elapsed: time.Since(start), Budget: budget,
					Reason: fmt.Sprintf("gave up after %s of unreachable provider (retry budget %s): %v",
						time.Since(start).Round(time.Second), budget, err)})
				return nil, fmt.Errorf("llm: retry budget %s exhausted after %s of transport failures (last: %v)",
					budget, time.Since(start).Round(time.Second), err)
			}
			c.emitRetry(RetryEvent{Kind: RetryTransport, Attempt: attempt + 1, Err: err,
				Delay: backoff, Elapsed: time.Since(start), Budget: budget})
			if attempt < retryLogEvery || attempt%retryLogEvery == 0 {
				log.Printf("llm: cannot reach provider (attempt %d): %v; retrying in %s",
					attempt+1, err, backoff)
			}
			if !sleepOrCancel(ctx, backoff) {
				return nil, ctx.Err()
			}
			backoff *= 2
			if backoff > retryMaxBackoff {
				backoff = retryMaxBackoff
			}
			continue
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
			// The queue detail (slots busy, requests ahead, why) comes back with the
			// same parse that yields the delay, so reporting it costs nothing — and
			// it is the difference between "429" and "4/4 busy, 2 ahead of you".
			bp, asked := backpressureFrom(resp.Header, bodyBytes)
			if asked {
				sleep = bp.RetryAfter
			}

			if time.Now().Add(sleep).After(deadline) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Status: status,
					Elapsed: time.Since(start), Budget: budget, BP: &bp,
					Reason: fmt.Sprintf("gave up after %s of 429 backpressure (retry budget %s)",
						time.Since(start).Round(time.Second), budget)})
				return nil, fmt.Errorf("llm: retry budget %s exhausted after %s of 429 backpressure",
					budget, time.Since(start).Round(time.Second))
			}
			c.emitRetry(RetryEvent{Kind: Retry429, Attempt: attempt + 1, Status: status,
				Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))),
				Delay: sleep, Elapsed: time.Since(start), Budget: budget,
				BP: &bp, ServerAsked: asked})
			if attempt < retryLogEvery || attempt%retryLogEvery == 0 {
				if q := bp.String(); q != "" {
					log.Printf("llm: provider returned 429 — %s (attempt %d); retrying in %s",
						q, attempt+1, sleep)
				} else {
					log.Printf("llm: provider returned 429 (attempt %d); retrying in %s",
						attempt+1, sleep)
				}
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
			d, serverAsked := retryAfterFrom(resp.Header, bodyBytes)
			if serverAsked {
				sleep = d
			}

			// A 5xx that reports unparseable tool-call arguments is NOT
			// transient: the model produced them deterministically from this
			// exact context, so every retry reproduces the same bytes. Observed
			// on llama.cpp — five attempts, all failing at the identical column,
			// 15s of backoff spent on an outcome that could not change.
			if bodyIsTruncatedToolCall(string(bodyBytes)) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Status: status,
					Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))), Elapsed: time.Since(start), Budget: budget,
					Reason: "not retried: the provider rejected a tool call the model produced, so every retry reproduces it"})
				return nil, &TruncatedToolCallError{Status: status, Body: string(bodyBytes)}
			}
			// A 5xx that says the INPUT was too large is not transient either: the
			// request is the same bytes every time, so every retry reproduces it.
			// Observed as llama.cpp's "input (35871 tokens) is too large to
			// process. increase the physical batch size (current batch size:
			// 8192)" — five attempts and ~15s of backoff spent on an outcome that
			// could not change, and reported as an upstream fault when the caller
			// had simply sent too much at once.
			if bodyIsInputTooLarge(string(bodyBytes)) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Status: status,
					Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))), Elapsed: time.Since(start), Budget: budget,
					Reason: "not retried: the request is too large for this endpoint, so every retry fails identically"})
				return nil, fmt.Errorf("llm: input too large for this endpoint (status %d, not retried): %s",
					status, strings.TrimSpace(firstLineOfBody(string(bodyBytes))))
			}
			if fiveXXAttempts >= c.retry5xxAttempts() {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Status: status,
					Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))), Elapsed: time.Since(start), Budget: budget,
					Attempts5xx: fiveXXAttempts, Max5xx: c.retry5xxAttempts(),
					Reason: fmt.Sprintf("gave up after %d attempts against status %d", fiveXXAttempts, status)})
				// Carry the SERVER'S OWN MESSAGE. Reporting only the status code
				// discards the one thing that says what to do about it: a corpus
				// spent hours on "llm: status 500 (after 4 retries)" repeated in
				// every failed job, while the endpoint had been saying "input
				// (35871 tokens) is too large to process. increase the physical
				// batch size (current batch size: 8192)" the whole time. The body
				// is already read for retry-after and is capped at 8 KiB.
				if body := strings.TrimSpace(firstLineOfBody(string(bodyBytes))); body != "" {
					return nil, fmt.Errorf("llm: status %d (after %d retries): %s", status, fiveXXAttempts-1, body)
				}
				return nil, fmt.Errorf("llm: status %d (after %d retries)", status, fiveXXAttempts-1)
			}
			if time.Now().Add(sleep).After(deadline) {
				c.emitRetry(RetryEvent{Kind: RetryGiveUp, Attempt: attempt + 1, Status: status,
					Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))), Elapsed: time.Since(start), Budget: budget,
					Attempts5xx: fiveXXAttempts, Max5xx: c.retry5xxAttempts(),
					Reason: fmt.Sprintf("gave up after %s against status %d (retry budget %s)",
						time.Since(start).Round(time.Second), status, budget)})
				return nil, fmt.Errorf("llm: retry budget %s exhausted after %s (last status %d)",
					budget, time.Since(start).Round(time.Second), status)
			}
			c.emitRetry(RetryEvent{Kind: Retry5xx, Attempt: attempt + 1, Status: status,
				Body: strings.TrimSpace(firstLineOfBody(string(bodyBytes))),
				Delay: sleep, Elapsed: time.Since(start), Budget: budget,
				Attempts5xx: fiveXXAttempts, Max5xx: c.retry5xxAttempts(), ServerAsked: serverAsked})
			log.Printf("llm: upstream returned %d (5xx attempt %d/%d); retrying in %s",
				status, fiveXXAttempts, c.retry5xxAttempts(), sleep)
			if !sleepOrCancel(ctx, sleep) {
				return nil, ctx.Err()
			}
		default:
			// Anything else (2xx success, 4xx non-429): hand back. A retried call
			// that finally worked says so, so a UI can take its banner down —
			// only for a real success, since a 400 arriving after five 503s is
			// the end of the retrying, not a recovery.
			if attempt > 0 && status < 400 {
				c.emitRetry(RetryEvent{Kind: RetryRecovered, Attempt: attempt + 1, Status: status,
					Elapsed: time.Since(start), Budget: budget})
			}
			return resp, nil
		}
		backoff *= 2
		if backoff > retryMaxBackoff {
			backoff = retryMaxBackoff
		}
	}
}

// transportRetryable reports whether an error from http.Client.Do is worth
// waiting out. True for the ways a restarting or briefly-unreachable gateway
// fails — connection refused, reset, EOF, timeouts, DNS blips.
//
// False for the two classes where waiting is pure delay before the same
// failure:
//
//   - the caller's ctx ended. Retrying past a cancellation is a bug, and the
//     ctx error arrives here wrapped in *url.Error rather than as a status.
//   - configuration faults — a bad scheme or a TLS chain the client will never
//     accept. These are deterministic; no amount of backoff changes them, and
//     retrying buries the real error under budget-exhaustion noise.
func transportRetryable(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx.Err() != nil ||
		errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var (
		unknownAuthority x509.UnknownAuthorityError
		hostnameErr      x509.HostnameError
		certInvalid      x509.CertificateInvalidError
	)
	if errors.As(err, &unknownAuthority) || errors.As(err, &hostnameErr) || errors.As(err, &certInvalid) {
		return false
	}
	// net/http reports an unusable URL as a plain string on *url.Error; there is
	// no sentinel to match against.
	if strings.Contains(err.Error(), "unsupported protocol scheme") {
		return false
	}
	return true
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

	// Temperature and Seed pin sampling for reproducibility. Both are pointers
	// so "unset" stays distinct from a deliberate 0 — temperature 0 is exactly
	// the value a caller most wants to send, and a plain float64 could never
	// express it without also forcing it on every caller that had no opinion.
	//
	// A measurement harness needs these: without them the server's own sampling
	// config decides, so a model launched with --temp 0.7 makes a pass/fail
	// probe a coin flip and single-shot runs disagree with themselves.
	// Providers that do not support seed ignore the field.
	Temperature *float64
	Seed        *int

	// Stop ends generation when any of these strings is produced. The provider
	// excludes the matched string from the returned text.
	//
	// Needed with Grammar: a completed grammar does NOT force EOS. Measured with
	// `root ::= call`, the model finished a valid call and then emitted the same
	// call again, repeatedly, to the token cap. Stopping on the block terminator
	// is what actually ends the turn.
	Stop []string

	// MaxTokens caps the completion. 0 = leave it to the server.
	//
	// This matters most with Grammar set. A grammar that permits repetition (a
	// heredoc body is `bodyline*`) never REQUIRES termination, so an unbounded
	// request can run to the context limit instead of finishing — observed as a
	// hang, not an error.
	MaxTokens int

	// Repetition controls. Pointers for the same reason as Temperature: 0 is a
	// meaningful value (it is "off"), and a plain float64 could not distinguish
	// "the caller wants it off" from "the caller had no opinion".
	//
	// These exist because a CUT is not a cure. RepetitionGuard stops a runaway
	// after the fact; it does nothing about the cause, and at temperature 0 the
	// cause is deterministic — measured on one survey page, the same generation
	// was cut three times in 34 seconds with byte-identical output, because a
	// greedy decoder in a repeating basin has no way out. Only sampling does.
	//
	// FrequencyPenalty and PresencePenalty are the OpenAI-standard pair
	// (penalize by count seen, and by seen-at-all). RepeatPenalty is
	// llama.cpp's own multiplicative penalty over a trailing window, ignored by
	// providers that do not implement it — 1.0 is off, ~1.1 is a light touch.
	FrequencyPenalty *float64
	PresencePenalty  *float64
	RepeatPenalty    *float64
}

// applySampling copies pinned sampling params onto an outgoing request body.
func applySampling(body map[string]any, opts *ChatOpts) {
	if opts == nil {
		return
	}
	if opts.Temperature != nil {
		body["temperature"] = *opts.Temperature
	}
	if opts.Seed != nil {
		body["seed"] = *opts.Seed
	}
	if opts.MaxTokens > 0 {
		body["max_tokens"] = opts.MaxTokens
	}
	if len(opts.Stop) > 0 {
		body["stop"] = opts.Stop
	}
	if opts.FrequencyPenalty != nil {
		body["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		body["presence_penalty"] = *opts.PresencePenalty
	}
	if opts.RepeatPenalty != nil {
		body["repeat_penalty"] = *opts.RepeatPenalty
	}
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
	applySampling(body, opts)

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal: %w", err)
	}

	traceID := ""
	if opts != nil {
		traceID = opts.TraceID
	}
	// Measured from BEFORE the request so latency includes retries, queueing and
	// a cold model load — the parts that make a call slow are exactly the parts
	// a server-side number would omit.
	callStart := time.Now()
	resp, err := c.postChatWithRetry(ctx, payload, traceID)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, statusError(resp)
	}

	ch := make(chan StreamChunk, 64)
	go c.readStream(ctx, resp.Body, ch, callStart)
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
	return c.ChatWithOpts(ctx, messages, tools, nil)
}

// ChatWithOpts is Chat with sampling and constrained-decoding options.
func (c *Client) ChatWithOpts(ctx context.Context, messages []Message, tools []ToolDef, opts *ChatOpts) (string, []ToolCall, error) {
	body := map[string]any{
		"model":    c.model,
		"messages": messages,
		"stream":   false,
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	applySampling(body, opts)

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
// readStream parses the SSE body. callStart is when the request was issued, so
// usage chunks can carry the round-trip latency no provider reports.
func (c *Client) readStream(ctx context.Context, body io.ReadCloser, ch chan<- StreamChunk, callStart time.Time) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 65536), 65536)

	// Tool-call accumulator. Keyed by index from the streaming protocol.
	// Order is maintained via toolOrder (deltas might arrive out of
	// numeric order, but typically index 0 streams first).
	toolBuf := map[int]*ToolCall{}
	var toolOrder []int
	// Stream forensics: on an abrupt end these are the difference between
	// "the provider truncated us" and "we mishandled a normal stream".
	var sawDone, sawFinish bool
	var events, contentChars int

	// Degenerate-loop watchers, one per logical byte stream: content, plus each
	// tool call's arguments (keyed by the protocol's index). Created lazily so a
	// stream that never uses a channel never allocates its window.
	var contentRep *repWatcher
	toolRep := map[int]*repWatcher{}
	// cutLoop reports the loop and ends the stream. Returning closes the body
	// (deferred above), which disconnects — and the disconnect is what makes the
	// server abandon the generation instead of running it to the context limit.
	cutLoop := func(info *RepetitionInfo) {
		log.Printf(
			"llm: CUT A LOOPING GENERATION after %d events (%d content chars) — the model %s. "+
				"The stream was closed client-side; output up to the cut is kept. This is a MODEL "+
				"fault (degenerate repetition), not a transport error, and is not worth retrying "+
				"unchanged.", events, contentChars, info)
		ch <- StreamChunk{Done: true, StopReason: StopReasonRepetition, Repetition: info}
	}

	// flushTools emits the accumulated calls. reason is the provider's
	// finish_reason ("" when the stream simply ended).
	//
	// Calls with unparseable arguments ARE emitted. The transport's job is to
	// report faithfully what the provider sent, not to decide the turn's fate:
	// swallowing the call here would abort the turn with no tool_result, so the
	// model never learns its call was cut off and cannot retry smaller. The
	// consumer (agent.Session) refuses to DISPATCH such a call and answers it
	// with an error result instead — which keeps the call/result pair intact and
	// closes the loop.
	//
	// A clean finish is NOT evidence the payload is whole: a generation stopped
	// by the context window or a token cap terminates cleanly — measured
	// finish_reason="length" WITH [DONE], carrying 450 chars of unterminated
	// JSON. So the log below fires on every path, not just an abrupt end.
	flushTools := func(reason string) {
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
			if !json.Valid([]byte(tc.Function.Arguments)) {
				why := "the stream ended before they were complete"
				if reason != "" {
					why = "generation stopped early (finish_reason=" + reason + ")"
				}
				log.Printf(
					"llm: TRUNCATED TOOL CALL — %q has unparseable arguments (%d chars) because %s. "+
						"It is emitted so the caller can answer it with an error result, but it must "+
						"NOT be dispatched or persisted verbatim. Cause is upstream, not the caller.",
					tc.Function.Name, len(tc.Function.Arguments), why)
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
		events++
		if data == "[DONE]" {
			sawDone = true
			flushTools("")
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
							Name string `json:"name,omitempty"`
							// RawMessage, not string: a provider that sends the
							// object directly would otherwise fail this event's
							// unmarshal, and the loop below drops unparseable
							// events without a word. See normalizeArgs.
							Arguments json.RawMessage `json:"arguments,omitempty"`
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
			// Hand back whatever arguments had accumulated. The buffer holds the
			// partial write at exactly this moment; returning without it drops
			// work the caller could otherwise resume from.
			ch <- StreamChunk{Error: event.Error.Message, PartialToolCalls: partialTools(toolBuf, toolOrder)}
			return
		}
		// Usage-only chunk (no choices) — providers using
		// stream_options.include_usage send one of these after content.
		if len(event.Choices) == 0 {
			if event.Usage != nil {
				event.Usage.LatencyMS = time.Since(callStart).Milliseconds()
				ch <- StreamChunk{Usage: event.Usage}
			}
			continue
		}
		choice := event.Choices[0]

		// Forward content immediately for token-level streaming.
		if choice.Delta.Content != "" {
			contentChars += len(choice.Delta.Content)
			ch <- StreamChunk{Content: choice.Delta.Content}
			if !c.Repetition.Off {
				if contentRep == nil {
					contentRep = newRepWatcher(c.Repetition, "content")
				}
				if info := contentRep.feed(choice.Delta.Content); info != nil {
					cutLoop(info)
					return
				}
			}
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
			// Fragments concatenate: the protocol splits one call's arguments
			// across many deltas. normalizeArgs unquotes each fragment back to
			// the raw characters the model produced.
			if frag := normalizeArgs(td.Function.Arguments); frag != "" {
				tc.Function.Arguments += frag
				if c.Repetition.Off {
					continue
				}
				w := toolRep[td.Index]
				if w == nil {
					w = newRepWatcher(c.Repetition, "tool arguments")
					toolRep[td.Index] = w
				}
				if info := w.feed(frag); info != nil {
					// The name is known by now even though it arrived on an
					// earlier delta than the fragment that tripped this.
					if tc.Function.Name != "" {
						info.Where = "tool:" + tc.Function.Name + " arguments"
					}
					// The accumulated call is DISCARDED, not flushed: its
					// arguments are mid-loop and unparseable, and emitting it
					// would put a call built from garbage into the history. The
					// consumer answers the repetition itself.
					cutLoop(info)
					return
				}
			}
		}

		// finish_reason marks the end of the assistant's response. For
		// "tool_calls" it's the trigger to flush the accumulated calls;
		// for "stop" with no tool calls accumulated, just done.
		if choice.FinishReason != "" {
			sawFinish = true
			flushTools(choice.FinishReason)
			done := StreamChunk{Done: true}
			if event.Usage != nil {
				event.Usage.LatencyMS = time.Since(callStart).Milliseconds()
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
	// The stream ended WITHOUT a finish_reason — a dropped connection, a server
	// that gave up mid-generation, a truncated body. Anything still in the
	// accumulator is therefore INCOMPLETE.
	//
	// Flushing it as a normal tool call is how a malformed call enters a
	// conversation in the first place: the provider never sent one, we
	// manufacture it from a partial buffer, the caller dispatches it and
	// persists it, and every later request carrying that history is rejected at
	// parse time by any server that deserializes arguments to render them.
	//
	// So emit partials as an ERROR carrying the fragments, never as ToolCalls.
	// The caller can salvage the text; it cannot mistake it for a real call.
	if partial := partialTools(toolBuf, toolOrder); len(partial) > 0 {
		// Loud and specific: this is a PROVIDER fault (it closed the body
		// mid-message), and the numbers below are what distinguish it from a
		// client-side parsing mistake. Without them a reader sees only "bad
		// JSON somewhere" and blames the wrong component — which is exactly
		// what happened when this bug was first hit.
		var names []string
		bytes := 0
		for _, tc := range partial {
			names = append(names, tc.Function.Name)
			bytes += len(tc.Function.Arguments)
		}
		detail := fmt.Sprintf(
			"llm: PROVIDER TRUNCATED THE STREAM — body closed after %d events (%d content chars) "+
				"with finish_reason=%v and [DONE]=%v; %d tool call(s) %v left incomplete with %d "+
				"argument chars buffered. The partial call is NOT usable and was discarded rather "+
				"than emitted: persisting it would make every later request carrying this history "+
				"fail to parse. Cause is upstream (context exhausted mid-generation, or a dropped "+
				"connection), not the caller.",
			events, contentChars, sawFinish, sawDone, len(partial), names, bytes)
		log.Print(detail)
		ch <- StreamChunk{Error: detail, PartialToolCalls: partial}
	}
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

// partialTools snapshots the tool-call accumulator for an error chunk. The
// entries are incomplete by construction — the point is to let a caller see how
// far a cut-off write got, not to execute them.
func partialTools(buf map[int]*ToolCall, order []int) []ToolCall {
	if len(buf) == 0 {
		return nil
	}
	out := make([]ToolCall, 0, len(buf))
	for _, idx := range order {
		if tc := buf[idx]; tc != nil {
			out = append(out, *tc)
		}
	}
	return out
}

// TruncatedToolCallError reports a response whose tool-call arguments could not
// be parsed because generation was CUT OFF — typically the context window
// filling mid-argument while the model writes a large file.
//
// It is separated from ordinary 5xx precisely so a caller does not retry it: at
// temperature 0 the same context reproduces the same truncation byte for byte.
// The useful responses are to compact the context and re-issue, or to ask for a
// smaller edit — not to try again unchanged.
type TruncatedToolCallError struct {
	Status int
	Body   string
}

func (e *TruncatedToolCallError) Error() string {
	return fmt.Sprintf("llm: tool-call arguments truncated (status %d) — context likely exhausted mid-write", e.Status)
}

// bodyIsInputTooLarge recognises the endpoint complaining that the REQUEST did
// not fit — a batch or context overflow. Deterministic: the same bytes fail the
// same way, so retrying is pure delay before an identical failure. The fix is in
// the caller (send less), which it cannot learn while this looks transient.
func bodyIsInputTooLarge(body string) bool {
	b := strings.ToLower(body)
	if b == "" {
		return false
	}
	return strings.Contains(b, "too large to process") ||
		strings.Contains(b, "physical batch size") ||
		strings.Contains(b, "exceeds the available context") ||
		strings.Contains(b, "exceed_context_size")
}

// firstLineOfBody trims a provider error body to something loggable.
func firstLineOfBody(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

// bodyIsTruncatedToolCall recognises the provider's complaint that it could not
// parse tool-call arguments. Deliberately narrow: a generic 5xx stays retryable,
// since only THIS shape is known to be deterministic.
func bodyIsTruncatedToolCall(body string) bool {
	if body == "" {
		return false
	}
	b := strings.ToLower(body)
	if !strings.Contains(b, "tool call") && !strings.Contains(b, "tool_call") {
		return false
	}
	return strings.Contains(b, "parse") &&
		(strings.Contains(b, "unexpected end") ||
			strings.Contains(b, "missing closing quote") ||
			strings.Contains(b, "parse_error") ||
			strings.Contains(b, "invalid string"))
}
