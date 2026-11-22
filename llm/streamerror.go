package llm

// StreamError is an error a provider delivered INSIDE a 200 response: an SSE
// `data:` event whose payload is `{"error":{...}}`. The HTTP status was already
// 200 by then, so postWithRetry never saw it, and until now the client kept
// only error.message — dropping the code, type and provider metadata that say
// which upstream failed and why (OpenRouter puts the upstream's own body in
// metadata.raw and its name in metadata.provider_name). A log line reading
// "JSON error injected into SSE stream" was undiagnosable for exactly that
// reason.
//
// It rides on StreamChunk.StreamErr beside the unchanged Error string, and
// agent.Session returns it as the Turn's error, so errors.As reaches it.
//
// A stream error that arrives before anything was forwarded to the caller and
// looks transient is RETRIED inside the client (see Client.pumpStream); this
// type is what the caller sees when it is not, or when the budget ran out.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

type StreamError struct {
	// Message is error.message, verbatim.
	Message string
	// Code is error.code as text. Providers send a number (OpenRouter: 502) or a
	// string ("rate_limit_exceeded"); both land here, a number as its digits.
	Code string
	// Type is error.type ("server_error", "overloaded_error", ...).
	Type string
	// Param is error.param when it is a scalar.
	Param string
	// Metadata is error.metadata raw (provider_name, raw, ...); nil when absent.
	Metadata json.RawMessage
	// Raw is the whole error object as received, for a host that wants a field
	// this type does not name.
	Raw string
	// RetryAfter is the server's own wait hint (error.retry_after, or
	// error.metadata.retry_after), clamped like an HTTP Retry-After. 0 = none.
	RetryAfter time.Duration
	// Reason is error.reason, the fair-share proxy's own classification
	// (corrallm: "queue-timeout", "rejected", ...). Empty when absent.
	Reason string
	// Status is error.status, an HTTP-like status some proxies repeat inside
	// the event. 0 when absent. Retryable falls back to it when Code is empty.
	Status int
}

// streamErrMetaClip bounds the metadata shown in Error(): enough to see the
// upstream's reason, not a multi-KB provider body in every log line.
const streamErrMetaClip = 300

func (e *StreamError) Error() string {
	var b strings.Builder
	b.WriteString(e.Message)
	var parts []string
	if e.Code != "" {
		parts = append(parts, "code "+e.Code)
	}
	if e.Type != "" {
		parts = append(parts, "type "+e.Type)
	}
	if len(parts) > 0 {
		fmt.Fprintf(&b, " (%s)", strings.Join(parts, ", "))
	}
	if len(e.Metadata) > 0 {
		m := string(e.Metadata)
		if len(m) > streamErrMetaClip {
			m = m[:streamErrMetaClip] + "...(clipped)"
		}
		b.WriteString(" metadata=" + m)
	}
	return b.String()
}

// Retryable reports whether the error looks transient. The matcher is small on
// purpose: a numeric code of 429 or any 5xx, or "overloaded", "rate limit" /
// "rate_limit", "unavailable" or "resource_exhausted" in the code, type or
// message. Anything else (400s, auth, content policy, an unknown message such
// as a proxy's "JSON error injected") is NOT retried: the same bytes would
// fail the same way, and a wrong guess costs a whole regenerated reply.
func (e *StreamError) Retryable() bool {
	var n int
	if _, err := fmt.Sscanf(e.Code, "%d", &n); err != nil && e.Code == "" {
		n = e.Status
	}
	if n == 429 || (n >= 500 && n <= 599) {
		return true
	}
	text := strings.ToLower(e.Code + " " + e.Type + " " + e.Message)
	for _, k := range []string{"overloaded", "rate limit", "rate_limit", "unavailable", "resource_exhausted"} {
		if strings.Contains(text, k) {
			return true
		}
	}
	return false
}

// parseStreamError reads an `error` value from a stream event: an object, or a
// bare string. nil when it carries no message (an event with `"error":null`
// or `{}` is not an error).
func parseStreamError(raw json.RawMessage) *StreamError {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return &StreamError{Message: s, Raw: string(raw)}
	}
	var o struct {
		Message    string          `json:"message"`
		Code       json.RawMessage `json:"code"`
		Type       string          `json:"type"`
		Param      json.RawMessage `json:"param"`
		Metadata   json.RawMessage `json:"metadata"`
		RetryAfter float64         `json:"retry_after"`
		Reason     string          `json:"reason"`
		Status     json.RawMessage `json:"status"`
	}
	if json.Unmarshal(raw, &o) != nil || o.Message == "" {
		return nil
	}
	e := &StreamError{Message: o.Message, Code: scalarText(o.Code), Type: o.Type,
		Param: scalarText(o.Param), Raw: string(raw), Reason: o.Reason}
	_, _ = fmt.Sscanf(scalarText(o.Status), "%d", &e.Status)
	if string(o.Metadata) != "null" {
		e.Metadata = o.Metadata
	}
	secs := o.RetryAfter
	if secs == 0 && len(e.Metadata) > 0 {
		var m struct {
			RetryAfter float64 `json:"retry_after"`
		}
		if json.Unmarshal(e.Metadata, &m) == nil {
			secs = m.RetryAfter
		}
	}
	if secs > 0 {
		e.RetryAfter = clampRetry(secs)
	}
	return e
}

// scalarText renders a JSON string or number as text; "" for null, absent or
// a composite.
func scalarText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var n json.Number
	if json.Unmarshal(raw, &n) == nil {
		return n.String()
	}
	return ""
}

// Err is the chunk's failure as an error: the typed *StreamError when the
// provider reported one, else a plain error carrying Error. Nil when the chunk
// carries no error. Consumers that return the failure should use this, so
// errors.As(err, &*StreamError) works for the caller of a Turn.
func (c StreamChunk) Err() error {
	if c.StreamErr != nil {
		return c.StreamErr
	}
	if c.Stall != nil {
		return c.Stall
	}
	if c.Error == "" {
		return nil
	}
	return errors.New(c.Error)
}
