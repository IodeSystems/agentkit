package llm

import (
	"net/http"
	"strings"
)

// Passthrough: the capability record for a request body this Client did not
// build.
//
// A proxy (corrallm) forwards a caller's chat body to a backend as it came,
// so the translation in chatBody and the learning in sendChat never see it.
// These two calls give a proxy the same behavior around its own forwarding:
// translate the body by the backend's record before sending, learn from the
// backend's answer after. Use one Client per (backend, model), seeded and
// reported like any other (SetCapabilities, OnLearn), so the record is the
// same one a direct caller of that backend would have.
//
// Scope, deliberately narrow: TranslateBody changes only what chatBody's
// translation changes (reasoning_effort, max_tokens). It does not know a
// dialect's other reasoning fields (chat_template_kwargs, thinking): a proxy
// that rewrites those keeps doing so itself. LearnResponse learns what a
// non-streaming answer shows: refusals, overflow counts, an effort accepted.
// What only a stream shows (stream.usage, the reasoning field) is learned
// when the proxy's own Client streams, not here.

// TranslateBody rewrites a caller's chat body by this Client's capability
// record, reports each change through OnTranslate, and returns them. A body
// decoded from JSON works as is: numbers may be float64 or json.Number. An
// empty record changes nothing.
func (c *Client) TranslateBody(body map[string]any) []Translation {
	trs := c.translate(body, nil)
	c.reportTranslations(trs)
	return trs
}

// LearnResponse learns from a backend's answer to sent, a body the Client did
// not send itself (pass the body as forwarded, after TranslateBody). status
// is the HTTP status; respBody is the response body, or at least its first few
// KiB for an error. It returns whether any fact was added or changed.
//
// A 2xx records the effort that was sent as accepted. A 4xx is read with the
// same refusal patterns and overflow parsing the Client uses on its own
// requests. A 429 or 5xx teaches nothing. LearnResponse never sends a request:
// a resend is the proxy's decision (see Client.ResendOnRefusal for the
// Client's own).
func (c *Client) LearnResponse(sent map[string]any, status int, respBody []byte) bool {
	if status >= 200 && status < 300 {
		before := c.Capabilities()
		c.observeAccepted(sentFrom(sent))
		return !sameFacts(before.Facts, c.Capabilities().Facts)
	}
	msg := strings.TrimSpace(string(respBody))
	taught := false
	for _, u := range refusalFacts(status, msg, sent) {
		if c.apply(u, SourceRefused) {
			taught = true
		}
	}
	if (status == http.StatusBadRequest || status == http.StatusRequestEntityTooLarge) && bodyIsInputTooLarge(msg) {
		if c.learnFromChatError(newInputTooLarge(status, msg)) {
			taught = true
		}
	}
	return taught
}

func sameFacts(a, b map[string]Fact) bool {
	if len(a) != len(b) {
		return false
	}
	for k, f := range a {
		g, ok := b[k]
		if !ok || g.Value != f.Value || g.Source != f.Source {
			return false
		}
	}
	return true
}
