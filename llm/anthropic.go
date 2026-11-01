package llm

// anthropic.go — the Claude Messages API behind the same interface.
//
// Callers speak the OpenAI shape to every provider: Message, ToolDef, ChatOpts,
// StreamChunk. A client pointed at Anthropic builds that same body and sends it
// through openai-claude-api-shim, so the retries, the headers, the loop watcher
// and readStream are the code every other provider uses, and nothing above this
// package knows. The reason is cost: Anthropic's OpenAI-compatible endpoint reads
// nothing from the prompt cache. Measured: 0 tokens on a second round there,
// 11,484 on /v1/messages.

import (
	"io"
	"strings"

	shim "github.com/iodesystems/openai-claude-api-shim"
)

// usesMessagesAPI reports whether this client talks to /v1/messages.
func (c *Client) usesMessagesAPI() bool {
	return c.MessagesAPI || strings.Contains(c.baseURL, "api.anthropic.com")
}

// streamBody is what readStream reads: the response, in OpenAI SSE. A refusal is
// an error, because an empty reply that ended normally reads as an answer.
func (c *Client) streamBody(body io.ReadCloser) io.ReadCloser {
	if !c.usesMessagesAPI() {
		return body
	}
	return transcoded{Reader: shim.Transcode(body, shim.StreamOptions{RefusalAsError: true, Prices: c.ClaudePrices}), Closer: body}
}

// transcoded reads the transcoder and closes the response, which is what cuts a
// looping generation.
type transcoded struct {
	io.Reader
	io.Closer
}
