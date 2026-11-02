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
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

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

// aliasTTL is how long an alias stays resolved. A new model is a new alias target,
// and a session should not change model under itself more than it has to.
const aliasTTL = time.Hour

// wireModel is the model to send: an alias (opus-latest) resolved from the
// endpoint's model list, anything else as it is. A list that cannot be read leaves
// the alias, and the provider's refusal says what is wrong.
func (c *Client) wireModel(ctx context.Context) string {
	if !c.usesMessagesAPI() || !shim.IsAlias(c.model) {
		return c.model
	}
	c.aliasMu.Lock()
	defer c.aliasMu.Unlock()
	if c.aliasID != "" && time.Since(c.aliasAt) < aliasTTL {
		return c.aliasID
	}
	id, err := c.resolveAlias(ctx)
	if err != nil {
		log.Printf("llm: could not resolve %s: %v", c.model, err)
		if c.aliasID != "" {
			return c.aliasID // the last answer beats none
		}
		return c.model
	}
	c.aliasID, c.aliasAt = id, time.Now()
	return id
}

func (c *Client) resolveAlias(ctx context.Context) (string, error) {
	u := c.baseURL
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u+"/models?limit=100", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", shim.APIVersion)
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", statusError(resp)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	id, ok := shim.LatestIn(body, c.model)
	if !ok {
		return "", fmt.Errorf("no model in that line")
	}
	return id, nil
}
