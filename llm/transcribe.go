package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"strings"
)

// transcribeURL returns the speech-to-text endpoint (sibling of chatURL).
func (c *Client) transcribeURL() string {
	u := c.baseURL
	if !strings.HasSuffix(u, "/v1") {
		u += "/v1"
	}
	return u + "/audio/transcriptions"
}

// Transcribe returns the text of one audio clip from an OpenAI-compatible
// /v1/audio/transcriptions endpoint. model is passed explicitly because speech
// uses a different model than chat (the Client's configured model is the chat
// model). filename carries the container type to servers that sniff it from the
// extension (e.g. "memo.m4a"). Honors the same auth + retry policy as Chat.
func (c *Client) Transcribe(ctx context.Context, model, filename string, audio []byte) (string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("model", model); err != nil {
		return "", fmt.Errorf("llm: transcribe form: %w", err)
	}
	if err := mw.WriteField("response_format", "json"); err != nil {
		return "", fmt.Errorf("llm: transcribe form: %w", err)
	}
	fw, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return "", fmt.Errorf("llm: transcribe form: %w", err)
	}
	if _, err := fw.Write(audio); err != nil {
		return "", fmt.Errorf("llm: transcribe form: %w", err)
	}
	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("llm: transcribe form: %w", err)
	}
	resp, err := c.postTypedWithRetry(ctx, c.transcribeURL(), mw.FormDataContentType(), body.Bytes(), callIdent{})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", statusError(resp)
	}
	var result struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("llm: decode transcription: %w", err)
	}
	return result.Text, nil
}
