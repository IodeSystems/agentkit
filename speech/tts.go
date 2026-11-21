package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"
)

// ttsDefaultModel is the model used when a caller does not name one.
const ttsDefaultModel = "aura-2-thalia-en"

// ttsStart is the first text message the TTS endpoint sends on the
// WebSocket, confirming the connection and reporting the model that
// will synthesize.
type ttsStart struct {
	Type  string `json:"type"`
	Model string `json:"model"`
}

// ttsText is the envelope a caller sends to ask the endpoint to
// synthesize text.
type ttsText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// TTSConfig configures a TTS stream.
type TTSConfig struct {
	// Model is the Deepgram TTS model (for example
	// "aura-asteria-en"). It is sent in the /speak query string and
	// is expected back in the start event. Empty means ttsDefaultModel.
	Model string
}

// TTSUsage is the per-stream billing record for a TTS stream.
type TTSUsage struct {
	// Chars is the number of UTF-8 characters of text sent for
	// synthesis. Deepgram bills TTS by characters.
	Chars int
	// Model is the model the server reported in its start event. It is
	// empty until the stream has started.
	Model string
}

// TTSStream is one open TTS session against the provider's /speak
// WebSocket. Text is sent with SendText; the synthesized audio comes
// out of Read, which satisfies io.Reader so the audio can be piped
// straight into a decoder or a file.
//
// The first Read drives the protocol handshake with the server: it
// blocks until the start event arrives (which is when the model name
// becomes known) and returns the first chunk of audio. A host that
// wants the start event to happen eagerly can call Start first.
//
// A TTSStream is not safe for concurrent use; a host reads and writes
// it from the goroutine that owns the session.
type TTSStream struct {
	stream *Stream

	buf  bytes.Buffer // audio frames, not yet handed to Read
	sent int
	done bool

	model   string
	modelOK bool
}

// OpenTTS dials the provider's /speak endpoint and returns a TTSStream.
// The model is taken from cfg.Model (ttsDefaultModel when empty). The
// stream is ready to use immediately; the start event is read on the
// first Start or Read.
func (c *Client) OpenTTS(ctx context.Context, cfg TTSConfig) (*TTSStream, error) {
	model := cfg.Model
	if model == "" {
		model = ttsDefaultModel
	}
	stream, err := c.Open(ctx, "/speak?model="+model)
	if err != nil {
		return nil, err
	}
	return &TTSStream{
		stream:  stream,
		model:   model,
		modelOK: false,
	}, nil
}

// Model returns the model name. It is the model that was requested
// until the server's start event names the one that will synthesize,
// which then replaces it.
func (s *TTSStream) Model() string {
	return s.model
}

// Usage returns the stream's billing record: the number of characters
// of text sent for synthesis and the model the server reported.
func (s *TTSStream) Usage() TTSUsage {
	return TTSUsage{Chars: s.sent, Model: s.model}
}

// Start drives the protocol handshake: it reads the server's start
// event and records the model it names. Binary frames that arrive
// before the start event (the server may interleave) are buffered
// for Read. Subsequent Start calls return nil; a stream that has
// already closed returns nil.
func (s *TTSStream) Start() error {
	if s.modelOK || s.done {
		return nil
	}
	for !s.modelOK && !s.done {
		op, payload, err := s.stream.ReadMessage()
		if err != nil {
			s.done = true
			var cerr *CloseError
			if errors.As(err, &cerr) || s.buf.Len() > 0 {
				return nil
			}
			return err
		}
		if op == opBinary {
			if _, err := s.buf.Write(payload); err != nil {
				return err
			}
			continue
		}
		var env ttsStart
		if err := json.Unmarshal(payload, &env); err != nil {
			return fmt.Errorf("speech: parse tts event: %w", err)
		}
		if env.Type == "start" {
			if env.Model != "" {
				s.model = env.Model
			}
			s.modelOK = true
			return nil
		}
		if env.Type == "close" {
			s.done = true
			return nil
		}
	}
	return nil
}

// SendText asks the endpoint to synthesize text. The text is wrapped
// in the provider's envelope and its character count is added to the
// stream's billing record.
func (s *TTSStream) SendText(text string) error {
	env, err := json.Marshal(ttsText{Type: "text", Text: text})
	if err != nil {
		return err
	}
	if err := s.stream.SendText(string(env)); err != nil {
		return err
	}
	s.sent += utf8.RuneCountInString(text)
	return nil
}

// Read implements io.Reader over the synthesized audio. It returns
// whatever audio has arrived, reading from the server as needed; when
// the server closes the stream (or the connection ends) it returns
// io.EOF after the last audio bytes have been delivered.
func (s *TTSStream) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if s.buf.Len() > 0 {
		n, _ := s.buf.Read(p)
		return n, nil
	}
	// Buffer is empty: pull the next message from the server.
	for s.buf.Len() == 0 && !s.done {
		op, payload, err := s.stream.ReadMessage()
		if err != nil {
			if s.buf.Len() > 0 {
				break
			}
			s.done = true
			var cerr *CloseError
			if errors.As(err, &cerr) {
				return 0, io.EOF
			}
			return 0, err
		}
		if op == opBinary {
			if _, err := s.buf.Write(payload); err != nil {
				return 0, err
			}
			break
		}
		var env ttsStart
		if err := json.Unmarshal(payload, &env); err != nil {
			return 0, fmt.Errorf("speech: parse tts event: %w", err)
		}
		if env.Type == "start" {
			if env.Model != "" {
				s.model = env.Model
			}
			s.modelOK = true
		}
		if env.Type == "close" {
			s.done = true
		}
	}
	if s.buf.Len() > 0 {
		n, _ := s.buf.Read(p)
		return n, nil
	}
	if s.done {
		return 0, io.EOF
	}
	return 0, nil
}

// Close sends a close frame to the provider and tears down the
// connection. It is safe to call more than once, and it is safe to
// call after the server has already closed the stream.
func (s *TTSStream) Close() error {
	s.done = true
	return s.stream.Close()
}
