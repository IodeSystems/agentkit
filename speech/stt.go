// Package speech — STT stream.
//
// STT opens a Deepgram /v1/listen WebSocket, accepts audio frames
// (16-bit PCM by default), and delivers transcript events on a
// channel. The read loop runs in a background goroutine; the host
// writes audio from its own goroutine and reads events from the
// Events() channel.
//
// Usage() records the total stream duration in seconds and the model
// name, so a host can bill per-minute without re-parsing the events.

package speech

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"
)

// STTEvent is one transcript event delivered on the Events() channel.
type STTEvent struct {
	// Type is one of: "partial", "final", "utterance_end",
	// "speech_started", "speech_stopped", "metadata".
	Type string

	// Transcript is the recognised text (empty for event types
	// that carry no transcript).
	Transcript string

	// Confidence is the model's confidence in the transcript,
	// 0.0–1.0 (0 for event types that carry no transcript).
	Confidence float64

	// Duration is the stream duration in seconds at the time of
	// this event (0 for event types that carry no duration).
	Duration float64
}

// STTUsage is the cost-metering record for a completed STT stream.
type STTUsage struct {
	// Duration is the total stream duration in seconds.
	Duration float64
	// Model is the model name that served the stream (for
	// example "nova-2"), as reported by the provider.
	Model string
}

// STTOptions configures the STT stream.
type STTOptions struct {
	// Model is the Deepgram model name. Empty means "nova-2".
	Model string
	// SampleRate is the audio sample rate in Hz. Zero means 16000.
	SampleRate int
	// Encoding is the audio encoding. Empty means "linear16".
	Encoding string
	// Punctuate enables automatic punctuation.
	Punctuate bool
	// Interim enables partial (interim) results in addition to
	// finals.
	Interim bool
}

func (o STTOptions) normalise() STTOptions {
	// An all-zero options struct means "use the defaults", and
	// the default is interim results on. Check the original
	// values before filling in the defaults.
	isZero := o.Model == "" && o.SampleRate == 0 && o.Encoding == "" &&
		!o.Punctuate && !o.Interim

	n := o
	if n.Model == "" {
		n.Model = "nova-2"
	}
	if n.SampleRate == 0 {
		n.SampleRate = 16000
	}
	if n.Encoding == "" {
		n.Encoding = "linear16"
	}
	if isZero {
		n.Interim = true
	}
	return n
}

// STT is one open Deepgram STT session.
//
// Write sends audio frames to the provider; Events returns the
// channel of transcript events; Usage returns the cost-metering
// record; Close sends the WS close frame and waits for the read
// loop to finish.
//
// A STT is safe for concurrent use of Write and the read-only
// methods (Events, Usage, Final, Done); the read loop owns the
// stream.
type STT struct {
	stream *Stream
	events chan STTEvent
	done   chan struct{} // closed when the read loop exits

	mu     sync.Mutex
	usage  STTUsage
	final  string // last final transcript
	closed bool
}

// OpenSTT opens a Deepgram STT WebSocket at /v1/listen and starts
// the read loop. The returned *STT is ready for Write calls.
//
// The context is used only for the dial; the stream outlives it.
func OpenSTT(ctx context.Context, c *Client, opts STTOptions) (*STT, error) {
	o := opts.normalise()

	q := url.Values{}
	q.Set("model", o.Model)
	q.Set("encoding", o.Encoding)
	q.Set("sample_rate", fmt.Sprintf("%d", o.SampleRate))
	if o.Punctuate {
		q.Set("punctuate", "true")
	}
	q.Set("interim_results", fmt.Sprintf("%t", o.Interim))

	stream, err := c.Open(ctx, "/v1/listen?"+q.Encode())
	if err != nil {
		return nil, err
	}

	s := &STT{
		stream: stream,
		events: make(chan STTEvent, 64),
		done:   make(chan struct{}),
	}

	go s.readLoop()

	return s, nil
}

// Write sends one audio frame to the provider. Audio is expected to
// be 16-bit little-endian PCM at the configured sample rate.
func (s *STT) Write(audio []byte) error {
	return s.stream.Send(audio)
}

// Events returns the channel of transcript events. The channel is
// closed when the stream ends (by Close, by a peer close frame, or
// by a read error).
func (s *STT) Events() <-chan STTEvent {
	return s.events
}

// Usage returns the cost-metering record for this stream. It is
// safe to call at any time; the values reflect the state of the
// stream as of the last event processed.
func (s *STT) Usage() STTUsage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage
}

// Final returns the last final transcript received. It is empty
// until a final Results event has been delivered.
func (s *STT) Final() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.final
}

// Done is closed when the read loop exits.
func (s *STT) Done() <-chan struct{} {
	return s.done
}

// Close sends the WS close frame and waits for the read loop to
// finish. It is idempotent.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()

	err := s.stream.Close()
	<-s.done
	return err
}

// readLoop reads messages from the stream until it ends, parses the
// JSON envelopes, and sends STTEvents on the events channel. It
// closes the events and done channels on the way out.
func (s *STT) readLoop() {
	defer close(s.events)
	defer close(s.done)

	for {
		op, payload, err := s.stream.ReadMessage()
		if err != nil {
			return
		}
		if op != opText {
			// Binary frames from the server are not part of the
			// STT protocol; skip them.
			continue
		}

		ev, model, ok := parseSTTEnvelope(payload)
		if !ok {
			continue
		}

		s.mu.Lock()
		if model != "" {
			s.usage.Model = model
		}
		if ev.Duration > s.usage.Duration {
			s.usage.Duration = ev.Duration
		}
		if ev.Type == "final" && ev.Transcript != "" {
			s.final = ev.Transcript
		}
		s.mu.Unlock()

		s.events <- ev
	}
}

// --- Deepgram JSON envelope parsing ---

// dgResults is a Deepgram Results envelope (both partial and final;
// SpeechFinal distinguishes them).
type dgResults struct {
	Type        string  `json:"type"`
	SpeechFinal bool    `json:"speech_final"`
	Duration    float64 `json:"duration"`
	Model       string  `json:"model"`
	Channel     *dgChan `json:"channel"`
}

type dgChan struct {
	Alternatives []dgAlt `json:"alternatives"`
}

type dgAlt struct {
	Transcript string  `json:"transcript"`
	Confidence float64 `json:"confidence"`
}

// dgMetadata is a Deepgram Metadata envelope.
type dgMetadata struct {
	Type     string  `json:"type"`
	Model    string  `json:"model"`
	Duration float64 `json:"duration"`
}

// dgUtteranceEnd is a Deepgram UtteranceEnd envelope.
type dgUtteranceEnd struct {
	Type     string  `json:"type"`
	Duration float64 `json:"duration"`
}

// dgSpeechEnded is a Deepgram SpeechEnded envelope.
type dgSpeechEnded struct {
	Type     string  `json:"type"`
	Duration float64 `json:"duration"`
}

// parseSTTEnvelope decodes one Deepgram JSON envelope into an
// STTEvent and the model name carried by the envelope (which is not
// part of the public STTEvent). The second-to-last return value is
// false when the payload is not a recognised event type; the caller
// should skip it.
func parseSTTEnvelope(payload []byte) (ev STTEvent, model string, ok bool) {
	// Dispatch on the type field without decoding the whole
	// struct twice.
	var peek struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(payload, &peek); err != nil {
		return STTEvent{}, "", false
	}

	switch peek.Type {
	case "Results":
		var r dgResults
		if err := json.Unmarshal(payload, &r); err != nil {
			return STTEvent{}, "", false
		}
		ev = STTEvent{
			Type:     "final",
			Duration: r.Duration,
		}
		if !r.SpeechFinal {
			ev.Type = "partial"
		}
		if r.Channel != nil && len(r.Channel.Alternatives) > 0 {
			ev.Transcript = r.Channel.Alternatives[0].Transcript
			ev.Confidence = r.Channel.Alternatives[0].Confidence
		}
		return ev, r.Model, true

	case "Metadata":
		var m dgMetadata
		if err := json.Unmarshal(payload, &m); err != nil {
			return STTEvent{}, "", false
		}
		return STTEvent{
			Type:     "metadata",
			Duration: m.Duration,
		}, m.Model, true

	case "UtteranceEnd":
		var u dgUtteranceEnd
		if err := json.Unmarshal(payload, &u); err != nil {
			return STTEvent{}, "", false
		}
		return STTEvent{
			Type:     "utterance_end",
			Duration: u.Duration,
		}, "", true

	case "SpeechStarted":
		return STTEvent{Type: "speech_started"}, "", true

	case "SpeechEnded":
		var se dgSpeechEnded
		if err := json.Unmarshal(payload, &se); err != nil {
			return STTEvent{}, "", false
		}
		return STTEvent{
			Type:     "speech_stopped",
			Duration: se.Duration,
		}, "", true

	default:
		return STTEvent{}, "", false
	}
}
