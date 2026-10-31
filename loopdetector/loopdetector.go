// Package loopdetector answers one question about a model's output: is it going
// round in a loop, and how many times has it gone round.
//
// llm.RepetitionGuard is on by default and catches one shape, the same bytes
// back to back. Loops seen in practice also come with random data between the
// copies, or go round ten or more variations before repeating, and those have
// no period for it to find. This package finds them, for a host that installs it.
//
// THE INPUT IS THE SESSION PLUS THE STREAM. Begin takes the messages the request
// was sent with, and reads the turn since the last user message; Watch takes the reply's deltas as they arrive, thinking
// included. There is no cache to size or to go stale, and nothing lives past
// the reply: what the detector compares against is the conversation itself,
// which any party that can see the request already holds, the corrallm proxy
// included. The history has no thinking in it (clients do not send it back), so
// a loop in thinking is found within the reply being built.
//
// THE OUTPUT IS A COUNT. Loops says how many times the output has gone round;
// Config.Tolerance says at what count Watch asks for the stream to be cut. A
// host that wants to watch before it trusts this sets a negative Tolerance and
// reads the Counters.
//
// Only output being built can be interrupted. A session whose history already
// loops is counted, but a reply is cut only once it continues the loop: a model
// told to stop must be given the reply in which to stop.
//
// Four detectors feed the count:
//
//	exact   the same block back to back: llm.RepetitionGuard.
//	cycle   the same sentences recurring, in any order, around other output.
//	phrase  the same, reworded a little. Words are stemmed and sentences matched
//	        by BM25: word order and word forms do not matter, synonyms do. BM25
//	        rather than a count of shared words, because it discounts what every
//	        sentence has: rows of a table share almost all their words and
//	        differ in a few, and it is those few that decide the score.
//	tool    the same call returning the same result. A result that changed
//	        is progress, so the count for that call starts again.
package loopdetector

import (
	"sync/atomic"

	"github.com/iodesystems/agentkit/llm"
)

// Kinds of loop, as reported in llm.RepetitionInfo.Kind and the Counters.
const (
	KindExact  = "exact"
	KindCycle  = "cycle"
	KindPhrase = "phrase"
	KindTool   = llm.KindTool
)

// Config is the detector's settings. The zero value is usable.
type Config struct {
	// Off disables detection: a Session counts nothing and interrupts nothing.
	Off bool

	// Tolerance is the loop count at which Watch returns a finding, which cuts
	// the stream. 0 → 3; negative → never, the count is only reported.
	Tolerance int

	// ToolTolerance is Tolerance for the tool detector, for a host that refuses
	// repeated calls itself and wants text cut but tools only counted. 0 →
	// Tolerance; negative → never.
	ToolTolerance int

	// Exact configures the exact detector.
	Exact llm.RepetitionGuard

	// Window is how many recent sentences the cycle and phrase detectors look at
	// (default 24), and MinShare the share of them that must be repeats (default
	// 0.6). What is not a repeat is the room left for other output between the
	// copies. Measured on 1,315 sessions of production traffic: 0.75 finds 3
	// sessions, 0.5 finds 6, and 0.5 is also where a document that alternates a
	// boilerplate line with a distinct one reaches the bar.
	Window   int
	MinShare float64
	// MinSpan is the bytes the Window must cover (default 1024), so that a run
	// of short lines does not count as a loop.
	MinSpan int
	// MinTokens is the fewest words that make a sentence (default 4). Shorter
	// ones, closing braces and "Let me check.", are not evidence of anything.
	MinTokens int
	// MaxSegment splits text with no sentence end in it (default 512 bytes).
	MaxSegment int
	// Similarity is the BM25 score, as a share of a sentence's score against
	// itself, at which two sentences are the same (default 0.8). Words are
	// stemmed first. Below 0.8 a list whose lines differ only in a number is
	// taken for one line said thirty times. Negative turns the phrase detector
	// off.
	Similarity float64

	// SampleBytes caps the First, Last and Sample of a finding (default 200).
	SampleBytes int

	// ToolWindow is how many recent tool calls the tool detector looks at
	// (default 12). MinShare applies to them too. No tool is exempt: polling for
	// a result that has not changed is the waste this finds. A host waits on an
	// event instead.
	ToolWindow int
}

func (c Config) resolve() Config {
	if c.Tolerance == 0 {
		c.Tolerance = 3
	}
	if c.Window <= 0 {
		c.Window = 24
	}
	if c.MinShare <= 0 {
		c.MinShare = 0.6
	}
	if c.MinSpan <= 0 {
		c.MinSpan = 1024
	}
	if c.MinTokens <= 0 {
		c.MinTokens = 4
	}
	if c.MaxSegment <= 0 {
		c.MaxSegment = 512
	}
	if c.Similarity == 0 {
		c.Similarity = 0.8
	}
	if c.SampleBytes <= 0 {
		c.SampleBytes = 200
	}
	if c.ToolWindow <= 0 {
		c.ToolWindow = 12
	}
	return c
}

func (d *LoopDetector) toleranceFor(kind string) int {
	if kind == KindTool && d.cfg.ToolTolerance != 0 {
		return d.cfg.ToolTolerance
	}
	return d.cfg.Tolerance
}

// LoopDetector holds the settings and the counters. It is safe for concurrent
// use; each reply gets its own Session from Begin.
type LoopDetector struct {
	cfg Config

	sessions, bytes, sentences, repeats atomic.Int64
	detected, interrupted               [4]atomic.Int64 // by kind, see kindIndex
}

// New returns a detector with cfg's settings.
func New(cfg Config) *LoopDetector {
	return &LoopDetector{cfg: cfg.resolve()}
}

// Install makes the detector watch every stream of c, in place of c.Repetition.
func (d *LoopDetector) Install(c *llm.Client) {
	c.Watch = func(history []llm.Message) llm.StreamWatcher { return d.Begin(history) }
}

// Counters are totals since New, for a host to export as metrics.
type Counters struct {
	// Sessions is how many replies were watched. Bytes and Sentences are how much
	// of them was; Repeats is how many sentences matched an earlier one.
	Sessions, Bytes, Sentences, Repeats int64
	// Detected counts sessions, by kind, in which a loop was found, whatever the
	// Tolerance. Interrupted counts those where Watch asked for a cut.
	Detected, Interrupted map[string]int64
}

var kinds = [...]string{KindExact, KindCycle, KindPhrase, KindTool}

func kindIndex(kind string) int {
	for i, k := range kinds {
		if k == kind {
			return i
		}
	}
	return 0
}

// Counters returns the totals so far.
func (d *LoopDetector) Counters() Counters {
	c := Counters{
		Sessions:    d.sessions.Load(),
		Bytes:       d.bytes.Load(),
		Sentences:   d.sentences.Load(),
		Repeats:     d.repeats.Load(),
		Detected:    map[string]int64{},
		Interrupted: map[string]int64{},
	}
	for i, k := range kinds {
		c.Detected[k] = d.detected[i].Load()
		c.Interrupted[k] = d.interrupted[i].Load()
	}
	return c
}
