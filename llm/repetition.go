package llm

import (
	"bytes"
	"strconv"
)

// Degenerate-repetition detection, and the reason it lives in the transport.
//
// A model that falls into a loop emits the same block of text over and over
// until something else stops it — and with MaxTokens unset, "something else" is
// the server's context limit. Measured: one generation spent ~30 MINUTES
// re-emitting an 85-character fragment of a legal property description. Nothing
// upstream cut it. `Stop` cannot: the loop's text is not known in advance.
// A grammar cannot: the output was well-formed the whole time. Only the party
// WATCHING THE BYTES ARRIVE can see it, which is readStream.
//
// The cut is the point. Closing the response body disconnects, and the server
// abandons the generation — so the detector does not just fail faster, it stops
// paying for tokens nobody will read.
//
// It watches BOTH channels. The measured case arrived inside a tool call's
// `arguments` (the fragment carried literal `\n` escapes, i.e. it was already
// JSON-encoded), streamed as tool_calls deltas — a detector on content deltas
// alone would have watched that 30 minutes go by.

// StopReason names why a stream ended other than the model finishing. Empty on
// a normal completion, so a consumer that ignores it behaves as before.
type StopReason string

// StopReasonRepetition means the client cut the stream because generation had
// collapsed into a repeating block. The output up to the cut is still valid;
// what follows would have been more copies of the same bytes.
const StopReasonRepetition StopReason = "repetition"

// RepetitionInfo describes the loop that was cut: enough to log it, to tell the
// model what it did, and to trim the redundant copies before they are persisted
// into the next context (which would otherwise TEACH the loop).
type RepetitionInfo struct {
	// Where the loop was detected: "content", or "tool:<name>" for a call's
	// arguments.
	Where string
	// Period is the length in bytes of the repeating block; Reps is how many
	// consecutive copies of it were seen.
	Period int
	Reps   int
	// Trailing is the byte count of redundant copies — (Reps-1)*Period — i.e.
	// how much can be trimmed from the end of the accumulated text while leaving
	// one intact copy of the block.
	Trailing int
	// Sample is one copy of the repeating block, capped for logging.
	Sample string
}

// String renders the finding for a log line or a message to the model.
func (r *RepetitionInfo) String() string {
	if r == nil {
		return ""
	}
	return "repeated a " + strconv.Itoa(r.Period) + "-character block " +
		strconv.Itoa(r.Reps) + " times in " + r.Where + ": " + strconv.Quote(r.Sample)
}

// RepetitionGuard configures degenerate-loop detection for a Client's streams.
// The zero value means "on, with defaults" — the failure it prevents is
// expensive and silent, so it is not opt-in. Set Off to disable it.
//
// The thresholds exist to separate a LOOP from legitimate repetition. Real
// output repeats constantly: markdown table rules, blank lines, boilerplate
// headers, near-identical rows. What a loop adds is scale — the same bytes,
// exactly, for a long span with nothing else between. Requiring several copies
// AND a non-trivial period AND a large total span is what makes a false
// positive cost real work to construct, while the measured case (an 85-byte
// period, hundreds of copies) trips within the first kilobyte.
type RepetitionGuard struct {
	// Off disables detection entirely.
	Off bool

	// MinPeriod is the shortest repeating block considered a loop (default 24).
	// Below this, ordinary formatting repeats freely.
	MinPeriod int
	// MaxPeriod is the longest block scanned for (default 1024). Cost is linear
	// in this, so it bounds the check rather than the phenomenon.
	MaxPeriod int
	// MinReps is how many consecutive identical copies are required (default 3).
	MinReps int
	// MinSpan is the minimum total bytes the repetition must cover (default
	// 512). This is what stops three copies of a short header from tripping it.
	MinSpan int
	// CheckEvery is how many newly-arrived bytes trigger a scan (default 256).
	// The scan is not run per delta: a delta is a token, and scanning per token
	// would multiply the cost by ~4 for no earlier detection than this.
	CheckEvery int
}

func (g RepetitionGuard) resolve() RepetitionGuard {
	if g.MinPeriod <= 0 {
		g.MinPeriod = 24
	}
	if g.MaxPeriod <= 0 {
		g.MaxPeriod = 1024
	}
	if g.MinReps <= 1 {
		g.MinReps = 3
	}
	if g.MinSpan <= 0 {
		g.MinSpan = 512
	}
	if g.CheckEvery <= 0 {
		g.CheckEvery = 256
	}
	return g
}

// window is how much history the scan needs: enough for MinReps copies of the
// longest period, and enough to reach MinSpan with a period one byte longer
// than the minimum.
func (g RepetitionGuard) window() int {
	n := g.MaxPeriod * g.MinReps
	if s := g.MinSpan + g.MaxPeriod; s > n {
		n = s
	}
	return n
}

// repWatcher accumulates one logical byte stream (the content channel, or one
// tool call's arguments) and reports the first degenerate repetition in it.
type repWatcher struct {
	cfg   RepetitionGuard
	where string
	tail  []byte
	since int
}

func newRepWatcher(cfg RepetitionGuard, where string) *repWatcher {
	return &repWatcher{cfg: cfg.resolve(), where: where}
}

// feed appends a delta and scans when enough new bytes have arrived. It returns
// non-nil exactly once the stream has collapsed into a loop; the caller cuts.
func (w *repWatcher) feed(s string) *RepetitionInfo {
	if w == nil || s == "" || w.cfg.Off {
		return nil
	}
	w.tail = append(w.tail, s...)
	if max := w.cfg.window(); len(w.tail) > max {
		// Keep the most recent window. Copy down rather than reslice so the
		// backing array does not grow without bound over a long generation.
		w.tail = w.tail[:copy(w.tail, w.tail[len(w.tail)-max:])]
	}
	w.since += len(s)
	if w.since < w.cfg.CheckEvery {
		return nil
	}
	w.since = 0
	period, reps, ok := periodicSuffix(w.tail, w.cfg)
	if !ok {
		return nil
	}
	sample := w.tail[len(w.tail)-period:]
	if len(sample) > 200 {
		sample = sample[:200]
	}
	return &RepetitionInfo{
		Where:    w.where,
		Period:   period,
		Reps:     reps,
		Trailing: (reps - 1) * period,
		Sample:   string(sample),
	}
}

// periodicSuffix reports whether tail ends in a block repeated back-to-back,
// returning the block's TRUE period and how many copies of it are present.
//
// The search is a plain ascending scan over candidate periods rather than a
// suffix-automaton trick, because a loop is a property of the SUFFIX only, and
// the comparison at each candidate almost always fails on its first byte —
// bytes.Equal short-circuits, so the real cost is closer to MaxPeriod than to
// the MaxPeriod*MinReps the shape suggests.
//
// The first match is then REDUCED to its minimal period, and that is what
// MinPeriod is tested against. Skipping this step made MinPeriod decorative:
// every multiple of a period is also a period, so a stream of "abab…" matched
// at p=24 and sailed past a MinPeriod of 24 — measured, as a false positive on
// blank lines and on indentation.
//
// The consequence is deliberate: a cycle SHORTER than MinPeriod (a flood of
// newlines, one repeated word) is not reported as a loop at all, at any length.
// That is the price of never firing on ordinary formatting, and such output is
// bounded by ChatOpts.MaxTokens rather than here.
func periodicSuffix(tail []byte, cfg RepetitionGuard) (period, reps int, ok bool) {
	n := len(tail)
	for p := cfg.MinPeriod; p <= cfg.MaxPeriod && p*cfg.MinReps <= n; p++ {
		if !bytes.Equal(tail[n-2*p:n-p], tail[n-p:]) {
			continue
		}
		// Only the FIRST match is examined. Any further match is a multiple of
		// the same underlying cycle, so it cannot change the verdict — and
		// scanning on would turn a degenerate stream into a quadratic one.
		d := minimalPeriod(tail[n-p:])
		if d < cfg.MinPeriod {
			return 0, 0, false
		}
		r := countReps(tail, d)
		if r >= cfg.MinReps && r*d >= cfg.MinSpan {
			return d, r, true
		}
		return 0, 0, false
	}
	return 0, 0, false
}

// countReps returns how many consecutive copies of tail's last p bytes end it.
func countReps(tail []byte, p int) int {
	n, last, r := len(tail), tail[len(tail)-p:], 1
	for (r+1)*p <= n && bytes.Equal(tail[n-(r+1)*p:n-r*p], last) {
		r++
	}
	return r
}

// minimalPeriod returns the length of the shortest block whose repetition
// builds b exactly — b itself when b is not internally periodic.
func minimalPeriod(b []byte) int {
	n := len(b)
	for d := 1; d <= n/2; d++ {
		if n%d != 0 {
			continue
		}
		if bytes.Equal(b[:n-d], b[d:]) {
			return d
		}
	}
	return n
}
