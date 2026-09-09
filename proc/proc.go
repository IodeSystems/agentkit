// Package proc is a long-running thing that produces output.
//
// A shell command and a child agent are the same shape: something started that
// keeps going, emits a stream while it does, can be asked about while it runs,
// and eventually settles with a result. dun has both and wrote them twice —
// `bgjob.go` for commands and `subagent.go` for children, each with its own
// monitor tool taking the same (id, tail) and each buffering output its own way.
// Nothing here is specific to either.
//
// What this owns is the part that is genuinely hard and was solved once already:
// where the output goes, how much of it a model is allowed to hear, and when.
//
//   - Output goes to a FILE and the path goes to the caller. A 50k-line build
//     log is grep-able; the same log spent on a context window is a session
//     ended. The inline tail is bounded and the file has the rest.
//   - Reporting is LINE-COMPLETE. Half a line tells a model nothing and a
//     pattern cannot judge one, so bytes since the last newline are held back.
//   - Silence is the DEFAULT. A process nobody asked about does not speak.
//     Streaming without a bound is worse than silence: an unfiltered build
//     narrates itself into the window one line at a time.
//
// Signals, exit status and the shape of a result belong to whoever runs the
// thing, so they arrive through Settle rather than being modelled here.
package proc

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"sync"
	"time"
)

// TailBytes is how much of the output an inline read carries — enough to see a
// failure, with the file holding the rest.
const TailBytes = 4000

// EarlyMax bounds what is kept in memory before a log file exists.
//
// A process may start before anyone has decided to keep it, and its output is
// usually also accumulating somewhere else for the caller that started it. Past
// the cap the log simply begins later rather than growing without limit.
const EarlyMax = 1 << 20

// Monitor is what a process is allowed to say while it runs.
//
// The zero value is silence until it settles, which is the right default: a
// process nobody asked about should not spend context.
type Monitor struct {
	// BufferBytes reports once this many matched-but-unreported bytes have
	// piled up. Zero never reports progress.
	BufferBytes int
	// Grep limits reports to lines matching it. Empty keeps every line.
	Grep string
	// Ignore mutes the process entirely, including its completion — for one
	// whose answer stopped mattering.
	Ignore bool
}

// Proc is one running thing: its log, what it may say, and what it has produced.
//
// Safe for concurrent use: Write is called from whatever is producing output
// while a monitor read may arrive from anywhere else.
type Proc struct {
	// ID addresses this process in a monitor tool.
	ID int
	// Kind is what sort of thing it is — "job", "agent" — carried for display
	// and never interpreted here.
	Kind string
	// Label is how a person recognises it: a command line, a child's prompt.
	Label string
	// LogPath is where the full output is, and is what the caller hands the
	// model instead of the output itself.
	LogPath string

	started time.Time

	mu      sync.Mutex
	mon     Monitor
	re      *regexp.Regexp
	log     *os.File
	early   []byte // held in memory until a log file exists
	partial []byte // bytes since the last newline; not reportable yet
	pending []byte // matched whole lines not yet reported
	written int64
	rows    *Rows
	extract Extract
	done    bool
	settled bool
	result  string
	ended   time.Time
}

// New starts tracking a process. logPath may be empty, in which case output is
// held in memory until OpenLog is called.
func New(id int, kind, label, logPath string, now time.Time) *Proc {
	return &Proc{ID: id, Kind: kind, Label: label, LogPath: logPath, started: now}
}

// Collect turns this process's output into rows as well as text.
//
// The extract runs on the same complete lines the monitor sees, so a process is
// never parsed twice and a half-line is never handed to a parser. Passing a nil
// Rows stops collection without disturbing the text side.
func (p *Proc) Collect(rows *Rows, e Extract) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.rows, p.extract = rows, e
}

// Rows is the structured side, or nil when nothing is collecting.
func (p *Proc) Rows() *Rows {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.rows
}

// Watch replaces the monitor policy and returns an error only when the pattern
// will not compile — a bad pattern is the caller's mistake and must not
// silently become "match everything".
func (p *Proc) Watch(m Monitor) error {
	var re *regexp.Regexp
	if m.Grep != "" {
		c, err := regexp.Compile(m.Grep)
		if err != nil {
			return fmt.Errorf("proc: watch pattern %q: %w", m.Grep, err)
		}
		re = c
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.mon, p.re = m, re
	return nil
}

// OpenLog begins writing to path, flushing whatever was held in memory.
func (p *Proc) OpenLog(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.early) > 0 {
		_, _ = f.Write(p.early)
		p.early = nil
	}
	p.log, p.LogPath = f, path
	return nil
}

// Write takes output as it is produced. It returns the progress note that is
// now due, or "" — the caller sends it, OUTSIDE any lock of its own, because
// notifying usually reaches a queue or a UI and nothing good comes of holding a
// process's lock across either.
func (p *Proc) Write(b []byte) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.log != nil {
		_, _ = p.log.Write(b)
	} else if len(p.early) < EarlyMax {
		p.early = append(p.early, b...)
	}
	p.written += int64(len(b))

	p.partial = append(p.partial, b...)
	if i := bytes.LastIndexByte(p.partial, '\n'); i >= 0 {
		whole := p.partial[:i+1]
		p.partial = append([]byte(nil), p.partial[i+1:]...)
		p.collect(whole)
		p.pending = append(p.pending, p.keep(whole)...)
	}
	return p.report(false, time.Now())
}

// collect extracts rows from whole lines. An extract that fails on one line
// must not stop the process: its output is still going to the log, and a parser
// that cannot read one line of a build is the ordinary case rather than an
// error.
func (p *Proc) collect(b []byte) {
	if p.rows == nil || p.extract == nil {
		return
	}
	for _, line := range bytes.SplitAfter(b, []byte("\n")) {
		line = bytes.TrimRight(line, "\n")
		if len(line) == 0 {
			continue
		}
		if row, ok := p.extract(line); ok {
			_ = p.rows.Append(row)
		}
	}
}

// keep filters whole lines down to what the monitor asked for.
func (p *Proc) keep(b []byte) []byte {
	if p.re == nil {
		return b
	}
	var out []byte
	for _, line := range bytes.SplitAfter(b, []byte("\n")) {
		if len(line) > 0 && p.re.Match(line) {
			out = append(out, line...)
		}
	}
	return out
}

// report returns the note that is due and clears what it reports. force ignores
// the byte threshold, which is what a monitor read wants: whatever is there now.
func (p *Proc) report(force bool, now time.Time) string {
	if p.mon.Ignore || len(p.pending) == 0 {
		return ""
	}
	if !force && (p.mon.BufferBytes <= 0 || len(p.pending) < p.mon.BufferBytes) {
		return ""
	}
	body := bytes.TrimRight(p.pending, "\n")
	p.pending = nil
	return fmt.Sprintf("%s #%d — %s (still running, %s):\n%s",
		p.Kind, p.ID, p.Label, now.Sub(p.started).Round(time.Second), body)
}

// Settle records the outcome. The second caller gets false: a result is claimed
// once, by whoever is waiting for it.
func (p *Proc) Settle(result string, now time.Time) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.settled {
		return false
	}
	p.done, p.settled, p.result, p.ended = true, true, result, now
	if p.log != nil {
		_ = p.log.Close()
		p.log = nil
	}
	return true
}

// Done reports whether the process has finished.
func (p *Proc) Done() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.done
}
