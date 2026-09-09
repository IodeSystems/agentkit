package proc

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func at(s int) time.Time { return time.Unix(1_700_000_000+int64(s), 0) }

/*
Half a line is not a report.

Output arrives in whatever chunks the producer happens to write, and a pattern
cannot judge a fragment: `go build` emitting "inter" is not an error, and the
model reading it learns nothing it can act on. So bytes since the last newline
are held back until the line finishes.
*/
func TestOnlyWholeLinesAreReported(t *testing.T) {
	p := New(1, "job", "go build", "", at(0))
	if err := p.Watch(Monitor{BufferBytes: 1}); err != nil {
		t.Fatal(err)
	}
	if note := p.Write([]byte("half a li")); note != "" {
		t.Errorf("a fragment was reported: %q", note)
	}
	note := p.Write([]byte("ne\n"))
	if !strings.Contains(note, "half a line") {
		t.Errorf("the finished line was not reported: %q", note)
	}
	if strings.Count(note, "half a line") != 1 {
		t.Errorf("the line was reported more than once: %q", note)
	}
}

/*
Silence is the default.

A process nobody asked about does not speak. Streaming without a bound is worse
than silence — an unfiltered build narrates itself into the window one line at a
time — so progress is reported only once a caller has said how much output is
worth interrupting for.
*/
func TestAProcessNobodyAskedAboutSaysNothing(t *testing.T) {
	p := New(1, "job", "go build", "", at(0))
	for i := 0; i < 50; i++ {
		if note := p.Write([]byte("chatter chatter chatter\n")); note != "" {
			t.Fatalf("an unwatched process spoke: %q", note)
		}
	}
}

/*
A pattern keeps the lines that were asked for, and nothing else.
*/
func TestAWatchPatternFiltersWhatIsReported(t *testing.T) {
	p := New(1, "job", "suite", "", at(0))
	if err := p.Watch(Monitor{BufferBytes: 1, Grep: "^FAIL"}); err != nil {
		t.Fatal(err)
	}
	note := p.Write([]byte("ok pkg/a\nFAIL pkg/b\nok pkg/c\n"))
	if !strings.Contains(note, "FAIL pkg/b") {
		t.Errorf("the matching line was dropped: %q", note)
	}
	if strings.Contains(note, "ok pkg/a") {
		t.Errorf("a non-matching line was reported: %q", note)
	}
	// A pattern that will not compile is the caller's mistake and must not
	// quietly become "match everything".
	if err := p.Watch(Monitor{Grep: "("}); err == nil {
		t.Error("an invalid pattern was accepted")
	}
}

/*
Ignore mutes a process completely, including its completion.
*/
func TestIgnoreSilencesEvenCompletion(t *testing.T) {
	p := New(1, "job", "superseded", "", at(0))
	if err := p.Watch(Monitor{BufferBytes: 1, Ignore: true}); err != nil {
		t.Fatal(err)
	}
	if note := p.Write([]byte("still going\n")); note != "" {
		t.Errorf("a muted process spoke: %q", note)
	}
}

/*
A result is claimed once.

Two things race for it — the call that started the process and whatever is
monitoring it — and both acting on the same completion is how a job gets
reported twice.
*/
func TestAResultIsClaimedOnce(t *testing.T) {
	p := New(1, "job", "x", "", at(0))
	if !p.Settle("exit 0", at(5)) {
		t.Fatal("the first claim was refused")
	}
	if p.Settle("exit 0", at(5)) {
		t.Error("a second caller also claimed the result")
	}
	if !p.Done() {
		t.Error("a settled process does not report itself done")
	}
}

/*
Output goes to a file, and the tail comes back from it.

The point of the log is that asking about a long-running thing costs the same
whether it has produced ten lines or fifty thousand. Held-in-memory output is
flushed into the file when one is opened, so nothing produced before that is
lost.
*/
func TestOutputGoesToTheLogAndTheTailComesBack(t *testing.T) {
	dir := t.TempDir()
	p := New(1, "job", "build", "", at(0))
	p.Write([]byte("early line\n"))
	if err := p.OpenLog(filepath.Join(dir, "job.log")); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 500; i++ {
		p.Write([]byte("a line of output that goes on for a while\n"))
	}
	p.Write([]byte("LAST\n"))

	tail := p.Tail(200)
	if !strings.Contains(tail, "LAST") {
		t.Errorf("the tail does not reach the end: %q", tail)
	}
	if len(tail) > 220 {
		t.Errorf("the tail is %d bytes, past what was asked for", len(tail))
	}
	// Cut to a line boundary: a tail that opens mid-line reads as corruption.
	if strings.HasPrefix(tail, "line of") || strings.HasPrefix(tail, " line") {
		t.Errorf("the tail opens mid-line: %q", tail[:40])
	}
	// And the early output survived into the file.
	if !strings.Contains(p.Tail(1<<20), "early line") {
		t.Error("output produced before the log existed was lost")
	}
}
