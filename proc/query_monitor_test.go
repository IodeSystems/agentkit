package proc

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func collecting(t *testing.T) (*Proc, *Rows) {
	t.Helper()
	rows, err := OpenRows(filepath.Join(t.TempDir(), "r.tsv"), "level text, pkg text")
	if err != nil {
		t.Fatal(err)
	}
	p := New(1, "job", "suite", "", at(0))
	p.Collect(rows, suiteExtract)
	return p, rows
}

/*
A query monitor signals on the ANSWER changing, not on the output.

`Monitor` reports lines: every line matching a pattern, once enough pile up.
That is volume dressed as a signal — a fifteen-minute suite matching ^FAIL four
hundred times narrates itself into the window. What is worth interrupting for is
that the failure COUNT moved.
*/
func TestAQuerySignalsOnlyWhenItsAnswerChanges(t *testing.T) {
	needSQLite(t)
	p, _ := collecting(t)
	if err := p.WatchQuery(QueryMonitor{
		Name: "failures", SQL: "select count(*) from rows where level='FAIL';"}); err != nil {
		t.Fatal(err)
	}

	// The first poll PRIMES: there is no change yet, and reporting the starting
	// value as news would fire every monitor once for nothing.
	p.Write([]byte("PASS pkg/a\n"))
	if notes := p.Poll(at(1)); len(notes) != 0 {
		t.Errorf("the priming poll spoke: %v", notes)
	}

	// More passing rows do not change the failure count.
	p.Write([]byte("PASS pkg/b\nPASS pkg/c\n"))
	if notes := p.Poll(at(2)); len(notes) != 0 {
		t.Errorf("an unchanged answer was reported: %v", notes)
	}

	// A failure does.
	p.Write([]byte("FAIL pkg/d\n"))
	notes := p.Poll(at(3))
	if len(notes) != 1 {
		t.Fatalf("a changed answer produced %d notes: %v", len(notes), notes)
	}
	if !strings.Contains(notes[0], "failures") || !strings.Contains(notes[0], "was") {
		t.Errorf("the note does not say what changed from what: %q", notes[0])
	}
}

/*
A burst and a dribble cost the same.

Both bounds must be satisfied, which is the point: 50,000 rows arriving at once
re-run a query ONCE, and a row a second cannot re-run it every second. A query
shells out for tens of milliseconds; polling it per row would make monitoring
cost more than the work.
*/
func TestABurstRunsTheQueryOnceAndAnIntervalBoundsADribble(t *testing.T) {
	needSQLite(t)
	p, _ := collecting(t)
	if err := p.WatchQuery(QueryMonitor{
		Name: "failures", SQL: "select count(*) from rows where level='FAIL';",
		EveryRows: 10, MinInterval: 30 * time.Second}); err != nil {
		t.Fatal(err)
	}
	// Nine rows is under EveryRows: not due at all, so not even primed.
	for i := 0; i < 9; i++ {
		p.Write([]byte("FAIL pkg/x\n"))
	}
	if notes := p.Poll(at(1)); len(notes) != 0 {
		t.Errorf("a query under its row bound ran: %v", notes)
	}
	// The tenth makes it due; this poll primes.
	p.Write([]byte("FAIL pkg/y\n"))
	p.Poll(at(2))

	// Ten more rows, but only a second later: the interval holds it.
	for i := 0; i < 10; i++ {
		p.Write([]byte("FAIL pkg/z\n"))
	}
	if notes := p.Poll(at(3)); len(notes) != 0 {
		t.Errorf("a query ran inside its MinInterval: %v", notes)
	}
	// Past the interval it runs, and the answer has changed.
	if notes := p.Poll(at(40)); len(notes) != 1 {
		t.Errorf("past its interval with new rows, the query produced %d notes", len(notes))
	}
}

/*
A query that will not run is reported once and dropped.

The caller wrote the SQL and cannot fix it from here, so repeating the same
complaint on every poll for the rest of a long process is exactly the noise this
type exists to avoid.
*/
func TestABrokenQueryIsReportedOnceThenDropped(t *testing.T) {
	needSQLite(t)
	p, _ := collecting(t)
	if err := p.WatchQuery(QueryMonitor{Name: "bad", SQL: "select nosuchcol from rows;"}); err != nil {
		t.Fatal(err)
	}
	p.Write([]byte("FAIL pkg/a\n"))
	notes := p.Poll(at(1))
	if len(notes) != 1 || !strings.Contains(notes[0], "nosuchcol") {
		t.Fatalf("a broken query was not reported with its cause: %v", notes)
	}
	p.Write([]byte("FAIL pkg/b\n"))
	if again := p.Poll(at(2)); len(again) != 0 {
		t.Errorf("a dropped query complained again: %v", again)
	}
}

/*
And a query monitor on a process collecting no rows is refused.

It could never fire, and a monitor that cannot fire is worse than none: it reads
as coverage.
*/
func TestAQueryMonitorWithoutRowsIsRefused(t *testing.T) {
	p := New(1, "job", "x", "", at(0))
	err := p.WatchQuery(QueryMonitor{Name: "n", SQL: "select 1;"})
	if err == nil {
		t.Fatal("a query monitor was accepted on a process collecting nothing")
	}
	if !strings.Contains(err.Error(), "Extract") {
		t.Errorf("the refusal does not say what is missing: %v", err)
	}
}
