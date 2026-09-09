package proc

import (
	"fmt"
	"strings"
	"time"
)

// A query as a monitor: signal on the ANSWER changing, not on the output.
//
// `Monitor` reports lines — every line matching a pattern, once enough of them
// pile up. That is volume dressed as a signal: a fifteen-minute suite matching
// ^FAIL four hundred times narrates itself into the window, and the thing
// actually worth interrupting for is that the failure COUNT moved from 0 to 3.
//
// So a query is registered, re-run as rows arrive, and reported only when its
// result differs from last time. Bytes are a proxy for interestingness; a
// changed answer is the thing itself.
type QueryMonitor struct {
	// Name is what the note calls this, e.g. "failures".
	Name string
	// SQL runs against the `rows` table. Aggregates are the point — a query
	// returning a thousand rows will report a thousand rows.
	SQL string
	// EveryRows is the minimum number of NEW rows before re-running. Zero means
	// one: any new row may change the answer.
	EveryRows int
	// MinInterval is the minimum wall time between runs of this query, whatever
	// the row count. Both bounds must be satisfied, which is what makes a burst
	// and a dribble behave the same: 50,000 rows arriving at once re-run this
	// query once, and a row a second cannot re-run it a second.
	MinInterval time.Duration

	last     string
	lastRun  time.Time
	lastRows int64
	primed   bool
	// broken disarms a query whose SQL will not run. Reported once and then
	// silent: the caller wrote it and cannot fix it from here, and repeating
	// the same complaint every poll for the rest of a long process is the
	// noise this whole type exists to avoid.
	broken bool
}

// WatchQuery registers a query to be re-run as rows arrive.
//
// Refused when nothing is collecting rows: a query monitor on a process with no
// Extract would be silently permanent, and a monitor that can never fire is
// worse than none — it reads as coverage.
func (p *Proc) WatchQuery(q QueryMonitor) error {
	if strings.TrimSpace(q.SQL) == "" {
		return fmt.Errorf("proc: a query monitor needs SQL")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.rows == nil {
		return fmt.Errorf(
			"proc: #%d is not collecting rows, so there is nothing to query — give it an "+
				"Extract first, or watch its output with a pattern instead", p.ID)
	}
	if q.EveryRows <= 0 {
		q.EveryRows = 1
	}
	p.queries = append(p.queries, &q)
	return nil
}

// Poll runs the query monitors that are due and returns a note for each whose
// answer CHANGED.
//
// Called by the host from wherever it already pumps — a heartbeat, a turn
// boundary — rather than from Write. A query costs tens of milliseconds and
// shells out; doing that on the producer's write path would make a process's
// own output wait on its monitoring, and holding the lock across it would make
// every other write wait too.
//
// The first run of a query PRIMES it and says nothing. There is no change yet,
// and reporting the starting value as news would fire every monitor once for
// nothing.
func (p *Proc) Poll(now time.Time) []string {
	p.mu.Lock()
	rows, n := p.rows, int64(0)
	if rows != nil {
		n = rows.Count()
	}
	type due struct {
		q *QueryMonitor
	}
	var run []due
	for _, q := range p.queries {
		if q.broken {
			continue
		}
		if n-q.lastRows < int64(q.EveryRows) {
			continue
		}
		if q.MinInterval > 0 && !q.lastRun.IsZero() && now.Sub(q.lastRun) < q.MinInterval {
			continue
		}
		run = append(run, due{q})
	}
	id, kind := p.ID, p.Kind
	p.mu.Unlock()

	var notes []string
	for _, d := range run {
		// Outside the lock: this shells out.
		out, err := rows.Query(d.q.SQL)
		if err != nil {
			p.mu.Lock()
			d.q.broken, d.q.lastRun, d.q.lastRows = true, now, n
			p.mu.Unlock()
			notes = append(notes, fmt.Sprintf("%s #%d — query %q will not run and has been "+
				"dropped: %v", kind, id, d.q.Name, err))
			continue
		}
		p.mu.Lock()
		changed := d.q.primed && out != d.q.last
		was := d.q.last
		d.q.last, d.q.lastRun, d.q.lastRows, d.q.primed = out, now, n, true
		p.mu.Unlock()
		if changed {
			notes = append(notes, fmt.Sprintf("%s #%d — %s changed:\n%s\n(was)\n%s",
				kind, id, d.q.Name, out, was))
		}
	}
	return notes
}
