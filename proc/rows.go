package proc

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// Structured output, without a database driver.
//
// A build log and a test run are structured data pretending to be text: `FAIL
// pkg/b` is a row. Grep-as-monitor is the weak reading of that — it streams
// every matching line, which is volume. The useful question is an aggregate
// that CHANGED: "the failure count went from 0 to 3", not four hundred lines
// each matching ^FAIL.
//
// So an Extract turns lines into rows, the rows append to a TSV beside the log,
// and a query is `sqlite3` on the end of a pipe. No driver: agentkit takes no
// third-party dependencies (invariant 0), and a Go SQLite is either cgo or a
// very large pure-Go module — but `os/exec` is stdlib and the sqlite3 CLI is
// everywhere. Measured 2026-09-08: 50,000 rows imported into :memory: and
// aggregated in 51ms, which is cheaper than the parsing that produced them.
//
// Detected at USE, not at start. A host without sqlite3 still gets the rows
// file — greppable, and the thing it was extracted into — and a query says
// plainly what is missing instead of the feature silently not existing.

// Row is one extracted record. Fields are written in order and must match the
// column order of Schema.
type Row []string

// Extract turns one complete line into a row. Returning false drops the line,
// which is the common case: most of a build log is not data.
type Extract func(line []byte) (Row, bool)

// Rows is the structured side of a process's output.
type Rows struct {
	// Path is the TSV the rows are appended to.
	Path string
	// Schema is the CREATE TABLE column list, e.g. "level text, pkg text".
	// Column order must match what Extract emits.
	Schema string

	f *os.File
	n int64
}

// OpenRows starts collecting rows into path. The caller supplies the schema
// because only it knows what its Extract produces.
func OpenRows(path, schema string) (*Rows, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Rows{Path: path, Schema: schema, f: f}, nil
}

// Append writes one row. Tabs and newlines inside a field would break the TSV,
// so they are replaced rather than escaped: this is a query surface over log
// output, and a field containing a literal tab is a parsing bug upstream, not
// something to preserve faithfully.
func (r *Rows) Append(row Row) error {
	if r == nil || r.f == nil {
		return nil
	}
	clean := make([]string, len(row))
	for i, v := range row {
		v = strings.ReplaceAll(v, "\t", " ")
		clean[i] = strings.ReplaceAll(v, "\n", " ")
	}
	if _, err := fmt.Fprintln(r.f, strings.Join(clean, "\t")); err != nil {
		return err
	}
	r.n++
	return nil
}

// Count is how many rows have been written.
func (r *Rows) Count() int64 {
	if r == nil {
		return 0
	}
	return r.n
}

// Close stops collection. The file stays: it is the queryable artifact.
func (r *Rows) Close() error {
	if r == nil || r.f == nil {
		return nil
	}
	err := r.f.Close()
	r.f = nil
	return err
}

// ErrNoSQLite says the feature was reached on a machine without the tool.
var ErrNoSQLite = fmt.Errorf(
	"sqlite3 is not on PATH, so these rows cannot be queried here — the rows " +
		"themselves are still on disk as tab-separated text, which grep and awk read")

// Query runs one SQL statement against the rows, which are imported into an
// in-memory database as the table `rows`.
//
// A fresh import per query rather than a long-lived database: the rows file is
// append-only and the import is measured in tens of milliseconds, so there is
// no state to keep consistent, nothing to migrate, and a query always sees
// exactly what has been written by the time it ran.
func (r *Rows) Query(sql string) (string, error) {
	if r == nil || r.Path == "" {
		return "", fmt.Errorf("this process collected no rows")
	}
	if _, err := exec.LookPath("sqlite3"); err != nil {
		return "", ErrNoSQLite
	}
	if err := r.flush(); err != nil {
		return "", err
	}
	cmd := exec.Command("sqlite3", ":memory:",
		"-cmd", "create table rows("+r.Schema+");",
		"-cmd", ".mode tabs",
		"-cmd", ".import "+r.Path+" rows",
		"-cmd", ".mode column",
		sql)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		// sqlite's own message names the column or the syntax, which is what a
		// caller needs; the exit status is not.
		msg := strings.TrimSpace(errb.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("%s", msg)
	}
	return strings.TrimRight(out.String(), "\n"), nil
}

// flush makes everything written so far visible to a reader of the file.
func (r *Rows) flush() error {
	if r.f == nil {
		return nil
	}
	return r.f.Sync()
}
