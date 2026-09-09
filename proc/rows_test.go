package proc

import (
	"bytes"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// suiteExtract reads `PASS name` / `FAIL name` and ignores everything else,
// which is what most of a build log is.
func suiteExtract(line []byte) (Row, bool) {
	f := bytes.Fields(line)
	if len(f) < 2 || (string(f[0]) != "PASS" && string(f[0]) != "FAIL") {
		return nil, false
	}
	return Row{string(f[0]), string(f[1])}, true
}

func needSQLite(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("sqlite3"); err != nil {
		t.Skip("sqlite3 not on PATH")
	}
}

/*
A stream becomes rows, and rows answer aggregates.

A build log is structured data pretending to be text. Grep-as-monitor is the
weak reading: it streams every matching line, which is volume. The useful
question is the aggregate — "how many failed" — and that needs the rows, not the
lines.

No driver. agentkit takes no third-party dependencies, and a Go SQLite is either
cgo or a very large pure-Go module; `os/exec` is stdlib and the sqlite3 CLI is
everywhere. Measured 2026-09-08: 50,000 rows imported and aggregated in 51ms.
*/
func TestAStreamBecomesRowsAndRowsAnswerAggregates(t *testing.T) {
	needSQLite(t)
	dir := t.TempDir()
	rows, err := OpenRows(filepath.Join(dir, "rows.tsv"), "level text, pkg text")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	p := New(1, "job", "suite", "", at(0))
	p.Collect(rows, suiteExtract)
	p.Write([]byte("=== RUN everything\nPASS pkg/a\nFAIL pkg/b\n"))
	p.Write([]byte("PASS pkg/c\nFAIL pkg/d\nsome noise nobody parses\n"))

	if rows.Count() != 4 {
		t.Fatalf("collected %d rows from 4 results", rows.Count())
	}
	out, err := rows.Query("select level, count(*) from rows group by level order by level;")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "FAIL") || !strings.Contains(out, "2") {
		t.Errorf("the aggregate does not answer the question:\n%s", out)
	}
}

/*
A half-line is never handed to a parser.

The extract runs on the same complete lines the monitor reports, so a chunk
boundary in the middle of `FAIL pkg/b` cannot produce a row for `FAIL pkg`.
*/
func TestAnExtractOnlySeesWholeLines(t *testing.T) {
	dir := t.TempDir()
	rows, _ := OpenRows(filepath.Join(dir, "r.tsv"), "level text, pkg text")
	defer rows.Close()

	p := New(1, "job", "suite", "", at(0))
	p.Collect(rows, suiteExtract)
	p.Write([]byte("FAIL pkg"))
	if rows.Count() != 0 {
		t.Fatalf("a fragment was parsed into %d row(s)", rows.Count())
	}
	p.Write([]byte("/b\n"))
	if rows.Count() != 1 {
		t.Fatalf("the finished line produced %d rows", rows.Count())
	}
}

/*
A query that will not run says what is wrong, in sqlite's words.

The caller wrote the SQL and the column name it got wrong is the thing it needs
back; an exit status is not.
*/
func TestABadQueryReturnsSQLitesOwnComplaint(t *testing.T) {
	needSQLite(t)
	dir := t.TempDir()
	rows, _ := OpenRows(filepath.Join(dir, "r.tsv"), "level text, pkg text")
	defer rows.Close()
	rows.Append(Row{"FAIL", "pkg/a"})

	_, err := rows.Query("select nosuchcolumn from rows;")
	if err == nil {
		t.Fatal("a query naming a column that does not exist succeeded")
	}
	if !strings.Contains(err.Error(), "nosuchcolumn") {
		t.Errorf("the error does not name the problem: %v", err)
	}
}

/*
Tabs and newlines inside a field do not break the file.

This is a query surface over log output: a field containing a literal tab is a
parsing bug upstream, and silently shifting every later column is a worse answer
than replacing the character.
*/
func TestAFieldCannotBreakTheRowFormat(t *testing.T) {
	needSQLite(t)
	dir := t.TempDir()
	rows, _ := OpenRows(filepath.Join(dir, "r.tsv"), "level text, pkg text")
	defer rows.Close()
	rows.Append(Row{"FAIL", "pkg/a\twith a tab\nand a newline"})

	out, err := rows.Query("select count(*) from rows;")
	if err != nil {
		t.Fatal(err)
	}
	// `.mode column` prints a header and a rule, which is what a model reading
	// the answer wants; the count is the last line.
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if got := strings.TrimSpace(lines[len(lines)-1]); got != "1" {
		t.Errorf("a field with a tab became %q rows:\n%s", got, out)
	}
}
