package files

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// edit replaces exact text, once per pair, all pairs or none.
//
// Each old must match EXACTLY ONCE. A replace-all on text a model chose is how
// an edit meant for one call site lands on five, and the diff that would have
// shown it is never read.
func (t *Toolset) edit(a args) (string, error) {
	if strings.TrimSpace(a.Path) == "" {
		return "", errors.New("edit needs a path")
	}
	var pairs []editPair
	switch {
	case len(a.Edits) > 0 && a.Old != nil:
		return "", errors.New("give either edits:[{old,new},…] or old/new, not both")
	case len(a.Edits) > 0:
		pairs = a.Edits
	case a.Old != nil:
		if a.New == nil {
			return "", errors.New("old needs a new (an empty string deletes the text)")
		}
		pairs = []editPair{{Old: *a.Old, New: *a.New}}
	default:
		return "", fmt.Errorf("edit needs old and new, or edits:[{old,new},…]. To create or replace a whole file use %s", t.Name(Write))
	}
	abs, display, err := t.resolve(a.Path)
	if err != nil {
		return "", err
	}
	if err := t.writable(abs, display); err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "", fmt.Errorf("%w — to create it, use %s", t.notFound(abs, display), t.Name(Write))
	case err != nil:
		return "", err
	case st.IsDir():
		return "", fmt.Errorf("%s is a directory", display)
	case st.Size() > maxEditSize:
		return "", fmt.Errorf("%s is %s; edit loads files up to %s", display, sizeOf(st.Size()), sizeOf(maxEditSize))
	}
	raw, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	if i := bytes.IndexByte(raw[:min(len(raw), binaryScan)], 0); i >= 0 {
		return "", fmt.Errorf("%s is a binary file (a zero byte at offset %d); edit changes text", display, i)
	}
	if err := t.refuseGenerated(raw, display); err != nil {
		return "", err
	}

	orig := string(raw)
	// Line endings are the FILE's: read shows CRLF lines without the CR, so an
	// anchor arrives with bare LFs and is matched — and written — as CRLF.
	crlf := strings.Contains(orig, "\r\n") && strings.Count(orig, "\r\n") == strings.Count(orig, "\n")
	conv := func(s string) string {
		if !crlf {
			return s
		}
		return strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\n", "\r\n")
	}
	content := orig
	var notes []string
	for i, p := range pairs {
		where := ""
		if len(pairs) > 1 {
			where = fmt.Sprintf("edit %d of %d: ", i+1, len(pairs))
		}
		if p.Old == "" {
			return "", fmt.Errorf("%sold is empty; it must be text in the file. To write a whole file use %s", where, t.Name(Write))
		}
		old, nw := conv(p.Old), conv(p.New)
		n := strings.Count(content, old)
		if n == 0 {
			// A model that copied read's "N<TAB>" prefixes: the anchor is
			// unambiguous once they are gone, so take it and say so.
			if so, ok := stripNumbers(p.Old); ok && strings.Count(content, conv(so)) == 1 {
				old, n = conv(so), 1
				if sn, ok := stripNumbers(p.New); ok {
					nw = conv(sn)
				}
				notes = append(notes, where+"the line-number prefixes copied from read were removed from the anchor")
			}
		}
		if n != 1 {
			msg := editMiss(display, p.Old, lf(content), n)
			if len(pairs) > 1 {
				msg += " No edits were applied."
			}
			return "", errors.New(where + msg)
		}
		if old == nw {
			return "", fmt.Errorf("%sold and new are the same text", where)
		}
		content = strings.Replace(content, old, nw, 1)
	}
	if content == orig {
		return fmt.Sprintf("%s: no change (the edits cancel out)", display), nil
	}
	if err := writeAtomic(abs, []byte(content), st.Mode()); err != nil {
		return "", err
	}
	t.markSeen(abs)
	t.afterWrite(t.rel(abs))
	out := fmt.Sprintf("edited %s: %s", display, diffSummary(lf(orig), lf(content)))
	if len(notes) > 0 {
		out += "\n(" + strings.Join(notes, "; ") + ")"
	}
	return out, nil
}

func lf(s string) string { return strings.ReplaceAll(s, "\r\n", "\n") }

var numPrefix = regexp.MustCompile(`^ *\d+\t`)

// stripNumbers removes read's "N<TAB>" prefix from every line, if every
// non-empty line has one.
func stripNumbers(s string) (string, bool) {
	lines := strings.Split(s, "\n")
	any := false
	for i, l := range lines {
		if l == "" && i == len(lines)-1 {
			continue
		}
		loc := numPrefix.FindStringIndex(l)
		if loc == nil {
			return "", false
		}
		lines[i] = l[loc[1]:]
		any = true
	}
	return strings.Join(lines, "\n"), any
}

// writeAtomic replaces abs with data through a temp file and a rename in the
// same directory, keeping mode. A crash leaves the old file or the new one,
// never half of either.
func writeAtomic(abs string, data []byte, mode fs.FileMode) error {
	dir := filepath.Dir(abs)
	f, err := os.CreateTemp(dir, "."+filepath.Base(abs)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	ok := false
	defer func() {
		if !ok {
			_ = os.Remove(tmp)
		}
	}()
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp, mode&(fs.ModePerm|fs.ModeSetuid|fs.ModeSetgid|fs.ModeSticky)); err != nil {
		return err
	}
	if err := os.Rename(tmp, abs); err != nil {
		return err
	}
	ok = true
	return nil
}

// ── miss reporting ──────────────────────────────────────────────────

// editMiss says why an anchor did not match exactly once, and what would.
//
// "appears 0 times" made the next attempt another guess at the same text (aw4,
// 2026-09-05: 85 failures across 36 sessions, the largest wall in the log).
// The common cause is whitespace — a line reproduced with the wrong
// indentation — which a count cannot show. So: the same text modulo
// whitespace, then the closest window with the lines that differ, then the
// closest single line, each with the file's own line numbers.
func editMiss(display, old, body string, n int) string {
	short := clipQ(old, 60)
	if n > 1 {
		return fmt.Sprintf("%s appears %d times in %s (lines %s); an edit must name exactly one place — "+
			"include more of the surrounding lines so the anchor is unique", short, n, display, joinInts(lineNumbers(body, old, 20)))
	}
	fl := strings.Split(body, "\n")
	ol := strings.Split(strings.TrimRight(old, "\n"), "\n")
	if at := lineNumbers(normalise(body), normalise(old), 2); len(at) == 1 {
		return fmt.Sprintf("%s does not appear in %s exactly, but the same text with different whitespace does, at "+
			"line %d. Copy the anchor exactly as read shows it, after the line-number tab:\n%s%s",
			short, display, at[0], region(fl, at[0], at[0]+len(ol)-1), nearestMatch(ol, fl).report())
	}
	nm := nearestMatch(ol, fl)
	if nm.good() {
		return fmt.Sprintf("%s does not appear in %s. Closest: lines %d-%d, %d of %d lines equal:\n%s%s",
			short, display, nm.from, nm.to, nm.matched, nm.total, region(fl, nm.from, nm.to), nm.report())
	}
	if ln, score := bestLine(firstNonBlank(ol), fl); score >= 0.6 {
		return fmt.Sprintf("%s does not appear in %s. The closest single line to its first line is line %d:\n%s"+
			"Re-read that span and copy the anchor from it.", short, display, ln, region(fl, max(1, ln-2), ln+2))
	}
	hint := ""
	if nm.total > 1 {
		hint = fmt.Sprintf(" Only %d of its %d lines appear anywhere in the file, so it was not copied from this file.", nm.present, nm.total)
	}
	return fmt.Sprintf("%s does not appear in %s at all.%s Re-read the file — it may have changed since you read it — "+
		"and use a short anchor (a few lines) copied from that read.", short, display, hint)
}

// region renders lines from..to (1-based, inclusive, clipped to 12) as read does.
func region(fl []string, from, to int) string {
	var b strings.Builder
	to = min(to, len(fl), from+11)
	for i := max(1, from); i <= to; i++ {
		fmt.Fprintf(&b, "%d\t%s\n", i, clipLine(fl[i-1], 200))
	}
	return b.String()
}

// nearest is the window of the file that best lines up with a missed anchor.
type nearest struct {
	from, to       int // 1-based inclusive
	matched, total int // anchor lines equal (modulo whitespace) to their file line
	present        int // anchor lines found anywhere in the file
	diffs          []string
}

func (n nearest) good() bool { return n.total > 0 && n.matched > 0 && n.matched*2 >= n.total }

const maxMissDiff = 1200

// nearestMatch slides the anchor down the file and keeps the window where most
// lines are equal ignoring whitespace (exact equality breaks ties).
func nearestMatch(ol, fl []string) nearest {
	best, bestExact := nearest{total: len(ol)}, -1
	norm := make(map[string]bool, len(fl))
	for _, l := range fl {
		norm[normaliseLine(l)] = true
	}
	for _, l := range ol {
		if strings.TrimSpace(l) != "" && norm[normaliseLine(l)] {
			best.present++
		}
	}
	if best.present == 0 {
		return best
	}
	nol := make([]string, len(ol))
	for j, l := range ol {
		nol[j] = normaliseLine(l)
	}
	for i := range fl {
		m, exact := 0, 0
		for j := 0; j < len(ol) && i+j < len(fl); j++ {
			if normaliseLine(fl[i+j]) == nol[j] {
				m++
				if fl[i+j] == ol[j] {
					exact++
				}
			}
		}
		if m > best.matched || (m == best.matched && m > 0 && exact > bestExact) {
			best.matched, bestExact = m, exact
			best.from, best.to = i+1, min(i+len(ol), len(fl))
		}
	}
	if best.matched == 0 {
		return best
	}
	for j := 0; j < len(ol) && best.from-1+j < len(fl); j++ {
		at := best.from + j
		if want, got := ol[j], fl[at-1]; want != got {
			best.diffs = append(best.diffs, fmt.Sprintf("line %d\n  anchor: %s\n  file:   %s", at, visible(want), visible(got)))
		}
	}
	if over := best.from - 1 + len(ol) - len(fl); over > 0 {
		best.diffs = append(best.diffs, fmt.Sprintf("the anchor runs %d line(s) past the end of the file", over))
	}
	return best
}

func (n nearest) report() string {
	if len(n.diffs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Differing lines (tab shown as \\t, a space at either end as \\s):")
	for i, d := range n.diffs {
		if b.Len()+len(d) > maxMissDiff {
			fmt.Fprintf(&b, "\n… %d more differing line(s)", len(n.diffs)-i)
			break
		}
		b.WriteString("\n" + d)
	}
	return b.String()
}

// bestLine finds the file line most similar to want (bigram Dice coefficient,
// whitespace-normalised).
func bestLine(want string, fl []string) (int, float64) {
	w := bigrams(normaliseLine(want))
	if len(w) == 0 {
		return 0, 0
	}
	bestN, bestS := 0, 0.0
	for i, l := range fl {
		s := dice(w, bigrams(normaliseLine(l)))
		if s > bestS {
			bestN, bestS = i+1, s
		}
	}
	return bestN, bestS
}

func bigrams(s string) map[string]int {
	m := map[string]int{}
	r := []rune(s)
	for i := 0; i+1 < len(r); i++ {
		m[string(r[i:i+2])]++
	}
	return m
}

func dice(a, b map[string]int) float64 {
	na, nb, both := 0, 0, 0
	for k, v := range a {
		na += v
		both += min(v, b[k])
	}
	for _, v := range b {
		nb += v
	}
	if na+nb == 0 {
		return 0
	}
	return 2 * float64(both) / float64(na+nb)
}

func firstNonBlank(ls []string) string {
	for _, l := range ls {
		if strings.TrimSpace(l) != "" {
			return l
		}
	}
	return ""
}

func normaliseLine(s string) string { return strings.Join(strings.Fields(s), " ") }

// normalise collapses whitespace inside lines and KEEPS the newlines, so a
// position found in it is still a line number.
func normalise(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = normaliseLine(l)
	}
	return strings.Join(lines, "\n")
}

// lineNumbers is the 1-based line each occurrence of sub starts on, at most n.
func lineNumbers(body, sub string, n int) []int {
	if sub == "" {
		return nil
	}
	var out []int
	for off := 0; len(out) < n; {
		i := strings.Index(body[off:], sub)
		if i < 0 {
			break
		}
		at := off + i
		out = append(out, strings.Count(body[:at], "\n")+1)
		off = at + max(1, len(sub))
	}
	return out
}

// visible makes the whitespace that decides a match readable: tabs, and
// spaces at either end of the line.
func visible(s string) string {
	s = strings.ReplaceAll(s, "\t", "\\t")
	body := strings.TrimLeft(s, " ")
	lead := len(s) - len(body)
	trimmed := strings.TrimRight(body, " ")
	s = strings.Repeat("\\s", lead) + trimmed + strings.Repeat("\\s", len(body)-len(trimmed))
	return clipLine(s, 160)
}

func clipLine(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return safeCut(s[:n]) + "…"
}

func clipQ(s string, n int) string { return strconv.Quote(clipLine(s, n)) }

func joinInts(ns []int) string {
	parts := make([]string, len(ns))
	for i, n := range ns {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ", ")
}
