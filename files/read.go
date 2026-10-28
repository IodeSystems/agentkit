package files

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The read caps (Config.ReadMax, Config.UnrangedMax) bound what one read
// RETURNS, for the context window and not for memory. Measured in aw4:
//
//   - 64KB ranged is ~16k tokens, ~9% of a 170k window. 256KB was 28% of the
//     window in one call; across twelve hours file.read returned 9.7MB in 2779
//     calls, one of them 191KB, while turn latency went 58s → 337s.
//   - 8KB unranged: reads that named no range averaged 4.8KB, so the cap
//     changes nothing for most of them. It bites the 2.2% above 16KB that
//     carried 17% of a day's read bytes — exactly the set that fills a
//     window. The cost is a second call for those files; a call is transient,
//     bytes in the window are not.
//
// A capped result is not a loss: it names the file's length and the next
// range, and a session that needs more asks for more.

// binaryScan is how far the NUL check looks: git's heuristic, and right for
// the same reason — a zero byte never occurs in text and almost always occurs
// in a format that is not text.
const binaryScan = 8000

// maxEditSize bounds the files edit will load whole.
const maxEditSize = 16 << 20

// largeFile is the size from which a head or a tail read gets a note: search
// before reading. A benchmark (dun, 2026-10-27) read the last 120 lines of a
// 218KB CI log, ~5.8k tokens carried for four more requests, where a grep of
// the same log returned 265 bytes.
const largeFile = 64 * 1024

// largeHead is what a read of a large file with no range returns: enough to
// see what the file is, not a page to carry. The same benchmark re-run with
// the note: a model read the 8KB head first (~5k tokens, kept in context)
// and only then grepped; a short head makes the search the next step.
const largeHead = 2 << 10

// read reads a file or a range of it; on a directory it is list; with grep it
// is grep, on a file or a tree. One tool for the three, because a model that
// has read in hand reaches for it, and a search that needs a second tool is a
// search it does not make.
func (t *Toolset) read(ctx context.Context, a args) (string, error) {
	if a.Grep == "" && a.Pattern != "" {
		a.Grep = a.Pattern // the grep tool's name for it
	}
	if a.Grep != "" {
		a.Pattern = a.Grep
		return t.grep(ctx, a)
	}
	if strings.TrimSpace(a.Path) == "" {
		return "", errors.New("read needs a path")
	}
	abs, display, err := t.resolve(a.Path)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(abs)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return "", t.notFound(abs, display)
	case err != nil:
		return "", err
	case st.IsDir():
		return t.list(ctx, a)
	}
	offset, limit := int(a.Offset), int(a.Limit)
	if limit < 0 {
		return "", fmt.Errorf("limit counts lines and cannot be negative (got %d). To read the END of a file, "+
			"give a negative offset: offset=-50 is its last 50 lines", limit)
	}
	f, err := os.Open(abs)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()

	head := make([]byte, binaryScan)
	n, _ := io.ReadFull(f, head)
	head = head[:n]
	if i := bytes.IndexByte(head, 0); i >= 0 {
		return "", fmt.Errorf("%s is a binary file (%s, %s; a zero byte at offset %d), so there is no text "+
			"to show. Use a tool that understands the format", display, sizeOf(st.Size()), http.DetectContentType(head), i)
	}
	if st.Size() == 0 {
		t.markSeen(abs)
		return display + ": empty (0 bytes)", nil
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}

	ranged := offset != 0 || limit != 0
	large := st.Size() >= largeFile
	capBytes := t.cfg.ReadMax
	if !ranged {
		capBytes = t.cfg.UnrangedMax
		if large {
			capBytes = min(capBytes, largeHead)
		}
	}

	// The total is needed for the header, for a negative offset and for the
	// paging hint, so the file is counted first; it is streamed, never held.
	total, err := countLines(f)
	if err != nil {
		return "", err
	}
	start := 1
	switch {
	case offset < 0:
		start = max(1, total+offset+1)
	case offset > total:
		return "", fmt.Errorf("offset %d is past the end of %s, which has %d lines. Lines are numbered from 1, "+
			"the numbers a grep reports", offset, display, total)
	case offset > 0:
		start = offset
	}
	end := total
	if limit > 0 {
		end = min(total, start+limit-1)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	body, last, err := numbered(f, start, end, capBytes)
	if err != nil {
		return "", err
	}
	t.markSeen(abs)

	var b strings.Builder
	switch {
	case start == 1 && last == total:
		fmt.Fprintf(&b, "%s: %s\n", display, plural(total, "line"))
	default:
		fmt.Fprintf(&b, "%s: lines %d-%d of %d\n", display, start, last, total)
	}
	b.WriteString(body)
	if last < end {
		// Capped: say what was cut and the exact next range, so paging needs
		// no arithmetic the caller has to get right.
		next := last + 1
		switch {
		case !ranged && large:
			// The note below says what to do instead; no paging advice.
		case !ranged:
			fmt.Fprintf(&b, "\n[cut at %s: this read named no range, and gets a head. %s has %d lines (%s). "+
				"Continue with offset=%d limit=200, or offset=-200 for the end; a ranged read returns up to %s]",
				sizeOf(int64(capBytes)), display, total, sizeOf(st.Size()), next, sizeOf(int64(t.cfg.ReadMax)))
		default:
			fmt.Fprintf(&b, "\n[cut at %s, the most one read returns. %s has %d lines. Continue with offset=%d]",
				sizeOf(int64(capBytes)), display, total, next)
		}
	}
	out := strings.TrimRight(b.String(), "\n")
	if large && (!ranged || offset < 0) {
		out += fmt.Sprintf("\n[%s is %s, %d lines: search it with grep (e.g. grep:\"ERROR|FAIL\" for a log), or read a range]",
			display, sizeOf(st.Size()), total)
	}
	return out, nil
}

// countLines counts lines the way numbered numbers them: a final line without
// a newline still counts, a trailing newline does not add an empty one.
func countLines(r io.Reader) (int, error) {
	buf := make([]byte, 64*1024)
	n, last := 0, byte('\n')
	for {
		k, err := r.Read(buf)
		if k > 0 {
			n += bytes.Count(buf[:k], []byte{'\n'})
			last = buf[k-1]
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, err
		}
	}
	if last != '\n' {
		n++
	}
	return n, nil
}

// numbered renders lines start..end as "N<TAB>text", stopping before capBytes
// of text. A CR before the newline is dropped from the display (edit puts it
// back). It returns the last line rendered.
//
// The number and its tab are not part of the file. Tab rather than padding or
// a "│", because "everything after the first tab" is one rule a model can
// apply, and edit also strips the prefix if a model copies it anyway.
func numbered(r io.Reader, start, end, capBytes int) (string, int, error) {
	br := bufio.NewReaderSize(r, 64*1024)
	var b strings.Builder
	used, line, last := 0, 0, start-1
	for line < end {
		text, full, err := readLine(br, line+1 >= start, capBytes-used+1)
		if err != nil && err != io.EOF {
			return "", last, err
		}
		if err == io.EOF && text == nil {
			break
		}
		line++
		if line >= start {
			s := strings.TrimSuffix(string(text), "\r")
			if used+len(s) > capBytes {
				if last >= start {
					break // stop on a whole line
				}
				// One line longer than the cap: show its head.
				s = safeCut(s[:max(0, capBytes-used)]) + fmt.Sprintf("…[line is %s; cut]", sizeOf(int64(full)))
			}
			used += len(s)
			b.WriteString(strconv.Itoa(line))
			b.WriteByte('\t')
			b.WriteString(s)
			b.WriteByte('\n')
			last = line
			if used >= capBytes {
				break
			}
		}
		if err == io.EOF {
			break
		}
	}
	return b.String(), last, nil
}

// readLine returns the next line without its newline, and the line's full
// length. When keep is false the line is skipped without being held; when
// keep is true at most limit bytes of it are held (the rest is consumed), so
// one enormous line cannot be loaded whole. A nil slice with io.EOF means
// there was no line.
func readLine(br *bufio.Reader, keep bool, limit int) ([]byte, int, error) {
	var out []byte
	got := false
	full := 0
	for {
		chunk, err := br.ReadSlice('\n')
		if len(chunk) > 0 {
			got = true
			full += len(bytes.TrimSuffix(chunk, []byte{'\n'}))
			if keep && len(out) < limit {
				room := limit - len(out)
				c := bytes.TrimSuffix(chunk, []byte{'\n'})
				if len(c) > room {
					c = c[:room]
				}
				out = append(out, c...)
			}
		}
		switch {
		case err == bufio.ErrBufferFull:
			continue
		case err == io.EOF:
			if !got {
				return nil, 0, io.EOF
			}
			if out == nil {
				out = []byte{}
			}
			return out, full, io.EOF
		case err != nil:
			return nil, 0, err
		}
		if out == nil {
			out = []byte{}
		}
		return out, full, nil
	}
}

// safeCut trims s back to a rune boundary: a cut mid-rune is invalid UTF-8,
// which can break the transport rather than merely read badly.
func safeCut(s string) string {
	for len(s) > 0 && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

func sizeOf(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1fKB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d bytes", n)
}
