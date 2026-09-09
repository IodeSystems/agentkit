package proc

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// Tail returns the last n bytes of what a process has produced, cut to a line
// boundary so it never opens mid-line.
//
// From the LOG, not from memory: the point of writing output to a file is that
// asking about a long-running thing costs the same whether it has produced ten
// lines or fifty thousand.
func (p *Proc) Tail(n int) string {
	p.mu.Lock()
	path, early := p.LogPath, append([]byte(nil), p.early...)
	p.mu.Unlock()
	if n <= 0 {
		n = TailBytes
	}
	body := early
	if path != "" {
		if b, err := os.ReadFile(path); err == nil {
			body = b
		}
	}
	if len(body) > n {
		body = body[len(body)-n:]
		if i := indexByte(body, '\n'); i >= 0 && i+1 < len(body) {
			body = body[i+1:]
		}
	}
	return strings.TrimRight(string(body), "\n")
}

func indexByte(b []byte, c byte) int {
	for i := range b {
		if b[i] == c {
			return i
		}
	}
	return -1
}

// Status is one line: what this is, how long it has run, and how it ended.
func (p *Proc) Status(now time.Time) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	state := "running " + now.Sub(p.started).Round(time.Second).String()
	if p.done {
		state = "done in " + p.ended.Sub(p.started).Round(time.Second).String()
		if p.result != "" {
			state += ", " + p.result
		}
	}
	var muted string
	if p.mon.Ignore {
		muted = " [muted]"
	}
	return fmt.Sprintf("#%d %s %s — %s, %s%s",
		p.ID, p.Kind, p.Label, state, bytesHuman(p.written), muted)
}

func bytesHuman(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%dKB", n/(1<<10))
	}
	return fmt.Sprintf("%dB", n)
}

// Compose renders a monitor answer as the text a model reads.
//
// ONE renderer for every kind of process, which is the whole reason this
// package exists: dun grew `agent_monitor` and a background-job monitor
// separately, taking the same (id, tail) and answering in two different shapes.
// A model that has learned to read one should not have to learn the other.
//
// With no id: every process, one line each, newest last — "what is running".
// With an id: that one's status, then its tail, which is the question actually
// being asked ("how is it going") answered in the order a person reads it.
func Compose(all []*Proc, id int, tail int, now time.Time) string {
	if len(all) == 0 {
		return "nothing is running."
	}
	sorted := append([]*Proc(nil), all...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	if id <= 0 {
		var b strings.Builder
		for _, p := range sorted {
			b.WriteString(p.Status(now))
			b.WriteString("\n")
		}
		b.WriteString("\nAsk about one by id to read its output.")
		return strings.TrimLeft(b.String(), "\n")
	}
	for _, p := range sorted {
		if p.ID != id {
			continue
		}
		out := p.Status(now)
		if p.LogPath != "" {
			out += "\nlog: " + p.LogPath
		}
		if t := p.Tail(tail); t != "" {
			out += "\n\n" + t
		} else {
			out += "\n\n(no output yet)"
		}
		return out
	}
	return fmt.Sprintf("there is no #%d. Running now:\n%s", id, Compose(all, 0, 0, now))
}
