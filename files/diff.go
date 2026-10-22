package files

import (
	"fmt"
	"strings"
)

// diffSummary is "+N -M" and the changed hunks, unified-diff style with two
// lines of context, clipped. Lines are compared whole: the common head and
// tail are trimmed first, so the Myers search below only ever sees the
// changed middle.
func diffSummary(a, b string) string {
	al, bl := splitLines(a), splitLines(b)
	ops := diffLines(al, bl)
	add, del := 0, 0
	for _, o := range ops {
		switch o.kind {
		case '+':
			add++
		case '-':
			del++
		}
	}
	hunks := hunksOf(ops, 2)
	var s strings.Builder
	fmt.Fprintf(&s, "+%d -%d", add, del)
	if len(hunks) > 1 {
		fmt.Fprintf(&s, " in %d hunks", len(hunks))
	}
	shown := 0
	const maxShown = 40
	for hi, h := range hunks {
		if shown >= maxShown {
			fmt.Fprintf(&s, "\n… %d more hunk(s)", len(hunks)-hi)
			break
		}
		fmt.Fprintf(&s, "\n@@ -%d,%d +%d,%d @@", h.aStart, h.aLen, h.bStart, h.bLen)
		for li, o := range h.ops {
			if shown >= maxShown {
				fmt.Fprintf(&s, "\n… %d more line(s) in this hunk", len(h.ops)-li)
				break
			}
			s.WriteString("\n" + string(o.kind) + clipLine(o.text, 200))
			shown++
		}
	}
	return s.String()
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

type diffOp struct {
	kind rune // ' ', '-', '+'
	text string
}

// diffMaxD bounds the Myers search. Past it the middle is reported as one
// replaced block — still correct, just not minimal — rather than spending
// O(D²) memory on a rewrite.
const diffMaxD = 1000

func diffLines(a, b []string) []diffOp {
	p := 0
	for p < len(a) && p < len(b) && a[p] == b[p] {
		p++
	}
	s := 0
	for s < len(a)-p && s < len(b)-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	var ops []diffOp
	for _, l := range a[:p] {
		ops = append(ops, diffOp{' ', l})
	}
	ops = append(ops, myers(a[p:len(a)-s], b[p:len(b)-s])...)
	for _, l := range a[len(a)-s:] {
		ops = append(ops, diffOp{' ', l})
	}
	return ops
}

// myers is the O((N+M)D) shortest edit script.
func myers(a, b []string) []diffOp {
	n, m := len(a), len(b)
	whole := func() []diffOp {
		var ops []diffOp
		for _, l := range a {
			ops = append(ops, diffOp{'-', l})
		}
		for _, l := range b {
			ops = append(ops, diffOp{'+', l})
		}
		return ops
	}
	if n == 0 || m == 0 {
		return whole()
	}
	maxD := min(n+m, diffMaxD)
	off := maxD + 1
	v := make([]int, 2*maxD+3)
	var trace [][]int
	found := -1
	for d := 0; d <= maxD && found < 0; d++ {
		trace = append(trace, append([]int(nil), v...))
		for k := -d; k <= d; k += 2 {
			var x int
			if k == -d || (k != d && v[off+k-1] < v[off+k+1]) {
				x = v[off+k+1]
			} else {
				x = v[off+k-1] + 1
			}
			y := x - k
			for x < n && y < m && a[x] == b[y] {
				x++
				y++
			}
			v[off+k] = x
			if x >= n && y >= m {
				found = d
				break
			}
		}
	}
	if found < 0 {
		return whole()
	}
	// Walk the trace back from (n, m).
	var rev []diffOp
	x, y := n, m
	for d := found; d > 0; d-- {
		pv := trace[d]
		k := x - y
		var pk int
		if k == -d || (k != d && pv[off+k-1] < pv[off+k+1]) {
			pk = k + 1
		} else {
			pk = k - 1
		}
		px := pv[off+pk]
		py := px - pk
		for x > px && y > py {
			x--
			y--
			rev = append(rev, diffOp{' ', a[x]})
		}
		if x == px {
			y--
			rev = append(rev, diffOp{'+', b[y]})
		} else {
			x--
			rev = append(rev, diffOp{'-', a[x]})
		}
	}
	for x > 0 && y > 0 {
		x--
		y--
		rev = append(rev, diffOp{' ', a[x]})
	}
	ops := make([]diffOp, len(rev))
	for i := range rev {
		ops[i] = rev[len(rev)-1-i]
	}
	return ops
}

type hunk struct {
	aStart, aLen, bStart, bLen int
	ops                        []diffOp
}

// hunksOf groups changes with ctx lines of context, merging hunks whose
// context would touch.
func hunksOf(ops []diffOp, ctx int) []hunk {
	var out []hunk
	ai, bi := make([]int, len(ops)+1), make([]int, len(ops)+1)
	for i, o := range ops {
		ai[i+1], bi[i+1] = ai[i], bi[i]
		if o.kind != '+' {
			ai[i+1]++
		}
		if o.kind != '-' {
			bi[i+1]++
		}
	}
	i := 0
	for i < len(ops) {
		if ops[i].kind == ' ' {
			i++
			continue
		}
		start := max(0, i-ctx)
		end := i
		for end < len(ops) {
			if ops[end].kind != ' ' {
				end++
				continue
			}
			run := end
			for run < len(ops) && ops[run].kind == ' ' {
				run++
			}
			if run < len(ops) && run-end <= 2*ctx {
				end = run
				continue
			}
			end = min(len(ops), end+ctx)
			break
		}
		h := hunk{ops: ops[start:end]}
		h.aStart, h.bStart = ai[start]+1, bi[start]+1
		h.aLen, h.bLen = ai[end]-ai[start], bi[end]-bi[start]
		out = append(out, h)
		i = end
	}
	return out
}
