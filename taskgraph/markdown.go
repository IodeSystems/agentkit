// Markdown is the plan as text a person can read and edit, and the check a host
// runs on an edit before it keeps it.
//
// Form (Graph.Markdown writes it, ParseMarkdown reads it):
//
//	# Plan: <goal>                       optional; Graph.Goal
//
//	## <id> [<box>] <title>              one section per node
//	effort: high
//	owner: any non-empty text, not checked here
//	needs: 1, 2
//	done when: what has to be true
//	account: how it was met
//	reason: why blocked or dropped
//	replaces: 3
//
//	Free Markdown: Node.Details. Any length.
//
// Section rule: a line starts a section ONLY when it matches
// "## <id> [<box>] <title>": an id with no space, then a box below. Any other
// "## ..." line (say "## Notes") is Details, and so is every line inside a code
// fence. The one ambiguous case: a Details line outside a fence that does match,
// such as "## 3 [x] title", reads back as a new section; fence it.
//
// Box: " " todo, ">" doing, "x" done, "!" blocked, "-" dropped. Field lines run
// to the first blank line. The writer puts one field per line, in the order
// above, and leaves out an empty one; the reader also takes several fields on
// one line joined by " · ". Nodes come out topologically, ties by id, so the
// same plan is the same text and a diff of two texts is a diff of the plan.
//
// Not in the text, so a round trip does not keep them: ClosedAt, ClosedBy,
// Meta, Seq (the reader sets Seq to the largest numeric id). Status changes are
// not checked against the transition table: the text says where a node is, not
// how it got there.
//
// Why the reader reports every problem instead of the first: the text is
// edited by a person or a cheap model, and one round of "fix these" beats five.
package taskgraph

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// Problem is one thing wrong with a plan text or a plan. Line is 1-based in
// the text, 0 when the plan did not come from text or the problem has no
// place (a node that disappeared). ID is the node, "" when none.
type Problem struct {
	Line int
	ID   string
	Msg  string
}

func (p Problem) String() string {
	var b strings.Builder
	if p.Line > 0 {
		fmt.Fprintf(&b, "line %d: ", p.Line)
	}
	if p.ID != "" {
		fmt.Fprintf(&b, "node %s: ", p.ID)
	}
	b.WriteString(p.Msg)
	return b.String()
}

var boxOf = map[Status]string{Todo: " ", Doing: ">", Done: "x", Blocked: "!", Dropped: "-"}

var statusOfBox = map[string]Status{" ": Todo, ">": Doing, "x": Done, "!": Blocked, "-": Dropped}

// fieldKeys is the field order of the canonical form.
var fieldKeys = []string{"effort", "owner", "needs", "done when", "account", "reason", "replaces"}

// idLess orders ids: numbers by value and before words, words by text.
func idLess(a, b string) bool {
	x, ex := strconv.Atoi(a)
	y, ey := strconv.Atoi(b)
	switch {
	case ex == nil && ey == nil && x != y:
		return x < y
	case ex == nil && ey != nil:
		return true
	case ex != nil && ey == nil:
		return false
	}
	return a < b
}

// order is the nodes with every need before its dependents, ties by id. Nodes
// caught in a cycle, or needing a missing id, come last by id: Markdown writes
// a plan Validate would refuse, so a person can see and fix it.
func (g *Graph) order() []*Node {
	byID := map[string]*Node{}
	for _, n := range g.Nodes {
		if _, dup := byID[n.ID]; !dup {
			byID[n.ID] = n
		}
	}
	out := make([]*Node, 0, len(g.Nodes))
	placed := map[*Node]bool{}
	for len(out) < len(g.Nodes) {
		var ready []*Node
		for _, n := range g.Nodes {
			if placed[n] {
				continue
			}
			ok := true
			for _, d := range n.Needs {
				if dn := byID[d]; dn != nil && !placed[dn] {
					ok = false
					break
				}
			}
			if ok {
				ready = append(ready, n)
			}
		}
		if len(ready) == 0 { // a cycle: the rest by id
			for _, n := range g.Nodes {
				if !placed[n] {
					ready = append(ready, n)
				}
			}
			sort.SliceStable(ready, func(i, j int) bool { return idLess(ready[i].ID, ready[j].ID) })
			for _, n := range ready {
				placed[n] = true
				out = append(out, n)
			}
			break
		}
		sort.SliceStable(ready, func(i, j int) bool { return idLess(ready[i].ID, ready[j].ID) })
		// One at a time, so a node that just became ready competes by id with the rest.
		placed[ready[0]] = true
		out = append(out, ready[0])
	}
	return out
}

// Markdown is the canonical text of the plan; see the file's doc block.
func (g *Graph) Markdown() string {
	var b strings.Builder
	if g.Goal != "" {
		b.WriteString("# Plan: " + g.Goal + "\n")
	}
	for _, n := range g.order() {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		box, ok := boxOf[n.Status]
		if !ok {
			box = " "
		}
		fmt.Fprintf(&b, "## %s [%s] %s\n", n.ID, box, n.Title)
		vals := map[string]string{
			"effort": n.Effort, "owner": n.Owner, "needs": strings.Join(n.Needs, ", "), "done when": n.DoneWhen,
			"account": n.Account, "reason": n.Why, "replaces": n.Replaces,
		}
		for _, k := range fieldKeys {
			if v := vals[k]; v != "" {
				b.WriteString(k + ": " + v + "\n")
			}
		}
		if d := strings.TrimSpace(n.Details); d != "" {
			b.WriteString("\n" + d + "\n")
		}
	}
	return b.String()
}

// fence says whether line opens or closes a code fence of marker m (what
// opened it, "" when none is open): returns the marker now in force.
func fence(line, m string) string {
	t := strings.TrimLeft(line, " ")
	if len(t) < 3 || len(line)-len(t) > 3 {
		return m
	}
	c := t[:1]
	if c != "`" && c != "~" || !strings.HasPrefix(t, strings.Repeat(c, 3)) {
		return m
	}
	if m == "" {
		return c
	}
	if c == m && strings.Trim(t, c+" \t") == "" {
		return ""
	}
	return m
}

// splitFields cuts a field line at each " · " that is followed by a known key
// and ":", so a value may hold " · " itself.
func splitFields(line string) []string {
	var parts []string
	for {
		cut := -1
		for i := 0; ; {
			j := strings.Index(line[i:], " · ")
			if j < 0 {
				break
			}
			rest := line[i+j+len(" · "):]
			if k, _, ok := fieldKey(rest); ok && k != "" {
				cut = i + j
				break
			}
			i += j + len(" · ")
		}
		if cut < 0 {
			return append(parts, line)
		}
		parts = append(parts, line[:cut])
		line = line[cut+len(" · "):]
	}
}

// fieldKey reads "key: value" with a known key.
func fieldKey(s string) (key, val string, ok bool) {
	k, v, found := strings.Cut(s, ":")
	if !found {
		return "", "", false
	}
	k = strings.ToLower(strings.TrimSpace(k))
	if !slices.Contains(fieldKeys, k) {
		return "", "", false
	}
	return k, strings.TrimSpace(v), true
}

// ParseMarkdown reads a plan text. It reports every problem it finds, of the
// text and of the plan (Validate with no previous plan), and returns the graph
// it could read either way.
func ParseMarkdown(text string) (*Graph, []Problem) {
	g := &Graph{}
	var probs []Problem
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	var cur *Node
	var body []string
	inFields := false
	fenced := ""
	flush := func() {
		if cur != nil {
			cur.Details = strings.TrimSpace(strings.Join(body, "\n"))
		}
		cur, body = nil, nil
	}
	for i, line := range lines {
		no := i + 1
		if fenced == "" {
			if id, box, title, ok := sectionHeading(line); ok {
				flush()
				cur = &Node{ID: id, Status: statusOfBox[box], Title: clean(title), line: no}
				g.Nodes = append(g.Nodes, cur)
				inFields = true
				continue
			}
			if msg := brokenHeading(line); msg != "" {
				id := ""
				if cur != nil {
					id = cur.ID
				}
				probs = append(probs, Problem{Line: no, ID: id, Msg: msg})
				continue
			}
		}
		if cur == nil {
			if t := strings.TrimSpace(line); t == "" {
				continue
			} else if goal, ok := strings.CutPrefix(line, "# Plan:"); ok && len(g.Nodes) == 0 && g.Goal == "" {
				g.Goal = strings.TrimSpace(goal)
			} else {
				probs = append(probs, Problem{Line: no, Msg: "text outside a node section; a node starts with \"## <id> [<box>] <title>\", box one of \" \", \">\", \"x\", \"!\", \"-\""})
			}
			continue
		}
		if inFields {
			if strings.TrimSpace(line) == "" {
				inFields = false
				continue
			}
			for _, part := range splitFields(line) {
				k, v, ok := fieldKey(part)
				if !ok {
					probs = append(probs, Problem{Line: no, ID: cur.ID, Msg: fmt.Sprintf("%q is not a field (%s); a blank line ends the fields", Clip(part, 40), strings.Join(fieldKeys, ", "))})
					continue
				}
				setField(cur, k, v)
			}
			continue
		}
		fenced = fence(line, fenced)
		body = append(body, line)
	}
	flush()
	for _, n := range g.Nodes {
		if v, err := strconv.Atoi(n.ID); err == nil && v > g.Seq {
			g.Seq = v
		}
	}
	return g, append(probs, Validate(nil, g)...)
}

func setField(n *Node, k, v string) {
	switch k {
	case "effort":
		n.Effort = v
	case "owner":
		n.Owner = v
	case "needs":
		n.Needs = cleanList(strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' }))
	case "done when":
		n.DoneWhen = v
	case "account":
		n.Account = v
	case "reason":
		n.Why = v
	case "replaces":
		n.Replaces = v
	}
}

// sectionHeading reads "## <id> [<box>] <title>": an id with no space and a
// known box. Anything else is not a section.
func sectionHeading(line string) (id, box, title string, ok bool) {
	rest, found := strings.CutPrefix(line, "## ")
	if !found {
		return
	}
	id, rest, _ = strings.Cut(rest, " ")
	if id == "" || len(rest) < 3 || rest[0] != '[' || rest[2] != ']' {
		return "", "", "", false
	}
	box = rest[1:2]
	if _, known := statusOfBox[box]; !known {
		return "", "", "", false
	}
	rest = rest[3:]
	if rest != "" && rest[0] != ' ' {
		return "", "", "", false
	}
	return id, box, strings.TrimSpace(rest), true
}

// brokenHeading is the problem text for a line that tries to be a heading,
// "## <id> [<anything up to ]>] ...", and is not one: read as body text it
// would drop a step with no word. "## Notes" and "## 1 a" have no bracket and
// are body.
func brokenHeading(line string) string {
	rest, ok := strings.CutPrefix(line, "## ")
	if !ok {
		return ""
	}
	id, rest, _ := strings.Cut(rest, " ")
	end := strings.Index(rest, "]")
	if id == "" || !strings.HasPrefix(rest, "[") || end < 0 {
		return ""
	}
	if _, known := statusOfBox[rest[1:end]]; !known {
		return fmt.Sprintf("box not known: use one of \" \" todo, \">\" doing, \"x\" done, \"!\" blocked, \"-\" dropped; got %q", rest[1:end])
	}
	return "the heading needs a space after the box: \"## <id> [<box>] <title>\""
}

// Validate is every rule a plan must keep, for a host to run on an edit before
// it keeps the edit. prev is the plan before it, or nil. It returns all
// problems, by line (no line first); Line is set for a node that came from text.
//
//   - ids are unique and not empty;
//   - an id in prev is still in next: a step is dropped, never deleted;
//   - every node has a title and done_when;
//   - needs name a node, not the node itself, and form no cycle;
//   - done has an account; blocked and dropped have a reason;
//   - replaces names a node;
//   - effort is "", low, medium or high.
func Validate(prev, next *Graph) []Problem {
	var probs []Problem
	add := func(n *Node, format string, a ...any) {
		probs = append(probs, Problem{Line: n.line, ID: n.ID, Msg: fmt.Sprintf(format, a...)})
	}
	ids := map[string]*Node{}
	for _, n := range next.Nodes {
		switch {
		case n.ID == "":
			add(n, "a node needs an id")
		case ids[n.ID] != nil:
			add(n, "id is used twice (first at line %d)", ids[n.ID].line)
		default:
			ids[n.ID] = n
		}
	}
	if prev != nil {
		for _, p := range prev.Nodes {
			if ids[p.ID] == nil {
				probs = append(probs, Problem{ID: p.ID, Msg: "the node is gone; a step is never deleted, only dropped (box \"-\" with a reason)"})
			}
		}
	}
	for _, n := range next.Nodes {
		if n.Title == "" {
			add(n, "needs a title")
		}
		if n.DoneWhen == "" {
			add(n, "needs done when: what has to be true for it to be done")
		}
		for _, d := range n.Needs {
			switch {
			case d == n.ID:
				add(n, "needs itself")
			case ids[d] == nil:
				add(n, "needs %s, which is not a node", d)
			}
		}
		switch n.Status {
		case Done:
			if n.Account == "" {
				add(n, "is done and needs an account: how done when was met")
			}
		case Blocked, Dropped:
			if n.Why == "" {
				add(n, "is %s and needs a reason", n.Status)
			}
		}
		if n.Replaces != "" && ids[n.Replaces] == nil {
			add(n, "replaces %s, which is not a node", n.Replaces)
		}
		if err := checkEffort(n.Effort); err != nil {
			add(n, "%s", err)
		}
	}
	// Cycles: a depth-first walk; each cycle is reported once, at the node
	// where it was found closing.
	const (
		fresh = iota
		open
		shut
	)
	state := map[*Node]int{}
	var stack []*Node
	var walk func(n *Node)
	walk = func(n *Node) {
		state[n] = open
		stack = append(stack, n)
		for _, d := range n.Needs {
			dn := ids[d]
			if dn == nil || dn == n {
				continue
			}
			switch state[dn] {
			case fresh:
				walk(dn)
			case open:
				at := slices.Index(stack, dn)
				var path []string
				for _, s := range stack[at:] {
					path = append(path, s.ID)
				}
				add(n, "needs %s: a cycle (%s needs %s)", d, strings.Join(path, " needs "), d)
			}
		}
		stack = stack[:len(stack)-1]
		state[n] = shut
	}
	for _, n := range next.Nodes {
		if state[n] == fresh && ids[n.ID] == n {
			walk(n)
		}
	}
	sort.SliceStable(probs, func(i, j int) bool { return probs[i].Line < probs[j].Line })
	return probs
}
