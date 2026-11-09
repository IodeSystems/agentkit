package taskgraph

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Row is one line of the graph as a tree.
type Row struct {
	Node  *Node
	Depth int
	// Repeat: the node already showed under another parent, so this row
	// is a pointer to it and its dependents are not shown again.
	Repeat bool
}

// Tree lays the graph out for a person: a node with no needs is a root, and a
// node hangs under each node it needs, so one with several parents shows under
// each, marked as a repeat after the first. Siblings are in node order.
func (g *Graph) Tree() []Row {
	var rows []Row
	shown := map[string]bool{}
	var walk func(n *Node, depth int)
	walk = func(n *Node, depth int) {
		if shown[n.ID] {
			rows = append(rows, Row{Node: n, Depth: depth, Repeat: true})
			return
		}
		shown[n.ID] = true
		rows = append(rows, Row{Node: n, Depth: depth})
		for _, d := range g.Dependents(n.ID) {
			walk(d, depth+1)
		}
	}
	for _, n := range g.Nodes {
		if len(n.Needs) == 0 {
			walk(n, 0)
		}
	}
	return rows
}

// Clip cuts s to max runes, with an ellipsis; "" when max is not positive.
func Clip(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	if max <= 0 {
		return ""
	}
	r := []rune(s)
	return string(r[:max-1]) + "…"
}

// Line is a node on one line.
func (g *Graph) Line(n *Node) string {
	s := fmt.Sprintf("%s [%s] %s", n.ID, n.Status, n.Title)
	if n.Owner != "" {
		s += " (owner " + n.Owner + ")"
	}
	if n.Replaces != "" {
		s += " (replaces " + n.Replaces + ")"
	}
	return s
}

// Detail is a node in full, for the model: its fields and why it waits.
func (g *Graph) Detail(n *Node) string {
	var b strings.Builder
	b.WriteString(g.Line(n))
	add := func(k, v string) {
		if v != "" {
			b.WriteString("\n  " + k + ": " + v)
		}
	}
	add("details", n.Details)
	add("done when", n.DoneWhen)
	if len(n.Needs) > 0 {
		add("needs", strings.Join(n.Needs, ", "))
	}
	if n.Status == Blocked || n.Status == Dropped {
		add("why", n.Why)
	}
	add("account", n.Account)
	if n.Status == Todo {
		for _, w := range g.Waiting(n.ID) {
			add("waiting", w)
		}
	}
	return b.String()
}

// ReadyLine is "ready: 4 Title; 5 Title" for owner, at most max of them.
func (g *Graph) ReadyLine(owner string, max int) string {
	r := g.Ready(owner)
	if len(r) == 0 {
		return "ready: none"
	}
	return "ready: " + listNodes(r, max)
}

func listNodes(ns []*Node, max int) string {
	var parts []string
	for i, n := range ns {
		if i == max {
			parts = append(parts, fmt.Sprintf("+%d more", len(ns)-max))
			break
		}
		parts = append(parts, n.ID+" "+Clip(n.Title, 60))
	}
	return strings.Join(parts, "; ")
}

// List is the whole graph as the tree for the model: per node its Line and,
// under it, what done means. Details stay out (Detail has them): the model
// re-reads the whole plan often, and done_when is what it needs to pick the
// next node, while details are for the one node it is working.
func (g *Graph) List() string {
	if len(g.Nodes) == 0 {
		return "plan: empty"
	}
	var b strings.Builder
	done := 0
	for _, n := range g.Nodes {
		if n.Status == Done {
			done++
		}
	}
	fmt.Fprintf(&b, "plan: %d of %d done", done, len(g.Nodes))
	for _, r := range g.Tree() {
		b.WriteString("\n" + strings.Repeat("  ", r.Depth) + g.Line(r.Node))
		if r.Repeat {
			b.WriteString(" (again)")
		}
		if r.Node.Status == Todo {
			if w := g.Waiting(r.Node.ID); len(w) > 0 && !r.Repeat {
				b.WriteString(" — " + w[0])
			}
		}
		if !r.Repeat {
			b.WriteString("\n" + strings.Repeat("  ", r.Depth) + "  done when: " + Clip(r.Node.DoneWhen, 100))
		}
	}
	b.WriteString("\n" + g.ReadyLine("", 5))
	return b.String()
}

// Brief is the line a turn carries while the plan has open nodes: where it
// stands, the node in progress, the next ready ones, what is stuck. At most
// five lines; "" when nothing is open. owner is the agent it is for.
func (g *Graph) Brief(owner string) string {
	if g.Open() == 0 {
		return ""
	}
	var doing, blocked, stuck []*Node
	done, total := 0, 0
	for _, n := range g.Nodes {
		if n.Owner != owner {
			continue
		}
		total++
		switch n.Status {
		case Done:
			done++
		case Doing:
			doing = append(doing, n)
		case Blocked:
			blocked = append(blocked, n)
		case Todo:
			for _, d := range n.Needs {
				if dn := g.Get(d); dn != nil && dn.Status == Dropped {
					stuck = append(stuck, n)
					break
				}
			}
		}
	}
	if total == 0 {
		return "" // open work belongs to other owners
	}
	lines := []string{fmt.Sprintf("plan: %d of %d done", done, total)}
	if len(doing) > 0 {
		lines = append(lines, "doing: "+listNodes(doing, 2))
	}
	if r := g.Ready(owner); len(r) > 0 {
		lines = append(lines, "ready: "+listNodes(r, 3))
	}
	if len(blocked) > 0 {
		lines = append(lines, "blocked: "+g.reasons(blocked, 2))
	}
	if len(stuck) > 0 {
		lines = append(lines, "waiting on a dropped node (update needs): "+listNodes(stuck, 3))
	}
	return strings.Join(lines, "\n")
}

func (g *Graph) reasons(ns []*Node, max int) string {
	var parts []string
	for i, n := range ns {
		if i == max {
			parts = append(parts, fmt.Sprintf("+%d more", len(ns)-max))
			break
		}
		parts = append(parts, n.ID+" "+Clip(n.Title, 40)+" ("+Clip(n.Why, 60)+")")
	}
	return strings.Join(parts, "; ")
}
