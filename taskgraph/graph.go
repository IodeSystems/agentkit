// Package taskgraph is a plan the model keeps for itself: a DAG of tasks a
// person can review, shared by a main agent and the sub-agents it hands parts
// to. It holds the nodes, the edges they wait on, the rules for a status change,
// the one answer to "can this start", an opaque owner per node, the `plan` tool
// (Tool), and what each write changed and for whom (Diff, Apply).
//
// It is optional, like files and loopdetector: a host mounts it; agent does not
// import it. It imports llm (the tool definition) and stdlib only.
//
// Where the line is. A plan the model keeps decides nothing about which
// session runs, when, or why, so it is not orchestration. Everything that is
// stays with the host: storage (Store), what an owner is and which session it
// maps to, who may change what (WithMayChange), approval or a plan mode,
// delegation (spawning the agent a part is assigned to), scheduling, and how a
// change reaches an owner (WithOnChange, Apply).
package taskgraph

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Status is where a node stands.
//
// blocked is only ever somebody saying "stuck", with a reason. Waiting on
// other nodes is computed from the edges (Waiting) and never stored, so a node
// cannot go stale when a prerequisite moves.
type Status string

const (
	Todo    Status = "todo"
	Doing   Status = "doing"
	Done    Status = "done"
	Blocked Status = "blocked"
	Dropped Status = "dropped"
)

// transitions is every status change that is allowed, from → to. The one place
// that says so; callers ask SetStatus.
var transitions = map[Status][]Status{
	Todo:    {Doing, Blocked, Dropped},
	Doing:   {Done, Todo, Blocked, Dropped},
	Blocked: {Todo, Dropped},
	Done:    {Todo}, // reopened
	Dropped: {Todo}, // taken back
}

// Node is one task.
type Node struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	// Details is free text: what the node is for and anything the next reader
	// needs. Only Detail shows it; the whole-plan List leaves it out, so a long
	// plan stays cheap to re-read.
	Details string `json:"details,omitempty"`
	// Status is never "waiting": see Status.
	Status Status `json:"status"`
	// Needs are the ids this node waits on, until each is done.
	Needs []string `json:"needs,omitempty"`
	// Owner is an opaque string the host gives meaning to, kept as it is; "" is
	// nobody in particular. Ready filters on it.
	Owner string `json:"owner,omitempty"`
	// DoneWhen is what done means, set when the node is made.
	DoneWhen string `json:"done_when"`
	// Replaces is a dropped node this one stands in for. It does not move the
	// dropped node's dependents: an update of their needs does.
	Replaces string `json:"replaces,omitempty"`
	// Why is the reason a node is blocked or dropped.
	Why string `json:"why,omitempty"`
	// Account is how DoneWhen was met.
	Account string `json:"account,omitempty"`
}

// Graph is the whole plan.
type Graph struct {
	Nodes []*Node `json:"nodes"`
	// Seq numbers the next node; ids are never reused.
	Seq int `json:"seq,omitempty"`
}

// Clone is a deep copy.
func (g *Graph) Clone() Graph {
	c := *g
	c.Nodes = make([]*Node, len(g.Nodes))
	for i, n := range g.Nodes {
		cn := *n
		cn.Needs = slices.Clone(n.Needs)
		c.Nodes[i] = &cn
	}
	return c
}

// Get is the node with id, or nil.
func (g *Graph) Get(id string) *Node {
	for _, n := range g.Nodes {
		if n.ID == id {
			return n
		}
	}
	return nil
}

func clean(s string) string { return strings.Join(strings.Fields(s), " ") }

func cleanList(in []string) []string {
	var out []string
	for _, s := range in {
		if s = strings.TrimSpace(s); s != "" && !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	return out
}

// Add makes a node from n's Title, Details, Needs, Owner, DoneWhen
// and Replaces. It starts as todo.
func (g *Graph) Add(n Node) (*Node, error) {
	n.Title, n.DoneWhen, n.Details = clean(n.Title), clean(n.DoneWhen), strings.TrimSpace(n.Details)
	if n.Title == "" {
		return nil, fmt.Errorf("a node needs a title")
	}
	if n.DoneWhen == "" {
		return nil, fmt.Errorf("a node needs done_when: what has to be true for it to be done")
	}
	n.Needs = cleanList(n.Needs)
	if err := g.checkNeeds("", n.Needs); err != nil {
		return nil, err
	}
	if n.Replaces != "" {
		r := g.Get(n.Replaces)
		switch {
		case r == nil:
			return nil, fmt.Errorf("replaces: no node %s", n.Replaces)
		case r.Status != Dropped:
			return nil, fmt.Errorf("replaces: node %s is %s; drop it first (a node stands in only for a dropped one)", r.ID, r.Status)
		}
	}
	g.Seq++
	n.ID = strconv.Itoa(g.Seq)
	n.Status, n.Why, n.Account = Todo, "", ""
	g.Nodes = append(g.Nodes, &n)
	return &n, nil
}

// checkNeeds refuses a need that is missing, is the node itself, or would close
// a cycle. id "" is a node not made yet, which nothing can depend on.
func (g *Graph) checkNeeds(id string, needs []string) error {
	for _, d := range needs {
		if g.Get(d) == nil {
			return fmt.Errorf("needs: no node %s", d)
		}
		if id == "" {
			continue
		}
		if d == id {
			return fmt.Errorf("needs: node %s cannot need itself", id)
		}
		if path := g.path(d, id); path != nil {
			return fmt.Errorf("needs: a cycle: %s", strings.Join(append(path, d), " needs "))
		}
	}
	return nil
}

// path is the chain of needs from `from` down to `to` ([from … to]), or nil
// when `from` does not depend on `to`.
func (g *Graph) path(from, to string) []string {
	seen := map[string]bool{}
	var walk func(id string) []string
	walk = func(id string) []string {
		if id == to {
			return []string{id}
		}
		if seen[id] {
			return nil
		}
		seen[id] = true
		n := g.Get(id)
		if n == nil {
			return nil
		}
		for _, d := range n.Needs {
			if p := walk(d); p != nil {
				return append([]string{id}, p...)
			}
		}
		return nil
	}
	return walk(from)
}

// Patch is the fields of a node an update may change; nil leaves one alone.
// Status is todo or doing only: done, blocked and dropped carry what they need
// (an account, a reason) through SetStatus.
type Patch struct {
	Title, Details, DoneWhen *string
	Needs                    *[]string
	Status                   *Status
}

// Update applies p to node id. A change of Needs is the re-pointing of an edge.
func (g *Graph) Update(id string, p Patch) error {
	n := g.Get(id)
	if n == nil {
		return fmt.Errorf("no node %s", id)
	}
	c := *n
	if p.Title != nil {
		if c.Title = clean(*p.Title); c.Title == "" {
			return fmt.Errorf("a node needs a title")
		}
	}
	if p.Details != nil {
		c.Details = strings.TrimSpace(*p.Details)
	}
	if p.DoneWhen != nil {
		if c.DoneWhen = clean(*p.DoneWhen); c.DoneWhen == "" {
			return fmt.Errorf("a node needs done_when")
		}
	}
	if p.Needs != nil {
		c.Needs = cleanList(*p.Needs)
		if err := g.checkNeeds(id, c.Needs); err != nil {
			return err
		}
	}
	if p.Status != nil {
		if *p.Status != Todo && *p.Status != Doing {
			return fmt.Errorf("status: update sets todo or doing; use done (with an account), block or drop (with a reason)")
		}
		if *p.Status != c.Status {
			// The check reads the node with its new needs; put it back if it fails.
			orig, to := *n, *p.Status
			c.Status = orig.Status
			*n = c
			if err := g.SetStatus(id, to, ""); err != nil {
				*n = orig
				return err
			}
			c = *n
		}
	}
	*n = c
	return nil
}

// SetStatus moves node id to a status, if the table allows it. why is the
// account for done and the reason for blocked and dropped; each needs one.
func (g *Graph) SetStatus(id string, to Status, why string) error {
	n := g.Get(id)
	if n == nil {
		return fmt.Errorf("no node %s", id)
	}
	why = clean(why)
	if n.Status == to {
		return fmt.Errorf("node %s is already %s", id, to)
	}
	if !slices.Contains(transitions[n.Status], to) {
		return fmt.Errorf("node %s is %s; it cannot go to %s (from %s: %s)", id, n.Status, to, n.Status, wordList(transitions[n.Status]))
	}
	switch to {
	case Doing:
		if ok, reasons := g.CanStart(id); !ok {
			return fmt.Errorf("node %s cannot start: %s", id, strings.Join(reasons, "; "))
		}
	case Done:
		if why == "" {
			return fmt.Errorf("node %s: say how done_when was met (%q)", id, n.DoneWhen)
		}
		if w := g.Waiting(id); len(w) > 0 {
			return fmt.Errorf("node %s cannot be done: %s", id, strings.Join(w, "; "))
		}
	case Blocked, Dropped:
		if why == "" {
			return fmt.Errorf("node %s: say why it is %s", id, to)
		}
	}
	n.Status = to
	n.Why = ""
	switch to {
	case Blocked, Dropped:
		n.Why = why
	case Done:
		n.Account = why
	case Todo:
		n.Account = ""
	}
	return nil
}

func wordList(ss []Status) string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = string(s)
	}
	return strings.Join(out, ", ")
}

// Waiting says what a node waits on, one reason per need that is not done,
// computed from the edges. Empty: nothing holds it. A dropped need does NOT
// release its dependents; the reason says to replace the edge.
func (g *Graph) Waiting(id string) []string {
	n := g.Get(id)
	if n == nil {
		return nil
	}
	var out []string
	for _, d := range n.Needs {
		dn := g.Get(d)
		switch {
		case dn == nil:
			out = append(out, "needs "+d+", which is gone")
		case dn.Status == Done:
		case dn.Status == Dropped:
			s := "needs " + d + ", dropped (" + dn.Why + "): re-point the need with update"
			if r := g.replacement(d); r != nil {
				s += " to " + r.ID
			}
			out = append(out, s)
		default:
			out = append(out, "needs "+d+" ("+string(dn.Status)+")")
		}
	}
	return out
}

// replacement is a node that stands in for dropped node id, if one does.
func (g *Graph) replacement(id string) *Node {
	for _, n := range g.Nodes {
		if n.Replaces == id && n.Status != Dropped {
			return n
		}
	}
	return nil
}

// CanStart is the one answer to "can this node start": it is todo and nothing
// holds it. The ready set, the tool, the screen and the turn's line all read
// this and no other.
func (g *Graph) CanStart(id string) (bool, []string) {
	n := g.Get(id)
	if n == nil {
		return false, []string{"no node " + id}
	}
	if n.Status != Todo {
		return false, []string{"it is " + string(n.Status)}
	}
	if w := g.Waiting(id); len(w) > 0 {
		return false, w
	}
	return true, nil
}

// Ready is the nodes owned by owner that can start now, in node order.
func (g *Graph) Ready(owner string) []*Node {
	var out []*Node
	for _, n := range g.Nodes {
		if n.Owner != owner {
			continue
		}
		if ok, _ := g.CanStart(n.ID); ok {
			out = append(out, n)
		}
	}
	return out
}

// Open is how many nodes are not finished: todo, doing or blocked.
func (g *Graph) Open() int {
	c := 0
	for _, n := range g.Nodes {
		if n.Status == Todo || n.Status == Doing || n.Status == Blocked {
			c++
		}
	}
	return c
}

// Dependents are the nodes that need id directly, in node order.
func (g *Graph) Dependents(id string) []*Node {
	var out []*Node
	for _, n := range g.Nodes {
		if slices.Contains(n.Needs, id) {
			out = append(out, n)
		}
	}
	return out
}

// Subtree is id and every node that hangs from it: whatever needs it, directly
// or not. It is the unit a sub-agent is given.
func (g *Graph) Subtree(id string) []*Node {
	if g.Get(id) == nil {
		return nil
	}
	in := map[string]bool{id: true}
	for changed := true; changed; {
		changed = false
		for _, n := range g.Nodes {
			if in[n.ID] {
				continue
			}
			for _, d := range n.Needs {
				if in[d] {
					in[n.ID], changed = true, true
					break
				}
			}
		}
	}
	var out []*Node
	for _, n := range g.Nodes {
		if in[n.ID] {
			out = append(out, n)
		}
	}
	return out
}

// Assign gives the subtree of id to owner. It refuses a subtree with a node
// someone else already owns.
func (g *Graph) Assign(id, owner string) error {
	sub := g.Subtree(id)
	if sub == nil {
		return fmt.Errorf("no node %s", id)
	}
	for _, n := range sub {
		if n.Owner != "" && n.Owner != owner {
			return fmt.Errorf("node %s is owned by %s", n.ID, n.Owner)
		}
	}
	for _, n := range sub {
		n.Owner = owner
	}
	return nil
}
