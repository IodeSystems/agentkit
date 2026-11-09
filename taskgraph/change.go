package taskgraph

// Changes: what a write did to the plan, and who should hear of it.
//
// Several agents share one plan (see Store). An agent learns what it did from
// its own tool result; it does NOT learn what the others did, nor what the
// person did on a screen. So every write yields a []Change, and the host sends
// each to the owners it names, out of band: typically agent.Session.Inject of a
// KindNotification into each owner's session. Mapping an owner to a session is
// the host's (that is orchestration); this file only computes, from the graph,
// which owners a change touches.
//
// Who is told of a change to node N, never counting the actor that made it:
//
//   - N's owner, before and after (a reassignment tells both). This is what
//     tells an agent its part changed under it: the person edited a node it
//     owns, or the main agent re-pointed one of its edges.
//   - The owners of N's dependents, when N's status changed. An edge that
//     crosses owners is a dependency between agents: when sub-agent 1 finishes
//     A, sub-agent 2, whose B needs A, must hear of it.
//
// A node that BECOMES ready is its own Change (Ready set), told to its owner;
// often its only change is that something it needed was finished.
//
// What this does not do: tell the agent that delegated a part. Which agent
// handed a part to which is the host's record, not the graph's (an Owner is an
// opaque string here), so a host that wants status to flow back adds the
// delegator to Tell itself.
//
// Batching needs nothing here: KindNotification entries pending for a session
// are all claimed at the top of its next turn, so a burst of changes is one
// turn. A notice that went stale before that turn ("4 is ready", but another
// agent took 4) is what agent.NotificationPreparer is for.

import (
	"context"
	"slices"
)

// Change is one node a write changed.
type Change struct {
	ID string
	// Before is the node as it was; nil when the write added it.
	Before *Node
	// After is the node as it is now; nil when the write removed it (nothing
	// in this package removes a node, but an fn given to Apply may).
	After *Node
	// Ready: the node could not start before the write, and can now.
	Ready bool
	// Tell is the owners that should hear of it, in a stable order, without
	// the actor. "" is an owner like any other: what it means is the host's.
	Tell []string
}

// Diff is the changes from before to after, made by actor, in node order
// (removed nodes last). Fields that did not change make no Change.
func Diff(before, after Graph, actor string) []Change {
	var out []Change
	seen := map[string]bool{}
	for _, a := range after.Nodes {
		seen[a.ID] = true
		b := before.Get(a.ID)
		ready := false
		if ok, _ := after.CanStart(a.ID); ok {
			was := false
			if b != nil {
				was, _ = before.CanStart(a.ID)
			}
			ready = !was
		}
		edited := b == nil || !sameNode(b, a)
		if !edited && !ready {
			continue
		}
		c := Change{ID: a.ID, Before: copyNode(b), After: copyNode(a), Ready: ready}
		var tell []string
		if b != nil {
			tell = append(tell, b.Owner)
		}
		tell = append(tell, a.Owner)
		if b == nil || b.Status != a.Status {
			for _, d := range after.Dependents(a.ID) {
				tell = append(tell, d.Owner)
			}
		}
		c.Tell = others(tell, actor)
		out = append(out, c)
	}
	for _, b := range before.Nodes {
		if seen[b.ID] {
			continue
		}
		tell := []string{b.Owner}
		for _, d := range before.Dependents(b.ID) {
			tell = append(tell, d.Owner)
		}
		out = append(out, Change{ID: b.ID, Before: copyNode(b), Tell: others(tell, actor)})
	}
	return out
}

// Apply runs fn on the plan in st as actor and returns what it changed. It is
// the one write path: the plan tool goes through it, and so should every write
// the host makes itself (a person's edit on a screen, an assignment), or those
// changes are never told to the owners they touch.
//
// The changes are computed inside Update, from the plan Update hands fn, and
// returned only if Update succeeds: a write the store refused or failed to save
// tells no one.
func Apply(ctx context.Context, st Store, actor string, fn func(g *Graph) error) ([]Change, error) {
	var changes []Change
	err := st.Update(ctx, func(g *Graph) error {
		before := g.Clone()
		if err := fn(g); err != nil {
			return err
		}
		changes = Diff(before, *g, actor) // recomputed if the store calls fn again
		return nil
	})
	if err != nil {
		return nil, err
	}
	return changes, nil
}

// copyNode is a deep copy, so a Change never shares memory with a graph the
// store goes on to keep and change; nil stays nil.
func copyNode(n *Node) *Node {
	if n == nil {
		return nil
	}
	c := *n
	c.Needs = slices.Clone(n.Needs)
	return &c
}

func sameNode(a, b *Node) bool {
	return a.Title == b.Title && a.Details == b.Details && a.Status == b.Status && a.Owner == b.Owner &&
		a.DoneWhen == b.DoneWhen && a.Replaces == b.Replaces && a.Why == b.Why &&
		a.Account == b.Account && slices.Equal(a.Needs, b.Needs)
}

// others is owners without actor and without repeats, sorted.
func others(owners []string, actor string) []string {
	var out []string
	for _, o := range owners {
		if o != actor && !slices.Contains(out, o) {
			out = append(out, o)
		}
	}
	slices.Sort(out)
	return out
}
