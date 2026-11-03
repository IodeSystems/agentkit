package taskgraph

import (
	"context"
	"errors"
	"slices"
	"testing"
)

func changeOf(cs []Change, id string) *Change {
	for i := range cs {
		if cs[i].ID == id {
			return &cs[i]
		}
	}
	return nil
}

// Two sub-agents share a plan: 1 owns A, 2 owns B, and B needs A. When 1
// finishes A, 2 must hear that B is ready; 1 must not be told of its own write.
func TestDiffTellsTheOwnerAcrossAnEdge(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "A", DoneWhen: "x", Owner: "1"})
	g.Add(Node{Title: "B", DoneWhen: "x", Owner: "2", Needs: []string{"1"}})
	g.SetStatus("1", Doing, "")
	before := g.Clone()
	if err := g.SetStatus("1", Done, "ok"); err != nil {
		t.Fatal(err)
	}
	cs := Diff(before, g, "1")
	a, b := changeOf(cs, "1"), changeOf(cs, "2")
	if a == nil || a.Before.Status != Doing || a.After.Status != Done || !slices.Equal(a.Tell, []string{"2"}) {
		t.Errorf("A: %+v", a)
	}
	if b == nil || !b.Ready || !slices.Equal(b.Tell, []string{"2"}) {
		t.Errorf("B did not become ready for its owner: %+v", b)
	}
	if len(cs) != 2 {
		t.Errorf("changes: %+v", cs)
	}
}

// The person (or the main agent) edits a node a sub-agent owns: the owner is
// told its part changed under it. A reassignment tells both owners.
func TestDiffTellsTheOwnerOfAnEditedPart(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "A", DoneWhen: "x", Owner: "1"})
	before := g.Clone()
	title := "A, smaller"
	g.Update("1", Patch{Title: &title})
	if c := changeOf(Diff(before, g, "person"), "1"); c == nil || c.Ready || !slices.Equal(c.Tell, []string{"1"}) {
		t.Errorf("edit: %+v", c)
	}
	before = g.Clone()
	g.Get("1").Owner = "2" // a host's own reassignment; Assign refuses a node someone owns
	if c := changeOf(Diff(before, g, ""), "1"); c == nil || !slices.Equal(c.Tell, []string{"1", "2"}) {
		t.Errorf("reassign: %+v", c)
	}
}

// A write that changes nothing (a no-op update, a host flag saved through
// Update) makes no Change, so it tells no one.
func TestDiffOfNoChangeIsEmpty(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "A", DoneWhen: "x", Owner: "1", Needs: nil})
	if cs := Diff(g.Clone(), g, ""); len(cs) != 0 {
		t.Errorf("%+v", cs)
	}
}

// A Change must not share memory with the graph the store keeps: a host that
// queues the notice and a later write that changes the node in place would
// otherwise rewrite the notice.
func TestDiffCopiesNodes(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "A", DoneWhen: "x", Needs: nil})
	cs := Diff(Graph{}, g, "")
	g.Nodes[0].Title = "changed"
	if cs[0].After.Title != "A" {
		t.Error("Change.After aliases the graph")
	}
}

// A node an fn removes (nothing here does, but a host's fn may) is a Change
// with no After, told to its owner and to the dependents left waiting.
func TestDiffOfARemovedNode(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "A", DoneWhen: "x", Owner: "1"})
	g.Add(Node{Title: "B", DoneWhen: "x", Owner: "2", Needs: []string{"1"}})
	before := g.Clone()
	g.Nodes = g.Nodes[1:]
	c := changeOf(Diff(before, g, ""), "1")
	if c == nil || c.After != nil || !slices.Equal(c.Tell, []string{"1", "2"}) {
		t.Errorf("%+v", c)
	}
}

// Apply returns changes only for a write the store kept, and a store that runs
// fn twice (a retried transaction) yields the changes of the run it kept, once.
func TestApply(t *testing.T) {
	ctx := context.Background()
	st := newMemStore(Graph{}, nil)
	st.twice = true
	cs, err := Apply(ctx, st, "", func(g *Graph) error { _, err := g.Add(Node{Title: "A", DoneWhen: "x"}); return err })
	if err != nil || len(cs) != 1 || cs[0].ID != "1" || st.calls != 2 {
		t.Fatalf("retried: %+v %v calls=%d", cs, err, st.calls)
	}
	st.twice = false
	st.save = func(Graph) error { return errors.New("disk full") }
	cs, err = Apply(ctx, st, "", func(g *Graph) error { _, err := g.Add(Node{Title: "B", DoneWhen: "x"}); return err })
	if err == nil || cs != nil {
		t.Errorf("a failed save returned changes: %+v %v", cs, err)
	}
	st.save = nil
	cs, err = Apply(ctx, st, "", func(g *Graph) error { return errors.New("no") })
	if err == nil || cs != nil {
		t.Errorf("a refused fn returned changes: %+v %v", cs, err)
	}
}
