package taskgraph

import (
	"strings"
	"testing"
)

// An unknown id must name the nodes that exist, so the model does not spend a
// plan_show round trip to learn them. Open nodes carry a clipped title; closed
// ones are bare ids.
func TestUnknownIDListsOpenNodesWithTitlesAndClosedNodesBare(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "first", "done_when": "x"})
	call(st, "", ToolAdd, map[string]any{"title": "second node with a title that is far too long to list", "done_when": "x"})
	call(st, "", ToolAdd, map[string]any{"title": "third", "done_when": "x"})
	call(st, "", ToolUpdate, map[string]any{"id": "1", "status": "doing"})
	call(st, "", ToolClose, map[string]any{"id": "1", "outcome": "done", "account": "ok"})
	call(st, "", ToolClose, map[string]any{"id": "3", "outcome": "drop", "reason": "no"})
	out := call(st, "", ToolShow, map[string]any{"id": "71"})
	want := "ERROR: plan: no node 71. Open nodes: 2 (second node with a title that…). Closed: 1, 3."
	if out != want {
		t.Errorf("got %s\nwant %s", out, want)
	}
}

func TestUnknownIDListCapsAtTwentyAndCountsTheRest(t *testing.T) {
	var g Graph
	for i := 0; i < 25; i++ {
		if _, err := g.Add(Node{Title: "n", DoneWhen: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	msg := g.NoNode("99")
	if strings.Count(msg, " (n)") != 20 || !strings.HasSuffix(msg, " +5 more.") || strings.Contains(msg, "Closed:") {
		t.Errorf("got %s", msg)
	}
}

func TestUnknownIDOnAnEmptyPlanSaysThePlanHasNoNodes(t *testing.T) {
	out := call(newMemStore(Graph{}, nil), "", ToolShow, map[string]any{"id": "1"})
	if want := "ERROR: plan: no node 1. The plan has no nodes."; out != want {
		t.Errorf("got %s", out)
	}
}

// Every site that refuses an unknown id goes through Graph.NoNode.
func TestEveryUnknownIDSiteListsTheNodes(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "keep", "done_when": "x"})
	for name, tc := range map[string]struct {
		tool string
		args map[string]any
	}{
		"show":    {ToolShow, map[string]any{"id": "9"}},
		"update":  {ToolUpdate, map[string]any{"id": "9", "title": "t"}},
		"close":   {ToolClose, map[string]any{"id": "9", "outcome": "done", "account": "x"}},
		"needs":   {ToolAdd, map[string]any{"title": "t", "done_when": "x", "needs": []string{"9"}}},
		"replace": {ToolAdd, map[string]any{"title": "t", "done_when": "x", "replaces": "9"}},
	} {
		if out := call(st, "", tc.tool, tc.args); !strings.Contains(out, "no node 9. Open nodes: 1 (keep).") {
			t.Errorf("%s: %s", name, out)
		}
	}
	g := view(t, st)
	for name, err := range map[string]error{
		"Graph.Update":    g.Update("9", Patch{}),
		"Graph.SetStatus": g.SetStatus("9", Done, "x"),
		"Graph.Assign":    g.Assign("9", "o"),
	} {
		if err == nil || !strings.Contains(err.Error(), "Open nodes: 1 (keep).") {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, why := g.CanStart("9"); len(why) != 1 || !strings.Contains(why[0], "Open nodes: 1 (keep).") {
		t.Errorf("CanStart: %v", why)
	}
}
