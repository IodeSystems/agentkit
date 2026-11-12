package taskgraph

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// mine lets an actor change a node it owns, and the main one ("") any.
func mine(actor string, n *Node) bool { return actor == "" || n.Owner == actor }

func call(st Store, actor, tool string, a map[string]any) string {
	b, _ := json.Marshal(a)
	return New(st, actor, WithMayChange(mine)).Dispatch(context.Background(), tool, string(b))
}

func view(t *testing.T, st Store) Graph {
	t.Helper()
	g, err := st.View(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestToolWorkflow(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	out := call(st, "", ToolAdd, map[string]any{"title": "parse", "done_when": "tests pass"})
	if !strings.Contains(out, "1 [todo] parse") || !strings.Contains(out, "ready: 1 parse") {
		t.Errorf("add: %s", out)
	}
	out = call(st, "", ToolAdd, map[string]any{"title": "emit", "done_when": "x", "needs": []string{"1"}, "details": "writes the AST out"})
	if !strings.Contains(out, "2 [todo] emit") || !strings.Contains(out, "ready: 1 parse") {
		t.Errorf("add 2: %s", out)
	}
	if out = call(st, "", ToolUpdate, map[string]any{"id": "2", "status": "doing"}); !strings.Contains(out, "ERROR: plan: node 2 cannot start") {
		t.Errorf("start waiting: %s", out)
	}
	call(st, "", ToolUpdate, map[string]any{"id": "1", "status": "doing"})
	if out = call(st, "", ToolClose, map[string]any{"outcome": "done", "id": "1"}); !strings.Contains(out, "how done_when was met") {
		t.Errorf("done with no account: %s", out)
	}
	out = call(st, "", ToolClose, map[string]any{"outcome": "done", "id": "1", "account": "go test ok"})
	if !strings.Contains(out, "1 [done] parse") || !strings.Contains(out, "ready: 2 emit") {
		t.Errorf("done: %s", out)
	}
	if out = call(st, "", ToolShow, map[string]any{"id": "1"}); !strings.Contains(out, "account: go test ok") {
		t.Errorf("show: %s", out)
	}
	out = call(st, "", ToolShow, map[string]any{})
	if !strings.HasPrefix(out, "plan: 1 of 2 done") || !strings.Contains(out, "  2 [todo] emit") {
		t.Errorf("show all: %s", out)
	}
	if out = call(st, "", ToolShow, map[string]any{}); !strings.HasSuffix(out, "\nready: 2 emit") {
		t.Errorf("show ends with what is ready: %s", out)
	}
}

func TestToolDropAndReplace(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x"})
	call(st, "", ToolAdd, map[string]any{"title": "b", "done_when": "x", "needs": []string{"1"}})
	out := call(st, "", ToolClose, map[string]any{"outcome": "drop", "id": "1"})
	if !strings.Contains(out, "reason is required") {
		t.Errorf("drop with no reason: %s", out)
	}
	out = call(st, "", ToolClose, map[string]any{"outcome": "drop", "id": "1", "reason": "bad idea"})
	if !strings.Contains(out, "1 [dropped]") || !strings.Contains(out, "wait on it still") || !strings.Contains(out, "replaces 1") || !strings.Contains(out, "ready: none") {
		t.Errorf("drop: %s", out)
	}
	out = call(st, "", ToolAdd, map[string]any{"title": "a2", "done_when": "x", "replaces": "1"})
	if !strings.Contains(out, "(replaces 1)") {
		t.Errorf("replace: %s", out)
	}
	out = call(st, "", ToolUpdate, map[string]any{"id": "2", "needs": []string{"3"}})
	if !strings.Contains(out, "ready: 3 a2") {
		t.Errorf("re-point: %s", out)
	}
	if g := view(t, st); g.Get("1") == nil {
		t.Error("the dropped node was deleted")
	}
}

func TestToolErrors(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x"})
	for name, tc := range map[string]struct {
		tool string
		args map[string]any
		want string
	}{
		"unknown outcome": {ToolClose, map[string]any{"id": "1", "outcome": "nuke"}, "outcome"},
		"no id":           {ToolClose, map[string]any{"outcome": "done", "account": "x"}, "id is required"},
		"no outcome":      {ToolClose, map[string]any{"id": "1"}, "outcome"},
		"missing node":    {ToolClose, map[string]any{"id": "9", "outcome": "block", "reason": "x"}, "no node 9"},
		"show missing":    {ToolShow, map[string]any{"id": "9"}, "no node 9"},
		"cycle":           {ToolUpdate, map[string]any{"id": "1", "needs": []string{"1"}}, "itself"},
		"no title":        {ToolAdd, map[string]any{"done_when": "x"}, "title is required"},
		"bad status":      {ToolUpdate, map[string]any{"id": "1", "status": "done"}, "status"},
		"block no reason": {ToolClose, map[string]any{"id": "1", "outcome": "block"}, "reason is required"},
		"unknown tool":    {"plan", map[string]any{}, "plan_add"},
	} {
		if out := call(st, "", tc.tool, tc.args); !strings.HasPrefix(out, "ERROR: plan: ") || !strings.Contains(out, tc.want) {
			t.Errorf("%s: %s", name, out)
		}
	}
	if out := New(st, "").Dispatch(context.Background(), ToolShow, "{nope"); !strings.HasPrefix(out, "ERROR: plan: ") || !strings.Contains(out, "not valid JSON") {
		t.Errorf("bad json: %s", out)
	}
}

// A call with a missing required field must say what to do next and write
// nothing: a bare complaint is what a cheap model repeats.
func TestToolAddWithoutDoneWhenIsRefusedWithARetryHintAndWritesNothing(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	out := call(st, "", ToolAdd, map[string]any{"title": "a"})
	want := "ERROR: plan: nothing was added: done_when is required (what has to be true for the node to be done). Call plan_add again with done_when."
	if out != want {
		t.Errorf("got %s", out)
	}
	if g := view(t, st); len(g.Nodes) != 0 {
		t.Errorf("a refused add wrote %d nodes", len(g.Nodes))
	}
}

func TestToolCloseDoneWithoutAccountIsRefusedAndTheNodeStaysOpen(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x"})
	call(st, "", ToolUpdate, map[string]any{"id": "1", "status": "doing"})
	out := call(st, "", ToolClose, map[string]any{"id": "1", "outcome": "done"})
	if !strings.HasPrefix(out, "ERROR: plan: ") || !strings.Contains(out, "account is required") || !strings.Contains(out, "Call plan_close again") {
		t.Errorf("%s", out)
	}
	if g := view(t, st); g.Get("1").Status != Doing {
		t.Errorf("status %s", g.Get("1").Status)
	}
}

// Only a refusal starts with ERROR: a host or a model may treat that prefix as failure.
func TestToolSuccessNeverStartsWithError(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	for _, o := range []string{
		call(st, "", ToolShow, map[string]any{}),
		call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x"}),
		call(st, "", ToolShow, map[string]any{}),
		call(st, "", ToolShow, map[string]any{"id": "1"}),
	} {
		if strings.HasPrefix(o, "ERROR") {
			t.Errorf("success starts with ERROR: %s", o)
		}
	}
}

// The old single tool took action; a model that still sends it is told the new names.
func TestToolActionFieldGivesTheHintAndWritesNothing(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	out := call(st, "", ToolAdd, map[string]any{"action": "add", "title": "a", "done_when": "x"})
	if !strings.HasPrefix(out, "ERROR: plan: ") || !strings.Contains(out, "the plan tools are plan_add, plan_update, plan_close, plan_show") {
		t.Errorf("%s", out)
	}
	if g := view(t, st); len(g.Nodes) != 0 {
		t.Errorf("a refused call wrote %d nodes", len(g.Nodes))
	}
}

func TestToolWrapRoutesPlanCallsAndPassesTheRest(t *testing.T) {
	tl := New(newMemStore(Graph{}, nil), "")
	d := tl.Wrap(func(_ context.Context, tc llm.ToolCall) (string, error) { return "inner:" + tc.Function.Name, nil })
	var tc llm.ToolCall
	tc.Function.Name = "other"
	if out, _ := d(context.Background(), tc); out != "inner:other" {
		t.Errorf("%s", out)
	}
	tc.Function.Name, tc.Function.Arguments = ToolShow, "{}"
	if out, _ := d(context.Background(), tc); out != "plan: empty" {
		t.Errorf("%s", out)
	}
	if !tl.Has(ToolClose) || tl.Has("plan") {
		t.Error("Has")
	}
}

func TestToolOwnership(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "a", DoneWhen: "x"})
	g.Add(Node{Title: "b", DoneWhen: "x", Needs: []string{"1"}})
	g.Add(Node{Title: "c", DoneWhen: "x"})
	if err := g.Assign("1", "7"); err != nil {
		t.Fatal(err)
	}
	st := newMemStore(g, nil)
	if out := call(st, "7", ToolUpdate, map[string]any{"id": "3", "title": "mine now"}); !strings.Contains(out, "not yours") {
		t.Errorf("actor changed main's node: %s", out)
	}
	if out := call(st, "7", ToolUpdate, map[string]any{"id": "2", "title": "b2"}); !strings.Contains(out, "2 [todo] b2") {
		t.Errorf("actor changed its own: %s", out)
	}
	// It adds under its part, and the node is its own.
	if out := call(st, "7", ToolAdd, map[string]any{"title": "d", "done_when": "x", "needs": []string{"2"}}); !strings.Contains(out, "(owner 7)") {
		t.Errorf("add: %s", out)
	}
	if out := call(st, "7", ToolAdd, map[string]any{"title": "e", "done_when": "x", "needs": []string{"3"}}); !strings.Contains(out, "not yours") {
		t.Errorf("actor hung a node off main's: %s", out)
	}
	// The main agent may change any node.
	if out := call(st, "", ToolUpdate, map[string]any{"id": "2", "title": "b3"}); !strings.Contains(out, "b3") {
		t.Errorf("main: %s", out)
	}
	// A sub-agent's ready line is its own.
	if out := call(st, "7", ToolShow, map[string]any{}); !strings.HasSuffix(out, "\nready: 1 a") {
		t.Errorf("ready: %s", out)
	}
}

// A write the store failed to save, or the graph refused, is told to no one:
// the hook fires only for a change the plan kept.
func TestToolTellsOnlyKeptChanges(t *testing.T) {
	fail := false
	st := newMemStore(Graph{}, func(Graph) error {
		if fail {
			return errors.New("disk full")
		}
		return nil
	})
	var told [][]Change
	tl := New(st, "", WithOnChange(func(_ context.Context, _ string, cs []Change) { told = append(told, cs) }))
	run := func(tool string, a map[string]any) string {
		b, _ := json.Marshal(a)
		return tl.Dispatch(context.Background(), tool, string(b))
	}

	run(ToolAdd, map[string]any{"title": "a", "done_when": "x"})
	if len(told) != 1 || len(told[0]) != 1 || told[0][0].ID != "1" || told[0][0].Before != nil {
		t.Fatalf("add: %+v", told)
	}
	fail = true
	if out := run(ToolAdd, map[string]any{"title": "b", "done_when": "x"}); !strings.Contains(out, "disk full") {
		t.Errorf("save failure: %s", out)
	}
	fail = false
	run(ToolClose, map[string]any{"id": "1", "outcome": "done", "account": "x"}) // refused: 1 is todo, not doing
	run(ToolShow, map[string]any{})
	if len(told) != 1 {
		t.Errorf("told of a write that was not kept, or of a read: %d", len(told))
	}
}

func TestDefs(t *testing.T) {
	d := New(newMemStore(Graph{}, nil), "", WithNote("Host words.")).Defs()
	bare := New(newMemStore(Graph{}, nil), "").Defs()
	want := []struct {
		name     string
		required []string
		props    []string
		note     bool
	}{
		{ToolAdd, []string{"title", "done_when"}, []string{"title", "details", "done_when", "needs", "replaces"}, true},
		{ToolUpdate, []string{"id"}, []string{"id", "title", "details", "done_when", "needs", "status"}, false},
		{ToolClose, []string{"id", "outcome"}, []string{"id", "outcome", "account", "reason"}, false},
		{ToolShow, []string{}, []string{"id"}, true},
	}
	if len(d) != len(want) {
		t.Fatalf("%d defs", len(d))
	}
	for i, w := range want {
		f := d[i].Function
		if f.Name != w.name {
			t.Errorf("def %d is %s", i, f.Name)
		}
		pm := f.Parameters.(map[string]any)
		if got := pm["required"]; !reflect.DeepEqual(got, w.required) {
			t.Errorf("%s required %v, want %v", f.Name, got, w.required)
		}
		props := pm["properties"].(map[string]any)
		if len(props) != len(w.props) {
			t.Errorf("%s has %d properties", f.Name, len(props))
		}
		for _, k := range w.props {
			if _, ok := props[k]; !ok {
				t.Errorf("%s: no %s parameter", f.Name, k)
			}
		}
		if has := strings.HasSuffix(f.Description, " Host words."); has != w.note {
			t.Errorf("%s: host note present=%v, want %v", f.Name, has, w.note)
		}
		for _, x := range []string{"plan mode", "approve", "files"} {
			if strings.Contains(bare[i].Function.Description, x) {
				t.Errorf("%s names %q: the host says that", f.Name, x)
			}
		}
	}
}

func TestToolWithoutAHookAnyoneMayChange(t *testing.T) {
	g := Graph{}
	g.Add(Node{Title: "a", DoneWhen: "x", Owner: "z"})
	st := newMemStore(g, nil)
	if out := New(st, "q").Dispatch(context.Background(), ToolUpdate, `{"id":"1","title":"b"}`); !strings.Contains(out, "1 [todo] b") {
		t.Errorf("%s", out)
	}
	if out := New(st, "q", WithMayChange(func(string, *Node) bool { return false })).Dispatch(context.Background(), ToolUpdate, `{"id":"1","title":"c"}`); !strings.Contains(out, "not yours") {
		t.Errorf("%s", out)
	}
}

func TestShowListsDoneWhenAndOnlyShowIDGivesDetails(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	call(st, "", ToolAdd, map[string]any{"title": "parse", "done_when": "tests pass", "details": "SECRET-DETAILS for the next reader"})
	all := call(st, "", ToolShow, map[string]any{})
	if strings.Contains(all, "SECRET-DETAILS") {
		t.Errorf("whole-plan show carries details; a long plan is re-read often:\n%s", all)
	}
	if !strings.Contains(all, "1 [todo] parse") || !strings.Contains(all, "done when: tests pass") {
		t.Errorf("whole-plan show lacks the status line or done_when:\n%s", all)
	}
	one := call(st, "", ToolShow, map[string]any{"id": "1"})
	if !strings.Contains(one, "details: SECRET-DETAILS for the next reader") {
		t.Errorf("show <id> lacks details:\n%s", one)
	}
}

// A model that sends a removed or made-up field must be told, with nothing
// written: a silent drop loses its text and nothing tells it to fix the call.
func TestToolRefusesUnknownFieldsAndNamesTheReplacement(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	out := call(st, "", ToolAdd, map[string]any{"title": "parse", "done_when": "tests pass", "goal": "g", "thought": "x"})
	for _, want := range []string{"nothing was changed", `"goal": use details`, `"thought"`, "details, done_when", "Call plan_add again"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in %s", want, out)
		}
	}
	if g := view(t, st); len(g.Nodes) != 0 {
		t.Errorf("a refused call wrote %d nodes", len(g.Nodes))
	}
	if out = call(st, "", ToolAdd, map[string]any{"title": "parse", "done_when": "tests pass", "details": "d"}); !strings.Contains(out, "1 [todo] parse") {
		t.Errorf("known fields only: %s", out)
	}
}
