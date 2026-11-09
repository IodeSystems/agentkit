package taskgraph

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// ToolName is the model's one tool for the plan.
const ToolName = "plan"

// Option shapes Tool for its host.
type Option func(*options)

type options struct {
	note      string
	mayChange func(actor string, n *Node) bool
	onChange  func(ctx context.Context, actor string, changes []Change)
}

// WithOnChange gives the host every write the tool made: the changes (see
// Diff for who each names in Tell) after the store kept them, never for a
// refused or failed write, and only when there is at least one. It runs on the
// tool call's goroutine, outside Update, so it may View the store; it should
// hand the notices off (agent.Session.Inject) rather than run a turn.
func WithOnChange(f func(ctx context.Context, actor string, changes []Change)) Option {
	return func(o *options) { o.onChange = f }
}

// WithNote appends host sentences to the tool's description: what the plan
// means here (a mode, what a node's details hold) is the host's to say.
func WithNote(s string) Option { return func(o *options) { o.note = s } }

// WithMayChange sets who may change a node: no hook, anyone. An actor that
// may not gets a result saying so. It also gates what a new node may need.
func WithMayChange(f func(actor string, n *Node) bool) Option {
	return func(o *options) { o.mayChange = f }
}

// Tool is the plan tool for actor, over st: its definition and its handler.
// actor is whom the calls are for ("" when the host has none to tell apart);
// a node an actor adds is owned by it.
func Tool(st Store, actor string, opts ...Option) (llm.ToolDef, func(ctx context.Context, args string) string) {
	var o options
	for _, f := range opts {
		f(&o)
	}
	return def(o.note), func(ctx context.Context, raw string) string { return handle(ctx, st, actor, raw, o) }
}

func def(note string) llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = ToolName
	td.Function.Description = "Keep the plan: a graph of tasks the user can read. Make nodes for work with " +
		"several steps; a one-step task needs none. Each node has a title, details, done_when (what has to be true for " +
		"it to be done) and needs (ids it waits on; a node is ready when everything it needs is done). Work the ready " +
		"nodes: update status doing when you begin one, then done with an account of how done_when was met. A dropped " +
		"node keeps its dependents waiting: add a node with replaces, then update the dependents' needs. block means " +
		"you are stuck, with a reason; waiting on other nodes is not blocked, it shows by itself. Actions: add, " +
		"update, show (no id lists each node's title, status and done_when; show <id> gives the full node with its details), ready, done, block, drop. The user may edit the plan."
	if note != "" {
		td.Function.Description += " " + note
	}
	td.Function.Parameters = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action":    map[string]any{"type": "string", "enum": []string{"add", "update", "show", "ready", "done", "block", "drop"}},
			"id":        map[string]any{"type": "string", "description": "update/show/done/block/drop: the node"},
			"title":     map[string]any{"type": "string", "description": "add/update: a short name"},
			"details":   map[string]any{"type": "string", "description": "add/update: what the node is for and anything the next reader needs; shown only by show <id>"},
			"done_when": map[string]any{"type": "string", "description": "add (required)/update: what has to be true for it to be done"},
			"needs":     map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "add/update: ids this node waits on; an update replaces the list (the way to re-point an edge)"},
			"replaces":  map[string]any{"type": "string", "description": "add: the id of a dropped node this one stands in for"},
			"status":    map[string]any{"type": "string", "enum": []string{"todo", "doing"}, "description": "update: doing when you begin; todo to take a block back"},
			"account":   map[string]any{"type": "string", "description": "done: how done_when was met"},
			"reason":    map[string]any{"type": "string", "description": "block/drop: why"},
		},
		"required": []string{"action"},
	}
	return td
}

type args struct {
	Action   string    `json:"action"`
	ID       string    `json:"id"`
	Title    *string   `json:"title"`
	Details  *string   `json:"details"`
	DoneWhen *string   `json:"done_when"`
	Needs    *[]string `json:"needs"`
	Replaces string    `json:"replaces"`
	Status   *Status   `json:"status"`
	Account  string    `json:"account"`
	Reason   string    `json:"reason"`
}

// removed names the fields a node no longer has, so a model that sends an
// old shape (from training, or from earlier in the conversation) is told
// where the text goes instead of losing it.
var removed = map[string]string{"goal": "use details", "notes": "use details", "files": "not kept: put paths in details"}

// unknownFields refuses a call with a field the schema does not have, and
// writes nothing. json.Unmarshal drops such a field without a word, so the
// model's text is lost and nothing tells it to fix the call; an error it can
// read is what cheap models recover from (Carl, 2026-11-09). The schema is
// the list of known fields, so the two cannot drift apart.
func unknownFields(raw string) string {
	var m map[string]json.RawMessage
	if json.Unmarshal([]byte(raw), &m) != nil {
		return ""
	}
	props := def("").Function.Parameters.(map[string]any)["properties"].(map[string]any)
	var bad []string
	for k := range m {
		if _, ok := props[k]; ok {
			continue
		}
		if hint, ok := removed[k]; ok {
			bad = append(bad, fmt.Sprintf("%q: %s", k, hint))
		} else {
			bad = append(bad, fmt.Sprintf("%q", k))
		}
	}
	if len(bad) == 0 {
		return ""
	}
	slices.Sort(bad)
	known := make([]string, 0, len(props))
	for k := range props {
		known = append(known, k)
	}
	slices.Sort(known)
	return "plan: nothing was changed. Unknown field " + strings.Join(bad, ", ") +
		". The fields are: " + strings.Join(known, ", ")
}

func str(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// handle runs one call of the tool as actor and returns what the model reads: the node touched and
// what is ready now. An error is a result, never a Go error.
func handle(ctx context.Context, st Store, actor, raw string, o options) string {
	may := func(g *Graph, id string) bool {
		return o.mayChange == nil || o.mayChange(actor, g.Get(id))
	}
	var a args
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		return "plan: arguments are not valid JSON: " + err.Error()
	}
	if msg := unknownFields(raw); msg != "" {
		return msg
	}
	a.ID = strings.TrimSpace(a.ID)
	switch a.Action {
	case "show":
		g, err := st.View(ctx)
		if err != nil {
			return "plan: " + err.Error()
		}
		if a.ID == "" {
			return g.List()
		}
		n := g.Get(a.ID)
		if n == nil {
			return "plan: no node " + a.ID
		}
		return g.Detail(n)
	case "ready":
		g, err := st.View(ctx)
		if err != nil {
			return "plan: " + err.Error()
		}
		return g.ReadyLine(actor, 20)
	case "add", "update", "done", "block", "drop":
	default:
		return fmt.Sprintf("plan: action %q: add, update, show, ready, done, block or drop", a.Action)
	}
	var out string
	changes, err := Apply(ctx, st, actor, func(g *Graph) error {
		var n *Node
		if a.Action != "add" {
			if a.ID == "" {
				return fmt.Errorf("%s needs an id", a.Action)
			}
			if g.Get(a.ID) == nil {
				return fmt.Errorf("no node %s", a.ID)
			}
			if !may(g, a.ID) {
				return fmt.Errorf("node %s is not yours to change", a.ID)
			}
		}
		switch a.Action {
		case "add":
			nn := Node{Title: str(a.Title), Details: str(a.Details), DoneWhen: str(a.DoneWhen), Replaces: a.Replaces, Owner: actor}
			if a.Needs != nil {
				nn.Needs = *a.Needs
			}
			// An actor adds under its own part only: what it needs must be its to change.
			for _, d := range nn.Needs {
				if g.Get(d) != nil && !may(g, d) {
					return fmt.Errorf("needs %s: that node is not yours", d)
				}
			}
			var err error
			if n, err = g.Add(nn); err != nil {
				return err
			}
		case "update":
			if err := g.Update(a.ID, Patch{Title: a.Title, Details: a.Details, DoneWhen: a.DoneWhen,
				Needs: a.Needs, Status: a.Status}); err != nil {
				return err
			}
			n = g.Get(a.ID)
		case "done":
			if err := g.SetStatus(a.ID, Done, a.Account); err != nil {
				return err
			}
			n = g.Get(a.ID)
		case "block":
			if err := g.SetStatus(a.ID, Blocked, a.Reason); err != nil {
				return err
			}
			n = g.Get(a.ID)
		case "drop":
			if err := g.SetStatus(a.ID, Dropped, a.Reason); err != nil {
				return err
			}
			n = g.Get(a.ID)
		}
		out = g.Line(n)
		if a.Action == "drop" {
			if ds := g.Dependents(n.ID); len(ds) > 0 {
				out += "\n" + fmt.Sprintf("%d node(s) wait on it still (%s): add a node with replaces %s, then update their needs", len(ds), listNodes(ds, 5), n.ID)
			}
		}
		out += "\n" + g.ReadyLine(actor, 5)
		return nil
	})
	if err != nil {
		return "plan: " + err.Error()
	}
	if o.onChange != nil && len(changes) > 0 {
		o.onChange(ctx, actor, changes)
	}
	return out
}
