package taskgraph

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// The four plan tools. One tool with an action enum made the model send fields
// that belong to other actions and left "required" empty; one flat schema per
// tool lets the provider enforce what is required (Carl, 2026-11-12).
const (
	ToolAdd    = "plan_add"
	ToolUpdate = "plan_update"
	ToolClose  = "plan_close"
	ToolShow   = "plan_show"
)

// Dispatcher is agent.ToolDispatcher's shape, restated so this package needs
// only llm. A value of either type converts to the other.
type Dispatcher = func(ctx context.Context, tc llm.ToolCall) (string, error)

// Option shapes New for its host.
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

// WithNote appends host sentences to the descriptions of plan_add and
// plan_show: what the plan means here (a mode, what a node's details hold) is
// the host's to say. plan_add is where the model writes a node and plan_show is
// where it reads the plan back; the other two act on a node it already knows.
func WithNote(s string) Option { return func(o *options) { o.note = s } }

// WithMayChange sets who may change a node: no hook, anyone. An actor that
// may not gets a result saying so. It also gates what a new node may need.
func WithMayChange(f func(actor string, n *Node) bool) Option {
	return func(o *options) { o.mayChange = f }
}

// Tools is the four plan tools for one actor, over one Store.
type Tools struct {
	st    Store
	actor string
	o     options
	defs  []llm.ToolDef
}

// New builds the plan tools for actor, over st. actor is whom the calls are
// for ("" when the host has none to tell apart); a node an actor adds is owned
// by it.
func New(st Store, actor string, opts ...Option) *Tools {
	t := &Tools{st: st, actor: actor}
	for _, f := range opts {
		f(&t.o)
	}
	t.defs = defs(t.o.note)
	return t
}

// Defs are the four tool definitions, in a fixed order.
func (t *Tools) Defs() []llm.ToolDef { return slices.Clone(t.defs) }

// Has reports whether name is one of the plan tools.
func (t *Tools) Has(name string) bool {
	return name == ToolAdd || name == ToolUpdate || name == ToolClose || name == ToolShow
}

// Wrap routes the plan tools' calls and passes every other call to inner.
func (t *Tools) Wrap(inner Dispatcher) Dispatcher {
	return func(ctx context.Context, tc llm.ToolCall) (string, error) {
		if !t.Has(tc.Function.Name) {
			return inner(ctx, tc)
		}
		return t.Dispatch(ctx, tc.Function.Name, tc.Function.Arguments), nil
	}
}

func prop(typ, desc string) map[string]any { return map[string]any{"type": typ, "description": desc} }

func strs(desc string) map[string]any {
	return map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": desc}
}

func mkdef(name, desc string, props map[string]any, required ...string) llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = name
	td.Function.Description = desc
	if required == nil {
		required = []string{} // a null "required" makes llama.cpp reject the request (see CLAUDE.md)
	}
	td.Function.Parameters = map[string]any{"type": "object", "properties": props, "required": required}
	return td
}

func defs(note string) []llm.ToolDef {
	if note != "" {
		note = " " + note
	}
	idProp := func(d string) map[string]any { return prop("string", d) }
	return []llm.ToolDef{
		mkdef(ToolAdd, "Add a node to the plan: a graph of tasks the user can read. Make nodes for work with "+
			"several steps; a one-step task needs none. Each node has a title, details, done_when (what has to be true for "+
			"it to be done) and needs (ids it waits on; a node is ready when everything it needs is done). Work the ready "+
			"nodes: plan_update status doing when you begin one, then plan_close done with an account of how done_when was met. "+
			"A dropped node keeps its dependents waiting: plan_add a node with replaces, then plan_update the dependents' needs. "+
			"block (plan_close) means you are stuck, with a reason; waiting on other nodes is not blocked, it shows by itself. "+
			"The user may edit the plan."+note,
			map[string]any{
				"title":     prop("string", "a short name"),
				"details":   prop("string", "what the node is for and anything the next reader needs; shown only by plan_show with an id"),
				"done_when": prop("string", "what has to be true for it to be done"),
				"needs":     strs("ids this node waits on"),
				"replaces":  prop("string", "the id of a dropped node this one stands in for"),
			}, "title", "done_when"),
		mkdef(ToolUpdate, "Change a node: its text, what it needs, or its status (doing when you begin it).",
			map[string]any{
				"id":        idProp("the node"),
				"title":     prop("string", "a short name"),
				"details":   prop("string", "what the node is for and anything the next reader needs"),
				"done_when": prop("string", "what has to be true for it to be done"),
				"needs":     strs("ids this node waits on; replaces the list (the way to re-point an edge)"),
				"status": map[string]any{"type": "string", "enum": []string{"todo", "doing"},
					"description": "doing when you begin; todo to take a block back"},
			}, "id"),
		mkdef(ToolClose, "Finish a node: done (give account), block (you are stuck; give reason) or drop (give reason).",
			map[string]any{
				"id":      idProp("the node"),
				"outcome": map[string]any{"type": "string", "enum": []string{"done", "block", "drop"}},
				"account": prop("string", "outcome done: how done_when was met"),
				"reason":  prop("string", "outcome block or drop: why"),
			}, "id", "outcome"),
		mkdef(ToolShow, "Read the plan. No id: each node's title, status and done_when, then what is ready. With id: the full node with details."+note,
			map[string]any{"id": idProp("the node; leave out for the whole plan")}),
	}
}

type args struct {
	ID       string    `json:"id"`
	Title    *string   `json:"title"`
	Details  *string   `json:"details"`
	DoneWhen *string   `json:"done_when"`
	Needs    *[]string `json:"needs"`
	Replaces string    `json:"replaces"`
	Status   *Status   `json:"status"`
	Outcome  string    `json:"outcome"`
	Account  string    `json:"account"`
	Reason   string    `json:"reason"`
}

// removed names the fields a node no longer has, so a model that sends an
// old shape (from training, or from earlier in the conversation) is told
// where the text goes instead of losing it. "action" is the old single tool.
var removed = map[string]string{
	"goal": "use details", "notes": "use details", "files": "not kept: put paths in details",
	"action": "the plan tools are plan_add, plan_update, plan_close, plan_show",
}

// unknownFields refuses a call with a field the tool's schema does not have,
// and writes nothing. json.Unmarshal drops such a field without a word, so the
// model's text is lost and nothing tells it to fix the call; an error it can
// read is what cheap models recover from (Carl, 2026-11-09). The schema is
// the list of known fields, so the two cannot drift apart.
func unknownFields(name, raw string, d llm.ToolDef) string {
	var m map[string]json.RawMessage
	if json.Unmarshal([]byte(raw), &m) != nil {
		return ""
	}
	props := d.Function.Parameters.(map[string]any)["properties"].(map[string]any)
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
	return "ERROR: plan: nothing was changed. Unknown field " + strings.Join(bad, ", ") +
		". The fields of " + name + " are: " + strings.Join(known, ", ") + ". Call " + name + " again with those."
}

func str(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// need is the refusal for a missing required field. A model that reads
// "Call X again with F" fixes the call; one that reads a bare complaint
// tends to repeat it (house wording: agent/session.go, "NOT executed").
func need(name, what, field, meaning string) string {
	return fmt.Sprintf("ERROR: plan: nothing was %s: %s is required (%s). Call %s again with %s.", what, field, meaning, name, field)
}

// Dispatch runs one call by its tool name with JSON arguments as the actor and
// returns what the model reads. A result that starts with "ERROR: plan: " is a
// refusal, and a successful result never starts with "ERROR". An error is a
// result, never a Go error.
func (t *Tools) Dispatch(ctx context.Context, name, raw string) string {
	var d llm.ToolDef
	for _, x := range t.defs {
		if x.Function.Name == name {
			d = x
		}
	}
	if d.Function.Name == "" {
		return fmt.Sprintf("ERROR: plan: unknown tool %q. The plan tools are plan_add, plan_update, plan_close, plan_show.", name)
	}
	var a args
	if strings.TrimSpace(raw) == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		return "ERROR: plan: arguments are not valid JSON: " + err.Error() + ". Call " + name + " again with a JSON object."
	}
	if msg := unknownFields(name, raw, d); msg != "" {
		return msg
	}
	a.ID = strings.TrimSpace(a.ID)
	st, actor, o := t.st, t.actor, t.o
	may := func(g *Graph, id string) bool {
		return o.mayChange == nil || o.mayChange(actor, g.Get(id))
	}

	switch name {
	case ToolShow:
		g, err := st.View(ctx)
		if err != nil {
			return "ERROR: plan: " + err.Error()
		}
		if a.ID == "" {
			out := g.List()
			// List ends with the ready line of the main agent; an actor's own is what it can work.
			if actor != "" && len(g.Nodes) > 0 {
				out = out[:strings.LastIndex(out, "\n")] + "\n" + g.ReadyLine(actor, 20)
			}
			return out
		}
		n := g.Get(a.ID)
		if n == nil {
			return "ERROR: plan: no node " + a.ID + ". Call plan_show without id to list the nodes."
		}
		return g.Detail(n)
	case ToolAdd:
		if strings.TrimSpace(str(a.Title)) == "" {
			return need(name, "added", "title", "a short name")
		}
		if strings.TrimSpace(str(a.DoneWhen)) == "" {
			return need(name, "added", "done_when", "what has to be true for the node to be done")
		}
	case ToolUpdate:
		if a.ID == "" {
			return need(name, "changed", "id", "the node")
		}
	case ToolClose:
		if a.ID == "" {
			return need(name, "changed", "id", "the node")
		}
		switch a.Outcome {
		case "done":
			if strings.TrimSpace(a.Account) == "" {
				return need(name, "closed", "account", "how done_when was met")
			}
		case "block", "drop":
			if strings.TrimSpace(a.Reason) == "" {
				return need(name, "closed", "reason", "why the node is "+map[string]string{"block": "blocked", "drop": "dropped"}[a.Outcome])
			}
		default:
			return fmt.Sprintf("ERROR: plan: nothing was closed: outcome %q is not known (done, block or drop). Call %s again with one of them.", a.Outcome, name)
		}
	}

	var out string
	changes, err := Apply(ctx, st, actor, func(g *Graph) error {
		var n *Node
		if name != ToolAdd {
			if g.Get(a.ID) == nil {
				return fmt.Errorf("no node %s. Call plan_show to list the nodes", a.ID)
			}
			if !may(g, a.ID) {
				return fmt.Errorf("node %s is not yours to change", a.ID)
			}
		}
		switch name {
		case ToolAdd:
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
		case ToolUpdate:
			if err := g.Update(a.ID, Patch{Title: a.Title, Details: a.Details, DoneWhen: a.DoneWhen,
				Needs: a.Needs, Status: a.Status}); err != nil {
				return err
			}
			n = g.Get(a.ID)
		case ToolClose:
			status, why := Done, a.Account
			switch a.Outcome {
			case "block":
				status, why = Blocked, a.Reason
			case "drop":
				status, why = Dropped, a.Reason
			}
			if err := g.SetStatus(a.ID, status, why); err != nil {
				return err
			}
			n = g.Get(a.ID)
		}
		out = g.Line(n)
		if name == ToolClose && a.Outcome == "drop" {
			if ds := g.Dependents(n.ID); len(ds) > 0 {
				out += "\n" + fmt.Sprintf("%d node(s) wait on it still (%s): plan_add a node with replaces %s, then plan_update their needs", len(ds), listNodes(ds, 5), n.ID)
			}
		}
		out += "\n" + g.ReadyLine(actor, 5)
		return nil
	})
	if err != nil {
		return "ERROR: plan: " + err.Error()
	}
	if o.onChange != nil && len(changes) > 0 {
		o.onChange(ctx, actor, changes)
	}
	return out
}
