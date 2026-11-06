package taskgraph

import (
	"strings"
	"testing"
)

// add makes a node or fails the test.
func add(t *testing.T, g *Graph, title string, needs ...string) string {
	t.Helper()
	n, err := g.Add(Node{Title: title, DoneWhen: title + " works", Needs: needs})
	if err != nil {
		t.Fatalf("add %s: %v", title, err)
	}
	return n.ID
}

func mustStatus(t *testing.T, g *Graph, id string, to Status, why string) {
	t.Helper()
	if err := g.SetStatus(id, to, why); err != nil {
		t.Fatalf("%s -> %s: %v", id, to, err)
	}
}

func ids(ns []*Node) string {
	var s []string
	for _, n := range ns {
		s = append(s, n.ID)
	}
	return strings.Join(s, ",")
}

func TestAdd(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	for name, tc := range map[string]struct {
		n    Node
		want string
	}{
		"no title":      {Node{DoneWhen: "x"}, "title"},
		"no done_when":  {Node{Title: "x"}, "done_when"},
		"unknown need":  {Node{Title: "x", DoneWhen: "y", Needs: []string{"9"}}, "no node 9"},
		"replace none":  {Node{Title: "x", DoneWhen: "y", Replaces: "9"}, "no node 9"},
		"replace live":  {Node{Title: "x", DoneWhen: "y", Replaces: a}, "drop it first"},
		"blank title":   {Node{Title: "  \n", DoneWhen: "y"}, "title"},
		"blank need ok": {Node{Title: "x", DoneWhen: "y", Needs: []string{" ", a, a}}, ""},
	} {
		_, err := g.Add(tc.n)
		switch {
		case tc.want == "" && err != nil:
			t.Errorf("%s: %v", name, err)
		case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)):
			t.Errorf("%s: err %v, want %q", name, err, tc.want)
		}
	}
	n := g.Nodes[len(g.Nodes)-1]
	if len(n.Needs) != 1 || n.Status != Todo || n.ID != "2" {
		t.Errorf("node: %+v", n)
	}
	// A failed add leaves no node and burns no id.
	if len(g.Nodes) != 2 || g.Seq != 2 {
		t.Errorf("after refusals: %d nodes, seq %d", len(g.Nodes), g.Seq)
	}
}

func TestCycleRefused(t *testing.T) {
	for name, tc := range map[string]struct {
		id    string
		needs []string
		ok    bool
	}{
		"self":          {"1", []string{"1"}, false},
		"direct":        {"1", []string{"2"}, false},
		"transitive":    {"1", []string{"3"}, false},
		"diamond ok":    {"3", []string{"1", "2"}, true},
		"unrelated ok":  {"2", nil, true},
		"gone":          {"1", []string{"9"}, false},
		"skip level ok": {"3", []string{"1"}, true},
	} {
		g := &Graph{} // 1 <- 2 <- 3, a fresh one each so cases do not mix
		a := add(t, g, "a")
		b := add(t, g, "b", a)
		add(t, g, "c", b)
		needs := tc.needs
		err := g.Update(tc.id, Patch{Needs: &needs})
		if (err == nil) != tc.ok {
			t.Errorf("%s: %v", name, err)
		}
		if !tc.ok && len(g.Get(tc.id).Needs) != len(map[string][]string{"1": nil, "2": {"1"}, "3": {"2"}}[tc.id]) {
			t.Errorf("%s: a refused update changed needs: %v", name, g.Get(tc.id).Needs)
		}
	}
}

func TestTransitions(t *testing.T) {
	all := []Status{Todo, Doing, Done, Blocked, Dropped}
	allowed := map[Status]string{
		Todo: "doing blocked dropped", Doing: "done todo blocked dropped",
		Blocked: "todo dropped", Done: "todo", Dropped: "todo",
	}
	for _, from := range all {
		for _, to := range all {
			g := &Graph{}
			id := add(t, g, "x")
			// Walk to `from` by an allowed path.
			switch from {
			case Doing:
				mustStatus(t, g, id, Doing, "")
			case Done:
				mustStatus(t, g, id, Doing, "")
				mustStatus(t, g, id, Done, "ok")
			case Blocked:
				mustStatus(t, g, id, Blocked, "stuck")
			case Dropped:
				mustStatus(t, g, id, Dropped, "no")
			}
			err := g.SetStatus(id, to, "why")
			want := strings.Contains(" "+allowed[from]+" ", " "+string(to)+" ")
			if from == to {
				want = false
			}
			if (err == nil) != want {
				t.Errorf("%s -> %s: err %v, allowed %v", from, to, err, want)
			}
		}
	}
}

func TestStatusNeedsAnAccountOrReason(t *testing.T) {
	g := &Graph{}
	id := add(t, g, "x")
	mustStatus(t, g, id, Doing, "")
	for _, to := range []Status{Done, Blocked, Dropped} {
		if err := g.SetStatus(id, to, "  "); err == nil {
			t.Errorf("%s with no words was allowed", to)
		}
	}
	if g.Get(id).Status != Doing {
		t.Errorf("a refused change moved the node: %s", g.Get(id).Status)
	}
	mustStatus(t, g, id, Done, "tests pass")
	if g.Get(id).Account != "tests pass" {
		t.Errorf("account: %q", g.Get(id).Account)
	}
	mustStatus(t, g, id, Todo, "") // reopened: the account goes
	if g.Get(id).Account != "" {
		t.Errorf("account kept after reopening: %q", g.Get(id).Account)
	}
	mustStatus(t, g, id, Blocked, "waiting on the user")
	if g.Get(id).Why != "waiting on the user" {
		t.Errorf("why: %q", g.Get(id).Why)
	}
	mustStatus(t, g, id, Todo, "")
	if g.Get(id).Why != "" {
		t.Errorf("why kept after unblocking: %q", g.Get(id).Why)
	}
}

func TestReadyNeedsEveryNeedDone(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b")
	c := add(t, g, "c", a, b)
	if got := ids(g.Ready("")); got != "1,2" {
		t.Fatalf("ready: %s", got)
	}
	mustStatus(t, g, a, Doing, "")
	if got := ids(g.Ready("")); got != "2" {
		t.Errorf("a doing: ready %s", got)
	}
	mustStatus(t, g, a, Done, "ok")
	if ok, _ := g.CanStart(c); ok {
		t.Error("c ready with b open")
	}
	mustStatus(t, g, b, Doing, "")
	mustStatus(t, g, b, Done, "ok")
	if got := ids(g.Ready("")); got != c {
		t.Errorf("both done: ready %s", got)
	}
	// Waiting is computed: reopening a prerequisite holds c again.
	mustStatus(t, g, a, Todo, "")
	if ok, why := g.CanStart(c); ok || len(why) != 1 || !strings.Contains(why[0], "needs "+a+" (todo)") {
		t.Errorf("after reopening a: %v %v", ok, why)
	}
	if g.Get(c).Status != Todo {
		t.Errorf("waiting changed c's stored status: %s", g.Get(c).Status)
	}
}

func TestStartRefusedWhileWaiting(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b", a)
	if err := g.SetStatus(b, Doing, ""); err == nil || !strings.Contains(err.Error(), "needs 1 (todo)") {
		t.Errorf("start b: %v", err)
	}
	mustStatus(t, g, a, Doing, "")
	mustStatus(t, g, a, Done, "ok")
	mustStatus(t, g, b, Doing, "")
	// Done is checked against the edges too.
	mustStatus(t, g, a, Todo, "")
	if err := g.SetStatus(b, Done, "ok"); err == nil {
		t.Error("b done while its prerequisite was reopened")
	}
}

func TestDroppedDoesNotRelease(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b", a)
	mustStatus(t, g, a, Dropped, "wrong approach")
	ok, why := g.CanStart(b)
	if ok {
		t.Fatal("a dependent of a dropped node started")
	}
	if len(why) != 1 || !strings.Contains(why[0], "dropped (wrong approach)") || !strings.Contains(why[0], "re-point") {
		t.Errorf("reason: %v", why)
	}
	// Replace, then re-point: the dependent waits on the new node.
	r, err := g.Add(Node{Title: "a2", DoneWhen: "x", Replaces: a})
	if err != nil {
		t.Fatal(err)
	}
	if _, why := g.CanStart(b); !strings.Contains(why[0], "to "+r.ID) {
		t.Errorf("reason does not name the replacement: %v", why)
	}
	needs := []string{r.ID}
	if err := g.Update(b, Patch{Needs: &needs}); err != nil {
		t.Fatal(err)
	}
	if _, why := g.CanStart(b); len(why) != 1 || !strings.Contains(why[0], "needs "+r.ID+" (todo)") {
		t.Errorf("after re-pointing: %v", why)
	}
	mustStatus(t, g, r.ID, Doing, "")
	mustStatus(t, g, r.ID, Done, "ok")
	if ok, _ := g.CanStart(b); !ok {
		t.Error("b not ready after its replacement prerequisite is done")
	}
	// The dropped node is kept.
	if n := g.Get(a); n == nil || n.Status != Dropped {
		t.Errorf("dropped node: %+v", n)
	}
}

func TestUpdate(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b", a)
	s := func(v string) *string { return &v }
	if err := g.Update(b, Patch{Title: s("  new   title "), Goal: s("g"), Notes: s("n"), DoneWhen: s("dw"), Files: &[]string{"x.go", "x.go", " "}}); err != nil {
		t.Fatal(err)
	}
	n := g.Get(b)
	if n.Title != "new title" || n.Goal != "g" || n.DoneWhen != "dw" || n.Notes != "n" || len(n.Files) != 1 {
		t.Errorf("%+v", n)
	}
	for name, p := range map[string]Patch{
		"empty title":     {Title: s(" ")},
		"empty done_when": {DoneWhen: s("")},
		"status done":     {Status: ptr(Done)},
		"status dropped":  {Status: ptr(Dropped)},
		"start waiting":   {Status: ptr(Doing)},
	} {
		before := *g.Get(b)
		if err := g.Update(b, p); err == nil {
			t.Errorf("%s: allowed", name)
		}
		if got := *g.Get(b); got.Title != before.Title || got.Status != before.Status || got.DoneWhen != before.DoneWhen {
			t.Errorf("%s: a refused update changed the node: %+v", name, got)
		}
	}
	// A failing status change undoes the needs change in the same patch.
	c := add(t, g, "c")
	needs := []string{c}
	if err := g.Update(b, Patch{Needs: &needs, Status: ptr(Doing)}); err == nil {
		t.Error("start with an open need")
	}
	if got := g.Get(b).Needs; len(got) != 1 || got[0] != a {
		t.Errorf("needs after a refused patch: %v", got)
	}
	if err := g.Update("9", Patch{}); err == nil {
		t.Error("update of a missing node")
	}
	// Start and unblock through update.
	mustStatus(t, g, c, Blocked, "stuck")
	if err := g.Update(c, Patch{Status: ptr(Todo)}); err != nil {
		t.Error(err)
	}
	if err := g.Update(c, Patch{Status: ptr(Doing)}); err != nil {
		t.Error(err)
	}
}

func ptr[T any](v T) *T { return &v }

func TestBlockedIsNotWaiting(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b", a)
	if g.Get(b).Status != Todo {
		t.Fatal("a waiting node is todo")
	}
	mustStatus(t, g, a, Blocked, "need creds")
	if ok, _ := g.CanStart(b); ok {
		t.Error("dependent of a blocked node is ready")
	}
	if got := ids(g.Ready("")); got != "" {
		t.Errorf("ready: %q", got)
	}
}

func TestSubtreeAndOwnership(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b", a)
	c := add(t, g, "c", b)
	d := add(t, g, "d", a)
	e := add(t, g, "e") // unrelated
	f := add(t, g, "f", c, e)
	for id, want := range map[string]string{a: "1,2,3,4,6", b: "2,3,6", c: "3,6", d: "4", e: "5,6", f: "6"} {
		if got := ids(g.Subtree(id)); got != want {
			t.Errorf("subtree %s: %s, want %s", id, got, want)
		}
	}
	if g.Subtree("9") != nil {
		t.Error("subtree of a missing node")
	}
	if err := g.Assign(b, "2"); err != nil {
		t.Fatal(err)
	}
	for id, want := range map[string]string{a: "", b: "2", c: "2", d: "", e: "", f: "2"} {
		if got := g.Get(id).Owner; got != want {
			t.Errorf("owner of %s: %q, want %q", id, got, want)
		}
	}
	// Another owner cannot take part of it; the same one can again.
	if err := g.Assign(c, "3"); err == nil || !strings.Contains(err.Error(), "owned by 2") {
		t.Errorf("assign owned: %v", err)
	}
	if g.Get(c).Owner != "2" {
		t.Error("a refused assign changed an owner")
	}
	if err := g.Assign(c, "2"); err != nil {
		t.Error(err)
	}
	// Ready is per owner.
	if got := ids(g.Ready("")); got != "1,5" {
		t.Errorf("main ready: %s", got)
	}
	mustStatus(t, g, a, Doing, "")
	mustStatus(t, g, a, Done, "ok")
	if got := ids(g.Ready("2")); got != "2" {
		t.Errorf("owner 2 ready: %s", got)
	}
}

func TestTree(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	b := add(t, g, "b")
	c := add(t, g, "c", a, b) // two parents
	add(t, g, "d", c)
	add(t, g, "e", a)
	var got []string
	for _, r := range g.Tree() {
		s := strings.Repeat(".", r.Depth) + r.Node.ID
		if r.Repeat {
			s += "*"
		}
		got = append(got, s)
	}
	want := "1 .3 ..4 .5 2 .3*"
	if strings.Join(got, " ") != want {
		t.Errorf("tree: %s, want %s", strings.Join(got, " "), want)
	}
	if (&Graph{}).Tree() != nil {
		t.Error("tree of nothing")
	}
}

func TestCloneIsDeep(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	add(t, g, "b", a)
	c := g.Clone()
	c.Nodes[1].Needs[0] = "x"
	c.Nodes[0].Title = "changed"
	if g.Nodes[1].Needs[0] != a || g.Nodes[0].Title != "a" {
		t.Error("clone shares memory")
	}
}

func TestBrief(t *testing.T) {
	g := &Graph{}
	if g.Brief("") != "" {
		t.Error("brief of an empty plan")
	}
	a := add(t, g, "a")
	b := add(t, g, "b")
	c := add(t, g, "c", a)
	add(t, g, "d")
	add(t, g, "e")
	add(t, g, "f")
	mustStatus(t, g, a, Doing, "")
	mustStatus(t, g, b, Blocked, "no key")
	got := g.Brief("")
	for _, want := range []string{"plan: 0 of 6 done", "doing: 1 a", "ready: 4 d; 5 e; 6 f", "blocked: 2 b (no key)"} {
		if !strings.Contains(got, want) {
			t.Errorf("brief lacks %q:\n%s", want, got)
		}
	}
	// A dependent of a dropped node is named.
	mustStatus(t, g, a, Dropped, "bad")
	if got := g.Brief(""); !strings.Contains(got, "waiting on a dropped node") || !strings.Contains(got, c+" c") {
		t.Errorf("stuck not shown:\n%s", got)
	}
	// At most five lines, however much there is.
	for i := 0; i < 20; i++ {
		add(t, g, "x")
	}
	if n := strings.Count(g.Brief(""), "\n") + 1; n > 5 {
		t.Errorf("brief is %d lines", n)
	}
	// Nothing open: no line.
	h := &Graph{}
	x := add(t, h, "x")
	mustStatus(t, h, x, Doing, "")
	mustStatus(t, h, x, Done, "ok")
	if h.Brief("") != "" {
		t.Errorf("brief with nothing open: %q", h.Brief(""))
	}
	y := add(t, h, "y")
	mustStatus(t, h, y, Dropped, "no")
	if h.Brief("") != "" {
		t.Errorf("brief with only dropped and done: %q", h.Brief(""))
	}
}

func TestBriefIsPerOwner(t *testing.T) {
	g := &Graph{}
	a := add(t, g, "a")
	add(t, g, "b", a)
	add(t, g, "c")
	if err := g.Assign(a, "1"); err != nil {
		t.Fatal(err)
	}
	if got := g.Brief(""); !strings.Contains(got, "ready: 3 c") || strings.Contains(got, "1 a") {
		t.Errorf("main:\n%s", got)
	}
	if got := g.Brief("1"); !strings.Contains(got, "ready: 1 a") {
		t.Errorf("agent 1:\n%s", got)
	}
}

func TestBriefIsEmptyForOwnerWithNoNodes(t *testing.T) {
	g := &Graph{}
	if _, err := g.Add(Node{Title: "a", DoneWhen: "a works", Owner: "other"}); err != nil {
		t.Fatal(err)
	}
	if got := g.Brief("me"); got != "" {
		t.Errorf("brief for an owner with no nodes = %q, want empty", got)
	}
}

func TestClipNonPositiveMaxDoesNotPanic(t *testing.T) {
	for _, n := range []int{0, -1} {
		if got := Clip("abc", n); got != "" {
			t.Errorf("Clip(abc, %d) = %q", n, got)
		}
	}
	if got := Clip("abc", 1); got != "…" {
		t.Errorf("Clip(abc, 1) = %q", got)
	}
}
