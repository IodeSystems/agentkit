package taskgraph

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

const canonical = "# Plan: ship the parser\n" +
	"\n## 1 [x] parse\n" +
	"effort: low\n" +
	"done when: tests pass\n" +
	"account: go test ok\n" +
	"\nThe lexer is in lex.go.\n" +
	"\n## 2 [>] emit\n" +
	"effort: high\n" +
	"needs: 1\n" +
	"done when: output matches\n" +
	"\n```go\n## not a section\nfunc f() {}\n```\n\nAfter the fence.\n" +
	"\n## 3 [!] lint\n" +
	"needs: 1, 2\n" +
	"done when: clean\n" +
	"reason: linter is down\n" +
	"\n## 4 [-] old way\n" +
	"done when: n/a\n" +
	"reason: superseded\n" +
	"\n## 5 [ ] new way\n" +
	"effort: medium\n" +
	"done when: works\n" +
	"replaces: 4\n"

func TestMarkdownRoundTripKeepsEveryBoxAndField(t *testing.T) {
	g, probs := ParseMarkdown(canonical)
	if len(probs) > 0 {
		t.Fatalf("problems: %v", probs)
	}
	if got := g.Markdown(); got != canonical {
		t.Errorf("round trip changed the text:\n%s\n--- want\n%s", got, canonical)
	}
	if g.Goal != "ship the parser" || g.Seq != 5 || g.Get("2").Status != Doing || g.Get("4").Status != Dropped {
		t.Errorf("graph: %+v", g)
	}
}

func TestMarkdownSharedLineParsesAndWritesCanonical(t *testing.T) {
	in := "## 1 [ ] a\neffort: low · done when: x · y · needs: \n"
	g, probs := ParseMarkdown(in)
	if len(probs) > 0 {
		t.Fatal(probs)
	}
	if n := g.Get("1"); n.Effort != "low" || n.DoneWhen != "x · y" {
		t.Errorf("node: %+v", n)
	}
	if want := "## 1 [ ] a\neffort: low\ndone when: x · y\n"; g.Markdown() != want {
		t.Errorf("canonical: %q", g.Markdown())
	}
}

func TestMarkdownFencedHeadingStaysInDetails(t *testing.T) {
	g, probs := ParseMarkdown("## 1 [ ] a\ndone when: x\n\n~~~\n## 2 [ ] fake\n~~~\n")
	if len(probs) > 0 || len(g.Nodes) != 1 || !strings.Contains(g.Nodes[0].Details, "## 2 [ ] fake") {
		t.Errorf("%v %+v", probs, g.Nodes)
	}
}

func TestMarkdownOrderIsTopologicalTiesByID(t *testing.T) {
	g, _ := ParseMarkdown("## 10 [ ] c\ndone when: x\nneeds: 2\n\n## 2 [ ] b\ndone when: x\n\n## 1 [ ] a\ndone when: x\nneeds: 10\n\n## 3 [ ] d\ndone when: x\n")
	var ids []string
	for _, n := range g.order() {
		ids = append(ids, n.ID)
	}
	if want := []string{"2", "3", "10", "1"}; !slices.Equal(ids, want) {
		t.Errorf("%v want %v", ids, want)
	}
}

func viewp(t *testing.T, st Store) *Graph { g := view(t, st); return &g }

func hasProblem(ps []Problem, id, sub string, line int) bool {
	for _, p := range ps {
		if p.ID == id && strings.Contains(p.Msg, sub) && (line < 0 || p.Line == line) {
			return true
		}
	}
	return false
}

func TestValidateRulesEachReportAProblemWithLine(t *testing.T) {
	cases := []struct{ name, text, id, sub string }{
		{"dup id", "## 1 [ ] a\ndone when: x\n\n## 1 [ ] b\ndone when: x\n", "1", "used twice"},
		{"missing need", "## 1 [ ] a\ndone when: x\nneeds: 9\n", "1", "needs 9"},
		{"self need", "## 1 [ ] a\ndone when: x\nneeds: 1\n", "1", "itself"},
		{"cycle", "## 1 [ ] a\ndone when: x\nneeds: 2\n\n## 2 [ ] b\ndone when: x\nneeds: 1\n", "2", "cycle"},
		{"no done when", "## 1 [ ] a\neffort: low\n", "1", "done when"},
		{"done no account", "## 1 [x] a\ndone when: x\n", "1", "account"},
		{"blocked no reason", "## 1 [!] a\ndone when: x\n", "1", "reason"},
		{"dropped no reason", "## 1 [-] a\ndone when: x\n", "1", "reason"},
		{"bad replaces", "## 1 [ ] a\ndone when: x\nreplaces: 7\n", "1", "replaces 7"},
		{"bad effort", "## 1 [ ] a\ndone when: x\neffort: huge\n", "1", "effort"},
		{"bad box", "## 1 [?] a\ndone when: x\n", "1", "box"},
		{"not a field", "## 1 [ ] a\ndone when: x\nwhat is this\n", "1", "not a field"},
	}
	for _, c := range cases {
		_, probs := ParseMarkdown(c.text)
		if !hasProblem(probs, c.id, c.sub, -1) {
			t.Errorf("%s: %v", c.name, probs)
		}
		for _, p := range probs {
			if p.Line == 0 {
				t.Errorf("%s: no line: %v", c.name, p)
			}
		}
	}
	_, probs := ParseMarkdown("## 1 [ ] a\ndone when: x\n\n## 2 [ ] b\ndone when: x\nneeds: 1\n")
	if len(probs) != 0 {
		t.Errorf("a clean plan: %v", probs)
	}
}

func TestValidateNodeDeletedIsRefused(t *testing.T) {
	prev, _ := ParseMarkdown("## 1 [ ] a\ndone when: x\n\n## 2 [ ] b\ndone when: x\n")
	next, _ := ParseMarkdown("## 1 [ ] a\ndone when: x\n")
	probs := Validate(prev, next)
	if len(probs) != 1 || probs[0].ID != "2" || !strings.Contains(probs[0].Msg, "dropped") {
		t.Errorf("%v", probs)
	}
	if len(Validate(prev, prev)) != 0 {
		t.Error("same plan refused")
	}
}

func TestValidateReportsSeveralProblemsAtOnce(t *testing.T) {
	_, probs := ParseMarkdown("## 1 [x] a\n\n## 2 [!] b\ndone when: x\neffort: max\nneeds: 8\n")
	for _, w := range [][2]string{{"1", "done when"}, {"1", "account"}, {"2", "reason"}, {"2", "effort"}, {"2", "needs 8"}} {
		if !hasProblem(probs, w[0], w[1], -1) {
			t.Errorf("missing %v in %v", w, probs)
		}
	}
	if !hasProblem(probs, "2", "reason", 3) {
		t.Errorf("line: %v", probs)
	}
}

func TestOldStoredNodeWithoutEffortOrMetaLoads(t *testing.T) {
	var g Graph
	old := `{"nodes":[{"id":"1","title":"a","status":"todo","done_when":"x"}],"seq":1}`
	if err := json.Unmarshal([]byte(old), &g); err != nil {
		t.Fatal(err)
	}
	n := g.Get("1")
	if n.Effort != "" || n.Meta != nil {
		t.Errorf("%+v", n)
	}
	b, _ := json.Marshal(g)
	if strings.Contains(string(b), "effort") || strings.Contains(string(b), "meta") || strings.Contains(string(b), "goal") {
		t.Errorf("empty fields written: %s", b)
	}
}

func TestCloneDeepCopiesMeta(t *testing.T) {
	g := Graph{Nodes: []*Node{{ID: "1", Meta: map[string]string{"k": "v"}}}}
	c := g.Clone()
	c.Nodes[0].Meta["k"] = "changed"
	if g.Nodes[0].Meta["k"] != "v" {
		t.Error("Clone shares Meta")
	}
}

func TestToolRejectsBadEffortAndKeepsGoodOne(t *testing.T) {
	st := newMemStore(Graph{}, nil)
	out := call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x", "effort": "huge"})
	if !strings.HasPrefix(out, "ERROR: plan: effort") || len(view(t, st).Nodes) != 0 {
		t.Errorf("add: %s", out)
	}
	call(st, "", ToolAdd, map[string]any{"title": "a", "done_when": "x", "effort": "high"})
	if viewp(t, st).Get("1").Effort != "high" {
		t.Error("effort not kept")
	}
	if out = call(st, "", ToolUpdate, map[string]any{"id": "1", "effort": "max"}); !strings.HasPrefix(out, "ERROR: plan: effort") {
		t.Errorf("update: %s", out)
	}
	call(st, "", ToolUpdate, map[string]any{"id": "1", "effort": "low"})
	if viewp(t, st).Get("1").Effort != "low" {
		t.Error("effort not updated")
	}
}
