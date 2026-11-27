package taskgraph

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMarkdownReportAndResultRoundTripInCanonicalOrder(t *testing.T) {
	const text = "## 1 [x] a\n" +
		"effort: low\n" +
		"owner: sub\n" +
		"done when: d\n" +
		"report: files changed and test output\n" +
		"result: .dun/plans/s/1.report.md\n" +
		"account: ok\n"
	g, probs := ParseMarkdown(text)
	if len(probs) > 0 {
		t.Fatalf("problems: %v", probs)
	}
	n := g.Get("1")
	if n.Report != "files changed and test output" || n.Result != ".dun/plans/s/1.report.md" {
		t.Fatalf("report %q result %q", n.Report, n.Result)
	}
	if got := g.Markdown(); got != text {
		t.Errorf("round trip changed the text:\n%s\n--- want\n%s", got, text)
	}
}

func TestMarkdownReportAndResultParseOnSharedLine(t *testing.T) {
	g, probs := ParseMarkdown("## 1 [ ] a\ndone when: d · report: r · result: x · y\n")
	if len(probs) > 0 {
		t.Fatalf("problems: %v", probs)
	}
	if n := g.Get("1"); n.DoneWhen != "d" || n.Report != "r" || n.Result != "x · y" {
		t.Errorf("got %+v", n)
	}
}

func TestPlanAddSchemaHasReportNotResultAndUpdateHasBoth(t *testing.T) {
	d := New(newMemStore(Graph{}, nil), "").Defs()
	has := func(i int, k string) bool {
		_, ok := d[i].Function.Parameters.(map[string]any)["properties"].(map[string]any)[k]
		return ok
	}
	if !has(0, "report") || has(0, "result") {
		t.Errorf("plan_add: report %v result %v", has(0, "report"), has(0, "result"))
	}
	if !has(1, "report") || !has(1, "result") {
		t.Errorf("plan_update must take report and result")
	}
}

func TestUpdateSetsAndClearsReportAndResult(t *testing.T) {
	g := &Graph{}
	n, err := g.Add(Node{Title: "a", DoneWhen: "d", Report: " what\nto say ", Result: "ignored"})
	if err != nil {
		t.Fatal(err)
	}
	if n.Report != "what to say" || n.Result != "" {
		t.Fatalf("report %q result %q", n.Report, n.Result)
	}
	r, p := "x.md", "new brief"
	if err := g.Update(n.ID, Patch{Report: &p, Result: &r}); err != nil {
		t.Fatal(err)
	}
	if n.Report != "new brief" || n.Result != "x.md" {
		t.Fatalf("report %q result %q", n.Report, n.Result)
	}
	e := ""
	if err := g.Update(n.ID, Patch{Report: &e, Result: &e}); err != nil {
		t.Fatal(err)
	}
	if n.Report != "" || n.Result != "" {
		t.Errorf("not cleared: %q %q", n.Report, n.Result)
	}
}

func TestOldStoredNodeWithoutReportOrResultLoads(t *testing.T) {
	var g Graph
	if err := json.Unmarshal([]byte(`{"nodes":[{"id":"1","title":"a","status":"todo","done_when":"d"}]}`), &g); err != nil {
		t.Fatal(err)
	}
	if n := g.Get("1"); n == nil || n.Report != "" || n.Result != "" {
		t.Errorf("got %+v", n)
	}
	if b, _ := json.Marshal(g.Get("1")); strings.Contains(string(b), "report") || strings.Contains(string(b), "result") {
		t.Errorf("empty fields were written: %s", b)
	}
}
