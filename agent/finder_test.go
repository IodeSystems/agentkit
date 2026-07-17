package agent

import (
	"context"
	"errors"
	"testing"
)

// fakeFinder is a scripted DocFinder: it returns hits (or an error) and records
// how many times Find was called + the texts it last saw — so a test can prove
// the watermark gates re-queries and fail-open retries.
type fakeFinder struct {
	hits    []DocHit
	err     error
	calls   int
	lastTxt []string
}

func (f *fakeFinder) Find(_ context.Context, texts []string) ([]DocHit, error) {
	f.calls++
	f.lastTxt = texts
	return f.hits, f.err
}

// notices counts injected KindNotification entries with the given tag.
func notices(t *testing.T, st *memStore, tag string) []Entry {
	t.Helper()
	ctx := context.Background()
	all, err := st.Context(ctx, "s")
	if err != nil {
		t.Fatalf("context: %v", err)
	}
	var out []Entry
	for _, e := range all {
		if e.Kind == KindNotification && e.Tag == tag {
			out = append(out, e)
		}
	}
	return out
}

// clock hands out strictly increasing timestamps for deterministic ordering.
type clk struct{ n int64 }

func (c *clk) next() int64 { c.n++; return c.n }

func TestFinderPreparer_InjectsPointerAboveThreshold(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	st.Append(context.Background(), "s", Entry{ID: "u1", Kind: KindUser, Content: "how does auth work?", CreatedAt: c.next()})

	f := &fakeFinder{hits: []DocHit{
		{DocID: "d1", Title: "Auth", Score: 0.9, Line: "auth overview"},
		{DocID: "d2", Title: "Weak", Score: 0.1, Line: "unrelated"},
	}}
	p := FinderPreparer(st, f, FinderOpts{MinScore: 0.5, Now: c.next})

	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	got := notices(t, st, "rag")
	if len(got) != 1 {
		t.Fatalf("want 1 notice (d2 below threshold), got %d", len(got))
	}
	if !contains(got[0].Content, "d1") || !contains(got[0].Content, "search tool") {
		t.Errorf("notice should point at d1 and tell the model to fetch: %q", got[0].Content)
	}
}

func TestFinderPreparer_CapsAtMaxHitsBestFirst(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	st.Append(context.Background(), "s", Entry{ID: "u1", Kind: KindUser, Content: "q", CreatedAt: c.next()})
	f := &fakeFinder{hits: []DocHit{
		{DocID: "d1", Score: 0.5}, {DocID: "d2", Score: 0.9}, {DocID: "d3", Score: 0.7},
	}}
	p := FinderPreparer(st, f, FinderOpts{MaxHits: 2, Now: c.next})
	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	got := notices(t, st, "rag")
	if len(got) != 2 {
		t.Fatalf("want 2 (MaxHits), got %d", len(got))
	}
	// Strongest kept: d2 (0.9) + d3 (0.7); d1 (0.5) dropped.
	if !contains(got[0].Content, "d2") || !contains(got[1].Content, "d3") {
		t.Errorf("want best-first d2,d3; got %q / %q", got[0].Content, got[1].Content)
	}
}

func TestFinderPreparer_DedupsAcrossPasses(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	f := &fakeFinder{hits: []DocHit{{DocID: "d1", Score: 0.9}}}
	p := FinderPreparer(st, f, FinderOpts{Now: c.next})

	// Two separate user messages, two passes — same doc surfaces both times.
	st.Append(context.Background(), "s", Entry{ID: "u1", Kind: KindUser, Content: "q1", CreatedAt: c.next()})
	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	st.Append(context.Background(), "s", Entry{ID: "u2", Kind: KindUser, Content: "q2", CreatedAt: c.next()})
	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	if n := len(notices(t, st, "rag")); n != 1 {
		t.Fatalf("dedup: d1 should notify once across passes, got %d", n)
	}
	if f.calls != 2 {
		t.Errorf("finder should be queried on each new message, got %d calls", f.calls)
	}
}

func TestFinderPreparer_WatermarkGatesRequery(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	st.Append(context.Background(), "s", Entry{ID: "u1", Kind: KindUser, Content: "q", CreatedAt: c.next()})
	f := &fakeFinder{}
	p := FinderPreparer(st, f, FinderOpts{Now: c.next})

	// First pass observes u1; second pass has nothing new → no Find call.
	_ = p.PrepareNotifications(context.Background(), "s")
	_ = p.PrepareNotifications(context.Background(), "s")
	if f.calls != 1 {
		t.Fatalf("watermark should gate the re-query; want 1 Find call, got %d", f.calls)
	}
}

func TestFinderPreparer_IgnoresUnobservedKinds(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	st.Append(context.Background(), "s", Entry{ID: "t1", Kind: KindToolResult, Content: "big tool dump", CreatedAt: c.next()})
	f := &fakeFinder{hits: []DocHit{{DocID: "d1", Score: 0.9}}}
	p := FinderPreparer(st, f, FinderOpts{Now: c.next}) // default: user+assistant only
	_ = p.PrepareNotifications(context.Background(), "s")
	if f.calls != 0 {
		t.Fatalf("tool_result is not an observed kind; want 0 Find calls, got %d", f.calls)
	}
}

func TestFinderPreparer_FailOpenRetriesNextPass(t *testing.T) {
	st := &memStore{}
	c := &clk{}
	st.Append(context.Background(), "s", Entry{ID: "u1", Kind: KindUser, Content: "q", CreatedAt: c.next()})
	f := &fakeFinder{err: errors.New("ragtag down")}
	p := FinderPreparer(st, f, FinderOpts{Now: c.next})

	// Pass 1: finder errors → no notice, watermark NOT advanced.
	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatalf("fail-open should not return an error, got %v", err)
	}
	if n := len(notices(t, st, "rag")); n != 0 {
		t.Fatalf("errored pass must inject nothing, got %d", n)
	}
	// Pass 2: finder recovers → the SAME u1 is retried (watermark held) and hits.
	f.err = nil
	f.hits = []DocHit{{DocID: "d1", Score: 0.9}}
	if err := p.PrepareNotifications(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	if n := len(notices(t, st, "rag")); n != 1 {
		t.Fatalf("recovered pass should retry u1 and notify once, got %d", n)
	}
	if f.calls != 2 {
		t.Errorf("want 2 Find calls (retry), got %d", f.calls)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
