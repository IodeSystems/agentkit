package agent

import (
	"context"
	"testing"
)

func TestIsResolvedTruth(t *testing.T) {
	resolved := []string{"", "   ", "null", "{}", "[]", `""`, "\n\t"}
	for _, s := range resolved {
		if !IsResolvedTruth(s) {
			t.Errorf("IsResolvedTruth(%q) = false; want true", s)
		}
	}
	live := []string{`{"errors":1}`, `[{"line":4}]`, "still failing", "0"}
	for _, s := range live {
		if IsResolvedTruth(s) {
			t.Errorf("IsResolvedTruth(%q) = true; want false", s)
		}
	}
}

type fakeRevalStore struct {
	notices []PendingNotice
	cleared []string // "groupBy/key"
}

func (f *fakeRevalStore) PendingNotices(_ context.Context, _ string) ([]PendingNotice, error) {
	return f.notices, nil
}
func (f *fakeRevalStore) Clear(_ context.Context, _, groupBy, key string) error {
	f.cleared = append(f.cleared, groupBy+"/"+key)
	return nil
}

type fakeRevalidator struct{ truth map[string]string } // key → current truth ("" = resolved)

func (f *fakeRevalidator) Revalidate(_ context.Context, groupBy, key string) (string, bool, error) {
	v, ok := f.truth[key]
	return v, ok, nil
}

func TestMCPPreparer_ClearsResolvedKeepsLive(t *testing.T) {
	store := &fakeRevalStore{notices: []PendingNotice{
		{GroupBy: "file", Key: "resolved.go"},
		{GroupBy: "file", Key: "still-bad.go"},
		{GroupBy: "file", Key: "no-revalidator.go"},
	}}
	rv := &fakeRevalidator{truth: map[string]string{
		"resolved.go":  "",              // resolved → clear
		"still-bad.go": `[{"line":10}]`, // live → keep
		// no-revalidator.go absent → ok=false → keep
	}}

	if err := MCPPreparer(store, rv).PrepareNotifications(context.Background(), "s1"); err != nil {
		t.Fatal(err)
	}
	if len(store.cleared) != 1 || store.cleared[0] != "file/resolved.go" {
		t.Fatalf("cleared = %v; want [file/resolved.go]", store.cleared)
	}
}
