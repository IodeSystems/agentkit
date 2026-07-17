package ragnotify

import "testing"

func TestParseHits_Shapes(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []struct {
			id    string
			score float64
			line  string
		}
	}{
		{
			name: "wrapped under hits, doc_id + snippet",
			raw:  `{"hits":[{"doc_id":"d1","title":"Auth","score":0.9,"snippet":"how auth works"}]}`,
			want: []struct {
				id    string
				score float64
				line  string
			}{{"d1", 0.9, "how auth works"}},
		},
		{
			name: "bare array, id + text fallbacks",
			raw:  `[{"id":"d2","score":0.5,"text":"body text"}]`,
			want: []struct {
				id    string
				score float64
				line  string
			}{{"d2", 0.5, "body text"}},
		},
		{
			name: "results wrapper",
			raw:  `{"results":[{"doc_id":"d3","score":0.7,"content":"c"}]}`,
			want: []struct {
				id    string
				score float64
				line  string
			}{{"d3", 0.7, "c"}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseHits(tc.raw)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("want %d hits, got %d", len(tc.want), len(got))
			}
			for i, w := range tc.want {
				if got[i].DocID != w.id || got[i].Score != w.score || got[i].Line != w.line {
					t.Errorf("hit %d = %+v, want id=%s score=%v line=%q", i, got[i], w.id, w.score, w.line)
				}
			}
		})
	}
}

func TestParseHits_DropsIdlessAndEmpty(t *testing.T) {
	got, err := ParseHits(`{"hits":[{"title":"no id","score":1.0},{"doc_id":"ok","score":0.5}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].DocID != "ok" {
		t.Fatalf("id-less hit should be dropped; got %+v", got)
	}
	if h, err := ParseHits("   "); err != nil || h != nil {
		t.Fatalf("empty result → (nil,nil); got %+v / %v", h, err)
	}
}

func TestClip(t *testing.T) {
	if clip("short", 240) != "short" {
		t.Error("no clip under limit")
	}
	if got := clip("aaaa", 2); got != "aa…" {
		t.Errorf("clip = %q", got)
	}
}
