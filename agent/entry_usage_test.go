package agent

import (
	"encoding/json"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func TestEntryUsageProjectsTheSplit(t *testing.T) {
	u := &llm.Usage{
		PromptTokens:        1000,
		CompletionTokens:    42,
		LatencyMS:           2818,
		PromptTokensDetails: &llm.PromptTokensDetails{CachedTokens: 900},
	}
	eu := entryUsage(u)
	if eu == nil {
		t.Fatal("nil usage record for a reported call")
	}
	// The split, not a total: cached is ~free, processed is real input work,
	// generated is the priciest. A single number hides all three.
	if eu.Cached != 900 || eu.Processed != 100 || eu.Generated != 42 {
		t.Errorf("split wrong: %+v", eu)
	}
	if eu.LatencyMS != 2818 {
		t.Errorf("latency lost: %+v", eu)
	}
}

// A provider that reports nothing must leave the field ABSENT — a row of zeros
// would read as "this call was free", which is worse than no data.
func TestEntryUsageNilWhenNothingReported(t *testing.T) {
	if entryUsage(nil) != nil {
		t.Error("nil usage should project to nil")
	}
	if got := entryUsage(&llm.Usage{}); got != nil {
		t.Errorf("empty usage should project to nil, got %+v", got)
	}
}

// The record must survive the round-trip a JSONL store performs, and must be
// omitted entirely on entries that have none.
func TestEntryUsageJSONRoundTrip(t *testing.T) {
	e := Entry{ID: "1", Kind: KindAssistant, Content: "hi",
		Usage: &EntryUsage{LatencyMS: 500, Cached: 10, Processed: 20, Generated: 30}}
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var back Entry
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Usage == nil || *back.Usage != *e.Usage {
		t.Fatalf("usage did not round-trip: %+v", back.Usage)
	}

	plain, _ := json.Marshal(Entry{ID: "2", Kind: KindToolResult, Content: "x"})
	if containsSub(string(plain), "Usage") {
		t.Errorf("usage should be omitted when absent: %s", plain)
	}
}

func containsSub(h, n string) bool {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return true
		}
	}
	return false
}
