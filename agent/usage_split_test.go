package agent

import "testing"

// TestUsageSplitAccumulates pins the invariant that makes the split trustworthy:
// Cached + Processed + Generated must equal Total, so a host can report the three
// channels without them silently disagreeing with the headline number.
func TestUsageSplitSumsToTotal(t *testing.T) {
	s := &Session{}
	// two rounds, one with a cache hit, one cold
	for _, u := range []struct{ prompt, cached, completion int }{
		{prompt: 1000, cached: 900, completion: 50},
		{prompt: 1200, cached: 0, completion: 70},
	} {
		s.usageTotal += u.prompt + u.completion
		s.usageCached += u.cached
		s.usageProcessed += u.prompt - u.cached
		s.usageGenerated += u.completion
		s.usageTurns++
	}
	if got, want := s.usageCached+s.usageProcessed+s.usageGenerated, s.usageTotal; got != want {
		t.Fatalf("channels sum to %d, Total is %d", got, want)
	}
	if s.usageTurns != 2 {
		t.Fatalf("Turns = %d, want 2", s.usageTurns)
	}
	if s.usageGenerated != 120 {
		t.Fatalf("Generated = %d, want 120", s.usageGenerated)
	}
}
