package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// compactingSession builds a session whose budget is small enough that LOD
// cannot fit the window, forcing a fold.
func compactingSession(t *testing.T) *Session {
	t.Helper()
	store := &memStore{}
	big := strings.Repeat("lorem ipsum ", 30) // ~360 chars ≈ 90 tokens each
	for i := 0; i < 4; i++ {
		store.entries = append(store.entries, Entry{
			ID: string(rune('a' + i)), Kind: KindAssistant, Content: big, CreatedAt: int64(i + 1),
		})
	}
	store.entries = append(store.entries, Entry{ID: "q", Kind: KindUser, Content: "recap?", CreatedAt: 10})

	shaper := &Shaper{
		Store:  store,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}},
		Policy: ShaperPolicy{BudgetTokens: 80, PreserveLastMessages: 1,
			LODTruncateAboveChars: 40, LODHeadroomTokens: -1},
	}
	var clock int64 = 100
	return &Session{
		SessionID: "s", Store: store, Build: shaper.Build, MaxTurns: 3,
		Runner: &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "ok"}}}},
		Now:    func() int64 { clock++; return clock },
	}
}

// The pair must be exactly that: a start with no completion is a host left
// showing "compacting…" forever, which is worse than never having said it.
func TestCompactionStartIsPairedWithComplete(t *testing.T) {
	s := compactingSession(t)
	var starts []CompactionStart
	var completes []CompactionInfo
	s.OnCompactionStart = func(c CompactionStart) { starts = append(starts, c) }
	s.OnCompactionComplete = func(c CompactionInfo) { completes = append(completes, c) }

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if len(starts) == 0 {
		t.Fatal("no compaction happened; this test asserts nothing")
	}
	if len(starts) != len(completes) {
		t.Fatalf("%d starts against %d completes — they must pair", len(starts), len(completes))
	}
}

// The start has to carry enough to render "compacting N entries" before the
// summarizer returns. A start with nothing on it is a spinner.
func TestCompactionStartCarriesWhatItIsAbout(t *testing.T) {
	s := compactingSession(t)
	var start CompactionStart
	s.OnCompactionStart = func(c CompactionStart) { start = c }

	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if start.SessionID != "s" {
		t.Errorf("SessionID = %q, want s", start.SessionID)
	}
	if start.Subsuming <= 0 {
		t.Errorf("Subsuming = %d; a fold that subsumes nothing should not have announced",
			start.Subsuming)
	}
	if start.TokensBefore <= 0 || start.Target <= 0 {
		t.Errorf("TokensBefore=%d Target=%d; both come from the measurement that "+
			"triggered the fold and must survive to the hook",
			start.TokensBefore, start.Target)
	}
	if start.TokensBefore <= start.Target {
		t.Errorf("TokensBefore %d is not above Target %d — then why did it fold?",
			start.TokensBefore, start.Target)
	}
	if start.At.IsZero() {
		t.Error("At must be set")
	}
}

// The summary IS the record for an append-only host, and Duration is the part
// of a slow turn that was not the work.
func TestCompactionCompleteCarriesSummaryAndDuration(t *testing.T) {
	s := compactingSession(t)
	var got CompactionInfo
	s.OnCompactionComplete = func(c CompactionInfo) { got = c }

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if got.Summary == "" {
		t.Fatal("Summary empty — it is the whole record of what was folded")
	}
	if got.SubsumedCount <= 0 {
		t.Errorf("SubsumedCount = %d", got.SubsumedCount)
	}
	if got.Duration <= 0 {
		t.Error("Duration must be measured; it is the summarizer's round-trip")
	}
	if len(res.Compactions) == 0 || res.Compactions[0].Summary != got.Summary {
		t.Errorf("TurnResult.Compactions must carry the same fold: %+v", res.Compactions)
	}
}

// Hooks are optional, and a session that sets neither must behave as before.
func TestCompactionHooksAreOptional(t *testing.T) {
	s := compactingSession(t)
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if len(res.Compactions) == 0 {
		t.Error("compaction still has to happen with no hooks attached")
	}
}
