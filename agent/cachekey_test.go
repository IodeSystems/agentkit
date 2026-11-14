package agent

import (
	"context"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

func cacheKeyTurn(t *testing.T, s *Session, store *memStore, say string) {
	t.Helper()
	store.entries = append(store.entries, Entry{ID: NewID(), Kind: KindUser, Content: say, CreatedAt: int64(len(store.entries) + 1)})
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("Turn: %v", err)
	}
}

func reply() []llm.StreamChunk { return []llm.StreamChunk{{Content: "ok"}} }

// A key that moved between turns of one conversation would send every turn to
// a cold slot, which is the cost the key exists to avoid.
func TestPromptCacheKeyIsStableAcrossTurnsWithoutCompaction(t *testing.T) {
	store := &memStore{}
	r := &scriptRunner{turns: [][]llm.StreamChunk{reply(), reply()}}
	s := &Session{SessionID: "s1", Store: store, Runner: r, MaxTurns: 3}
	cacheKeyTurn(t, s, store, "one")
	cacheKeyTurn(t, s, store, "two")
	a, b := r.opts[0].PromptCacheKey, r.opts[1].PromptCacheKey
	if a == "" || len(a) != 32 || a != b {
		t.Fatalf("keys %q, %q: want the same 32-char key", a, b)
	}
}

// After a fold the old prefix is gone; a key still naming it routes to a slot
// holding nothing. A host that rebuilds its Session per unit of work (aw4)
// must get the key of the store's newest marker on the first round.
func TestPromptCacheKeyChangesAfterCompaction(t *testing.T) {
	store := &memStore{}
	r := &scriptRunner{turns: [][]llm.StreamChunk{reply(), reply(), reply()}}
	session := func() *Session { return &Session{SessionID: "s1", Store: store, Runner: r, MaxTurns: 3} }
	cacheKeyTurn(t, session(), store, "one")
	if err := store.Compact(context.Background(), "s1", Compaction{
		Marker: Entry{ID: "m1", Kind: KindCompaction, Content: "sum", CreatedAt: 100},
	}); err != nil {
		t.Fatal(err)
	}
	cacheKeyTurn(t, session(), store, "two")
	if err := store.Compact(context.Background(), "s1", Compaction{
		Marker: Entry{ID: "m2", Kind: KindCompaction, Content: "sum2", CreatedAt: 200},
	}); err != nil {
		t.Fatal(err)
	}
	cacheKeyTurn(t, session(), store, "three")
	k0, k1, k2 := r.opts[0].PromptCacheKey, r.opts[1].PromptCacheKey, r.opts[2].PromptCacheKey
	if k0 == k1 || k1 == k2 || k0 == k2 {
		t.Fatalf("keys %q %q %q: each compaction must start a new key", k0, k1, k2)
	}
	if k1 != promptCacheKey("s1", "m1") {
		t.Fatalf("key %q is not derived from session id and marker id", k1)
	}
}

// A fold inside a long-lived Session's Turn reaches the key through the
// compaction sink, not a store read: the round after the fold names the new
// marker, though the Session cached the old one on its first round.
func TestPromptCacheKeyFollowsFoldInsideTurn(t *testing.T) {
	store := &memStore{}
	r := &scriptRunner{turns: [][]llm.StreamChunk{reply(), reply()}}
	fold := false
	s := &Session{SessionID: "s1", Store: store, Runner: r, MaxTurns: 3}
	s.Build = func(ctx context.Context, sid, system string) ([]llm.Message, error) {
		if fold {
			marker := Entry{ID: "m9", Kind: KindCompaction, Content: "sum", CreatedAt: 100}
			if err := store.Compact(ctx, sid, Compaction{Marker: marker}); err != nil {
				return nil, err
			}
			ctx.Value(compactionSinkKey{}).(func(CompactionInfo))(CompactionInfo{MarkerID: marker.ID})
		}
		return DefaultContextBuilder(ctx, store, sid, system)
	}
	cacheKeyTurn(t, s, store, "one")
	fold = true
	cacheKeyTurn(t, s, store, "two")
	if got, want := r.opts[1].PromptCacheKey, promptCacheKey("s1", "m9"); got != want {
		t.Fatalf("key after fold = %q, want %q", got, want)
	}
	if r.opts[0].PromptCacheKey == r.opts[1].PromptCacheKey {
		t.Fatal("key did not change after the fold")
	}
}

// The host knows its own conversation identity (a fork, a shared prefix); the
// derived key must not overwrite it, and must not mutate the shared ChatOpts.
func TestPromptCacheKeyHostValueWins(t *testing.T) {
	store := &memStore{}
	r := &scriptRunner{turns: [][]llm.StreamChunk{reply()}}
	shared := &llm.ChatOpts{PromptCacheKey: "host-key"}
	s := &Session{SessionID: "s1", Store: store, Runner: r, ChatOpts: shared, MaxTurns: 3}
	cacheKeyTurn(t, s, store, "one")
	if got := r.opts[0].PromptCacheKey; got != "host-key" {
		t.Fatalf("key = %q, want host-key", got)
	}
}

// Without a session id there is no stable identity to name, so send nothing
// rather than a hash of the empty string shared by every anonymous session.
func TestPromptCacheKeyAbsentWithoutSessionID(t *testing.T) {
	store := &memStore{}
	r := &scriptRunner{turns: [][]llm.StreamChunk{reply()}}
	s := &Session{Store: store, Runner: r, MaxTurns: 3}
	cacheKeyTurn(t, s, store, "one")
	if got := r.opts[0].PromptCacheKey; got != "" {
		t.Fatalf("key = %q, want none", got)
	}
}

// The fold's summarizer call shares no prefix with the conversation; a key on
// it would pin it to the conversation's slot for no cache gain.
func TestSummarizerCallCarriesNoPromptCacheKey(t *testing.T) {
	store := &memStore{entries: []Entry{
		{ID: "u", Kind: KindUser, Content: "go", CreatedAt: 1},
		{ID: "a", Kind: KindAssistant, Content: "done", CreatedAt: 2},
		{ID: "u2", Kind: KindUser, Content: "more", CreatedAt: 3},
	}}
	r := &scriptRunner{turns: [][]llm.StreamChunk{{{Content: "SUMMARY"}}}}
	sh := &Shaper{Store: store, Runner: r,
		Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1, PreserveLastToolCalls: 1}}
	if _, _, err := sh.Compact(context.Background(), "s1"); err != nil {
		t.Fatal(err)
	}
	if len(r.opts) != 1 || r.opts[0].PromptCacheKey != "" {
		t.Fatalf("summarizer opts = %+v", r.opts)
	}
}
