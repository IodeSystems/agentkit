package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

type fakeCompactor struct {
	calls int
	ok    bool
	err   error
}

func (f *fakeCompactor) Compact(context.Context, string) (CompactionInfo, bool, error) {
	f.calls++
	return CompactionInfo{}, f.ok, f.err
}

func newRecoverySession(c *fakeCompactor) (*Session, *memStore) {
	st := &memStore{}
	return &Session{SessionID: "s", Store: st, Compactor: c}, st
}

// The note must SAY the write was cut off and TELL the model to go smaller.
// Compacting silently is the failure mode: the model reissues the same
// oversized write into the freed window, and the next big file fails the same way.
func TestRecoveryCompactsAndExplains(t *testing.T) {
	c := &fakeCompactor{ok: true}
	s, st := newRecoverySession(c)
	err := s.recoverFromTruncation(context.Background(), &llm.TruncatedToolCallError{
		Status: 500,
		Body:   `... last read: '\"package com.termux.app;\\n\\nclass Foo {\\n'`,
	})
	if err != nil {
		t.Fatalf("recovery failed: %v", err)
	}
	if c.calls != 1 {
		t.Fatalf("compactor called %d times, want 1", c.calls)
	}
	if len(st.entries) != 1 {
		t.Fatalf("appended %d entries, want 1", len(st.entries))
	}
	e := st.entries[0]
	if e.Kind != KindNotification {
		t.Errorf("kind = %q; a notification is tail-kept and budget-neutral, so it "+
			"survives the compaction it follows", e.Kind)
	}
	for _, want := range []string{"CUT OFF", "compacted", "SMALLER"} {
		if !strings.Contains(e.Content, want) {
			t.Errorf("note missing %q:\n%s", want, e.Content)
		}
	}
	// It should hand back the partial so the model resumes instead of regenerating.
	if !strings.Contains(e.Content, "package com.termux.app") {
		t.Errorf("note dropped the partial write preview:\n%s", e.Content)
	}
}

// Nothing left to fold means a retry hits the identical wall — say so rather
// than loop.
func TestRecoveryRefusesWhenNothingToCompact(t *testing.T) {
	c := &fakeCompactor{ok: false}
	s, st := newRecoverySession(c)
	err := s.recoverFromTruncation(context.Background(), &llm.TruncatedToolCallError{Status: 500})
	if err == nil {
		t.Fatal("expected an error when the context cannot be reduced")
	}
	if !strings.Contains(err.Error(), "too large") {
		t.Errorf("unhelpful error: %v", err)
	}
	if len(st.entries) != 0 {
		t.Error("should not append a retry hint when retrying cannot help")
	}
}

// No compactor configured: report, do not pretend to recover.
func TestRecoveryWithoutCompactorStillExplains(t *testing.T) {
	s, st := newRecoverySession(nil)
	s.Compactor = nil
	if err := s.recoverFromTruncation(context.Background(), &llm.TruncatedToolCallError{Status: 500}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(st.entries) != 1 {
		t.Fatalf("appended %d, want 1", len(st.entries))
	}
}
