package agent

import (
	"context"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// OnIdle runs where the Turn would end: a non-empty note is appended and the
// Turn takes another round; "" lets it end. The host bounds itself by asks.
func TestOnIdleAsksOnceThenEnds(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Content: "Done."}},
		{{Content: "Checked: all requirements met. Done."}},
	}}
	var calls []int
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5,
		OnIdle: func(_ context.Context, rounds, asks int) string {
			calls = append(calls, asks)
			if asks > 0 {
				return ""
			}
			return "check each requirement"
		}}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "Checked: all requirements met. Done." {
		t.Errorf("Reply = %q, want the round after the check", res.Reply)
	}
	if len(calls) != 2 || calls[0] != 0 || calls[1] != 1 {
		t.Errorf("OnIdle asks = %v, want [0 1]", calls)
	}
	n := 0
	for _, e := range store.entries {
		if e.Kind == KindNotification && e.Tag == "idle_check" {
			n++
		}
	}
	if n != 1 {
		t.Errorf("%d idle_check notes, want 1", n)
	}
}
