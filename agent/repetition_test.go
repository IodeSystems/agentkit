package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

const repBlock = "EXCLUSIVE OF THE EASTERLY 100 FEET OF SAID LOTS 7 TO 10, "

// loopedTurn is what the transport hands back when it cuts a looping
// generation: the content that arrived before the cut, then a Done chunk
// carrying the finding.
func loopedTurn(prefix string, reps int) []llm.StreamChunk {
	chunks := []llm.StreamChunk{{Content: prefix}}
	for i := 0; i < reps; i++ {
		chunks = append(chunks, llm.StreamChunk{Content: repBlock})
	}
	return append(chunks, llm.StreamChunk{
		Done:       true,
		StopReason: llm.StopReasonRepetition,
		Repetition: &llm.RepetitionInfo{
			Where: "content", Period: len(repBlock), Reps: reps,
			Trailing: (reps - 1) * len(repBlock), Sample: repBlock,
		},
	})
}

// A cut round must not end the Turn: the model gets told what it did and
// re-prompted, because the text before the loop is usually a real answer in
// progress.
func TestRepetitionIsRetriedNotFatal(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		loopedTurn("The parcel is described as: ", 6),
		{{Content: "The parcel covers lots 7 to 10."}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "The parcel covers lots 7 to 10." {
		t.Errorf("Reply = %q, want the recovered second round", res.Reply)
	}

	// The loop must NOT be persisted in full: re-prompting from a context that
	// contains it primes the model to continue it.
	var assistant string
	var notice string
	for _, e := range store.entries {
		switch {
		case e.Kind == KindAssistant && assistant == "":
			assistant = e.Content
		case e.Kind == KindNotification && e.Tag == "repetition":
			notice = e.Content
		}
	}
	if want := "The parcel is described as: " + repBlock; assistant != want {
		t.Errorf("persisted assistant reply = %q,\nwant one copy of the block kept: %q", assistant, want)
	}
	if notice == "" {
		t.Fatal("no repetition notice was injected; the model is never told why it was cut")
	}
	if !strings.Contains(notice, "CUT OFF") {
		t.Errorf("notice does not say the response was cut: %q", notice)
	}

	// The notice has to actually REACH the model on the retry.
	if len(runner.seen) < 2 {
		t.Fatalf("only %d chat rounds; the cut round was not retried", len(runner.seen))
	}
	var sawNotice bool
	for _, m := range runner.seen[1] {
		if strings.Contains(m.Content, "[repetition]") {
			sawNotice = true
		}
	}
	if !sawNotice {
		t.Error("the retry round did not carry the repetition notice")
	}
}

// Retrying forever is the same 30 minutes paid in installments.
func TestRepeatedCollapseGivesUp(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		loopedTurn("a", 6), loopedTurn("b", 6), loopedTurn("c", 6), loopedTurn("d", 6),
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 20}

	_, err := s.Turn(context.Background())
	if err == nil {
		t.Fatal("a model that collapses every round never stopped the Turn")
	}
	if !strings.Contains(err.Error(), "looped") {
		t.Errorf("error does not name the cause: %v", err)
	}
	// Default MaxRepetitionRetries=2 → two retries, then the third trip aborts.
	if len(runner.seen) != 3 {
		t.Errorf("made %d chat rounds, want 3 (the cut round plus 2 retries)", len(runner.seen))
	}
}

// A loop in TOOL ARGUMENTS arrives with no content and no usable call. The Turn
// must still recover rather than idling out as if the model had nothing to say.
func TestRepetitionInToolArgumentsRetries(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		{{Done: true, StopReason: llm.StopReasonRepetition, Repetition: &llm.RepetitionInfo{
			Where: "tool:record_legal arguments", Period: len(repBlock), Reps: 6,
			Trailing: 5 * len(repBlock), Sample: repBlock,
		}}},
		{{Content: "done"}},
	}}
	s := &Session{SessionID: "s", Store: store, Runner: runner, MaxTurns: 5}

	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.Reply != "done" {
		t.Errorf("Reply = %q, want the recovered round", res.Reply)
	}
	var notice string
	for _, e := range store.entries {
		if e.Kind == KindNotification && e.Tag == "repetition" {
			notice = e.Content
		}
	}
	if !strings.Contains(notice, "record_legal") {
		t.Errorf("notice does not name the offending tool: %q", notice)
	}
}
