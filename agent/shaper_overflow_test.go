package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// foldRunner answers fold calls with "S<n>" and records each request's size.
// fail, when set, is consulted per call and may refuse it.
type foldRunner struct {
	sizes []int
	users []string
	fail  func(call int) error
}

func (r *foldRunner) ChatStream(_ context.Context, msgs []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	call := len(r.sizes)
	n := 0
	for _, m := range msgs {
		n += Default().Estimate(m.Content)
	}
	r.sizes = append(r.sizes, n)
	r.users = append(r.users, msgs[len(msgs)-1].Content)
	if r.fail != nil {
		if err := r.fail(call); err != nil {
			return nil, err
		}
	}
	ch := make(chan llm.StreamChunk, 2)
	ch <- llm.StreamChunk{Content: fmt.Sprintf("S%d", call)}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

// bigHistory is ten ~1000-token events and a short pristine tail.
func bigHistory() *memStore {
	st := &memStore{}
	for i := 0; i < 10; i++ {
		st.entries = append(st.entries, Entry{ID: fmt.Sprintf("e%d", i), Kind: KindUser,
			Content: fmt.Sprintf("event %d %s", i, strings.Repeat("x", 3000)), CreatedAt: int64(i + 1)})
	}
	st.entries = append(st.entries, Entry{ID: "tail", Kind: KindAssistant, Content: "last", CreatedAt: 100})
	return st
}

// A fold whose events exceed the window is read in stages that each fit it,
// and the marker carries the last stage's summary of everything.
func TestFold_StagesAWindowSizedBacklog(t *testing.T) {
	const window = 4000
	st := bigHistory()
	r := &foldRunner{}
	sh := &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{BudgetTokens: window, PreserveLastMessages: 1}}
	info, did, err := sh.Compact(context.Background(), "s")
	if err != nil || !did {
		t.Fatalf("fold: did=%v err=%v", did, err)
	}
	if len(r.sizes) < 2 {
		t.Fatalf("a 10k-token backlog against a %d window must fold in stages, got %d call(s)", window, len(r.sizes))
	}
	for i, n := range r.sizes {
		if n > window {
			t.Errorf("stage %d sent ~%d tokens, over the %d window", i, n, window)
		}
	}
	for i := 1; i < len(r.users); i++ {
		if !strings.Contains(r.users[i], fmt.Sprintf("S%d", i-1)) {
			t.Errorf("stage %d does not carry the previous summary", i)
		}
	}
	if want := fmt.Sprintf("S%d", len(r.sizes)-1); info.Summary != want || info.SubsumedCount != 10 {
		t.Errorf("summary %q subsumed %d, want %q and 10", info.Summary, info.SubsumedCount, want)
	}
}

// Under the window nothing changes: one call with every event.
func TestFold_FitsInOneCall(t *testing.T) {
	st := bigHistory()
	r := &foldRunner{}
	sh := &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1}}
	if _, _, err := sh.Compact(context.Background(), "s"); err != nil {
		t.Fatal(err)
	}
	if len(r.sizes) != 1 || !strings.Contains(r.users[0], "event 0") || !strings.Contains(r.users[0], "event 9") {
		t.Fatalf("want one call with all events, got %d call(s)", len(r.sizes))
	}
}

// One refused stage shrinks by the endpoint's counts and is sent again; a
// second refusal is returned, typed, instead of looping.
func TestFold_RefusedStageShrinksOnceThenFails(t *testing.T) {
	refuse := &llm.InputTooLargeError{Status: 400, PromptTokens: 12000, ContextTokens: 8000}

	st := bigHistory()
	r := &foldRunner{fail: func(call int) error {
		if call == 0 {
			return refuse
		}
		return nil
	}}
	sh := &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{PreserveLastMessages: 1}} // unbudgeted: one call
	if _, _, err := sh.Compact(context.Background(), "s"); err != nil {
		t.Fatalf("one refusal must be recovered: %v", err)
	}
	if len(r.sizes) < 3 || r.sizes[1] >= r.sizes[0] {
		t.Fatalf("after the refusal the stage must be smaller and staged: sizes %v", r.sizes)
	}

	st = bigHistory()
	r = &foldRunner{fail: func(int) error { return refuse }}
	sh = &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{PreserveLastMessages: 1}}
	_, _, err := sh.Compact(context.Background(), "s")
	var tl *llm.InputTooLargeError
	if !errors.As(err, &tl) {
		t.Fatalf("a second refusal must return the typed error, got %v", err)
	}
	if len(r.sizes) != 2 {
		t.Errorf("want exactly one retry, got %d calls", len(r.sizes))
	}
}

// A hint from a refused prompt narrows the budget: an unbudgeted Shaper gets
// the endpoint's window, and a measured undercount shrinks it.
func TestLimits_ShapeHint(t *testing.T) {
	sh := &Shaper{Policy: ShaperPolicy{}}
	if b, _ := sh.limits(context.Background()); b != 0 {
		t.Fatalf("no hint, no budget: got %d", b)
	}
	if b, _ := sh.limits(WithShapeHint(context.Background(), ShapeHint{Window: 100_000})); b != Budget(100_000, DefaultOutputReservePct) {
		t.Errorf("window hint: got %d", b)
	}
	sh.Policy.BudgetTokens = 60_000
	if b, _ := sh.limits(WithShapeHint(context.Background(), ShapeHint{Window: 100_000, Scale: 1.5})); b != 40_000 {
		t.Errorf("a larger window must not raise the budget, and scale 1.5 must cut it to 40000: got %d", b)
	}
}

// mainRunner answers fold calls with a summary and main calls via main(n).
type mainRunner struct {
	main  func(call int) error
	sizes []int // estimated tokens of each main call
}

func (r *mainRunner) ChatStream(_ context.Context, msgs []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	out := "SUMMARY"
	if msgs[0].Content != DefaultSummarySystem {
		n := 0
		for _, m := range msgs {
			n += Default().Estimate(m.Content)
		}
		r.sizes = append(r.sizes, n)
		if err := r.main(len(r.sizes) - 1); err != nil {
			return nil, err
		}
		out = "ok"
	}
	ch := make(chan llm.StreamChunk, 2)
	ch <- llm.StreamChunk{Content: out}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

func refusalSession(r *mainRunner) *Session {
	st := bigHistory()
	sh := &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{PreserveLastMessages: 1}} // unbudgeted
	return &Session{SessionID: "s", Store: st, Runner: r, MaxTurns: 3, Build: sh.Build}
}

// A prompt the endpoint refuses is rebuilt under the window it reported (a
// fold here, since the Shaper had no budget) and sent once more.
func TestTurn_RefusedPromptShrinksAndRetriesOnce(t *testing.T) {
	refuse := &llm.InputTooLargeError{Status: 400, PromptTokens: 20000, ContextTokens: 8000}
	r := &mainRunner{main: func(call int) error {
		if call == 0 {
			return refuse
		}
		return nil
	}}
	s := refusalSession(r)
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("one refusal must be recovered: %v", err)
	}
	if len(r.sizes) != 2 || r.sizes[1] >= r.sizes[0] || len(res.Compactions) == 0 {
		t.Fatalf("want a fold and a smaller second send: sizes %v, %d compaction(s)", r.sizes, len(res.Compactions))
	}
	if s.shapeHint.Window != 8000 || s.shapeHint.Scale <= 1 {
		t.Errorf("hint not learned: %+v", s.shapeHint)
	}
}

// A refusal that survives the rebuild is returned, typed, after one retry.
func TestTurn_RefusedTwiceReturnsTypedError(t *testing.T) {
	refuse := &llm.InputTooLargeError{Status: 400, PromptTokens: 20000, ContextTokens: 8000}
	r := &mainRunner{main: func(int) error { return refuse }}
	s := refusalSession(r)
	var reason TurnEndReason
	s.OnTurnEnd = func(e TurnEnd) { reason = e.Reason }
	_, err := s.Turn(context.Background())
	var tl *llm.InputTooLargeError
	if !errors.As(err, &tl) || reason != TurnOverflow {
		t.Fatalf("want typed overflow, got reason %q err %v", reason, err)
	}
	if len(r.sizes) != 2 {
		t.Errorf("want exactly one retry, got %d main calls", len(r.sizes))
	}
}

// A builder that ignores the hint cannot shrink: fail at once, not resend.
func TestTurn_RefusalWithoutShrinkDoesNotResend(t *testing.T) {
	refuse := &llm.InputTooLargeError{Status: 400}
	r := &mainRunner{main: func(int) error { return refuse }}
	s := &Session{SessionID: "s", Store: bigHistory(), Runner: r, MaxTurns: 3}
	_, err := s.Turn(context.Background())
	var tl *llm.InputTooLargeError
	if !errors.As(err, &tl) {
		t.Fatalf("want typed error, got %v", err)
	}
	if len(r.sizes) != 1 {
		t.Errorf("an unshrunk context must not be resent: %d main calls", len(r.sizes))
	}
}

func TestTurn_RefusalOnLastRoundStillRetries(t *testing.T) {
	refuse := &llm.InputTooLargeError{Status: 400, PromptTokens: 20000, ContextTokens: 8000}
	r := &mainRunner{main: func(call int) error {
		if call == 0 {
			return refuse
		}
		return nil
	}}
	s := refusalSession(r)
	s.MaxTurns = 1
	if _, err := s.Turn(context.Background()); err != nil {
		t.Fatalf("a refused send must not spend the only round: %v", err)
	}
	if len(r.sizes) != 2 {
		t.Fatalf("want the refused send and one resend, got %d sends", len(r.sizes))
	}
}

// The aw4 case at its real scale (2026-11-06 16:59): a fold of ~90k estimated
// tokens of raw history against an 86,016-token endpoint whose real count is
// 1.15× or 1.5× the estimate and which refuses anything over its window, with
// llama.cpp's counts. One event alone is larger than the window. The fold
// must get through, refused at most once.
func TestFold_AW4ScaleBacklogGetsThrough(t *testing.T) {
	for _, pct := range []int{115, 150} {
		t.Run(fmt.Sprintf("undercount%d", pct), func(t *testing.T) { aw4ScaleFold(t, pct) })
	}
}

func aw4ScaleFold(t *testing.T, pct int) {
	const window = 86016
	st := &memStore{}
	for i := 0; i < 36; i++ {
		st.entries = append(st.entries, Entry{ID: fmt.Sprintf("e%d", i), Kind: KindToolResult,
			Content: fmt.Sprintf("result %d %s", i, strings.Repeat("y", 9000)), CreatedAt: int64(i + 1)})
	}
	st.entries = append(st.entries, Entry{ID: "huge", Kind: KindToolResult,
		Content: "huge " + strings.Repeat("z", 400000), CreatedAt: 50})
	st.entries = append(st.entries, Entry{ID: "tail", Kind: KindAssistant, Content: "last", CreatedAt: 100})

	refusals := 0
	r := &foldRunner{}
	r.fail = func(call int) error {
		real := r.sizes[call] * pct / 100
		if real > window {
			refusals++
			return &llm.InputTooLargeError{Status: 400, PromptTokens: real, ContextTokens: window,
				Body: fmt.Sprintf("request (%d tokens) exceeds the available context size (%d tokens)", real, window)}
		}
		return nil
	}
	sh := &Shaper{Store: st, Runner: r, Policy: ShaperPolicy{BudgetTokens: Budget(window, DefaultOutputReservePct), PreserveLastMessages: 1}}
	info, did, err := sh.Compact(context.Background(), "s")
	if err != nil || !did {
		t.Fatalf("fold: did=%v err=%v (refusals %d, sizes %v)", did, err, refusals, r.sizes)
	}
	if refusals > 1 {
		t.Errorf("refused %d times, want at most 1", refusals)
	}
	accepted := 0
	for _, n := range r.sizes {
		if real := n * pct / 100; real <= window {
			accepted++
		}
	}
	if info.SubsumedCount != 37 {
		t.Errorf("subsumed %d, want 37", info.SubsumedCount)
	}
	t.Logf("fold: %d call(s), %d accepted stage(s), %d refusal(s), estimated sizes %v", len(r.sizes), accepted, refusals, r.sizes)
}
