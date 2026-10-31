package loopdetector

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// feedAll streams text in small deltas and returns the first finding and how
// many bytes it took.
func feedAll(s *Session, channel, text string) (*llm.RepetitionInfo, int) {
	const step = 48
	for i := 0; i < len(text); i += step {
		j := min(i+step, len(text))
		if info := s.Watch(channel, text[i:j]); info != nil {
			return info, j
		}
	}
	return nil, len(text)
}

// variations is the shape measured in production: a reply of 291 sentences of
// which 40 were distinct, each said about 21 times, with no two copies adjacent.
// The exact detector found nothing in it.
func variations(n int) []string {
	topics := []string{"the login page", "the billing export", "the session alarm", "the schema guard",
		"the worktree flow", "the audit report", "the KPI endpoint", "the retry queue",
		"the upload limit", "the cache header", "the deploy script", "the theme switch"}
	out := make([]string, n)
	for i := range out {
		t := topics[i%len(topics)]
		out[i] = fmt.Sprintf("Looking again at %s item %d, it blocks nothing urgent so it stays at priority two for now. ", t, i)
	}
	return out
}

func TestCyclesOfManyVariationsAreCounted(t *testing.T) {
	var b strings.Builder
	for range 21 {
		for _, v := range variations(40) {
			b.WriteString(v)
		}
	}
	s := New(Config{}).Begin(nil)
	info, fed := feedAll(s, llm.ChannelContent, b.String())
	if info == nil {
		t.Fatal("a 40-sentence cycle said 21 times was not found")
	}
	if info.Kind != KindCycle || info.Reps != 3 {
		t.Errorf("got kind %q, %d loops, want cycle at the default tolerance of 3", info.Kind, info.Reps)
	}
	if want := 4 * len(strings.Join(variations(40), "")); fed > want {
		t.Errorf("found after %d bytes, want within four rounds (%d)", fed, want)
	}
}

func TestAShuffledCycleHasNoPeriodForTheGuardToFind(t *testing.T) {
	// Reordered each round: no period to find.
	rng := rand.New(rand.NewSource(1))
	var shuffled strings.Builder
	for range 21 {
		v := variations(40)
		rng.Shuffle(len(v), func(i, j int) { v[i], v[j] = v[j], v[i] })
		shuffled.WriteString(strings.Join(v, ""))
	}
	w := llm.NewStreamWatcher(llm.RepetitionGuard{})
	if info := w.Watch(llm.ChannelContent, shuffled.String()); info != nil {
		t.Fatalf("the exact guard found %v, so this test shows nothing", info)
	}
	s := New(Config{}).Begin(nil)
	if info, _ := feedAll(s, llm.ChannelContent, shuffled.String()); info == nil {
		t.Fatal("a shuffled cycle was not found")
	}
}

func TestRandomDataBetweenCopiesDoesNotHideALoop(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	noise := func() string {
		w := make([]string, 8)
		for i := range w {
			w[i] = fmt.Sprintf("%x", rng.Uint64())
		}
		return strings.Join(w, " ") + ".\n"
	}
	var b strings.Builder
	for range 30 {
		for _, v := range variations(4) {
			b.WriteString(v)
		}
		b.WriteString(noise())
	}
	s := New(Config{}).Begin(nil)
	if info, _ := feedAll(s, llm.ChannelContent, b.String()); info == nil {
		t.Fatal("four sentences repeated around random data were not found")
	}
}

func TestRewordedCopiesAreCountedAsPhrases(t *testing.T) {
	sentences := []string{
		"I should check whether the migration ran against the replica before touching the primary database. ",
		"The replica lag alarm fired twice this morning and nobody acknowledged the second page yet. ",
		"Before the release goes out someone needs to confirm the rollback script still works end to end. ",
		"The staging certificate expires next week so the renewal job has to be looked at soon. ",
		"Customer exports are slow again because the nightly compaction is overlapping with the batch window. ",
	}
	var b strings.Builder
	openers := []string{"So, ", "Well, ", "Hmm, ", "Right, "}
	for i := range 80 {
		b.WriteString(openers[i%len(openers)])
		b.WriteString(sentences[i%len(sentences)])
	}
	s := New(Config{}).Begin(nil)
	info, _ := feedAll(s, llm.ChannelContent, b.String())
	if info == nil {
		t.Fatal("reworded copies were not found")
	}
	if info.Kind != KindPhrase {
		t.Errorf("kind = %q, want phrase", info.Kind)
	}
}

func TestStemmingMakesOneWordOfItsForms(t *testing.T) {
	for _, group := range [][]string{
		{"check", "checks", "checked", "checking"},
		{"stop", "stops", "stopped", "stopping"},
		{"retry", "retries", "retried"},
		{"migrate", "migrates", "migrated", "migrating"},
	} {
		// retry/retries meet at the stem of the -ies form.
		base := stem(group[len(group)-1])
		for _, w := range group[:len(group)-1] {
			if w == "retry" {
				continue
			}
			if got := stem(w); got != base {
				t.Errorf("stem(%q) = %q, want %q", w, got, base)
			}
		}
	}
	for _, short := range []string{"is", "as", "this", "bus"} {
		if got := stem(short); got == "" {
			t.Errorf("stem(%q) is empty", short)
		}
	}
}

func TestReworkedWordFormsStillMatch(t *testing.T) {
	x := newIndex()
	text := "the worker checked the migrated database replicas before stopping the primary service"
	x.match(tokenize(text), text, 0.8)
	x.match(tokenize("nightly export jobs overlap the batch window every single night"), "other", 0.8)
	again := "workers checking the migrating database replica before they stop the primary services"
	if _, kind := x.match(tokenize(again), again, 0.7); kind != KindPhrase {
		t.Errorf("kind = %q, want phrase: word forms did not meet", kind)
	}
}

func TestAUserAskingAgainIsNotALoop(t *testing.T) {
	report := strings.Join(variations(30), "")
	var history []llm.Message
	for range 4 {
		history = append(history, llm.Message{Role: "user", Content: "status?"},
			llm.Message{Role: "assistant", Content: report})
	}
	history = append(history, llm.Message{Role: "user", Content: "status?"})
	s := New(Config{}).Begin(history)
	if info, _ := feedAll(s, llm.ChannelContent, report); info != nil {
		t.Fatalf("the same answer to the same question was cut: %v", info)
	}
}

// dun's trace: the same failed call 11 times, a loop warning after the fifth. The
// warning is a user message on the wire and must not restart what it warns about.
func TestAHostNoticeDoesNotEndTheTurn(t *testing.T) {
	calls := make([]string, 7)
	for i := range calls {
		calls[i] = fmt.Sprintf("tool%d", i)
	}
	same := func(_, c int) string { return fmt.Sprint("same ", c) }
	history := []llm.Message{{Role: "user", Content: "go"}}
	history = append(history, toolHistory(2, calls, same)...)
	history = append(history, llm.Message{Role: "user", Content: "[tool_loop] You have called tool0 5 times"})
	history = append(history, toolHistory(3, calls, same)...)
	history = append(history, llm.Message{Role: "user", Content: "[compacted]\nearlier"})
	s := New(Config{}).Begin(history)
	if info := s.WatchToolCall("tool0", `{"call":0,"path":"a"}`); info == nil {
		t.Fatal("a notice restarted the count")
	}
	// A person ends it, even one writing a bracket mid-sentence.
	s = New(Config{}).Begin(append(history, llm.Message{Role: "user", Content: "stop [now] and rethink"}))
	if info := s.WatchToolCall("tool0", `{"call":0,"path":"a"}`); info != nil {
		t.Fatalf("a person's message did not end the turn: %v", info)
	}
}

func TestAReplyDraftedInThinkingIsNotALoop(t *testing.T) {
	draft := strings.Join(variations(30), "")
	s := New(Config{}).Begin(nil)
	feedAll(s, llm.ChannelReasoning, draft)
	feedAll(s, llm.ChannelReasoning, draft)
	if info, _ := feedAll(s, llm.ChannelContent, draft); info != nil {
		t.Fatalf("a reply that says what was thought was cut: %v", info)
	}
}

// Ten times the text must cost about ten times as much, not a hundred. Begin
// runs on every request.
func TestBeginScalesWithTheTurn(t *testing.T) {
	took := func(n int) time.Duration {
		var b strings.Builder
		for i := range n {
			fmt.Fprintf(&b, "The package number %d builds cleanly and every test in it passes on this branch today.\n", i)
		}
		history := []llm.Message{{Role: "assistant", Content: b.String()}}
		start := time.Now()
		New(Config{}).Begin(history)
		return time.Since(start)
	}
	small, large := took(5000), took(50000)
	if large > 30*small {
		t.Errorf("5,000 sentences took %v, 50,000 took %v", small, large)
	}
}

func TestToolToleranceCountsToolsWhileTextIsCut(t *testing.T) {
	calls := make([]string, 7)
	for i := range calls {
		calls[i] = fmt.Sprintf("tool%d", i)
	}
	d := New(Config{ToolTolerance: -1})
	s := d.Begin(toolHistory(4, calls, func(_, c int) string { return fmt.Sprint("same ", c) }))
	if info := s.WatchToolCall("tool0", `{"call":0,"path":"a"}`); info != nil {
		t.Fatalf("a count-only tool detector cut: %v", info)
	}
	if loops, _ := s.Loops(); loops < 3 {
		t.Errorf("Loops() = %d", loops)
	}
	text := strings.Repeat(strings.Join(variations(40), ""), 9)
	if info, _ := feedAll(s, llm.ChannelContent, text); info == nil {
		t.Error("text was not cut")
	}
}

func TestAFindingShowsTheFirstAndLastRepeat(t *testing.T) {
	calls := make([]string, 7)
	for i := range calls {
		calls[i] = fmt.Sprintf("tool%d", i)
	}
	s := New(Config{SampleBytes: 12}).Begin(toolHistory(4, calls, func(_, c int) string { return fmt.Sprint("same ", c) }))
	info := s.WatchToolCall("tool3", `{"call":3,"path":"a"}`)
	if info == nil {
		t.Fatal("not cut")
	}
	if info.First == "" || info.Last != `tool3 {"call` || len(info.First) > 12 {
		t.Errorf("First = %q, Last = %q", info.First, info.Last)
	}

	text := strings.Repeat(strings.Join(variations(40), ""), 9)
	found, _ := feedAll(New(Config{}).Begin(nil), llm.ChannelContent, text)
	if found == nil || !strings.HasPrefix(found.First, "Looking again") || !strings.HasPrefix(found.Last, "Looking again") {
		t.Errorf("finding = %+v", found)
	}
}

// Same call, same failure text, honest work: the tree changed between the runs.
// An edit is a different call each time, so it holds the share of repeats down.
func TestEditingAndRerunningTheSameFailureIsNotALoop(t *testing.T) {
	var history []llm.Message
	add := func(id, name, args, result string) {
		tc := llm.ToolCall{ID: id}
		tc.Function.Name, tc.Function.Arguments = name, args
		history = append(history, llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{tc}},
			llm.Message{Role: "tool", ToolCallID: id, Content: result})
	}
	for i := range 20 {
		add(fmt.Sprint("e", i), "edit", fmt.Sprintf(`{"attempt":%d}`, i), "ok")
		add(fmt.Sprint("t", i), "test.run", `{"pkg":"a"}`, "FAIL: TestX expected 2 got 1")
	}
	s := New(Config{}).Begin(history)
	if loops, _ := s.Loops(); loops != 0 {
		t.Errorf("counted %d loops", loops)
	}
	if info := s.WatchToolCall("test.run", `{"pkg":"a"}`); info != nil {
		t.Errorf("cut: %v", info)
	}
}

func TestToleranceSetsWhereTheStreamIsCut(t *testing.T) {
	text := strings.Repeat(strings.Join(variations(40), ""), 21)
	_, at3 := feedAll(New(Config{Tolerance: 3}).Begin(nil), llm.ChannelContent, text)
	info, at6 := feedAll(New(Config{Tolerance: 6}).Begin(nil), llm.ChannelContent, text)
	if info == nil || info.Reps != 6 {
		t.Fatalf("tolerance 6 gave %+v", info)
	}
	if at6 <= at3 {
		t.Errorf("tolerance 6 cut at %d, tolerance 3 at %d", at6, at3)
	}
}

func TestNegativeToleranceCountsAndNeverInterrupts(t *testing.T) {
	d := New(Config{Tolerance: -1})
	s := d.Begin(nil)
	text := strings.Repeat(strings.Join(variations(40), ""), 21)
	if info, _ := feedAll(s, llm.ChannelContent, text); info != nil {
		t.Fatalf("interrupted with a negative tolerance: %v", info)
	}
	loops, kind := s.Loops()
	if loops < 20 || kind != KindCycle {
		t.Errorf("Loops() = %d, %q, want about 21 cycle", loops, kind)
	}
	if f := s.Finding(); f == nil || f.Reps != loops || f.First == "" {
		t.Errorf("Finding() = %+v", f)
	}
	c := d.Counters()
	if c.Detected[KindCycle] != 1 || c.Interrupted[KindCycle] != 0 {
		t.Errorf("counters: %+v", c)
	}
}

func TestLoopInThinkingIsCutWithNothingToTrim(t *testing.T) {
	s := New(Config{}).Begin(nil)
	info, _ := feedAll(s, llm.ChannelReasoning, strings.Repeat(strings.Join(variations(40), ""), 21))
	if info == nil {
		t.Fatal("a loop in thinking was not found")
	}
	if info.Where != llm.ChannelReasoning {
		t.Errorf("where = %q", info.Where)
	}
}

func TestExactCountGoesOnPastItsWindow(t *testing.T) {
	const block = "EXCLUSIVE OF THE EASTERLY 100 FEET\\nOF SAID LOTS 7 TO 10, EXCLUSIVE OF THE EASTERLY "
	s := New(Config{Tolerance: -1, Similarity: -1}).Begin(nil)
	feedAll(s, llm.ChannelContent, strings.Repeat(block, 500))
	loops, kind := s.Loops()
	if kind != KindExact && kind != KindCycle {
		t.Errorf("kind = %q", kind)
	}
	if loops < 100 {
		t.Errorf("Loops() = %d after 500 copies", loops)
	}
}

func TestTheCapturedOCRRunawayStillCuts(t *testing.T) {
	b, err := os.ReadFile("../llm/testdata/looping_ocr.txt")
	if err != nil {
		t.Fatal(err)
	}
	s := New(Config{}).Begin(nil)
	info, fed := feedAll(s, llm.ChannelContent, string(b))
	if info == nil {
		t.Fatal("the runaway was not cut")
	}
	if fed > 2048 {
		t.Errorf("cut after %d bytes, the guard alone does it inside 2048", fed)
	}
}

// Output that repeats and is not looping. Each of these would make the detector
// unusable if it fired.
func TestOrdinaryRepetitionIsNotALoop(t *testing.T) {
	rows := func(format string, n int) string {
		var b strings.Builder
		for i := range n {
			fmt.Fprintf(&b, format, i, i*7+3)
		}
		return b.String()
	}
	cases := map[string]string{
		"blank lines":   strings.Repeat("\n", 4000),
		"table rule":    strings.Repeat("| --- | --- |\n", 300),
		"indentation":   strings.Repeat("\t\t\t\t", 2000),
		"numbered list": rows("The package number %d builds cleanly and every test in it passes on this branch %d.\n", 30),
		"numbered rows": rows("| item %04d | ok | 2026-07-28 | a reasonably long description %d |\n", 400),
		"log lines":     rows("2026-07-28T10:00:00Z INFO worker %d finished job in %d milliseconds without error\n", 400),
		"boilerplate": rows("EXCLUSIVE OF THE EASTERLY 100 FEET OF SAID LOTS %d TO %d;\n"+
			"together with the appurtenances thereto, situate in the County of Record.\n", 200),
		"go code": rows("if err := step%d(ctx); err != nil {\n\treturn fmt.Errorf(\"step %d: %%w\", err)\n}\n", 200),
	}
	for name, text := range cases {
		t.Run(name, func(t *testing.T) {
			s := New(Config{Tolerance: -1}).Begin(nil)
			feedAll(s, llm.ChannelContent, text)
			if loops, kind := s.Loops(); loops != 0 {
				t.Errorf("counted %d %s loops", loops, kind)
			}
		})
	}
}

func TestHistoryIsCountedButDoesNotInterruptUntilTheReplyContinues(t *testing.T) {
	round := strings.Join(variations(30), "")
	history := []llm.Message{{Role: "user", Content: "go"}}
	for range 4 {
		history = append(history, llm.Message{Role: "assistant", Content: round})
	}
	s := New(Config{}).Begin(history)
	if loops, _ := s.Loops(); loops < 3 {
		t.Fatalf("Loops() = %d from four identical replies", loops)
	}
	// A new subject: the model got out of the loop, and must be let.
	if info, _ := feedAll(s, llm.ChannelContent, "That was going nowhere, so here is the actual answer, which is new.\n"); info != nil {
		t.Fatalf("a reply that left the loop was cut: %v", info)
	}
	if info, _ := feedAll(s, llm.ChannelContent, round); info == nil {
		t.Fatal("a reply that continued the loop was not cut")
	}
}

func toolHistory(rounds int, calls []string, result func(round, call int) string) []llm.Message {
	var h []llm.Message
	id := 0
	for r := range rounds {
		for c, name := range calls {
			tc := llm.ToolCall{ID: fmt.Sprint("c", id)}
			tc.Function.Name = name
			tc.Function.Arguments = fmt.Sprintf(`{"call":%d,"path":"a"}`, c)
			h = append(h, llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{tc}},
				llm.Message{Role: "tool", ToolCallID: tc.ID, Content: result(r, c)})
			id++
		}
	}
	return h
}

func TestToolCallsWithTheSameResultAreALoopAcrossACycle(t *testing.T) {
	calls := make([]string, 7)
	for i := range calls {
		calls[i] = fmt.Sprintf("tool%d", i)
	}
	s := New(Config{}).Begin(toolHistory(4, calls, func(_, c int) string { return fmt.Sprint("same ", c) }))
	if loops, kind := s.Loops(); loops < 3 || kind != KindTool {
		t.Fatalf("Loops() = %d, %q", loops, kind)
	}
	if info := s.WatchToolCall("tool0", `{"path":"a","call":0}`); info == nil {
		t.Fatal("the next call of the cycle was not cut (key order must not matter)")
	}
}

func TestAToolCallWhoseResultChangesIsProgress(t *testing.T) {
	s := New(Config{}).Begin(toolHistory(20, []string{"status"}, func(r, _ int) string { return fmt.Sprint("step ", r) }))
	if loops, _ := s.Loops(); loops != 0 {
		t.Fatalf("a poll whose answer advances counted %d loops", loops)
	}
}

func TestPollingAnUnchangedResultIsALoop(t *testing.T) {
	s := New(Config{}).Begin(toolHistory(20, []string{"jobs"}, func(_, _ int) string { return "running" }))
	if loops, kind := s.Loops(); loops < 3 || kind != KindTool {
		t.Fatalf("Loops() = %d, %q", loops, kind)
	}
}

func TestToolArgumentsGetTheExactDetectorOnly(t *testing.T) {
	file := strings.Join(variations(40), "\n")
	history := []llm.Message{{Role: "assistant", Content: file}, {Role: "assistant", Content: file}}
	s := New(Config{}).Begin(history)
	if info, _ := feedAll(s, llm.ToolChannel(0), file); info != nil {
		t.Fatalf("writing out text from the conversation was cut: %v", info)
	}
	block := strings.Repeat("0123456789abcdefghijklmnopqrstuvwxyz", 3)
	if info, _ := feedAll(s, llm.ToolChannel(1), strings.Repeat(block, 40)); info == nil {
		t.Fatal("a loop inside arguments was not cut")
	}
}

func TestOffDoesNothing(t *testing.T) {
	d := New(Config{Off: true})
	s := d.Begin(toolHistory(9, []string{"a"}, func(_, _ int) string { return "x" }))
	if info, _ := feedAll(s, llm.ChannelContent, strings.Repeat(strings.Join(variations(40), ""), 9)); info != nil {
		t.Fatal(info)
	}
	if loops, _ := s.Loops(); loops != 0 {
		t.Fatal(loops)
	}
}

func TestSentenceSplitting(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"no end yet", -1},
		{"line\nnext", 5},
		{"Pi is 3.14 and more. Next", 20},
		{"see example.com now", -1},
		{"ends here.", -1}, // what follows is not known yet
		{strings.Repeat("x", 600), 512},
	}
	for _, c := range cases {
		if got := sentenceEnd([]byte(c.in), 512); got != c.want {
			t.Errorf("sentenceEnd(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

// The detector end to end through llm.Client: a reply looping in its thinking is
// cut, and the caller is told, with nothing to trim from the reply.
func TestInstalledOnAClientItCutsAStream(t *testing.T) {
	sent := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; ; i++ {
			v := variations(40)[i%40]
			if _, err := fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":%q}}]}\n\n", v); err != nil {
				return
			}
			w.(http.Flusher).Flush()
			sent++
			if r.Context().Err() != nil || i > 5000 {
				return
			}
		}
	}))
	defer srv.Close()
	c := llm.NewClient(srv.URL, "", "m")
	d := New(Config{})
	d.Install(c)
	ch, err := c.ChatStream(context.Background(), []llm.Message{{Role: "user", Content: "go"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var last llm.StreamChunk
	for chunk := range ch {
		if chunk.Done {
			last = chunk
		}
	}
	if last.StopReason != llm.StopReasonRepetition || last.Repetition == nil {
		t.Fatalf("final chunk: %+v", last)
	}
	if r := last.Repetition; r.Where != llm.ChannelReasoning || r.Trailing != 0 || r.Reps != 3 {
		t.Errorf("finding: %+v", r)
	}
	if got := d.Counters().Interrupted[KindCycle]; got != 1 {
		t.Errorf("Interrupted = %d", got)
	}
}

func BenchmarkReplyOf64KB(b *testing.B) {
	text := strings.Repeat(strings.Join(variations(200), ""), 8)[:64<<10]
	d := New(Config{Tolerance: -1})
	b.SetBytes(int64(len(text)))
	for range b.N {
		feedAll(d.Begin(nil), llm.ChannelContent, text)
	}
}
