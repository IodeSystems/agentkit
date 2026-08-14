package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// The measured failure: an 85-character fragment of a legal property
// description, re-emitted until the server's context limit — ~30 minutes.
const loopingBlock = "EXCLUSIVE OF THE EASTERLY 100 FEET\\nOF SAID LOTS 7 TO 10, EXCLUSIVE OF THE EASTERLY "

func TestRepetitionTripsOnTheMeasuredLoop(t *testing.T) {
	w := newRepWatcher(RepetitionGuard{}, "content")
	var info *RepetitionInfo
	for i := 0; i < 200 && info == nil; i++ {
		info = w.feed(loopingBlock)
	}
	if info == nil {
		t.Fatal("a block repeated 200 times was not detected as a loop")
	}
	if info.Period != len(loopingBlock) {
		t.Errorf("period = %d, want the block length %d (a multiple means the shortest period was missed)",
			info.Period, len(loopingBlock))
	}
	if info.Trailing != (info.Reps-1)*info.Period {
		t.Errorf("Trailing = %d, want (%d-1)*%d — the redundant copies, leaving one intact",
			info.Trailing, info.Reps, info.Period)
	}
	// It must cut EARLY. Detecting the loop after a megabyte is not a fix.
	if got := info.Reps * info.Period; got > 4096 {
		t.Errorf("detected only after %d bytes; the point is to cut before the tokens are spent", got)
	}
}

// The real thing: 3,953 captured characters of the generation that ran 29.8
// minutes and produced 176,309 tokens from a 3,912-token prompt (corrallm
// activity id 107421, 2026-07-28 13:43, Qwen3-6-27B-MPT, finish_reason=length).
// It is a county surveyor's plat being transcribed by the VLM OCR path; the
// model transcribes cleanly for fifteen lines, then locks onto one clause.
//
// Note WHY this document seeds a loop: the source text legitimately repeats
// ("EXCLUSIVE OF ... EXCLUSIVE OF ..."), which is ordinary in a legal
// description. So the fixture also pins the boundary — the guard must fire on
// the runaway without firing on the document's own repetition.
func TestRepetitionCutsTheCapturedOCRRunaway(t *testing.T) {
	b, err := os.ReadFile("testdata/looping_ocr.txt")
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}
	text := string(b)

	w := newRepWatcher(RepetitionGuard{}, "content")
	var info *RepetitionInfo
	var fed int
	for i := 0; i < len(text) && info == nil; i += 64 {
		j := min(i+64, len(text))
		fed = j
		info = w.feed(text[i:j])
	}
	if info == nil {
		t.Fatal("the guard did not fire on the generation that burned 30 minutes of GPU")
	}
	// The whole point is cutting EARLY. This ran to 176,309 tokens; the fixture
	// is the first ~4 KB, and detection must land well inside it.
	if fed > 2048 {
		t.Errorf("detected only after %d chars; too late to matter", fed)
	}
	if info.Period < 24 {
		t.Errorf("period %d is below MinPeriod — the minimal-period reduction is wrong", info.Period)
	}
	t.Logf("cut after %d chars: %s", fed, info)
}

// Legitimate output repeats constantly. These are the shapes that would make
// the guard unusable if it fired on them.
func TestRepetitionIgnoresOrdinaryRepetition(t *testing.T) {
	cases := map[string]string{
		"blank lines":         strings.Repeat("\n", 4000),
		"markdown table rule": strings.Repeat("| --- | --- |\n", 300),
		"indentation":         strings.Repeat("\t\t\t\t", 2000),
		"near-identical rows": func() string {
			var b strings.Builder
			for i := 0; i < 400; i++ {
				fmt.Fprintf(&b, "| item %04d | ok | 2026-07-28 | a reasonably long description |\n", i)
			}
			return b.String()
		}(),
		// Boilerplate that recurs but never back-to-back — the shape of a real
		// document, and the one most likely to be mistaken for a loop.
		"recurring boilerplate": func() string {
			var b strings.Builder
			for i := 0; i < 200; i++ {
				fmt.Fprintf(&b, "EXCLUSIVE OF THE EASTERLY 100 FEET OF SAID LOTS %d TO %d;\n"+
					"together with the appurtenances thereto, situate in the County of Record.\n",
					i, i+3)
			}
			return b.String()
		}(),
	}
	for name, text := range cases {
		t.Run(name, func(t *testing.T) {
			w := newRepWatcher(RepetitionGuard{}, "content")
			for i := 0; i+64 <= len(text); i += 64 {
				if info := w.feed(text[i : i+64]); info != nil {
					t.Fatalf("false positive on %s: %s", name, info)
				}
			}
		})
	}
}

// Blank lines and single characters ARE repetition, but a guard that fires on
// them fires on everything. Only MinPeriod separates the two, so pin it.
func TestRepetitionRespectsMinPeriod(t *testing.T) {
	short := strings.Repeat("ab", 4000)
	if info := newRepWatcher(RepetitionGuard{}, "content").feed(short); info != nil {
		t.Fatalf("a 2-char cycle tripped the default guard: %s", info)
	}
	loose := RepetitionGuard{MinPeriod: 2, MinSpan: 64}
	if info := newRepWatcher(loose, "content").feed(short); info == nil {
		t.Fatal("a 2-char cycle was not detected even with MinPeriod=2")
	}
}

func TestRepetitionOffDisablesDetection(t *testing.T) {
	w := newRepWatcher(RepetitionGuard{Off: true}, "content")
	for i := 0; i < 200; i++ {
		if info := w.feed(loopingBlock); info != nil {
			t.Fatalf("Off did not disable detection: %s", info)
		}
	}
}

// The stream must be CUT, not merely reported: the caller sees a Done chunk
// carrying the reason, and content stops arriving.
func TestChatStreamCutsALoopingGeneration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("test server cannot flush")
		}
		for i := 0; i < 5000; i++ {
			delta, _ := json.Marshal(map[string]any{
				"choices": []any{map[string]any{"delta": map[string]any{"content": loopingBlock}}},
			})
			fmt.Fprintf(w, "data: %s\n\n", delta)
			fl.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var chars int
	var stop StopReason
	var info *RepetitionInfo
	for chunk := range ch {
		chars += len(chunk.Content)
		if chunk.StopReason != "" {
			stop = chunk.StopReason
			info = chunk.Repetition
		}
	}
	if stop != StopReasonRepetition {
		t.Fatalf("StopReason = %q, want %q — the loop was not cut", stop, StopReasonRepetition)
	}
	if info == nil || info.Period != len(loopingBlock) {
		t.Fatalf("Repetition = %+v, want the %d-char block", info, len(loopingBlock))
	}
	// 5000 blocks were on offer; the cut must land in the first handful.
	if chars > 8192 {
		t.Errorf("consumed %d chars before cutting; the server offered %d",
			chars, 5000*len(loopingBlock))
	}
}

// The measured loop arrived inside a tool call's arguments, not in content.
func TestChatStreamCutsALoopInToolArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		head, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{
			"tool_calls": []any{map[string]any{"index": 0, "id": "c1", "type": "function",
				"function": map[string]any{"name": "record_legal", "arguments": `{"text":"`}}},
		}}}})
		fmt.Fprintf(w, "data: %s\n\n", head)
		fl.Flush()
		for i := 0; i < 5000; i++ {
			frag, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{
				"tool_calls": []any{map[string]any{"index": 0,
					"function": map[string]any{"arguments": loopingBlock}}},
			}}}})
			fmt.Fprintf(w, "data: %s\n\n", frag)
			fl.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var stop StopReason
	var where string
	var calls int
	for chunk := range ch {
		if chunk.ToolCall != nil {
			calls++
		}
		if chunk.StopReason != "" {
			stop = chunk.StopReason
			where = chunk.Repetition.Where
		}
	}
	if stop != StopReasonRepetition {
		t.Fatalf("StopReason = %q, want %q — a loop in tool arguments went undetected", stop, StopReasonRepetition)
	}
	if !strings.Contains(where, "record_legal") {
		t.Errorf("Where = %q, want the offending tool named", where)
	}
	// A call built from mid-loop arguments must never reach the caller.
	if calls != 0 {
		t.Errorf("emitted %d tool call(s) from a looping argument buffer; want 0", calls)
	}
}

// A cut is not a cure: it stops the burn but leaves the CAUSE, and at
// temperature 0 the cause is deterministic. The sampling knobs are the only
// thing that gives a retry a different outcome, so they must reach the wire.
func TestSamplingCarriesRepetitionControls(t *testing.T) {
	f, p, r := 0.5, 0.2, 1.1
	body := map[string]any{}
	NewClient("http://x/v1", "", "m").applySampling(body, &ChatOpts{FrequencyPenalty: &f, PresencePenalty: &p, RepeatPenalty: &r})
	for k, want := range map[string]float64{
		"frequency_penalty": 0.5, "presence_penalty": 0.2, "repeat_penalty": 1.1,
	} {
		got, ok := body[k]
		if !ok {
			t.Errorf("%s never reached the request body", k)
			continue
		}
		if got != want {
			t.Errorf("%s = %v, want %v", k, got, want)
		}
	}
	// Unset must stay absent — sending a 0 would silently DISABLE a penalty the
	// server was configured with.
	empty := map[string]any{}
	NewClient("http://x/v1", "", "m").applySampling(empty, &ChatOpts{})
	for _, k := range []string{"frequency_penalty", "presence_penalty", "repeat_penalty"} {
		if _, ok := empty[k]; ok {
			t.Errorf("%s was sent despite being unset", k)
		}
	}
}
