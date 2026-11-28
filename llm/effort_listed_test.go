package llm

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// A refusal that LISTS the levels the endpoint takes teaches the whole ladder
// in one pass: unlisted levels refused, listed ones accepted. none is not part
// of that (a level list says nothing about thinking off).

// llamaCppRefusal is verbatim from corrallm → llama.cpp (HTTP 400), for "max".
const llamaCppRefusal = `{"error":{"code":400,"message":"reasoning_effort \"max\" is not supported by the chat template, supported levels: low, medium, xhigh","type":"invalid_request_error"}}`

// levelsOf groups the list facts of us by key.
func levelsOf(us []factUpdate) map[string][]string {
	m := map[string][]string{}
	for _, u := range us {
		m[u.Key] = append(m[u.Key], u.Value)
	}
	return m
}

func llamaRefusing() func(int, map[string]any) (int, string) {
	return func(_ int, b map[string]any) (int, string) {
		switch e := sentEffortOf(b); e {
		case "low", "medium", "xhigh":
			return 200, okReply
		default:
			return 400, strings.ReplaceAll(llamaCppRefusal, `\"max\"`, `\"`+e+`\"`)
		}
	}
}

func chatAt(c *Client, e Effort) error {
	_, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: e})
	return err
}

func TestLlamaCppTemplateRefusalOfMaxIsLearnedResentOnceAndNeverAskedAgain(t *testing.T) {
	srv := newCapServer(t, llamaRefusing())
	c := openaiClient(srv.URL)
	if err := chatAt(c, EffortMax); err != nil {
		t.Fatalf("resend did not succeed: %v", err)
	}
	var got []string
	for _, b := range srv.sent() {
		got = append(got, sentEffortOf(b))
	}
	// Lower first (CLAUDE.md "One effort path"): xhigh is the nearest level below max
	// that the list accepts.
	if strings.Join(got, ",") != "max,xhigh" {
		t.Fatalf("efforts sent %v, want max then xhigh", got)
	}
	r, _ := c.Capabilities().RefusedEfforts()
	a, _ := c.Capabilities().AcceptedEfforts()
	if !slices.Equal(r, []string{"minimal", "high", "max"}) || !slices.Equal(a, []string{"low", "medium", "xhigh"}) {
		t.Fatalf("refused %v accepted %v, want the full ladder split by the list (none untouched)", r, a)
	}
	for _, e := range []Effort{EffortMax, EffortHigh} {
		before := len(srv.sent())
		if err := chatAt(c, e); err != nil {
			t.Fatal(err)
		}
		if n := len(srv.sent()) - before; n != 1 {
			t.Fatalf("%s after learning cost %d requests, want 1 (no 400)", e, n)
		}
	}
}

func TestLlamaCppTemplateRefusalOfHighIsLearnedAndResentOnce(t *testing.T) {
	srv := newCapServer(t, llamaRefusing())
	c := openaiClient(srv.URL)
	if err := chatAt(c, EffortHigh); err != nil {
		t.Fatalf("resend did not succeed: %v", err)
	}
	var got []string
	for _, b := range srv.sent() {
		got = append(got, sentEffortOf(b))
	}
	if strings.Join(got, ",") != "high,medium" {
		t.Fatalf("efforts sent %v, want high then medium", got)
	}
}

func TestOpenAISupportedValuesListTeachesTheFullSetNotJustTheSentLevel(t *testing.T) {
	f := levelsOf(refusalFacts(400, openaiMaxRefusal, map[string]any{"reasoning_effort": "max"}))
	if !slices.Contains(f[CapReasoningEffortsRefused], "max") || !slices.Contains(f[CapReasoningEffortsRefused], "xhigh") || slices.Contains(f[CapReasoningEffortsRefused], "none") {
		t.Fatalf("refused = %v", f[CapReasoningEffortsRefused])
	}
	for _, l := range []string{"low", "medium", "high"} {
		if !slices.Contains(f[CapReasoningEffortsAccepted], l) {
			t.Fatalf("accepted = %v, want %s", f[CapReasoningEffortsAccepted], l)
		}
	}
}

func TestRefusalNamingAnotherEffortThanSentTeachesNothingEvenWithAList(t *testing.T) {
	for _, body := range []string{llamaCppRefusal, openaiMaxRefusal} {
		if f := refusalFacts(400, body, map[string]any{"reasoning_effort": "high"}); len(f) != 0 {
			t.Fatalf("learned from a value not sent: %+v", f)
		}
	}
}

func TestLlamaCppListEvidenceNamesTheBody(t *testing.T) {
	us := refusalFacts(400, llamaCppRefusal, map[string]any{"reasoning_effort": "max"})
	for _, u := range us {
		if !strings.Contains(u.Evidence, "supported levels") && u.Value != "max" {
			t.Errorf("evidence %q does not name the list", u.Evidence)
		}
	}
}
