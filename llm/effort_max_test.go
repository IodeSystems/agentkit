package llm

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// Max is sensed per endpoint and model on openai and openrouter: sent as
// asked, a 400 that names it is learned into reasoning.efforts.refused and the
// request is resent once a level down; a 200 is learned as accepted. A static
// row rewrote max to xhigh for every model, including the ones that take it.

// openaiMaxRefusal is OpenAI's shape for an effort value the model does not take.
const openaiMaxRefusal = `{"error":{"message":"Invalid value: 'max'. Supported values are: 'low', 'medium', and 'high'.","type":"invalid_request_error","param":"reasoning_effort","code":"unsupported_value"}}`

func openrouterClient(url string) *Client {
	c := NewClient(url, "k", "some/model")
	c.SetReasoningDialect(DialectOpenRouter)
	return c
}

// effortDialects: each dialect under test, with its client.
var effortDialects = []struct {
	name   string
	client func(url string) *Client
}{
	{DialectOpenAI, openaiClient},
	{DialectOpenRouter, openrouterClient},
}

func sentEffortOf(b map[string]any) string { e, _ := sentEffort(b); return e }

// refuseEfforts answers 400 (OpenAI's shape, naming the value) to each listed
// effort and 200 to anything else.
func refuseEfforts(levels ...string) func(int, map[string]any) (int, string) {
	return refuseEffortsWith(openaiMaxRefusal, levels...)
}

// bareRefusal names the value and lists nothing, so only the sent level is
// learned and the ladder's one-step-at-a-time clamp is what the tests see.
const bareRefusal = `{"error":{"message":"'max' is not one of ['low', 'medium', 'high']"}}`

func refuseEffortsBare(levels ...string) func(int, map[string]any) (int, string) {
	return refuseEffortsWith(bareRefusal, levels...)
}

func refuseEffortsWith(body string, levels ...string) func(int, map[string]any) (int, string) {
	return func(_ int, b map[string]any) (int, string) {
		if e := sentEffortOf(b); slices.Contains(levels, e) {
			return 400, strings.ReplaceAll(body, "'max'", "'"+e+"'")
		}
		return 200, okReply
	}
}

func chatMax(t *testing.T, c *Client) error {
	t.Helper()
	_, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: EffortMax})
	return err
}

func TestFreshRecordSendsMaxNotAStaticXHighRewrite(t *testing.T) {
	for _, d := range effortDialects {
		t.Run(d.name, func(t *testing.T) {
			srv := newCapServer(t, refuseEfforts())
			c := d.client(srv.URL)
			var trs []Translation
			c.OnTranslate = func(tr Translation) { trs = append(trs, tr) }
			if err := chatMax(t, c); err != nil {
				t.Fatal(err)
			}
			if sent := srv.sent(); len(sent) != 1 || sentEffortOf(sent[0]) != "max" {
				t.Fatalf("sent %v, want one request at max", sent)
			}
			if len(trs) != 0 {
				t.Fatalf("an exact level was reported as changed: %+v", trs)
			}
		})
	}
}

func TestRefusedMaxIsLearnedAndResentOnceAtXHighWithAReason(t *testing.T) {
	for _, d := range effortDialects {
		t.Run(d.name, func(t *testing.T) {
			srv := newCapServer(t, refuseEffortsBare("max"))
			c := d.client(srv.URL)
			var trs []Translation
			c.OnTranslate = func(tr Translation) { trs = append(trs, tr) }
			if err := chatMax(t, c); err != nil {
				t.Fatalf("resend did not succeed: %v", err)
			}
			sent := srv.sent()
			if len(sent) != 2 || sentEffortOf(sent[0]) != "max" || sentEffortOf(sent[1]) != "xhigh" {
				t.Fatalf("sent %d requests, efforts %v; want max then xhigh", len(sent), sent)
			}
			if r, _ := c.Capabilities().RefusedEfforts(); !slices.Contains(r, "max") {
				t.Fatalf("refused = %v, want max", r)
			}
			if len(trs) != 1 || trs[0].Key != "effort" || trs[0].Wanted != "max" || trs[0].Sent != "xhigh" ||
				!strings.Contains(trs[0].Reason, CapReasoningEffortsRefused) {
				t.Fatalf("translations = %+v, want one effort max→xhigh naming the refusal", trs)
			}
		})
	}
}

func TestAfterMaxRefusalALaterRequestGoesStraightToXHigh(t *testing.T) {
	for _, d := range effortDialects {
		t.Run(d.name, func(t *testing.T) {
			srv := newCapServer(t, refuseEffortsBare("max"))
			c := d.client(srv.URL)
			if err := chatMax(t, c); err != nil {
				t.Fatal(err)
			}
			if err := chatMax(t, c); err != nil {
				t.Fatal(err)
			}
			sent := srv.sent()
			if len(sent) != 3 || sentEffortOf(sent[2]) != "xhigh" {
				t.Fatalf("sent %d requests, want 3 with the last at xhigh (no refused round trip)", len(sent))
			}
		})
	}
}

// The step down from a refused max is the existing ladder: xhigh first, and
// high once xhigh is refused too, never a jump straight to high.
func TestRefusedXHighAfterMaxStepsOnToHigh(t *testing.T) {
	for _, d := range effortDialects {
		t.Run(d.name, func(t *testing.T) {
			srv := newCapServer(t, refuseEffortsBare("max", "xhigh"))
			c := d.client(srv.URL)
			if err := chatMax(t, c); err == nil {
				t.Fatal("want the resend's 400 (one resend only)")
			}
			if err := chatMax(t, c); err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, b := range srv.sent() {
				got = append(got, sentEffortOf(b))
			}
			if strings.Join(got, ",") != "max,xhigh,high" {
				t.Fatalf("efforts sent %v, want max,xhigh,high", got)
			}
		})
	}
}

func TestAcceptedMaxIsRecordedAsAccepted(t *testing.T) {
	for _, d := range effortDialects {
		t.Run(d.name, func(t *testing.T) {
			srv := newCapServer(t, refuseEfforts())
			c := d.client(srv.URL)
			if err := chatMax(t, c); err != nil {
				t.Fatal(err)
			}
			if a, _ := c.Capabilities().AcceptedEfforts(); !slices.Contains(a, "max") {
				t.Fatalf("accepted = %v, want max", a)
			}
		})
	}
}

func TestRefusalFactsReadsOpenAIInvalidValueShapeOnlyForTheSentEffort(t *testing.T) {
	if f := factsOf(refusalFacts(400, bareRefusal, map[string]any{"reasoning_effort": "max"})); f[CapReasoningEffortsRefused] != "max" {
		t.Fatalf("openai field: %v", f)
	}
	if f := levelsOf(refusalFacts(400, openaiMaxRefusal, map[string]any{"reasoning": map[string]any{"effort": "max"}})); !slices.Contains(f[CapReasoningEffortsRefused], "max") {
		t.Fatalf("openrouter field: %v", f)
	}
	// A different value was sent: the refusal is not about it.
	if f := factsOf(refusalFacts(400, openaiMaxRefusal, map[string]any{"reasoning_effort": "high"})); len(f) != 0 {
		t.Fatalf("learned from a value not sent: %v", f)
	}
	// "max" in some other field's refusal is not an effort refusal.
	other := strings.ReplaceAll(openaiMaxRefusal, `"param":"reasoning_effort"`, `"param":"verbosity"`)
	if f := factsOf(refusalFacts(400, other, map[string]any{"reasoning_effort": "max"})); len(f) != 0 {
		t.Fatalf("learned from another field's refusal: %v", f)
	}
}

func TestProbeEffortLadderSendsMax(t *testing.T) {
	p := &probeServer{refuse: map[string]int{"max": 400}, refuseBody: openaiMaxRefusal}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	efforts := append([]string(nil), p.efforts...)
	p.mu.Unlock()
	if !slices.Contains(efforts, "max") {
		t.Fatalf("probe sent %v, want max among them", efforts)
	}
	if got := factValue(caps, CapReasoningEffortsRefused); got != "max" {
		t.Fatalf("refused = %q, want max", got)
	}
}
