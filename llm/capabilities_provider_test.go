package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"
)

// routedReply is OpenRouter's non-stream reply naming the upstream that served it.
func routedReply(provider string) string {
	return `{"provider":"` + provider + `","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`
}

// routedMaxRefusal is OpenAI's max refusal as OpenRouter relays it: the
// upstream named in error.metadata.provider_name.
func routedMaxRefusal(provider string) string {
	return `{"error":{"message":"Invalid value: 'max'. Supported values are: 'low', 'medium', and 'high'.","code":400,"param":"reasoning_effort","metadata":{"provider_name":"` + provider + `"}}}`
}

func TestProviderOfJoinsServedHeaderAndBodyProvider(t *testing.T) {
	h := http.Header{}
	h.Set(HeaderServed, "glm-lane-b")
	cases := []struct {
		name string
		h    http.Header
		body string
		want string
	}{
		{"direct endpoint", nil, okReply, ""},
		{"corrallm header only", h, okReply, "glm-lane-b"},
		{"openrouter body provider", nil, routedReply("DeepInfra"), "DeepInfra"},
		{"openrouter error metadata", nil, routedMaxRefusal("OpenAI"), "OpenAI"},
		{"both", h, routedReply("DeepInfra"), "glm-lane-b/DeepInfra"},
		{"not json", nil, "bad gateway", ""},
	}
	for _, tc := range cases {
		if got := providerOf(tc.h, []byte(tc.body)); got != tc.want {
			t.Errorf("%s: %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestRefusalIsLearnedIntoTheRecordOfTheRouteThatRefused(t *testing.T) {
	srv := newCapServer(t, func(n int, b map[string]any) (int, string) {
		if sentEffortOf(b) == "max" {
			return 400, routedMaxRefusal("OpenAI")
		}
		return 200, routedReply("OpenAI")
	})
	c := openrouterClient(srv.URL)
	var learned []Learned
	c.OnLearn = func(l Learned) { learned = append(learned, l) }
	if err := chatMax(t, c); err != nil {
		t.Fatal(err)
	}
	for _, l := range learned {
		if l.Provider != "OpenAI" {
			t.Errorf("learned %s into %q, want OpenAI", l.Key, l.Provider)
		}
	}
	recs := c.ProviderCapabilities()
	if len(recs) != 1 || recs[0].Provider != "OpenAI" {
		t.Fatalf("records %+v, want one for OpenAI", recs)
	}
	if r, _ := recs[0].RefusedEfforts(); !slices.Contains(r, "max") {
		t.Fatalf("OpenAI refused = %v", r)
	}
}

// Unpinned: the next request may land on any route, so a refusal learned on
// one route shapes it, whichever route answered before.
func TestUnpinnedRequestUsesEveryRoutesRefusalNarrowestWins(t *testing.T) {
	c := openrouterClient("http://x/v1")
	c.SetCapabilities(Capabilities{Provider: "OpenAI", Facts: map[string]Fact{
		CapReasoningEffortsRefused: fact("max", SourceRefused),
		CapLimitsOutput:            fact("8000", SourceRefused),
	}})
	c.SetCapabilities(Capabilities{Provider: "DeepInfra", Facts: map[string]Fact{
		CapReasoningEffortsAccepted: fact("max", SourceObserved),
		CapLimitsOutput:             fact("4000", SourceRefused),
	}})
	body, _ := bodyOf(t, c, &ChatOpts{Effort: EffortMax})
	if sentEffortOf(body) != "xhigh" || body["max_tokens"] != 4000 {
		t.Fatalf("effort %q max_tokens %v, want xhigh and the smallest limit 4000", sentEffortOf(body), body["max_tokens"])
	}
	if acc, _ := c.Capabilities().AcceptedEfforts(); slices.Contains(acc, "max") {
		t.Fatalf("merged accepted %v still holds a refused effort", acc)
	}
}

// Pinned: a route the request cannot reach does not narrow it, and the pinned
// route's own refusal does.
func TestPinnedProviderUsesOnlyThatRoutesRecordPlusEndpointWide(t *testing.T) {
	c := openrouterClient("http://x/v1")
	c.SetCapabilities(Capabilities{Provider: "OpenAI", Facts: map[string]Fact{CapReasoningEffortsRefused: fact("max", SourceRefused)}})
	c.SetCapabilities(Capabilities{Provider: "", Facts: map[string]Fact{CapLimitsOutput: fact("4000", SourceDeclared)}})
	no := false
	toDeepInfra := &ChatOpts{Effort: EffortMax, Provider: &ProviderRouting{Order: []string{"deepinfra"}, AllowFallbacks: &no}}
	body, _ := bodyOf(t, c, toDeepInfra)
	if sentEffortOf(body) != "max" || body["max_tokens"] != 4000 {
		t.Fatalf("pinned to deepinfra: effort %q max_tokens %v, want max and the endpoint-wide 4000", sentEffortOf(body), body["max_tokens"])
	}
	body, _ = bodyOf(t, c, &ChatOpts{Effort: EffortMax, Provider: &ProviderRouting{Only: []string{"openai"}}})
	if sentEffortOf(body) != "xhigh" {
		t.Fatalf("pinned to openai (case differs from the response name): effort %q, want xhigh", sentEffortOf(body))
	}
	// Order WITH fallbacks pins nothing: any route can still serve it.
	body, _ = bodyOf(t, c, &ChatOpts{Effort: EffortMax, Provider: &ProviderRouting{Order: []string{"deepinfra"}}})
	if sentEffortOf(body) != "xhigh" {
		t.Fatalf("order with fallbacks: effort %q, want xhigh", sentEffortOf(body))
	}
}

func TestProviderRoutingIsSentAsGiven(t *testing.T) {
	c := openrouterClient("http://x/v1")
	no := false
	body, _ := bodyOf(t, c, &ChatOpts{Provider: &ProviderRouting{Order: []string{"a", "b"}, AllowFallbacks: &no}})
	raw, _ := json.Marshal(body["provider"])
	if string(raw) != `{"order":["a","b"],"allow_fallbacks":false}` {
		t.Fatalf("provider = %s", raw)
	}
	if body, _ := bodyOf(t, c, &ChatOpts{}); body["provider"] != nil {
		t.Fatalf("provider sent unasked: %v", body["provider"])
	}
}

func TestProviderMatchesTheLastSegmentOfAJoinedRoute(t *testing.T) {
	if !providerMatches("openai/gpt-x-lane/DeepInfra", []string{"deepinfra"}) {
		t.Error("joined key not matched by its provider")
	}
	if providerMatches("NotDeepInfra", []string{"deepinfra"}) {
		t.Error("a suffix that is not a segment matched")
	}
}

func TestMergeNarrowestRules(t *testing.T) {
	old, mid, now := time.Unix(1, 0), time.Unix(2, 0), time.Unix(3, 0)
	at := func(v string, s Source, t time.Time) Fact { return Fact{Value: v, Source: s, At: t} }
	a := map[string]Fact{
		CapReasoningEffortsRefused: at("max", SourceRefused, old),
		CapLimitsContext:           at("32000", SourceObserved, old),
		CapReasoningBudgetMin:      at("512", SourceRefused, old),
		CapDecodeGrammar:           at("true", SourceProbed, old),
		CapSamplingPrefix + "seed": at("accepted", SourceObserved, old),
		CapReasoningOff:            at("minimal", SourceProbed, old),
		CapLimitsOutput:            at("9000", SourceHost, old),
	}
	b := map[string]Fact{
		CapReasoningEffortsRefused: at("none,xhigh", SourceRefused, mid),
		CapLimitsContext:           at("16000", SourceObserved, mid),
		CapReasoningBudgetMin:      at("1024", SourceRefused, mid),
		CapDecodeGrammar:           at("false", SourceProbed, mid),
		CapSamplingPrefix + "seed": at("refused", SourceRefused, mid),
		CapReasoningOff:            at("low", SourceProbed, now),
		CapLimitsOutput:            at("2000", SourceRefused, now),
	}
	m := mergeNarrowest([]map[string]Fact{a, b})
	want := map[string]string{
		CapReasoningEffortsRefused: "none,xhigh,max",
		CapLimitsContext:           "16000",
		CapReasoningBudgetMin:      "1024",
		CapDecodeGrammar:           "false",
		CapSamplingPrefix + "seed": "refused",
		CapReasoningOff:            "low",  // newest
		CapLimitsOutput:            "9000", // a host pin beats a learned limit
	}
	for k, v := range want {
		if m[k].Value != v {
			t.Errorf("%s = %q, want %q", k, m[k].Value, v)
		}
	}
}

func TestSetCapabilitiesReplacesOnlyItsProvidersRecord(t *testing.T) {
	c := openrouterClient("http://x/v1")
	c.SetCapabilities(Capabilities{Provider: "A", Facts: map[string]Fact{CapLimitsOutput: fact("1", SourceHost)}})
	c.SetCapabilities(Capabilities{Provider: "B", Facts: map[string]Fact{CapLimitsOutput: fact("2", SourceHost)}})
	c.SetCapabilities(Capabilities{Provider: "A", Facts: map[string]Fact{CapLimitsContext: fact("3", SourceHost)}})
	recs := c.ProviderCapabilities()
	if len(recs) != 2 || recs[0].Provider != "A" || recs[1].Provider != "B" ||
		recs[0].Facts[CapLimitsOutput].Value != "" || recs[1].Facts[CapLimitsOutput].Value != "2" {
		t.Fatalf("%+v", recs)
	}
}

// A stream names its route in its chunks (OpenRouter) or in a late SSE comment
// (corrallm); the accepted effort and the stream facts go to that route.
func TestStreamLearnsIntoTheRouteItsChunksAndLateHeaderName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set(HeaderServed, "lane-a")
		w.Write([]byte(": x-corrallm-served: lane-b\n\n" +
			sse(`{"provider":"DeepInfra","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`, `[DONE]`)))
	}))
	t.Cleanup(srv.Close)
	c := openrouterClient(srv.URL)
	drain(t, c, &ChatOpts{Effort: EffortMax}, nil)
	recs := c.ProviderCapabilities()
	if len(recs) != 1 || recs[0].Provider != "lane-b/DeepInfra" {
		t.Fatalf("records %+v, want one for lane-b/DeepInfra", recs)
	}
	if acc, _ := recs[0].AcceptedEfforts(); !slices.Contains(acc, "max") {
		t.Fatalf("accepted = %v", acc)
	}
}

func TestStreamErrorChunkNamesItsRoute(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sse(`{"choices":[{"delta":{"reasoning":"r"}}]}`,
			`{"error":{"code":400,"message":"nope","metadata":{"provider_name":"Google"}}}`)))
	}))
	t.Cleanup(srv.Close)
	c := openrouterClient(srv.URL)
	drain(t, c, &ChatOpts{Effort: EffortHigh}, nil)
	recs := c.ProviderCapabilities()
	if len(recs) != 1 || recs[0].Provider != "Google" {
		t.Fatalf("records %+v, want one for Google", recs)
	}
}

func TestLearnResponseFromUsesTheBackendsServedHeader(t *testing.T) {
	c := openaiClient("http://x/v1")
	h := http.Header{}
	h.Set(HeaderServed, "lane-a")
	if !c.LearnResponseFrom(map[string]any{"reasoning_effort": "max"}, 400, h, []byte(openaiMaxRefusal)) {
		t.Fatal("taught nothing")
	}
	recs := c.ProviderCapabilities()
	if len(recs) != 1 || recs[0].Provider != "lane-a" {
		t.Fatalf("%+v", recs)
	}
	// The unpinned view carries it.
	if r, _ := c.Capabilities().RefusedEfforts(); !strings.Contains(strings.Join(r, ","), "max") {
		t.Fatalf("merged refused = %v", r)
	}
}
