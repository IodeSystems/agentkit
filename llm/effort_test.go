package llm

import (
	"strings"
	"testing"
)

func effortCaps(kv ...string) Capabilities {
	m := map[string]Fact{}
	for i := 0; i < len(kv); i += 2 {
		m[kv[i]] = Fact{Value: kv[i+1], Source: SourceRefused, Evidence: "sent X → 400"}
	}
	return Capabilities{Facts: m}
}

// TestParseEffortAcceptsOnlyCanonicalNames: "off" is dun's alias, not ours; a
// second spelling here would be a second table.
func TestParseEffortAcceptsOnlyCanonicalNames(t *testing.T) {
	for _, e := range append([]Effort{EffortUnset}, Efforts...) {
		got, err := ParseEffort(string(e))
		if err != nil || got != e {
			t.Errorf("ParseEffort(%q) = %q, %v", e, got, err)
		}
	}
	for _, bad := range []string{"off", "on", "extreme", "0"} {
		if _, err := ParseEffort(bad); err == nil {
			t.Errorf("ParseEffort(%q) accepted", bad)
		}
	}
}

func TestEffortAboutCoversEveryLevel(t *testing.T) {
	for _, e := range append([]Effort{EffortUnset}, Efforts...) {
		if EffortAbout[e] == "" {
			t.Errorf("no description for %q", e)
		}
	}
}

// TestEffortForStaticTable: with an empty record, per dialect, what each level
// becomes and whether it is native or a budget approximation.
func TestEffortForStaticTable(t *testing.T) {
	type row struct {
		dialect string
		msgs    bool
		model   string
		e, want Effort
		native  bool
		reason  string // substring; "" = no reason expected
	}
	rows := []row{
		{DialectOpenAI, false, "m", EffortNone, EffortNone, true, ""},
		{DialectOpenAI, false, "m", EffortMax, EffortMax, true, ""},
		{DialectOpenRouter, false, "m", EffortNone, EffortNone, true, ""},
		{DialectOpenRouter, false, "m", EffortLow, EffortLow, true, ""},
		{DialectOpenRouter, false, "m", EffortMax, EffortMax, true, ""},
		{DialectLlamaCpp, false, "m", EffortNone, EffortNone, true, ""},
		{DialectLlamaCpp, false, "m", EffortMinimal, EffortMinimal, false, ""},
		{DialectLlamaCpp, false, "m", EffortMedium, EffortMedium, false, ""},
		{DialectLlamaCpp, false, "m", EffortHigh, EffortHigh, true, ""},
		{DialectLlamaCpp, false, "m", EffortXHigh, EffortHigh, true, "no xhigh"},
		{DialectLlamaCpp, false, "m", EffortMax, EffortHigh, true, "no max"},
		{DialectAnthropic, false, "claude-x", EffortNone, EffortNone, true, ""},
		{DialectAnthropic, false, "claude-x", EffortMinimal, EffortLow, false, "1024"},
		{DialectAnthropic, false, "claude-x", EffortMax, EffortMax, false, ""},
		{DialectAnthropic, true, "claude-opus-4-6", EffortNone, EffortLow, true, "cannot be turned off"},
		{DialectAnthropic, true, "claude-opus-4-6", EffortMinimal, EffortLow, true, "no minimal"},
		{DialectAnthropic, true, "claude-opus-4-6", EffortMax, EffortMax, true, ""},
		{DialectAnthropic, true, "claude-haiku-4-5", EffortHigh, EffortUnset, false, "model predates effort"},
		{DialectAnthropic, true, "claude-sonnet-4-5-20251001", EffortLow, EffortUnset, false, "model predates effort"},
	}
	for _, r := range rows {
		ch := EffortFor(r.dialect, r.msgs, r.model, Capabilities{}, r.e)
		if ch.Wanted != r.e || ch.Effective != r.want || (r.want != EffortUnset && ch.Native != r.native) {
			t.Errorf("%s msgs=%v %s %s: %+v, want %s native=%v", r.dialect, r.msgs, r.model, r.e, ch, r.want, r.native)
		}
		if (r.reason == "") != (ch.Reason == "") || !strings.Contains(ch.Reason, r.reason) {
			t.Errorf("%s %s: reason %q, want it to contain %q", r.dialect, r.e, ch.Reason, r.reason)
		}
	}
	if ch := EffortFor(DialectOpenAI, false, "m", Capabilities{}, EffortUnset); ch != (EffortChoice{}) {
		t.Errorf("unset = %+v", ch)
	}
}

// TestEffortForClampsRefusedLevelLowerFirst: decision 1 (Carl, 2026-11-18). A
// refused level goes to the nearest lower level not refused, then higher, and
// never down to none; none goes up from minimal, or to reasoning.off.
func TestEffortForClampsRefusedLevelLowerFirst(t *testing.T) {
	cases := []struct {
		name string
		e    Effort
		caps Capabilities
		want Effort
	}{
		{"high refused goes to medium", EffortHigh, effortCaps(CapReasoningEffortsRefused, "high"), EffortMedium},
		{"low refused with minimal ok goes to minimal", EffortLow, effortCaps(CapReasoningEffortsRefused, "low"), EffortMinimal},
		{"nothing lower goes higher", EffortLow, effortCaps(CapReasoningEffortsRefused, "minimal,low,medium"), EffortHigh},
		{"never down to none", EffortMinimal, effortCaps(CapReasoningEffortsRefused, "minimal,low"), EffortMedium},
		{"max refused goes to xhigh", EffortMax, effortCaps(CapReasoningEffortsRefused, "max"), EffortXHigh},
		{"none thinking-only goes up to minimal", EffortNone, effortCaps(CapReasoningThinkingOnly, "true"), EffortMinimal},
		{"none refused, minimal refused", EffortNone, effortCaps(CapReasoningEffortsRefused, "none,minimal"), EffortLow},
		{"none uses reasoning.off", EffortNone, effortCaps(CapReasoningOff, "low", CapReasoningEffortsRefused, "none"), EffortLow},
		{"everything refused is unset", EffortNone, effortCaps(CapReasoningEffortsRefused, "none,minimal,low,medium,high,xhigh,max"), EffortUnset},
	}
	for _, tc := range cases {
		ch := EffortFor(DialectOpenAI, false, "m", tc.caps, tc.e)
		if ch.Effective != tc.want {
			t.Errorf("%s: %+v, want %s", tc.name, ch, tc.want)
		}
		if ch.Effective != tc.e && (ch.Reason == "" || ch.Cause == "") {
			t.Errorf("%s: changed without a reason: %+v", tc.name, ch)
		}
	}
}

// TestEffortForClampOnlyLandsOnLevelsTheDialectExpresses: the Messages API has
// no minimal (it maps to low), so a refused low must not clamp DOWN to a level
// that maps back to low. openai expresses max, so with everything below max
// refused, max IS a landing place for xhigh's clamp.
func TestEffortForClampOnlyLandsOnLevelsTheDialectExpresses(t *testing.T) {
	ch := EffortFor(DialectOpenAI, false, "m", effortCaps(CapReasoningEffortsRefused, "minimal,low,medium,high,xhigh"), EffortXHigh)
	if ch.Effective != EffortMax {
		t.Errorf("%+v, want max", ch)
	}
	ch = EffortFor(DialectAnthropic, true, "claude-opus-4-6", effortCaps(CapReasoningEffortsRefused, "low"), EffortNone)
	if ch.Effective != EffortMedium {
		t.Errorf("messages none with low refused: %+v, want medium", ch)
	}
}

func TestEffortForReasonNamesTheFact(t *testing.T) {
	ch := EffortFor(DialectOpenAI, false, "m", effortCaps(CapReasoningThinkingOnly, "true"), EffortNone)
	if ch.Cause != CapReasoningThinkingOnly || !strings.HasPrefix(ch.Reason, "minimal (none not supported: reasoning.thinking_only=true") ||
		!strings.Contains(ch.Reason, "refused") || !strings.Contains(ch.Reason, "sent X") {
		t.Errorf("%+v", ch)
	}
}

// TestEffortForRefusedFieldIsUnsetWithReason: the endpoint refused
// reasoning_effort itself, so nothing is sendable in a dialect that uses it, and
// the caller is told why instead of getting silence.
func TestEffortForRefusedFieldIsUnsetWithReason(t *testing.T) {
	caps := effortCaps(CapSamplingPrefix+"reasoning_effort", "refused")
	ch := EffortFor(DialectOpenAI, false, "m", caps, EffortLow)
	if ch.Effective != EffortUnset || !strings.Contains(ch.Reason, "sampling.reasoning_effort") {
		t.Errorf("%+v", ch)
	}
	// llama.cpp never sends that field.
	if ch := EffortFor(DialectLlamaCpp, false, "m", caps, EffortLow); ch.Effective != EffortLow {
		t.Errorf("llama.cpp affected by a field it does not send: %+v", ch)
	}
}

func TestEffortSupportListsEveryLevelWithReasonWhenUnavailable(t *testing.T) {
	list := EffortSupport(DialectOpenAI, false, "m", effortCaps(CapReasoningThinkingOnly, "true"))
	if len(list) != len(Efforts) {
		t.Fatalf("len %d", len(list))
	}
	for i, ch := range list {
		if ch.Wanted != Efforts[i] {
			t.Errorf("row %d wanted %s", i, ch.Wanted)
		}
		if ch.Effective != ch.Wanted && ch.Reason == "" {
			t.Errorf("%s unavailable without a reason", ch.Wanted)
		}
	}
	if list[0].Effective != EffortMinimal {
		t.Errorf("none row: %+v", list[0])
	}
	c := openaiClient("http://x/v1")
	if c.Effective(EffortMax).Effective != EffortMax || len(c.EffortSupport()) != len(Efforts) {
		t.Error("client wrappers disagree with the pure function")
	}
}

// TestEffortTranslationIsReportedPerRequestWhenLevelChanges: a static change
// (max to high on llama.cpp) is reported too; an exact level is not.
func TestEffortTranslationIsReportedPerRequestWhenLevelChanges(t *testing.T) {
	c := llamaClient()
	_, trs := c.chatBody(t.Context(), nil, nil, &ChatOpts{Effort: EffortMax}, false)
	if len(trs) != 1 || trs[0].Key != "effort" || trs[0].Sent != "high" || trs[0].Wanted != "max" || trs[0].Reason == "" {
		t.Fatalf("%+v", trs)
	}
	if _, trs := c.chatBody(t.Context(), nil, nil, &ChatOpts{Effort: EffortHigh}, false); len(trs) != 0 {
		t.Fatalf("exact level reported: %+v", trs)
	}
}

// TestRawProbeIsNotShapedByTheRecord: the probe is how the record is made. A
// record that refuses none must not turn the probe's none into minimal.
func TestRawProbeIsNotShapedByTheRecord(t *testing.T) {
	c := openaiClient("http://x/v1")
	c.SetCapabilities(effortCaps(CapReasoningThinkingOnly, "true"))
	body, trs := c.chatBody(withRawProbe(t.Context()), nil, nil, &ChatOpts{Effort: EffortNone}, false)
	if body["reasoning_effort"] != "none" || len(trs) != 0 {
		t.Fatalf("%v %v", body, trs)
	}
}

func bodyOf(t *testing.T, c *Client, opts *ChatOpts) (map[string]any, []Translation) {
	t.Helper()
	body, trs := c.chatBody(t.Context(), nil, nil, opts, false)
	return body, trs
}

// TestEffortRefusalsBindOnlyDialectsThatSendAnEffortField: reasoning.efforts.refused
// and reasoning.off are facts about the reasoning_effort field. llama.cpp
// expresses a level as enable_thinking + a budget, which the endpoint never
// judged, so clamping there would drop a level nobody refused.
func TestEffortRefusalsBindOnlyDialectsThatSendAnEffortField(t *testing.T) {
	c := llamaClient()
	c.SetCapabilities(effortCaps(CapReasoningEffortsRefused, "high"))
	body, trs := bodyOf(t, c, &ChatOpts{Effort: EffortHigh})
	kw, _ := body["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != true || body["reasoning_budget_tokens"] != nil || len(trs) != 0 {
		t.Fatalf("high refused on llama: %v %v", body, trs)
	}
	c.SetCapabilities(effortCaps(CapReasoningOff, "low"))
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortNone})
	kw, _ = body["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != false || body["reasoning_budget_tokens"] != 0 || len(trs) != 0 {
		t.Fatalf("reasoning.off on llama: %v %v", body, trs)
	}
	// The model-level fact still binds: none is not possible on a thinking-only model.
	if ch := EffortFor(DialectLlamaCpp, false, "m", effortCaps(CapReasoningThinkingOnly, "true"), EffortNone); ch.Effective == EffortNone {
		t.Errorf("thinking_only ignored on llama: %+v", ch)
	}
	// An effort-field dialect is still bound.
	if ch := EffortFor(DialectOpenAI, false, "m", effortCaps(CapReasoningEffortsRefused, "high"), EffortHigh); ch.Effective != EffortMedium {
		t.Errorf("openai: %+v", ch)
	}
}

// TestAnthropicBudgetStaysBelowMaxTokens: Anthropic answers 400 when
// budget_tokens >= max_tokens.
func TestAnthropicBudgetStaysBelowMaxTokens(t *testing.T) {
	budget := func(body map[string]any) any {
		th, _ := body["thinking"].(map[string]any)
		return th["budget_tokens"]
	}
	c := claudeClient()
	c.SetReasoningDialect(DialectAnthropic)
	body, trs := bodyOf(t, c, &ChatOpts{Effort: EffortHigh, MaxTokens: 8192})
	if budget(body) != 8191 || len(trs) != 1 || !strings.Contains(trs[0].Reason, "8191") {
		t.Fatalf("high + MaxTokens 8192: %v %+v", body["thinking"], trs)
	}
	c.SetCapabilities(effortCaps(CapLimitsOutput, "64000"))
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortMax})
	if budget(body) != 63999 || len(trs) != 2 || trs[0].Key != "effort" {
		t.Fatalf("max + OutputLimit 64000: %v %+v", body["thinking"], trs)
	}
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortHigh, MaxTokens: 800})
	if body["thinking"] != nil || len(trs) != 1 || trs[0].Sent != "none" ||
		trs[0].Reason != "none (max_tokens 800 leaves no room for thinking)" {
		t.Fatalf("MaxTokens 800: %v %+v", body["thinking"], trs)
	}
	// Room to spare: untouched, nothing reported.
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortLow, MaxTokens: 8192})
	if budget(body) != 1024 || len(trs) != 0 {
		t.Fatalf("low + MaxTokens 8192: %v %+v", body["thinking"], trs)
	}
}

// TestExplicitBudgetIsACapNotAnOverride: the level's budget is the ceiling an
// explicit ReasoningBudgetTokens can only lower; with no Effort it is sent as
// it always was.
func TestExplicitBudgetIsACapNotAnOverride(t *testing.T) {
	ptr := func(n int) *int { return &n }
	c := llamaClient()
	// A WithAutoReasoningBudget-sized budget must not raise low (1024).
	body, trs := bodyOf(t, c, &ChatOpts{Effort: EffortLow, ReasoningBudgetTokens: ptr(60000)})
	if body["reasoning_budget_tokens"] != 1024 || len(trs) != 0 {
		t.Fatalf("low + 60000: %v %+v", body["reasoning_budget_tokens"], trs)
	}
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortMedium, ReasoningBudgetTokens: ptr(500)})
	if body["reasoning_budget_tokens"] != 500 || len(trs) != 1 || !strings.Contains(trs[0].Reason, "capped to 500") {
		t.Fatalf("medium + 500: %v %+v", body["reasoning_budget_tokens"], trs)
	}
	// No Effort: exactly the explicit budget, no enable_thinking, no report.
	body, trs = bodyOf(t, c, &ChatOpts{ReasoningBudgetTokens: ptr(2500)})
	if body["reasoning_budget_tokens"] != 2500 || body["chat_template_kwargs"] != nil || len(trs) != 0 {
		t.Fatalf("unset + 2500: %v %+v", body, trs)
	}
	// A level with no budget (llama high) takes the explicit budget as before.
	body, _ = bodyOf(t, c, &ChatOpts{Effort: EffortHigh, ReasoningBudgetTokens: ptr(3000)})
	if body["reasoning_budget_tokens"] != 3000 {
		t.Fatalf("high + 3000: %v", body["reasoning_budget_tokens"])
	}
	// Explicit 0 caps at zero, and says so.
	body, trs = bodyOf(t, c, &ChatOpts{Effort: EffortLow, ReasoningBudgetTokens: ptr(0)})
	if body["reasoning_budget_tokens"] != 0 || len(trs) != 1 {
		t.Fatalf("low + 0: %v %+v", body["reasoning_budget_tokens"], trs)
	}
	a := claudeClient()
	a.SetReasoningDialect(DialectAnthropic)
	body, trs = bodyOf(t, a, &ChatOpts{Effort: EffortHigh, ReasoningBudgetTokens: ptr(0)})
	if body["thinking"] != nil || len(trs) != 1 || trs[0].Sent != "none" {
		t.Fatalf("anthropic high + 0: %v %+v", body["thinking"], trs)
	}
	body, _ = bodyOf(t, a, &ChatOpts{Effort: EffortHigh, ReasoningBudgetTokens: ptr(2048)})
	if th, _ := body["thinking"].(map[string]any); th["budget_tokens"] != 2048 {
		t.Fatalf("anthropic high + 2048: %v", body["thinking"])
	}
}

// TestChatBodyTranslatesByTheSameSnapshotItResolvedTheLevelBy: translateWith
// takes the record chatBody already holds, so a fact learned after that
// snapshot cannot make the level pass and the body pass disagree.
func TestChatBodyTranslatesByTheSameSnapshotItResolvedTheLevelBy(t *testing.T) {
	c := openaiClient("http://x/v1")
	body := map[string]any{"reasoning_effort": "high"}
	snap := effortCaps(CapReasoningEffortsRefused, "high")
	c.SetCapabilities(Capabilities{}) // the live record is empty
	if trs := c.translateWith(body, nil, snap); len(trs) != 1 || body["reasoning_effort"] != "medium" {
		t.Fatalf("translateWith ignored the snapshot: %v %+v", body, trs)
	}
}
