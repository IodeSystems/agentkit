package llm

import (
	"reflect"
	"testing"
	"time"
)

func capClient() *Client { return NewClient("https://api.example.com/v1/", "k", "glm") }

func fact(v string, s Source) Fact { return Fact{Value: v, Source: s, At: time.Unix(1, 0)} }

// Every (new, old) source pair: evidence replaces declared and evidence,
// declared replaces only declared, nothing replaces host.
func TestCapabilityPrecedenceMatrix(t *testing.T) {
	all := []Source{SourceHost, SourceRefused, SourceObserved, SourceProbed, SourceDeclared}
	for _, old := range all {
		for _, nw := range []Source{SourceRefused, SourceObserved, SourceProbed, SourceDeclared} {
			want := false
			switch {
			case old == SourceHost:
			case nw.isEvidence():
				want = true
			case nw == SourceDeclared:
				want = old == SourceDeclared
			}
			c := capClient()
			c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapLimitsContext: fact("100", old)}})
			c.learn(CapLimitsContext, fact("200", nw))
			got := c.Capabilities().Facts[CapLimitsContext].Value == "200"
			if got != want {
				t.Errorf("new %s over old %s: replaced=%v, want %v", nw, old, got, want)
			}
		}
	}
}

func TestCapabilityHostPinSurvivesLearning(t *testing.T) {
	c := capClient()
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningOff: fact("low", SourceHost)}})
	n := 0
	c.OnLearn = func(Learned) { n++ }
	c.learn(CapReasoningOff, fact("none", SourceRefused))
	c.learn(CapReasoningOff, fact("none", SourceHost)) // learning cannot pin either
	if v, _ := c.Capabilities().OffEffort(); v != "low" || n != 0 {
		t.Fatalf("host pin lost: off=%q callbacks=%d", v, n)
	}
}

func TestCapabilitySeededRefusedBeatsFreshDeclared(t *testing.T) {
	c := capClient()
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapLimitsOutput: fact("4096", SourceRefused)}})
	c.learn(CapLimitsOutput, fact("8192", SourceDeclared))
	if v, _ := c.Capabilities().OutputLimit(); v != 4096 {
		t.Fatalf("stored refused fact lost to declared: %d", v)
	}
}

func TestCapabilityOnLearnFiresOnceOnChange(t *testing.T) {
	c := capClient()
	var got []Learned
	c.OnLearn = func(l Learned) { got = append(got, l) }
	c.learn(CapStreamUsage, fact("true", SourceObserved))
	c.learn(CapStreamUsage, fact("true", SourceObserved)) // identical re-learn
	if len(got) != 1 || got[0].Previous != nil || got[0].Key != CapStreamUsage ||
		got[0].Endpoint != "https://api.example.com/v1" || got[0].Model != "glm" {
		t.Fatalf("first learn: %+v", got)
	}
	c.learn(CapStreamUsage, fact("false", SourceObserved)) // value change
	c.learn(CapStreamUsage, fact("false", SourceProbed))   // source change
	if len(got) != 3 || got[1].Previous == nil || got[1].Previous.Value != "true" ||
		got[2].Previous == nil || got[2].Previous.Source != SourceObserved {
		t.Fatalf("changes: %+v", got)
	}
}

func TestCapabilityQuietRefreshUpdatesTimestamp(t *testing.T) {
	c := capClient()
	n := 0
	c.OnLearn = func(Learned) { n++ }
	c.learn(CapStreamUsage, fact("true", SourceObserved))
	later := Fact{Value: "true", Source: SourceObserved, Evidence: "again", At: time.Unix(99, 0)}
	c.learn(CapStreamUsage, later)
	if n != 1 || c.Capabilities().Facts[CapStreamUsage].Evidence != "again" {
		t.Fatalf("callbacks=%d fact=%+v", n, c.Capabilities().Facts[CapStreamUsage])
	}
}

// A callback that snapshots the record would deadlock if learn held the lock.
func TestCapabilityOnLearnRunsOutsideLock(t *testing.T) {
	c := capClient()
	done := make(chan Capabilities, 1)
	c.OnLearn = func(Learned) { done <- c.Capabilities() }
	go c.learn(CapLimitsContext, fact("32000", SourceObserved))
	select {
	case snap := <-done:
		if v, ok := snap.ContextWindow(); !ok || v != 32000 {
			t.Fatalf("snapshot in callback: %v %v", v, ok)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("deadlock: OnLearn ran under the record lock")
	}
}

func TestCapabilitySnapshotIsACopy(t *testing.T) {
	c := capClient()
	c.learn(CapLimitsContext, fact("1000", SourceObserved))
	snap := c.Capabilities()
	snap.Facts[CapLimitsContext] = fact("1", SourceHost)
	snap.Facts["x"] = fact("y", SourceHost)
	again := c.Capabilities()
	if again.Facts[CapLimitsContext].Value != "1000" || len(again.Facts) != 1 {
		t.Fatalf("snapshot aliased the record: %+v", again.Facts)
	}
	// Seeding copies too: the host's map is not kept.
	seed := map[string]Fact{CapLimitsContext: fact("5", SourceHost)}
	c.SetCapabilities(Capabilities{Facts: seed})
	seed[CapLimitsContext] = fact("6", SourceHost)
	if v, _ := c.Capabilities().ContextWindow(); v != 5 {
		t.Fatalf("seed aliased: %d", v)
	}
}

func TestCapabilityEndpointNormalization(t *testing.T) {
	for in, want := range map[string]string{
		"https://API.Example.com/v1/":     "https://api.example.com/v1",
		"http://localhost:8080":           "http://localhost:8080",
		"http://localhost:8080//":         "http://localhost:8080",
		"https://u:p@h.io/v1/?key=s#frag": "https://h.io/v1",
		"  https://h.io/a/b/ ":            "https://h.io/a/b",
		"not a url/":                      "not a url",
	} {
		if got := normalizeEndpoint(in); got != want {
			t.Errorf("normalizeEndpoint(%q) = %q, want %q", in, got, want)
		}
	}
	c := NewClient("https://H.io/v1/", "", "m")
	if cp := c.Capabilities(); cp.Endpoint != "https://h.io/v1" || cp.Model != "m" {
		t.Fatalf("record identity: %+v", cp)
	}
}

func TestCapabilityGettersTreatBadValuesAsUnknown(t *testing.T) {
	c := Capabilities{Facts: map[string]Fact{
		CapLimitsContext:           fact("lots", SourceObserved),
		CapLimitsOutput:            fact("-5", SourceObserved),
		CapReasoningOff:            fact("  ", SourceObserved),
		CapReasoningEffortsRefused: fact(" , ,", SourceObserved),
		CapReasoningThinkingOnly:   fact("maybe", SourceObserved),
		CapToolsChoiceForced:       fact("", SourceObserved),
		CapStreamUsage:             fact("2", SourceObserved),
		CapStreamReasoningField:    fact("", SourceObserved),
	}}
	if _, ok := c.ContextWindow(); ok {
		t.Error("ContextWindow known for non-number")
	}
	if _, ok := c.OutputLimit(); ok {
		t.Error("OutputLimit known for negative")
	}
	if _, ok := c.OffEffort(); ok {
		t.Error("OffEffort known for blank")
	}
	if _, ok := c.RefusedEfforts(); ok {
		t.Error("RefusedEfforts known for empty list")
	}
	if _, ok := c.AcceptedEfforts(); ok {
		t.Error("AcceptedEfforts known for missing key")
	}
	if _, ok := c.ThinkingOnly(); ok {
		t.Error("ThinkingOnly known for non-bool")
	}
	if _, ok := c.ToolChoiceHonored(); ok {
		t.Error("ToolChoiceHonored known for empty")
	}
	if _, ok := c.StreamUsage(); ok {
		t.Error("StreamUsage known for 2")
	}
	if _, ok := c.ReasoningField(); ok {
		t.Error("ReasoningField known for empty")
	}
}

func TestCapabilityGettersParseGoodValues(t *testing.T) {
	c := Capabilities{Facts: map[string]Fact{
		CapLimitsContext:            fact("131072", SourceDeclared),
		CapLimitsOutput:             fact("8192", SourceRefused),
		CapReasoningOff:             fact("none", SourceRefused),
		CapReasoningEffortsRefused:  fact("none, minimal", SourceRefused),
		CapReasoningEffortsAccepted: fact("low,high", SourceObserved),
		CapReasoningThinkingOnly:    fact("true", SourceRefused),
		CapToolsChoiceForced:        fact("false", SourceProbed),
		CapStreamUsage:              fact("true", SourceObserved),
		CapStreamReasoningField:     fact("reasoning_content", SourceObserved),
	}}
	if v, ok := c.ContextWindow(); !ok || v != 131072 {
		t.Error("ContextWindow")
	}
	if v, ok := c.OutputLimit(); !ok || v != 8192 {
		t.Error("OutputLimit")
	}
	if v, ok := c.OffEffort(); !ok || v != "none" {
		t.Error("OffEffort")
	}
	if v, ok := c.RefusedEfforts(); !ok || !reflect.DeepEqual(v, []string{"none", "minimal"}) {
		t.Error("RefusedEfforts", v)
	}
	if v, ok := c.AcceptedEfforts(); !ok || !reflect.DeepEqual(v, []string{"low", "high"}) {
		t.Error("AcceptedEfforts", v)
	}
	if v, ok := c.ThinkingOnly(); !ok || !v {
		t.Error("ThinkingOnly")
	}
	if v, ok := c.ToolChoiceHonored(); !ok || v {
		t.Error("ToolChoiceHonored")
	}
	if v, ok := c.StreamUsage(); !ok || !v {
		t.Error("StreamUsage")
	}
	if v, ok := c.ReasoningField(); !ok || v != "reasoning_content" {
		t.Error("ReasoningField")
	}
}
