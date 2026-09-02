package agent_test

import (
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/agent"
)

/*
A policy built from a window has the field that hosts forget.

Every host assembled a ShaperPolicy by hand, and more than one of them shipped
the same bug: a budget set, a Shaper attached, and no LOD threshold — so a
14,000-character file read stayed in the context verbatim for the rest of the
session. A policy each host must assemble correctly is one each host gets wrong
once, and the second learns nothing from the first.
*/
func TestPolicyForFillsInWhatHostsForget(t *testing.T) {
	p := agent.PolicyFor(188_160)

	if p.LODTruncateAboveChars <= 0 {
		t.Error("no LOD threshold: oversized entries stay whole, which is the bug this exists to stop")
	}
	if p.PreserveLastMessages <= 0 || p.PreserveLastToolCalls <= 0 {
		t.Error("no preserved tail: a fold can eat the result the model was about to act on")
	}

	// The reserve is real: shaping to the whole window leaves no room to answer.
	if p.BudgetTokens >= 188_160 {
		t.Errorf("budget %d is the whole window; nothing is left for the response or the fold",
			p.BudgetTokens)
	}
	// And not so aggressive that most of the window is unusable.
	if p.BudgetTokens < 188_160*3/4 {
		t.Errorf("budget %d reserves more than a quarter of the window", p.BudgetTokens)
	}

	// Zero means UNBUDGETED, which is not a budget of nothing. A host that has
	// not been told its window must not have its history destroyed on that basis.
	if z := agent.PolicyFor(0); z.BudgetTokens != 0 || z.LODTruncateAboveChars != 0 {
		t.Errorf("PolicyFor(0) = %+v, want a zero policy the Shaper reads as unbudgeted", z)
	}
	if n := agent.PolicyFor(-1); n.BudgetTokens != 0 {
		t.Errorf("a negative window produced a budget of %d", n.BudgetTokens)
	}
}

/*
The default estimator errs HIGH, because the two errors are not comparable.

Over-estimating folds early and costs a summarization call. Under-estimating lets
a session run past its window believing it is under budget, and the provider
refuses mid-turn. Measured on a real agent session: 604,565 characters counted as
173,648 tokens — 3.48 chars per token — where chars/4 said 151,141 and nothing
folded.
*/
func TestTheDefaultEstimatorErrsHigh(t *testing.T) {
	// The measured ratio on real agent traffic.
	const measured = 3.48

	def := agent.Default()
	for _, n := range []int{1_000, 604_565} {
		s := strings.Repeat("x", n)
		actual := float64(n) / measured
		if got := float64(def.Estimate(s)); got < actual {
			t.Errorf("default estimated %d chars at %.0f tokens; a provider counts ~%.0f — "+
				"under the truth is the direction that kills sessions", n, got, actual)
		}
		// chars/4 fails that, which is why the default changed. If this ever
		// stops being true the file has lost its reason to exist.
		if got := float64(agent.CharsByFour{}.Estimate(s)); got >= actual {
			t.Errorf("CharsByFour estimated %.0f for %d chars, at or above the real ~%.0f; "+
				"then it was safe all along", got, n, actual)
		}
	}
	if def.Estimate("") != 0 {
		t.Error("empty content costs tokens")
	}
	// Not so high that it folds history nobody needed to lose.
	if got := float64(agent.Default().Estimate(strings.Repeat("x", 604_565))); got > 604_565/measured*1.6 {
		t.Errorf("default overestimates by more than 60%%: %.0f", got)
	}
}
