package agent

// TokenEstimator estimates the token count of a string for budget
// calculations. Real tokenizers plug in behind the same interface.
//
// ## The error must point UP
//
// An estimator is wrong in one of two directions and they are not comparable:
//
//	too HIGH  the Shaper folds earlier than it had to. Costs a summarization
//	          call and some history that could have stayed.
//	too LOW   the Shaper believes it is under budget while the real context
//	          passes the window. The provider refuses the request and the
//	          session dies mid-turn, having done the work and lost it.
//
// So an estimator that is occasionally expensive is correct and one that is
// occasionally optimistic is not, and every default here errs high on purpose.
type TokenEstimator interface {
	Estimate(s string) int
}

// CharsByFour is the v1 default: byte length / 4, rounded up.
//
// KEPT, and no longer the default. It was documented here as "conservative for
// English+code" and for prose it is. For AGENT traffic it is not:
//
//	measured 2026-09-02, a real agent session against a 188k window
//	604,565 rendered characters → 173,648 tokens counted by the provider
//	= 3.48 chars per token, so chars/4 estimated 151,141 — 13% under
//
// That gap sat below the Shaper's restructure threshold, nothing folded, and the
// next request was refused. Agent context is not prose: hex identifiers tokenize
// near one token per two characters, and source, JSON tool payloads, stack
// traces and diffs are all denser than English.
//
// Use it when the traffic really is prose and the extra headroom matters.
type CharsByFour struct{}

func (CharsByFour) Estimate(s string) int {
	if len(s) == 0 {
		return 0
	}
	return (len(s) + 3) / 4
}

// CharsByThree is the default: byte length / 3, rounded up.
//
// Three rather than the measured 3.48 because one workload on one tokenizer is
// not a constant, and the margin is the cheap side of the asymmetry above.
type CharsByThree struct{}

func (CharsByThree) Estimate(s string) int {
	if len(s) == 0 {
		return 0
	}
	return (len(s) + 2) / 3
}

// Default returns the package-level default estimator.
//
// Changed from CharsByFour on 2026-09-02. A host that wants the old behaviour
// sets Estimate: CharsByFour{} explicitly, which is the right way round: the
// default should be the safe one and the optimistic one should be a choice.
func Default() TokenEstimator { return CharsByThree{} }

// Budget returns the active-context budget for a model: the context-window
// size scaled down by the reserve percentage. reserve_pct=25 on a 254 000-
// token model yields 190 500, leaving room for the model's own response and
// any compaction work.
func Budget(contextTokens, reservePct int) int {
	if contextTokens <= 0 {
		return 0
	}
	if reservePct < 0 {
		reservePct = 0
	}
	if reservePct >= 100 {
		reservePct = 99
	}
	return contextTokens * (100 - reservePct) / 100
}
