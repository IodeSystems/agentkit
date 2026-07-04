package agent

// TokenEstimator estimates the token count of a string for budget
// calculations. The default is a deliberately conservative chars/4 heuristic
// that overcounts most real tokenizers; the context shaper relies on this
// conservatism for safety. Real tokenizers plug in behind the same interface.
type TokenEstimator interface {
	Estimate(s string) int
}

// CharsByFour is the v1 default: byte length / 4, rounded up. Cheap,
// dependency-free, conservative for English+code.
type CharsByFour struct{}

func (CharsByFour) Estimate(s string) int {
	if len(s) == 0 {
		return 0
	}
	return (len(s) + 3) / 4
}

// Default returns the package-level default estimator.
func Default() TokenEstimator { return CharsByFour{} }

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
