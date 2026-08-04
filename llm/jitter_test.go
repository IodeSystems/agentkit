package llm

// Jitter tests. The property that matters is not "the sleep is random" but the
// two-sided bound around it: a draw that lands BELOW the server's Retry-After
// buys a guaranteed second 429 (the proxy told us when a slot frees; arriving
// early cannot help), and a draw with no spread at all leaves concurrent callers
// waking in lockstep — the defect jitter exists to remove. Cover both edges,
// plus the pass-through cases and the schedule's own integrity.

import (
	"testing"
	"time"
)

// retryJitterSet swaps the package's jitter fraction and returns the prior value
// for restore. Test-only — production never mutates.
func retryJitterSet(f float64) float64 {
	prev := retryJitterFraction
	retryJitterFraction = f
	return prev
}

// TestJitterNeverSleepsLessThanAsked — the failure this guards is a jittered
// sleep landing before the server's Retry-After, which turns the proxy's one
// piece of real knowledge into a wasted round-trip carrying a full payload.
// Additive-only jitter means the floor is the input, every draw.
func TestJitterNeverSleepsLessThanAsked(t *testing.T) {
	const d = 10 * time.Second
	for range 10000 {
		got := jittered(d)
		if got < d {
			t.Fatalf("jittered(%v) = %v; want ≥ %v — never retry before the server asked", d, got, d)
		}
	}
}

// TestJitterStaysWithinItsFraction — the other edge: unbounded slack would let a
// 30s ceiling become an arbitrary wait and quietly eat the retry budget. The
// ceiling is d*(1+fraction), exclusive.
func TestJitterStaysWithinItsFraction(t *testing.T) {
	const d = 10 * time.Second
	max := d + time.Duration(retryJitterFraction*float64(d))
	for range 10000 {
		got := jittered(d)
		if got > max {
			t.Fatalf("jittered(%v) = %v; want ≤ %v (fraction %.2f)", d, got, max, retryJitterFraction)
		}
	}
}

// TestJitterSpreadsCallersOffOneSchedule — the herd. N callers handed the
// IDENTICAL Retry-After must not resolve to the identical sleep, or they
// re-POST together into the box that just rejected them. One repeated value
// across a large sample means jitter is not wired in.
func TestJitterSpreadsCallersOffOneSchedule(t *testing.T) {
	const d = time.Second
	seen := map[time.Duration]int{}
	for range 1000 {
		seen[jittered(d)]++
	}
	if len(seen) < 900 {
		t.Errorf("1000 callers on one Retry-After produced %d distinct sleeps; want ~1000 — they are still in lockstep", len(seen))
	}
	if seen[d] > 1 {
		t.Errorf("%d callers drew the bare schedule value %v; jitter is not being applied", seen[d], d)
	}
}

// TestJitterDisabledIsExact — zeroing the fraction must restore the exact
// schedule, so a test (or an operator) can get deterministic timings back.
func TestJitterDisabledIsExact(t *testing.T) {
	defer retryJitterSet(retryJitterSet(0))
	for _, d := range []time.Duration{time.Millisecond, time.Second, 30 * time.Second} {
		if got := jittered(d); got != d {
			t.Errorf("jittered(%v) with jitter off = %v; want exactly %v", d, got, d)
		}
	}
}

// TestJitterPassesThroughNonPositive — a zero sleep means "retry now" and a
// negative one is nonsense; neither has anything to spread, and multiplying a
// negative by the fraction would move the retry the wrong way.
func TestJitterPassesThroughNonPositive(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		if got := jittered(d); got != d {
			t.Errorf("jittered(%v) = %v; want it untouched", d, got)
		}
	}
}

// TestJitterDoesNotCompoundAcrossTheSchedule — jitter is drawn per sleep from
// the clean backoff, never folded back into it. If the loop jittered the
// schedule variable itself the slack would multiply with every doubling, and
// six attempts would overshoot the ceiling badly. This asserts the invariant
// the loop relies on: the schedule doubles, the SLEEP is what gets slack.
func TestJitterDoesNotCompoundAcrossTheSchedule(t *testing.T) {
	saveInit, saveMax := retryInitialBackoffSet(time.Second, 30*time.Second)
	defer retryInitialBackoffSet(saveInit, saveMax)

	backoff := retryInitialBackoff
	want := []time.Duration{1, 2, 4, 8, 16, 30, 30}
	for i, w := range want {
		wantD := w * time.Second
		if backoff != wantD {
			t.Fatalf("attempt %d: schedule = %v; want %v — jitter leaked into the doubling", i+1, backoff, wantD)
		}
		sleep := jittered(backoff)
		if sleep < backoff {
			t.Fatalf("attempt %d: sleep %v < schedule %v", i+1, sleep, backoff)
		}
		backoff *= 2
		if backoff > retryMaxBackoff {
			backoff = retryMaxBackoff
		}
	}
}
