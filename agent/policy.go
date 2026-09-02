package agent

// policy.go — a shaping policy from the one number a host actually knows.
//
// Before this, every host assembled a ShaperPolicy field by field, and the cost
// of that showed up as the same bug in more than one of them: a host set a
// budget, attached a Shaper, and still blew its window, because it had not set
// LODTruncateAboveChars and a 14,000-character file read stayed in the context
// verbatim for the rest of the session.
//
// dun set it and never hit this. aw4 did not and hit it twice in one day, then
// spent three commits rediscovering three quarters of dun's four lines. A policy
// that has to be assembled correctly by each host is a policy each host gets
// wrong once, and the second host learns nothing from the first.
//
// So: a host declares its WINDOW, which is the number it can actually look up,
// and gets a policy that works.

// DefaultOutputReservePct is how much of the window is held back for the
// model's own response and for compaction's summarizer call. Shaping to the
// full window leaves no room to answer.
const DefaultOutputReservePct = 12

// DefaultLODTruncateAboveChars is where level-of-detail truncation starts on
// older entries.
//
// 4000 characters is roughly a screen of source. Below it, entries are small
// enough that keeping them whole is not what fills a window; above it, they are
// file reads and command output, which is exactly what does. Truncating these
// is cheaper than compaction — no summarizer call — and it targets the growth
// rather than folding conversation that was never the problem.
const DefaultLODTruncateAboveChars = 4000

// DefaultPreserveLastMessages and DefaultPreserveLastToolCalls are the tail a
// fold must not touch. A session whose most recent exchanges are summarized
// loses the tool result it was about to act on, which is the one part of the
// history it cannot reconstruct.
const (
	DefaultPreserveLastMessages  = 8
	DefaultPreserveLastToolCalls = 4
)

// PolicyFor builds a shaping policy for a model's context window.
//
// `window` is the model's real context size in tokens — what the provider will
// refuse above, not a budget somebody already reduced. The reserve is taken off
// here, so a caller passing an already-reduced budget reserves twice and merely
// folds earlier than it needed to; passing the window is both correct and the
// easier thing to look up.
//
// A window of zero returns a zero policy, which the Shaper reads as UNBUDGETED —
// render everything, fold nothing. That is deliberate and is not the same as a
// budget of nothing: a host that has not been told its window must not have its
// history destroyed on that basis.
func PolicyFor(window int) ShaperPolicy {
	if window <= 0 {
		return ShaperPolicy{}
	}
	return ShaperPolicy{
		BudgetTokens:          Budget(window, DefaultOutputReservePct),
		LODTruncateAboveChars: DefaultLODTruncateAboveChars,
		PreserveLastMessages:  DefaultPreserveLastMessages,
		PreserveLastToolCalls: DefaultPreserveLastToolCalls,
	}
}
