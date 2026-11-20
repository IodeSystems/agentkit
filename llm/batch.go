package llm

// Batch token pricing.
//
// A batch completion is submitted to a provider's batch endpoint (Claude:
// /v1/messages/batches) and priced at a fraction of the standard per-token
// rate. The fraction is a property of the DIALECT, not of the call: the
// provider publishes one batch rate per model family, and the shim owns the
// absolute standard cost. agentkit owns the relationship — how much cheaper
// batch is relative to real-time — and nothing else.
//
// The absolute standard cost is the shim's job: it knows the model's price
// list. This package knows only the ratio, so a caller can compute the batch
// cost as standard_cost × BatchRate(dialect) without importing a price list.

// BatchRate returns the batch token price as a fraction of the standard
// per-token rate for the given dialect.
//
// Anthropic (Claude family): 0.5 — Claude's Batch API prices tokens at 50% of
// the real-time rate. This is the only provider with a published, fixed batch
// discount; the constant is 0.5, not a lookup, because Claude does not vary
// it by model size.
//
// All other dialects: 1.0 — no batch discount is published for them. A caller
// that batch-submits to an endpoint without a known batch price treats the
// cost as the standard rate, which is the safe direction: over-reporting a
// batch cost as real-time is a smaller error than under-reporting it as free.
func BatchRate(dialect string) float64 {
	switch dialect {
	case DialectAnthropic:
		return 0.5
	default:
		return 1.0
	}
}

// Batched reports whether this completion came from a batch endpoint.
//
// A batch Usage has Batch=true: the call was submitted to the provider's
// batch API and is priced at BatchRate(dialect) × the standard per-token rate.
// A real-time Usage has Batch=false (the zero value): the call went through
// the standard chat path.
func (u Usage) Batched() bool { return u.Batch }
