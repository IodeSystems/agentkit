package agent

import "context"

// Tracer optionally captures spans for the turn loop, chat round-trips, and
// context builds. A host adapts its own tracing (e.g. autowork3's internal/
// trace) onto this. Nil Tracer = no tracing, zero overhead.
type Tracer interface {
	// Start opens a span named `name` and returns it plus a child context
	// carrying the span, so nested Starts chain into a tree.
	Start(ctx context.Context, name string) (Span, context.Context)
}

// Span is one open span. Set attaches an attribute (chainable). Exactly one
// of End / EndError closes it.
type Span interface {
	Set(key string, value any) Span
	End()
	EndError(err error)
}

// startSpan is the nil-safe entry point the engine uses.
func startSpan(t Tracer, ctx context.Context, name string) (Span, context.Context) {
	if t == nil {
		return noopSpan{}, ctx
	}
	return t.Start(ctx, name)
}

type noopSpan struct{}

func (n noopSpan) Set(string, any) Span { return n }
func (noopSpan) End()                   {}
func (noopSpan) EndError(error)         {}
