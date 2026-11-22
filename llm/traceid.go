package llm

import "context"

// traceKey is the ctx key for WithTraceID.
type traceKey struct{}

// WithTraceID returns a ctx whose calls send id as X-Trace-Id. It exists for
// the calls that have no ChatOpts: DiscoverContext, Embed, Transcribe and the
// capability, tokenize and budget probes. A host sets it once on the ctx of a
// thread and every call on that ctx joins the thread in the server log.
//
// Client.do adds it to any request that has no X-Trace-Id yet. For a chat call
// ChatOpts.TraceID wins; the ctx id is used only when that is empty. An empty id returns ctx unchanged.
func WithTraceID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}
	return context.WithValue(ctx, traceKey{}, id)
}

// traceIDFrom is the id WithTraceID set, or "".
func traceIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(traceKey{}).(string)
	return id
}
