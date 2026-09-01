package llm

import (
	"net/http"
	"strconv"
)

// Timing headers a corrallm-style proxy stamps on every response it serves.
//
// They are the answer to a question a client cannot answer for itself: of the
// wall clock this call took, how much was the model working, and how much was
// the proxy holding the request?
//
// The 429 path is already visible — a rejected caller gets Retry-After and the
// queue's shape (see Backpressure). A caller that queues nine seconds and is
// then SERVED sees one slow response, indistinguishable from a slow generation.
// Anything measuring what work cost — a benchmark, or a host billing elapsed
// time against a budget — charges that queue to the model, so the same work
// prices differently depending on how busy the box was.
const (
	// HeaderQueuedMS — time blocked in admission control, summed across every
	// backend the request queued on before one admitted it.
	HeaderQueuedMS = "X-Corrallm-Queued-Ms"
	// HeaderLoadMS — time waiting for a backend to become resident: a cold
	// spawn and its health wait. Zero when the model was already up.
	HeaderLoadMS = "X-Corrallm-Load-Ms"
	// HeaderUpstreamTTFBMS — time from dispatch upstream to its response
	// headers. Unlike the other two this IS execution: the model's prompt-eval
	// latency, separable from generation.
	HeaderUpstreamTTFBMS = "X-Corrallm-Upstream-Ttfb-Ms"
)

// ServerTiming is what the proxy said about its own overhead on one round-trip.
//
// Reported distinguishes "the proxy said zero" from "nothing told us", which are
// different facts and must not collapse: a plain OpenAI endpoint reports
// nothing, and treating that as a confident zero would let a host subtract
// overhead it never measured and call the remainder work.
type ServerTiming struct {
	// QueuedMS is time the request spent waiting for a slot. NOT execution.
	QueuedMS int64
	// LoadMS is time spent bringing a backend up. NOT execution.
	LoadMS int64
	// UpstreamTTFBMS is the model's prompt-eval latency. This IS execution.
	UpstreamTTFBMS int64
	// Reported is true when the response carried the headers at all.
	Reported bool
}

// OverheadMS is the part of a round-trip the proxy admits was not the model:
// queueing plus a cold load. Zero when nothing was reported, which is the safe
// direction — an unreported overhead is counted as work rather than silently
// discounted.
func (t ServerTiming) OverheadMS() int64 { return t.QueuedMS + t.LoadMS }

// serverTimingFrom reads the headers, if the server sent them. A single header
// is enough to count as reported: a proxy that stamps one stamps all three, and
// requiring all three would turn a future added field into a silent regression.
func serverTimingFrom(h http.Header) ServerTiming {
	if h == nil {
		return ServerTiming{}
	}
	var t ServerTiming
	for _, f := range []struct {
		name string
		dst  *int64
	}{
		{HeaderQueuedMS, &t.QueuedMS},
		{HeaderLoadMS, &t.LoadMS},
		{HeaderUpstreamTTFBMS, &t.UpstreamTTFBMS},
	} {
		v := h.Get(f.name)
		if v == "" {
			continue
		}
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n < 0 {
			// A malformed value is not a measurement. Skip it rather than
			// recording a zero that would read as "did not queue".
			continue
		}
		*f.dst = n
		t.Reported = true
	}
	return t
}
