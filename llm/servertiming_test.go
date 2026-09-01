package llm

import (
	"net/http"
	"testing"
)

func hdr(kv ...string) http.Header {
	h := http.Header{}
	for i := 0; i+1 < len(kv); i += 2 {
		h.Set(kv[i], kv[i+1])
	}
	return h
}

func TestServerTimingParsesTheBreakdown(t *testing.T) {
	got := serverTimingFrom(hdr(
		HeaderQueuedMS, "9000",
		HeaderLoadMS, "1500",
		HeaderUpstreamTTFBMS, "220",
	))
	if !got.Reported {
		t.Fatal("Reported must be true when the headers are present")
	}
	if got.QueuedMS != 9000 || got.LoadMS != 1500 || got.UpstreamTTFBMS != 220 {
		t.Errorf("got %+v", got)
	}
	// TTFB is execution and must NOT be counted as overhead.
	if got.OverheadMS() != 10500 {
		t.Errorf("OverheadMS = %d, want 10500 (queued+load, not ttfb)", got.OverheadMS())
	}
}

// The distinction the whole type exists for: an endpoint that says nothing is
// not an endpoint reporting zero overhead.
func TestServerTimingAbsentIsNotZero(t *testing.T) {
	got := serverTimingFrom(hdr())
	if got.Reported {
		t.Error("Reported must be false when no header was sent")
	}
	if got.OverheadMS() != 0 {
		t.Errorf("OverheadMS = %d, want 0", got.OverheadMS())
	}
}

// A garbage value is not a measurement. Recording it as 0 would read as
// "did not queue", which is a claim the server never made.
func TestServerTimingIgnoresMalformedValues(t *testing.T) {
	got := serverTimingFrom(hdr(HeaderQueuedMS, "not-a-number", HeaderLoadMS, "-5"))
	if got.Reported {
		t.Error("a malformed value must not count as a report")
	}
	if got.QueuedMS != 0 || got.LoadMS != 0 {
		t.Errorf("got %+v, want zeroes", got)
	}
}

func TestWorkMSSubtractsOnlyReportedOverhead(t *testing.T) {
	// Reported: subtract it.
	u := Usage{LatencyMS: 12000, Server: ServerTiming{QueuedMS: 9000, LoadMS: 1000, Reported: true}}
	if got := u.WorkMS(); got != 2000 {
		t.Errorf("WorkMS = %d, want 2000", got)
	}
	// Not reported: the whole latency is work, because discounting an
	// unmeasured overhead is a guess dressed as a measurement.
	u = Usage{LatencyMS: 12000}
	if got := u.WorkMS(); got != 12000 {
		t.Errorf("WorkMS = %d, want the full latency when nothing was reported", got)
	}
	// Skew must not produce negative work.
	u = Usage{LatencyMS: 100, Server: ServerTiming{QueuedMS: 9000, Reported: true}}
	if got := u.WorkMS(); got != 0 {
		t.Errorf("WorkMS = %d, want 0 — never negative", got)
	}
}
