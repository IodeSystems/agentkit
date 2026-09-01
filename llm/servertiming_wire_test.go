package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The whole point, end to end: corrallm stamps its overhead on a SERVED
// response, and the client must carry it as far as Usage. Before this, a
// request that queued nine seconds and was then served looked exactly like a
// slow generation.
func TestServerTimingReachesUsageOverTheWire(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(HeaderQueuedMS, "9000")
		w.Header().Set(HeaderLoadMS, "1200")
		w.Header().Set(HeaderUpstreamTTFBMS, "180")
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var usage *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if !usage.Server.Reported {
		t.Fatal("Server.Reported false — the headers were dropped between response and Usage")
	}
	if usage.Server.QueuedMS != 9000 || usage.Server.LoadMS != 1200 {
		t.Errorf("Server = %+v, want queued 9000 / load 1200", usage.Server)
	}
	if usage.Server.OverheadMS() != 10200 {
		t.Errorf("OverheadMS = %d, want 10200", usage.Server.OverheadMS())
	}
	// LatencyMS is a local measurement of a fast loopback call, so the overhead
	// the server claims exceeds it. WorkMS must floor at zero rather than go
	// negative — the assertion that matters is that it did not report the full
	// latency as work.
	if got := usage.WorkMS(); got != 0 {
		t.Errorf("WorkMS = %d, want 0 for a claimed overhead larger than local latency", got)
	}
}

// An endpoint that reports nothing must leave Reported false, so a host can
// tell "no overhead" from "no information".
func TestPlainEndpointReportsNoTiming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],"+
			"\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	var usage *Usage
	for chunk := range ch {
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk arrived")
	}
	if usage.Server.Reported {
		t.Error("Reported must stay false against an endpoint that sends no timing headers")
	}
	if usage.WorkMS() != usage.LatencyMS {
		t.Errorf("WorkMS = %d, want the full LatencyMS %d", usage.WorkMS(), usage.LatencyMS)
	}
}
