package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// llama.cpp puts "timings" beside "usage" on its final stream event, not inside
// it. A host that bills a round's wall clock needs it to tell the model's own
// compute from time spent waiting behind other requests on the slot, which no
// proxy header reports. The shape below is llama-server b11253's include_usage
// final event, trimmed.
func TestTimingsReachUsageOverTheWire(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":27834,\"completion_tokens\":103,\"total_tokens\":27937},"+
			"\"timings\":{\"cache_n\":0,\"prompt_n\":27834,\"prompt_ms\":12911.5,\"prompt_per_second\":2155.8,"+
			"\"predicted_n\":103,\"predicted_ms\":992.7,\"predicted_per_second\":103.7,\"draft_n\":60,\"draft_n_accepted\":41}}\n\n")
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
	if usage.Timings == nil {
		t.Fatal("Usage.Timings nil — the sibling timings object was dropped")
	}
	if got := usage.Timings.ComputeMS(); got != 13904 {
		t.Errorf("ComputeMS = %d, want 13904 (12911.5 + 992.7)", got)
	}
	if usage.Timings.DraftN != 60 || usage.Timings.DraftNAccepted != 41 {
		t.Errorf("draft counts = %d/%d, want 60/41 (the MTP counts llama.cpp sends)", usage.Timings.DraftN, usage.Timings.DraftNAccepted)
	}
	if usage.Timings.PromptN != 27834 || usage.Timings.PredictedN != 103 {
		t.Errorf("Timings = %+v", *usage.Timings)
	}
}

// Nothing sent, nothing claimed: nil, and ComputeMS of nil is zero.
func TestNoTimingsLeavesUsageTimingsNil(t *testing.T) {
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
	for chunk := range ch {
		if chunk.Usage != nil && chunk.Usage.Timings != nil {
			t.Fatalf("Timings = %+v from an endpoint that sent none", *chunk.Usage.Timings)
		}
		if chunk.Usage != nil && chunk.Usage.Timings.ComputeMS() != 0 {
			t.Fatal("ComputeMS of nil must be 0")
		}
	}
}
