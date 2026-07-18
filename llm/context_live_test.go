package llm

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestLive_DiscoverContext runs the real probe against a configured endpoint.
// Guarded: LLM_LIVE=1 LLM_URL=... LLM_KEY=... LLM_MODEL=... go test -run TestLive_DiscoverContext ./llm
func TestLive_DiscoverContext(t *testing.T) {
	if os.Getenv("LLM_LIVE") == "" {
		t.Skip("set LLM_LIVE=1 (+ LLM_URL/LLM_KEY/LLM_MODEL) to probe a real endpoint")
	}
	url := os.Getenv("LLM_URL")
	if url == "" {
		url = "https://llm.iodesystems.com"
	}
	model := os.Getenv("LLM_MODEL")
	if model == "" {
		model = "ternary-bonsai-27b"
	}
	c := NewClient(url, os.Getenv("LLM_KEY"), model)

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	start := time.Now()
	n, err := c.DiscoverContext(ctx)
	if err != nil {
		t.Fatalf("DiscoverContext: %v", err)
	}
	t.Logf("discovered context for %s: %d tokens (in %s)", model, n, time.Since(start).Round(time.Second))
	if n <= 0 {
		t.Fatalf("expected a positive context estimate, got %d", n)
	}
	// Must be bounded by the ceiling (no runaway climb).
	if n > contextProbeCeiling {
		t.Fatalf("probe exceeded the ceiling: %d", n)
	}
}
