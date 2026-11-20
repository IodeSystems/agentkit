package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// An endpoint that refuses reasoning-off with OpenRouter's "mandatory" wording
// (dun report, 2026-11-18). The fold must learn thinking_only, resend once
// at a level the model takes, and send that level first from then on.
func mandatoryServer(t *testing.T) (*httptest.Server, func() []string) {
	var mu sync.Mutex
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		s := string(b)
		mu.Lock()
		bodies = append(bodies, s)
		mu.Unlock()
		off := strings.Contains(s, `"enable_thinking":false`) || strings.Contains(s, `"enabled":false`) || strings.Contains(s, `"effort":"none"`) ||
			strings.Contains(s, `"reasoning_effort":"none"`)
		if off {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400,"metadata":{"provider_name":null}}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"SUMMARY\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, func() []string { mu.Lock(); defer mu.Unlock(); return append([]string(nil), bodies...) }
}

func TestMandatoryReasoning_ClientLearnsAndResends(t *testing.T) {
	srv, bodies := mandatoryServer(t)
	c := llm.NewClient(srv.URL, "", "m")
	c.SetReasoningDialect(llm.DialectLlamaCpp)
	for i := 0; i < 2; i++ {
		ch, err := c.ChatStream(context.Background(), []llm.Message{{Role: "user", Content: "hi"}}, nil, &llm.ChatOpts{Effort: llm.EffortNone})
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		for range ch {
		}
	}
	for i, b := range bodies() {
		t.Logf("req %d reasoning bits: %.300s", i, b)
	}
	only, ok := c.Capabilities().ThinkingOnly()
	t.Logf("thinking_only=%v known=%v requests=%d", only, ok, len(bodies()))
	if !ok || !only {
		t.Error("refusal not learned")
	}
	if n := len(bodies()); n != 3 {
		t.Errorf("want 3 requests (refuse, resend, first-try ok), got %d", n)
	}
}

// dun pins reasoning.off (provider effort_off, SourceHost) before the first
// request (dun capstore.go TrackCaps). The pin must not stop the resend.
func TestMandatoryReasoning_HostPinnedOffStillResends(t *testing.T) {
	for _, pin := range []string{"none", "minimal"} {
		srv, bodies := mandatoryServer(t)
		c := llm.NewClient(srv.URL, "", "m")
		c.SetReasoningDialect(llm.DialectLlamaCpp)
		c.SetCapabilities(llm.Capabilities{Facts: map[string]llm.Fact{llm.CapReasoningOff: {Value: pin, Source: llm.SourceHost}}})
		ch, err := c.ChatStream(context.Background(), []llm.Message{{Role: "user", Content: "hi"}}, nil, &llm.ChatOpts{Effort: llm.EffortNone})
		if err != nil {
			t.Errorf("pin=%s: %v (requests=%d)", pin, err, len(bodies()))
			continue
		}
		for range ch {
		}
		t.Logf("pin=%s requests=%d", pin, len(bodies()))
	}
}

func TestMandatoryReasoning_FoldThroughClient(t *testing.T) {
	srv, bodies := mandatoryServer(t)
	c := llm.NewClient(srv.URL, "", "m")
	c.SetReasoningDialect(llm.DialectLlamaCpp)
	s := &memStore{}
	for i, k := range []EntryKind{KindUser, KindToolCall, KindToolResult, KindAssistant, KindUser} {
		s.entries = append(s.entries, Entry{ID: fmt.Sprintf("e%d", i), Kind: k, ToolCallID: "c1",
			Content: strings.Repeat("body ", 2_000), CreatedAt: int64(i + 1)})
	}
	sh := &Shaper{Store: s, Runner: c,
		SummaryOpts: &llm.ChatOpts{Effort: llm.EffortNone},
		Policy:      ShaperPolicy{BudgetTokens: 1_000_000, PreserveLastMessages: 1}}
	_, ok, err := sh.Compact(context.Background(), "s")
	t.Logf("requests=%d ok=%v err=%v", len(bodies()), ok, err)
	if err != nil || !ok {
		t.Fatalf("fold failed: ok=%v err=%v", ok, err)
	}
}
