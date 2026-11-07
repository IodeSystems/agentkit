package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// probeServer is a fake endpoint with one switch per behavior the probe reads.
type probeServer struct {
	mu          sync.Mutex
	echo        string         // "exact" (default), "lossy", "nocall"
	refuse      map[string]int // reasoning_effort -> status to answer
	refuseBody  string
	reasonChars map[string]int // reasoning_effort -> streamed reasoning chars
	dropEffort  bool           // close the connection on any reasoning_effort request
	contextReqs int
	efforts     []string
}

func (p *probeServer) handler(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	_ = json.NewDecoder(r.Body).Decode(&req)
	p.mu.Lock()
	defer p.mu.Unlock()

	sse := func(delta map[string]any, finish string) {
		w.Header().Set("Content-Type", "text/event-stream")
		b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": delta}}})
		fmt.Fprintf(w, "data: %s\n\n", b)
		f, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", f)
	}

	if eff, ok := req["reasoning_effort"].(string); ok {
		p.efforts = append(p.efforts, eff)
		if p.dropEffort {
			conn, _, _ := w.(http.Hijacker).Hijack()
			conn.Close()
			return
		}
		if st := p.refuse[eff]; st != 0 {
			w.WriteHeader(st)
			fmt.Fprint(w, p.refuseBody)
			return
		}
		delta := map[string]any{"content": "ok"}
		if n := p.reasonChars[eff]; n > 0 {
			delta["reasoning_content"] = strings.Repeat("r", n)
		}
		sse(delta, "stop")
		return
	}
	if mt, _ := req["max_tokens"].(float64); mt == 1 {
		p.contextReqs++
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"}}]}`)
		return
	}
	if _, ok := req["grammar"]; ok {
		sse(map[string]any{"content": "yes"}, "stop")
		return
	}
	// Echo case: the prompt carries the payload as a JSON string literal.
	msgs, _ := req["messages"].([]any)
	content := ""
	if len(msgs) > 0 {
		content, _ = msgs[0].(map[string]any)["content"].(string)
	}
	if p.echo == "nocall" {
		sse(map[string]any{"content": "sure"}, "stop")
		return
	}
	var payload string
	if i := strings.Index(content, "decoded:\n"); i >= 0 {
		_ = json.Unmarshal([]byte(content[i+len("decoded:\n"):]), &payload)
	}
	if p.echo == "lossy" && strings.Contains(payload, "</parameter>") {
		payload = strings.SplitN(payload, "\n", 2)[0]
	}
	args, _ := json.Marshal(map[string]string{"text": payload})
	sse(map[string]any{"tool_calls": []any{map[string]any{
		"index": 0, "id": "c1", "type": "function",
		"function": map[string]any{"name": "echo", "arguments": string(args)},
	}}}, "tool_calls")
}

func (p *probeServer) client(t *testing.T) *Client {
	ts := httptest.NewServer(http.HandlerFunc(p.handler))
	t.Cleanup(ts.Close)
	c := NewClient(ts.URL, "", "glm")
	c.SetReasoningDialect(DialectOpenAI)
	return c
}

func factValue(caps Capabilities, key string) string { return caps.Facts[key].Value }

func TestProbeCapabilitiesRefusedNoneMakesMinimalTheOffEffort(t *testing.T) {
	p := &probeServer{refuse: map[string]int{"none": 400}, refuseBody: `{"error":{"message":"invalid reasoning_effort"}}`}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if got := factValue(caps, CapReasoningEffortsRefused); got != "none" {
		t.Errorf("refused = %q", got)
	}
	if got := factValue(caps, CapReasoningEffortsAccepted); got != "minimal,low,medium,high,xhigh" {
		t.Errorf("accepted = %q", got)
	}
	if got := factValue(caps, CapReasoningOff); got != "minimal" {
		t.Errorf("off = %q, want minimal (no reasoning measured: ladder order)", got)
	}
	if _, ok := caps.Facts[CapReasoningThinkingOnly]; ok {
		t.Errorf("thinking_only set for an unrelated refusal")
	}
	for k, f := range caps.Facts {
		if f.Source != SourceProbed || f.Evidence == "" {
			t.Errorf("%s: source %q evidence %q", k, f.Source, f.Evidence)
		}
	}
	if factValue(caps, CapToolsChoiceForced) != "true" || factValue(caps, CapToolsArgFidelity) != "exact" ||
		factValue(caps, CapDecodeGrammar) != "true" || factValue(caps, CapDecodeGrammarWithTools) != "true" {
		t.Errorf("tool surface facts: %+v", caps.Facts)
	}
}

// GLM-5.3 measured: minimal reasons 220 chars, low 107. Ladder order would call
// minimal the off effort and spend more thinking than the rung above it.
func TestProbeCapabilitiesOffIsTheEffortThatReasonsLeastNotTheLowestRung(t *testing.T) {
	p := &probeServer{
		refuse: map[string]int{"none": 400}, refuseBody: `bad`,
		reasonChars: map[string]int{"minimal": 220, "low": 107, "medium": 500, "high": 900, "xhigh": 1500},
	}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if got := factValue(caps, CapReasoningOff); got != "low" {
		t.Fatalf("off = %q, want low", got)
	}
	ev := caps.Facts[CapReasoningOff].Evidence
	if !strings.Contains(ev, "minimal=220c") || !strings.Contains(ev, "low=107c") || !strings.Contains(ev, "one sample") {
		t.Errorf("evidence lacks per-effort numbers or the sample caveat: %q", ev)
	}
}

func TestProbeCapabilitiesAcceptedNoneWinsOutright(t *testing.T) {
	p := &probeServer{reasonChars: map[string]int{"none": 300, "minimal": 0}}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if got := factValue(caps, CapReasoningOff); got != "none" {
		t.Errorf("off = %q, want none", got)
	}
	if _, ok := caps.Facts[CapReasoningEffortsRefused]; ok {
		t.Errorf("refused recorded though nothing was refused")
	}
}

func TestProbeCapabilitiesNoneRefusedAsUndisableableIsThinkingOnly(t *testing.T) {
	p := &probeServer{refuse: map[string]int{"none": 400}, refuseBody: `{"error":"thinking cannot be disabled for this model"}`}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if factValue(caps, CapReasoningThinkingOnly) != "true" {
		t.Errorf("thinking_only not recorded: %+v", caps.Facts)
	}
}

func TestProbeCapabilitiesNoToolCallMeansChoiceNotForced(t *testing.T) {
	p := &probeServer{echo: "nocall"}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if got := factValue(caps, CapToolsChoiceForced); got != "false" {
		t.Errorf("choice_forced = %q", got)
	}
	if _, ok := caps.Facts[CapToolsArgFidelity]; ok {
		t.Errorf("arg_fidelity claimed with no argument to compare")
	}
}

func TestProbeCapabilitiesLossyEchoNamesTheFailingCase(t *testing.T) {
	p := &probeServer{echo: "lossy"}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	f := caps.Facts[CapToolsArgFidelity]
	if f.Value != "lossy" || !strings.Contains(f.Evidence, "param delimiter") {
		t.Errorf("arg_fidelity = %+v", f)
	}
}

func TestProbeCapabilitiesReportsProbedFactsThroughOnLearn(t *testing.T) {
	p := &probeServer{refuse: map[string]int{"none": 400}, refuseBody: `bad`}
	c := p.client(t)
	var got []Learned
	c.OnLearn = func(l Learned) { got = append(got, l) }
	if _, err := ProbeCapabilities(context.Background(), c, ProbeOpts{}); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, l := range got {
		if l.Fact.Source != SourceProbed {
			t.Errorf("%s reported with source %q", l.Key, l.Fact.Source)
		}
		seen[l.Key] = true
	}
	for _, k := range []string{CapReasoningOff, CapReasoningEffortsRefused, CapToolsChoiceForced, CapDecodeGrammar} {
		if !seen[k] {
			t.Errorf("OnLearn never saw %s", k)
		}
	}
}

func TestProbeCapabilitiesTransportErrorAbortsKeepingEarlierFacts(t *testing.T) {
	p := &probeServer{dropEffort: true}
	c := p.client(t)
	c.RetryBudget = 1 // nanosecond: do not retry the dropped connection for minutes
	caps, err := ProbeCapabilities(context.Background(), c, ProbeOpts{})
	if err == nil {
		t.Fatal("transport error did not abort the probe")
	}
	if factValue(caps, CapToolsChoiceForced) != "true" || factValue(caps, CapDecodeGrammar) != "true" {
		t.Errorf("tool surface facts lost on abort: %+v", caps.Facts)
	}
	if _, ok := caps.Facts[CapReasoningOff]; ok {
		t.Errorf("off recorded from an aborted ladder with nothing accepted")
	}
	if len(p.efforts) != 1 {
		t.Errorf("probe went on after the failure: %v", p.efforts)
	}
}

func TestProbeCapabilitiesContextOptionOffSendsNoContextRequests(t *testing.T) {
	p := &probeServer{}
	if _, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{}); err != nil {
		t.Fatal(err)
	}
	if p.contextReqs != 0 {
		t.Fatalf("%d DiscoverContext requests with Context off", p.contextReqs)
	}
}

func TestProbeCapabilitiesContextOptionRecordsCeilingAsLowerBound(t *testing.T) {
	p := &probeServer{}
	caps, err := ProbeCapabilities(context.Background(), p.client(t), ProbeOpts{Context: true})
	if err != nil {
		t.Fatal(err)
	}
	f := caps.Facts[CapLimitsContext]
	if f.Value != "32768" || f.Source != SourceProbed || !strings.Contains(f.Evidence, "lower bound") {
		t.Errorf("limits.context = %+v", f)
	}
}

func TestProbeCapabilitiesSkipsEffortLadderOutsideEffortDialects(t *testing.T) {
	p := &probeServer{}
	c := p.client(t)
	c.SetReasoningDialect(DialectLlamaCpp)
	caps, err := ProbeCapabilities(context.Background(), c, ProbeOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.efforts) != 0 || factValue(caps, CapReasoningOff) != "" {
		t.Errorf("ladder ran on a dialect that does not send reasoning_effort: %v", p.efforts)
	}
}
