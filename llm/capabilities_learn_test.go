package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// capServer answers each chat request with reply(n, body), n counting from 0,
// and keeps every body it was sent.
type capServer struct {
	*httptest.Server
	mu     sync.Mutex
	bodies []map[string]any
}

func newCapServer(t *testing.T, reply func(n int, body map[string]any) (int, string)) *capServer {
	t.Helper()
	s := &capServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		s.mu.Lock()
		n := len(s.bodies)
		s.bodies = append(s.bodies, body)
		s.mu.Unlock()
		status, out := reply(n, body)
		if body["stream"] == true && status == http.StatusOK {
			w.Header().Set("Content-Type", "text/event-stream")
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, out)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *capServer) sent() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[string]any(nil), s.bodies...)
}

const okReply = `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`

// glmRefusal is the shape of GLM-5.3's 400 on reasoning_effort "none". The
// exact Fireworks sentence was not captured (see refusalPatterns).
// glmRefusal is verbatim from Fireworks GLM-5.3 through corrallm (2026-11-07).
const glmRefusal = `{"error":{"message":"GLM-5.3 is a thinking-only model; disabling thinking (reasoning_effort='none') is not supported.","type":"invalid_request_error","code":400}}`

// glmBudgetRefusal is verbatim from the same endpoint for thinking.budget_tokens 64.
const glmBudgetRefusal = `{"error":{"message":"thinking.budget_tokens must be >= 1024","type":"invalid_request_error","code":400}}`

func openaiClient(url string) *Client {
	c := NewClient(url, "k", "glm-5.3")
	c.SetReasoningDialect(DialectOpenAI)
	return c
}

func factsOf(us []factUpdate) map[string]string {
	m := map[string]string{}
	for _, u := range us {
		m[u.Key] = u.Value
	}
	return m
}

// Each refusal wording teaches exactly its facts, and only for what was sent.
func TestRefusalPatternsTeachOnlyWhatTheBodyNames(t *testing.T) {
	none := map[string]any{"reasoning_effort": "none", "temperature": 0.0, "foo": 1, "grammar": "root ::= x"}
	cases := []struct {
		name   string
		status int
		body   string
		sent   map[string]any
		want   map[string]string
	}{
		{"openai unsupported parameter", 400,
			`{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model.","type":"invalid_request_error","param":"temperature"}}`,
			none, map[string]string{"sampling.temperature": "refused"}},
		{"openai unsupported parameter not sent", 400,
			`{"error":{"message":"Unsupported parameter: 'top_p' is not supported with this model."}}`,
			none, map[string]string{}},
		{"openai unsupported effort value", 400,
			`{"error":{"message":"Unsupported value: 'reasoning_effort' does not support 'none' with this model. Supported values are: 'low', 'medium', and 'high'."}}`,
			none, map[string]string{CapReasoningEffortsRefused: "none"}},
		{"azure unrecognized argument", 400,
			`{"error":{"message":"Unrecognized request argument supplied: reasoning_effort","type":"invalid_request_error"}}`,
			none, map[string]string{"sampling.reasoning_effort": "refused"}},
		{"azure unrecognized grammar", 400,
			`{"error":{"message":"Unrecognized request argument supplied: grammar"}}`,
			none, map[string]string{CapDecodeGrammar: "false"}},
		{"pydantic json extra_forbidden", 422,
			`{"detail":[{"type":"extra_forbidden","loc":["body","foo"],"msg":"Extra inputs are not permitted","input":1}]}`,
			none, map[string]string{"sampling.foo": "refused"}},
		{"vllm repr extra_forbidden", 400,
			`{"object":"error","message":"[{'type': 'extra_forbidden', 'loc': ('body', 'foo'), 'msg': 'Extra inputs are not permitted', 'input': 1}]","type":"BadRequestError","code":400}`,
			none, map[string]string{"sampling.foo": "refused"}},
		{"pydantic literal on effort", 400,
			`{"detail":[{"type":"literal_error","loc":["body","reasoning_effort"],"msg":"Input should be 'low', 'medium' or 'high'","input":"none"}]}`,
			none, map[string]string{CapReasoningEffortsRefused: "none"}},
		{"pydantic entries do not bleed", 422,
			`{"detail":[{"type":"extra_forbidden","loc":["body","foo"],"msg":"Extra inputs are not permitted"},{"type":"missing","loc":["body","reasoning_effort"],"msg":"Field required"}]}`,
			none, map[string]string{"sampling.foo": "refused"}},
		{"jsonschema enum", 400,
			`{"error":{"message":"'none' is not one of ['low', 'medium', 'high']"}}`,
			none, map[string]string{CapReasoningEffortsRefused: "none"}},
		{"jsonschema enum other value", 400,
			`{"error":{"message":"'auto' is not one of ['low', 'medium', 'high']"}}`,
			none, map[string]string{}},
		{"fireworks glm thinking-only (verbatim)", 400, glmRefusal,
			none, map[string]string{CapReasoningThinkingOnly: "true", CapReasoningEffortsRefused: "none"}},
		{"thinking cannot be disabled", 400,
			`{"error":{"message":"Thinking cannot be disabled for this model"}}`,
			none, map[string]string{CapReasoningThinkingOnly: "true", CapReasoningEffortsRefused: "none"}},
		{"fireworks glm budget floor (verbatim)", 400, glmBudgetRefusal,
			none, map[string]string{CapReasoningBudgetMin: "1024", CapReasoningBudgetField: "thinking.budget_tokens"}},
		{"thinking is required, effort low", 400,
			`{"error":{"message":"thinking is required for this model"}}`,
			map[string]any{"reasoning_effort": "low"}, map[string]string{CapReasoningThinkingOnly: "true"}},
		{"reasoning is mandatory, alone", 400,
			`{"error":{"message":"Reasoning is mandatory for this model."}}`,
			map[string]any{"reasoning_effort": "low"}, map[string]string{CapReasoningThinkingOnly: "true"}},
		{"reasoning is required, alone", 400,
			`{"error":{"message":"reasoning is required"}}`,
			map[string]any{"reasoning_effort": "low"}, map[string]string{CapReasoningThinkingOnly: "true"}},
		{"escaped json in json", 400,
			`{"error":{"message":"{\"error\":{\"message\":\"Unsupported parameter: 'temperature'\"}}"}}`,
			none, map[string]string{"sampling.temperature": "refused"}},
		{"anthropic budget floor", 400,
			`{"type":"error","error":{"type":"invalid_request_error","message":"thinking.budget_tokens: Input should be greater than or equal to 1024"}}`,
			none, map[string]string{CapReasoningBudgetMin: "1024", CapReasoningBudgetField: "thinking.budget_tokens"}},
		{"budget >= form", 400,
			`{"error":{"message":"budget_tokens must be >= 1024"}}`,
			none, map[string]string{CapReasoningBudgetMin: "1024", CapReasoningBudgetField: "budget_tokens"}},
		{"openai output cap", 400,
			`{"error":{"message":"max_tokens is too large: 50000. This model supports at most 16384 completion tokens, whereas you provided 50000."}}`,
			none, map[string]string{CapLimitsOutput: "16384"}},
		{"anthropic output cap", 400,
			`{"type":"error","error":{"message":"max_tokens: 100000 > 64000, which is the maximum allowed number of output tokens for claude-x"}}`,
			none, map[string]string{CapLimitsOutput: "64000"}},
		{"unknown body", 400, `{"error":{"message":"something else went wrong"}}`, none, map[string]string{}},
		{"5xx teaches nothing", 500, glmRefusal, none, map[string]string{}},
		{"429 teaches nothing", 429, glmRefusal, none, map[string]string{}},
		{"402 teaches nothing", 402, glmRefusal, none, map[string]string{}},
		{"403 teaches nothing", 403, glmRefusal, none, map[string]string{}},
		{"404 teaches nothing", 404, glmRefusal, none, map[string]string{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			us := refusalFacts(tc.status, tc.body, tc.sent)
			got := factsOf(us)
			if fmt.Sprint(got) != fmt.Sprint(tc.want) {
				t.Fatalf("facts = %v, want %v", got, tc.want)
			}
			for _, u := range us {
				if !strings.HasPrefix(u.Evidence, "sent ") || !strings.Contains(u.Evidence, fmt.Sprint(tc.status)) {
					t.Errorf("evidence %q does not say what was sent and what came back", u.Evidence)
				}
			}
		})
	}
}

// GLM-5.3 end to end: off is refused once, the record learns it, and the next
// off request sends the lowest effort not refused instead of "none" again.
func TestOffRefusedByThinkingOnlyModelIsTranslatedOnTheNextRequest(t *testing.T) {
	srv := newCapServer(t, func(_ int, b map[string]any) (int, string) {
		if b["reasoning_effort"] == "none" {
			return 400, glmRefusal
		}
		return 200, okReply
	})
	c := openaiClient(srv.URL)
	c.NoResendOnRefusal = true // the next-request path, not the resend
	var learned []Learned
	var trs []Translation
	c.OnLearn = func(l Learned) { learned = append(learned, l) }
	c.OnTranslate = func(tr Translation) { trs = append(trs, tr) }
	ctx := context.Background()
	off := &ChatOpts{Effort: "none"}

	if _, _, err := c.ChatWithOpts(ctx, []Message{{Role: "user", Content: "hi"}}, nil, off); err == nil {
		t.Fatal("first request: want the 400 back, got nil")
	}
	caps := c.Capabilities()
	if only, ok := caps.ThinkingOnly(); !ok || !only {
		t.Fatalf("thinking_only not learned: %+v", caps.Facts)
	}
	if f := caps.Facts[CapReasoningEffortsRefused]; f.Value != "none" || f.Source != SourceRefused {
		t.Fatalf("refused efforts = %+v", f)
	}
	if len(learned) == 0 {
		t.Fatal("OnLearn did not fire")
	}

	if _, _, err := c.ChatWithOpts(ctx, []Message{{Role: "user", Content: "hi"}}, nil, off); err != nil {
		t.Fatalf("second request: %v", err)
	}
	sent := srv.sent()
	if len(sent) != 2 {
		t.Fatalf("requests = %d, want 2 (NoResendOnRefusal)", len(sent))
	}
	if got := sent[1]["reasoning_effort"]; got != "minimal" {
		t.Fatalf("second request sent %v, want minimal", got)
	}
	if len(trs) != 1 || trs[0].Key != "effort" || trs[0].Sent != "minimal" || trs[0].Wanted != "none" ||
		!strings.Contains(trs[0].Reason, "thinking_only") {
		t.Fatalf("translations = %+v", trs)
	}
	if acc, _ := c.Capabilities().AcceptedEfforts(); fmt.Sprint(acc) != "[minimal]" {
		t.Fatalf("accepted = %v", acc)
	}
}

func TestHostPinnedOffIsSentAndNeverOverwritten(t *testing.T) {
	srv := newCapServer(t, func(int, map[string]any) (int, string) { return 400, glmRefusal })
	c := openaiClient(srv.URL)
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningOff: fact("low", SourceHost)}})
	_, _, _ = c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: "none"})
	if got := srv.sent()[0]["reasoning_effort"]; got != "low" {
		t.Fatalf("sent %v, want the pinned low", got)
	}
	c.learn(CapReasoningOff, fact("minimal", SourceObserved))
	if f := c.Capabilities().Facts[CapReasoningOff]; f.Value != "low" || f.Source != SourceHost {
		t.Fatalf("pin overwritten: %+v", f)
	}
}

func TestTranslateEffortLadder(t *testing.T) {
	caps := func(kv ...string) Capabilities {
		m := map[string]Fact{}
		for i := 0; i < len(kv); i += 2 {
			m[kv[i]] = fact(kv[i+1], SourceRefused)
		}
		return Capabilities{Facts: m}
	}
	cases := []struct {
		want string
		caps Capabilities
		send string
	}{
		{"none", caps(), "none"},
		{"none", caps(CapReasoningEffortsRefused, "none"), "minimal"},
		{"none", caps(CapReasoningEffortsRefused, "none,minimal"), "low"},
		{"none", caps(CapReasoningThinkingOnly, "true"), "minimal"},
		{"none", caps(CapReasoningOff, "low", CapReasoningEffortsRefused, "none"), "low"},
		{"minimal", caps(CapReasoningEffortsRefused, "minimal,low"), "medium"},
		{"high", caps(CapReasoningEffortsRefused, "high"), "medium"},
		{"low", caps(CapReasoningEffortsRefused, "minimal,low,medium"), "high"},
		{"medium", caps(CapReasoningEffortsRefused, "none,minimal,low,medium"), "high"},
		{"xhigh", caps(CapReasoningEffortsRefused, "high,xhigh"), "medium"},
		{"low", caps(CapReasoningEffortsRefused, "none"), "low"},
		{"max", caps(CapReasoningEffortsRefused, "max"), "xhigh"},
		{"max", caps(), "max"},
		{"banana", caps(), "banana"},
	}
	for _, tc := range cases {
		if got := translateEffort(tc.want, tc.caps); got != tc.send {
			t.Errorf("translate(%q, %v) = %q, want %q", tc.want, tc.caps.Facts, got, tc.send)
		}
	}
}

func TestEmptyRecordTranslatesNothing(t *testing.T) {
	c := openaiClient("http://x/v1")
	body, trs := c.chatBody(context.Background(), nil, nil, &ChatOpts{Effort: "none", MaxTokens: 50}, false)
	if trs != nil || body["reasoning_effort"] != "none" || body["max_tokens"] != 50 {
		t.Fatalf("body changed by an empty record: %v %v", body, trs)
	}
}

func TestOutputLimitCapsMaxTokens(t *testing.T) {
	c := openaiClient("http://x/v1")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapLimitsOutput: fact("1000", SourceRefused)}})
	for _, tc := range []struct{ want, sent int }{{0, 1000}, {5000, 1000}, {500, 500}} {
		body, _ := c.chatBody(context.Background(), nil, nil, &ChatOpts{MaxTokens: tc.want}, false)
		if body["max_tokens"] != tc.sent {
			t.Errorf("MaxTokens %d sent %v, want %d", tc.want, body["max_tokens"], tc.sent)
		}
	}
	if body, _ := c.chatBody(context.Background(), nil, nil, nil, false); body["max_tokens"] != 1000 {
		t.Errorf("nil opts: max_tokens %v, want 1000", body["max_tokens"])
	}
}

func TestRefusedEffortFieldIsDropped(t *testing.T) {
	c := openaiClient("http://x/v1")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{"sampling.reasoning_effort": fact("refused", SourceRefused)}})
	body, trs := c.chatBody(context.Background(), nil, nil, &ChatOpts{Effort: "low"}, false)
	if _, ok := body["reasoning_effort"]; ok || len(trs) != 1 || trs[0].Sent != "" || trs[0].Wanted != "low" {
		t.Fatalf("body %v translations %+v", body, trs)
	}
}

// Messages API: the shim reads reasoning_effort from the OpenAI-shaped body,
// so the same translation applies before it.
func TestMessagesAPIOffIsTranslatedToo(t *testing.T) {
	c := NewClient("https://api.anthropic.com", "k", "claude-x")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningEffortsRefused: fact("none", SourceRefused)}})
	body, _ := c.chatBody(context.Background(), nil, nil, &ChatOpts{Effort: "none"}, false)
	// The Messages API has no none and no minimal: both are low (the shim does
	// the same), so the refusal of none no longer decides the level.
	if body["reasoning_effort"] != "low" {
		t.Fatalf("sent %v, want low", body["reasoning_effort"])
	}
}

func TestAcceptedEffortsAreDedupedInLadderOrder(t *testing.T) {
	srv := newCapServer(t, func(int, map[string]any) (int, string) { return 200, okReply })
	c := openaiClient(srv.URL)
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningEffortsRefused: fact("high", SourceRefused)}})
	n := 0
	c.OnLearn = func(Learned) { n++ }
	for _, e := range []string{"high", "low", "xhigh", "low"} {
		// "high" is refused, so it goes out as medium (lower first).
		if _, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: Effort(e)}); err != nil {
			t.Fatal(err)
		}
	}
	if acc, _ := c.Capabilities().AcceptedEfforts(); fmt.Sprint(acc) != "[low medium xhigh]" {
		t.Fatalf("accepted = %v", acc)
	}
	if n != 3 {
		t.Fatalf("OnLearn fired %d times, want 3 (a repeat confirmation is quiet)", n)
	}
}

// An effort served after being refused leaves the refused list: the model was
// fixed upstream, and the latest evidence wins.
func TestAcceptedEffortLeavesRefusedList(t *testing.T) {
	srv := newCapServer(t, func(int, map[string]any) (int, string) { return 200, okReply })
	c := openaiClient(srv.URL)
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningEffortsRefused: fact("none,medium", SourceRefused)}})
	c.observeAccepted(sentRequest{effort: "medium"})
	if r, _ := c.Capabilities().RefusedEfforts(); fmt.Sprint(r) != "[none]" {
		t.Fatalf("refused = %v", r)
	}
}

func TestOverflowTeachesContextWindow(t *testing.T) {
	srv := newCapServer(t, func(int, map[string]any) (int, string) {
		return 400, `{"error":{"code":400,"message":"request (89653 tokens) exceeds the available context size (86016 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":89653,"n_ctx":86016}}`
	})
	c := openaiClient(srv.URL)
	_, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, nil)
	var big *InputTooLargeError
	if !errors.As(err, &big) {
		t.Fatalf("err = %v, want *InputTooLargeError unchanged", err)
	}
	f := c.Capabilities().Facts[CapLimitsContext]
	if f.Value != "86016" || f.Source != SourceObserved || !strings.Contains(f.Evidence, "89653") {
		t.Fatalf("limits.context = %+v", f)
	}
}

// --- streams ---

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

func drain(t *testing.T, c *Client, opts *ChatOpts, tools []ToolDef) {
	t.Helper()
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "hi"}}, tools, opts)
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
}

func TestStreamUsageAndReasoningFieldObserved(t *testing.T) {
	cases := []struct {
		name, body, usage, field string
	}{
		{"reasoning_content with usage", sse(
			`{"choices":[{"delta":{"reasoning_content":"hm"}}]}`,
			`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			`[DONE]`), "true", "reasoning_content"},
		{"reasoning without usage", sse(
			`{"choices":[{"delta":{"reasoning":"hm"}}]}`,
			`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
			`[DONE]`), "false", "reasoning"},
		{"cut stream says nothing about usage", sse(
			`{"choices":[{"delta":{"content":"ok"}}]}`), "", ""},
		{"error event says nothing about usage", sse(
			`{"error":{"message":"boom"}}`, `[DONE]`), "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newCapServer(t, func(int, map[string]any) (int, string) { return 200, tc.body })
			c := openaiClient(srv.URL)
			drain(t, c, nil, nil)
			caps := c.Capabilities()
			if got := caps.Facts[CapStreamUsage].Value; got != tc.usage {
				t.Errorf("stream.usage = %q, want %q", got, tc.usage)
			}
			if got := caps.Facts[CapStreamReasoningField].Value; got != tc.field {
				t.Errorf("stream.reasoning_field = %q, want %q", got, tc.field)
			}
		})
	}
}

var echoTool = []ToolDef{{Type: "function"}}

func TestForcedToolChoiceObserved(t *testing.T) {
	call := `{"choices":[{"message":{"content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"echo","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`
	length := `{"choices":[{"message":{"content":"aaa"},"finish_reason":"length"}]}`
	cases := []struct {
		name, choice, reply, want string
	}{
		{"required, no call", "required", okReply, "false"},
		{"required, call", "required", call, "true"},
		{"named, no call", `{"type":"function","function":{"name":"echo"}}`, okReply, "false"},
		{"length is no verdict", "required", length, ""},
		{"auto is not forced", "auto", okReply, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newCapServer(t, func(int, map[string]any) (int, string) { return 200, tc.reply })
			c := openaiClient(srv.URL)
			if _, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, echoTool, &ChatOpts{ToolChoice: tc.choice}); err != nil {
				t.Fatal(err)
			}
			if got := c.Capabilities().Facts[CapToolsChoiceForced].Value; got != tc.want {
				t.Fatalf("tools.choice_forced = %q, want %q", got, tc.want)
			}
		})
	}
	t.Run("stream, no call", func(t *testing.T) {
		srv := newCapServer(t, func(int, map[string]any) (int, string) {
			return 200, sse(`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`, `[DONE]`)
		})
		c := openaiClient(srv.URL)
		drain(t, c, &ChatOpts{ToolChoice: "required"}, echoTool)
		if got := c.Capabilities().Facts[CapToolsChoiceForced].Value; got != "false" {
			t.Fatalf("tools.choice_forced = %q, want false", got)
		}
	})
}

// --- resend ---

func TestResendOnRefusal(t *testing.T) {
	refuseNone := func(_ int, b map[string]any) (int, string) {
		if b["reasoning_effort"] == "none" {
			return 400, glmRefusal
		}
		return 200, okReply
	}
	off := &ChatOpts{Effort: "none"}
	msgs := []Message{{Role: "user", Content: "hi"}}

	t.Run("off: the refusal is returned, one request", func(t *testing.T) {
		srv := newCapServer(t, refuseNone)
		c := openaiClient(srv.URL)
		c.NoResendOnRefusal = true
		if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, off); err == nil {
			t.Fatal("want the 400")
		}
		if n := len(srv.sent()); n != 1 {
			t.Fatalf("requests = %d, want 1", n)
		}
	})
	t.Run("default: resent once, translated", func(t *testing.T) {
		srv := newCapServer(t, refuseNone)
		c := openaiClient(srv.URL)
		var trs []Translation
		c.OnTranslate = func(tr Translation) { trs = append(trs, tr) }
		if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, off); err != nil {
			t.Fatalf("resend did not succeed: %v", err)
		}
		sent := srv.sent()
		if len(sent) != 2 || sent[1]["reasoning_effort"] != "minimal" {
			t.Fatalf("requests = %d, second sent %v", len(sent), sent[len(sent)-1]["reasoning_effort"])
		}
		if len(trs) != 1 {
			t.Fatalf("translations = %+v, want one (the resend)", trs)
		}
	})
	t.Run("on, stream: resent once, translated", func(t *testing.T) {
		srv := newCapServer(t, func(n int, b map[string]any) (int, string) {
			if b["reasoning_effort"] == "none" {
				return 400, glmRefusal
			}
			return 200, sse(`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`, `[DONE]`)
		})
		c := openaiClient(srv.URL)
		drain(t, c, off, nil)
		if sent := srv.sent(); len(sent) != 2 || sent[1]["reasoning_effort"] != "minimal" {
			t.Fatalf("requests = %d", len(sent))
		}
	})
	t.Run("on: a refusal that teaches nothing is not resent", func(t *testing.T) {
		srv := newCapServer(t, func(int, map[string]any) (int, string) {
			return 400, `{"error":{"message":"bad request"}}`
		})
		c := openaiClient(srv.URL)
		_, _, err := c.ChatWithOpts(context.Background(), msgs, nil, off)
		if err == nil || !strings.Contains(err.Error(), "bad request") {
			t.Fatalf("err = %v", err)
		}
		if n := len(srv.sent()); n != 1 {
			t.Fatalf("requests = %d, want 1", n)
		}
	})
	t.Run("on: a fact that does not change the bytes is not resent", func(t *testing.T) {
		srv := newCapServer(t, func(int, map[string]any) (int, string) {
			return 400, `{"error":{"message":"budget_tokens must be >= 1024"}}`
		})
		c := openaiClient(srv.URL)
		_, _, _ = c.ChatWithOpts(context.Background(), msgs, nil, off)
		if n := len(srv.sent()); n != 1 {
			t.Fatalf("requests = %d, want 1", n)
		}
		if v, _ := c.Capabilities().Facts[CapReasoningBudgetMin]; v.Value != "1024" {
			t.Fatalf("budget_min not learned: %+v", v)
		}
	})
	t.Run("on: never more than one resend", func(t *testing.T) {
		// Refuses every effort it is sent, each time with a new fact.
		srv := newCapServer(t, func(_ int, b map[string]any) (int, string) {
			return 400, fmt.Sprintf(`{"error":{"message":"'%v' is not one of []"}}`, b["reasoning_effort"])
		})
		c := openaiClient(srv.URL)
		if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, off); err == nil {
			t.Fatal("want the second 400")
		}
		if n := len(srv.sent()); n != 2 {
			t.Fatalf("requests = %d, want 2", n)
		}
		if r, _ := c.Capabilities().RefusedEfforts(); fmt.Sprint(r) != "[none minimal]" {
			t.Fatalf("refused = %v (the second refusal is still learned)", r)
		}
	})
}

// OpenAI's wording for a field the endpoint does not take, as Azure-style and
// strict-schema servers send it for prompt_cache_key.
const cacheKeyRefusal = `{"error":{"message":"Unrecognized request argument supplied: prompt_cache_key","type":"invalid_request_error","param":null,"code":null}}`

func TestPromptCacheKeyIsWrittenForOpenAIShape(t *testing.T) {
	c := openaiClient("http://x/v1")
	body, _ := c.chatBody(context.Background(), nil, nil, &ChatOpts{PromptCacheKey: "k1"}, false)
	if body["prompt_cache_key"] != "k1" {
		t.Fatalf("body %v", body)
	}
}

func TestPromptCacheKeyAbsentWhenEmpty(t *testing.T) {
	c := openaiClient("http://x/v1")
	for _, o := range []*ChatOpts{nil, {}} {
		body, _ := c.chatBody(context.Background(), nil, nil, o, false)
		if _, ok := body["prompt_cache_key"]; ok {
			t.Fatalf("body %v", body)
		}
	}
}

// Anthropic caches by cache_control, not by key; the shim would not know the
// field.
func TestPromptCacheKeyNeverSentOnMessagesAPI(t *testing.T) {
	c := NewClient("https://api.anthropic.com", "k", "claude-x")
	body, _ := c.chatBody(context.Background(), nil, nil, &ChatOpts{PromptCacheKey: "k1"}, false)
	if _, ok := body["prompt_cache_key"]; ok {
		t.Fatalf("body %v", body)
	}
}

// An endpoint that 400s on the field would otherwise fail every request: the
// refusal is learned once and the field dropped after.
func TestRefusedPromptCacheKeyIsLearnedAndDropped(t *testing.T) {
	srv := newCapServer(t, func(_ int, b map[string]any) (int, string) {
		if _, ok := b["prompt_cache_key"]; ok {
			return 400, cacheKeyRefusal
		}
		return 200, okReply
	})
	c := openaiClient(srv.URL)
	c.NoResendOnRefusal = true
	learned := 0
	c.OnLearn = func(Learned) { learned++ }
	opts := &ChatOpts{PromptCacheKey: "k1"}
	msgs := []Message{{Role: "user", Content: "hi"}}
	if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, opts); err == nil {
		t.Fatal("want the 400 returned with NoResendOnRefusal")
	}
	if v, _ := c.Capabilities().fact("sampling.prompt_cache_key"); v != "refused" || learned != 1 {
		t.Fatalf("fact %q, OnLearn %d", v, learned)
	}
	if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, opts); err != nil {
		t.Fatal(err)
	}
	if sent := srv.sent(); len(sent) != 2 || sent[1]["prompt_cache_key"] != nil {
		t.Fatalf("requests %d, second %v", len(sent), sent[len(sent)-1])
	}
}

func TestRefusedPromptCacheKeyIsResentOnceWithResendOnRefusal(t *testing.T) {
	srv := newCapServer(t, func(_ int, b map[string]any) (int, string) {
		if _, ok := b["prompt_cache_key"]; ok {
			return 400, cacheKeyRefusal
		}
		return 200, okReply
	})
	c := openaiClient(srv.URL)
	if _, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{PromptCacheKey: "k1"}); err != nil {
		t.Fatal(err)
	}
	if sent := srv.sent(); len(sent) != 2 || sent[1]["prompt_cache_key"] != nil {
		t.Fatalf("requests %d", len(sent))
	}
}

// 402/403 are about the account; their text can quote anything.
func TestPromptCacheKeyRefusalOnAccountStatusTeachesNothing(t *testing.T) {
	for _, st := range []int{402, 403} {
		if us := refusalFacts(st, cacheKeyRefusal, map[string]any{"prompt_cache_key": "k1"}); len(us) != 0 {
			t.Errorf("%d taught %v", st, factsOf(us))
		}
	}
	if got := factsOf(refusalFacts(400, cacheKeyRefusal, map[string]any{"prompt_cache_key": "k1"})); got["sampling.prompt_cache_key"] != "refused" {
		t.Errorf("400 taught %v", got)
	}
}

// A thinking-only model refuses EVERY off form, not just reasoning_effort. Each
// form left on the wire makes the retry the same 400 (Gemini behind a
// llama.cpp-dialect proxy).
func TestThinkingOnlyDropsTemplateKwargThinkingOff(t *testing.T) {
	c := openaiClient("http://x")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningThinkingOnly: fact("true", SourceRefused)}})
	body := map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}
	trs := c.translate(body, nil)
	if _, ok := body["chat_template_kwargs"]; ok || len(trs) != 1 || trs[0].Key != "chat_template_kwargs.enable_thinking" {
		t.Fatalf("body %v, translations %v", body, trs)
	}
	// Other kwargs stay; enable_thinking=true is not an off form.
	body = map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false, "x": 1}}
	c.translate(body, nil)
	if kw := body["chat_template_kwargs"].(map[string]any); len(kw) != 1 || kw["x"] != 1 {
		t.Fatalf("kwargs = %v", kw)
	}
	body = map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}}
	if trs := c.translate(body, nil); len(trs) != 0 || body["chat_template_kwargs"] == nil {
		t.Fatalf("enable_thinking=true touched: %v %v", body, trs)
	}
}

func TestThinkingOnlyDropsZeroReasoningBudgetOnly(t *testing.T) {
	c := openaiClient("http://x")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapReasoningThinkingOnly: fact("true", SourceRefused)}})
	body := map[string]any{"reasoning_budget_tokens": 0}
	trs := c.translate(body, nil)
	if _, ok := body["reasoning_budget_tokens"]; ok || len(trs) != 1 || trs[0].Key != "reasoning_budget_tokens" {
		t.Fatalf("body %v, translations %v", body, trs)
	}
	body = map[string]any{"reasoning_budget_tokens": 512}
	if trs := c.translate(body, nil); len(trs) != 0 || body["reasoning_budget_tokens"] != 512 {
		t.Fatalf("positive budget touched: %v %v", body, trs)
	}
}

func TestThinkingOnlyRewritesReasoningEnabledFalseToLowestEffort(t *testing.T) {
	c := openaiClient("http://x")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{
		CapReasoningThinkingOnly:   fact("true", SourceRefused),
		CapReasoningEffortsRefused: fact("minimal", SourceRefused),
	}})
	body := map[string]any{"reasoning": map[string]any{"enabled": false}}
	trs := c.translate(body, nil)
	r, _ := body["reasoning"].(map[string]any)
	if r["effort"] != "low" || len(r) != 1 || len(trs) != 1 || trs[0].Key != "reasoning.enabled" || trs[0].Sent != "low" {
		t.Fatalf("body %v, translations %v", body, trs)
	}
	// Nothing chosen: every rung refused.
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{
		CapReasoningThinkingOnly:   fact("true", SourceRefused),
		CapReasoningEffortsRefused: fact("minimal,low,medium,high,xhigh,max", SourceRefused),
	}})
	body = map[string]any{"reasoning": map[string]any{"enabled": false}}
	c.translate(body, nil)
	if _, ok := body["reasoning"]; ok {
		t.Fatalf("reasoning kept: %v", body)
	}
}

func TestEmptyRecordKeepsOffForms(t *testing.T) {
	c := openaiClient("http://x")
	body := map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}, "reasoning_budget_tokens": 0}
	if trs := c.translate(body, nil); len(trs) != 0 || body["reasoning_budget_tokens"] != 0 {
		t.Fatalf("empty record changed %v: %v", body, trs)
	}
}

// The refusal that carries no reasoning_effort at all (llama.cpp dialect, off
// by enable_thinking=false) still teaches thinking_only, and the default
// resend sends the request again without the off field: ONE call succeeds.
// Before, the caller got the 400 (no resend by default), and a resend would
// have failed too, because translate kept enable_thinking=false.
const geminiMandatory = `{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","type":"invalid_request_error","code":400}}`

func TestThinkingMandatoryRefusalIsLearnedAndResentWithoutOffField(t *testing.T) {
	srv := newCapServer(t, func(n int, body map[string]any) (int, string) {
		if kw, _ := body["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] == false {
			return 400, geminiMandatory
		}
		return 200, okReply
	})
	c := NewClient(srv.URL, "k", "gemini")
	c.SetReasoningDialect(DialectLlamaCpp)
	off := &ChatOpts{Effort: EffortNone}
	msgs := []Message{{Role: "user", Content: "hi"}}
	if _, _, err := c.ChatWithOpts(context.Background(), msgs, nil, off); err != nil {
		t.Fatalf("one call must succeed through the resend: %v", err)
	}
	if only, ok := c.Capabilities().ThinkingOnly(); !ok || !only {
		t.Fatalf("thinking_only not learned: %+v", c.Capabilities().Facts)
	}
	sent := srv.sent()
	if len(sent) != 2 {
		t.Fatalf("requests = %d, want 2 (refused, resent)", len(sent))
	}
	// The resend clamps none to minimal: thinking on, 512-token budget.
	if kw, _ := sent[1]["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != true {
		t.Fatalf("off field still sent on the resend: %v", sent[1])
	}
	if sent[1]["reasoning_budget_tokens"] != float64(512) {
		t.Fatalf("resend budget = %v, want 512 (minimal)", sent[1]["reasoning_budget_tokens"])
	}
}

// N6, end to end: a thinking-only model refuses none. ONE Chat call with
// Effort none shows the lesson (OnLearn), then the clamp (OnTranslate "effort",
// Effective minimal, with a reason that names the fact), then the 200 from the
// default resend. Order is the contract: a host reads "learned, so now this".
func TestThinkingOnlyRefusalIsLearnedThenReportedAsEffortClampThenServed(t *testing.T) {
	srv := newCapServer(t, func(_ int, b map[string]any) (int, string) {
		if b["reasoning_effort"] == "none" {
			return 400, glmRefusal
		}
		return 200, okReply
	})
	c := openaiClient(srv.URL)
	var events []string
	var eff []Translation
	c.OnLearn = func(l Learned) { events = append(events, "learn:"+l.Key) }
	c.OnTranslate = func(tr Translation) {
		events = append(events, "translate:"+tr.Key)
		if tr.Key == "effort" {
			eff = append(eff, tr)
		}
	}
	if _, _, err := c.ChatWithOpts(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, &ChatOpts{Effort: EffortNone}); err != nil {
		t.Fatalf("one call must succeed through the resend: %v", err)
	}
	if n := len(srv.sent()); n != 2 {
		t.Fatalf("requests = %d, want 2", n)
	}
	if srv.sent()[1]["reasoning_effort"] != "minimal" {
		t.Fatalf("resend sent %v, want minimal", srv.sent()[1]["reasoning_effort"])
	}
	if len(eff) != 1 || eff[0].Wanted != "none" || eff[0].Sent != "minimal" || !strings.Contains(eff[0].Reason, CapReasoningThinkingOnly) {
		t.Fatalf("effort translations = %+v", eff)
	}
	firstLearn, firstEffort := -1, -1
	for i, e := range events {
		if strings.HasPrefix(e, "learn:") && firstLearn < 0 {
			firstLearn = i
		}
		if e == "translate:effort" && firstEffort < 0 {
			firstEffort = i
		}
	}
	if firstLearn < 0 || firstEffort < firstLearn {
		t.Fatalf("want OnLearn before the effort translation, got %v", events)
	}
}
