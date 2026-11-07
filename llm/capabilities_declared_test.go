package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const corrallmModels = `{"object":"list","data":[
{"id":"other","context_length":1000},
{"id":"local-Swift-1.5-Qwen3.8-27B-Q6_K_S","state":"ready","kind":"model","type":"chat","context_length":86016,"slots":2,"modalities":{"image":{"formats":["png","jpeg"]},"text":{}},"outputs":["text"],"capability":"chat","thinking_default":"on","thinking_modes":["off","on"],"thinking_effort":"medium","thinking_budget":{"fraction_of_remaining":0.2,"min":2048,"max":8192}},
{"id":"think-only","context_length":0,"modalities":{"text":{"maxTokens":4096}},"thinking_default":"on","thinking_modes":["on"]}
]}`

const openRouterModels = `{"data":[{"id":"vendor/other","context_length":8}, {
"id":"qwen/qwen3-235b","name":"Qwen3","created":1745000000,"context_length":131072,
"architecture":{"modality":"text+image->text","input_modalities":["text","image"],"output_modalities":["text"],"tokenizer":"Qwen3"},
"pricing":{"prompt":"0.0000001","completion":"0.0000003"},
"top_provider":{"context_length":131072,"max_completion_tokens":32768,"is_moderated":false},
"supported_parameters":["tools","tool_choice","reasoning","include_reasoning","max_tokens","temperature","top_p","top_k","min_p","frequency_penalty","presence_penalty","repetition_penalty","seed","stop","logit_bias","logprobs","response_format","structured_outputs"]
}]}`

func declaredServer(t *testing.T, body string, hits *atomic.Int32, auth *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		if auth != nil {
			*auth = r.Header.Get("Authorization")
		}
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(body))
	}))
}

func wantFact(t *testing.T, caps Capabilities, key, value string) Fact {
	t.Helper()
	f, ok := caps.Facts[key]
	if !ok {
		t.Fatalf("fact %s missing; have %v", key, caps.Keys())
	}
	if f.Value != value || f.Source != SourceDeclared {
		t.Fatalf("fact %s = %q/%s, want %q/declared", key, f.Value, f.Source, value)
	}
	return f
}

func TestLoadDeclaredCorrallmSwiftEntry(t *testing.T) {
	srv := declaredServer(t, corrallmModels, nil, nil)
	defer srv.Close()
	c := NewClient(srv.URL, "", "local-Swift-1.5-Qwen3.8-27B-Q6_K_S")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	caps := c.Capabilities()
	f := wantFact(t, caps, CapLimitsContext, "86016")
	if f.Evidence != "GET /models: context_length=86016" {
		t.Errorf("evidence = %q", f.Evidence)
	}
	wantFact(t, caps, CapModalityPrefix+"image", "true")
	wantFact(t, caps, CapModalityPrefix+"text", "true")
	if _, ok := caps.Facts[CapModalityPrefix+"audio"]; ok {
		t.Error("audio learned though not listed")
	}
	to := wantFact(t, caps, CapReasoningThinkingOnly, "false")
	if !strings.Contains(to.Evidence, "thinking_effort=medium") {
		t.Errorf("thinking_effort missing from evidence: %q", to.Evidence)
	}
	for _, k := range []string{CapReasoningBudgetMin, CapReasoningEffortsAccepted, CapReasoningBudgetField} {
		if _, ok := caps.Facts[k]; ok {
			t.Errorf("corrallm thinking_budget/effort mapped to %s", k)
		}
	}
}

func TestLoadDeclaredCorrallmOnlyOnMeansThinkingOnlyAndZeroContextSkipped(t *testing.T) {
	srv := declaredServer(t, corrallmModels, nil, nil)
	defer srv.Close()
	c := NewClient(srv.URL, "", "think-only")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	caps := c.Capabilities()
	wantFact(t, caps, CapReasoningThinkingOnly, "true")
	wantFact(t, caps, CapLimitsOutput, "4096")
	if _, ok := caps.Facts[CapLimitsContext]; ok {
		t.Error("context_length 0 recorded as a window")
	}
}

func TestLoadDeclaredModelNotListedLearnsNothing(t *testing.T) {
	srv := declaredServer(t, corrallmModels, nil, nil)
	defer srv.Close()
	c := NewClient(srv.URL, "", "an-alias")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatalf("a miss is not an error: %v", err)
	}
	if n := len(c.Capabilities().Facts); n != 0 {
		t.Errorf("learned %d facts for an unlisted model", n)
	}
}

func TestLoadDeclaredOpenRouterShape(t *testing.T) {
	srv := declaredServer(t, openRouterModels, nil, nil)
	defer srv.Close()
	c := NewClient(srv.URL+"/v1", "", "qwen/qwen3-235b")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	caps := c.Capabilities()
	wantFact(t, caps, CapLimitsContext, "131072")
	wantFact(t, caps, CapLimitsOutput, "32768")
	wantFact(t, caps, CapModalityPrefix+"image", "true")
	wantFact(t, caps, CapDecodeJSONSchema, "true")
	wantFact(t, caps, "reasoning.supported", "true")
	wantFact(t, caps, CapSamplingPrefix+"top_k", "accepted")
	wantFact(t, caps, CapSamplingPrefix+"logit_bias", "accepted")
	wantFact(t, caps, CapSamplingPrefix+"temperature", "accepted")
	if _, ok := caps.Facts[CapToolsChoiceForced]; ok {
		t.Error("tool_choice listed support was taken as a honored forced choice")
	}
	if _, ok := caps.Facts[CapSamplingPrefix+"stop"]; ok {
		t.Error("stop is not a sampling parameter")
	}
}

func TestLoadDeclaredDoesNotOverwriteRefusedFact(t *testing.T) {
	srv := declaredServer(t, corrallmModels, nil, nil)
	defer srv.Close()
	c := NewClient(srv.URL, "", "local-Swift-1.5-Qwen3.8-27B-Q6_K_S")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{
		CapLimitsContext: {Value: "32768", Source: SourceRefused, Evidence: "400 overflow"},
	}})
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	f := c.Capabilities().Facts[CapLimitsContext]
	if f.Value != "32768" || f.Source != SourceRefused {
		t.Errorf("declared replaced evidence: %+v", f)
	}
}

func TestLoadDeclaredSendsBearerAuth(t *testing.T) {
	var auth string
	srv := declaredServer(t, corrallmModels, nil, &auth)
	defer srv.Close()
	c := NewClient(srv.URL, "sekret", "x")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer sekret" {
		t.Errorf("Authorization = %q", auth)
	}
}

func TestLoadDeclaredMessagesAPIClientMakesNoRequest(t *testing.T) {
	var hits atomic.Int32
	srv := declaredServer(t, corrallmModels, &hits, nil)
	defer srv.Close()
	c := NewClient(srv.URL, "k", "claude-x")
	c.MessagesAPI = true
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	if hits.Load() != 0 {
		t.Errorf("Messages API client sent %d requests", hits.Load())
	}
}

func TestLoadDeclaredHTTPErrorIsReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "x")
	if err := c.LoadDeclared(context.Background()); err == nil {
		t.Error("401 from /models returned nil")
	}
}

func TestLoadDeclaredHostOnlyBaseGetsV1AndHTMLIsNotAnError(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.URL.Path == "/v1/models" {
			w.Write([]byte(corrallmModels))
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("<html>dashboard</html>"))
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "think-only")
	if err := c.LoadDeclared(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(paths) != 1 || paths[0] != "/v1/models" {
		t.Errorf("paths = %v, want [/v1/models]", paths)
	}
	wantFact(t, c.Capabilities(), CapReasoningThinkingOnly, "true")

	html := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html>dashboard</html>"))
	}))
	defer html.Close()
	c2 := NewClient(html.URL, "", "think-only")
	if err := c2.LoadDeclared(context.Background()); err != nil {
		t.Fatalf("HTML answer must not be an error: %v", err)
	}
	if n := len(c2.Capabilities().Facts); n != 0 {
		t.Errorf("learned %d facts from HTML", n)
	}
}
