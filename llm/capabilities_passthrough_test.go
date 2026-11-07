package llm

import (
	"encoding/json"
	"testing"
)

const passGLMNone = `{"error":{"message":"GLM-5.3 is a thinking-only model; disabling thinking (reasoning_effort='none') is not supported.","type":"invalid_request_error","code":400}}`

func decodeBody(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// A proxy forwards a caller's body: the refusal it gets back teaches the
// record, and the next caller's body is translated by it before forwarding.
func TestPassthroughLearnsFromARefusalAndTranslatesTheNextBody(t *testing.T) {
	c := NewClient("https://fireworks.example/v1", "", "glm")
	var learned []Learned
	c.OnLearn = func(l Learned) { learned = append(learned, l) }

	first := decodeBody(t, `{"model":"glm","messages":[],"reasoning_effort":"none"}`)
	if trs := c.TranslateBody(first); len(trs) != 0 {
		t.Fatalf("empty record translated: %v", trs)
	}
	if !c.LearnResponse(first, 400, []byte(passGLMNone)) {
		t.Fatal("the verbatim GLM refusal taught nothing")
	}
	if only, ok := c.Capabilities().ThinkingOnly(); !ok || !only || len(learned) == 0 {
		t.Fatalf("thinking_only not learned/reported: %+v", c.Capabilities().Facts)
	}

	next := decodeBody(t, `{"model":"glm","messages":[],"reasoning_effort":"none"}`)
	var reported []Translation
	c.OnTranslate = func(tr Translation) { reported = append(reported, tr) }
	trs := c.TranslateBody(next)
	if next["reasoning_effort"] == "none" || len(trs) != 1 || len(reported) != 1 {
		t.Fatalf("next body not translated: %v, %v", next, trs)
	}
}

// A max_tokens decoded from JSON is a float64. Read as "unset", the output
// cap would RAISE a caller's 512 to the model's limit.
func TestPassthroughOutputCapDoesNotRaiseADecodedMaxTokens(t *testing.T) {
	c := NewClient("https://x.example/v1", "", "m")
	c.SetCapabilities(Capabilities{Facts: map[string]Fact{CapLimitsOutput: {Value: "4096", Source: SourceDeclared}}})
	small := decodeBody(t, `{"max_tokens":512}`)
	c.TranslateBody(small)
	if bodyInt(small["max_tokens"]) != 512 {
		t.Errorf("512 changed to %v", small["max_tokens"])
	}
	big := decodeBody(t, `{"max_tokens":100000}`)
	c.TranslateBody(big)
	if bodyInt(big["max_tokens"]) != 4096 {
		t.Errorf("100000 not capped: %v", big["max_tokens"])
	}
}

func TestPassthroughSuccessRecordsTheAcceptedEffort(t *testing.T) {
	c := NewClient("https://x.example/v1", "", "m")
	sent := decodeBody(t, `{"reasoning_effort":"low"}`)
	if !c.LearnResponse(sent, 200, nil) {
		t.Fatal("accepted effort not learned")
	}
	if c.LearnResponse(sent, 200, nil) {
		t.Error("a repeat taught something new")
	}
	if got, _ := c.Capabilities().AcceptedEfforts(); len(got) != 1 || got[0] != "low" {
		t.Errorf("accepted = %v", got)
	}
}

func TestPassthroughOverflowTeachesTheWindowAnd5xxTeachesNothing(t *testing.T) {
	c := NewClient("https://x.example/v1", "", "m")
	if c.LearnResponse(map[string]any{}, 503, []byte(passGLMNone)) {
		t.Error("a 503 taught something")
	}
	body := `{"error":{"code":400,"message":"request (89653 tokens) exceeds the available context size (86016 tokens)","type":"exceed_context_size_error","n_prompt_tokens":89653,"n_ctx":86016}}`
	if !c.LearnResponse(map[string]any{}, 400, []byte(body)) {
		t.Fatal("overflow taught nothing")
	}
	if n, ok := c.Capabilities().ContextWindow(); !ok || n != 86016 {
		t.Errorf("window = %d %v", n, ok)
	}
}
