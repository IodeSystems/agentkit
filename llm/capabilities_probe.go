// ProbeCapabilities: the one path that SPENDS requests to learn what an
// endpoint does, and records each answer in the Client's capability record.
//
// Everything else in the record is learned from traffic the host was sending
// anyway, or read from /models for one GET. A probe costs real generations (and
// with ProbeOpts.Context, up to 32k-token prompts), so it runs only on demand:
// dun's /model probe. See plan/capabilities-design.md.
package llm

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ProbeOpts selects the probes that cost more than a few tokens.
type ProbeOpts struct {
	// Context also runs DiscoverContext: O(log N) requests, some with prompts of
	// up to 32k tokens. Off by default for that reason. Ignored for a Messages
	// API client, which DiscoverContext refuses.
	Context bool
}

// effortLadder is the standard reasoning_effort ladder, lowest first. The order
// is the order the facts are written in and decides which accepted effort is
// "off": the lowest one the endpoint serves.
var effortLadder = []string{"none", "minimal", "low", "medium", "high", "xhigh"}

// ProbeCapabilities measures one (endpoint, model) and records each result as a
// probed fact (so OnLearn reports it, and a stored refused/observed fact is
// replaced by what the probe saw). It returns the whole record.
//
//   - ProbeToolSurface supplies tools.arg_fidelity, decode.grammar and
//     decode.grammar_with_tools. Its behavior and report type are unchanged.
//   - The echo cases already force a call with tool_choice "required", so a
//     missing call in any of them is tools.choice_forced=false.
//   - One tiny request per effort in the ladder, only where the client sends
//     reasoning_effort at all: the openai dialect and the Messages API. The
//     other dialects (llamacpp, anthropic compat, openrouter) put the effort in a
//     different field, which answers 200 whether or not it was understood, so a
//     probe would record an acceptance that means nothing.
//   - The window, when opts.Context is set.
//
// A 4xx on an effort is the endpoint's answer (refused). Any other failure is
// not an answer about the model, so the probe stops and returns the error with
// the facts learned so far still in the record. A partial record that is wrong
// would be worse than a short one.
func ProbeCapabilities(ctx context.Context, c *Client, opts ProbeOpts) (Capabilities, error) {
	rep, err := ProbeToolSurface(ctx, c)
	if err != nil {
		return c.Capabilities(), err
	}
	// ProbeToolSurface folds transport failures into the report. If every echo
	// case failed that way the endpoint was never reached, and recording "choice
	// not forced" or "grammar refused" from that would be false.
	if first, all := echoTransportFailure(rep); all {
		return c.Capabilities(), fmt.Errorf("llm: probe: tool surface unreachable: %s", first)
	}
	learnToolSurface(c, rep)

	if usesReasoningEffort(c) {
		if err := probeEfforts(ctx, c); err != nil {
			return c.Capabilities(), err
		}
	}

	if opts.Context && !c.usesMessagesAPI() {
		n, err := c.DiscoverContext(ctx)
		if err != nil {
			return c.Capabilities(), fmt.Errorf("llm: probe: context: %w", err)
		}
		ev := fmt.Sprintf("DiscoverContext: prompts up to %d tokens accepted, boundary found", n)
		if n >= contextProbeCeiling {
			ev = fmt.Sprintf("DiscoverContext: %d tokens accepted and no boundary by the probe ceiling; a lower bound, the window may be larger", n)
		}
		c.learn(CapLimitsContext, Fact{Value: strconv.Itoa(n), Source: SourceProbed, Evidence: ev})
	}
	return c.Capabilities(), nil
}

// usesReasoningEffort reports whether the client puts reasoning_effort on the
// wire; see applyReasoning.
func usesReasoningEffort(c *Client) bool {
	return c.usesMessagesAPI() || c.dialect() == DialectOpenAI
}

// echoTransportFailure returns the first echo error, and whether every echo
// case ended in one.
func echoTransportFailure(rep ToolSurfaceReport) (first string, all bool) {
	if len(rep.Echo) == 0 {
		return "", false
	}
	for _, e := range rep.Echo {
		if e.Err == "" {
			return first, false
		}
		if first == "" {
			first = e.Err
		}
	}
	return first, true
}

func learnToolSurface(c *Client, rep ToolSurfaceReport) {
	var lossy, noCall, errored []string
	judged := 0
	for _, e := range rep.Echo {
		switch {
		case e.Err != "", e.BadJSON:
			errored = append(errored, e.Name)
		case e.NoCall:
			noCall = append(noCall, e.Name)
		default:
			judged++
			if e.Lossy {
				lossy = append(lossy, fmt.Sprintf("%s (got %q)", e.Name, trimForLog(e.Got)))
			}
		}
	}

	switch {
	case len(lossy) > 0:
		c.learn(CapToolsArgFidelity, Fact{Value: "lossy", Source: SourceProbed,
			Evidence: "echo tool call returned a different argument: " + strings.Join(lossy, "; ")})
	case judged == len(rep.Echo) && judged > 0:
		c.learn(CapToolsArgFidelity, Fact{Value: "exact", Source: SourceProbed,
			Evidence: fmt.Sprintf("%d echo tool calls returned the argument byte for byte", judged)})
	}
	// Otherwise some case produced no argument to compare, and the rest were
	// exact: that is not evidence of fidelity.

	switch {
	case len(noCall) > 0:
		c.learn(CapToolsChoiceForced, Fact{Value: "false", Source: SourceProbed,
			Evidence: `tool_choice "required" returned no tool call: ` + strings.Join(noCall, ", ")})
	case len(errored) == 0 && judged > 0:
		c.learn(CapToolsChoiceForced, Fact{Value: "true", Source: SourceProbed,
			Evidence: fmt.Sprintf(`tool_choice "required" returned a tool call in all %d requests`, judged)})
	}

	c.learn(CapDecodeGrammar, Fact{Value: strconv.FormatBool(rep.GrammarAccepted), Source: SourceProbed,
		Evidence: grammarEvidence("grammar alone", rep.GrammarAccepted, rep.GrammarError)})
	c.learn(CapDecodeGrammarWithTools, Fact{Value: strconv.FormatBool(rep.GrammarWithToolsAccepted), Source: SourceProbed,
		Evidence: grammarEvidence("grammar with tools", rep.GrammarWithToolsAccepted, "")})
}

func grammarEvidence(what string, ok bool, errText string) string {
	if ok {
		return what + ": request accepted"
	}
	if errText != "" {
		return what + ": request failed: " + trimForLog(errText)
	}
	return what + ": request failed"
}

var statusInError = regexp.MustCompile(`^llm: status (\d{3})`)

// refusalStatus returns the HTTP status of an error that statusError built, and
// whether it is a 4xx. Everything else (transport failure, retry budget spent, a
// 5xx that outlived its retries) is not the endpoint answering about the field.
func refusalStatus(err error) (int, bool) {
	m := statusInError.FindStringSubmatch(err.Error())
	if m == nil {
		return 0, false
	}
	n, _ := strconv.Atoi(m[1])
	return n, n >= 400 && n < 500
}

// thinkingOnlyHints mark a refusal of effort "none" that says thinking cannot
// be switched off, as opposed to "none is not a value I know".
var thinkingOnlyHints = []string{
	"cannot be disabled", "can't be disabled", "cannot be turned off",
	"cannot disable", "can't disable", "cannot be switched off", "always enabled", "always on",
}

func bodyMeansThinkingOnly(msg string) bool {
	s := strings.ToLower(msg)
	if !strings.Contains(s, "thinking") && !strings.Contains(s, "reasoning") {
		return false
	}
	for _, h := range thinkingOnlyHints {
		if strings.Contains(s, h) {
			return true
		}
	}
	return false
}

// effortSample is what one effort's reply showed about thinking.
type effortSample struct {
	effort string
	chars  int // streamed reasoning text
	tokens int // provider-counted reasoning tokens; valid when hasTok
	hasTok bool
}

// sampleEffort sends one tiny streamed request and measures the reasoning.
// max_tokens is 256, not smaller: a cap near 16 clips every effort to the same
// length and erases the difference being measured.
func sampleEffort(ctx context.Context, c *Client, eff string) (effortSample, error) {
	s := effortSample{effort: eff}
	ch, err := c.ChatStream(ctx, []Message{{Role: "user", Content: "Reply with the word ok."}}, nil,
		&ChatOpts{ReasoningEffort: eff, MaxTokens: 256})
	if err != nil {
		return s, err
	}
	var streamErr string
	for chunk := range ch {
		s.chars += len(chunk.Reasoning)
		if chunk.Usage != nil {
			if n, ok := chunk.Usage.ReportedReasoningTokens(); ok {
				s.tokens, s.hasTok = n, true
			}
		}
		if chunk.Error != "" && streamErr == "" {
			streamErr = chunk.Error
		}
	}
	if streamErr != "" {
		return s, fmt.Errorf("%s", streamErr)
	}
	return s, nil
}

// lowestThinking picks the accepted effort that thought least. A ladder is not
// monotone in practice (GLM-5.3: minimal produced 220 reasoning chars, low 107),
// so ladder order alone would name an effort that thinks more than a higher one.
// "none" wins outright when accepted; otherwise the fewest reasoning tokens when
// every sample has a provider count, else the fewest streamed characters; ties
// and no measurement at all fall back to ladder order. It is ONE sample per
// effort, so a close call is noise; the Evidence says so.
func lowestThinking(samples []effortSample) (best string, detail string) {
	useTokens := len(samples) > 0
	for _, s := range samples {
		if !s.hasTok {
			useTokens = false
		}
	}
	var parts []string
	var bestN int
	for i, s := range samples {
		n, unit := s.chars, "c"
		if useTokens {
			n, unit = s.tokens, "t"
		}
		parts = append(parts, fmt.Sprintf("%s=%d%s", s.effort, n, unit))
		if s.effort == "none" {
			return "none", "none accepted; " + strings.Join(parts, " ")
		}
		if i == 0 || n < bestN { // strict: a tie keeps the lower rung
			best, bestN = s.effort, n
		}
	}
	return best, strings.Join(parts, " ")
}

// probeEfforts sends one request per effort and records the ladder.
func probeEfforts(ctx context.Context, c *Client) error {
	var accepted, refused []string
	var samples []effortSample
	thinkingOnlyEv := ""

	var abort error
	for _, eff := range effortLadder {
		s, err := sampleEffort(ctx, c, eff)
		if err == nil {
			accepted = append(accepted, eff)
			samples = append(samples, s)
			continue
		}
		status, is4xx := refusalStatus(err)
		if !is4xx {
			abort = fmt.Errorf("llm: probe: reasoning_effort %q: %w", eff, err)
			break
		}
		refused = append(refused, eff)
		if eff == "none" && bodyMeansThinkingOnly(err.Error()) {
			thinkingOnlyEv = fmt.Sprintf(`reasoning_effort "none" answered %d and the body says thinking cannot be disabled: %s`,
				status, trimForLog(err.Error()))
		}
	}

	// Record what was learned even when a later effort aborted the ladder: each
	// effort already answered is a fact. A partial ladder gives a partial pick of
	// "off", which a later complete probe replaces.
	probed := strings.Join(effortLadder[:len(accepted)+len(refused)], ",")
	if len(accepted) > 0 {
		c.learn(CapReasoningEffortsAccepted, Fact{Value: strings.Join(accepted, ","), Source: SourceProbed,
			Evidence: "one request per effort (max_tokens 256) over " + probed + "; 200 for these"})
		off, detail := lowestThinking(samples)
		c.learn(CapReasoningOff, Fact{Value: off, Source: SourceProbed,
			Evidence: "accepted effort with the least reasoning, one sample each: " + detail})
	}
	if len(refused) > 0 {
		c.learn(CapReasoningEffortsRefused, Fact{Value: strings.Join(refused, ","), Source: SourceProbed,
			Evidence: "one request per effort (max_tokens 256) over " + probed + "; 4xx for these"})
	} else if _, had := c.Capabilities().Facts[CapReasoningEffortsRefused]; had && abort == nil {
		// An empty value reads as "not known", which retires a stale refusal.
		c.learn(CapReasoningEffortsRefused, Fact{Value: "", Source: SourceProbed,
			Evidence: "every effort in the ladder answered 200"})
	}
	if thinkingOnlyEv != "" {
		c.learn(CapReasoningThinkingOnly, Fact{Value: "true", Source: SourceProbed, Evidence: thinkingOnlyEv})
	}
	return abort
}
