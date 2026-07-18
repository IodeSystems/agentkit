package agent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/iodesystems/agentkit/llm"
)

// Session bundles everything needed to run one agent conversation through the
// unified turn loop. The host constructs it per unit of work and calls Turn.
// The turn model is event-driven: async arrivals reach the model at two seams
// only — the return of a tool result, or the end of an assistant turn — so
// Turn claims the pending inbox at the top of every iteration and re-checks
// after an idle response.
type Session struct {
	ThreadID  string // opaque grouping label (stamped on spans + the chat trace id)
	SessionID string
	System    string // baseline system prompt

	Store    Store
	Runner   LLMRunner
	Build    ContextBuilder // nil → DefaultContextBuilder over Store
	Tools    []llm.ToolDef
	Dispatch ToolDispatcher

	// ChatOpts forwards LLM-call options on every round-trip. Nil OK. Used
	// by protocol-bound roles to force tool_choice=required.
	ChatOpts *llm.ChatOpts

	// OnAssistantToken, if set, receives streamed content chunks for
	// SSE / live broadcast.
	OnAssistantToken func(string)

	// OnCompaction, if set, fires whenever the Shaper folds history mid-Turn —
	// the same info Turn returns in TurnResult.Compactions. A host persists the
	// summary as a hidden field on the turn and/or shows meta about it.
	OnCompaction func(CompactionInfo)

	// OnUsage, if set, fires after each chat round with the running token tally
	// (cumulative Total + current Active window). Mirrors TurnResult.Usage.
	OnUsage func(TokenUsage)

	// Estimate counts tokens for the Active-window figure. Nil → Default()
	// (chars/4). Set to the same estimator the Shaper uses for consistency.
	Estimate TokenEstimator

	// usageTotal accumulates cumulative billed tokens across this Session's
	// Turn calls (persisted state, not set by the host).
	usageTotal int

	// slots holds transclusion slots captured from tool results during the
	// current Turn (see transclude.go). Reset at the top of each Turn; results
	// merge last-writer-wins and {OUTPUT} tracks the most-recent one. The
	// engine expands an assistant reply's {name}/{OUTPUT} placeholders against
	// this before returning it as TurnResult.Reply — the model writes the short
	// placeholder, the host displays the spliced-in bytes.
	slots map[string]string

	// MaxTurns caps the loop (default 100 — generous, since a role may chain
	// read tool calls before its terminal output; better to pay extra
	// round-trips than wedge the pipeline). Tests set this small.
	MaxTurns int

	// ForcedTerminalTool, when non-empty, names the single tool the caller
	// pinned via ChatOpts.ToolChoice as the session's only legitimate exit.
	// If MaxTurns elapses without the model ever invoking it, Turn returns a
	// diagnostic naming the tool — distinguishing "ran out of budget" from
	// "provider silently ignored tool_choice and never called the right
	// tool". Empty for roles where every tool is acceptable.
	ForcedTerminalTool string

	// Preparer, if set, revalidates + clears stale pending notifications
	// before each iteration's context is built (the prepareNotifications-
	// BeforeSend seam). Nil = no preparation.
	Preparer NotificationPreparer

	// SpanPrefix names the span namespace (default "agent"): spans are
	// "<prefix>.Turn" and "<prefix>.streamChat". A host sets this to keep
	// its existing observability labels stable.
	SpanPrefix string

	// OmitSlotInstructions suppresses the automatic append of SlotSystemNote to
	// the system prompt (see transclude.go). By default, when Tools are attached
	// agent teaches the model the {OUTPUT}/{name} transclusion convention; set
	// this when the host writes those instructions itself or wants them gone.
	OmitSlotInstructions bool

	// MaxToolResultChars, if > 0, caps how much of each tool result the MODEL
	// sees: a longer result is truncated (head+tail, with a marker) in the
	// context sent this round. Storage keeps the COMPLETE result, and the
	// truncation is view-only — a reply can still surface the full bytes to the
	// user with {OUTPUT}/{name} (see transclude.go). The marker tells the model
	// the escape hatch exists. 0 = no truncation (default; no behavior change).
	MaxToolResultChars int

	// EncodeToolResult, if set, re-encodes a raw tool-result string BEFORE it is
	// stored + rendered into the model's context — e.g. JSON → YAML/TOON/CSV for a
	// terser, more token-efficient representation. nil = passthrough (the raw
	// result is stored unchanged). The transform must be information-preserving:
	// the model only READS the result, so a non-JSON representation is safe.
	EncodeToolResult func(raw string) string

	// Now is overridable in tests.
	Now func() int64

	// Tracer optionally captures spans. Nil = no tracing.
	Tracer Tracer
}

func (s *Session) spanName(n string) string {
	p := s.SpanPrefix
	if p == "" {
		p = "agent"
	}
	return p + "." + n
}

// Inject appends an entry to this session's log so the NEXT Turn renders it —
// notification / message injection. ID and CreatedAt are filled if zero.
// Injection into a different session's inbox is a host concern (broadcast /
// delivery); this is the self-inbox primitive.
func (s *Session) Inject(ctx context.Context, e Entry) error {
	if e.ID == "" {
		e.ID = uuid.New().String()
	}
	if e.CreatedAt == 0 {
		if s.Now != nil {
			e.CreatedAt = s.Now()
		} else {
			e.CreatedAt = time.Now().UnixNano()
		}
	}
	return s.Store.Append(ctx, s.SessionID, e)
}

// Turn runs the unified loop and returns a TurnResult: the model's final reply,
// any compactions the Shaper performed to stay in budget, and the running token
// tally (cumulative Total + current Active window).
//
// Queued-message BATCHING is inherent here: ClaimPending marks ALL pending
// inbox arrivals shown at the top of an iteration, and build() renders every
// non-subsumed entry — so N messages that queued between activations are seen
// in ONE turn, not one turn each.
func (s *Session) Turn(ctx context.Context) (result TurnResult, err error) {
	sp, ctx := startSpan(s.Tracer, ctx, s.spanName("Turn"))
	sp.Set("thread_id", s.ThreadID).Set("session_id", s.SessionID)
	defer func() {
		if err != nil {
			sp.EndError(err)
		} else {
			sp.End()
		}
	}()

	// Install the compaction sink so a Shaper's mid-Turn compaction surfaces
	// into result.Compactions + OnCompaction, without changing ContextBuilder.
	ctx = withCompactionSink(ctx, func(ci CompactionInfo) {
		result.Compactions = append(result.Compactions, ci)
		if s.OnCompaction != nil {
			s.OnCompaction(ci)
		}
	})

	maxTurns := s.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 100
	}
	now := s.Now
	if now == nil {
		now = func() int64 { return time.Now().UnixNano() }
	}
	build := s.Build
	if build == nil {
		build = func(ctx context.Context, sessionID, system string) ([]llm.Message, error) {
			return DefaultContextBuilder(ctx, s.Store, sessionID, system)
		}
	}

	// Effective system prompt: when tools are attached, teach the model the
	// transclusion convention (SlotSystemNote) so {OUTPUT}/{name} work out of the
	// box — no placeholders make sense with no tools, so only append then. Opt
	// out with OmitSlotInstructions when the host manages the prompt itself.
	// Computed once; constant across the loop's iterations.
	system := s.System
	if len(s.Tools) > 0 && !s.OmitSlotInstructions {
		if system == "" {
			system = SlotSystemNote
		} else {
			system += "\n\n" + SlotSystemNote
		}
	}

	// Fresh transclusion registry per Turn: a reply only references tool output
	// produced within the same Turn's loop.
	s.slots = map[string]string{}

	var sawForcedToolCall bool
	for i := 0; i < maxTurns; i++ {
		// Claim any pending inbox arrivals (marks them shown). They're
		// already persisted, so the model sees them via build() next call;
		// this is just "mark shown".
		if _, e := s.Store.ClaimPending(ctx, s.SessionID, now()); e != nil {
			return result, fmt.Errorf("agent: claim pending: %w", e)
		}

		// prepareNotificationsBeforeSend: revalidate + clear stale notices so
		// the model never re-validates a resolved condition.
		if s.Preparer != nil {
			if e := s.Preparer.PrepareNotifications(ctx, s.SessionID); e != nil {
				return result, fmt.Errorf("agent: prepare notifications: %w", e)
			}
		}

		messages, e := build(ctx, s.SessionID, system)
		if e != nil {
			return result, fmt.Errorf("agent: build context: %w", e)
		}
		// Cap the model's view of large tool results (storage keeps the
		// complete bytes; {OUTPUT}/{name} still surface them to the user).
		if s.MaxToolResultChars > 0 {
			truncateToolMessages(messages, s.MaxToolResultChars)
		}
		// Active window = the tokens actually sent this round (post compaction +
		// LOD + tool-result truncation). The last build of the Turn wins.
		result.Usage.Active = s.estimateTokens(messages)

		resp, toolCalls, usage, e := s.streamChat(ctx, messages)
		if e != nil {
			return result, fmt.Errorf("agent: chat: %w", e)
		}
		if usage != nil {
			s.usageTotal += usage.PromptTokens + usage.CompletionTokens
		}
		result.Usage.Total = s.usageTotal
		if s.OnUsage != nil {
			s.OnUsage(result.Usage)
		}

		if resp != "" {
			// Persist the reply RAW (placeholders intact) so model replay stays
			// token-lean; expand only for display in TurnResult.Reply. A reply
			// with no known {name}/{OUTPUT} placeholder is unchanged.
			if e := s.Store.Append(ctx, s.SessionID, Entry{
				ID:        uuid.New().String(),
				Kind:      KindAssistant,
				Content:   resp,
				CreatedAt: now(),
			}); e != nil {
				return result, fmt.Errorf("agent: persist llm reply: %w", e)
			}
			result.Reply = expandSlots(resp, s.slots)
		}

		if len(toolCalls) == 0 {
			// Idle. If new events arrived between the chat call and now, loop
			// to deliver them. Otherwise we're done.
			pending, e := s.Store.ClaimPending(ctx, s.SessionID, now())
			if e != nil {
				return result, fmt.Errorf("agent: re-check pending: %w", e)
			}
			if pending == 0 {
				return result, nil
			}
			continue
		}

		// Persist the assistant's tool CALLS (id + name + arguments) so a
		// rebuilt context has a valid assistant(tool_calls) → tool(tool_call_id)
		// structure, not orphan tool messages that confuse the model on replay.
		for _, tc := range toolCalls {
			if e := s.Store.Append(ctx, s.SessionID, Entry{
				ID:         uuid.New().String(),
				Kind:       KindToolCall,
				Content:    tc.Function.Arguments,
				ToolCallID: tc.ID,
				ToolName:   tc.Function.Name,
				CreatedAt:  now(),
			}); e != nil {
				return result, fmt.Errorf("agent: persist tool call: %w", e)
			}
		}

		var sessionClosed bool
		for _, tc := range toolCalls {
			if s.ForcedTerminalTool != "" && tc.Function.Name == s.ForcedTerminalTool {
				sawForcedToolCall = true
			}
			toolResult, e := s.Dispatch(ctx, tc)
			if e != nil {
				if errors.Is(e, ErrSessionClosed) {
					sessionClosed = true
				} else {
					toolResult = fmt.Sprintf("ERROR: %v", e)
				}
			}
			// Register this result's transclusion slots ({OUTPUT} + any
			// <name>...</name> sections) from the RAW result, so a reply can
			// surface the COMPLETE, human-readable bytes to the user via
			// {OUTPUT}/{name} — independent of the model-side encoding /
			// truncation below.
			maps.Copy(s.slots, captureSlots(toolResult))
			if s.EncodeToolResult != nil {
				toolResult = s.EncodeToolResult(toolResult)
			}
			if e := s.Store.Append(ctx, s.SessionID, Entry{
				ID:         uuid.New().String(),
				Kind:       KindToolResult,
				Content:    toolResult,
				ToolCallID: tc.ID,
				ToolName:   tc.Function.Name,
				CreatedAt:  now(),
			}); e != nil {
				return result, fmt.Errorf("agent: persist tool result: %w", e)
			}
		}
		if sessionClosed {
			// A terminal tool fired. Don't loop into another chat round.
			return result, nil
		}
	}
	if s.ForcedTerminalTool != "" && !sawForcedToolCall {
		return result, fmt.Errorf(
			"agent: max turns (%d) exceeded without ever calling forced terminal tool %q "+
				"(the LLM provider may be ignoring tool_choice forcing — verify against a "+
				"spec-compliant endpoint)",
			maxTurns, s.ForcedTerminalTool)
	}
	return result, fmt.Errorf("agent: max turns (%d) exceeded", maxTurns)
}

// estimateTokens sizes a built message list with the session's estimator (nil →
// Default). Used for the Active-window figure.
func (s *Session) estimateTokens(msgs []llm.Message) int {
	est := s.Estimate
	if est == nil {
		est = Default()
	}
	t := 0
	for _, m := range msgs {
		t += est.Estimate(m.Content) + est.Estimate(m.Role) + est.Estimate(m.Name)
	}
	return t
}

// streamChat consumes the streaming response, accumulates content, collects
// tool calls, captures token usage (when reported), and emits per-token
// notifications.
func (s *Session) streamChat(ctx context.Context, messages []llm.Message) (out string, toolCalls []llm.ToolCall, usage *llm.Usage, err error) {
	sp, ctx := startSpan(s.Tracer, ctx, s.spanName("streamChat"))
	sp.Set("thread_id", s.ThreadID).Set("session_id", s.SessionID).Set("n_messages", len(messages))
	defer func() {
		sp.Set("n_tool_calls", len(toolCalls)).Set("response_chars", len(out))
		if usage != nil {
			sp.Set("prompt_tokens", usage.PromptTokens).
				Set("completion_tokens", usage.CompletionTokens).
				Set("total_tokens", usage.TotalTokens)
			if usage.CacheReadInputTokens > 0 {
				sp.Set("cache_read_input_tokens", usage.CacheReadInputTokens)
			}
			if usage.CacheCreationInputTokens > 0 {
				sp.Set("cache_creation_input_tokens", usage.CacheCreationInputTokens)
			}
		}
		if err != nil {
			sp.EndError(err)
		} else {
			sp.End()
		}
	}()
	// Stamp the trace id on every round-trip so server-side logs attribute
	// requests back to a thread. Copy ChatOpts so we don't mutate shared
	// state (the pointer is shared across Turns within a Session).
	opts := llm.ChatOpts{}
	if s.ChatOpts != nil {
		opts = *s.ChatOpts
	}
	opts.TraceID = s.ThreadID
	ch, err := s.Runner.ChatStream(ctx, messages, s.Tools, &opts)
	if err != nil {
		return "", nil, nil, err
	}
	var content strings.Builder
	done := false
	for chunk := range ch {
		if chunk.Error != "" {
			return content.String(), toolCalls, usage, fmt.Errorf("%s", chunk.Error)
		}
		if chunk.Content != "" {
			content.WriteString(chunk.Content)
			if s.OnAssistantToken != nil {
				s.OnAssistantToken(chunk.Content)
			}
		}
		if chunk.ToolCall != nil {
			toolCalls = append(toolCalls, *chunk.ToolCall)
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		if chunk.Done {
			done = true
		}
		if done && usage != nil {
			go func() {
				for range ch {
				}
			}()
			break
		}
	}
	return content.String(), toolCalls, usage, nil
}
