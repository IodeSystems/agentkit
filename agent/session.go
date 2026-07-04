package agent

import (
	"context"
	"errors"
	"fmt"
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

// Turn runs the unified loop and returns the model's final assistant reply
// (possibly empty if the loop ended on tool calls without text).
//
// Queued-message BATCHING is inherent here: ClaimPending marks ALL pending
// inbox arrivals shown at the top of an iteration, and build() renders every
// non-subsumed entry — so N messages that queued between activations are seen
// in ONE turn, not one turn each.
func (s *Session) Turn(ctx context.Context) (lastReplyOut string, err error) {
	sp, ctx := startSpan(s.Tracer, ctx, s.spanName("Turn"))
	sp.Set("thread_id", s.ThreadID).Set("session_id", s.SessionID)
	defer func() {
		if err != nil {
			sp.EndError(err)
		} else {
			sp.End()
		}
	}()

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

	var (
		lastReply         string
		sawForcedToolCall bool
	)
	for i := 0; i < maxTurns; i++ {
		// Claim any pending inbox arrivals (marks them shown). They're
		// already persisted, so the model sees them via build() next call;
		// this is just "mark shown".
		if _, err := s.Store.ClaimPending(ctx, s.SessionID, now()); err != nil {
			return lastReply, fmt.Errorf("agent: claim pending: %w", err)
		}

		// prepareNotificationsBeforeSend: revalidate + clear stale notices so
		// the model never re-validates a resolved condition.
		if s.Preparer != nil {
			if err := s.Preparer.PrepareNotifications(ctx, s.SessionID); err != nil {
				return lastReply, fmt.Errorf("agent: prepare notifications: %w", err)
			}
		}

		messages, err := build(ctx, s.SessionID, s.System)
		if err != nil {
			return lastReply, fmt.Errorf("agent: build context: %w", err)
		}

		resp, toolCalls, err := s.streamChat(ctx, messages)
		if err != nil {
			return lastReply, fmt.Errorf("agent: chat: %w", err)
		}

		if resp != "" {
			if err := s.Store.Append(ctx, s.SessionID, Entry{
				ID:        uuid.New().String(),
				Kind:      KindAssistant,
				Content:   resp,
				CreatedAt: now(),
			}); err != nil {
				return resp, fmt.Errorf("agent: persist llm reply: %w", err)
			}
			lastReply = resp
		}

		if len(toolCalls) == 0 {
			// Idle. If new events arrived between the chat call and now, loop
			// to deliver them. Otherwise we're done.
			pending, err := s.Store.ClaimPending(ctx, s.SessionID, now())
			if err != nil {
				return lastReply, fmt.Errorf("agent: re-check pending: %w", err)
			}
			if pending == 0 {
				return lastReply, nil
			}
			continue
		}

		var sessionClosed bool
		for _, tc := range toolCalls {
			if s.ForcedTerminalTool != "" && tc.Function.Name == s.ForcedTerminalTool {
				sawForcedToolCall = true
			}
			result, err := s.Dispatch(ctx, tc)
			if err != nil {
				if errors.Is(err, ErrSessionClosed) {
					sessionClosed = true
				} else {
					result = fmt.Sprintf("ERROR: %v", err)
				}
			}
			if err := s.Store.Append(ctx, s.SessionID, Entry{
				ID:         uuid.New().String(),
				Kind:       KindToolResult,
				Content:    result,
				ToolCallID: tc.ID,
				ToolName:   tc.Function.Name,
				CreatedAt:  now(),
			}); err != nil {
				return lastReply, fmt.Errorf("agent: persist tool result: %w", err)
			}
		}
		if sessionClosed {
			// A terminal tool fired. Don't loop into another chat round.
			return lastReply, nil
		}
	}
	if s.ForcedTerminalTool != "" && !sawForcedToolCall {
		return lastReply, fmt.Errorf(
			"agent: max turns (%d) exceeded without ever calling forced terminal tool %q "+
				"(the LLM provider may be ignoring tool_choice forcing — verify against a "+
				"spec-compliant endpoint)",
			maxTurns, s.ForcedTerminalTool)
	}
	return lastReply, fmt.Errorf("agent: max turns (%d) exceeded", maxTurns)
}

// streamChat consumes the streaming response, accumulates content, collects
// tool calls, captures token usage (when reported), and emits per-token
// notifications.
func (s *Session) streamChat(ctx context.Context, messages []llm.Message) (out string, toolCalls []llm.ToolCall, err error) {
	sp, ctx := startSpan(s.Tracer, ctx, s.spanName("streamChat"))
	sp.Set("thread_id", s.ThreadID).Set("session_id", s.SessionID).Set("n_messages", len(messages))
	var usage *llm.Usage
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
		return "", nil, err
	}
	var content strings.Builder
	done := false
	for chunk := range ch {
		if chunk.Error != "" {
			return content.String(), toolCalls, fmt.Errorf("%s", chunk.Error)
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
	return content.String(), toolCalls, nil
}
