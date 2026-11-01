package llm

// anthropic.go — the Claude Messages API behind the same interface.
//
// Callers speak the OpenAI shape to every provider: Message, ToolDef, ChatOpts,
// StreamChunk. Anthropic's OpenAI-compatible endpoint takes that shape as it is,
// and quietly drops what makes a long agent session affordable: measured on
// api.anthropic.com, a second round read 0 tokens from cache there and 11,484 on
// /v1/messages. So a client pointed at Anthropic talks to /v1/messages and
// nothing above this file knows.
//
// THE REQUEST is translated (anthropicParams). THE RESPONSE is transcoded: each
// stream event is rewritten as the OpenAI SSE event that means the same, and
// readStream reads that. It owns everything a stream can do wrong (the loop
// watcher, truncated tool calls, usage, finish reasons), and a second reader
// would have to get all of it right again.
//
// Credentials are the SDK's: a key when the client has one, else what it finds
// in the environment, Workload Identity Federation included.
//
// What the shape cannot carry:
//   - Thinking blocks are not replayed: Message has no place for their
//     signatures. The model thinks again from the transcript.
//   - temperature, seed and the penalties are not sent: current models reject
//     them with a 400. Neither are Grammar and ResponseFormat.
//   - A forced ToolChoice becomes auto: the newest models reject forced choice.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicMaxTokens is max_tokens when the caller sets none. The API requires
// one, and the stream is why it can be this large.
const anthropicMaxTokens = 64000

// usesMessagesAPI reports whether this client talks to /v1/messages.
func (c *Client) usesMessagesAPI() bool {
	return c.MessagesAPI || strings.Contains(c.baseURL, "api.anthropic.com")
}

func (c *Client) anthropicClient() anthropic.Client {
	retries := c.Retry5xxAttempts
	if retries <= 0 {
		retries = 5
	}
	opts := []option.RequestOption{
		option.WithBaseURL(strings.TrimSuffix(strings.TrimRight(c.baseURL, "/"), "/v1")),
		option.WithMaxRetries(retries),
	}
	if c.apiKey != "" {
		opts = append(opts, option.WithAPIKey(c.apiKey))
	}
	return anthropic.NewClient(opts...)
}

// anthropicChatStream is ChatStream on the Messages API.
func (c *Client) anthropicChatStream(ctx context.Context, messages []Message, tools []ToolDef, opts *ChatOpts) (<-chan StreamChunk, error) {
	params, err := c.anthropicParams(messages, tools, opts)
	if err != nil {
		return nil, err
	}
	callStart := time.Now()
	// Closing the pipe's read end is how readStream cuts a looping generation;
	// the cancel is what carries that to the server.
	sctx, cancel := context.WithCancel(ctx)
	client := c.anthropicClient()
	stream := client.Messages.NewStreaming(sctx, params)
	r, w := io.Pipe()
	go func() {
		defer cancel()
		defer stream.Close()
		w.CloseWithError(transcodeAnthropic(stream.Next, func() anthropic.MessageStreamEventUnion { return stream.Current() }, stream.Err, w))
	}()
	ch := make(chan StreamChunk, 64)
	go func() {
		c.readStream(ctx, &cancelOnClose{ReadCloser: r, cancel: cancel}, ch, messages, callStart, responseMeta{})
	}()
	return ch, nil
}

// anthropicChat is ChatWithOpts on the Messages API: the stream, collected.
func (c *Client) anthropicChat(ctx context.Context, messages []Message, tools []ToolDef, opts *ChatOpts) (string, []ToolCall, error) {
	ch, err := c.anthropicChatStream(ctx, messages, tools, opts)
	if err != nil {
		return "", nil, err
	}
	var text strings.Builder
	var calls []ToolCall
	for chunk := range ch {
		if chunk.Error != "" {
			err = fmt.Errorf("llm: %s", chunk.Error)
		}
		text.WriteString(chunk.Content)
		if chunk.ToolCall != nil {
			calls = append(calls, *chunk.ToolCall)
		}
	}
	return text.String(), calls, err
}

type cancelOnClose struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b *cancelOnClose) Close() error {
	b.cancel()
	return b.ReadCloser.Close()
}

// anthropicParams translates an OpenAI-shaped request.
func (c *Client) anthropicParams(messages []Message, tools []ToolDef, opts *ChatOpts) (anthropic.MessageNewParams, error) {
	p := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: anthropicMaxTokens,
		// One breakpoint at the end of the conversation: every round reads what
		// the last one wrote. The system prompt has its own, below, because it
		// outlives the conversation's prefix.
		CacheControl: anthropic.NewCacheControlEphemeralParam(),
	}
	var system []string
	var rest []Message
	for _, m := range messages {
		if m.Role == "system" {
			system = append(system, m.Content)
			continue
		}
		rest = append(rest, m)
	}
	if len(system) > 0 {
		p.System = []anthropic.TextBlockParam{{
			Text:         strings.Join(system, "\n\n"),
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}}
	}
	p.Messages = anthropicMessages(rest)
	for _, t := range tools {
		tool, err := anthropicTool(t)
		if err != nil {
			return p, err
		}
		p.Tools = append(p.Tools, tool)
	}
	if opts == nil {
		return p, nil
	}
	if opts.MaxTokens > 0 {
		p.MaxTokens = int64(opts.MaxTokens)
	}
	p.StopSequences = opts.Stop
	if opts.ToolChoice == "none" {
		p.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	}
	if e := anthropicEffort(c.model, opts.ReasoningEffort); e != "" {
		p.OutputConfig.Effort = e
	}
	return p, nil
}

// anthropicEffort is the effort to send, "" for none. Thinking cannot be turned
// off on every model, so off is the lowest effort. The models before effort
// reject the field.
func anthropicEffort(model, level string) anthropic.OutputConfigEffort {
	for _, old := range []string{"claude-haiku-4-5", "claude-sonnet-4-5", "claude-3"} {
		if strings.HasPrefix(model, old) {
			return ""
		}
	}
	switch level {
	case "none":
		return anthropic.OutputConfigEffortLow
	case "low", "medium", "high", "xhigh", "max":
		return anthropic.OutputConfigEffort(level)
	}
	return ""
}

func anthropicTool(t ToolDef) (anthropic.ToolUnionParam, error) {
	var schema anthropic.ToolInputSchemaParam
	b, err := json.Marshal(t.Function.Parameters)
	if err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf("llm: tool %s schema: %w", t.Function.Name, err)
	}
	var s struct {
		Properties any      `json:"properties"`
		Required   []string `json:"required"`
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf("llm: tool %s schema: %w", t.Function.Name, err)
	}
	schema.Properties, schema.Required = s.Properties, s.Required
	tool := anthropic.ToolParam{
		Name:                t.Function.Name,
		Description:         anthropic.String(t.Function.Description),
		InputSchema:         schema,
		EagerInputStreaming: anthropic.Bool(true), // readStream checks the JSON it gets
	}
	return anthropic.ToolUnionParam{OfTool: &tool}, nil
}

// anthropicMessages folds the OpenAI turn order into the one the Messages API
// requires. Roles alternate, starting and ending with the user; a tool's results
// sit together at the head of the user message after its call; and every call
// has a result and every result a call, which a transcript that was folded or
// interrupted need not satisfy.
func anthropicMessages(messages []Message) []anthropic.MessageParam {
	results := map[string]bool{}
	for _, m := range messages {
		if m.Role == "tool" {
			results[m.ToolCallID] = true
		}
	}
	var out []anthropic.MessageParam
	var user []anthropic.ContentBlockParamUnion // text and images of the open user turn
	var tool []anthropic.ContentBlockParamUnion // its tool results, which go first
	called := map[string]bool{}
	var missing []string
	flush := func() {
		for _, id := range missing {
			tool = append(tool, anthropic.NewToolResultBlock(id, "(no result was recorded for this call)", true))
		}
		missing = nil
		if blocks := append(tool, user...); len(blocks) > 0 {
			out = append(out, anthropic.NewUserMessage(blocks...))
		}
		user, tool = nil, nil
	}
	for _, m := range messages {
		switch m.Role {
		case "assistant":
			var blocks []anthropic.ContentBlockParamUnion
			if strings.TrimSpace(m.Content) != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var input any = map[string]any{}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, input, tc.Function.Name))
			}
			if len(blocks) == 0 {
				continue // nothing said: the turns either side are one
			}
			flush()
			out = append(out, anthropic.NewAssistantMessage(blocks...))
			for _, tc := range m.ToolCalls {
				called[tc.ID] = true
				if !results[tc.ID] {
					missing = append(missing, tc.ID)
				}
			}
		case "tool":
			if called[m.ToolCallID] {
				tool = append(tool, anthropic.NewToolResultBlock(m.ToolCallID, m.Content, false))
			} else if m.Content != "" {
				// Its call was folded away. The text may still matter.
				user = append(user, anthropic.NewTextBlock(m.Content))
			}
		default:
			user = append(user, anthropicUserBlocks(m)...)
		}
	}
	if len(missing) == 0 && len(user) == 0 && len(tool) == 0 && len(out) > 0 && out[len(out)-1].Role == anthropic.MessageParamRoleAssistant {
		user = append(user, anthropic.NewTextBlock("Continue."))
	}
	flush()
	if len(out) > 0 && out[0].Role != anthropic.MessageParamRoleUser {
		out = append([]anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("(earlier conversation)"))}, out...)
	}
	return out
}

func anthropicUserBlocks(m Message) []anthropic.ContentBlockParamUnion {
	if len(m.Parts) == 0 {
		if strings.TrimSpace(m.Content) == "" {
			return nil
		}
		return []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(m.Content)}
	}
	var out []anthropic.ContentBlockParamUnion
	for _, p := range m.Parts {
		switch {
		case p.ImageURL != nil:
			if media, data, ok := strings.Cut(strings.TrimPrefix(p.ImageURL.URL, "data:"), ";base64,"); ok {
				out = append(out, anthropic.NewImageBlockBase64(media, data))
			} else {
				out = append(out, anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: p.ImageURL.URL}))
			}
		case strings.TrimSpace(p.Text) != "":
			out = append(out, anthropic.NewTextBlock(p.Text))
		}
	}
	return out
}

// transcodeAnthropic writes a Messages stream as OpenAI SSE, for readStream.
// A failure, a refusal included, is an error event, which readStream reports.
func transcodeAnthropic(next func() bool, current func() anthropic.MessageStreamEventUnion, failed func() error, w io.Writer) error {
	emit := func(v any) error {
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(w, "data: %s\n\n", b)
		return err
	}
	delta := func(d map[string]any) error {
		return emit(map[string]any{"choices": []any{map[string]any{"delta": d}}})
	}
	toolDelta := func(index int64, f map[string]any, id string) error {
		call := map[string]any{"index": index, "function": f}
		if id != "" {
			call["id"], call["type"] = id, "function"
		}
		return delta(map[string]any{"tool_calls": []any{call}})
	}
	var prompt, read, created int64
	blocks := map[int64]*struct{ tool, args bool }{}
	for next() {
		var err error
		switch e := current().AsAny().(type) {
		case anthropic.MessageStartEvent:
			read = e.Message.Usage.CacheReadInputTokens
			created = e.Message.Usage.CacheCreationInputTokens
			prompt = e.Message.Usage.InputTokens + read + created
		case anthropic.ContentBlockStartEvent:
			b := &struct{ tool, args bool }{}
			blocks[e.Index] = b
			if t, ok := e.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
				b.tool = true
				err = toolDelta(e.Index, map[string]any{"name": t.Name, "arguments": ""}, t.ID)
			}
		case anthropic.ContentBlockDeltaEvent:
			switch d := e.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				err = delta(map[string]any{"content": d.Text})
			case anthropic.ThinkingDelta:
				if d.Thinking != "" {
					err = delta(map[string]any{"reasoning_content": d.Thinking})
				}
			case anthropic.InputJSONDelta:
				if d.PartialJSON != "" {
					if b := blocks[e.Index]; b != nil {
						b.args = true
					}
					err = toolDelta(e.Index, map[string]any{"arguments": d.PartialJSON}, "")
				}
			}
		case anthropic.ContentBlockStopEvent:
			// A call with no arguments streams none, and "" is not JSON.
			if b := blocks[e.Index]; b != nil && b.tool && !b.args {
				err = toolDelta(e.Index, map[string]any{"arguments": "{}"}, "")
			}
		case anthropic.MessageDeltaEvent:
			reason := e.Delta.StopReason
			if reason == anthropic.StopReasonRefusal {
				return emit(map[string]any{"error": map[string]any{"message": anthropicRefusal(e)}})
			}
			out := e.Usage.OutputTokens
			err = emit(map[string]any{
				"choices": []any{map[string]any{"delta": map[string]any{}, "finish_reason": anthropicFinish(reason)}},
				"usage": map[string]any{
					"prompt_tokens":               prompt,
					"completion_tokens":           out,
					"total_tokens":                prompt + out,
					"cache_read_input_tokens":     read,
					"cache_creation_input_tokens": created,
				},
			})
		}
		if err != nil {
			return err
		}
	}
	if err := failed(); err != nil {
		log.Printf("llm: anthropic stream: %v", err)
		return emit(map[string]any{"error": map[string]any{"message": err.Error()}})
	}
	_, err := fmt.Fprint(w, "data: [DONE]\n\n")
	return err
}

func anthropicFinish(r anthropic.StopReason) string {
	switch r {
	case anthropic.StopReasonToolUse:
		return "tool_calls"
	case anthropic.StopReasonMaxTokens:
		return "length"
	}
	return "stop"
}

func anthropicRefusal(e anthropic.MessageDeltaEvent) string {
	d := e.Delta.StopDetails
	if d.Category != "" {
		return fmt.Sprintf("the model refused the request (%s): %s", d.Category, d.Explanation)
	}
	return "the model refused the request"
}
