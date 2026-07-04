package llm

import (
	"context"
	"fmt"
	"sync"
)

// Session manages the message history and tool configuration for one LLM conversation.
type Session struct {
	mu       sync.RWMutex
	ID       string
	Messages []Message
	tools    map[string]ToolDef
	system   string
}

func NewSession(id, system string) *Session {
	s := &Session{
		ID:     id,
		system: system,
		tools:  make(map[string]ToolDef),
	}
	if system != "" {
		s.Messages = append(s.Messages, Message{Role: "system", Content: system})
	}
	return s
}

func (s *Session) AddUserMessage(content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: content})
}

func (s *Session) AddAssistantMessage(content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: content})
}

func (s *Session) AddToolMessage(toolCallID, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{
		Role:    "tool",
		Content: content,
		Name:    toolCallID,
	})
}

// AddToolResult adds a tool call from the assistant followed by the tool's result message.
func (s *Session) AddToolResult(toolCallID, result string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// The assistant message with the tool call is already added by the caller
	s.Messages = append(s.Messages, Message{
		Role:    "tool",
		Content: result,
		Name:    toolCallID,
	})
}

func (s *Session) AddToolCalls(calls []ToolCall) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// OpenAI places tool_calls on the assistant message
	msg := Message{Role: "assistant", Content: ""}
	s.Messages = append(s.Messages, msg)
}

func (s *Session) SetTools(tools []ToolDef) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools = make(map[string]ToolDef, len(tools))
	for _, t := range tools {
		s.tools[t.Function.Name] = t
	}
}

func (s *Session) GetTools() []ToolDef {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]ToolDef, 0, len(s.tools))
	for _, t := range s.tools {
		out = append(out, t)
	}
	return out
}

func (s *Session) GetMessages() []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Message, len(s.Messages))
	copy(out, s.Messages)
	return out
}

// ChatStream sends messages + tools to the LLM and returns the stream.
func (s *Session) ChatStream(ctx context.Context, client *Client) (<-chan StreamChunk, error) {
	messages := s.GetMessages()
	tools := s.GetTools()
	return client.ChatStream(ctx, messages, tools, nil)
}

// TrimMessages removes oldest messages to stay within context window.
func (s *Session) TrimMessages(max int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Messages) <= max {
		return
	}
	keep := max
	if keep < 1 {
		keep = 1
	}
	// Always keep the system message if present
	if s.system != "" {
		trimmed := []Message{s.Messages[0]}
		trimmed = append(trimmed, s.Messages[len(s.Messages)-keep+1:]...)
		s.Messages = trimmed
	} else {
		s.Messages = s.Messages[len(s.Messages)-keep:]
	}
}

func (s *Session) String() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fmt.Sprintf("Session(%s, %d messages, %d tools)", s.ID, len(s.Messages), len(s.tools))
}
