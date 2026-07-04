package agent

import (
	"context"
	"sync"

	"github.com/iodesystems/agentkit/llm"
)

// memStore is a minimal in-memory agent.Store for tests. Appended entries are
// visible to Context immediately; "pending" is modeled as entries not yet
// claimed — ClaimPending returns the unclaimed count and marks them claimed.
type memStore struct {
	mu        sync.Mutex
	entries   []Entry
	unclaimed int
}

func (m *memStore) ClaimPending(_ context.Context, _ string, _ int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := m.unclaimed
	m.unclaimed = 0
	return n, nil
}

func (m *memStore) Append(_ context.Context, _ string, e Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, e)
	return nil
}

func (m *memStore) Context(_ context.Context, _ string) ([]Entry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Entry, len(m.entries))
	copy(out, m.entries)
	return out, nil
}

func (m *memStore) Compact(_ context.Context, _ string, c Compaction) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	subsumed := map[string]bool{}
	for _, e := range c.Subsumes {
		subsumed[e.ID] = true
	}
	kept := m.entries[:0:0]
	for _, e := range m.entries {
		if !subsumed[e.ID] {
			kept = append(kept, e)
		}
	}
	m.entries = append(kept, c.Marker)
	return nil
}

// queue adds an entry and marks it pending (an inbox arrival).
func (m *memStore) queue(e Entry) {
	m.entries = append(m.entries, e)
	m.unclaimed++
}

// scriptRunner is a canned LLMRunner: it replays queued responses (content +
// optional tool calls) turn by turn, and records the messages each chat saw.
type scriptRunner struct {
	turns [][]llm.StreamChunk
	seen  [][]llm.Message
	i     int
}

func (r *scriptRunner) ChatStream(_ context.Context, msgs []llm.Message, _ []llm.ToolDef, _ *llm.ChatOpts) (<-chan llm.StreamChunk, error) {
	cp := make([]llm.Message, len(msgs))
	copy(cp, msgs)
	r.seen = append(r.seen, cp)
	var chunks []llm.StreamChunk
	if r.i < len(r.turns) {
		chunks = r.turns[r.i]
	}
	r.i++
	ch := make(chan llm.StreamChunk, len(chunks)+1)
	for _, c := range chunks {
		ch <- c
	}
	ch <- llm.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}
