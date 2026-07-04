package main

import (
	"context"
	"sort"
	"sync"

	"github.com/google/uuid"

	"github.com/iodesystems/agentkit/agent"
)

// demoStore is a minimal in-memory agent.Store — the piece every integrator
// supplies. A real host backs this with its own tables (autowork3 uses
// Postgres event streams); the loop only needs the six methods below plus,
// for the notification demos, the two RevalidateStore methods.
//
// It is deliberately faithful to the contract:
//   - Append persists loop output (assistant replies, tool results) WITHOUT
//     marking anything pending.
//   - publish/publishNotice model an inbox arrival — persisted AND marked
//     pending, so ClaimPending reports it and the loop re-delivers.
//   - notices carry a (groupBy, key) partition so supersede + clear + the
//     MCP-revalidator preparer can target them.
type demoStore struct {
	mu        sync.Mutex
	entries   []agent.Entry
	unclaimed int
	notices   []*notice
}

// notice is one revalidatable, retractable notification.
type notice struct {
	entry          agent.Entry
	groupBy, key   string
	shown, cleared bool
}

func newDemoStore() *demoStore { return &demoStore{} }

// ── agent.Store ────────────────────────────────────────────────────

func (s *demoStore) ClaimPending(_ context.Context, _ string, at int64) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.unclaimed
	s.unclaimed = 0
	for _, nt := range s.notices {
		nt.shown = true
	}
	return n, nil
}

func (s *demoStore) Append(_ context.Context, _ string, e agent.Entry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, e)
	return nil
}

func (s *demoStore) Context(_ context.Context, _ string) ([]agent.Entry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]agent.Entry, 0, len(s.entries)+len(s.notices))
	out = append(out, s.entries...)
	for _, nt := range s.notices {
		if !nt.cleared {
			out = append(out, nt.entry)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].CreatedAt < out[j].CreatedAt })
	return out, nil
}

func (s *demoStore) Compact(_ context.Context, _ string, c agent.Compaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	subsumed := map[string]bool{}
	for _, e := range c.Subsumes {
		subsumed[e.ID] = true
	}
	kept := s.entries[:0:0]
	for _, e := range s.entries {
		if !subsumed[e.ID] {
			kept = append(kept, e)
		}
	}
	s.entries = append(kept, c.Marker)
	return nil
}

// ── inbox helpers (host-side; not part of the Store interface) ──────

// publish adds an entry AND marks it a pending inbox arrival, so the next
// ClaimPending reports it and an idle loop wakes to deliver it. This is how a
// host injects a message into an in-flight conversation.
func (s *demoStore) publish(e agent.Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, e)
	s.unclaimed++
}

// publishNotice publishes a notification partitioned by (groupBy, key). When
// supersede is true, a live UNSHOWN notice for the same partition is replaced
// wholesale (newest wins) rather than stacked — the inbox holds at most one
// live notice per key.
func (s *demoStore) publishNotice(e agent.Entry, groupBy, key string, supersede bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if supersede {
		for _, nt := range s.notices {
			if nt.cleared || nt.shown || nt.groupBy != groupBy || nt.key != key {
				continue
			}
			nt.entry = e // replace content + metadata wholesale
			return
		}
	}
	s.notices = append(s.notices, &notice{entry: e, groupBy: groupBy, key: key})
	s.unclaimed++
}

// ── agent.RevalidateStore (for the MCP-revalidator preparer) ────────

func (s *demoStore) PendingNotices(_ context.Context, _ string) ([]agent.PendingNotice, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []agent.PendingNotice
	for _, nt := range s.notices {
		if nt.cleared || nt.groupBy == "" {
			continue
		}
		out = append(out, agent.PendingNotice{GroupBy: nt.groupBy, Key: nt.key})
	}
	return out, nil
}

func (s *demoStore) Clear(_ context.Context, _ string, groupBy, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, nt := range s.notices {
		if nt.groupBy == groupBy && nt.key == key {
			nt.cleared = true
		}
	}
	return nil
}

// liveNoticeCount reports how many notices are still live (not cleared) — for
// the demos to show supersede/clear collapsing the inbox.
func (s *demoStore) liveNoticeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, nt := range s.notices {
		if !nt.cleared {
			n++
		}
	}
	return n
}

// entry is a small constructor stamping ID + CreatedAt so demos read cleanly.
func entry(kind agent.EntryKind, content string, at int64) agent.Entry {
	return agent.Entry{ID: uuid.New().String(), Kind: kind, Content: content, CreatedAt: at}
}
