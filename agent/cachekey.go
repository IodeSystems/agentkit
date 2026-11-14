package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
)

// Prompt-cache key: the conversation identity a slot-affinity proxy (corrallm)
// or a provider's prefix cache uses to send a request back to where its prefix
// already lives. It must be the same on every round of one conversation and
// change when the prefix changes shape: after a compaction the old prefix is
// gone, and a key that still names it routes the request to a slot holding
// nothing useful. So the key is derived from the session id plus the id of the
// newest compaction marker in the history being sent. No marker: session id
// only.
//
// Derived, not stored: nothing to persist, and a host that rebuilds a session
// from its store gets the same key.

// promptCacheKey is hex(sha256(sessionID NUL compactionID))[:32]. 32 hex chars
// stay far under the 256-char limit and leak neither id. Empty sessionID yields
// "": with no identity there is nothing stable to name.
func promptCacheKey(sessionID, compactionID string) string {
	if sessionID == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(sessionID + "\x00" + compactionID))
	return hex.EncodeToString(sum[:])[:32]
}

// compactionMarker is the id of the newest KindCompaction entry the store
// holds for the session, "" when none. The first call reads the store; later
// calls use the value the compaction sink keeps current (Turn). A failed read
// is not cached, so the next round tries again.
//
// Called AFTER the round's build: the Shaper may fold inside build, and the
// sink has run by then. Context returns non-subsumed entries, so a marker a
// newer one folded is already gone; the newest by order is taken.
func (s *Session) compactionMarker(ctx context.Context) string {
	if s.cacheMarkerKnown {
		return s.cacheMarker
	}
	entries, err := s.Store.Context(ctx, s.SessionID)
	if err != nil {
		// A key is an optimization. Fall back to the session id alone rather
		// than fail a round over it.
		return ""
	}
	sortEntries(entries)
	s.cacheMarker, s.cacheMarkerKnown = "", true
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Kind == KindCompaction {
			s.cacheMarker = entries[i].ID
			break
		}
	}
	return s.cacheMarker
}
