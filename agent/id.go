package agent

// Entry identity.
//
// Every Entry needs an ID before it can be persisted or referenced (a
// ToolCallID pairs a result with its call; a compaction points at what it
// subsumed). Hosts that mint their own IDs are free to set Entry.ID and this
// is never consulted; NewID is the default for hosts that don't.
//
// This is a hand-rolled RFC 4122 v4 rather than github.com/google/uuid because
// `agent` promises to import nothing but agentkit/llm, agentkit/mcpmgr and
// stdlib — and that promise was quietly false for as long as it pulled in a
// module to produce 36 characters.
//
// The FORMAT is the compatibility surface, not the implementation: hosts store
// these next to IDs they generated themselves (autowork3's are TEXT columns
// documented as "UUID v4 strings", filled from google/uuid). Output here is
// byte-identical to that — lowercase 8-4-4-4-12 hex, version nibble 4, variant
// bits 10 — so the two remain interchangeable in the same column.

import (
	"crypto/rand"
	"encoding/hex"
)

// NewID returns a random RFC 4122 version 4 UUID in canonical string form.
//
// It cannot fail: as of Go 1.24 crypto/rand.Read never returns an error (it
// panics internally if the system source is unavailable, which is not a
// condition a caller could do anything about anyway). That is why this returns
// one value and not two — an error here would be untestable noise at every one
// of its call sites.
func NewID() string {
	var b [16]byte
	rand.Read(b[:])

	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10x

	var out [36]byte
	hex.Encode(out[0:8], b[0:4])
	out[8] = '-'
	hex.Encode(out[9:13], b[4:6])
	out[13] = '-'
	hex.Encode(out[14:18], b[6:8])
	out[18] = '-'
	hex.Encode(out[19:23], b[8:10])
	out[23] = '-'
	hex.Encode(out[24:36], b[10:16])
	return string(out[:])
}
