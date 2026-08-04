package agent

import (
	"regexp"
	"strings"
	"testing"
)

// The format is the compatibility surface. Hosts store these in the same
// columns as IDs they minted themselves with github.com/google/uuid — if the
// shape drifts, the two stop being interchangeable and nothing complains until
// something downstream tries to parse one.
//
// canonicalV4 encodes RFC 4122 exactly: lowercase hex, 8-4-4-4-12, the version
// nibble pinned to 4 and the variant nibble to 8/9/a/b.
var canonicalV4 = regexp.MustCompile(
	`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func TestNewIDIsCanonicalUUIDv4(t *testing.T) {
	for range 1000 {
		id := NewID()
		if !canonicalV4.MatchString(id) {
			t.Fatalf("NewID() = %q; not a canonical RFC 4122 v4", id)
		}
	}
}

// A byte-order or masking slip shows up as a constant nibble somewhere, and a
// regex over random samples would pass right through it. Check the bits that
// are supposed to be FIXED are fixed, and separately that the rest actually
// vary.
func TestNewIDBitsAreSetWhereRequired(t *testing.T) {
	const n = 2000
	seen := make(map[string]bool, n)
	varies := make([]map[byte]bool, 36)
	for i := range varies {
		varies[i] = map[byte]bool{}
	}

	for range n {
		id := NewID()
		if seen[id] {
			t.Fatalf("NewID() repeated %q within %d draws", id, n)
		}
		seen[id] = true
		for i := range len(id) {
			varies[i][id[i]] = true
		}
	}

	// Positions that must never vary: the hyphens, the version, the variant.
	for _, i := range []int{8, 13, 18, 23} {
		if len(varies[i]) != 1 || !varies[i]['-'] {
			t.Errorf("position %d = %v; want only '-'", i, keysOf(varies[i]))
		}
	}
	if len(varies[14]) != 1 || !varies[14]['4'] {
		t.Errorf("version nibble = %v; want only '4'", keysOf(varies[14]))
	}
	for c := range varies[19] {
		if !strings.ContainsRune("89ab", rune(c)) {
			t.Errorf("variant nibble = %q; want one of 89ab", c)
		}
	}

	// Every other position must actually be random. A stuck nibble here is
	// the signature of slicing the wrong bytes.
	for i := range 36 {
		switch i {
		case 8, 13, 18, 23, 14, 19:
			continue
		}
		if len(varies[i]) < 8 {
			t.Errorf("position %d only ever held %v across %d draws; not random",
				i, keysOf(varies[i]), n)
		}
	}
}

func keysOf(m map[byte]bool) string {
	var b strings.Builder
	for c := range m {
		b.WriteByte(c)
	}
	return b.String()
}
