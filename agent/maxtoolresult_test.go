package agent

import (
	"strings"
	"testing"
)

/*
A single tool result is bounded even when it is the newest thing in the session.

LOD reaches only entries older than the preserved tail: `render` skips the tail
on purpose, because a session whose most recent tool result is stubbed loses the
thing it was about to act on. That exemption is correct and it leaves a hole —
one enormous result in the tail is sent whole, every turn, until it ages out.

Measured in aw4 on 2026-09-06: a `file.read` returned 191,623 characters, about
48k tokens against a 170k window, from one call. Nothing truncated it because it
was recent.
*/
func TestTheNewestToolResultIsStillBounded(t *testing.T) {
	sh := &Shaper{Policy: ShaperPolicy{
		BudgetTokens:       170_000,
		MaxToolResultChars: 1000,
		// LOD off, so this asserts the NEW floor and not the old path.
		LODTruncateAboveChars: 0,
	}}
	huge := strings.Repeat("x", 50_000)
	// pristineCount == len(entries): every entry is in the preserved tail, which
	// is exactly the case LOD does not cover.
	entries := []Entry{{ID: "e1", Kind: KindToolResult, Content: huge}}
	msgs := sh.render(entries, len(entries), "sys", true)

	var body string
	for _, m := range msgs {
		body += m.Content
	}
	if len(body) >= len(huge) {
		t.Fatalf("the newest result was rendered whole: %d chars", len(body))
	}
	if !strings.Contains(body, "truncated") || !strings.Contains(body, "len=50000") {
		t.Errorf("the stub does not say what was cut or how long it was:\n%.200s", body)
	}
}

// And a result under the bound is untouched — a floor that rewrote ordinary
// results would make every turn noisier for nothing.
func TestAToolResultUnderTheBoundIsUntouched(t *testing.T) {
	sh := &Shaper{Policy: ShaperPolicy{BudgetTokens: 170_000, MaxToolResultChars: 1000}}
	small := "ok\tgithub.com/iodesystems/aw4/internal/tool\t0.2s"
	entries := []Entry{{ID: "e1", Kind: KindToolResult, Content: small}}
	var body string
	for _, m := range sh.render(entries, len(entries), "sys", true) {
		body += m.Content
	}
	if !strings.Contains(body, small) {
		t.Errorf("a small result was rewritten:\n%q", body)
	}
}

// Zero disables it, like every other knob in the policy.
func TestZeroMeansNoToolResultBound(t *testing.T) {
	sh := &Shaper{Policy: ShaperPolicy{BudgetTokens: 170_000, MaxToolResultChars: 0}}
	huge := strings.Repeat("y", 20_000)
	entries := []Entry{{ID: "e1", Kind: KindToolResult, Content: huge}}
	var body string
	for _, m := range sh.render(entries, len(entries), "sys", true) {
		body += m.Content
	}
	if !strings.Contains(body, huge) {
		t.Error("MaxToolResultChars=0 truncated anyway; zero must mean unbounded")
	}
}
