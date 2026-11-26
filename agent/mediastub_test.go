package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// mediaRound appends one round: a file_read call and its result with an image
// whose bytes are unique to the round.
func mediaRound(st *memStore, i int) llm.ContentPart {
	img := llm.ImageData("image/png", []byte(fmt.Sprintf("png-bytes-%d", i)))
	id := fmt.Sprintf("c%d", i)
	at := int64(10 * (i + 1))
	st.entries = append(st.entries,
		Entry{ID: "call-" + id, Kind: KindToolCall, ToolCallID: id, ToolName: "file_read",
			Content: fmt.Sprintf(`{"path":"shots/%d.png"}`, i), CreatedAt: at},
		Entry{ID: "res-" + id, Kind: KindToolResult, ToolCallID: id, ToolName: "file_read",
			Content: fmt.Sprintf("shots/%d.png: image/png, 12 bytes", i), Parts: []llm.ContentPart{img}, CreatedAt: at + 1})
	return img
}

func mediaShaper(st *memStore, older int) *Shaper {
	return &Shaper{Store: st, Estimate: Default(), Policy: ShaperPolicy{
		StubMediaOlderThan: older, LODStep: 4, PreserveLastToolCalls: 2, PreserveLastMessages: 2,
		MediaReadTool: "file_read",
	}}
}

func build(t *testing.T, sh *Shaper) []llm.Message {
	t.Helper()
	msgs, err := sh.Build(context.Background(), "s", "sys")
	if err != nil {
		t.Fatal(err)
	}
	return msgs
}

// toolMedia reports, per tool message, whether it still carries an image part.
func toolMedia(msgs []llm.Message) []bool {
	var out []bool
	for _, m := range msgs {
		if m.Role != "tool" {
			continue
		}
		has := false
		for _, p := range m.Parts {
			if p.Type == "image_url" {
				has = true
			}
		}
		out = append(out, has)
	}
	return out
}

// An old read keeps its text and loses its image to a label that names the
// image and how to get it back; the newest reads keep theirs.
func TestStubMedia_OldResultLosesImageToALabelTextKept(t *testing.T) {
	st := &memStore{}
	first := mediaRound(st, 0)
	for i := 1; i < 10; i++ {
		mediaRound(st, i)
	}
	msgs := build(t, mediaShaper(st, 2))
	media := toolMedia(msgs)
	if media[0] || !media[len(media)-1] || !media[len(media)-2] {
		t.Fatalf("media per result = %v: want the oldest stubbed, the newest two kept", media)
	}
	var m llm.Message
	for _, x := range msgs {
		if x.Role == "tool" {
			m = x
			break
		}
	}
	ref := llm.MediaRef(first)
	want := fmt.Sprintf(`[image/png %s from file_read "shots/0.png", not resent; call file_read with path "%s" to see it again]`, ref, ref)
	// All text after the stub: a plain string body, not an array of text parts.
	if len(m.Parts) != 0 || m.Content != "shots/0.png: image/png, 12 bytes\n"+want {
		raw, _ := json.Marshal(m.Parts)
		t.Fatalf("stubbed tool message content %q parts %s\nwant text + label %s", m.Content, raw, want)
	}
	// The Store keeps the bytes.
	if e := st.entries[1]; len(e.Parts) != 1 || e.Parts[0].ImageURL == nil || e.Parts[0].ImageURL.URL != first.ImageURL.URL {
		t.Fatalf("stored entry was changed: %+v", e.Parts)
	}
}

// Off is the old render: every image is sent, however old.
func TestStubMedia_OffSendsEveryImage(t *testing.T) {
	st := &memStore{}
	for i := 0; i < 10; i++ {
		mediaRound(st, i)
	}
	for _, has := range toolMedia(build(t, mediaShaper(st, 0))) {
		if !has {
			t.Fatal("StubMediaOlderThan 0 stubbed an image")
		}
	}
}

// Round N's request must be a byte-exact prefix of round N+1's between steps,
// or the provider's prompt cache misses every round; the boundary moves only
// by LODStep, so the prefix breaks at a step and nowhere else.
func TestStubMedia_PrefixStableBetweenSteps(t *testing.T) {
	st := &memStore{}
	sh := mediaShaper(st, 2)
	var prev []llm.Message
	prevB, breaks, steps := 0, 0, 0
	for i := 0; i < 20; i++ {
		mediaRound(st, i)
		msgs := build(t, sh)
		es, _ := st.Context(context.Background(), "s")
		b := sh.mediaBoundary(es, len(es)-classifyPristineCount(es, 2, 2))
		if b%4 != 0 {
			t.Fatalf("boundary %d not a multiple of LODStep", b)
		}
		prefix := prev == nil || (len(msgs) >= len(prev) && reflect.DeepEqual(prev, msgs[:len(prev)]))
		if b != prevB {
			steps++
		}
		if !prefix {
			breaks++
			if b == prevB {
				t.Fatalf("round %d: prefix changed with the boundary unchanged at %d", i, b)
			}
		}
		prev, prevB = msgs, b
	}
	if steps == 0 || breaks == 0 {
		t.Fatalf("no step in 20 rounds (steps %d, breaks %d): the test does not reach the stub", steps, breaks)
	}
	if breaks > steps {
		t.Fatalf("%d prefix breaks for %d steps", breaks, steps)
	}
}

// The pristine tail keeps its media even when enough rounds followed it: the
// tail is what the session acts on next.
func TestStubMedia_PristineTailKeepsMedia(t *testing.T) {
	st := &memStore{}
	for i := 0; i < 10; i++ {
		mediaRound(st, i)
	}
	sh := mediaShaper(st, 1)
	sh.Policy.LODStep = 1
	sh.Policy.PreserveLastToolCalls = 6 // three call/result pairs
	media := toolMedia(build(t, sh))
	n := len(media)
	if !media[n-1] || !media[n-2] || !media[n-3] || media[n-4] {
		t.Fatalf("media per result = %v: want exactly the three in the tail kept", media)
	}
}

// A user's attachment is stubbed the same way, with its own wording; a pinned
// user entry keeps its image whatever its age.
func TestStubMedia_UserAttachmentStubbedUnlessPinned(t *testing.T) {
	for _, pin := range []bool{false, true} {
		st := &memStore{}
		img := llm.ImageData("image/jpeg", []byte("jpeg"))
		st.entries = append(st.entries, Entry{ID: "u0", Kind: KindUser, Content: "see",
			Parts: []llm.ContentPart{llm.TextPart("see"), img}, CreatedAt: 1})
		for i := 0; i < 10; i++ {
			mediaRound(st, i)
		}
		sh := mediaShaper(st, 2)
		sh.Policy.PinFirstUser = pin
		u := build(t, sh)[1]
		if u.Role != "user" || len(u.Parts) != 2 {
			t.Fatalf("user message: %+v", u)
		}
		if pin {
			if u.Parts[1].ImageURL == nil {
				t.Fatal("a pinned user entry lost its image")
			}
			continue
		}
		ref := llm.MediaRef(img)
		want := fmt.Sprintf(`[image/jpeg %s from an earlier message, not resent; call file_read with path "%s" to see it again]`, ref, ref)
		if u.Parts[0].Text != "see" || u.Parts[1].Text != want {
			t.Fatalf("user parts %+v, want label %s", u.Parts, want)
		}
	}
}

// Without the read tool's name or the call, the label words it generically and
// leaves out the source.
func TestStubMedia_LabelWithoutToolOrCall(t *testing.T) {
	img := llm.ImageData("image/png", []byte("x"))
	ref := llm.MediaRef(img)
	got := mediaLabel(img, sourceOf(Entry{}, ""), "")
	want := fmt.Sprintf(`[image/png %s, not resent; call the read file tool with path "%s" to see it again]`, ref, ref)
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	if s := sourceOf(Entry{ToolName: "shot", Content: "{}"}, ""); s != "shot" {
		t.Fatalf("source with no path = %q", s)
	}
}

// Session.MediaRef reads the part back from the Store while an entry holds it,
// and reports it gone once a fold removed that entry.
func TestSessionMediaRef_FoundThenGoneAfterFold(t *testing.T) {
	st := &memStore{}
	img := mediaRound(st, 0)
	mediaRound(st, 1)
	s := &Session{SessionID: "s", Store: st}
	ref := llm.MediaRef(img)
	got, ok := s.MediaRef(context.Background(), ref)
	if !ok || !reflect.DeepEqual(got, img) {
		t.Fatalf("MediaRef(%s) = %+v, %v", ref, got, ok)
	}
	if _, ok := s.MediaRef(context.Background(), "media:0000000000000000"); ok {
		t.Fatal("an unknown ref resolved")
	}
	_ = st.Compact(context.Background(), "s", Compaction{
		Marker:   Entry{ID: "m", Kind: KindCompaction, Content: "read two shots", CreatedAt: 15},
		Subsumes: st.entries[:2],
	})
	if _, ok := s.MediaRef(context.Background(), ref); ok {
		t.Fatal("a ref resolved after its entry was folded away")
	}
	if !strings.HasPrefix(llm.MediaRef(st.entries[1].Parts[0]), "media:") {
		t.Fatal("the kept entry has no ref")
	}
}
