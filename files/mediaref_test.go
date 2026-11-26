package files

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// store is a resolver over a fixed set of parts, standing in for
// Session.MediaRef.
func store(parts ...llm.ContentPart) func(context.Context, string) (llm.ContentPart, bool) {
	return func(_ context.Context, ref string) (llm.ContentPart, bool) {
		for _, p := range parts {
			if llm.MediaRef(p) == ref {
				return p, true
			}
		}
		return llm.ContentPart{}, false
	}
}

func TestReadMediaRefReturnsTheExactStoredPart(t *testing.T) {
	img := llm.ImageData("image/png", []byte(pngBytes(t, noise(10, 10, false))))
	ref := llm.MediaRef(img)
	var ev events
	ts := mount(t, Config{Root: t.TempDir(), MediaRef: store(img), OnMedia: ev.on})
	res := readRes(t, ts, ref)
	if len(res.Parts) != 1 || !reflect.DeepEqual(res.Parts[0], img) {
		t.Fatalf("not the stored part: %q %d", res.Text, len(res.Parts))
	}
	contains(t, res.Text, ref+": image/png, ", "(stored copy from earlier in this session)")
	if got := ev.list(); len(got) != 0 {
		t.Fatalf("a stored copy sent unchanged fired events: %+v", got)
	}
}

func TestReadMediaRefNotFoundAndNoResolver(t *testing.T) {
	ref := "media:0123456789abcdef"
	ts := mount(t, Config{Root: t.TempDir(), MediaRef: store()})
	mustErr(t, call(t, ts, Read, map[string]any{"path": ref}), ref+" is no longer in this session")
	ts = mount(t, Config{Root: t.TempDir()})
	mustErr(t, call(t, ts, Read, map[string]any{"path": ref}), "media references are not available in this session")
	mustErr(t, call(t, ts, Read, map[string]any{"path": "media:nope"}), "not a media reference", "./media:nope")
}

func TestReadMediaRefOffSendsALabel(t *testing.T) {
	img := llm.ImageData("image/png", []byte(pngBytes(t, noise(10, 10, false))))
	ref := llm.MediaRef(img)
	var ev events
	ts := mount(t, Config{Root: t.TempDir(), MediaRef: store(img), OnMedia: ev.on,
		Media: &MediaPolicy{Image: MediaLimit{Mode: MediaOff}}})
	res := readRes(t, ts, ref)
	if len(res.Parts) != 0 || strings.HasPrefix(res.Text, "ERROR") {
		t.Fatalf("off sent a part or refused: %q", res.Text)
	}
	contains(t, res.Text, ref, "image/png not sent: image media is off for this session")
	if got := ev.list(); len(got) != 1 || !got[0].Refused || got[0].Path != ref {
		t.Fatalf("events %+v", got)
	}
}

// A stored copy over a cap lowered since it was read goes through the same
// scaling as a file over it.
func TestReadMediaRefOverCapIsScaled(t *testing.T) {
	raw := pngBytes(t, noise(400, 200, false))
	img := llm.ImageData("image/png", []byte(raw))
	ref := llm.MediaRef(img)
	var ev events
	ts := mount(t, Config{Root: t.TempDir(), MediaRef: store(img), OnMedia: ev.on,
		Media: &MediaPolicy{Image: MediaLimit{MaxBytes: int64(len(raw) / 2)}, Tools: MediaTools{None: true}}})
	res := readRes(t, ts, ref)
	if len(res.Parts) != 1 || reflect.DeepEqual(res.Parts[0], img) {
		t.Fatalf("want a scaled copy: %q", res.Text)
	}
	contains(t, res.Text, ref+": image/png", "by stdlib", "over the current limit")
	if got := ev.list(); len(got) != 1 || !got[0].Scaled {
		t.Fatalf("events %+v", got)
	}
}

// "media:" never reaches the disk; a real file of that name is read by "./".
func TestReadMediaPathNeverTouchesTheTreeButDotSlashDoes(t *testing.T) {
	name := "media:0123456789abcdef"
	root := tree(t, map[string]string{name: "a file, not media\n"})
	ts := mount(t, Config{Root: root, MediaRef: store()})
	mustErr(t, call(t, ts, Read, map[string]any{"path": name}), "no longer in this session")
	contains(t, mustOK(t, call(t, ts, Read, map[string]any{"path": "./" + name})), "a file, not media")
	mustErr(t, call(t, ts, Read, map[string]any{"path": "media:../../etc/passwd"}), "not a media reference")
}
