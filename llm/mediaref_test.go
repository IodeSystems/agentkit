package llm

import (
	"strings"
	"testing"
)

func TestMediaRefNamesTheBytesNotTheEncoding(t *testing.T) {
	png := []byte("\x89PNG fake bytes")
	a := MediaRef(ImageData("image/png", png))
	if !IsMediaRef(a) || !strings.HasPrefix(a, "media:") || len(a) != len("media:")+16 {
		t.Fatalf("bad ref shape %q", a)
	}
	if b := MediaRef(ImageData("image/png", png)); b != a {
		t.Fatalf("same bytes, two refs: %q %q", a, b)
	}
	// The same bytes as a file part and as audio: one name for one payload.
	if b := MediaRef(FileData("image/png", "x.png", png)); b != a {
		t.Fatalf("same bytes in a file part got %q, want %q", b, a)
	}
	if b := MediaRef(AudioData("wav", png)); b != a {
		t.Fatalf("same bytes as audio got %q, want %q", b, a)
	}
	if b := MediaRef(ImageData("image/png", append(png, 0))); b == a {
		t.Fatal("different bytes, same ref")
	}
}

func TestMediaRefOfURLAndTextParts(t *testing.T) {
	u := MediaRef(ImagePart("https://example.com/a.png"))
	if !IsMediaRef(u) || u == MediaRef(ImagePart("https://example.com/b.png")) {
		t.Fatalf("URL parts: %q", u)
	}
	id := ContentPart{Type: "file", File: &FileRef{FileID: "file-1"}}
	if !IsMediaRef(MediaRef(id)) {
		t.Fatal("file id part has no ref")
	}
	if r := MediaRef(TextPart("hi")); r != "" {
		t.Fatalf("text part has a ref %q", r)
	}
}

func TestIsMediaRefRejectsNonRefs(t *testing.T) {
	for _, s := range []string{"media:", "media:xyz", "media:0123456789abcdeg", "./media:0123456789abcdef", "media:0123456789abcdef0"} {
		if IsMediaRef(s) {
			t.Errorf("%q accepted", s)
		}
	}
}
