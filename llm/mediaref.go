package llm

// mediaref.go — a short, stable name for one media part.
//
// A media part in the history rides every later request whole. A Shaper that
// stops resending an old one (agent.ShaperPolicy.StubMediaOlderThan) leaves a
// label in its place, and the label has to name the part so the model can ask
// for it again: a files toolset's read takes the name as its path and returns
// the stored part. The name is a CONTENT hash, the idea dun's blob store uses:
// the same bytes get the same name wherever and however often they occur, the
// name says nothing about where they came from, and it can be computed from
// the part alone, by agent (the label) and by files (the lookup) without either
// importing the other.

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"net/url"
	"strings"
)

// MediaRefPrefix starts every media reference.
const MediaRefPrefix = "media:"

// MediaRef is the reference of a media part: "media:" + the first 16 hex
// characters (64 bits) of the sha256 of its payload. The payload is the
// DECODED bytes of an inline part (a data: URI's base64 or percent-encoded
// body, input_audio's bare base64), so the same file read twice, or attached
// once and read once, has one reference. A part that carries no bytes (an
// http(s) URL, a provider file id) is hashed by that string. A text part, or a
// part with nothing in it, has no reference: "".
func MediaRef(p ContentPart) string {
	var sum [sha256.Size]byte
	if b, ok := MediaBytes(p); ok {
		sum = sha256.Sum256(b)
	} else if s := mediaLocator(p); s != "" {
		sum = sha256.Sum256([]byte(s))
	} else {
		return ""
	}
	return MediaRefPrefix + hex.EncodeToString(sum[:8])
}

// IsMediaRef says whether s has the shape MediaRef returns.
func IsMediaRef(s string) bool {
	h, ok := strings.CutPrefix(s, MediaRefPrefix)
	if !ok || len(h) != 16 {
		return false
	}
	_, err := hex.DecodeString(h)
	return err == nil
}

// MediaBytes is the decoded payload of an inline media part: a data: URI's
// body (image_url, video_url, file) or input_audio's base64. false for a text
// part, a part that names its content by URL or file id, and an encoding that
// does not decode.
func MediaBytes(p ContentPart) ([]byte, bool) {
	switch {
	case p.Type == "text":
		return nil, false
	case p.ImageURL != nil:
		return dataURIBytes(p.ImageURL.URL)
	case p.VideoURL != nil:
		return dataURIBytes(p.VideoURL.URL)
	case p.File != nil:
		return dataURIBytes(p.File.FileData)
	case p.InputAudio != nil:
		b, err := base64.StdEncoding.DecodeString(p.InputAudio.Data)
		return b, err == nil && p.InputAudio.Data != ""
	}
	return nil, false
}

// MediaMIME is a media part's type ("image/png"), from its data: URI, an audio
// format or a file name; the kind word ("image", "audio", …) when none says.
func MediaMIME(p ContentPart) string { return partMime(p) }

// mediaLocator is what a part without inline bytes names its content by.
func mediaLocator(p ContentPart) string {
	switch {
	case p.Type == "text":
		return ""
	case p.ImageURL != nil:
		return p.ImageURL.URL
	case p.VideoURL != nil:
		return p.VideoURL.URL
	case p.File != nil:
		if p.File.FileID != "" {
			return p.File.FileID
		}
		return p.File.FileData
	case p.InputAudio != nil:
		return p.InputAudio.Data
	}
	return ""
}

// dataURIBytes decodes "data:<type>[;base64],<body>".
func dataURIBytes(u string) ([]byte, bool) {
	rest, ok := strings.CutPrefix(u, "data:")
	if !ok {
		return nil, false
	}
	meta, body, ok := strings.Cut(rest, ",")
	if !ok {
		return nil, false
	}
	if strings.HasSuffix(strings.ToLower(meta), ";base64") {
		b, err := base64.StdEncoding.DecodeString(body)
		return b, err == nil
	}
	s, err := url.PathUnescape(body)
	return []byte(s), err == nil
}
