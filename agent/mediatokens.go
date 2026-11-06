package agent

// mediatokens.go — what a non-image media part costs, for messageTokens.
//
// No provider reports a part's cost before the request, and the client sees
// bytes, not tokens. Each estimate here is worked out from the bytes and errs
// HIGH (see TokenEstimator): a PDF counts every page at the top of Claude's
// published per-page range plus the page image, audio counts the densest
// encoder rate we know of, and a size the client cannot see (a URL, a file id)
// counts as a generous fixed amount.
//
// Each call decodes the part's base64 again: O(size) per estimate, every build.
// Not measured to matter yet; cache on the part if it does.

import (
	"encoding/base64"
	"encoding/binary"
	"regexp"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

const (
	// PDFPageTextTokens is one PDF page's text: the top of the 1,500–3,000
	// tokens per page the Claude PDF guide gives. Each page is also sent as an
	// image, so a page costs PDFPageTextTokens + ImageTokenEstimate.
	PDFPageTextTokens = 3000
	// PDFPageTokenEstimate is one PDF page: its text and its page image.
	PDFPageTokenEstimate = PDFPageTextTokens + ImageTokenEstimate
	// pdfBytesPerPage is the page count's fallback when no page object can be
	// seen (PDF 1.5 object streams compress them): one page per this many
	// bytes, at least one. A scanned page is ~100 KB–1 MB, a text page far less,
	// so a text PDF counts high here — the safe side.
	pdfBytesPerPage = 100 << 10

	// AudioTokensPerSecond is the densest audio encoder rate we know of: Gemini
	// documents 32 tokens per second; Whisper-style encoders behind Qwen2-Audio
	// and Qwen-Omni emit 25. Not measured against a llama.cpp or vLLM server.
	AudioTokensPerSecond = 32
	// audioBytesPerSecond is the duration fallback for a container whose header
	// is not read here (mp3, flac, ogg, m4a): 128 kbps. Lossless audio is
	// several times denser, so it counts long — the safe side.
	audioBytesPerSecond = 16000

	// VideoTokensPerSecond is one second of video: Gemini's documented ~258
	// tokens per second of frames plus its 32 of audio, rounded up. Frame
	// sampling on llama.cpp and vLLM is model-specific; not measured.
	VideoTokensPerSecond = 300
	// videoBytesPerSecond is the duration from size: 1 Mbps, low for any
	// camera or screen recording, so the duration counts long.
	videoBytesPerSecond = 125000
	// VideoURLTokenEstimate is a video given by URL, whose size the client
	// cannot see: one minute.
	VideoURLTokenEstimate = 60 * VideoTokensPerSecond
)

// partTokens is the estimated cost of one content part.
func partTokens(est TokenEstimator, p llm.ContentPart) int {
	switch {
	case p.ImageURL != nil || p.Type == "image_url":
		return ImageTokenEstimate
	case p.InputAudio != nil:
		return audioTokens(p.InputAudio)
	case p.VideoURL != nil:
		return videoTokens(p.VideoURL.URL)
	case p.File != nil:
		return fileTokens(est, p.File)
	}
	return est.Estimate(p.Text)
}

// audioTokens: duration from a WAV header when there is one, else from size at
// audioBytesPerSecond.
func audioTokens(a *llm.InputAudio) int {
	raw, err := base64.StdEncoding.DecodeString(a.Data)
	if err != nil || len(raw) == 0 {
		return AudioTokensPerSecond // at least a second: the part is not free
	}
	secs := float64(len(raw)) / audioBytesPerSecond
	if s, ok := wavSeconds(raw); ok {
		secs = s
	}
	return max(1, int(secs*AudioTokensPerSecond+0.999))
}

// wavSeconds reads a RIFF/WAVE header: the data chunk's size over the fmt
// chunk's byte rate.
func wavSeconds(b []byte) (float64, bool) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, false
	}
	var byteRate, dataLen uint32
	for off := 12; off+8 <= len(b); {
		id, n := string(b[off:off+4]), binary.LittleEndian.Uint32(b[off+4:off+8])
		body := off + 8
		switch id {
		case "fmt ":
			if body+12 <= len(b) {
				byteRate = binary.LittleEndian.Uint32(b[body+8 : body+12])
			}
		case "data":
			// A streamed WAV says 0xFFFFFFFF here; the bytes present are the
			// length that is real.
			dataLen = n
			if avail := int64(len(b) - body); int64(n) > avail {
				dataLen = uint32(avail)
			}
		}
		if byteRate > 0 && dataLen > 0 {
			return float64(dataLen) / float64(byteRate), true
		}
		next := body + int(n) + int(n&1) // chunks are word-aligned
		if next <= off {
			break
		}
		off = next
	}
	return 0, false
}

// videoTokens: duration from size at videoBytesPerSecond for a data: URI; a
// URL is VideoURLTokenEstimate.
func videoTokens(url string) int {
	raw, ok := dataURIBytes(url)
	if !ok {
		return VideoURLTokenEstimate
	}
	return max(VideoTokensPerSecond, len(raw)*VideoTokensPerSecond/videoBytesPerSecond)
}

// fileTokens: a PDF by its pages, text by its text, anything else (or a file
// by id) as one PDF page.
func fileTokens(est TokenEstimator, f *llm.FileRef) int {
	raw, ok := dataURIBytes(f.FileData)
	if !ok {
		return PDFPageTokenEstimate
	}
	media, _, _ := strings.Cut(strings.TrimPrefix(f.FileData, "data:"), ";")
	switch media = strings.ToLower(media); {
	case media == "application/pdf":
		return pdfPages(raw) * PDFPageTokenEstimate
	case strings.HasPrefix(media, "text/"), media == "application/json", media == "application/xml":
		return est.Estimate(string(raw))
	}
	return PDFPageTokenEstimate
}

// pdfPageObject matches a page object's type and not the page tree's
// ("/Type /Pages": \b fails between "e" and "s").
var pdfPageObject = regexp.MustCompile(`/Type\s*/Page\b`)

// pdfPages counts page objects; none visible (object streams) falls back to
// size.
func pdfPages(raw []byte) int {
	if n := len(pdfPageObject.FindAllIndex(raw, -1)); n > 0 {
		return n
	}
	return max(1, len(raw)/pdfBytesPerPage)
}

// dataURIBytes decodes a base64 data: URI's payload.
func dataURIBytes(uri string) ([]byte, bool) {
	if !strings.HasPrefix(uri, "data:") {
		return nil, false
	}
	meta, data, ok := strings.Cut(uri[len("data:"):], ",")
	if !ok || !strings.HasSuffix(meta, ";base64") {
		return nil, false
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	return raw, err == nil
}
