package agent

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// wav builds a RIFF/WAVE file of secs seconds at byteRate.
func wav(secs, byteRate int) []byte {
	data := secs * byteRate
	b := []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
	fmtc := make([]byte, 4+16)
	binary.LittleEndian.PutUint32(fmtc[0:4], 16)
	binary.LittleEndian.PutUint16(fmtc[4:6], 1) // PCM
	binary.LittleEndian.PutUint16(fmtc[6:8], 1) // mono
	binary.LittleEndian.PutUint32(fmtc[8:12], uint32(byteRate/2))
	binary.LittleEndian.PutUint32(fmtc[12:16], uint32(byteRate))
	binary.LittleEndian.PutUint16(fmtc[16:18], 2)
	binary.LittleEndian.PutUint16(fmtc[18:20], 16)
	b = append(b, fmtc...)
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, uint32(data))
	return append(b, make([]byte, data)...)
}

func pdfWithPages(n int) []byte {
	var sb strings.Builder
	sb.WriteString("%PDF-1.4\n1 0 obj << /Type /Pages /Count 3 >> endobj\n")
	for i := 0; i < n; i++ {
		sb.WriteString("2 0 obj << /Type /Page /Parent 1 0 R >> endobj\n")
	}
	return []byte(sb.String())
}

// A PDF counts every page at the top of the per-page range plus its image;
// the page TREE ("/Type /Pages") is not a page.
func TestPartTokens_PDFCountsPagesNotThePageTree(t *testing.T) {
	got := partTokens(CharsByThree{}, llm.PDFData("a.pdf", pdfWithPages(3)))
	if want := 3 * PDFPageTokenEstimate; got != want {
		t.Fatalf("tokens = %d, want %d", got, want)
	}
}

// A PDF whose page objects sit in compressed object streams shows none; it
// must still count by size, never zero.
func TestPartTokens_PDFWithNoVisiblePagesCountsBySize(t *testing.T) {
	if got := partTokens(CharsByThree{}, llm.PDFData("", []byte("%PDF-1.5 compressed"))); got != PDFPageTokenEstimate {
		t.Fatalf("small = %d, want one page", got)
	}
	big := make([]byte, 5*pdfBytesPerPage)
	if got := partTokens(CharsByThree{}, llm.PDFData("", big)); got != 5*PDFPageTokenEstimate {
		t.Fatalf("big = %d, want five pages", got)
	}
}

// A text file is its text.
func TestPartTokens_TextFileIsItsText(t *testing.T) {
	text := strings.Repeat("x", 300)
	if got := partTokens(CharsByThree{}, llm.FileData("text/plain", "n.txt", []byte(text))); got != 100 {
		t.Fatalf("tokens = %d, want 100", got)
	}
}

// A file the client cannot size (by id, or an unknown type) is not free.
func TestPartTokens_UnsizedFileIsOnePage(t *testing.T) {
	for name, p := range map[string]llm.ContentPart{
		"file id": {Type: "file", File: &llm.FileRef{FileID: "file-1"}},
		"docx":    llm.FileData("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "", []byte{1}),
	} {
		if got := partTokens(CharsByThree{}, p); got != PDFPageTokenEstimate {
			t.Errorf("%s = %d, want %d", name, got, PDFPageTokenEstimate)
		}
	}
}

// A WAV's duration comes from its header, not its size.
func TestPartTokens_WAVDurationFromHeader(t *testing.T) {
	got := partTokens(CharsByThree{}, llm.AudioData("wav", wav(2, 32000)))
	if want := 2 * AudioTokensPerSecond; got != want {
		t.Fatalf("tokens = %d, want %d", got, want)
	}
}

// A streamed WAV says its data chunk is 0xFFFFFFFF bytes; counting that would
// make a 1 s clip cost a day of audio.
func TestPartTokens_StreamedWAVCountsTheBytesPresent(t *testing.T) {
	b := wav(1, 32000)
	binary.LittleEndian.PutUint32(b[40:44], 0xFFFFFFFF)
	if got := partTokens(CharsByThree{}, llm.AudioData("wav", b)); got != AudioTokensPerSecond {
		t.Fatalf("tokens = %d, want %d", got, AudioTokensPerSecond)
	}
}

// Other containers count by size at 128 kbps.
func TestPartTokens_MP3DurationFromSize(t *testing.T) {
	got := partTokens(CharsByThree{}, llm.AudioData("mp3", make([]byte, 10*audioBytesPerSecond)))
	if want := 10 * AudioTokensPerSecond; got != want {
		t.Fatalf("tokens = %d, want %d", got, want)
	}
}

// Video by bytes at 1 Mbps; by URL, a fixed minute.
func TestPartTokens_Video(t *testing.T) {
	got := partTokens(CharsByThree{}, llm.VideoData("video/mp4", make([]byte, 4*videoBytesPerSecond)))
	if want := 4 * VideoTokensPerSecond; got != want {
		t.Fatalf("data = %d, want %d", got, want)
	}
	if got := partTokens(CharsByThree{}, llm.VideoPart("https://x/v.mp4")); got != VideoURLTokenEstimate {
		t.Fatalf("url = %d, want %d", got, VideoURLTokenEstimate)
	}
}

// Every media part goes through messageTokens, so the Shaper and the Session
// count it the same way.
func TestMessageTokens_CountsEveryMediaKind(t *testing.T) {
	m := llm.Message{Role: "user", Parts: []llm.ContentPart{
		llm.TextPart("x"),
		llm.PDFData("a.pdf", pdfWithPages(1)),
		llm.AudioData("wav", wav(1, 32000)),
		llm.VideoPart("https://x/v.mp4"),
	}}
	got := messageTokens(CharsByThree{}, m)
	if min := PDFPageTokenEstimate + AudioTokensPerSecond + VideoURLTokenEstimate; got < min {
		t.Fatalf("tokens = %d, want >= %d", got, min)
	}
	if (&Shaper{Estimate: CharsByThree{}}).tokens([]llm.Message{m}) != got {
		t.Fatal("Shaper.tokens and messageTokens disagree")
	}
}
