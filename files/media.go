package files

// media.go — read on an image, a recording, a video or a PDF.
//
// A file with a zero byte in its head used to be refused as "binary". For the
// formats a multimodal model can take, the bytes ARE the content: a screenshot
// of a failing UI, a recorded bug, a spec as a PDF. read now returns such a
// file as a media part of the tool result (llm.ToolResult.Parts) with a short
// text label, and llm routes the part to where the endpoint takes it. Every
// other binary is refused as before.
//
// The cost is the reason for the policy. A media part in a tool result stays in
// the history, so it rides EVERY later request of the session, not one: a
// 20 MB video is 27 MB of base64 resent each round. So each kind has a byte cap
// (MediaPolicy), an image is also held to a long edge, and a file over a limit
// is shrunk by the first tool found (scale.go) or refused with a label that
// says the user can raise the limit. The host owns the policy and can change it
// while the session runs (SetMediaPolicy); OnMedia tells the host each time a
// limit applied, so a UI can show "the agent saw a 1568 px copy" or offer to
// raise the cap.
//
// Detection is by content, never by extension: a .png that is really a text
// file reads as text, and an extension-less screenshot reads as an image.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/gif" // DecodeConfig / Decode for the stdlib scaler
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

// MediaKind is the class of a media file: it picks the limit and the part type.
type MediaKind string

const (
	MediaImage MediaKind = "image" // png, jpeg, gif, webp; heic, avif, tiff, bmp converted
	MediaAudio MediaKind = "audio" // wav, mp3, flac, ogg, m4a
	MediaVideo MediaKind = "video" // mp4, webm, mov; mkv converted
	MediaPDF   MediaKind = "pdf"   // sent as a file part
)

// MediaMode says whether one kind of media is sent.
type MediaMode string

const (
	// MediaSend (and "") sends the file as a part. The default.
	MediaSend MediaMode = "send"
	// MediaOff sends a text label instead ("media is off for this session"),
	// for a model or endpoint that cannot take the kind, or a host that wants
	// no media spend.
	MediaOff MediaMode = "off"
)

// Defaults for a MediaPolicy field left zero.
const (
	// DefaultMediaMaxBytes is the per-kind byte cap: 5 MiB. Claude's API refuses
	// an image over 5 MB, which sets the image number; the same number holds
	// for the other kinds because of the cost above — 5 MiB is ~6.7 MB of
	// base64 on every later request — and not because an endpoint refuses
	// more. A host that wants a long clip raises Audio/Video.MaxBytes.
	DefaultMediaMaxBytes = 5 << 20
	// DefaultMaxLongEdge is 1568 px: Claude's API scales any image with a
	// longer edge down to it before the model sees it, so the extra pixels
	// are only bytes on the wire.
	DefaultMaxLongEdge = 1568
	// DefaultMediaTimeout bounds ONE run of a scaling tool. A tool that hangs
	// on a hostile or broken file is killed at it and the file is refused.
	DefaultMediaTimeout = 30 * time.Second
)

// MediaLimit is the policy for one kind.
type MediaLimit struct {
	Mode     MediaMode // "" = MediaSend
	MaxBytes int64     // 0 = DefaultMediaMaxBytes
}

// MediaTools pins or turns off the external scaling tools. A pinned path is
// used as is, with no PATH lookup; an empty one is looked up on PATH (once per
// Toolset, cached).
type MediaTools struct {
	Magick        string // ImageMagick 7 `magick`
	Convert       string // ImageMagick 6 `convert` (never looked up on Windows: convert.exe there converts disks)
	FFmpeg        string // `ffmpeg`
	VipsThumbnail string // libvips `vipsthumbnail`
	// None looks up nothing on PATH: images use the stdlib scaler, and audio
	// or video over its cap is refused. A pinned path still wins.
	None bool
}

// MediaIn is what a Scaler is asked to shrink.
type MediaIn struct {
	Path          string // absolute
	MIME          string // sniffed from the content
	Kind          MediaKind
	Bytes         int64
	Width, Height int   // 0 when unknown
	MaxBytes      int64 // the cap the output must fit
	MaxLongEdge   int   // images: the longest edge the output may have
	Alpha         bool  // images: the source may have transparency
	// Attempt is 0 on the first pass and 1 on the one retry, made when the
	// first output was still over MaxBytes: shrink harder.
	Attempt int
}

// MediaOut is a Scaler's result. MIME must be a type the endpoint takes
// (image/jpeg, image/png, audio/mpeg, video/mp4, …). Note is what was done, for
// the label ("mono mp3 64 kbit/s"); optional.
type MediaOut struct {
	Data []byte
	MIME string
	Note string
}

// Scaler shrinks or converts one media file. A host Scaler replaces the
// built-in tools entirely (a sandboxed converter, a remote service). Its error
// refuses the file, with the error in the label.
type Scaler func(ctx context.Context, in MediaIn) (MediaOut, error)

// MediaPolicy is how read treats media files. The zero value is the default:
// every kind sent, 5 MiB caps, 1568 px images, tools from PATH.
type MediaPolicy struct {
	Image, Audio, Video, PDF MediaLimit
	// ScaleAlways re-encodes every image through the scaler, even one under
	// both limits. Off: an image under the limits is sent byte for byte.
	ScaleAlways bool
	// MaxLongEdge caps an image's longer side in pixels; 0 = DefaultMaxLongEdge.
	// An image over it is scaled down (never up).
	MaxLongEdge int
	Tools       MediaTools
	// Scaler, when set, is used instead of the built-in tools.
	Scaler Scaler
	// Timeout bounds one scaling run; 0 = DefaultMediaTimeout.
	Timeout time.Duration
}

func (p MediaPolicy) limit(k MediaKind) MediaLimit {
	var l MediaLimit
	switch k {
	case MediaImage:
		l = p.Image
	case MediaAudio:
		l = p.Audio
	case MediaVideo:
		l = p.Video
	case MediaPDF:
		l = p.PDF
	}
	if l.Mode == "" {
		l.Mode = MediaSend
	}
	if l.MaxBytes <= 0 {
		l.MaxBytes = DefaultMediaMaxBytes
	}
	return l
}

func (p MediaPolicy) edge() int {
	if p.MaxLongEdge <= 0 {
		return DefaultMaxLongEdge
	}
	return p.MaxLongEdge
}

func (p MediaPolicy) timeout() time.Duration {
	if p.Timeout <= 0 {
		return DefaultMediaTimeout
	}
	return p.Timeout
}

// MediaSize is one side of a MediaEvent. Width and Height are 0 when unknown.
type MediaSize struct {
	Bytes         int64
	Width, Height int
}

// MediaEvent reports a read where a media limit applied: the file was over a
// cap or an edge, was scaled or converted, was refused, or its kind is off. A
// file sent unchanged under every limit fires none.
type MediaEvent struct {
	Path    string // as the model named it, root-relative
	MIME    string // sniffed
	Kind    MediaKind
	Before  MediaSize
	After   MediaSize // zero when nothing was sent
	OutMIME string    // what was sent; "" when nothing was
	Scaled  bool      // the bytes sent are a scaled or converted copy
	// Tool made the copy: "magick", "convert", "ffmpeg", "vipsthumbnail",
	// "stdlib", "host", or "none".
	Tool    string
	Refused bool   // nothing was sent (over a limit and not shrunk, or off)
	Reason  string // why, in the words the model got
}

// SetMediaPolicy replaces the media policy. It takes effect on the next read,
// so a host can change it while the session runs (the user raised a cap).
func (t *Toolset) SetMediaPolicy(p MediaPolicy) { t.media.Store(&p) }

// MediaPolicy is the policy in force.
func (t *Toolset) MediaPolicy() MediaPolicy {
	if p := t.media.Load(); p != nil {
		return *p
	}
	return MediaPolicy{}
}

// mediaType is a sniffed format.
type mediaType struct {
	mime   string
	kind   MediaKind
	native bool   // the endpoint takes it as is; false = convert before sending
	format string // input_audio's format, audio only
	coder  string // ImageMagick coder, images only
}

var (
	tPNG  = mediaType{"image/png", MediaImage, true, "", "png"}
	tJPEG = mediaType{"image/jpeg", MediaImage, true, "", "jpeg"}
	tGIF  = mediaType{"image/gif", MediaImage, true, "", "gif"}
	tWEBP = mediaType{"image/webp", MediaImage, true, "", "webp"}
	tHEIC = mediaType{"image/heic", MediaImage, false, "", "heic"}
	tAVIF = mediaType{"image/avif", MediaImage, false, "", "avif"}
	tTIFF = mediaType{"image/tiff", MediaImage, false, "", "tiff"}
	tBMP  = mediaType{"image/bmp", MediaImage, false, "", "bmp"}
	tWAV  = mediaType{"audio/wav", MediaAudio, true, "wav", ""}
	tMP3  = mediaType{"audio/mpeg", MediaAudio, true, "mp3", ""}
	tFLAC = mediaType{"audio/flac", MediaAudio, true, "flac", ""}
	tOGG  = mediaType{"audio/ogg", MediaAudio, true, "ogg", ""}
	tM4A  = mediaType{"audio/mp4", MediaAudio, true, "m4a", ""}
	tMP4  = mediaType{"video/mp4", MediaVideo, true, "", ""}
	tWEBM = mediaType{"video/webm", MediaVideo, true, "", ""}
	tMOV  = mediaType{"video/quicktime", MediaVideo, true, "", ""}
	tMKV  = mediaType{"video/x-matroska", MediaVideo, false, "", ""}
	tPDF  = mediaType{"application/pdf", MediaPDF, true, "", ""}
)

// sniffMedia classifies a file by its first bytes. http.DetectContentType
// knows png, jpeg, gif, webp, bmp, wav, mp3 with an ID3 tag, ogg, webm and pdf;
// the rest is matched here. ftyp is matched FIRST: DetectContentType calls any
// ftyp box with an "mp4*" brand video/mp4, which is also an m4a's compatible
// brand.
func sniffMedia(head []byte) (mediaType, bool) {
	if m, ok := sniffFtyp(head); ok {
		return m, true
	}
	has := func(off int, s string) bool {
		return len(head) >= off+len(s) && string(head[off:off+len(s)]) == s
	}
	switch {
	case has(0, "fLaC"):
		return tFLAC, true
	case has(0, "RIFF") && has(8, "WAVE"):
		return tWAV, true
	case has(0, "RIFF") && has(8, "WEBP"):
		return tWEBP, true
	case has(0, "II*\x00"), has(0, "MM\x00*"):
		return tTIFF, true
	case has(0, "ID3"):
		return tMP3, true
	case len(head) >= 4 && head[0] == 0xFF && head[1]&0xE0 == 0xE0 && head[1]&0x18 != 0x08 && head[1]&0x06 != 0 &&
		head[2]&0xF0 != 0xF0:
		// An MPEG audio frame header with no ID3 tag: 11 sync bits, a version
		// and a layer that are not "reserved", a bitrate that is not "bad".
		// jpeg's FF D8 fails the sync bits.
		return tMP3, true
	case has(0, "OggS"):
		return tOGG, true
	case has(0, "\x1a\x45\xdf\xa3"):
		doc := head[:min(len(head), 64)]
		if bytes.Contains(doc, []byte("webm")) {
			return tWEBM, true
		}
		return tMKV, true
	}
	switch http.DetectContentType(head) {
	case "image/png":
		return tPNG, true
	case "image/jpeg":
		return tJPEG, true
	case "image/gif":
		return tGIF, true
	case "image/bmp":
		return tBMP, true
	case "application/pdf":
		return tPDF, true
	}
	return mediaType{}, false
}

// sniffFtyp reads an ISO base media file's ftyp box: the major brand and the
// compatible brands name the format.
func sniffFtyp(head []byte) (mediaType, bool) {
	if len(head) < 12 || string(head[4:8]) != "ftyp" {
		return mediaType{}, false
	}
	size := int(binary.BigEndian.Uint32(head[:4]))
	if size < 12 || size > len(head) {
		size = len(head)
	}
	brands := []string{string(head[8:12])}
	for i := 16; i+4 <= size; i += 4 {
		brands = append(brands, string(head[i:i+4]))
	}
	is := func(set ...string) bool {
		for _, b := range brands {
			for _, s := range set {
				if b == s {
					return true
				}
			}
		}
		return false
	}
	switch {
	case is("avif", "avis"):
		return tAVIF, true
	case is("heic", "heix", "hevc", "hevx", "heim", "heis", "mif1", "msf1"):
		return tHEIC, true
	case brands[0] == "M4A " || brands[0] == "M4B ":
		return tM4A, true
	case brands[0] == "qt  ":
		return tMOV, true
	}
	return tMP4, true
}

// imageDims reads an image's size without decoding its pixels: the stdlib
// decoders' config for png, jpeg and gif, the header for webp. 0, 0 when
// unknown.
func imageDims(abs string, m mediaType, head []byte) (int, int) {
	switch m {
	case tPNG, tJPEG, tGIF:
		f, err := os.Open(abs)
		if err != nil {
			return 0, 0
		}
		defer func() { _ = f.Close() }()
		c, _, err := image.DecodeConfig(bufio.NewReader(f))
		if err != nil {
			return 0, 0
		}
		return c.Width, c.Height
	case tWEBP:
		return webpDims(head)
	}
	return 0, 0
}

// webpDims parses the first chunk of a WebP: VP8X (extended), VP8L (lossless)
// or "VP8 " (lossy).
func webpDims(b []byte) (int, int) {
	if len(b) < 30 {
		return 0, 0
	}
	u24 := func(p []byte) int { return int(p[0]) | int(p[1])<<8 | int(p[2])<<16 }
	switch string(b[12:16]) {
	case "VP8X":
		return 1 + u24(b[24:27]), 1 + u24(b[27:30])
	case "VP8L":
		if b[20] != 0x2f {
			return 0, 0
		}
		v := binary.LittleEndian.Uint32(b[21:25])
		return 1 + int(v&0x3FFF), 1 + int((v>>14)&0x3FFF)
	case "VP8 ":
		return int(binary.LittleEndian.Uint16(b[26:28]) & 0x3FFF), int(binary.LittleEndian.Uint16(b[28:30]) & 0x3FFF)
	}
	return 0, 0
}

// alphaHint says whether an image MAY have transparency, from its header, so
// an external tool is asked for png (keeps it) instead of jpeg (drops it).
// png: a color type with alpha or a tRNS chunk; webp: the VP8X alpha flag or
// lossless; gif: a transparent color in the head. Unknown formats: no.
func alphaHint(m mediaType, head []byte) bool {
	switch m {
	case tPNG:
		if len(head) > 25 && (head[25] == 4 || head[25] == 6) {
			return true
		}
		return bytes.Contains(head, []byte("tRNS"))
	case tWEBP:
		if len(head) > 20 && string(head[12:16]) == "VP8X" {
			return head[20]&0x10 != 0
		}
		return len(head) > 16 && string(head[12:16]) == "VP8L"
	case tGIF:
		for i := 0; i+3 < len(head); i++ {
			if head[i] == 0x21 && head[i+1] == 0xF9 && head[i+2] == 0x04 && head[i+3]&1 == 1 {
				return true
			}
		}
	}
	return false
}

// readMedia is read for a sniffed media file: the part, a label, and an event
// when a limit applied.
func (t *Toolset) readMedia(ctx context.Context, abs, display string, size int64, head []byte, m mediaType) (llm.ToolResult, error) {
	p := t.MediaPolicy()
	lim := p.limit(m.kind)
	ev := MediaEvent{Path: display, MIME: m.mime, Kind: m.kind, Before: MediaSize{Bytes: size}, Tool: "none"}
	if m.kind == MediaImage {
		ev.Before.Width, ev.Before.Height = imageDims(abs, m, head)
	}
	desc := describe(m.mime, ev.Before)

	refuse := func(reason string) (llm.ToolResult, error) {
		ev.Refused, ev.Reason = true, reason
		t.emitMedia(ev)
		return llm.ToolResult{}, fmt.Errorf("%s: %s; %s", display, desc, reason)
	}
	if lim.Mode == MediaOff {
		ev.Refused = true
		ev.Reason = fmt.Sprintf("%s not sent: %s media is off for this session", m.mime, m.kind)
		t.emitMedia(ev)
		return llm.TextResult(fmt.Sprintf("%s: %s; %s", display, desc, ev.Reason)), nil
	}

	overBytes := size > lim.MaxBytes
	longEdge := max(ev.Before.Width, ev.Before.Height)
	overEdge := m.kind == MediaImage && longEdge > p.edge()
	capNote := fmt.Sprintf("the %s %s limit for this session", sizeOf(lim.MaxBytes), m.kind)
	if m.kind == MediaPDF && overBytes {
		return refuse(fmt.Sprintf("%s is %s, over %s (a PDF is not shrunk); the user can raise it", m.mime, sizeOf(size), capNote))
	}
	need := !m.native || overBytes || overEdge || (m.kind == MediaImage && p.ScaleAlways)
	if !need {
		data, err := readCapped(abs, lim.MaxBytes)
		if err != nil {
			return llm.ToolResult{}, err
		}
		return llm.ToolResult{Text: display + ": " + desc, Parts: []llm.ContentPart{mediaPart(m, display, m.mime, data)}}, nil
	}

	in := MediaIn{Path: abs, MIME: m.mime, Kind: m.kind, Bytes: size, Width: ev.Before.Width, Height: ev.Before.Height,
		MaxBytes: lim.MaxBytes, MaxLongEdge: p.edge(), Alpha: alphaHint(m, head)}
	out, tool, err := t.scale(ctx, p, m, in)
	ev.Tool = tool
	if err != nil {
		if m.native && !overBytes {
			// Over the edge only (or ScaleAlways): the original is within the
			// byte cap, so send it rather than nothing.
			data, rerr := readCapped(abs, lim.MaxBytes)
			if rerr != nil {
				return llm.ToolResult{}, rerr
			}
			ev.Reason = fmt.Sprintf("sent unscaled: %v", err)
			ev.After, ev.OutMIME = ev.Before, m.mime
			t.emitMedia(ev)
			return llm.ToolResult{Text: fmt.Sprintf("%s: %s (%s)", display, desc, ev.Reason),
				Parts: []llm.ContentPart{mediaPart(m, display, m.mime, data)}}, nil
		}
		switch {
		case overBytes:
			return refuse(fmt.Sprintf("%s is %s, over %s, and could not be shrunk (%v); the user can raise the limit",
				m.mime, sizeOf(size), capNote, err))
		default:
			return refuse(fmt.Sprintf("%s is not sent as is (many endpoints refuse it) and could not be converted (%v)", m.mime, err))
		}
	}
	ev.Scaled, ev.OutMIME = true, out.MIME
	ev.After = MediaSize{Bytes: int64(len(out.Data))}
	if strings.HasPrefix(out.MIME, "image/") {
		if c, _, err := image.DecodeConfig(bytes.NewReader(out.Data)); err == nil {
			ev.After.Width, ev.After.Height = c.Width, c.Height
		}
	}
	why := "converted"
	switch {
	case overBytes && overEdge:
		why = fmt.Sprintf("scaled to fit the %d px edge and %s", p.edge(), capNote)
	case overBytes:
		why = "shrunk to fit " + capNote
	case overEdge:
		why = fmt.Sprintf("scaled to fit the %d px edge", p.edge())
	case m.native:
		why = "re-encoded (scale always)"
	}
	ev.Reason = fmt.Sprintf("%s by %s", why, tool)
	if out.Note != "" {
		ev.Reason += ": " + out.Note
	}
	t.emitMedia(ev)
	om := m
	if out.MIME != m.mime {
		om = outType(out.MIME)
	}
	label := fmt.Sprintf("%s: %s → %s (%s)", display, desc, describe(out.MIME, ev.After), ev.Reason)
	return llm.ToolResult{Text: label, Parts: []llm.ContentPart{mediaPart(om, display, out.MIME, out.Data)}}, nil
}

// outType is the mediaType of a scaler's output MIME, for the part it builds.
func outType(mime string) mediaType {
	for _, m := range []mediaType{tPNG, tJPEG, tGIF, tWEBP, tWAV, tMP3, tFLAC, tOGG, tM4A, tMP4, tWEBM, tMOV, tPDF} {
		if m.mime == mime {
			return m
		}
	}
	switch {
	case strings.HasPrefix(mime, "image/"):
		return mediaType{mime: mime, kind: MediaImage}
	case strings.HasPrefix(mime, "audio/"):
		return mediaType{mime: mime, kind: MediaAudio, format: strings.TrimPrefix(mime, "audio/")}
	case strings.HasPrefix(mime, "video/"):
		return mediaType{mime: mime, kind: MediaVideo}
	}
	return mediaType{mime: mime, kind: MediaPDF}
}

// mediaPart builds the content part for one kind: image_url and video_url
// take a data: URI, input_audio BARE base64 plus a format, a PDF a file part.
func mediaPart(m mediaType, display, mime string, data []byte) llm.ContentPart {
	switch m.kind {
	case MediaImage:
		return llm.ImageData(mime, data)
	case MediaAudio:
		return llm.AudioData(m.format, data)
	case MediaVideo:
		return llm.VideoData(mime, data)
	}
	return llm.FileData(mime, filepath.Base(display), data)
}

// describe is the label's body: "image/png, 1920×1080, 402.3KB".
func describe(mime string, s MediaSize) string {
	if s.Width > 0 && s.Height > 0 {
		return fmt.Sprintf("%s, %d×%d, %s", mime, s.Width, s.Height, sizeOf(s.Bytes))
	}
	return fmt.Sprintf("%s, %s", mime, sizeOf(s.Bytes))
}

func (t *Toolset) emitMedia(ev MediaEvent) {
	if t.cfg.OnMedia != nil {
		t.cfg.OnMedia(ev)
	}
}

// readCapped reads a whole file of at most limit bytes; a file that grew past
// it since the stat is refused rather than read without bound.
func readCapped(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%s is over %s", filepath.Base(path), sizeOf(limit))
	}
	return data, nil
}
