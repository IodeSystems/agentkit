package files

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// noise is an image that does not compress, so a byte cap bites.
func noise(w, h int, alpha bool) *image.NRGBA {
	r := rand.New(rand.NewSource(1))
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(m.Pix); i += 4 {
		m.Pix[i], m.Pix[i+1], m.Pix[i+2] = uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256))
		m.Pix[i+3] = 255
		if alpha && i%8 == 0 {
			m.Pix[i+3] = 0
		}
	}
	return m
}

func pngBytes(t *testing.T, m image.Image) string {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, m); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

func jpegBytes(t *testing.T, m image.Image) string {
	t.Helper()
	var b bytes.Buffer
	if err := jpeg.Encode(&b, m, nil); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

func gifBytes(t *testing.T) string {
	t.Helper()
	p := image.NewPaletted(image.Rect(0, 0, 4, 3), []color.Color{color.Black, color.White})
	var b bytes.Buffer
	if err := gif.Encode(&b, p, nil); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

// webpHeader is a VP8X WebP header for w×h, padded to n bytes: enough for the
// sniff and the size, not a decodable image.
func webpHeader(w, h, n int) string {
	b := make([]byte, max(n, 30))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	copy(b[8:], "WEBPVP8X")
	binary.LittleEndian.PutUint32(b[16:], 10)
	put24 := func(p []byte, v int) { p[0], p[1], p[2] = byte(v), byte(v>>8), byte(v>>16) }
	put24(b[24:], w-1)
	put24(b[27:], h-1)
	return string(b)
}

func wavBytes(samples int) string {
	var b bytes.Buffer
	data := samples * 2
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(36+data))
	b.WriteString("WAVEfmt ")
	_ = binary.Write(&b, binary.LittleEndian, []any{uint32(16), uint16(1), uint16(1), uint32(8000), uint32(16000), uint16(2), uint16(16)})
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, uint32(data))
	r := rand.New(rand.NewSource(2))
	for i := 0; i < samples; i++ {
		_ = binary.Write(&b, binary.LittleEndian, int16(r.Intn(20000)-10000))
	}
	return b.String()
}

func ftyp(major string, compat ...string) string {
	body := major + "\x00\x00\x02\x00" + strings.Join(compat, "")
	var sz [4]byte
	binary.BigEndian.PutUint32(sz[:], uint32(8+len(body)))
	return string(sz[:]) + "ftyp" + body + strings.Repeat("\x00", 64)
}

// events collects OnMedia events.
type events struct {
	mu sync.Mutex
	ev []MediaEvent
}

func (e *events) on(ev MediaEvent) { e.mu.Lock(); e.ev = append(e.ev, ev); e.mu.Unlock() }
func (e *events) list() []MediaEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]MediaEvent(nil), e.ev...)
}

func readRes(t *testing.T, ts *Toolset, path string) llm.ToolResult {
	t.Helper()
	b, _ := json.Marshal(map[string]any{"path": path})
	return ts.Dispatch(context.Background(), ts.Name(Read), string(b))
}

// partBytes decodes the payload of a media part.
func partBytes(t *testing.T, p llm.ContentPart) (mime string, data []byte) {
	t.Helper()
	var uri string
	switch {
	case p.ImageURL != nil:
		uri = p.ImageURL.URL
	case p.VideoURL != nil:
		uri = p.VideoURL.URL
	case p.File != nil:
		uri = p.File.FileData
	case p.InputAudio != nil:
		d, err := base64.StdEncoding.DecodeString(p.InputAudio.Data)
		if err != nil {
			t.Fatal(err)
		}
		return "audio/" + p.InputAudio.Format, d
	default:
		t.Fatalf("not a media part: %+v", p)
	}
	head, b64, ok := strings.Cut(uri, ";base64,")
	if !ok || !strings.HasPrefix(head, "data:") {
		t.Fatalf("not a data URI: %.40s", uri)
	}
	d, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimPrefix(head, "data:"), d
}

func TestReadMediaDetectsEachKindAndSendsItsPart(t *testing.T) {
	small := noise(40, 30, false)
	root := tree(t, map[string]string{
		"shot.png":  pngBytes(t, small),
		"photo":     jpegBytes(t, small), // no extension: sniffed, not guessed
		"anim.gif":  gifBytes(t),
		"pic.webp":  webpHeader(640, 480, 200),
		"doc.pdf":   "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\n\x00trailer\n",
		"clip.wav":  wavBytes(100),
		"clip.mp4":  ftyp("isom", "isom", "iso2", "mp41"),
		"song.m4a":  ftyp("M4A ", "M4A ", "mp42", "isom"),
		"take.mov":  ftyp("qt  ", "qt  "),
		"clip.webm": "\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\x82\x84webm" + strings.Repeat("\x00", 40),
		"tune.mp3":  "\xff\xfb\x90\x00" + strings.Repeat("\x00", 100),
		"tag.mp3":   "ID3\x03\x00\x00\x00\x00\x00\x00" + strings.Repeat("\x00", 100),
		"a.flac":    "fLaC\x00\x00\x00\x22" + strings.Repeat("\x00", 40),
		"a.ogg":     "OggS\x00\x02" + strings.Repeat("\x00", 40),
	})
	ts := mount(t, Config{Root: root})
	cases := []struct {
		path, partType, mime, label string
	}{
		{"shot.png", "image_url", "image/png", "shot.png: image/png, 40×30, "},
		{"photo", "image_url", "image/jpeg", "photo: image/jpeg, 40×30, "},
		{"anim.gif", "image_url", "image/gif", "anim.gif: image/gif, 4×3, "},
		{"pic.webp", "image_url", "image/webp", "pic.webp: image/webp, 640×480, 200 bytes"},
		{"doc.pdf", "file", "application/pdf", "doc.pdf: application/pdf, "},
		{"clip.wav", "input_audio", "audio/wav", "clip.wav: audio/wav, "},
		{"clip.mp4", "video_url", "video/mp4", "clip.mp4: video/mp4, "},
		{"song.m4a", "input_audio", "audio/m4a", "song.m4a: audio/mp4, "},
		{"take.mov", "video_url", "video/quicktime", "take.mov: video/quicktime, "},
		{"clip.webm", "video_url", "video/webm", "clip.webm: video/webm, "},
		{"tune.mp3", "input_audio", "audio/mp3", "tune.mp3: audio/mpeg, "},
		{"tag.mp3", "input_audio", "audio/mp3", "tag.mp3: audio/mpeg, "},
		{"a.flac", "input_audio", "audio/flac", "a.flac: audio/flac, "},
		{"a.ogg", "input_audio", "audio/ogg", "a.ogg: audio/ogg, "},
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			res := readRes(t, ts, c.path)
			if strings.HasPrefix(res.Text, "ERROR") || len(res.Parts) != 1 {
				t.Fatalf("want one part, got %q with %d parts", res.Text, len(res.Parts))
			}
			contains(t, res.Text, c.label)
			p := res.Parts[0]
			if p.Type != c.partType {
				t.Fatalf("part type %q, want %q", p.Type, c.partType)
			}
			mime, data := partBytes(t, p)
			if mime != c.mime {
				t.Errorf("part MIME %q, want %q", mime, c.mime)
			}
			if string(data) != readFile(t, filepath.Join(root, c.path)) {
				t.Error("a file under every limit was not sent byte for byte")
			}
			if p.File != nil && p.File.Filename != "doc.pdf" {
				t.Errorf("pdf filename %q", p.File.Filename)
			}
		})
	}
}

func TestReadMediaSniffsFtypBrandsThatDetectContentTypeCallsMP4(t *testing.T) {
	for _, c := range []struct {
		head string
		want mediaType
	}{
		{ftyp("heic", "mif1", "heic"), tHEIC},
		{ftyp("mif1", "mif1", "avif"), tAVIF},
		{ftyp("M4A ", "mp42"), tM4A},
		{ftyp("mp42", "mp42", "isom"), tMP4},
		{"II*\x00\x08\x00\x00\x00", tTIFF},
		{"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x88matroska", tMKV},
	} {
		got, ok := sniffMedia([]byte(c.head))
		if !ok || got != c.want {
			t.Errorf("%q: got %v %v, want %s", c.head[:12], got.mime, ok, c.want.mime)
		}
	}
	for _, s := range []string{"plain text\n", "\x7fELF\x02\x01", "PK\x03\x04", "%PDX", "\xff\xd8\xff"} {
		if m, ok := sniffMedia([]byte(s)); ok && m != tJPEG {
			t.Errorf("%q sniffed as %s", s, m.mime)
		}
	}
}

func TestReadMediaOffSendsALabelAndAnEvent(t *testing.T) {
	root := tree(t, map[string]string{"shot.png": pngBytes(t, noise(10, 10, false))})
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{Image: MediaLimit{Mode: MediaOff}}})
	res := readRes(t, ts, "shot.png")
	if len(res.Parts) != 0 || strings.HasPrefix(res.Text, "ERROR") {
		t.Fatalf("off sent a part or refused: %q %d", res.Text, len(res.Parts))
	}
	contains(t, res.Text, "shot.png: image/png, 10×10", "image/png not sent: image media is off for this session")
	got := ev.list()
	if len(got) != 1 || !got[0].Refused || got[0].Tool != "none" || got[0].Kind != MediaImage {
		t.Fatalf("events %+v", got)
	}
}

func TestReadMediaPolicyChangeTakesEffectOnTheNextRead(t *testing.T) {
	root := tree(t, map[string]string{"shot.png": pngBytes(t, noise(10, 10, false))})
	ts := mount(t, Config{Root: root})
	if res := readRes(t, ts, "shot.png"); len(res.Parts) != 1 {
		t.Fatalf("default policy did not send the image: %q", res.Text)
	}
	ts.SetMediaPolicy(MediaPolicy{Image: MediaLimit{Mode: MediaOff}})
	if res := readRes(t, ts, "shot.png"); len(res.Parts) != 0 {
		t.Fatal("policy set to off still sent the image")
	}
	if ts.MediaPolicy().Image.Mode != MediaOff {
		t.Fatal("MediaPolicy does not return what was set")
	}
	ts.SetMediaPolicy(MediaPolicy{})
	if res := readRes(t, ts, "shot.png"); len(res.Parts) != 1 {
		t.Fatal("policy set back did not send the image")
	}
}

func TestReadMediaOverCapStdlibScalesToJPEG(t *testing.T) {
	root := tree(t, map[string]string{"big.png": pngBytes(t, noise(2000, 1000, false))})
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{Tools: MediaTools{None: true}}})
	res := readRes(t, ts, "big.png")
	if len(res.Parts) != 1 {
		t.Fatalf("no part: %s", res.Text)
	}
	mime, data := partBytes(t, res.Parts[0])
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || format != "jpeg" || mime != "image/jpeg" {
		t.Fatalf("sent %s/%s: %v", mime, format, err)
	}
	if cfg.Width != 1568 || cfg.Height != 784 {
		t.Errorf("scaled to %d×%d, want 1568×784", cfg.Width, cfg.Height)
	}
	contains(t, res.Text, "big.png: image/png, 2000×1000", "→ image/jpeg, 1568×784", "by stdlib")
	got := ev.list()
	if len(got) != 1 || !got[0].Scaled || got[0].Tool != "stdlib" || got[0].Refused ||
		got[0].After.Width != 1568 || got[0].Before.Bytes <= got[0].After.Bytes {
		t.Fatalf("event %+v", got)
	}
}

func TestReadMediaAlphaPNGStaysPNGWhenScaled(t *testing.T) {
	root := tree(t, map[string]string{"ui.png": pngBytes(t, noise(1800, 900, true))})
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Tools: MediaTools{None: true}}})
	res := readRes(t, ts, "ui.png")
	if len(res.Parts) != 1 {
		t.Fatalf("no part: %s", res.Text)
	}
	mime, data := partBytes(t, res.Parts[0])
	if _, format, err := image.DecodeConfig(bytes.NewReader(data)); err != nil || format != "png" || mime != "image/png" {
		t.Fatalf("an image with transparency was sent as %s/%s (jpeg turns it black): %v", mime, format, err)
	}
}

func TestReadMediaRetriesOnceThenRefusesOverTheCap(t *testing.T) {
	root := tree(t, map[string]string{"big.png": pngBytes(t, noise(1000, 800, false))})
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{
		Image: MediaLimit{MaxBytes: 200}, Tools: MediaTools{None: true}}})
	out := readRes(t, ts, "big.png").Text
	mustErr(t, out, "over the 200 bytes image limit", "still over", "the user can raise the limit")
	if got := ev.list(); len(got) != 1 || !got[0].Refused || got[0].Tool != "stdlib" {
		t.Fatalf("events %+v", got)
	}
}

func TestReadMediaWebPWithoutAToolPassesUnderCapAndIsRefusedOver(t *testing.T) {
	root := tree(t, map[string]string{"small.webp": webpHeader(800, 600, 2000), "big.webp": webpHeader(800, 600, 9000)})
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{
		Image: MediaLimit{MaxBytes: 4000}, Tools: MediaTools{None: true}}})
	if res := readRes(t, ts, "small.webp"); len(res.Parts) != 1 {
		t.Fatalf("a webp under the cap was not passed through: %s", res.Text)
	}
	if n := len(ev.list()); n != 0 {
		t.Fatalf("a passthrough fired %d events", n)
	}
	mustErr(t, readRes(t, ts, "big.webp").Text, "image/webp is 8.8KB, over the 3.9KB image limit",
		"no decoder for image/webp", "the user can raise the limit")
	if got := ev.list(); len(got) != 1 || !got[0].Refused || got[0].Tool != "none" {
		t.Fatalf("events %+v", got)
	}
}

func TestReadMediaHEICWithoutAToolIsRefusedEvenUnderTheCap(t *testing.T) {
	root := tree(t, map[string]string{"p.heic": ftyp("heic", "mif1", "heic")})
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Tools: MediaTools{None: true}}})
	mustErr(t, readRes(t, ts, "p.heic").Text, "image/heic is not sent as is", "no decoder for image/heic")
}

func TestReadMediaVideoOverCapWithoutFFmpegIsRefused(t *testing.T) {
	root := tree(t, map[string]string{"clip.mp4": ftyp("isom", "isom") + strings.Repeat("\x00", 3000)})
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Video: MediaLimit{MaxBytes: 1000}, Tools: MediaTools{None: true}}})
	mustErr(t, readRes(t, ts, "clip.mp4").Text, "video/mp4 is 3.0KB, over the 1000 bytes video limit for this session",
		"no ffmpeg on PATH", "the user can raise the limit")
}

func TestReadMediaPDFOverCapIsRefusedNotShrunk(t *testing.T) {
	root := tree(t, map[string]string{"doc.pdf": "%PDF-1.4\n" + strings.Repeat("x", 3000)})
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{PDF: MediaLimit{MaxBytes: 1000}}})
	mustErr(t, readRes(t, ts, "doc.pdf").Text, "a PDF is not shrunk", "the user can raise it")
}

func TestReadMediaHostScalerWinsOverTools(t *testing.T) {
	root := tree(t, map[string]string{"big.png": pngBytes(t, noise(2000, 100, false))})
	small := jpegBytes(t, noise(20, 1, false))
	var ev events
	var calls []MediaIn
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{
		Tools: MediaTools{Magick: "/nonexistent/magick"}, // would fail if run
		Scaler: func(ctx context.Context, in MediaIn) (MediaOut, error) {
			calls = append(calls, in)
			if _, ok := ctx.Deadline(); !ok {
				t.Error("host Scaler got no deadline")
			}
			return MediaOut{Data: []byte(small), MIME: "image/jpeg", Note: "host did it"}, nil
		}}})
	res := readRes(t, ts, "big.png")
	if len(res.Parts) != 1 {
		t.Fatalf("no part: %s", res.Text)
	}
	if len(calls) != 1 || calls[0].MaxLongEdge != 1568 || calls[0].Width != 2000 || calls[0].MIME != "image/png" {
		t.Fatalf("scaler calls %+v", calls)
	}
	contains(t, res.Text, "by host: host did it")
	if got := ev.list(); len(got) != 1 || got[0].Tool != "host" || !got[0].Scaled {
		t.Fatalf("events %+v", got)
	}
}

func TestReadMediaOverEdgeOnlyIsSentUnscaledWhenScalingFails(t *testing.T) {
	root := tree(t, map[string]string{"wide.png": pngBytes(t, noise(2000, 10, false))})
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{
		Scaler: func(context.Context, MediaIn) (MediaOut, error) { return MediaOut{}, os.ErrPermission }}})
	res := readRes(t, ts, "wide.png")
	if len(res.Parts) != 1 {
		t.Fatalf("an image under the byte cap was dropped because scaling failed: %s", res.Text)
	}
	contains(t, res.Text, "sent unscaled")
	if got := ev.list(); len(got) != 1 || got[0].Scaled || got[0].Refused {
		t.Fatalf("events %+v", got)
	}
}

func TestReadMediaTextFileWithImageExtensionReadsAsText(t *testing.T) {
	root := tree(t, map[string]string{"fake.png": "not an image\n"})
	ts := mount(t, Config{Root: root})
	res := readRes(t, ts, "fake.png")
	if len(res.Parts) != 0 {
		t.Fatal("a text file was sent as media because of its name")
	}
	contains(t, res.Text, "1\tnot an image")
}
