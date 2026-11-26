package files

// The external scaling tools are tested two ways. A FAKE tool — this test
// binary re-executed under the tool's name (the os/exec TestHelperProcess
// pattern, as mcpmgr/fakeserver_test.go does) — records the argv it got, so
// the safety argv is asserted on every machine. The REAL tools run only when
// they are on PATH, and skip otherwise.

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/jpeg"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

const (
	fakeToolEnv  = "FILES_FAKE_TOOL" // scenario: "stdout", "lastarg", "hang"
	fakeToolArgv = "FILES_FAKE_ARGV" // where the fake writes what it got
)

func TestMain(m *testing.M) {
	if s := os.Getenv(fakeToolEnv); s != "" {
		fakeToolMain(s)
		return
	}
	os.Exit(m.Run())
}

// fakeRecord is what one fake run saw.
type fakeRecord struct {
	Argv    []string
	OutMode string // the mode of the last argument's file, when it exists
}

func fakeToolMain(scenario string) {
	rec := fakeRecord{Argv: os.Args}
	last := os.Args[len(os.Args)-1]
	if st, err := os.Stat(last); err == nil {
		rec.OutMode = st.Mode().Perm().String()
	}
	if p := os.Getenv(fakeToolArgv); p != "" {
		b, _ := json.Marshal(rec)
		_ = os.WriteFile(p, b, 0o600)
	}
	var out bytes.Buffer
	_ = jpeg.Encode(&out, image.NewGray(image.Rect(0, 0, 8, 4)), nil)
	switch scenario {
	case "stdout":
		_, _ = os.Stdout.Write(out.Bytes())
	case "lastarg":
		_ = os.WriteFile(last, out.Bytes(), 0o600)
	case "hang":
		time.Sleep(time.Minute)
	}
	os.Exit(0)
}

// fakeTool puts this test binary on a fresh PATH under each name, and returns
// the dir and the file the fake records its argv to.
func fakeTool(t *testing.T, scenario string, names ...string) (dir, record string) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir = t.TempDir()
	for _, n := range names {
		if err := os.Symlink(self, filepath.Join(dir, n)); err != nil {
			t.Skip("no symlinks here:", err)
		}
	}
	record = filepath.Join(t.TempDir(), "argv.json")
	t.Setenv("PATH", dir)
	t.Setenv(fakeToolEnv, scenario)
	t.Setenv(fakeToolArgv, record)
	return dir, record
}

func readRecord(t *testing.T, path string) fakeRecord {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the fake tool did not run: %v", err)
	}
	var r fakeRecord
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestScaleMagickArgvIsSafeAndNoShellRuns(t *testing.T) {
	// A name a shell would act on, and one that starts with "-".
	names := []string{"a b;$(touch pwned)`id`.png", "-resize.png"}
	files := map[string]string{}
	for _, n := range names {
		files[n] = pngBytes(t, noise(2000, 50, false))
	}
	root := tree(t, files)
	dir, record := fakeTool(t, "stdout", "magick", "convert", "ffmpeg", "vipsthumbnail")
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on})
	for _, n := range names {
		res := readRes(t, ts, n)
		if len(res.Parts) != 1 {
			t.Fatalf("%s: no part: %s", n, res.Text)
		}
		r := readRecord(t, record)
		if r.Argv[0] != filepath.Join(dir, "magick") {
			t.Fatalf("ran %q, want magick first on PATH (and no shell)", r.Argv[0])
		}
		want := []string{"-limit", "memory", "256MiB", "-limit", "map", "512MiB", "-limit", "time", "30",
			"-limit", "disk", "1GiB", "png:" + filepath.Join(ts.Root(), n) + "[0]",
			"-auto-orient", "-resize", "1568x1568>", "-strip", "-quality", "85", "jpeg:-"}
		if !slices.Equal(r.Argv[1:], want) {
			t.Errorf("argv\n got %q\nwant %q", r.Argv[1:], want)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "pwned")); err == nil {
		t.Fatal("a file name was run through a shell")
	}
	if got := ev.list(); len(got) != 2 || got[0].Tool != "magick" || !got[0].Scaled {
		t.Fatalf("events %+v", got)
	}
}

func TestScaleConvertIsUsedWhenMagickIsAbsent(t *testing.T) {
	root := tree(t, map[string]string{"x.png": pngBytes(t, noise(2000, 50, false))})
	dir, record := fakeTool(t, "stdout", "convert", "ffmpeg")
	ts := mount(t, Config{Root: root})
	if res := readRes(t, ts, "x.png"); len(res.Parts) != 1 {
		t.Fatalf("no part: %s", res.Text)
	}
	if r := readRecord(t, record); r.Argv[0] != filepath.Join(dir, "convert") {
		t.Fatalf("ran %s", r.Argv[0])
	}
}

func TestScaleFFmpegArgvWhitelistsFileAndWritesA0600TempFile(t *testing.T) {
	root := tree(t, map[string]string{"clip.mp4": ftyp("isom", "isom") + strings.Repeat("\x00", 3000)})
	_, record := fakeTool(t, "lastarg", "ffmpeg")
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on, Media: &MediaPolicy{Video: MediaLimit{MaxBytes: 2000}}})
	res := readRes(t, ts, "clip.mp4")
	if len(res.Parts) != 1 || res.Parts[0].VideoURL == nil {
		t.Fatalf("no video part: %s", res.Text)
	}
	r := readRecord(t, record)
	a := r.Argv[1:]
	wantHead := []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
		"-i", "file:" + filepath.Join(ts.Root(), "clip.mp4")}
	if !slices.Equal(a[:len(wantHead)], wantHead) {
		t.Errorf("argv head\n got %q\nwant %q", a[:len(wantHead)], wantHead)
	}
	for _, w := range []string{"libx264", "aac", "+faststart"} {
		if !slices.Contains(a, w) {
			t.Errorf("argv lacks %q: %q", w, a)
		}
	}
	out := a[len(a)-1]
	if a[len(a)-2] != "-y" || !strings.Contains(out, "agentkit-media-") {
		t.Errorf("-y is not on our temp file: %q", a[len(a)-2:])
	}
	if r.OutMode != "-rw-------" {
		t.Errorf("temp file mode %s, want 0600", r.OutMode)
	}
	if _, err := os.Stat(filepath.Dir(out)); err == nil {
		t.Error("temp dir was not removed")
	}
	contains(t, res.Text, "by ffmpeg: H.264/AAC mp4, at most 720p")
	if got := ev.list(); len(got) != 1 || got[0].Tool != "ffmpeg" || got[0].OutMIME != "video/mp4" {
		t.Fatalf("events %+v", got)
	}
}

func TestScaleFFmpegAudioIsMonoMP3(t *testing.T) {
	root := tree(t, map[string]string{"talk.wav": wavBytes(4000)})
	_, record := fakeTool(t, "lastarg", "ffmpeg")
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Audio: MediaLimit{MaxBytes: 2000}}})
	res := readRes(t, ts, "talk.wav")
	if len(res.Parts) != 1 || res.Parts[0].InputAudio == nil || res.Parts[0].InputAudio.Format != "mp3" {
		t.Fatalf("no mp3 part: %s %+v", res.Text, res.Parts)
	}
	a := readRecord(t, record).Argv
	for _, w := range []string{"-vn", "libmp3lame", "64k"} {
		if !slices.Contains(a, w) {
			t.Errorf("argv lacks %q: %q", w, a)
		}
	}
	contains(t, res.Text, "mono mp3 64 kbit/s")
}

func TestScalePinnedToolPathWinsOverPATH(t *testing.T) {
	root := tree(t, map[string]string{"x.png": pngBytes(t, noise(2000, 50, false))})
	dir, record := fakeTool(t, "lastarg", "vipsthumbnail")
	pinned := filepath.Join(t.TempDir(), "my-vips")
	self, _ := os.Executable()
	if err := os.Symlink(self, pinned); err != nil {
		t.Skip(err)
	}
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Tools: MediaTools{VipsThumbnail: pinned}}})
	_ = readRes(t, ts, "x.png") // the fake writes to its last arg, not -o; the part is not the point
	r := readRecord(t, record)
	if r.Argv[0] != pinned {
		t.Fatalf("ran %s, want the pinned %s (PATH had %s)", r.Argv[0], pinned, dir)
	}
	if r.Argv[1] != filepath.Join(ts.Root(), "x.png") || r.Argv[2] != "--size=1568x1568>" || r.Argv[3] != "-o" {
		t.Errorf("vipsthumbnail argv %q", r.Argv)
	}
}

func TestScaleTimeoutKillsAHangingTool(t *testing.T) {
	root := tree(t, map[string]string{"x.webp": webpHeader(3000, 2000, 9000)})
	fakeTool(t, "hang", "magick")
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Timeout: 300 * time.Millisecond}})
	start := time.Now()
	out := readRes(t, ts, "x.webp").Text
	if d := time.Since(start); d > 10*time.Second {
		t.Fatalf("a hanging tool held the read %s", d)
	}
	// Under the byte cap, over the edge: sent unscaled, and the label says why.
	contains(t, out, "sent unscaled", "magick timed out after 300ms")
}

func TestScaleMagickIsNeverRunOnANonImage(t *testing.T) {
	if _, err := magickImage(context.Background(), time.Second, "/bin/false", tPDF, MediaIn{Path: "/x.pdf"}); err == nil ||
		!strings.Contains(err.Error(), "only run on images") {
		t.Fatalf("err %v", err)
	}
}

func TestScaleArgPathGuardsALeadingDash(t *testing.T) {
	if got := argPath("-rf"); got != "."+string(filepath.Separator)+"-rf" {
		t.Fatalf("argPath(-rf) = %q", got)
	}
}

// Real tools, when present.

func TestScaleRealMagickShrinksAPNGAndAWebP(t *testing.T) {
	bin, err := exec.LookPath("magick")
	if err != nil {
		t.Skip("magick not on PATH")
	}
	root := tree(t, map[string]string{"big.png": pngBytes(t, noise(3000, 1500, false))})
	webp := filepath.Join(root, "big.webp")
	if out, err := exec.Command(bin, "-size", "3000x1500", "gradient:red-blue", webp).CombinedOutput(); err != nil {
		t.Logf("no webp from magick (%v %s); png only", err, out)
		webp = ""
	}
	var ev events
	ts := mount(t, Config{Root: root, OnMedia: ev.on})
	for _, n := range []string{"big.png", "big.webp"} {
		if n == "big.webp" && webp == "" {
			continue
		}
		res := readRes(t, ts, n)
		if len(res.Parts) != 1 {
			t.Fatalf("%s: no part: %s", n, res.Text)
		}
		_, data := partBytes(t, res.Parts[0])
		c, format, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil || format != "jpeg" || c.Width != 1568 || c.Height != 784 {
			t.Fatalf("%s: magick made %s %d×%d: %v", n, format, c.Width, c.Height, err)
		}
	}
	for _, e := range ev.list() {
		if e.Tool != "magick" {
			t.Errorf("tool %s", e.Tool)
		}
	}
}

func TestScaleRealFFmpegShrinksAWav(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	root := tree(t, map[string]string{"talk.wav": wavBytes(80000)})
	ts := mount(t, Config{Root: root, Media: &MediaPolicy{Audio: MediaLimit{MaxBytes: 100000}}})
	res := readRes(t, ts, "talk.wav")
	if len(res.Parts) != 1 || res.Parts[0].InputAudio == nil || res.Parts[0].InputAudio.Format != "mp3" {
		t.Fatalf("no mp3: %s", res.Text)
	}
}
