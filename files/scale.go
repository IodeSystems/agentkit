package files

// scale.go — shrinking a media file with the tools already on the machine.
//
// The module takes no dependencies, and the stdlib decodes only png, jpeg and
// gif and encodes no audio or video. So scaling shells out, the way grep uses
// rg: the first tool found, in this order —
//
//  1. ImageMagick `magick` (7), else `convert` (6; never on Windows, where
//     convert.exe converts FAT disks to NTFS)   images
//  2. `ffmpeg`                                  images, audio, video
//  3. `vipsthumbnail`                           images
//  4. the stdlib (stdscale.go)                  png, jpeg, gif only
//
// A pinned path (MediaTools) wins over PATH; MediaTools.None looks nothing up.
// A host Scaler replaces all of it.
//
// These tools parse hostile formats — the file is whatever the model was
// pointed at, and an image decoder is a classic exploit target — so every run
// is held down the same way:
//
//   - exec, never a shell: an argument list, so no file name is ever parsed as
//     shell syntax.
//   - a timeout per run (MediaPolicy.Timeout, 30s), killed by the context, and
//     WaitDelay so a child that keeps a pipe open cannot hold the read.
//   - the input path is absolute, so a name starting with "-" is never read as
//     an option. ImageMagick gets an explicit coder prefix from the SNIFFED
//     type ("png:/abs/x.png") so it cannot pick a coder from the content (its
//     MVG, MSL and URL coders read other files or the network) and is refused
//     outright on a non-image; its resources are capped with -limit. ffmpeg
//     gets -protocol_whitelist file and a "file:" URL, so a playlist or a
//     concat file cannot make it open the network or another protocol.
//   - output goes to stdout or a 0600 file in a fresh os.MkdirTemp dir that is
//     always removed; -y only overwrites that file, which we created. What is
//     read back is capped.
//
// One pass aims at MaxLongEdge and the byte cap. If its output is still over
// the cap, one more pass shrinks harder (smaller edge, lower quality or
// bitrate); still over, the file is refused.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// maxToolOutput caps what one tool run may write back. The useful output is
// under the media cap; this only stops a runaway tool from filling memory.
const maxToolOutput = 64 << 20

// scale runs the scaler for in: the host's, or the first tool found, with one
// harder retry when the first output is over in.MaxBytes.
func (t *Toolset) scale(ctx context.Context, p MediaPolicy, m mediaType, in MediaIn) (MediaOut, string, error) {
	var last int
	tool := "none"
	for attempt := 0; attempt < 2; attempt++ {
		in.Attempt = attempt
		var out MediaOut
		var err error
		if p.Scaler != nil {
			tool = "host"
			rctx, cancel := context.WithTimeout(ctx, p.timeout())
			out, err = p.Scaler(rctx, in)
			cancel()
		} else {
			out, tool, err = t.scaleBuiltin(ctx, p, m, in)
		}
		if err != nil {
			return MediaOut{}, tool, err
		}
		if len(out.Data) == 0 {
			return MediaOut{}, tool, fmt.Errorf("%s returned no data", tool)
		}
		if int64(len(out.Data)) <= in.MaxBytes {
			return out, tool, nil
		}
		last = len(out.Data)
	}
	return MediaOut{}, tool, fmt.Errorf("%s got it to %s, still over %s", tool, sizeOf(int64(last)), sizeOf(in.MaxBytes))
}

// scaleBuiltin is one pass with the first tool found for the kind.
func (t *Toolset) scaleBuiltin(ctx context.Context, p MediaPolicy, m mediaType, in MediaIn) (MediaOut, string, error) {
	if m.kind != MediaImage {
		bin := t.mediaTool(p, "ffmpeg")
		if bin == "" {
			return MediaOut{}, "none", errors.New("no ffmpeg on PATH")
		}
		out, err := ffmpegMedia(ctx, p.timeout(), bin, m, in)
		return out, "ffmpeg", err
	}
	for _, name := range imageTools() {
		bin := t.mediaTool(p, name)
		if bin == "" {
			continue
		}
		var out MediaOut
		var err error
		switch name {
		case "magick", "convert":
			out, err = magickImage(ctx, p.timeout(), bin, m, in)
		case "ffmpeg":
			out, err = ffmpegMedia(ctx, p.timeout(), bin, m, in)
		case "vipsthumbnail":
			out, err = vipsImage(ctx, p.timeout(), bin, in)
		}
		if err != nil && stdDecodes(m) {
			// The tool lacks a delegate or failed on this file; the stdlib
			// can still do png, jpeg and gif.
			if o, serr := stdScale(in); serr == nil {
				return o, "stdlib", nil
			}
		}
		return out, name, err
	}
	if !stdDecodes(m) {
		return MediaOut{}, "none", fmt.Errorf("no decoder for %s: no magick, convert, ffmpeg or vipsthumbnail on PATH", m.mime)
	}
	out, err := stdScale(in)
	return out, "stdlib", err
}

func imageTools() []string {
	if runtime.GOOS == "windows" {
		return []string{"magick", "ffmpeg", "vipsthumbnail"}
	}
	return []string{"magick", "convert", "ffmpeg", "vipsthumbnail"}
}

// mediaTool is the path of a tool: pinned, else PATH (cached per Toolset), ""
// when absent or turned off.
func (t *Toolset) mediaTool(p MediaPolicy, name string) string {
	pinned := map[string]string{"magick": p.Tools.Magick, "convert": p.Tools.Convert,
		"ffmpeg": p.Tools.FFmpeg, "vipsthumbnail": p.Tools.VipsThumbnail}[name]
	if pinned != "" {
		return pinned
	}
	if p.Tools.None || (name == "convert" && runtime.GOOS == "windows") {
		return ""
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if v, ok := t.toolPath[name]; ok {
		return v
	}
	v, err := exec.LookPath(name)
	if err != nil {
		v = ""
	}
	t.toolPath[name] = v
	return v
}

// pass returns the target edge and a 0-100 quality for an attempt: the first
// pass fits the edge at quality 85, the retry takes 70% of that edge at 60.
func pass(in MediaIn) (edge, quality int) {
	edge = in.MaxLongEdge
	if l := max(in.Width, in.Height); l > 0 && l < edge {
		edge = l
	}
	quality = 85
	if in.Attempt > 0 {
		edge, quality = max(1, edge*7/10), 60
	}
	return edge, quality
}

// argPath makes a path safe as an argument: a relative one starting with "-"
// would be read as an option. Callers pass absolute paths; this is the guard.
func argPath(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return "." + string(filepath.Separator) + p
}

// magickArgs is the ImageMagick argument list for one pass. The input's coder
// comes from the sniffed type and [0] takes the first frame (a gif, a
// multi-page tiff). -limit time is in seconds.
func magickArgs(timeout time.Duration, m mediaType, in MediaIn) (args []string, outMIME string) {
	edge, q := pass(in)
	out, outMIME := "jpeg:-", "image/jpeg"
	if in.Alpha {
		out, outMIME = "png:-", "image/png"
	}
	secs := max(1, int(timeout/time.Second))
	return []string{
		"-limit", "memory", "256MiB", "-limit", "map", "512MiB",
		"-limit", "time", strconv.Itoa(secs), "-limit", "disk", "1GiB",
		m.coder + ":" + argPath(in.Path) + "[0]",
		"-auto-orient", "-resize", fmt.Sprintf("%dx%d>", edge, edge), "-strip", "-quality", strconv.Itoa(q),
		out,
	}, outMIME
}

func magickImage(ctx context.Context, timeout time.Duration, bin string, m mediaType, in MediaIn) (MediaOut, error) {
	if m.kind != MediaImage || m.coder == "" {
		return MediaOut{}, fmt.Errorf("ImageMagick is only run on images, not %s", m.mime)
	}
	args, mime := magickArgs(timeout, m, in)
	data, err := runTool(ctx, timeout, bin, args)
	if err != nil {
		return MediaOut{}, err
	}
	return MediaOut{Data: data, MIME: mime}, nil
}

// ffmpegArgs is the ffmpeg argument list for one pass, writing to out.
func ffmpegArgs(m mediaType, in MediaIn, out string) (args []string, outMIME, note string) {
	args = []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
		"-i", "file:" + argPath(in.Path)}
	switch m.kind {
	case MediaImage:
		edge, q := pass(in)
		// ffmpeg's mjpeg -q:v runs 2 (best) to 31.
		qv := 2 + (100-q)*29/100
		args = append(args, "-frames:v", "1", "-vf",
			fmt.Sprintf("scale=w='min(iw,%d)':h='min(ih,%d)':force_original_aspect_ratio=decrease", edge, edge))
		if in.Alpha {
			args = append(args, "-c:v", "png", "-f", "image2", "-update", "1")
			outMIME = "image/png"
		} else {
			args = append(args, "-c:v", "mjpeg", "-q:v", strconv.Itoa(qv), "-f", "image2", "-update", "1")
			outMIME = "image/jpeg"
		}
	case MediaAudio:
		rate := "64k"
		if in.Attempt > 0 {
			rate = "32k"
		}
		args = append(args, "-vn", "-ac", "1", "-c:a", "libmp3lame", "-b:a", rate, "-f", "mp3")
		outMIME, note = "audio/mpeg", "mono mp3 "+strings.TrimSuffix(rate, "k")+" kbit/s"
	case MediaVideo:
		h, crf := 720, 28
		if in.Attempt > 0 {
			h, crf = 480, 34
		}
		// The short edge to at most h, never up; even dimensions for yuv420p.
		vf := fmt.Sprintf("scale=w='if(gt(iw,ih),-2,trunc(min(iw,%d)/2)*2)':h='if(gt(iw,ih),trunc(min(ih,%d)/2)*2,-2)'", h, h)
		args = append(args, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", strconv.Itoa(crf),
			"-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-movflags", "+faststart", "-f", "mp4")
		outMIME, note = "video/mp4", fmt.Sprintf("H.264/AAC mp4, at most %dp, crf %d", h, crf)
	}
	return append(args, "-y", out), outMIME, note
}

func ffmpegMedia(ctx context.Context, timeout time.Duration, bin string, m mediaType, in MediaIn) (MediaOut, error) {
	var out MediaOut
	err := withTempFile("out", func(path string) error {
		args, mime, note := ffmpegArgs(m, in, path)
		if _, err := runTool(ctx, timeout, bin, args); err != nil {
			return err
		}
		data, err := readCapped(path, maxToolOutput)
		out = MediaOut{Data: data, MIME: mime, Note: note}
		return err
	})
	return out, err
}

// vipsArgs is the vipsthumbnail argument list for one pass. ">" in --size
// only shrinks; "[Q=..,strip]" are the jpeg saver's options.
func vipsArgs(in MediaIn, dir string) (args []string, out, outMIME string) {
	edge, q := pass(in)
	if in.Alpha {
		out, outMIME = filepath.Join(dir, "out.png"), "image/png"
		return []string{argPath(in.Path), fmt.Sprintf("--size=%dx%d>", edge, edge), "-o", out + "[strip]"}, out, outMIME
	}
	out, outMIME = filepath.Join(dir, "out.jpg"), "image/jpeg"
	return []string{argPath(in.Path), fmt.Sprintf("--size=%dx%d>", edge, edge), "-o",
		fmt.Sprintf("%s[Q=%d,strip]", out, q)}, out, outMIME
}

func vipsImage(ctx context.Context, timeout time.Duration, bin string, in MediaIn) (MediaOut, error) {
	var out MediaOut
	err := withTempFile("", func(path string) error {
		args, file, mime := vipsArgs(in, filepath.Dir(path))
		if _, err := runTool(ctx, timeout, bin, args); err != nil {
			return err
		}
		data, err := readCapped(file, maxToolOutput)
		out = MediaOut{Data: data, MIME: mime}
		return err
	})
	return out, err
}

// withTempFile runs fn with a path in a new 0700 directory that is removed
// after. With a name, the file is created first, 0600 and exclusive, so the
// tool writes into a file this process made (and -y overwrites only that).
func withTempFile(name string, fn func(path string) error) error {
	dir, err := os.MkdirTemp("", "agentkit-media-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	if name == "" {
		return fn(filepath.Join(dir, "unused"))
	}
	path := filepath.Join(dir, name)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	_ = f.Close()
	return fn(path)
}

// runTool runs bin with args — no shell — under a timeout, and returns its
// stdout, capped at maxToolOutput.
func runTool(ctx context.Context, timeout time.Duration, bin string, args []string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.WaitDelay = 2 * time.Second
	stdout := &capWriter{max: maxToolOutput}
	stderr := &capWriter{max: 2 << 10, truncate: true}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	name := filepath.Base(bin)
	switch {
	case ctx.Err() == context.DeadlineExceeded:
		return nil, fmt.Errorf("%s timed out after %s", name, timeout)
	case ctx.Err() != nil:
		return nil, ctx.Err()
	case stdout.over:
		return nil, fmt.Errorf("%s wrote over %s", name, sizeOf(maxToolOutput))
	case err != nil:
		msg := strings.TrimSpace(stderr.buf.String())
		if msg == "" {
			return nil, fmt.Errorf("%s failed: %v", name, err)
		}
		return nil, fmt.Errorf("%s failed: %v: %s", name, err, msg)
	}
	return stdout.buf.Bytes(), nil
}

// capWriter keeps at most max bytes. Over it, it fails the write (stdout: the
// output is unusable) or drops the rest (stderr: the head says why).
type capWriter struct {
	buf      bytes.Buffer
	max      int
	truncate bool
	over     bool
}

func (w *capWriter) Write(p []byte) (int, error) {
	room := w.max - w.buf.Len()
	if len(p) > room {
		w.buf.Write(p[:max(0, room)])
		if w.truncate {
			return len(p), nil
		}
		w.over = true
		return 0, io.ErrShortWrite
	}
	return w.buf.Write(p)
}
