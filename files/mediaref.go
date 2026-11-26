package files

// mediaref.go — read on a "media:<hex>" path: media stored earlier in the
// session, not a file.
//
// A Shaper that stops resending old media (agent.ShaperPolicy.StubMediaOlderThan)
// leaves a label that names each part by llm.MediaRef and says to read that
// path. read is the one path media takes into the model, so read answers it:
// Config.MediaRef (agent's Session.MediaRef) looks the reference up in the
// session's current entries and read returns the stored part as it was, with
// no disk read and no re-encoding. The current MediaPolicy still applies: a kind
// that is off returns a label, and a stored copy over the kind's byte cap goes
// through the same scaling path as a file over it (the bytes are written to a
// private temp file for the scaler, as a scaler's own output is).
//
// Safety: a path that starts with "media:" is never resolved against the root,
// so it cannot name, read or escape into the filesystem; only the resolver sees
// it. A real file of that name is still readable as "./media:…", which does
// not start with "media:". Other tools (edit, write, grep, list) treat such a
// name as an ordinary path inside the root.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// isMediaPath says whether a read path is a media reference rather than a file.
func isMediaPath(p string) bool { return strings.HasPrefix(strings.TrimSpace(p), llm.MediaRefPrefix) }

// readMediaRef is read for a "media:<hex>" path.
func (t *Toolset) readMediaRef(ctx context.Context, path string) (llm.ToolResult, error) {
	ref := strings.TrimSpace(path)
	if !llm.IsMediaRef(ref) {
		return llm.ToolResult{}, fmt.Errorf("%s is not a media reference (media: and 16 hex digits); to read a file of "+
			"that name, give it as ./%s", ref, ref)
	}
	if t.cfg.MediaRef == nil {
		return llm.ToolResult{}, errors.New("media references are not available in this session")
	}
	part, ok := t.cfg.MediaRef(ctx, ref)
	if !ok {
		return llm.ToolResult{}, fmt.Errorf("%s is no longer in this session", ref)
	}
	kind := partMediaKind(part)
	mime := llm.MediaMIME(part)
	data, inline := llm.MediaBytes(part)
	size := MediaSize{Bytes: int64(len(data))}
	desc := mime + " (stored reference from earlier in this session)"
	if inline {
		desc = fmt.Sprintf("%s, %s (stored copy from earlier in this session)", mime, sizeOf(size.Bytes))
	}

	p := t.MediaPolicy()
	lim := p.limit(kind)
	if lim.Mode == MediaOff {
		ev := MediaEvent{Path: ref, MIME: mime, Kind: kind, Before: size, Tool: "none", Refused: true,
			Reason: fmt.Sprintf("%s not sent: %s media is off for this session", mime, kind)}
		t.emitMedia(ev)
		return llm.TextResult(fmt.Sprintf("%s: %s; %s", ref, desc, ev.Reason)), nil
	}
	if inline && size.Bytes > lim.MaxBytes {
		// Over the cap now (the user lowered it): the same path as a file over it.
		return t.rescaleStored(ctx, ref, data)
	}
	return llm.ToolResult{Text: ref + ": " + desc, Parts: []llm.ContentPart{part}}, nil
}

// rescaleStored runs a stored copy over its cap through readMedia, from a 0600
// file in a fresh temp dir that is removed after.
func (t *Toolset) rescaleStored(ctx context.Context, ref string, data []byte) (llm.ToolResult, error) {
	head := data[:min(len(data), binaryScan)]
	m, ok := sniffMedia(head)
	if !ok {
		return llm.ToolResult{}, fmt.Errorf("%s: a stored copy of %s is over this session's limit and its format is not "+
			"recognized, so it cannot be shrunk", ref, sizeOf(int64(len(data))))
	}
	var res llm.ToolResult
	err := withTempFile("stored", func(path string) error {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			return err
		}
		var rerr error
		res, rerr = t.readMedia(ctx, path, ref, int64(len(data)), head, m)
		return rerr
	})
	if err != nil {
		return llm.ToolResult{}, err
	}
	res.Text += " (stored copy from earlier in this session, over the current limit)"
	return res, nil
}

// partMediaKind is the policy kind of a content part.
func partMediaKind(p llm.ContentPart) MediaKind {
	switch p.Type {
	case "image_url":
		return MediaImage
	case "input_audio":
		return MediaAudio
	case "video_url":
		return MediaVideo
	}
	return MediaPDF
}
