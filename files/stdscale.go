package files

// stdscale.go — the image scaler of last resort: the standard library only.
//
// With no ImageMagick, ffmpeg or vipsthumbnail on the machine, png, jpeg and
// gif can still be shrunk: decode with image/*, box-filter down to the long
// edge, encode jpeg (or png when the source has transparency, which jpeg would
// turn black). A box (area) filter, not a better kernel, because it is ~40
// lines, has no ringing, and a downscale for a model's eyes needs no more.
// webp, heic, avif, tiff and bmp have no stdlib decoder; they need a tool.
//
// Decoding allocates width × height × 4 bytes up front, so the size is read
// from the header first and an image over maxStdPixels is refused: a 100 KB
// png can claim 50000 × 50000.

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
)

// maxStdPixels bounds what the stdlib scaler decodes: 40 MP is ~160 MB of
// RGBA, and covers a 48 MP phone photo's common 12 MP binned output with room.
const maxStdPixels = 40_000_000

// maxStdInput bounds the file the stdlib scaler loads whole.
const maxStdInput = 64 << 20

func stdDecodes(m mediaType) bool { return m == tPNG || m == tJPEG || m == tGIF }

func stdScale(in MediaIn) (MediaOut, error) {
	if in.Width > 0 && in.Height > 0 && int64(in.Width)*int64(in.Height) > maxStdPixels {
		return MediaOut{}, fmt.Errorf("%d×%d is over the stdlib scaler's %d MP", in.Width, in.Height, maxStdPixels/1_000_000)
	}
	raw, err := readCapped(in.Path, maxStdInput)
	if err != nil {
		return MediaOut{}, err
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return MediaOut{}, err
	}
	if int64(cfg.Width)*int64(cfg.Height) > maxStdPixels {
		return MediaOut{}, fmt.Errorf("%d×%d is over the stdlib scaler's %d MP", cfg.Width, cfg.Height, maxStdPixels/1_000_000)
	}
	src, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return MediaOut{}, err
	}
	in.Width, in.Height = src.Bounds().Dx(), src.Bounds().Dy()
	edge, q := pass(in)
	dst := boxScale(src, edge)
	var buf bytes.Buffer
	if o, ok := src.(interface{ Opaque() bool }); ok && !o.Opaque() {
		if err := png.Encode(&buf, dst); err != nil {
			return MediaOut{}, err
		}
		return MediaOut{Data: buf.Bytes(), MIME: "image/png"}, nil
	}
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: q}); err != nil {
		return MediaOut{}, err
	}
	return MediaOut{Data: buf.Bytes(), MIME: "image/jpeg"}, nil
}

// boxScale fits src inside edge × edge (never up), each destination pixel the
// mean of the source pixels it covers. The mean is over premultiplied RGBA, so
// a transparent pixel adds no color to its neighbors.
func boxScale(src image.Image, edge int) *image.RGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dw, dh := sw, sh
	if l := max(sw, sh); l > edge {
		dw, dh = max(1, sw*edge/l), max(1, sh*edge/l)
	}
	s := image.NewRGBA(image.Rect(0, 0, sw, sh))
	draw.Draw(s, s.Bounds(), src, b.Min, draw.Src)
	if dw == sw && dh == sh {
		return s
	}
	d := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for dy := 0; dy < dh; dy++ {
		y0, y1 := dy*sh/dh, max(dy*sh/dh+1, (dy+1)*sh/dh)
		for dx := 0; dx < dw; dx++ {
			x0, x1 := dx*sw/dw, max(dx*sw/dw+1, (dx+1)*sw/dw)
			var r, g, bl, a, n uint64
			for y := y0; y < y1; y++ {
				row := s.Pix[y*s.Stride+x0*4 : y*s.Stride+x1*4]
				for i := 0; i < len(row); i += 4 {
					r += uint64(row[i])
					g += uint64(row[i+1])
					bl += uint64(row[i+2])
					a += uint64(row[i+3])
					n++
				}
			}
			o := dy*d.Stride + dx*4
			d.Pix[o], d.Pix[o+1], d.Pix[o+2], d.Pix[o+3] = uint8(r/n), uint8(g/n), uint8(bl/n), uint8(a/n)
		}
	}
	return d
}
