package llm

// Wire dump. Some provider bugs are invisible above the byte level: the SSE
// parser drops comment lines, joins data lines, and turns a malformed event into
// one Go error, so a report built from parsed chunks has already lost the
// evidence. The case that made this: Gemini through OpenRouter ends a stream
// with MALFORMED_FUNCTION_CALL, reproducible on one machine only. The person who
// can reproduce it needs a capture to send, without a proxy or a rebuild.
//
// How to use: set AGENTKIT_WIRE_DUMP to a directory. Every request that goes
// through Client.do (chat, embed, transcribe, the capability, tokenize and
// budget probes) then writes one file there. Unset or empty = off. The variable
// is read on each request, not at NewClient, so a host can turn it on and off
// without a restart. Off costs one os.Getenv per request; the response body is
// not wrapped and no per-chunk work runs.
//
// One file per HTTP attempt: a retry is a new file. The name is
//
//	<UTC time>-<seq>-<trace id>-a<attempt>.txt
//
// The time sorts the files; seq is a process-wide counter, so two requests in
// the same nanosecond still get two files. The trace id (X-Trace-Id, made safe
// for a file name) groups the attempts of one call, and a<N> is the attempt
// number inside postWithRetry (absent for a request that has no retry loop).
// Directory 0700, files 0600.
//
// File format, plain text so the raw bytes need no escaping:
//
//	> POST <url>                 request line
//	> Name: value                request headers, one per line
//	> body <N> bytes             then the body, byte for byte, then a newline
//	< 200 OK +0.412s             status, time since the request started
//	< Name: value                response headers
//	+0.413s data: {...}          every body line as received, with its time,
//	+0.900s : keep-alive         comment lines, event:, data:, [DONE] included
//	# eof +1.204s                how it ended: eof, read error, closed early
//
// Each body line is written with its own bytes, including a \r; nothing is
// decoded. A last line with no newline is written with " # no trailing newline".
// A non-streaming or error response is the same: its body, line by line.
//
// Redacted (name kept, value "[redacted]"): Authorization, Proxy-Authorization,
// X-Api-Key, Api-Key, Cookie, Set-Cookie, and every header whose name contains
// "key", "token" or "secret" (any case), in the request and the response. This
// also hides harmless headers such as X-Ratelimit-Remaining-Tokens; safe is
// preferred to complete. The same name rule applies to URL query parameters
// (?key=...), and a password in the URL is redacted.
//
// WARNING: the body is NOT redacted. A dump holds the full prompt, the tool
// definitions, every tool result in the history and the model's reply. Treat a
// dump directory as sensitive data.
//
// A dump failure (no directory, disk full) never fails or slows the request:
// the dump for that attempt stops and nothing is logged. There is no hook to
// report it; an empty or missing file is the signal.

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// WireDumpEnv names the directory for raw request/response captures. See the
// top of wiredump.go.
const WireDumpEnv = "AGENTKIT_WIRE_DUMP"

const redacted = "[redacted]"

// wireDumpSeq makes file names unique inside the process.
var wireDumpSeq atomic.Uint64

// wireAttemptKey is the ctx key for the attempt number postWithRetry sets.
type wireAttemptKey struct{}

// withWireAttempt marks ctx with attempt n (1-based) when the dump is on. Off,
// it returns ctx unchanged, so the retry loop allocates nothing extra.
func withWireAttempt(ctx context.Context, n int) context.Context {
	if os.Getenv(WireDumpEnv) == "" {
		return ctx
	}
	return context.WithValue(ctx, wireAttemptKey{}, n)
}

// wireDump is the capture of one attempt. Every write holds mu: Read and Close
// of a response body can run on different goroutines.
type wireDump struct {
	mu    sync.Mutex
	f     *os.File // nil after a failure or Close
	start time.Time
	part  []byte // a body line not yet ended by \n
}

// startWireDump opens the dump file for req and writes the request, or returns
// nil when the dump is off or the file cannot be made. It may replace req.Body
// with an identical reader when the request has no GetBody.
func startWireDump(req *http.Request) *wireDump {
	dir := os.Getenv(WireDumpEnv)
	if dir == "" {
		return nil
	}
	body := requestBody(req)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil
	}
	now := time.Now()
	name := fmt.Sprintf("%s-%06d-%s", now.UTC().Format("20060102T150405.000000000Z"),
		wireDumpSeq.Add(1), fileSafe(req.Header.Get("X-Trace-Id")))
	if n, ok := req.Context().Value(wireAttemptKey{}).(int); ok {
		name += fmt.Sprintf("-a%d", n)
	}
	f, err := os.OpenFile(filepath.Join(dir, name+".txt"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil
	}
	d := &wireDump{f: f, start: now}
	var b bytes.Buffer
	fmt.Fprintf(&b, "> %s %s\n", req.Method, redactURL(req.URL))
	writeHeaders(&b, "> ", req.Header)
	fmt.Fprintf(&b, "> body %d bytes\n", len(body))
	b.Write(body)
	b.WriteString("\n")
	d.write(b.Bytes())
	return d
}

// requestBody is a copy of the body that will be sent. GetBody gives a fresh
// reader and leaves req.Body alone; without it the body is read and replaced.
func requestBody(req *http.Request) []byte {
	if req.Body == nil || req.Body == http.NoBody {
		return nil
	}
	if req.GetBody != nil {
		if rc, err := req.GetBody(); err == nil {
			b, _ := io.ReadAll(rc)
			rc.Close()
			return b
		}
	}
	// A read error here would have failed the send too; what was read is sent.
	b, _ := io.ReadAll(req.Body)
	req.Body.Close()
	req.Body = io.NopCloser(bytes.NewReader(b))
	return b
}

// response records the outcome of c.http.Do and wraps the body. It returns
// resp and err unchanged otherwise.
func (d *wireDump) response(resp *http.Response, err error) (*http.Response, error) {
	if err != nil {
		d.write([]byte(fmt.Sprintf("# request error %s: %v\n", d.since(), err)))
		d.close()
		return resp, err
	}
	var b bytes.Buffer
	fmt.Fprintf(&b, "< %s %s\n", resp.Status, d.since())
	writeHeaders(&b, "< ", resp.Header)
	d.write(b.Bytes())
	resp.Body = &dumpBody{rc: resp.Body, d: d}
	return resp, nil
}

func (d *wireDump) since() string {
	return fmt.Sprintf("+%.3fs", time.Since(d.start).Seconds())
}

// write appends p; on the first failure the dump stops.
func (d *wireDump) write(p []byte) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.writeLocked(p)
}

func (d *wireDump) writeLocked(p []byte) {
	if d.f == nil {
		return
	}
	if _, err := d.f.Write(p); err != nil {
		d.f.Close()
		d.f = nil
	}
}

// body records one Read: each completed line with its time, and the error that
// ended the body, if any.
func (d *wireDump) body(p []byte, rerr error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.f == nil {
		return
	}
	ts := d.since()
	var b bytes.Buffer
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			d.part = append(d.part, p...)
			break
		}
		b.WriteString(ts)
		b.WriteByte(' ')
		b.Write(d.part)
		b.Write(p[:i+1])
		d.part = d.part[:0]
		p = p[i+1:]
	}
	if rerr != nil {
		d.flushPartLocked(&b, ts)
		if rerr == io.EOF {
			fmt.Fprintf(&b, "# eof %s\n", ts)
		} else {
			fmt.Fprintf(&b, "# read error %s: %v\n", ts, rerr)
		}
	}
	if b.Len() > 0 {
		d.writeLocked(b.Bytes())
	}
}

func (d *wireDump) flushPartLocked(b *bytes.Buffer, ts string) {
	if len(d.part) == 0 {
		return
	}
	b.WriteString(ts)
	b.WriteByte(' ')
	b.Write(d.part)
	b.WriteString(" # no trailing newline\n")
	d.part = d.part[:0]
}

// close ends the dump. A body closed before EOF (the caller stopped reading)
// says so, with any partial line kept.
func (d *wireDump) close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.f == nil {
		return
	}
	var b bytes.Buffer
	ts := d.since()
	d.flushPartLocked(&b, ts)
	fmt.Fprintf(&b, "# closed %s\n", ts)
	d.writeLocked(b.Bytes())
	if d.f != nil {
		d.f.Close()
		d.f = nil
	}
}

// dumpBody copies every byte the caller reads into the dump. It sits outside
// watchBody, so a StallError that watchBody returns is recorded too.
type dumpBody struct {
	rc io.ReadCloser
	d  *wireDump
}

func (b *dumpBody) Read(p []byte) (int, error) {
	n, err := b.rc.Read(p)
	if n > 0 || err != nil {
		b.d.body(p[:n], err)
	}
	return n, err
}

func (b *dumpBody) Close() error {
	err := b.rc.Close()
	b.d.close()
	return err
}

// writeHeaders writes h sorted by name, one value per line, redacted.
func writeHeaders(b *bytes.Buffer, prefix string, h http.Header) {
	names := make([]string, 0, len(h))
	for k := range h {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		for _, v := range h[k] {
			if sensitiveName(k) {
				v = redacted
			}
			fmt.Fprintf(b, "%s%s: %s\n", prefix, k, v)
		}
	}
}

// sensitiveName is the redaction rule for header and query parameter names.
func sensitiveName(name string) bool {
	n := strings.ToLower(name)
	switch n {
	case "authorization", "proxy-authorization", "cookie", "set-cookie":
		return true
	}
	return strings.Contains(n, "key") || strings.Contains(n, "token") || strings.Contains(n, "secret")
}

// redactURL is u with a password and the sensitive query values replaced.
// The query is rewritten by hand, not with url.Values, so the parameter order
// and the encoding of the other values stay as sent.
func redactURL(u *url.URL) string {
	if u == nil {
		return ""
	}
	c := *u
	if c.User != nil {
		if _, ok := c.User.Password(); ok {
			c.User = url.UserPassword(c.User.Username(), redacted)
		}
	}
	if c.RawQuery != "" {
		parts := strings.Split(c.RawQuery, "&")
		for i, p := range parts {
			k, _, ok := strings.Cut(p, "=")
			name, err := url.QueryUnescape(k)
			if err != nil {
				name = k
			}
			if ok && sensitiveName(name) {
				parts[i] = k + "=" + redacted
			}
		}
		c.RawQuery = strings.Join(parts, "&")
	}
	return c.String()
}

// fileSafe is s reduced to [A-Za-z0-9._-], at most 64 bytes; "notrace" when
// empty.
func fileSafe(s string) string {
	var b strings.Builder
	for _, r := range s {
		if b.Len() >= 64 {
			break
		}
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "notrace"
	}
	return b.String()
}
