package loopdetector

import (
	"encoding/json"
	"hash/fnv"
	"regexp"
	"slices"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// Session watches one reply against the conversation it continues. It belongs
// to one stream and is not safe for concurrent use.
type Session struct {
	d       *LoopDetector
	exact   llm.StreamWatcher
	tracks  map[string]*track
	calls   map[uint64]*callState
	recent  []mark // tool calls
	loops   int
	kind    string
	found   *llm.RepetitionInfo
	seen    [4]bool
	stopped bool
}

// mark is what one sentence or tool call left in a window.
type mark struct {
	count int  // copies of it so far, this one included
	size  int  // bytes, including the skipped text before it
	built bool // from the reply, not the history
	kind  string
	text  string // what was said or called, capped
}

// track is one channel's position: the sentences it has said, the unfinished
// one and the window. Sentences are per channel because a reply drafted in
// thinking is not a reply said twice.
type track struct {
	text    *index
	partial []byte
	skipped int // bytes of text too short to be a sentence, since the last mark
	window  []mark

	// The exact detector's count. Its window holds few copies of a long block,
	// so the count goes on from the bytes that arrive.
	period, exactLoops, exactAt int
	bytes                       int
}

type callState struct {
	count  int
	result uint64
}

// Begin starts watching a reply to history.
func (d *LoopDetector) Begin(history []llm.Message) *Session {
	s := &Session{
		d:      d,
		exact:  llm.NewStreamWatcher(d.cfg.Exact),
		tracks: map[string]*track{},
		calls:  map[uint64]*callState{},
	}
	d.sessions.Add(1)
	if d.cfg.Off {
		return s
	}
	// The turn starts at the last message from a person. A person who asks again
	// gets the same answer again, and that is not a loop. The cost is a loop that
	// goes on only across their messages, which this does not see.
	turn := history[lastPerson(history)+1:]
	results := map[string]string{}
	for _, m := range turn {
		if m.Role == "tool" {
			results[m.ToolCallID] = m.Content
		}
	}
	t := s.track(llm.ChannelContent)
	for _, m := range turn {
		if m.Role != "assistant" {
			continue
		}
		s.feed(t, m.Content+"\n", false)
		for _, tc := range m.ToolCalls {
			s.call(tc.Function.Name, tc.Function.Arguments, results[tc.ID], false)
		}
	}
	return s
}

func lastPerson(history []llm.Message) int {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" && !notice.MatchString(history[i].Content) {
			return i
		}
	}
	return -1
}

// notice matches a user message written by the host, not a person: agent's
// render.go gives those a "[tag] " or "[compacted]\n" prefix. It is read from the
// text because that is all a proxy has. Without this the warning about a loop
// restarts the count of that loop: measured in dun, 11 identical failed calls
// with the warning after the fifth, and the window never filled. A person who
// starts a message with a tag extends the turn, which costs a count, not a cut.
var notice = regexp.MustCompile(`^\[[A-Za-z0-9_.:-]+\][ \n]`)

func (s *Session) track(channel string) *track {
	t := s.tracks[channel]
	if t == nil {
		t = &track{text: newIndex()}
		s.tracks[channel] = t
	}
	return t
}

// Loops reports the highest loop count seen so far and the kind of loop it was
// in, 0 and "" when the session is not looping.
func (s *Session) Loops() (int, string) { return s.loops, s.kind }

// Finding returns the highest finding so far, cut or not, nil when the session is
// not looping. It is for a caller who wants to pass it on.
func (s *Session) Finding() *llm.RepetitionInfo { return s.found }

// Watch feeds one delta of the reply: llm.StreamWatcher. A non-nil return means
// the loop count reached Tolerance and the stream should be cut. It says so
// once.
func (s *Session) Watch(channel, delta string) *llm.RepetitionInfo {
	if s.d.cfg.Off || delta == "" {
		return nil
	}
	s.d.bytes.Add(int64(len(delta)))
	t := s.track(channel)
	t.bytes += len(delta)
	if info := s.exact.Watch(channel, delta); info != nil {
		if info.Period == t.period {
			t.exactLoops += (t.bytes - t.exactAt) / t.period
		}
		t.period, t.exactAt = info.Period, t.bytes
		t.exactLoops = max(t.exactLoops, info.Reps)
		info.Kind, info.Reps = KindExact, t.exactLoops
		info.Sample = s.capSample(info.Sample)
		info.First, info.Last = info.Sample, info.Sample
		if out := s.report(info); out != nil {
			return out
		}
	}
	// A tool's arguments can rightly hold text from earlier in the conversation:
	// a file being written again is not a sentence being repeated. Only the exact
	// detector reads them.
	if channel != llm.ChannelContent && channel != llm.ChannelReasoning {
		return nil
	}
	return s.feed(t, delta, true)
}

// WatchToolCall feeds a finished tool call of the reply: llm.ToolCallWatcher.
func (s *Session) WatchToolCall(name, args string) *llm.RepetitionInfo {
	if s.d.cfg.Off {
		return nil
	}
	return s.call(name, args, "", true)
}

// feed splits text into sentences and takes each in turn.
func (s *Session) feed(t *track, text string, built bool) *llm.RepetitionInfo {
	cfg := s.d.cfg
	t.partial = append(t.partial, text...)
	var found *llm.RepetitionInfo
	rest := t.partial
	for {
		end := sentenceEnd(rest, cfg.MaxSegment)
		if end < 0 {
			// Moved down once, not per sentence: a whole message arrives in one
			// piece from the history.
			t.partial = t.partial[:copy(t.partial, rest)]
			return found
		}
		// partial must not hold this sentence while it reports a Trailing.
		line := string(rest[:end])
		rest = rest[end:]
		t.partial = rest
		if info := s.sentence(t, line, built); info != nil && found == nil {
			found = info
		}
	}
}

// sentenceEnd returns the length of the first sentence in b, -1 when b holds no
// finished one. A full stop counts once what follows it is known, which is what
// keeps "3.14" and "example.com" in one piece.
func sentenceEnd(b []byte, maxLen int) int {
	for i, c := range b {
		switch {
		case i >= maxLen:
			return i
		case c == '\n':
			return i + 1
		case (c == '.' || c == '!' || c == '?') && i+1 < len(b) && (b[i+1] == ' ' || b[i+1] == '\n'):
			return i + 1
		}
	}
	return -1
}

func (s *Session) sentence(t *track, text string, built bool) *llm.RepetitionInfo {
	cfg := s.d.cfg
	tokens := tokenize(text)
	if len(tokens) < cfg.MinTokens {
		t.skipped += len(text)
		return nil
	}
	s.d.sentences.Add(1)
	p, kind := t.text.match(tokens, text, cfg.Similarity)
	p.count++
	if p.count > 1 {
		s.d.repeats.Add(1)
	}
	sample := s.capSample(text)
	t.window = pushMark(t.window, mark{count: p.count, size: t.skipped + len(text), built: built, kind: kind, text: sample}, cfg.Window)
	t.skipped = 0
	loops, loopKind := s.judge(t.window, cfg.Window)
	if loops == 0 {
		return nil
	}
	span, trailing, first := 0, 0, -1
	for i, m := range t.window {
		span += m.size
		if m.count > 1 && m.built && first < 0 {
			first = i
		}
		if first >= 0 {
			trailing += m.size
		}
	}
	if span < cfg.MinSpan {
		return nil
	}
	info := &llm.RepetitionInfo{
		Where: s.whereOf(t), Kind: loopKind, Reps: loops,
		Trailing: trailing + len(t.partial), Sample: s.capSample(p.text),
		First: firstRepeat(t.window), Last: sample,
	}
	s.note(info)
	// A window can be full of the history's loop. Only the reply's own repeats
	// may cut the reply, and this sentence must be one of them.
	replied := 0
	for _, m := range t.window {
		if m.built && m.count > 1 {
			replied++
		}
	}
	if !built || p.count < 2 || replied < s.d.toleranceFor(loopKind) {
		return nil
	}
	return s.report(info)
}

func (s *Session) whereOf(t *track) string {
	if s.tracks[llm.ChannelReasoning] == t {
		return llm.ChannelReasoning
	}
	return llm.ChannelContent
}

func pushMark(w []mark, m mark, size int) []mark {
	if len(w) == size {
		w = w[:copy(w, w[1:])]
	}
	return append(w, m)
}

// judge returns the loop count of a window, 0 when too little of it repeats. The
// count is the median of the repeats' copies: one sentence the model says often
// does not speak for a window of them.
func (s *Session) judge(w []mark, size int) (int, string) {
	if len(w) < size {
		return 0, ""
	}
	var counts []int
	phrases := 0
	for _, m := range w {
		if m.count > 1 {
			counts = append(counts, m.count)
			if m.kind == KindPhrase {
				phrases++
			}
		}
	}
	if float64(len(counts)) < s.d.cfg.MinShare*float64(len(w)) {
		return 0, ""
	}
	slices.Sort(counts)
	kind := w[len(w)-1].kind
	if kind != KindTool {
		kind = KindCycle
		if phrases*2 > len(counts) {
			kind = KindPhrase
		}
	}
	return counts[len(counts)/2], kind
}

// call counts one tool call. The result of a call in the reply is not known
// yet, so it continues the count of the same call in the history.
func (s *Session) call(name, args, result string, built bool) *llm.RepetitionInfo {
	cfg := s.d.cfg
	key, sum := hashString(name+"\x00"+normalizeArgs(args)), hashString(result)
	st := s.calls[key]
	switch {
	case st == nil:
		st = &callState{count: 1, result: sum}
		s.calls[key] = st
	case built || st.result == sum:
		st.count++
	default:
		st.count, st.result = 1, sum
	}
	sample := s.capSample(name + " " + args)
	s.recent = pushMark(s.recent, mark{count: st.count, built: built, kind: KindTool, text: sample}, cfg.ToolWindow)
	loops, kind := s.judge(s.recent, cfg.ToolWindow)
	if loops == 0 {
		return nil
	}
	info := &llm.RepetitionInfo{
		Where: "tool:" + name, Kind: kind, Reps: loops, Sample: sample,
		First: firstRepeat(s.recent), Last: sample,
	}
	s.note(info)
	if !built || st.count < 2 {
		return nil
	}
	return s.report(info)
}

// note records a finding in the count, whatever the Tolerance.
func (s *Session) note(info *llm.RepetitionInfo) {
	if i := kindIndex(info.Kind); !s.seen[i] {
		s.seen[i] = true
		s.d.detected[i].Add(1)
	}
	if info.Reps > s.loops {
		s.loops, s.kind, s.found = info.Reps, info.Kind, info
	}
}

// report returns info when the count has reached Tolerance, once.
func (s *Session) report(info *llm.RepetitionInfo) *llm.RepetitionInfo {
	s.note(info)
	tol := s.d.toleranceFor(info.Kind)
	if tol < 0 || info.Reps < tol || s.stopped {
		return nil
	}
	s.stopped = true
	s.d.interrupted[kindIndex(info.Kind)].Add(1)
	return info
}

func (s *Session) capSample(text string) string {
	text = strings.TrimSpace(text)
	if n := s.d.cfg.SampleBytes; len(text) > n {
		text = strings.ToValidUTF8(text[:n], "")
	}
	return text
}

// firstRepeat returns the earliest repeat in a window.
func firstRepeat(window []mark) string {
	for _, m := range window {
		if m.count > 1 {
			return m.text
		}
	}
	return ""
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// normalizeArgs makes two calls that mean the same compare equal: JSON key order
// goes, and nothing inside a value changes, because arguments are shell commands
// and file contents, where a space is data.
func normalizeArgs(args string) string {
	s := strings.TrimSpace(args)
	var v any
	if json.Unmarshal([]byte(s), &v) != nil {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return s
	}
	return string(b)
}
