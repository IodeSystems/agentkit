package loopdetector

import (
	"math"
	"strings"
	"unicode"
)

// BM25's usual constants: k1 is how fast a repeated word stops adding to the
// score, b how much a long sentence is marked down.
const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// maxPostings is how many phrases, the latest, one word is scored against. A word
// every sentence has would otherwise make scoring quadratic in the sentences:
// 8,000 templated ones took 15 s. A loop is recent, and exact copies do not come
// here at all.
const maxPostings = 128

// posting is one phrase that has a word, and how often. The count lives here
// rather than behind a map on the phrase because scoring reads it for every
// phrase of every word.
type posting struct {
	phrase *phrase
	tf     int
}

// phrase is one distinct sentence and how many times it has been said.
type phrase struct {
	text   string
	tokens int
	count  int
	score  float64 // match's working total
	stamp  int     // the match that score belongs to
}

// index holds the sentences of one session. It lives as long as the Session.
type index struct {
	exact    map[uint64]*phrase
	postings map[string][]posting
	phrases  int
	tokens   int
	stamp    int
}

func newIndex() *index {
	return &index{exact: map[uint64]*phrase{}, postings: map[string][]posting{}}
}

func tokenize(s string) []string {
	tokens := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for i, t := range tokens {
		tokens[i] = stem(t)
	}
	return tokens
}

// stem strips the English endings a reworded sentence changes: checks, checked
// and checking are one word. It need not make real words, only the same one
// each time, and a mistake costs one word of one match.
func stem(t string) string {
	for _, suffix := range []string{"ing", "ied", "ies", "ed", "es", "s"} {
		if len(t) >= len(suffix)+3 && strings.HasSuffix(t, suffix) {
			t = t[:len(t)-len(suffix)]
			break
		}
	}
	// checking loses "ing", stopping would keep "stopp".
	if n := len(t); n >= 4 && t[n-1] == t[n-2] && !strings.ContainsRune("aeiou", rune(t[n-1])) {
		t = t[:n-1]
	}
	// so does a final e: "use" and "using" meet at "us".
	if n := len(t); n >= 4 && t[n-1] == 'e' {
		t = t[:n-1]
	}
	return t
}

func termCounts(tokens []string) map[string]int {
	counts := make(map[string]int, len(tokens))
	for _, t := range tokens {
		counts[t]++
	}
	return counts
}

// match returns the phrase this sentence is a copy of, or a new one. kind is
// KindCycle for an exact copy or a new phrase, KindPhrase for a near one.
// similarity below zero looks for exact copies only.
func (x *index) match(tokens []string, text string, similarity float64) (*phrase, string) {
	key := hashString(strings.Join(tokens, " "))
	if p := x.exact[key]; p != nil {
		return p, KindCycle
	}
	if similarity >= 0 {
		if p := x.nearest(tokens, similarity); p != nil {
			return p, KindPhrase
		}
	}
	p := &phrase{text: text, tokens: len(tokens)}
	for t, tf := range termCounts(tokens) {
		x.postings[t] = append(x.postings[t], posting{p, tf})
	}
	x.exact[key] = p
	x.phrases++
	x.tokens += len(tokens)
	return p, KindCycle
}

// nearest returns the best-scoring phrase, if it scores at least similarity as a
// share of what the sentence scores against itself. A word never seen before
// adds to that self-score and to no other, so a sentence with new words in it
// is less like everything.
func (x *index) nearest(tokens []string, similarity float64) *phrase {
	if x.phrases == 0 {
		return nil
	}
	x.stamp++
	n := float64(x.phrases)
	avg := float64(x.tokens) / n
	weight := func(tf, length int) float64 {
		f := float64(tf)
		return f * (bm25K1 + 1) / (f + bm25K1*(1-bm25B+bm25B*float64(length)/avg))
	}
	var best *phrase
	var top, self float64
	for t, tf := range termCounts(tokens) {
		posted := x.postings[t]
		df := float64(len(posted))
		idf := math.Log(1 + (n-df+0.5)/(df+0.5))
		self += idf * weight(tf, len(tokens))
		for _, post := range posted[max(0, len(posted)-maxPostings):] {
			p := post.phrase
			if p.stamp != x.stamp {
				p.stamp, p.score = x.stamp, 0
			}
			p.score += idf * weight(post.tf, p.tokens)
			if p.score > top {
				top, best = p.score, p
			}
		}
	}
	if best == nil || top < similarity*self {
		return nil
	}
	return best
}
