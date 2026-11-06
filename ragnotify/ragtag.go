// Package ragnotify parses an MCP retrieval server's search result (e.g.
// ragtag, raglit) into []agent.DocHit, for a host's agent.DocFinder.
//
// Reranking is the server's job: ParseHits trusts the ordering and scores it
// returns, and agent's MinScore/MaxHits only trims a pre-ranked list. If the
// server returns raw vector top-k without a reranker, expect noise regardless
// of MinScore.
package ragnotify

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/iodesystems/agentkit/agent"
)

// ragHit is the expected per-document shape. Field names are lenient — a hit
// may spell the id/score/snippet several common ways.
type ragHit struct {
	DocID   string  `json:"doc_id"`
	ID      string  `json:"id"`
	Title   string  `json:"title"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet"`
	Text    string  `json:"text"`
	Content string  `json:"content"`
}

func (h ragHit) toDocHit() agent.DocHit {
	id := h.DocID
	if id == "" {
		id = h.ID
	}
	line := firstNonEmpty(h.Snippet, h.Text, h.Content)
	return agent.DocHit{DocID: id, Title: h.Title, Score: h.Score, Line: clip(line, 240)}
}

// ParseHits maps a ragtag search result string onto []agent.DocHit. It accepts
// either a bare JSON array of hits or an object wrapping them under "hits" /
// "results" / "documents" — the shapes retrieval servers commonly return. A
// hit missing an id is dropped (the id is the dedup + fetch key). Exported so a
// host with a different wire shape can test its own adapter against this one.
func ParseHits(raw string) ([]agent.DocHit, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}
	var rows []ragHit
	if s[0] == '[' {
		if err := json.Unmarshal([]byte(s), &rows); err != nil {
			return nil, fmt.Errorf("ragnotify: parse hit array: %w", err)
		}
	} else {
		var wrap struct {
			Hits      []ragHit `json:"hits"`
			Results   []ragHit `json:"results"`
			Documents []ragHit `json:"documents"`
		}
		if err := json.Unmarshal([]byte(s), &wrap); err != nil {
			return nil, fmt.Errorf("ragnotify: parse hit object: %w", err)
		}
		switch {
		case len(wrap.Hits) > 0:
			rows = wrap.Hits
		case len(wrap.Results) > 0:
			rows = wrap.Results
		default:
			rows = wrap.Documents
		}
	}
	out := make([]agent.DocHit, 0, len(rows))
	for _, r := range rows {
		h := r.toDocHit()
		if h.DocID == "" {
			continue // no stable id → can't dedup or fetch
		}
		out = append(out, h)
	}
	return out, nil
}

func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	// Back off to a rune boundary: a cut inside a rune is invalid UTF-8.
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n] + "…"
}
