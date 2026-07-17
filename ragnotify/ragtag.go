// Package ragnotify implements agent.DocFinder over an MCP retrieval server
// (e.g. ragtag) — the concrete half of agentkit's proactive-retrieval seam.
//
// agent owns the neutral machinery (DocFinder + FinderPreparer in
// agent/finder.go): observe the conversation, gate/cap/dedup, inject pointer
// notices. This package owns the one thing agent must NOT (invariant #2: agent
// imports only llm + mcpmgr + stdlib) — talking to an actual retrieval backend.
// It depends on agent + mcpmgr only.
//
// The whole integration is: point the SAME MCP server at both channels.
//   - Explicit: advertise the server's search tool to the model normally
//     (the mcpToolDefs bridge). The model calls it when it knows to.
//   - Passive: wrap that same tool in an MCPFinder and hand it to
//     agent.FinderPreparer. A pointer notice says "doc X looks relevant"; the
//     model fetches the body with the tool it already has.
//
// Reranking is the server's job. This finder passes the conversation text
// straight to the tool and trusts the ordering/scores it returns — agent's
// MinScore/MaxHits only trims a pre-ranked list. If ragtag returns raw vector
// top-k without a reranker, expect noise regardless of MinScore.
package ragnotify

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/iodesystems/agentkit/agent"
	"github.com/iodesystems/agentkit/mcpmgr"
)

// Opts configures how the conversation text maps onto an MCP tool call and how
// the tool's JSON result maps back onto []agent.DocHit. All fields default; a
// backend with different field names overrides them without touching agent.
type Opts struct {
	// QueryArg is the tool argument that carries the query text. Default "query".
	QueryArg string
	// ExtraArgs are merged into every call (e.g. {"top_k": 8, "strategy":
	// "hybrid"}) — this is where you pin the server's rerank strategy.
	ExtraArgs map[string]any
	// Join combines the observed entry texts into one query string. Default:
	// newline-join (most recent conversation turns as one query).
	Join func(texts []string) string
}

// MCPFinder returns an agent.DocFinder backed by an MCP search tool on mgr.
// serverID + toolName name the ragtag server and its search tool (the same tool
// you advertise to the model for the explicit channel).
func MCPFinder(mgr *mcpmgr.Manager, serverID, toolName string, opts Opts) agent.DocFinder {
	if opts.QueryArg == "" {
		opts.QueryArg = "query"
	}
	if opts.Join == nil {
		opts.Join = func(texts []string) string { return strings.Join(texts, "\n") }
	}
	return &mcpFinder{mgr: mgr, serverID: serverID, toolName: toolName, opts: opts}
}

type mcpFinder struct {
	mgr      *mcpmgr.Manager
	serverID string
	toolName string
	opts     Opts
}

func (f *mcpFinder) Find(ctx context.Context, texts []string) ([]agent.DocHit, error) {
	args := map[string]any{f.opts.QueryArg: f.opts.Join(texts)}
	maps.Copy(args, f.opts.ExtraArgs)
	raw, err := f.mgr.CallTool(ctx, f.serverID, f.toolName, args)
	if err != nil {
		return nil, fmt.Errorf("ragnotify: call %s: %w", f.toolName, err)
	}
	return ParseHits(raw)
}

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
	return s[:n] + "…"
}
