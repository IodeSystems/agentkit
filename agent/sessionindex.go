package agent

// sessionindex.go — what leaves the context stays searchable.
//
// A fold deletes the entries it summarizes (Store.Compact), and a recap moves
// its span out of the transcript. The summary stays in context; the detail it
// was written from does not. A SessionIndex keeps that detail where the model
// can find it again: the subsumed entries are the document, and the summary the
// model already wrote is its description, so the indexer has no summary of its
// own to write.
//
// An interface because agentkit cannot depend on an indexer (raglit, the one
// this was written for, imports agentkit). The implementation owns the naming
// and the lifetime: which index, which prefix a session's documents sit under,
// which sessions a search covers (a session that inherits another's transcript
// searches both), and what Close removes.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// SessionIndex is where folded and recapped history goes.
type SessionIndex interface {
	// Put stores one document. It runs inside a turn, after a fold or a recap
	// has applied, so it must return quickly: queue the work, do not wait for
	// it. An error does not undo the fold.
	Put(ctx context.Context, d SessionDoc) error
	// Search returns the n best matches for q.
	Search(ctx context.Context, q string, n int) ([]SessionHit, error)
	// Close is the session's terminal end: remove what it put, or leave it to
	// expire, as the implementation was configured. A host calls it once, and
	// never for a session that pauses to be continued.
	Close(ctx context.Context) error
}

// SessionDoc is one span of history that left the context.
type SessionDoc struct {
	// Key names the document within the session ("fold-<id>", "recap-<id>").
	// The implementation places it under the session's own prefix.
	Key string
	// Kind is "fold" or "recap".
	Kind string
	// Name is a short title: the task the span was worked under, or its first
	// request.
	Name string
	// Summary is what replaced the span in context.
	Summary string
	// Body is the span itself, one event per line.
	Body string
}

// SessionHit is one search result.
type SessionHit struct {
	Key     string  `json:"key"`
	Name    string  `json:"name,omitempty"`
	Snippet string  `json:"snippet"`
	Score   float64 `json:"score,omitempty"`
}

// Kinds of SessionDoc.
const (
	SessionDocFold  = "fold"
	SessionDocRecap = "recap"
)

// PutSpan stores entries that left the context, with the summary that replaced
// them. The Shaper calls it for a fold; a host that applies recaps calls it
// with the recap's span and summary.
func PutSpan(ctx context.Context, idx SessionIndex, kind, key, name, summary string, entries []Entry) error {
	if idx == nil || len(entries) == 0 {
		return nil
	}
	if strings.TrimSpace(name) == "" {
		name = spanTitle(entries)
	}
	return idx.Put(ctx, SessionDoc{
		Key:     key,
		Kind:    kind,
		Name:    oneLine(name, 120),
		Summary: summary,
		Body:    eventLines(entries),
	})
}

// eventLines is the span as the summarizer read it: numbered, labelled events.
func eventLines(entries []Entry) string {
	var b strings.Builder
	for i, e := range entries {
		label := string(e.Kind)
		if e.Tag != "" {
			label = e.Tag
		}
		fmt.Fprintf(&b, "%d. [%s] %s\n", i+1, label, e.Content)
	}
	return b.String()
}

// spanTitle is the span's first request, else its first line of anything.
func spanTitle(entries []Entry) string {
	for _, e := range entries {
		if e.Kind == KindUser && strings.TrimSpace(e.Content) != "" {
			return e.Content
		}
	}
	for _, e := range entries {
		if strings.TrimSpace(e.Content) != "" {
			return e.Content
		}
	}
	return "history"
}

// SessionSearchName is the session search tool's name.
const SessionSearchName = "session_search"

// SessionSearchTool is a tool that searches idx, and its handler. A host
// offers it beside its own tools when it sets Shaper.Index.
func SessionSearchTool(idx SessionIndex) (llm.ToolDef, func(ctx context.Context, args string) string) {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = SessionSearchName
	td.Function.Description = "Search this session's earlier history that was summarized out of your context: " +
		"tool results, file contents and messages that a summary replaced. Use it when a summary mentions " +
		"something you need the detail of."
	td.Function.Parameters = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "description": "What to find."},
			"limit": map[string]any{"type": "integer", "description": "How many results (default 5)."},
		},
		"required": []string{"query"},
	}
	return td, func(ctx context.Context, args string) string {
		var a struct {
			Query string `json:"query"`
			Limit int    `json:"limit"`
		}
		if err := json.Unmarshal([]byte(args), &a); err != nil || strings.TrimSpace(a.Query) == "" {
			return "ERROR: session_search needs a query"
		}
		if a.Limit <= 0 {
			a.Limit = 5
		}
		hits, err := idx.Search(ctx, a.Query, a.Limit)
		if err != nil {
			return "ERROR: session_search: " + err.Error()
		}
		if len(hits) == 0 {
			return "No earlier history matches."
		}
		var b strings.Builder
		for _, h := range hits {
			fmt.Fprintf(&b, "## %s", h.Key)
			if h.Name != "" {
				b.WriteString(" — " + h.Name)
			}
			b.WriteString("\n" + strings.TrimSpace(h.Snippet) + "\n\n")
		}
		return strings.TrimRight(b.String(), "\n")
	}
}
