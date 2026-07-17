package agent

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Proactive document surfacing — the "RAG-notify" finder.
//
// A normal RAG integration is a TOOL the model chooses to call
// (mcpToolDefs + a dispatcher). That covers the case where the model KNOWS it
// needs to search. It does NOT cover the ambient case: a user asks about a
// topic, or the assistant asserts something, and a relevant document exists
// that the model never thought to look for. Proactive retrieval closes that
// gap — it listens to the conversation and PINGS relevant docs without waiting
// for a tool call.
//
// This is the same seam every framework with the feature uses: run retrieval
// AFTER the input lands and BEFORE the model call (LangGraph's pre_model_hook,
// Claude Code's UserPromptSubmit hook, Zep's get_user_context). agentkit
// already has that seam — the NotificationPreparer (session.go), which runs at
// the top of every Turn iteration, after ClaimPending and before build. So
// proactive retrieval is just a Preparer that OBSERVES new entries, queries a
// finder, and INJECTS pointer notices. No Turn surgery, no new Store method.
//
// Neutral-seam discipline (invariant #2): agent defines only the DocFinder
// interface + the wiring. The concrete retrieval client (ragtag over MCP, an
// HTTP service, a local index) lives OUTSIDE agent — exactly like the
// Revalidator convention in prepare.go. See ragnotify/ for the MCP impl.
//
// Two deliberate divergences from a naive design, both matching industry
// practice:
//
//   - Notices are POINTERS, not payloads. A hit injects title + id + score +
//     one line, and tells the model to pull the body with the search tool it
//     already has. Stuffing full doc bodies into every turn is how you get
//     "context rot" / "lost in the middle"; claude-mem's index-plus-pull is
//     the pattern we follow. The passive channel and the explicit tool share
//     one MCP server: the notice says "doc X is relevant", the tool fetches it.
//   - Relevance is gated + capped + deduped, and reranking is NOT our job.
//     MinScore drops weak hits, MaxHits caps per pass, and a per-session
//     seen-set means a doc notifies once. The heavy quality lever —
//     over-retrieve then cross-encoder rerank — belongs SERVER-SIDE in the
//     finder (ragtag's "strategies"); a finder that returns raw vector top-k
//     will be noisy no matter what MinScore is set.

// DocFinder surfaces documents relevant to a batch of conversation text. It is
// the neutral seam: agent defines it, a sibling/host implements it (e.g. over a
// ragtag MCP search tool). texts is the content of the fresh conversation
// entries observed since the last pass (typically the latest user message and/
// or a prior assistant reply). The implementation owns query construction,
// retrieval, and — critically — RERANKING; agent only thresholds/caps/dedups.
type DocFinder interface {
	Find(ctx context.Context, texts []string) ([]DocHit, error)
}

// DocHit is one retrieved document reference. It is a POINTER, not the body:
// Line is a one-liner (title-ish / snippet), never the full text — the model
// pulls the body on demand via the search tool.
type DocHit struct {
	DocID string  // stable id — the dedup key AND what the model passes to the fetch tool
	Title string  // human title
	Score float64 // relevance; hits below FinderOpts.MinScore are dropped
	Line  string  // one-line pointer (snippet/summary); NOT the document body
}

// FinderOpts tunes the proactive preparer. Zero value is usable (see the
// per-field defaults); a host overrides what it cares about.
type FinderOpts struct {
	// MinScore drops hits below this relevance. Default 0 (keep all) — set it,
	// or the passive channel is a firehose.
	MinScore float64
	// MaxHits caps notices injected per pass (the "hard top-k into the prompt"
	// defense against context rot). Default 3; <=0 means uncapped.
	MaxHits int
	// Tag labels the injected KindNotification (render.go prefixes "[tag] ").
	// Default "rag".
	Tag string
	// Kinds selects which entry kinds to observe. Default: KindUser +
	// KindAssistant (listen to both the user's request and the agent's reply).
	Kinds []EntryKind
	// Timeout bounds a single Find call so a slow finder can't stall the Turn.
	// On timeout (or any Find error) the pass fails OPEN: no notice, and the
	// watermark does NOT advance, so the same entries are retried next pass.
	// Default 3s; <=0 means no timeout (inherit the ctx deadline only).
	Timeout time.Duration
	// Render formats a hit into the notice body. Default: a pointer line that
	// tells the model to fetch the body with the search tool.
	Render func(DocHit) string
	// Now overrides the clock (tests). Default time.Now().UnixNano.
	Now func() int64
}

// defaultRender is the pointer-not-payload notice body.
func defaultRender(h DocHit) string {
	return fmt.Sprintf(
		"possibly-relevant document: %q (id=%s, score=%.2f) — %s. "+
			"Call the document search tool with this id to read the full text if it helps.",
		h.Title, h.DocID, h.Score, h.Line)
}

// FinderPreparer returns a NotificationPreparer that, before each Turn
// iteration, observes conversation entries newer than a per-session watermark,
// asks the DocFinder for relevant documents, and injects one POINTER
// notification per fresh (unseen, above-threshold) hit — capped at MaxHits.
//
// Timing: a user message is observed and its hits injected in the SAME
// iteration, so the model sees the request and the doc pointers together,
// before it replies. An assistant reply is observed on the NEXT pass (the loop
// only re-enters a preparer when the turn continues or a new Turn starts), so
// an assistant-triggered hit surfaces on the following user Turn — the same
// one-turn lag inherent to the event-driven model.
//
// Dedup is per-session and cumulative: a DocID notifies at most once for the
// life of this preparer instance (in-memory state, like MCPPreparer's clearing
// leaves state host-side). A host that recreates the preparer per Turn re-
// observes history and re-notifies; keep one preparer per live session.
//
// Fail-open: a Find error/timeout skips the pass WITHOUT advancing the
// watermark, so a transient finder blip retries next pass rather than dropping
// the hit. A Store.Append error DOES abort (a broken store is not recoverable
// mid-Turn), matching the rest of the loop.
func FinderPreparer(store Store, finder DocFinder, opts FinderOpts) NotificationPreparer {
	if opts.MaxHits == 0 {
		opts.MaxHits = 3
	}
	if opts.Tag == "" {
		opts.Tag = "rag"
	}
	if len(opts.Kinds) == 0 {
		opts.Kinds = []EntryKind{KindUser, KindAssistant}
	}
	if opts.Timeout == 0 {
		opts.Timeout = 3 * time.Second
	}
	if opts.Render == nil {
		opts.Render = defaultRender
	}
	now := opts.Now
	if now == nil {
		now = func() int64 { return time.Now().UnixNano() }
	}
	observed := make(map[EntryKind]bool, len(opts.Kinds))
	for _, k := range opts.Kinds {
		observed[k] = true
	}

	var mu sync.Mutex
	watermark := map[string]int64{}          // sessionID → max observed CreatedAt
	seen := map[string]map[string]bool{}     // sessionID → set of notified DocIDs

	return PreparerFunc(func(ctx context.Context, sessionID string) error {
		entries, err := store.Context(ctx, sessionID)
		if err != nil {
			return err
		}

		mu.Lock()
		mark := watermark[sessionID]
		seenIDs := seen[sessionID]
		if seenIDs == nil {
			seenIDs = map[string]bool{}
			seen[sessionID] = seenIDs
		}
		mu.Unlock()

		// Collect fresh, observable text since the watermark.
		var texts []string
		var maxSeen int64 = mark
		for _, e := range entries {
			if e.CreatedAt <= mark || !observed[e.Kind] {
				continue
			}
			if s := e.Content; s != "" {
				texts = append(texts, s)
			}
			if e.CreatedAt > maxSeen {
				maxSeen = e.CreatedAt
			}
		}
		if len(texts) == 0 {
			return nil // nothing new to search on
		}

		fctx := ctx
		if opts.Timeout > 0 {
			var cancel context.CancelFunc
			fctx, cancel = context.WithTimeout(ctx, opts.Timeout)
			defer cancel()
		}
		hits, err := finder.Find(fctx, texts)
		if err != nil {
			return nil // fail-open: keep the watermark so we retry these entries
		}

		// Threshold, then best-first, so MaxHits keeps the strongest.
		kept := hits[:0:0]
		for _, h := range hits {
			if h.Score >= opts.MinScore && h.DocID != "" && !seenIDs[h.DocID] {
				kept = append(kept, h)
			}
		}
		sort.SliceStable(kept, func(i, j int) bool { return kept[i].Score > kept[j].Score })

		injected := 0
		for _, h := range kept {
			if opts.MaxHits > 0 && injected >= opts.MaxHits {
				break
			}
			if err := store.Append(ctx, sessionID, Entry{
				ID:        uuid.New().String(),
				Kind:      KindNotification,
				Tag:       opts.Tag,
				Content:   opts.Render(h),
				CreatedAt: now(),
			}); err != nil {
				return err
			}
			seenIDs[h.DocID] = true
			injected++
		}

		// Advance the watermark only after a successful pass so a fresh entry is
		// never both searched-and-lost on a finder error.
		mu.Lock()
		if maxSeen > watermark[sessionID] {
			watermark[sessionID] = maxSeen
		}
		mu.Unlock()
		return nil
	})
}
