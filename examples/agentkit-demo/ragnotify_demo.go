package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/iodesystems/agentkit/agent"
)

// ── ragnotify: proactive document surfacing (RAG-notify) ───────────
//
// A RAG tool covers "the model KNOWS to search". This covers the ambient case:
// the user (or the agent) mentions a topic, a relevant doc exists, and the
// model never thought to look. agent.FinderPreparer runs on the notification
// seam — after input lands, before the model call — queries a DocFinder, and
// injects POINTER notices (id + score + one line, "fetch it with the tool").
//
// This demo is OFFLINE: the DocFinder is a keyword-overlap stub over a tiny
// corpus (a real one is ragnotify.MCPFinder over a ragtag MCP server). It shows
// the pointer notice, the MinScore threshold, and per-doc dedup across passes.

func runRagnotify(ctx context.Context, _ config) error {
	clk := &clock{}
	store := newDemoStore()

	finder := &keywordFinder{docs: []stubDoc{
		{id: "auth-101", title: "Auth token refresh", body: "access tokens expire; refresh token rotates on use"},
		{id: "billing-3", title: "Invoicing", body: "invoices generate nightly; billing runs on stripe"},
		{id: "deploy-7", title: "Deploy runbook", body: "blue-green deploy; rollback flips the load balancer"},
	}}

	prep := agent.FinderPreparer(store, finder, agent.FinderOpts{
		MinScore: 0.15,
		MaxHits:  2,
		Now:      clk.next,
	})

	// Pass 1 — a user asks about auth. The preparer runs BEFORE the model would
	// reply, so the doc pointer lands in the same turn as the question.
	store.publish(entry(agent.KindUser, "why does my auth token keep expiring on refresh?", clk.next()))
	seen := runPass(ctx, prep, store, 0, "pass 1 — user asked about auth:")

	// Pass 2 — a follow-up still about auth. The SAME auth doc scores again but is
	// deduped (already notified); no new notice, no re-ping.
	store.publish(entry(agent.KindUser, "does the refresh token itself expire too?", clk.next()))
	seen = runPass(ctx, prep, store, seen, "\npass 2 — follow-up on auth (same doc → deduped):")

	// Pass 3 — topic shifts to deploys. A different doc surfaces.
	store.publish(entry(agent.KindUser, "walk me through the deploy rollback", clk.next()))
	runPass(ctx, prep, store, seen, "\npass 3 — topic shifts to deploy (new doc surfaces):")

	fmt.Println("\nEach ping is a POINTER — the model reads the full doc only if it")
	fmt.Println("calls the search tool with the id. Reranking would live in ragtag,")
	fmt.Println("not here; the finder only thresholds (MinScore), caps (MaxHits), dedups.")
	return nil
}

// runPass runs one preparer pass and prints only the [rag] notices injected
// THIS pass (indices past prevSeen), so dedup is visible: a re-surfaced doc
// injects nothing new. Returns the new total notice count.
func runPass(ctx context.Context, prep agent.NotificationPreparer, store *demoStore, prevSeen int, label string) int {
	if err := prep.PrepareNotifications(ctx, "demo"); err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return prevSeen
	}
	fmt.Println(label)
	entries, _ := store.Context(ctx, "demo")
	var rag []agent.Entry
	for _, e := range entries {
		if e.Kind == agent.KindNotification && e.Tag == "rag" {
			rag = append(rag, e)
		}
	}
	if len(rag) == prevSeen {
		fmt.Println("  (no new pings)")
	}
	for _, e := range rag[prevSeen:] {
		fmt.Printf("  [rag] %s\n", e.Content)
	}
	return len(rag)
}

// stubDoc + keywordFinder: a deterministic offline DocFinder. Score is the
// fraction of query words that appear in the doc's title+body — enough to make
// the threshold + dedup behavior observable without a real index.
type stubDoc struct{ id, title, body string }

type keywordFinder struct{ docs []stubDoc }

func (f *keywordFinder) Find(_ context.Context, texts []string) ([]agent.DocHit, error) {
	words := strings.Fields(strings.ToLower(strings.Join(texts, " ")))
	if len(words) == 0 {
		return nil, nil
	}
	var out []agent.DocHit
	for _, d := range f.docs {
		hay := strings.ToLower(d.title + " " + d.body)
		hitN := 0
		for _, w := range words {
			if len(w) >= 4 && strings.Contains(hay, w) {
				hitN++
			}
		}
		if hitN == 0 {
			continue
		}
		out = append(out, agent.DocHit{
			DocID: d.id,
			Title: d.title,
			Score: float64(hitN) / float64(len(words)),
			Line:  d.body,
		})
	}
	return out, nil
}
