package agent

import (
	"context"
	"strings"
)

// Notification preparation — the "prepareNotificationsBeforeSend" seam.
//
// Before a Turn sends context to the model, the integrator gets a chance to
// revalidate the session's pending notifications and drop the stale ones, so
// the model never pays to re-validate a notice whose condition already
// resolved (the MCP-LSP waste). This is where an ACTIVE "check back" belongs —
// NOT as daemon-side shell (a privilege-escalation surface), but as an
// integrator-owned hook that runs whatever it trusts.
//
// MCP has no notification API of its own, so the reusable pattern is a
// CONVENTION: an MCP integration designates one of its tools as a
// revalidate-by-key function (masked from the model's tool list), and the
// preparer calls it per pending notification key — stale → clear. See
// MaskRevalidators / the mcprevalidate helper (host-wired, since clearing
// touches the host's notification store).

// NotificationPreparer revalidates a session's pending notifications and
// clears the stale ones, mutating the host's notification store. It runs at
// the top of every Turn iteration, after the inbox is claimed and before the
// context is built — so a resolved notice is gone before it is rendered.
type NotificationPreparer interface {
	PrepareNotifications(ctx context.Context, sessionID string) error
}

// PreparerFunc adapts a function to NotificationPreparer.
type PreparerFunc func(ctx context.Context, sessionID string) error

func (f PreparerFunc) PrepareNotifications(ctx context.Context, sessionID string) error {
	return f(ctx, sessionID)
}

// ── the MCP-revalidator convention ──────────────────────────────────
//
// MCPPreparer is the ready-made preparer for the masked-MCP-tool convention:
// for each pending notice, it calls the integration's designated revalidator
// tool for the notice's group key; an EMPTY "current truth" means the
// condition resolved → clear the notice. The two host-specific halves — where
// notices live and how a tool is called — are the small interfaces below, so
// the convention logic itself is tested here, not re-derived per host.

// PendingNotice is one revalidatable notice: its group field + the value.
type PendingNotice struct {
	GroupBy string // e.g. "file"
	Key     string // e.g. "src/main.go"
}

// RevalidateStore is the host's notice substrate.
type RevalidateStore interface {
	// PendingNotices lists the session's unshown notices carrying a group key.
	PendingNotices(ctx context.Context, sessionID string) ([]PendingNotice, error)
	// Clear retracts the notice for (groupBy, key) in the session.
	Clear(ctx context.Context, sessionID, groupBy, key string) error
}

// Revalidator calls an integration's masked revalidator tool. ok is false
// when no tool is configured for groupBy (nothing to check → keep the
// notice); otherwise result is the tool's raw output ("" = resolved).
type Revalidator interface {
	Revalidate(ctx context.Context, groupBy, key string) (result string, ok bool, err error)
}

// MCPPreparer builds a NotificationPreparer over the convention. A
// revalidation error on one notice is skipped (fail-open: keep the notice)
// so a flaky tool can't wedge the Turn.
func MCPPreparer(store RevalidateStore, rv Revalidator) NotificationPreparer {
	return PreparerFunc(func(ctx context.Context, sessionID string) error {
		notes, err := store.PendingNotices(ctx, sessionID)
		if err != nil {
			return err
		}
		for _, n := range notes {
			res, ok, err := rv.Revalidate(ctx, n.GroupBy, n.Key)
			if err != nil || !ok {
				continue // no revalidator, or a transient failure → keep
			}
			if IsResolvedTruth(res) {
				if err := store.Clear(ctx, sessionID, n.GroupBy, n.Key); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// IsResolvedTruth reports whether a revalidator's "current truth" is empty —
// i.e. the condition resolved, so the notice should be cleared. Empty means:
// blank/whitespace, or a JSON empty value (null, {}, [], "").
func IsResolvedTruth(result string) bool {
	s := strings.TrimSpace(result)
	switch s {
	case "", "null", "{}", "[]", `""`:
		return true
	}
	return false
}
