package mcpmgr

// Per-request authorization for the Streamable HTTP transport. A static
// Authorization in MCPConfig.Headers is fixed for the life of the connection;
// an OAuth access token or an IdP's JWT expires, and a client that can only
// reconnect to use a new one loses its session. So the token is asked for on
// every request (MCPConfig.Authorization — the caller caches it), and a 401
// runs MCPConfig.OnUnauthorized and retries the request once.
//
// This package does no OAuth itself: no browser, no token storage. The
// mcpauth subpackage has the discovery steps every MCP OAuth client needs
// (WWW-Authenticate, RFC 9728, RFC 8414) as pure functions.

import (
	"context"
	"errors"
	"fmt"
)

// ErrUnauthorized is what errors.Is matches for an *UnauthorizedError: the
// server answered 401 and a refresh, if there was a hook to try one, did not
// help. It means "log in again", not "the server is broken".
var ErrUnauthorized = errors.New("mcp: unauthorized")

// UnauthorizedError is a 401 that survived the refresh-and-retry.
// WWWAuthenticate is the challenge as the server sent it; an MCP server puts
// resource_metadata in it (RFC 9728), which is where OAuth discovery starts —
// see mcpauth.ParseWWWAuthenticate.
type UnauthorizedError struct {
	URL             string
	WWWAuthenticate string
	Body            string // the first 512 bytes of the response body
}

func (e *UnauthorizedError) Error() string {
	return fmt.Sprintf("mcp: HTTP 401 from %s — the server refused the credentials: %s", e.URL, e.Body)
}

func (e *UnauthorizedError) Is(target error) bool { return target == ErrUnauthorized }

// authHooks carries MCPConfig.Authorization and OnUnauthorized into the HTTP
// client, plus the gate that makes concurrent 401s run one refresh.
type authHooks struct {
	Authorization  func(ctx context.Context) (string, error)
	OnUnauthorized func(ctx context.Context, wwwAuthenticate string) error
	// gate is a one-slot semaphore rather than a mutex so a caller waiting
	// behind a refresh (which may be an interactive login) can give up on
	// its own ctx.
	gate chan struct{}
}

func newAuthHooks(authz func(context.Context) (string, error), onUnauth func(context.Context, string) error) *authHooks {
	return &authHooks{Authorization: authz, OnUnauthorized: onUnauth, gate: make(chan struct{}, 1)}
}

func (a *authHooks) authorization(ctx context.Context) (string, error) {
	if a == nil || a.Authorization == nil {
		return "", nil
	}
	v, err := a.Authorization(ctx)
	if err != nil {
		return "", fmt.Errorf("mcp: authorization: %w", err)
	}
	return v, nil
}

func (a *authHooks) canRefresh() bool { return a != nil && a.OnUnauthorized != nil }

// refresh runs OnUnauthorized after a request sent with Authorization value
// sent got a 401 — unless another request already refreshed while this one
// was in flight, which shows as Authorization now returning something else.
// Then the retry uses the new value and the hook is not run again: N requests
// that fail together on one expired token cause one refresh, not N.
func (a *authHooks) refresh(ctx context.Context, sent, wwwAuthenticate string) error {
	select {
	case a.gate <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}
	defer func() { <-a.gate }()
	if a.Authorization != nil {
		if cur, err := a.Authorization(ctx); err == nil && cur != sent {
			return nil
		}
	}
	return a.OnUnauthorized(ctx, wwwAuthenticate)
}
