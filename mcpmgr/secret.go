package mcpmgr

import "context"

// SecretResolver resolves a secret reference (a row key in the secrets
// table) into a flat key/value map. The concrete implementation lives
// server-side (internal/server/secret_resolver.go) and wraps the
// at-rest *secrets.Store; mcpmgr intentionally does NOT import
// internal/secrets so it stays free of the crypto/db dependency edge
// and unit-testable with a fake resolver.
type SecretResolver interface {
	Resolve(ctx context.Context, ref string) (map[string]string, error)
}

// NewManagerWithSecrets builds a Manager wired to a SecretResolver.
// MCP configs that carry a non-empty SecretRef will have the resolved
// secret rendered to a 0600 temp file and substituted into Args/Env
// via the {{secret_file}} placeholder at spawn time.
func NewManagerWithSecrets(resolver SecretResolver) *Manager {
	m := NewManager()
	m.secrets = resolver
	return m
}

// SetSecretResolver attaches (or replaces) the resolver after
// construction. Used when the Manager is built before the secrets
// store is available.
func (m *Manager) SetSecretResolver(resolver SecretResolver) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.secrets = resolver
}
