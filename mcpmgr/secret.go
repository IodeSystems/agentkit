package mcpmgr

import "context"

// SecretResolver resolves a secret reference into a flat key/value map. The
// host implements it over its own secret store, so mcpmgr carries no crypto or
// database dependency.
type SecretResolver interface {
	Resolve(ctx context.Context, ref string) (map[string]string, error)
}

// SetSecretResolver attaches (or replaces) the resolver. An MCPConfig with a
// non-empty SecretRef then has the resolved secret rendered to a 0600 temp file
// and substituted into Args/Env via the {{secret_file}} placeholder at spawn.
func (m *Manager) SetSecretResolver(resolver SecretResolver) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.secrets = resolver
}
