package mcpmgr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

	"gopkg.in/yaml.v3"
)

// RenderSecret serialises a flat key/value secret map into one of the
// supported on-disk formats. The rendered bytes are written to a
// per-spawn 0600 temp file whose path replaces the {{secret_file}}
// placeholder in an MCP server's Args/Env (see manager.spawn).
//
// Supported formats:
//   - "env"        — KEY=VALUE lines (gmcpail consumes this)
//   - "properties" — key=value lines (java .properties variant)
//   - "json"       — a JSON object
//   - "yaml"       — KEY: VALUE mapping
//
// Keys are sorted so output is deterministic.
func RenderSecret(format string, kv map[string]string) ([]byte, error) {
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	switch format {
	case "env", "properties":
		var buf bytes.Buffer
		for _, k := range keys {
			buf.WriteString(k)
			buf.WriteByte('=')
			buf.WriteString(kv[k])
			buf.WriteByte('\n')
		}
		return buf.Bytes(), nil

	case "json":
		// Marshal via an ordered slice so output is deterministic
		// (encoding/json already sorts map keys, but be explicit).
		ordered := make(map[string]string, len(kv))
		for _, k := range keys {
			ordered[k] = kv[k]
		}
		b, err := json.Marshal(ordered)
		if err != nil {
			return nil, fmt.Errorf("mcpmgr: render json secret: %w", err)
		}
		return b, nil

	case "yaml":
		ordered := make(map[string]string, len(kv))
		for _, k := range keys {
			ordered[k] = kv[k]
		}
		b, err := yaml.Marshal(ordered)
		if err != nil {
			return nil, fmt.Errorf("mcpmgr: render yaml secret: %w", err)
		}
		return b, nil

	default:
		return nil, fmt.Errorf("mcpmgr: unknown secret_format %q (want env|properties|json|yaml)", format)
	}
}
