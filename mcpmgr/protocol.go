package mcpmgr

// The MCP layer: the handshake and the three request methods this package
// actually speaks, on top of the JSON-RPC transport in jsonrpc.go.
//
// The whole vocabulary is initialize + tools/list + tools/call, plus the
// notifications/initialized we owe the server and whatever notifications it
// pushes back. That subset has been byte-stable across every published
// revision of the MCP spec, which is what makes owning this code cheap.

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"strings"
)

// requestedProtocolVersion is what we offer in the handshake — the newest
// revision whose semantics for our three methods have been verified here. A
// server responds with the version it chose, which may be older.
const requestedProtocolVersion = "2025-06-18"

// maxToolPages bounds cursor-following in listTools. A server that keeps
// handing back a fresh cursor forever is broken, and an unbounded loop turns
// that into a hang at daemon boot.
const maxToolPages = 100

// toolDesc is one entry of tools/list.
//
// InputSchema is map[string]any ON PURPOSE. JSON Schema is an open vocabulary
// and a tool's schema is the only thing steering a constrained decoder: $defs,
// oneOf, additionalProperties, patternProperties, per-property descriptions.
// Decoding it into a struct with fields for {type, properties, required} —
// which is what the previous MCP library did — silently deleted the rest
// before any caller could see it. Round-trip it verbatim instead.
type toolDesc struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type listToolsResult struct {
	Tools      []toolDesc `json:"tools"`
	NextCursor string     `json:"nextCursor"`
}

// callToolResult is the tools/call reply. Content blocks are left as raw maps
// so an unrecognized block type degrades to its JSON rather than being dropped
// — a result the client cannot classify is still a result the model can read.
type callToolResult struct {
	Content           []map[string]any `json:"content"`
	IsError           bool             `json:"isError"`
	StructuredContent json.RawMessage  `json:"structuredContent,omitempty"`
}

type initializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"serverInfo"`
}

// initialize performs the MCP handshake: send initialize, then the
// notifications/initialized the server is entitled to wait for before
// answering anything else. Skipping that notification is a silent failure mode
// — a strict server accepts the connection and then refuses every call.
func (c *stdioClient) initialize(ctx context.Context, clientName, clientVersion string) (*initializeResult, error) {
	var res initializeResult
	err := c.call(ctx, "initialize", map[string]any{
		"protocolVersion": requestedProtocolVersion,
		"clientInfo":      map[string]any{"name": clientName, "version": clientVersion},
		// Present but empty: we implement no sampling, roots, or elicitation,
		// and the field is not optional.
		"capabilities": map[string]any{},
	}, &res)
	if err != nil {
		return nil, err
	}

	// Deliberately NOT policed against a list of known versions. We speak a
	// three-method subset that has not changed across revisions, so rejecting
	// a server for negotiating a version newer than this library knows about
	// would break working setups to enforce nothing. An empty version is a
	// different matter: it means the peer is not an MCP server.
	if res.ProtocolVersion == "" {
		return nil, fmt.Errorf("server negotiated no protocol version")
	}

	if err := c.notify("notifications/initialized", map[string]any{}); err != nil {
		return nil, fmt.Errorf("initialized notification: %w", err)
	}
	return &res, nil
}

// listTools enumerates the server's tools, following pagination cursors.
func (c *stdioClient) listTools(ctx context.Context, serverName string) ([]toolDesc, error) {
	var all []toolDesc
	cursor := ""
	for range maxToolPages {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		var res listToolsResult
		if err := c.call(ctx, "tools/list", params, &res); err != nil {
			return nil, err
		}
		all = append(all, res.Tools...)
		// A repeated cursor is the shape a paging bug actually takes; treat it
		// as the end rather than looping on it.
		if res.NextCursor == "" || res.NextCursor == cursor {
			return all, nil
		}
		cursor = res.NextCursor
	}
	// Return what we have rather than failing outright — a truncated tool set
	// is more useful than none — but say so, because silently advertising a
	// partial set is how a tool goes "missing" with nothing to point at.
	log.Printf("mcp: %s: stopped following tools/list cursors after %d pages; tool set may be truncated",
		serverName, maxToolPages)
	return all, nil
}

// callTool invokes one tool.
func (c *stdioClient) callTool(ctx context.Context, name string, args map[string]any) (*callToolResult, error) {
	if args == nil {
		// Send an empty object, not null: servers that unmarshal arguments
		// into a struct reject null, and a caller passing no arguments is
		// indistinguishable from one passing {}.
		args = map[string]any{}
	}
	var res callToolResult
	if err := c.call(ctx, "tools/call", map[string]any{
		"name":      name,
		"arguments": args,
	}, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// renderContent flattens MCP content blocks into the single string a tool
// result has to be on the wire to the model.
//
// The default branch is load-bearing: MCP keeps adding block types, and a
// client that recognizes only text would drop the payload of every one it does
// not know. Emitting the JSON keeps the information in front of the model even
// when this package has no idea what it is.
func renderContent(content []map[string]any) string {
	var parts []string
	for _, block := range content {
		typ, _ := block["type"].(string)
		switch typ {
		case "text":
			s, _ := block["text"].(string)
			parts = append(parts, s)
		case "image":
			data, _ := block["data"].(string)
			parts = append(parts, fmt.Sprintf("[image: %s]", data))
		default:
			b, err := json.Marshal(block)
			if err != nil {
				continue
			}
			parts = append(parts, string(b))
		}
	}
	return strings.Join(parts, "\n")
}

// contentText pulls just the text blocks out, for building an error message
// from an isError result.
func contentText(content []map[string]any) []string {
	var msgs []string
	for _, block := range content {
		if t, _ := block["type"].(string); t == "text" {
			if s, ok := block["text"].(string); ok {
				msgs = append(msgs, s)
			}
		}
	}
	return msgs
}

// normalizeSchema fills in the sub-fields a JSON Schema must have for strict
// consumers, WITHOUT rewriting the rest of it.
//
// An MCP tool with no arguments commonly advertises `{"type":"object"}` or
// omits the schema entirely. Marshalled back out with nil sub-fields that
// becomes `"required": null`, which is invalid JSON Schema — llama.cpp rejects
// the whole chat request with `type must be array, but is null`, taking every
// other tool down with it.
//
// The copy is shallow and every unrecognized key is carried over untouched:
// this normalizes, it does not reconstruct. Anything it does not know about is
// none of its business.
func normalizeSchema(raw map[string]any) map[string]any {
	out := make(map[string]any, len(raw)+3)
	maps.Copy(out, raw)
	if t, _ := out["type"].(string); t == "" {
		out["type"] = "object"
	}
	if out["properties"] == nil {
		out["properties"] = map[string]any{}
	}
	if out["required"] == nil {
		out["required"] = []any{}
	}
	return out
}
