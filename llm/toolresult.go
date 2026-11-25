package llm

// ToolResult is what one tool call returned: text, and optionally media parts
// (image, audio, video, file — any ContentPart kind, including kinds added
// later). It lives in llm, not agent, so a toolset that imports only llm (files,
// taskgraph) can return media without importing agent.
//
// Text is the result the model reads. Parts, when non-empty, are the media the
// result carries; a text part in Parts is allowed but Text is the usual place
// for words. How the parts reach the wire depends on the endpoint: inside the
// tool message where the endpoint accepts that kind there, else in one user
// message right after the round's tool messages. See the routing in the client.
type ToolResult struct {
	Text  string
	Parts []ContentPart
}

// TextResult is a ToolResult with text only.
func TextResult(s string) ToolResult { return ToolResult{Text: s} }
