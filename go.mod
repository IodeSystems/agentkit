module github.com/iodesystems/agentkit

go 1.26.2

require github.com/iodesystems/openai-claude-api-shim v0.1.1

// Carries the Anthropic SDK and its 72 modules, which broke invariant 0.
retract v0.17.0
