package llm

import "testing"

// The real body llama.cpp returned when a 19KB file write ran past the context
// window mid-argument. Kept verbatim (trimmed) because the whole point of the
// classifier is to recognise THIS shape and not retry it.
const realTruncationBody = `{"error":{"code":500,"message":"Failed to parse tool call arguments as JSON: ` +
	`[json.exception.parse_error.101] parse error at line 1, column 19315: syntax error while parsing value ` +
	`- invalid string: missing closing quote; last read: '\"package com.termux.app;\\n\\npublic class Foo {\\n'","type":"server_error"}}`

func TestBodyIsTruncatedToolCall(t *testing.T) {
	if !bodyIsTruncatedToolCall(realTruncationBody) {
		t.Fatal("did not recognise the observed truncation body")
	}
	// A generic 5xx must stay RETRYABLE: only the tool-arg parse shape is known
	// to be deterministic, and misclassifying kills recoverable failures.
	for _, retryable := range []string{
		"",
		`{"error":{"message":"internal server error"}}`,
		`{"error":{"message":"upstream timeout"}}`,
		`{"error":{"message":"model is loading"}}`,
		`{"error":{"message":"failed to parse request body"}}`, // parse, but not a tool call
	} {
		if bodyIsTruncatedToolCall(retryable) {
			t.Errorf("classified a retryable 5xx as terminal: %q", retryable)
		}
	}
}

func contains(h, n string) bool {
	for i := 0; i+len(n) <= len(h); i++ {
		if h[i:i+len(n)] == n {
			return true
		}
	}
	return false
}
