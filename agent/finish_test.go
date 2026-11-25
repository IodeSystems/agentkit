package agent

import (
	"context"
	"testing"

	"github.com/iodesystems/agentkit/llm"
)

// A host building an empty-reply error needs the provider's finish words for
// the round. They must reach TurnResult, from the LAST round, verbatim.
func TestTurnResultCarriesLastRoundFinishReasons(t *testing.T) {
	store := &memStore{}
	runner := &scriptRunner{turns: [][]llm.StreamChunk{
		sayAndCall("first", "c1"),
		{{Content: "done"}, {Done: true, FinishReason: "stop", NativeFinishReason: "MALFORMED_FUNCTION_CALL"}},
	}}
	s := &Session{
		SessionID: "s", Store: store, Runner: runner, MaxTurns: 10,
		Dispatch: func(context.Context, llm.ToolCall) (llm.ToolResult, error) { return llm.TextResult("ok"), nil },
	}
	res, err := s.Turn(context.Background())
	if err != nil {
		t.Fatalf("Turn: %v", err)
	}
	if res.FinishReason != "stop" || res.NativeFinishReason != "MALFORMED_FUNCTION_CALL" {
		t.Errorf("TurnResult finish = %q/%q, want stop/MALFORMED_FUNCTION_CALL", res.FinishReason, res.NativeFinishReason)
	}
}
