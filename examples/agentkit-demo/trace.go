package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/iodesystems/agentkit/agent"
)

// stdoutTracer is a tiny agent.Tracer that prints the loop's spans (Turn,
// streamChat, Shaper.Build) as a nested tree with timings — so `--trace` makes
// the otherwise-invisible machinery of a Turn legible. A host wires its real
// tracer (OpenTelemetry, autowork3's internal/trace) onto the same interface.
type stdoutTracer struct{}

type depthKey struct{}

func (stdoutTracer) Start(ctx context.Context, name string) (agent.Span, context.Context) {
	depth, _ := ctx.Value(depthKey{}).(int)
	fmt.Printf("    %s┌ %s\n", bars(depth), name)
	return &stdoutSpan{name: name, depth: depth, t0: time.Now()},
		context.WithValue(ctx, depthKey{}, depth+1)
}

type stdoutSpan struct {
	name  string
	depth int
	t0    time.Time
	attrs []string
}

// Set keeps only the high-signal attributes so the trace stays readable.
func (s *stdoutSpan) Set(key string, value any) agent.Span {
	switch key {
	case "n_messages", "n_tool_calls", "response_chars", "total_tokens",
		"prompt_tokens", "completion_tokens", "budget_tokens":
		s.attrs = append(s.attrs, fmt.Sprintf("%s=%v", key, value))
	}
	return s
}

func (s *stdoutSpan) End()             { s.close("") }
func (s *stdoutSpan) EndError(e error) { s.close("ERROR: " + e.Error()) }

func (s *stdoutSpan) close(extra string) {
	line := fmt.Sprintf("    %s└ %s (%s)", bars(s.depth), s.name, time.Since(s.t0).Round(time.Millisecond))
	if len(s.attrs) > 0 {
		line += "  " + strings.Join(s.attrs, " ")
	}
	if extra != "" {
		line += "  " + extra
	}
	fmt.Println(line)
}

func bars(d int) string { return strings.Repeat("│ ", d) }

// tracer returns the stdout tracer when --trace is set, else nil (zero overhead).
func (c config) tracer() agent.Tracer {
	if c.trace {
		return stdoutTracer{}
	}
	return nil
}
