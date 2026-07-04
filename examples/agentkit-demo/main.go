// Command agentkit-demo showcases every major agentkit capability against a
// live OpenAI-compatible endpoint (default: iode's corrallm at
// llm.iodesystems.com). Each subcommand isolates one feature.
//
//	agentkit-demo chat       streaming chat + fair-share 429 backpressure retry
//	agentkit-demo tools      local Go tool-call loop
//	agentkit-demo schema     client-side schema validation + fix loop
//	agentkit-demo grammar    server-side constrained decoding (GBNF + json)
//	agentkit-demo inject     notification injection into an in-flight session
//	agentkit-demo lift       async tool results (lifting)
//	agentkit-demo notify     notification lifecycle: supersede / clear / preparer
//	agentkit-demo compact    context shaping: LOD truncation + compaction
//
// The model at llm.iodesystems.com is small and often BUSY — expect 429
// backpressure. That is a feature demo, not an error: the client honors the
// server's retry_after (header or JSON body) and keeps trying until the
// deadline. Set --timeout to bound a live call.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/iodesystems/agentkit/llm"
)

const (
	defaultBaseURL = "https://llm.iodesystems.com"
	defaultModel   = "Qwen3-6-27B-MPT"
)

// config holds the connection settings shared by every subcommand.
type config struct {
	baseURL string
	model   string
	apiKey  string
	timeout time.Duration
	prompt  string
	trace   bool
	addr    string
	keep    bool
}

func (c config) client() *llm.Client {
	cl := llm.NewClient(c.baseURL, c.apiKey, c.model)
	// Bound the client's own 429/5xx retry to the same budget as the ctx
	// deadline — on a super-busy box it retries (honoring retry_after) up to
	// this long, then gives up cleanly instead of hanging.
	if c.timeout > 0 {
		cl.RetryBudget = c.timeout
	}
	return cl
}

// demo is one subcommand.
type demo struct {
	name  string
	blurb string
	run   func(ctx context.Context, cfg config) error
}

var demos = []demo{
	{"chat", "streaming chat + fair-share 429 backpressure retry", runChat},
	{"serve", "SSE streaming relay (stream a completion to a browser/curl)", runServe},
	{"tools", "local Go tool-call loop", runTools},
	{"mcp", "MCP server integration (spawn, discover, call)", runMCP},
	{"schema", "client-side schema validation + fix loop", runSchema},
	{"structured", "reliable typed extraction (tool_choice + validator + terminal tool)", runStructured},
	{"grammar", "server-side constrained decoding (GBNF + json)", runGrammar},
	{"inject", "notification injection into an in-flight session", runInject},
	{"lift", "async tool results (lifting)", runLift},
	{"notify", "notification lifecycle: supersede / clear / preparer", runNotify},
	{"compact", "context shaping: LOD truncation + compaction", runCompact},
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	name := os.Args[1]
	if name == "-h" || name == "--help" || name == "help" {
		usage()
		return
	}
	var d *demo
	for i := range demos {
		if demos[i].name == name {
			d = &demos[i]
			break
		}
	}
	if d == nil {
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", name)
		usage()
		os.Exit(2)
	}

	fs := flag.NewFlagSet(name, flag.ExitOnError)
	cfg := config{}
	fs.StringVar(&cfg.baseURL, "base-url", envOr("AGENTKIT_BASE_URL", defaultBaseURL), "OpenAI-compatible base URL")
	fs.StringVar(&cfg.model, "model", envOr("AGENTKIT_MODEL", defaultModel), "model id")
	fs.StringVar(&cfg.apiKey, "api-key", os.Getenv("AGENTKIT_API_KEY"), "API key (also $AGENTKIT_API_KEY); doubles as scheduling priority")
	fs.DurationVar(&cfg.timeout, "timeout", 5*time.Minute, "deadline + client retry budget for a live call (429 retries count against it)")
	fs.StringVar(&cfg.prompt, "prompt", "", "prompt override (command-specific default otherwise)")
	fs.BoolVar(&cfg.trace, "trace", false, "print the loop's spans (Turn / streamChat / Shaper.Build) with timings")
	fs.StringVar(&cfg.addr, "addr", "", "serve: listen address (default 127.0.0.1:<ephemeral>)")
	fs.BoolVar(&cfg.keep, "keep", false, "serve: keep serving after the self-request (for browser/curl clients)")
	_ = fs.Parse(os.Args[2:])

	// Ctrl-C cancels the context — mirrors how a host aborts a Turn on thread
	// close. The client's retry loop stops promptly on ctx cancel.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, cfg.timeout)
	defer cancel()

	fmt.Printf("▶ %s — %s\n  %s · model=%s\n\n", d.name, d.blurb, cfg.baseURL, cfg.model)
	if err := d.run(ctx, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "\n✗ %s\n", explain(err))
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "agentkit-demo <command> [flags]\n\nCommands:\n")
	for _, d := range demos {
		fmt.Fprintf(os.Stderr, "  %-9s %s\n", d.name, d.blurb)
	}
	fmt.Fprintf(os.Stderr, "\nCommon flags: --base-url --model --api-key --timeout --prompt --trace\n")
	fmt.Fprintf(os.Stderr, "Env: AGENTKIT_BASE_URL AGENTKIT_MODEL AGENTKIT_API_KEY\n")
}

// explain turns a bare ctx-deadline error into the real story: the model was
// saturated and the client kept honoring backpressure until the deadline.
func explain(err error) string {
	msg := err.Error()
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(msg, "context deadline exceeded") {
		return msg + "\n  (the backend was at capacity; the client honored the 429 retry_after " +
			"and kept trying until --timeout elapsed. Raise --timeout or retry when the model is idle.)"
	}
	return msg
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
