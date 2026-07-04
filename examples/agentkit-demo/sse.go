package main

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"

	"github.com/iodesystems/agentkit/llm"
)

// serve: relay a live completion to any HTTP client as Server-Sent Events using
// llm.StreamChunkToSSE — the batteries-included "stream agent tokens to a
// browser" helper. The whole relay is the handler below; StreamChunkToSSE does
// the framing (content deltas, tool-call events, [DONE], errors), so an
// integrator wiring a web UI writes almost nothing.
func runServe(ctx context.Context, cfg config) error {
	client := cfg.client()

	mux := http.NewServeMux()
	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		if q == "" {
			q = "In two sentences, what is an agent tool-call loop?"
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		ch, err := client.ChatStream(r.Context(), []llm.Message{{Role: "user", Content: q}}, nil, nil)
		if err != nil {
			fmt.Fprintf(w, "data: {\"type\":\"error\",\"text\":%q}\n\n", err.Error())
			flusher.Flush()
			return
		}
		// The entire relay: format each chunk as an SSE frame and flush.
		for chunk := range ch {
			if frame := llm.StreamChunkToSSE(chunk); frame != "" {
				_, _ = w.Write([]byte(frame))
				flusher.Flush()
			}
		}
	})

	// Listen on an ephemeral port unless --addr is given, so the demo never
	// collides with something already bound.
	addr := cfg.addr
	if addr == "" {
		addr = "127.0.0.1:0"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()
	defer func() { _ = srv.Shutdown(context.Background()) }()

	url := "http://" + ln.Addr().String()
	fmt.Printf("serving SSE at %s\n", url)
	fmt.Printf("connect with:  curl -N '%s/chat?q=your+question'\n\n", url)

	// Self-request: GET the endpoint and print the raw SSE frames the client
	// receives, so running `serve` alone shows StreamChunkToSSE's wire output —
	// no second terminal needed.
	fmt.Println("── self-request (raw SSE frames received by the client) ──")
	if err := streamSSE(ctx, url+"/chat"); err != nil {
		return err
	}
	fmt.Println("─────────────────────────────────────────────────────────")

	if !cfg.keep {
		fmt.Println("\n(done — pass --keep to leave the server up for browser/curl clients)")
		return nil
	}
	fmt.Printf("\nstill serving — connect more clients. Stops at --timeout or Ctrl-C.\n")
	<-ctx.Done()
	return nil
}

// streamSSE GETs an SSE endpoint and prints each non-blank frame as it arrives —
// a stand-in for a browser's EventSource, showing the exact bytes on the wire.
func streamSSE(ctx context.Context, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.TrimSpace(line) != "" {
			fmt.Printf("  ◀ %s\n", line)
		}
	}
	return sc.Err()
}
