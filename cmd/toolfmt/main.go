// Command toolfmt exposes the agent/toolfmt encoders as a stdin->stdout CLI.
//
// Usage:
//
//	toolfmt <format>   # reads JSON on stdin, writes the encoded form on stdout
//
// format is one of: json tightc
// json is the identity encoding (stdin passed through unchanged).
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/iodesystems/agentkit/agent/toolfmt"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: toolfmt <json|tightc>")
		os.Exit(2)
	}

	in, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "toolfmt: read stdin:", err)
		os.Exit(1)
	}
	raw := string(in)

	var out string
	switch os.Args[1] {
	case "json":
		out = raw
	case "tightc":
		out = toolfmt.EncodeTightC(raw)
	default:
		fmt.Fprintln(os.Stderr, "toolfmt: unknown format:", os.Args[1])
		os.Exit(2)
	}

	if _, err := io.WriteString(os.Stdout, out); err != nil {
		fmt.Fprintln(os.Stderr, "toolfmt: write stdout:", err)
		os.Exit(1)
	}
}
