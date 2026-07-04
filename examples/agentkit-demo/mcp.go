package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/iodesystems/agentkit/agent"
	"github.com/iodesystems/agentkit/llm"
	"github.com/iodesystems/agentkit/mcpmgr"
)

// mcp: real MCP tool integration.
//
// mcpmgr spawns a stdio MCP server, discovers its tools, and calls them. The
// only glue an integrator writes is the two bridges below — MCPTool → llm.ToolDef
// (advertise) and a dispatcher routing tool calls to Manager.CallTool (execute).
// That's the whole "give the agent an MCP server" story.

// mcpToolDefs bridges discovered MCP tools into the OpenAI tool format the model
// is advertised. The MCP InputSchema is already a JSON Schema, so it drops
// straight into ToolDef.Parameters.
func mcpToolDefs(tools []mcpmgr.MCPTool) []llm.ToolDef {
	out := make([]llm.ToolDef, 0, len(tools))
	for _, t := range tools {
		var td llm.ToolDef
		td.Type = "function"
		td.Function.Name = t.Name
		td.Function.Description = t.Description
		td.Function.Parameters = t.InputSchema
		out = append(out, td)
	}
	return out
}

// mcpDispatcher routes a model tool call to the owning MCP server. Errors meant
// for the model (unknown tool, bad args, tool failure) are formatted INTO the
// result so the loop stays alive (the ToolDispatcher contract).
func mcpDispatcher(mgr *mcpmgr.Manager, tools []mcpmgr.MCPTool) agent.ToolDispatcher {
	serverOf := make(map[string]string, len(tools))
	for _, t := range tools {
		serverOf[t.Name] = t.ServerID
	}
	return func(ctx context.Context, tc llm.ToolCall) (string, error) {
		serverID, ok := serverOf[tc.Function.Name]
		if !ok {
			return fmt.Sprintf("ERROR: unknown tool %q", tc.Function.Name), nil
		}
		var args map[string]any
		if s := strings.TrimSpace(tc.Function.Arguments); s != "" && s != "null" {
			if err := json.Unmarshal([]byte(s), &args); err != nil {
				return fmt.Sprintf("ERROR: bad arguments: %v", err), nil
			}
		}
		res, err := mgr.CallTool(ctx, serverID, tc.Function.Name, args)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err), nil
		}
		return res, nil
	}
}

func runMCP(ctx context.Context, cfg config) error {
	// Default to the MCP reference "everything" server (tools: echo, add,
	// printEnv, …). Override the whole command via --prompt "cmd arg arg".
	command, cmdArgs := "npx", []string{"-y", "@modelcontextprotocol/server-everything"}
	if cfg.prompt != "" {
		fields := strings.Fields(cfg.prompt)
		command, cmdArgs = fields[0], fields[1:]
	}
	fmt.Printf("spawning MCP server: %s %s\n(first run downloads the package — may take a bit)\n\n",
		command, strings.Join(cmdArgs, " "))

	mgr := mcpmgr.NewManager()
	defer mgr.Close()
	if err := mgr.StartServer(ctx, mcpmgr.MCPConfig{
		ID: "everything", Name: "everything", Command: command, Args: cmdArgs, Timeout: 90,
	}); err != nil {
		return fmt.Errorf("spawn MCP server: %w", err)
	}

	// GetTools returns nothing until discovery finishes; poll briefly.
	var tools []mcpmgr.MCPTool
	for i := 0; i < 60; i++ {
		if tools = mgr.GetTools(); len(tools) > 0 {
			break
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	if len(tools) == 0 {
		return fmt.Errorf("no MCP tools discovered (server failed to start or is still downloading — raise --timeout)")
	}
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	fmt.Printf("discovered %d MCP tools: %s\n", len(tools), strings.Join(names, ", "))

	clk := &clock{}
	store := newDemoStore()
	p := "Use the echo tool to echo the exact text 'hello from agentkit', then report what it returned."
	store.publish(entry(agent.KindUser, p, clk.next()))
	fmt.Printf("\nprompt: %s\n\nassistant: ", p)

	sess := &agent.Session{
		SessionID:        "demo",
		System:           "You are an assistant with MCP tools. Call them to answer, then summarize in one line.",
		Store:            store,
		Runner:           cfg.client(),
		Tools:            mcpToolDefs(tools),
		Dispatch:         verbose(mcpDispatcher(mgr, tools)),
		Now:              clk.next,
		MaxTurns:         8,
		Tracer:           cfg.tracer(),
		OnAssistantToken: func(s string) { fmt.Print(s) },
	}
	_, err := sess.Turn(ctx)
	fmt.Println()
	if err != nil {
		return err
	}
	fmt.Println("\nThe only integration code was mcpToolDefs + mcpDispatcher — mcpmgr owns")
	fmt.Println("the process spawn, tool discovery, and CallTool routing.")
	return nil
}
