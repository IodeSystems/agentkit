package files_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/iodesystems/agentkit/files"
	"github.com/iodesystems/agentkit/llm"
)

// A host mounts the toolset next to its own tools: Defs go in the tool list,
// Wrap decorates the dispatcher.
func Example() {
	root, _ := os.MkdirTemp("", "files-example")
	defer func() { _ = os.RemoveAll(root) }()
	_ = os.WriteFile(filepath.Join(root, "hello.txt"), []byte("hello\nworld\n"), 0o644)

	ts, err := files.New(files.Config{Root: root})
	if err != nil {
		panic(err)
	}
	var tools []llm.ToolDef
	tools = append(tools, ts.Defs()...)

	host := func(ctx context.Context, tc llm.ToolCall) (string, error) {
		return "the host's own tools", nil
	}
	dispatch := ts.Wrap(host)

	var tc llm.ToolCall
	tc.Function.Name = "file_read"
	tc.Function.Arguments = `{"path":"hello.txt"}`
	out, _ := dispatch(context.Background(), tc)
	fmt.Println(len(tools))
	fmt.Println(out)
	// Output:
	// 5
	// hello.txt: 2 lines
	// 1	hello
	// 2	world
}
