package files

import (
	"fmt"

	"github.com/iodesystems/agentkit/llm"
)

// Descriptions ride every request, so they say what changes a call and
// nothing else.

func (t *Toolset) def(bare string) llm.ToolDef {
	var td llm.ToolDef
	td.Type = "function"
	td.Function.Name = t.Name(bare)
	str := func(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }
	num := func(desc string) map[string]any { return map[string]any{"type": "integer", "description": desc} }
	boolean := func(desc string) map[string]any { return map[string]any{"type": "boolean", "description": desc} }
	obj := func(props map[string]any, required ...string) map[string]any {
		o := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			o["required"] = required
		}
		return o
	}
	pathDesc := "Path relative to the working tree."
	switch bare {
	case Read:
		// The list and grep tools' other arguments (max, files, ignore_case,
		// and pattern for grep) are accepted, unadvertised: every property
		// here rides every request.
		td.Function.Description = fmt.Sprintf("Read a file or a range of its lines, list a directory, or with grep search a file or "+
			"a directory tree. Search a large file or a log with grep before you read it. Lines come back as number, TAB, "+
			"text; leave the number and tab out of edit anchors. With no offset/limit you get the first %s; a ranged "+
			"read returns up to %s.", sizeOf(int64(t.cfg.UnrangedMax)), sizeOf(int64(t.cfg.ReadMax)))
		td.Function.Parameters = obj(map[string]any{
			"path":    str("File or directory, relative to the working tree."),
			"offset":  num("First line, from 1. Negative counts from the end: -50 is the last 50 lines."),
			"limit":   num("How many lines. Omit for the rest of the file (still capped)."),
			"grep":    str("Regular expression (RE2). Returns only the matching lines, as path:line:text."),
			"context": num("With grep: lines before and after each match (max 10)."),
			"glob":    str("On a directory: only files matching this glob (*.go at any depth; with a /, the path: internal/**/*_test.go)."),
			"sizes":   boolean("On a directory: show file sizes."),
		}, "path")
	case Edit:
		td.Function.Description = fmt.Sprintf("Change a file by exact replacement. Each old must occur EXACTLY ONCE in the file "+
			"(add surrounding lines to make it unique); edits apply in order, all or none. Copy old from a %s of the "+
			"span — a few lines, without the line-number prefix. Returns the diff.", t.Name(Read))
		// One shape, edits required: with old/new and edits both optional the
		// schema required only path, and a model that fills what is required
		// sent just that (DeepSeek, 2026-10-22). old/new at the top level are
		// still accepted, unadvertised.
		td.Function.Parameters = obj(map[string]any{
			"path": str(pathDesc),
			"edits": map[string]any{
				"type":        "array",
				"minItems":    1,
				"description": "Replacements, applied in order: [{old, new}, …]; one item for a single edit.",
				"items": obj(map[string]any{
					"old": str("Exact text to replace."),
					"new": str("Replacement text; empty deletes."),
				}, "old", "new"),
			},
		}, "path", "edits")
	case Write:
		td.Function.Description = fmt.Sprintf("Create a file, or replace one whole. Parent directories are created. Replacing a file "+
			"you have not read in this session is refused unless overwrite is true; for a change to an existing file use %s.",
			t.Name(Edit))
		td.Function.Parameters = obj(map[string]any{
			"path":      str(pathDesc),
			"content":   str("The whole file."),
			"overwrite": boolean("Replace an existing file you have not read."),
		}, "path", "content")
	case Grep:
		td.Function.Description = fmt.Sprintf("Search file contents with a regular expression (RE2 syntax; an invalid one is searched "+
			"literally). Skips .gitignored files, binaries and .git. Output is path:line:text, line numbers usable "+
			"as %s offset; context lines are path-line-text. The first line gives the totals.", t.Name(Read))
		td.Function.Parameters = obj(map[string]any{
			"pattern":     str("Regular expression."),
			"path":        str("Directory or file to search; default the whole tree."),
			"glob":        str("Only files matching this glob: *.go matches the name at any depth; a glob with / matches the path (internal/**/*_test.go)."),
			"files":       boolean("List matching file names only."),
			"context":     num("Lines of context around each match (max 10)."),
			"max":         num(fmt.Sprintf("Most matches to show (default %d).", t.cfg.GrepMax)),
			"ignore_case": boolean("Case-insensitive."),
		}, "pattern")
	case List:
		td.Function.Description = "List a directory (one level; directories end in /), or with glob every matching file beneath it. " +
			".gitignored entries are left out."
		td.Function.Parameters = obj(map[string]any{
			"path":  str("Directory; default the root."),
			"glob":  str("Recursive file filter, e.g. **/*.go or *_test.go."),
			"sizes": boolean("Show file sizes."),
			"max":   num(fmt.Sprintf("Most entries (default %d).", t.cfg.ListMax)),
		})
	case Delete:
		td.Function.Description = "Delete one file. Directories are refused."
		td.Function.Parameters = obj(map[string]any{"path": str(pathDesc)}, "path")
	case Move:
		td.Function.Description = "Move or rename a file or directory. Refuses to replace an existing path."
		td.Function.Parameters = obj(map[string]any{
			"path": str("What to move, relative to the working tree."),
			"to":   str("New path, relative to the working tree."),
		}, "path", "to")
	}
	return td
}
