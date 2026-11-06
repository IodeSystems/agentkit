package llm

import (
	"strings"
	"testing"
)

func TestParseToolCalls_EndTerminatorsDoNotLeakIntoProse(t *testing.T) {
	calls, prose, err := ParseToolCalls("@@call a\n{x: 1}\n@@end\n@@call b\n{y: 2}\n@@end\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %d; want 2", len(calls))
	}
	if prose != "" {
		t.Fatalf("prose = %q; want empty (terminators are not prose)", prose)
	}
	_, prose, err = ParseToolCalls("before\n@@call a\n{x: 1}\n@@end\nafter\n")
	if err != nil || prose != "before\n\nafter" {
		t.Fatalf("prose = %q err=%v; want text around the call only", prose, err)
	}
}

func TestParseToolCalls_CallWithoutTerminatorStillParses(t *testing.T) {
	// A stop sequence consumes the terminator, so its absence is normal.
	calls, prose, err := ParseToolCalls("@@call a\n{x: 1}\n")
	if err != nil || len(calls) != 1 || prose != "" {
		t.Fatalf("calls=%d prose=%q err=%v", len(calls), prose, err)
	}
}

func TestLooseHeredoc_BacktickFenceLanguageTagIsNotBody(t *testing.T) {
	in := "@@call write_file\n{content: ```md\n# Title\n```}\n"
	calls, _, err := ParseToolCalls(in)
	if err != nil {
		t.Fatal(err)
	}
	args := calls[0].Function.Arguments
	if strings.Contains(args, "md") || !strings.Contains(args, "# Title") {
		t.Fatalf("arguments = %s; the info string must be dropped, the body kept", args)
	}
}
