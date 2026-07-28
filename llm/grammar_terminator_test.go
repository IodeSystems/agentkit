package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// The terminator must end with a NEWLINE, or a batch is unrepresentable: the
// grammar would permit "@@end@@call" while forbidding "@@end\n@@call", and the
// model — which wants the newline — has EOS as its only legal continuation.
func TestCallRuleEndsWithNewline(t *testing.T) {
	var td ToolDef
	td.Function.Name = "read_file"
	_ = json.Unmarshal([]byte(`{"type":"object","properties":{"path":{}}}`), &td.Function.Parameters)
	g := HeredocGrammar([]ToolDef{td})

	want := `"\n` + HeredocEnd + `\n"`
	if !strings.Contains(g, want) {
		t.Errorf("call rule must end %s so a following call can start on a new line:\n%s", want, g)
	}
	if strings.Contains(g, `"\n`+HeredocEnd+`"`+"\n") {
		t.Errorf("call rule still ends without a newline:\n%s", g)
	}
}
