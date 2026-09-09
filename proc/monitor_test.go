package proc

import (
	"strings"
	"testing"
)

/*
One renderer, whatever the process is.

This package exists because dun grew two monitors — `agent_monitor` for children
and a separate one for background jobs — taking the same (id, tail) and
answering in two different shapes. A model that has learned to read one should
not have to learn the other, and a second implementation is a second set of
edges to get wrong.
*/
func TestOneRendererReadsTheSameForEveryKind(t *testing.T) {
	job := New(1, "job", "go build ./...", "", at(0))
	job.Write([]byte("compiling\n"))
	child := New(2, "agent", "find where the registry is built", "", at(0))
	child.Write([]byte("read 3 files\n"))
	all := []*Proc{child, job} // deliberately out of order

	list := Compose(all, 0, 0, at(30))
	// Both kinds appear, and by id rather than by however the caller held them.
	if i, j := strings.Index(list, "#1"), strings.Index(list, "#2"); i < 0 || j < 0 || i > j {
		t.Errorf("the listing is not both kinds in id order:\n%s", list)
	}
	for _, want := range []string{"job", "agent", "running 30s"} {
		if !strings.Contains(list, want) {
			t.Errorf("the listing does not say %q:\n%s", want, list)
		}
	}
}

/*
Asking about one answers "how is it going", in the order a person reads it:
what it is, then what it has said.
*/
func TestAskingAboutOneGivesItsStatusThenItsOutput(t *testing.T) {
	job := New(1, "job", "suite", "", at(0))
	job.Write([]byte("FAIL pkg/b\n"))
	out := Compose([]*Proc{job}, 1, 100, at(10))

	si, ti := strings.Index(out, "#1"), strings.Index(out, "FAIL pkg/b")
	if si < 0 || ti < 0 || si > ti {
		t.Errorf("status does not come before output:\n%s", out)
	}
}

/*
A wrong id is answered with what DOES exist.

"there is no #7" alone is a wall: the caller asked because it did not know, and
the list is the thing it needed.
*/
func TestAnUnknownIDNamesWhatIsRunning(t *testing.T) {
	job := New(1, "job", "go build", "", at(0))
	out := Compose([]*Proc{job}, 7, 0, at(5))
	if !strings.Contains(out, "no #7") {
		t.Errorf("the answer does not say the id is wrong: %q", out)
	}
	if !strings.Contains(out, "#1") {
		t.Errorf("the answer does not say what IS running, so it is a wall: %q", out)
	}
}

// Nothing running says so plainly rather than returning an empty string a
// caller has to interpret.
func TestNothingRunningSaysSo(t *testing.T) {
	if out := Compose(nil, 0, 0, at(0)); !strings.Contains(out, "nothing is running") {
		t.Errorf("an empty monitor returned %q", out)
	}
}
