package files

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// matcher is a gitignore-syntax rule set: comments, blank lines, `!`
// negation, trailing-`/` directory rules, `/`-anchored rules, `*`, `?`,
// `[…]` and `**`. Rules from nested .gitignore files carry their directory as
// base and apply only beneath it. The last matching rule wins.
//
// It is used where git cannot answer: ExtraIgnore everywhere, and .gitignore
// files when the root is not a git work tree. Inside a work tree git itself
// lists the files (walk.go), which is exact by construction.
type matcher struct {
	rules []rule
}

type rule struct {
	base    string // slash dir the rule's file lives in, "" for the root
	re      *regexp.Regexp
	negate  bool
	dirOnly bool
}

func newMatcher() *matcher { return &matcher{} }

// add parses lines as one ignore file in directory base.
func (m *matcher) add(base string, lines []string) {
	for _, l := range lines {
		if r, ok := parseRule(base, l); ok {
			m.rules = append(m.rules, r)
		}
	}
}

// addFile reads a .gitignore in base (abs is its path). A missing file adds
// nothing.
func (m *matcher) addFile(base, abs string) {
	f, err := os.Open(abs)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	m.add(base, lines)
}

func parseRule(base, line string) (rule, bool) {
	line = strings.TrimSuffix(line, "\r")
	// Trailing spaces are dropped unless escaped.
	for strings.HasSuffix(line, " ") && !strings.HasSuffix(line, `\ `) {
		line = line[:len(line)-1]
	}
	if line == "" || strings.HasPrefix(line, "#") {
		return rule{}, false
	}
	r := rule{base: base}
	switch {
	case strings.HasPrefix(line, "!"):
		r.negate = true
		line = line[1:]
	case strings.HasPrefix(line, `\!`), strings.HasPrefix(line, `\#`):
		line = line[1:]
	}
	if strings.HasSuffix(line, "/") {
		r.dirOnly = true
		line = strings.TrimRight(line, "/")
	}
	if line == "" {
		return rule{}, false
	}
	anchored := strings.Contains(line, "/")
	line = strings.TrimPrefix(line, "/")
	expr := "^(?:.*/)?" + globRegex(line) + "$"
	if anchored {
		expr = "^" + globRegex(line) + "$"
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return rule{}, false // a malformed rule matches nothing, as in git
	}
	r.re = re
	return r, true
}

// globRegex translates a gitignore glob to a regexp body.
func globRegex(g string) string {
	var b strings.Builder
	for i := 0; i < len(g); i++ {
		c := g[i]
		switch {
		case c == '*' && strings.HasPrefix(g[i:], "**"):
			// `**/` at the start or after `/`: zero or more directories.
			// `/**` at the end: everything beneath. Anything else: like `*`.
			atStart := i == 0 || g[i-1] == '/'
			switch {
			case atStart && strings.HasPrefix(g[i:], "**/"):
				b.WriteString("(?:.*/)?")
				i += 2
			case atStart && i+2 == len(g):
				b.WriteString(".*")
				i++
			default:
				b.WriteString("[^/]*")
				i++
			}
		case c == '*':
			b.WriteString("[^/]*")
		case c == '?':
			b.WriteString("[^/]")
		case c == '\\' && i+1 < len(g):
			i++
			b.WriteString(regexp.QuoteMeta(string(g[i])))
		case c == '[':
			end := strings.IndexByte(g[i+1:], ']')
			if end < 0 {
				b.WriteString(`\[`)
				continue
			}
			class := g[i+1 : i+1+end]
			if strings.HasPrefix(class, "!") {
				class = "^" + class[1:]
			}
			b.WriteString("[" + strings.ReplaceAll(class, `\`, `\\`) + "]")
			i += end + 1
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	return b.String()
}

// match reports the verdict of the last rule matching rel (slash path from
// the root) itself, and whether any rule matched.
func (m *matcher) match(rel string, isDir bool) (ignored, matched bool) {
	for _, r := range m.rules {
		if r.dirOnly && !isDir {
			continue
		}
		sub := rel
		if r.base != "" {
			if !strings.HasPrefix(rel, r.base+"/") {
				continue
			}
			sub = rel[len(r.base)+1:]
		}
		if r.re.MatchString(sub) {
			ignored, matched = !r.negate, true
		}
	}
	return ignored, matched
}

// ignored reports whether rel is excluded, by itself or through an ignored
// ancestor directory (git cannot re-include a file under an excluded
// directory, and neither does this).
func (m *matcher) ignored(rel string, isDir bool) bool {
	if m == nil || len(m.rules) == 0 {
		return false
	}
	for i := 0; i < len(rel); i++ {
		if rel[i] == '/' {
			if ig, _ := m.match(rel[:i], true); ig {
				return true
			}
		}
	}
	ig, _ := m.match(rel, isDir)
	return ig
}

// globMatcher filters by a `glob` argument with rg's -g semantics: a glob with
// no slash matches a file's name at any depth; one with a slash matches the
// root-relative path (`**` for any depth).
type globMatcher struct{ re *regexp.Regexp }

func newGlob(g string) (*globMatcher, error) {
	g = strings.TrimSpace(g)
	if g == "" {
		return nil, nil
	}
	expr := "^(?:.*/)?" + globRegex(g) + "$"
	if strings.Contains(strings.TrimSuffix(g, "/"), "/") {
		expr = "^" + globRegex(strings.TrimPrefix(g, "/")) + "$"
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return nil, fmt.Errorf("glob %q is malformed: %v", g, err)
	}
	return &globMatcher{re}, nil
}

func (g *globMatcher) match(rel string) bool { return g == nil || g.re.MatchString(rel) }
