// Package anthropicadmin reads what an Anthropic organization has spent.
//
// Anthropic's replies carry tokens and no money, and a Claude Console org has no
// read-only credential for its cost report: an Admin key or an org:admin login is
// full access to the org, whatever it is used for. So a host should hold one as
// little as it can. This package takes either, and prefers a login, which leaves
// no key for the host to keep:
//
//	KeyFile  a file holding an Admin API key (sk-ant-admin…)
//	Ant      Anthropic's `ant` CLI, logged in under Profile with the org:admin scope
//
// A key file wins when it exists: somebody put it there on purpose. With neither,
// How says so and a host shows estimates.
//
// The report is daily, in UTC, and usually within 5 minutes of the request. It is
// org-wide: spend per key has another API, behind by hours.
package anthropicadmin

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// DefaultProfile is the ant profile the admin login is kept under, apart from any
// day-to-day one.
const DefaultProfile = "admin"

const (
	apiVersion = "2023-06-01"
	oauthBeta  = "oauth-2025-04-20" // a bearer token on the API needs it
)

// ErrNoCredential means neither a key file nor an ant login is set up.
var ErrNoCredential = errors.New("no Anthropic admin credential")

// ErrNotPermitted means the credential works and Anthropic refuses this account
// the report. An individual account has no Admin API, and logging in again will
// not change that, so a host should say so and not retry.
var ErrNotPermitted = errors.New("this Anthropic account has no access to the cost report")

// Source is where an admin credential comes from.
type Source struct {
	KeyFile string // path; "" or a missing file is no key
	Ant     string // path to ant; "" is not installed
	Profile string // "" is DefaultProfile
	Base    string // "" is https://api.anthropic.com
	HTTP    *http.Client
}

func (s Source) profile() string {
	if s.Profile != "" {
		return s.Profile
	}
	return DefaultProfile
}

func (s Source) key() string {
	if s.KeyFile == "" {
		return ""
	}
	b, err := os.ReadFile(s.KeyFile)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// How names the flow this source would use: "key", "ant", or "" for neither.
// "ant" says it is installed, not that it is logged in.
func (s Source) How() string {
	switch {
	case s.key() != "":
		return "key"
	case s.Ant != "":
		return "ant"
	}
	return ""
}

// IsAnthropicAnt reports whether path is Anthropic's ant and not Apache Ant, the
// Java build tool, which has the same name.
func IsAnthropicAnt(ctx context.Context, path string) bool {
	out, err := exec.CommandContext(ctx, path, "--version").CombinedOutput()
	return err == nil && !strings.Contains(string(out), "Apache")
}

// LoginArgs are the arguments of the login, for a host that runs it with a
// terminal attached. noBrowser prints a URL and reads a pasted code.
func (s Source) LoginArgs(noBrowser bool) []string {
	args := []string{"auth", "login", "--profile", s.profile(), "--scope", "org:admin"}
	if noBrowser {
		args = append(args, "--no-browser")
	}
	return args
}

// LogoutArgs are the arguments that remove the login.
func (s Source) LogoutArgs() []string { return []string{"auth", "logout", "--profile", s.profile()} }

// authorize puts the credential on req.
func (s Source) authorize(ctx context.Context, req *http.Request) error {
	if k := s.key(); k != "" {
		req.Header.Set("x-api-key", k)
		return nil
	}
	if s.Ant == "" {
		return ErrNoCredential
	}
	cmd := exec.CommandContext(ctx, s.Ant, "auth", "print-credentials", "--profile", s.profile(), "--access-token")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("%w: ant is not logged in: %s", ErrNoCredential, strings.TrimSpace(stderr.String()))
	}
	tok := strings.TrimSpace(string(out))
	if tok == "" {
		return fmt.Errorf("%w: ant printed no token", ErrNoCredential)
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("anthropic-beta", oauthBeta)
	return nil
}

// Spend is real money, as Anthropic reports it.
type Spend struct {
	USD   float64
	Since time.Time // start of the first day counted, UTC
}

// MonthToDate is what the org has spent since the first of now's month, UTC.
func (s Source) MonthToDate(ctx context.Context, now time.Time) (Spend, error) {
	now = now.UTC()
	since := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)
	cents, err := s.costCents(ctx, since)
	if err != nil {
		return Spend{}, err
	}
	return Spend{USD: cents / 100, Since: since}, nil
}

func (s Source) costCents(ctx context.Context, since time.Time) (float64, error) {
	base := s.Base
	if base == "" {
		base = "https://api.anthropic.com"
	}
	hc := s.HTTP
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	var total float64
	page := ""
	for {
		q := url.Values{"starting_at": {since.Format(time.RFC3339)}, "bucket_width": {"1d"}, "limit": {"31"}}
		if page != "" {
			q.Set("page", page)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/v1/organizations/cost_report?"+q.Encode(), nil)
		if err != nil {
			return 0, err
		}
		req.Header.Set("anthropic-version", apiVersion)
		if err := s.authorize(ctx, req); err != nil {
			return 0, err
		}
		resp, err := hc.Do(req)
		if err != nil {
			return 0, err
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		resp.Body.Close()
		if err != nil {
			return 0, err
		}
		if resp.StatusCode == http.StatusForbidden {
			return 0, fmt.Errorf("%w: %s", ErrNotPermitted, firstLine(body))
		}
		if resp.StatusCode != http.StatusOK {
			return 0, fmt.Errorf("cost report: status %d: %s", resp.StatusCode, firstLine(body))
		}
		var r struct {
			Data []struct {
				Results []struct {
					Amount   string `json:"amount"`
					Currency string `json:"currency"`
				} `json:"results"`
			} `json:"data"`
			HasMore  bool   `json:"has_more"`
			NextPage string `json:"next_page"`
		}
		if err := json.Unmarshal(body, &r); err != nil {
			return 0, fmt.Errorf("cost report: %w", err)
		}
		for _, d := range r.Data {
			for _, x := range d.Results {
				if x.Currency != "" && x.Currency != "USD" {
					return 0, fmt.Errorf("cost report: currency %q", x.Currency)
				}
				v, err := strconv.ParseFloat(x.Amount, 64)
				if err != nil {
					return 0, fmt.Errorf("cost report: amount %q", x.Amount)
				}
				total += v
			}
		}
		if !r.HasMore || r.NextPage == "" {
			return total, nil
		}
		page = r.NextPage
	}
}

func firstLine(b []byte) string {
	s, _, _ := strings.Cut(strings.TrimSpace(string(b)), "\n")
	if len(s) > 300 {
		s = s[:300]
	}
	return s
}
