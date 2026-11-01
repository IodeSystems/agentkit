package anthropicadmin

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var now = time.Date(2026, 11, 14, 9, 0, 0, 0, time.UTC)

// report answers with two pages, $1.50 and $0.25, and keeps the last request.
func report(t *testing.T) (*httptest.Server, **http.Request) {
	t.Helper()
	var got *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r
		if r.URL.Path != "/v1/organizations/cost_report" {
			t.Errorf("path %s", r.URL.Path)
		}
		if r.URL.Query().Get("page") == "" {
			fmt.Fprint(w, `{"data":[{"results":[{"amount":"100.5","currency":"USD"},{"amount":"49.5","currency":"USD"}]},{"results":[]}],"has_more":true,"next_page":"p2"}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"results":[{"amount":"25","currency":"USD"}]}],"has_more":false,"next_page":null}`)
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func keyFile(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "admin.key")
	if err := os.WriteFile(p, []byte("sk-ant-admin01-x\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func fakeAnt(t *testing.T, script string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "ant")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+script+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestMonthToDate_Key(t *testing.T) {
	srv, got := report(t)
	s, err := Source{KeyFile: keyFile(t), Base: srv.URL}.MonthToDate(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if s.USD != 1.75 || !s.Since.Equal(time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)) {
		t.Errorf("%+v", s)
	}
	r := *got
	if r.Header.Get("x-api-key") != "sk-ant-admin01-x" || r.Header.Get("Authorization") != "" {
		t.Errorf("headers %v", r.Header)
	}
	if r.URL.Query().Get("starting_at") != "2026-11-01T00:00:00Z" {
		t.Errorf("query %v", r.URL.Query())
	}
}

func TestMonthToDate_Ant(t *testing.T) {
	srv, got := report(t)
	ant := fakeAnt(t, `[ "$1 $2 $4 $5" = "auth print-credentials admin --access-token" ] && echo tok-1`)
	s, err := Source{Ant: ant, Base: srv.URL}.MonthToDate(context.Background(), now)
	if err != nil || s.USD != 1.75 {
		t.Fatalf("%+v %v", s, err)
	}
	r := *got
	if r.Header.Get("Authorization") != "Bearer tok-1" || r.Header.Get("anthropic-beta") != oauthBeta || r.Header.Get("x-api-key") != "" {
		t.Errorf("headers %v", r.Header)
	}
}

func TestHow_KeyWins(t *testing.T) {
	k, ant := keyFile(t), fakeAnt(t, "echo")
	for _, tc := range []struct {
		s    Source
		want string
	}{
		{Source{KeyFile: k, Ant: ant}, "key"},
		{Source{KeyFile: "/nonexistent", Ant: ant}, "ant"},
		{Source{KeyFile: "/nonexistent"}, ""},
		{Source{}, ""},
	} {
		if got := tc.s.How(); got != tc.want {
			t.Errorf("%+v: %q, want %q", tc.s, got, tc.want)
		}
	}
}

func TestNoCredential(t *testing.T) {
	if _, err := (Source{}).MonthToDate(context.Background(), now); !errors.Is(err, ErrNoCredential) {
		t.Errorf("neither: %v", err)
	}
	srv, _ := report(t)
	ant := fakeAnt(t, `echo "run ant auth login" >&2; exit 1`)
	_, err := Source{Ant: ant, Base: srv.URL}.MonthToDate(context.Background(), now)
	if !errors.Is(err, ErrNoCredential) || !strings.Contains(err.Error(), "ant auth login") {
		t.Errorf("not logged in: %v", err)
	}
}

func TestRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprint(w, `{"error":{"message":"nope"}}`)
	}))
	defer srv.Close()
	if _, err := (Source{KeyFile: keyFile(t), Base: srv.URL}).MonthToDate(context.Background(), now); err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("%v", err)
	}
}

func TestIsAnthropicAnt(t *testing.T) {
	ctx := context.Background()
	if !IsAnthropicAnt(ctx, fakeAnt(t, "echo ant version 2.4.0")) {
		t.Error("Anthropic's was refused")
	}
	if IsAnthropicAnt(ctx, fakeAnt(t, "echo Apache Ant version 1.10.14")) {
		t.Error("Apache Ant was taken for it")
	}
}

func TestLoginArgs(t *testing.T) {
	if got := strings.Join(Source{}.LoginArgs(true), " "); got != "auth login --profile admin --scope org:admin --no-browser" {
		t.Errorf("%s", got)
	}
}
