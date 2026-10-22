package mcpauth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestParseWWWAuthenticate(t *testing.T) {
	cases := []struct {
		in   string
		want []Challenge
	}{
		{`Bearer resource_metadata="https://mcp.example/.well-known/oauth-protected-resource", scope="read write"`,
			[]Challenge{{Scheme: "Bearer", Params: map[string]string{
				"resource_metadata": "https://mcp.example/.well-known/oauth-protected-resource", "scope": "read write"}}}},
		// Several challenges, unquoted values, case-insensitive names, escapes.
		{`Basic realm="a", Bearer Error=invalid_token, error_description="say \"hi\", ok"`,
			[]Challenge{
				{Scheme: "Basic", Params: map[string]string{"realm": "a"}},
				{Scheme: "Bearer", Params: map[string]string{"error": "invalid_token", "error_description": `say "hi", ok`}}}},
		{`Bearer`, []Challenge{{Scheme: "Bearer", Params: map[string]string{}}}},
		{`Negotiate abc+/9==, Bearer realm="x"`, []Challenge{
			{Scheme: "Negotiate", Params: map[string]string{}, Token68: "abc+/9=="},
			{Scheme: "Bearer", Params: map[string]string{"realm": "x"}}}},
		{`Bearer realm = "spaced" , scope="s"`, []Challenge{{Scheme: "Bearer", Params: map[string]string{"realm": "spaced", "scope": "s"}}}},
		{``, nil},
	}
	for _, c := range cases {
		if got := ParseWWWAuthenticate(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s:\n got %+v\nwant %+v", c.in, got, c.want)
		}
	}
}

func TestBearerChallengeSkipsOtherSchemes(t *testing.T) {
	c, ok := BearerChallenge(`Basic realm="a", bearer resource_metadata="https://r/m"`)
	if !ok || c.Params["resource_metadata"] != "https://r/m" {
		t.Errorf("%+v %v", c, ok)
	}
	if _, ok := BearerChallenge(`Basic realm="a"`); ok {
		t.Error("no Bearer challenge there")
	}
}

func TestWellKnownURLsInsertBeforeThePath(t *testing.T) {
	got, err := ResourceMetadataURL("https://mcp.example/tenant/mcp/")
	if err != nil || got != "https://mcp.example/.well-known/oauth-protected-resource/tenant/mcp" {
		t.Errorf("%q %v", got, err)
	}
	got, _ = ResourceMetadataURL("https://mcp.example")
	if got != "https://mcp.example/.well-known/oauth-protected-resource" {
		t.Errorf("no path: %q", got)
	}
	urls, _ := AuthServerMetadataURLs("https://as.example/realms/a%2Fb")
	want := []string{
		"https://as.example/.well-known/oauth-authorization-server/realms/a%2Fb",
		"https://as.example/.well-known/openid-configuration/realms/a%2Fb",
		"https://as.example/realms/a%2Fb/.well-known/openid-configuration",
	}
	if !reflect.DeepEqual(urls, want) {
		t.Errorf("with a path (escapes kept):\n got %v\nwant %v", urls, want)
	}
	urls, _ = AuthServerMetadataURLs("https://as.example/")
	if !reflect.DeepEqual(urls, []string{"https://as.example/.well-known/oauth-authorization-server", "https://as.example/.well-known/openid-configuration"}) {
		t.Errorf("without a path: %v", urls)
	}
	if _, err := ResourceMetadataURL("/relative"); err == nil {
		t.Error("a relative URL must be refused")
	}
}

// metaServer serves JSON documents at fixed paths and 404s the rest, and logs
// what was asked.
type metaServer struct {
	docs map[string]any
	hits []string
}

func (m *metaServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.hits = append(m.hits, r.URL.EscapedPath())
	doc, ok := m.docs[r.URL.EscapedPath()]
	if !ok {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(doc)
}

func TestFetchResourceMetadataChecksTheResource(t *testing.T) {
	ms := &metaServer{docs: map[string]any{}}
	ts := httptest.NewServer(ms)
	defer ts.Close()
	res := ts.URL + "/mcp"
	ms.docs["/.well-known/oauth-protected-resource/mcp"] = map[string]any{
		"resource": res, "authorization_servers": []string{"https://as.example"}, "scopes_supported": []string{"read"}}
	u, _ := ResourceMetadataURL(res)
	md, err := FetchResourceMetadata(context.Background(), ts.Client(), u, res)
	if err != nil || md.AuthorizationServers[0] != "https://as.example" || md.ScopesSupported[0] != "read" {
		t.Fatalf("%+v %v", md, err)
	}
	if _, err := FetchResourceMetadata(context.Background(), ts.Client(), u, ts.URL+"/other"); err == nil || !strings.Contains(err.Error(), "describes resource") {
		t.Errorf("a document for another resource must be refused: %v", err)
	}
	if _, err := FetchResourceMetadata(context.Background(), ts.Client(), ts.URL+"/missing", ""); err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("want the 404: %v", err)
	}
}

func TestFetchAuthServerMetadataFallsBackInOrder(t *testing.T) {
	ms := &metaServer{docs: map[string]any{}}
	ts := httptest.NewServer(ms)
	defer ts.Close()
	issuer := ts.URL + "/tenant"
	// Only the last candidate — OIDC with the well-known appended — exists.
	ms.docs["/tenant/.well-known/openid-configuration"] = map[string]any{
		"issuer": issuer, "authorization_endpoint": issuer + "/authorize", "token_endpoint": issuer + "/token",
		"code_challenge_methods_supported": []string{"S256"}}
	md, err := FetchAuthServerMetadata(context.Background(), ts.Client(), issuer)
	if err != nil || md.TokenEndpoint != issuer+"/token" || md.CodeChallengeMethodsSupported[0] != "S256" {
		t.Fatalf("%+v %v", md, err)
	}
	want := []string{"/.well-known/oauth-authorization-server/tenant", "/.well-known/openid-configuration/tenant", "/tenant/.well-known/openid-configuration"}
	if !reflect.DeepEqual(ms.hits, want) {
		t.Errorf("order:\n got %v\nwant %v", ms.hits, want)
	}

	// RFC 8414 first when it is there; the OIDC URLs are not asked.
	ms.hits = nil
	ms.docs["/.well-known/oauth-authorization-server/tenant"] = map[string]any{"issuer": issuer + "/", "token_endpoint": "t"}
	md, err = FetchAuthServerMetadata(context.Background(), ts.Client(), issuer)
	if err != nil || md.TokenEndpoint != "t" || len(ms.hits) != 1 {
		t.Errorf("a trailing slash on the issuer is the same issuer: %+v %v %v", md, err, ms.hits)
	}
}

func TestFetchAuthServerMetadataRefusesAnotherIssuer(t *testing.T) {
	ms := &metaServer{docs: map[string]any{"/.well-known/oauth-authorization-server": map[string]any{"issuer": "https://evil.example"}}}
	ts := httptest.NewServer(ms)
	defer ts.Close()
	if _, err := FetchAuthServerMetadata(context.Background(), ts.Client(), ts.URL); err == nil || !strings.Contains(err.Error(), "describes issuer") {
		t.Errorf("want the issuer mismatch, got %v", err)
	}
	ms.docs = map[string]any{}
	if _, err := FetchAuthServerMetadata(context.Background(), ts.Client(), ts.URL); err == nil || !strings.Contains(err.Error(), "no authorization server metadata") {
		t.Errorf("want not-found after every candidate, got %v", err)
	}
}
