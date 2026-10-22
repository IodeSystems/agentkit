// Package mcpauth has the discovery steps an MCP client needs before it can
// run OAuth against a protected server (MCP authorization, 2025-06-18 and
// later):
//
//  1. a request gets 401 with WWW-Authenticate: Bearer resource_metadata="…"
//     — ParseWWWAuthenticate / BearerChallenge;
//  2. that URL (or, without one, the well-known URL derived from the server's
//     own — ResourceMetadataURL) gives the protected-resource metadata, RFC
//     9728, which names the authorization servers — FetchResourceMetadata;
//  3. each authorization server's metadata, RFC 8414 with the OpenID Connect
//     fallback, gives the endpoints — FetchAuthServerMetadata.
//
// What comes after — dynamic registration, the browser, PKCE, the token
// exchange, storing and refreshing the token — is the host's: it owns the
// user and the disk. Plug the result into mcpmgr.MCPConfig.Authorization and
// OnUnauthorized.
//
// stdlib only, like mcpmgr.
package mcpauth

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// maxMetadataBytes bounds a metadata document. They are a few KB; a server
// that sends more is not sending metadata.
const maxMetadataBytes = 1 << 20

// Challenge is one challenge of a WWW-Authenticate header (RFC 9110 §11.6.1).
// Scheme keeps the server's case; parameter names are lower-cased (they are
// case-insensitive) and values are unquoted.
type Challenge struct {
	Scheme  string
	Params  map[string]string
	Token68 string // the rare challenge that carries one instead of params
}

// ParseWWWAuthenticate parses a WWW-Authenticate value, which may hold several
// challenges ("Bearer realm=\"a\", Basic realm=\"b\""). Several header lines
// can be joined with ", " first. It is lenient: what it cannot parse ends the
// list rather than failing it.
func ParseWWWAuthenticate(header string) []Challenge {
	p := &parser{s: header}
	var out []Challenge
	for {
		p.skip(" \t,")
		scheme := p.token()
		if scheme == "" {
			return out
		}
		ch := Challenge{Scheme: scheme, Params: map[string]string{}}
		p.skip(" \t")
		if t68, ok := p.token68(); ok {
			ch.Token68 = t68
			out = append(out, ch)
			continue
		}
		for {
			start := p.i
			p.skip(" \t")
			name := p.token()
			p.skip(" \t")
			if name == "" || !p.eat('=') {
				// Not a parameter: the next challenge's scheme, or the end.
				p.i = start
				break
			}
			p.skip(" \t")
			var val string
			if p.peek() == '"' {
				val = p.quoted()
			} else {
				val = p.token()
			}
			ch.Params[strings.ToLower(name)] = val
			p.skip(" \t")
			if !p.eat(',') {
				break
			}
		}
		out = append(out, ch)
	}
}

// BearerChallenge returns the first Bearer challenge in a WWW-Authenticate
// value. An MCP server's carries resource_metadata, and often scope, which is
// the scope to ask for (MCP 2025-11-25).
func BearerChallenge(header string) (Challenge, bool) {
	for _, c := range ParseWWWAuthenticate(header) {
		if strings.EqualFold(c.Scheme, "Bearer") {
			return c, true
		}
	}
	return Challenge{}, false
}

type parser struct {
	s string
	i int
}

func (p *parser) peek() byte {
	if p.i < len(p.s) {
		return p.s[p.i]
	}
	return 0
}

func (p *parser) eat(b byte) bool {
	if p.peek() == b && b != 0 {
		p.i++
		return true
	}
	return false
}

func (p *parser) skip(set string) {
	for p.i < len(p.s) && strings.IndexByte(set, p.s[p.i]) >= 0 {
		p.i++
	}
}

// isTchar is RFC 9110's tchar.
func isTchar(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' ||
		strings.IndexByte("!#$%&'*+-.^_`|~", b) >= 0
}

func isToken68Char(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || strings.IndexByte("-._~+/", b) >= 0
}

func (p *parser) token() string {
	start := p.i
	for p.i < len(p.s) && isTchar(p.s[p.i]) {
		p.i++
	}
	return p.s[start:p.i]
}

// token68 consumes a token68 when that is what follows: its characters, then
// '=' padding, then the end or a comma. "name=value" is a parameter, not one.
func (p *parser) token68() (string, bool) {
	start := p.i
	for p.i < len(p.s) && isToken68Char(p.s[p.i]) {
		p.i++
	}
	if p.i == start {
		return "", false
	}
	for p.eat('=') {
	}
	end := p.i
	p.skip(" \t")
	if p.i == len(p.s) || p.peek() == ',' {
		return p.s[start:end], true
	}
	p.i = start
	return "", false
}

func (p *parser) quoted() string {
	p.i++ // the opening quote
	var b strings.Builder
	for p.i < len(p.s) {
		c := p.s[p.i]
		p.i++
		switch {
		case c == '"':
			return b.String()
		case c == '\\' && p.i < len(p.s):
			b.WriteByte(p.s[p.i])
			p.i++
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// ResourceMetadata is RFC 9728 protected-resource metadata. Only the fields an
// MCP client uses are decoded.
type ResourceMetadata struct {
	Resource               string   `json:"resource"`
	AuthorizationServers   []string `json:"authorization_servers,omitempty"`
	ScopesSupported        []string `json:"scopes_supported,omitempty"`
	BearerMethodsSupported []string `json:"bearer_methods_supported,omitempty"`
	ResourceName           string   `json:"resource_name,omitempty"`
	ResourceDocumentation  string   `json:"resource_documentation,omitempty"`
	JWKSURI                string   `json:"jwks_uri,omitempty"`
}

// AuthServerMetadata is RFC 8414 authorization-server metadata (which OpenID
// Connect discovery is a superset of). Only the fields an MCP client uses are
// decoded.
type AuthServerMetadata struct {
	Issuer                            string   `json:"issuer"`
	AuthorizationEndpoint             string   `json:"authorization_endpoint,omitempty"`
	TokenEndpoint                     string   `json:"token_endpoint,omitempty"`
	RegistrationEndpoint              string   `json:"registration_endpoint,omitempty"`
	RevocationEndpoint                string   `json:"revocation_endpoint,omitempty"`
	JWKSURI                           string   `json:"jwks_uri,omitempty"`
	ScopesSupported                   []string `json:"scopes_supported,omitempty"`
	ResponseTypesSupported            []string `json:"response_types_supported,omitempty"`
	GrantTypesSupported               []string `json:"grant_types_supported,omitempty"`
	TokenEndpointAuthMethodsSupported []string `json:"token_endpoint_auth_methods_supported,omitempty"`
	CodeChallengeMethodsSupported     []string `json:"code_challenge_methods_supported,omitempty"`
	ClientIDMetadataDocumentSupported bool     `json:"client_id_metadata_document_supported,omitempty"`
}

// wellKnown inserts /.well-known/<suffix> between the host and the path of u
// (RFC 8414 §3.1, RFC 9728 §3.1): https://a.example/tenant/mcp becomes
// https://a.example/.well-known/<suffix>/tenant/mcp. A trailing slash of the
// path is dropped; the query is kept, the fragment is not.
func wellKnown(raw, suffix string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, err
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("mcpauth: %q is not an absolute URL", raw)
	}
	path := strings.TrimSuffix(u.EscapedPath(), "/")
	out := &url.URL{Scheme: u.Scheme, Host: u.Host, RawQuery: u.RawQuery}
	out.Path = "/.well-known/" + suffix
	if path != "" {
		p, _ := url.PathUnescape(path)
		out.Path += p
		out.RawPath = "/.well-known/" + suffix + path
	}
	return out, nil
}

// ResourceMetadataURL is where RFC 9728 puts a resource's metadata when the
// 401 named none: /.well-known/oauth-protected-resource inserted before the
// resource URL's path.
func ResourceMetadataURL(resource string) (string, error) {
	u, err := wellKnown(resource, "oauth-protected-resource")
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

// AuthServerMetadataURLs lists, in the order to try them, where an issuer's
// metadata may be (MCP 2025-11-25, "Authorization Server Metadata Discovery").
// For an issuer with a path, https://as.example/tenant:
//
//	https://as.example/.well-known/oauth-authorization-server/tenant
//	https://as.example/.well-known/openid-configuration/tenant
//	https://as.example/tenant/.well-known/openid-configuration
//
// and for one without, the first two without the path.
func AuthServerMetadataURLs(issuer string) ([]string, error) {
	oauth, err := wellKnown(issuer, "oauth-authorization-server")
	if err != nil {
		return nil, err
	}
	oidc, _ := wellKnown(issuer, "openid-configuration")
	urls := []string{oauth.String(), oidc.String()}
	u, _ := url.Parse(issuer)
	if path := strings.TrimSuffix(u.EscapedPath(), "/"); path != "" {
		appended := &url.URL{Scheme: u.Scheme, Host: u.Host}
		p, _ := url.PathUnescape(path)
		appended.Path = p + "/.well-known/openid-configuration"
		appended.RawPath = path + "/.well-known/openid-configuration"
		urls = append(urls, appended.String())
	}
	return urls, nil
}

// statusError is a non-200 answer to a metadata GET.
type statusError struct {
	url    string
	status int
}

func (e *statusError) Error() string { return fmt.Sprintf("mcpauth: GET %s: HTTP %d", e.url, e.status) }

func getJSON(ctx context.Context, hc *http.Client, u string, out any) error {
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxMetadataBytes))
		return &statusError{url: u, status: resp.StatusCode}
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxMetadataBytes)).Decode(out); err != nil {
		return fmt.Errorf("mcpauth: GET %s: %w", u, err)
	}
	return nil
}

// FetchResourceMetadata GETs RFC 9728 metadata from metadataURL (the 401's
// resource_metadata, or ResourceMetadataURL). When resource is not empty the
// document's resource must equal it — RFC 9728 §3.3, the check that stops one
// server from pointing a client at another's authorization server. A nil hc
// means http.DefaultClient.
func FetchResourceMetadata(ctx context.Context, hc *http.Client, metadataURL, resource string) (*ResourceMetadata, error) {
	var md ResourceMetadata
	if err := getJSON(ctx, hc, metadataURL, &md); err != nil {
		return nil, err
	}
	if resource != "" && md.Resource != resource {
		return nil, fmt.Errorf("mcpauth: %s describes resource %q, not %q", metadataURL, md.Resource, resource)
	}
	return &md, nil
}

// FetchAuthServerMetadata tries AuthServerMetadataURLs(issuer) in order and
// returns the first document found. A non-200 moves on to the next URL; a
// transport error, a document that does not decode, or one whose issuer is
// not this issuer (RFC 8414 §3.3) stops the search. Issuers are compared
// exactly except for a trailing slash, which IdPs disagree about. A nil hc
// means http.DefaultClient.
func FetchAuthServerMetadata(ctx context.Context, hc *http.Client, issuer string) (*AuthServerMetadata, error) {
	urls, err := AuthServerMetadataURLs(issuer)
	if err != nil {
		return nil, err
	}
	var last error
	for _, u := range urls {
		var md AuthServerMetadata
		err := getJSON(ctx, hc, u, &md)
		if _, ok := err.(*statusError); ok {
			last = err
			continue
		}
		if err != nil {
			return nil, err
		}
		if strings.TrimSuffix(md.Issuer, "/") != strings.TrimSuffix(issuer, "/") {
			return nil, fmt.Errorf("mcpauth: %s describes issuer %q, not %q", u, md.Issuer, issuer)
		}
		return &md, nil
	}
	return nil, fmt.Errorf("mcpauth: no authorization server metadata for %s: %w", issuer, last)
}
