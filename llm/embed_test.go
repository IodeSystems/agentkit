package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmbed_ReturnsVectorsInInputOrder(t *testing.T) {
	var gotModel string
	var gotInput []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/embeddings") {
			t.Errorf("wrong path: %s", r.URL.Path)
		}
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		gotModel, gotInput = req.Model, req.Input
		// Return out of order to prove Embed re-sorts by index.
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[
			{"index":1,"embedding":[0.3,0.4]},
			{"index":0,"embedding":[0.1,0.2]}
		]}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "", "chat-model")
	vecs, err := c.Embed(context.Background(), "nomic-embed-text", []string{"alpha", "bravo"})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "nomic-embed-text" {
		t.Errorf("model not forwarded: %q", gotModel)
	}
	if len(gotInput) != 2 || gotInput[0] != "alpha" {
		t.Errorf("input not forwarded: %v", gotInput)
	}
	if len(vecs) != 2 || vecs[0][0] != 0.1 || vecs[1][0] != 0.3 {
		t.Fatalf("vectors not re-ordered by index: %v", vecs)
	}
}

func TestEmbed_EmptyInputNoCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("Embed should not call the server for empty input")
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	if v, err := c.Embed(context.Background(), "e", nil); err != nil || v != nil {
		t.Fatalf("want (nil,nil) for empty input, got %v / %v", v, err)
	}
}

func TestEmbed_CountMismatchErrors(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}]}`))
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	if _, err := c.Embed(context.Background(), "e", []string{"a", "b"}); err == nil {
		t.Fatal("expected a count-mismatch error")
	}
}
