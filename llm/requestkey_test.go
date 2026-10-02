package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A client's User and Header ride on every chat request, streaming or not, and
// the client's own headers still win a clash. A client with neither set sends
// no "user" field, so existing callers see no change on the wire.
func TestUserAndHeaderRideOnEveryChatRequest(t *testing.T) {
	type seen struct {
		body map[string]any
		h    http.Header
	}
	var got []seen
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		got = append(got, seen{body, r.Header.Clone()})
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hi"}}]}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", "m")
	c.User = "s_1"
	c.Header = http.Header{"X-Aw4-Session": {"s_1"}, "Content-Type": {"text/plain"}}
	ch, err := c.ChatStream(context.Background(), []Message{{Role: "user", Content: "x"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatal(err)
	}
	plain := NewClient(srv.URL, "", "m")
	if _, _, err := plain.Chat(context.Background(), []Message{{Role: "user", Content: "x"}}, nil); err != nil {
		t.Fatal(err)
	}

	if len(got) != 3 {
		t.Fatalf("%d requests, want 3", len(got))
	}
	for i, s := range got[:2] {
		if s.body["user"] != "s_1" {
			t.Errorf("request %d: user = %v, want s_1", i, s.body["user"])
		}
		if v := s.h.Get("X-Aw4-Session"); v != "s_1" {
			t.Errorf("request %d: X-Aw4-Session = %q, want s_1", i, v)
		}
		if v := s.h.Get("Content-Type"); v != "application/json" {
			t.Errorf("request %d: Content-Type = %q; the client's own header must win", i, v)
		}
	}
	if _, ok := got[2].body["user"]; ok {
		t.Errorf("a client with no User sent user = %v", got[2].body["user"])
	}
	if v := got[2].h.Get("X-Aw4-Session"); v != "" {
		t.Errorf("a client with no Header sent X-Aw4-Session = %q", v)
	}
}
