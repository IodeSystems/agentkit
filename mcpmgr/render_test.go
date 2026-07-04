package mcpmgr

import (
	"encoding/json"
	"testing"
)

func TestRenderSecret(t *testing.T) {
	kv := map[string]string{
		"GMAIL_USER":         "a@b.com",
		"GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
		"GMAIL_IMAP_ADDR":    "imap.gmail.com:993",
	}

	tests := []struct {
		format string
		want   string
	}{
		{
			// Keys sorted: APP_PASSWORD, IMAP_ADDR, USER.
			format: "env",
			want:   "GMAIL_APP_PASSWORD=abcd efgh ijkl mnop\nGMAIL_IMAP_ADDR=imap.gmail.com:993\nGMAIL_USER=a@b.com\n",
		},
		{
			format: "properties",
			want:   "GMAIL_APP_PASSWORD=abcd efgh ijkl mnop\nGMAIL_IMAP_ADDR=imap.gmail.com:993\nGMAIL_USER=a@b.com\n",
		},
		{
			format: "yaml",
			want:   "GMAIL_APP_PASSWORD: abcd efgh ijkl mnop\nGMAIL_IMAP_ADDR: imap.gmail.com:993\nGMAIL_USER: a@b.com\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.format, func(t *testing.T) {
			got, err := RenderSecret(tt.format, kv)
			if err != nil {
				t.Fatalf("RenderSecret(%q): %v", tt.format, err)
			}
			if string(got) != tt.want {
				t.Errorf("RenderSecret(%q)\n got: %q\nwant: %q", tt.format, got, tt.want)
			}
		})
	}

	t.Run("json", func(t *testing.T) {
		got, err := RenderSecret("json", kv)
		if err != nil {
			t.Fatalf("RenderSecret(json): %v", err)
		}
		var round map[string]string
		if err := json.Unmarshal(got, &round); err != nil {
			t.Fatalf("json output not valid object: %v", err)
		}
		if len(round) != len(kv) {
			t.Fatalf("json round-trip: got %d keys, want %d", len(round), len(kv))
		}
		for k, v := range kv {
			if round[k] != v {
				t.Errorf("json[%q]=%q, want %q", k, round[k], v)
			}
		}
	})

	t.Run("deterministic", func(t *testing.T) {
		// Same input renders byte-identical across calls (sorted keys).
		for _, format := range []string{"env", "properties", "json", "yaml"} {
			a, _ := RenderSecret(format, kv)
			b, _ := RenderSecret(format, kv)
			if string(a) != string(b) {
				t.Errorf("RenderSecret(%q) not deterministic", format)
			}
		}
	})

	t.Run("unknown format", func(t *testing.T) {
		if _, err := RenderSecret("toml", kv); err == nil {
			t.Fatal("expected error for unknown format")
		}
	})
}
