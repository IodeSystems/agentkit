package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTranscribe_SendsMultipartAndReturnsText(t *testing.T) {
	var gotModel, gotName, gotAuth string
	var gotAudio []byte
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("wrong path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("not multipart: %v", err)
			return
		}
		gotModel = r.FormValue("model")
		f, h, err := r.FormFile("file")
		if err != nil {
			t.Errorf("no file part: %v", err)
			return
		}
		gotName = h.Filename
		gotAudio, _ = io.ReadAll(f)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"text":"two pans and a colander"}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "k", "chat-model")
	text, err := c.Transcribe(context.Background(), "stt", "memo.m4a", []byte("RIFFdata"))
	if err != nil {
		t.Fatal(err)
	}
	if text != "two pans and a colander" {
		t.Errorf("text = %q", text)
	}
	if gotModel != "stt" || gotName != "memo.m4a" || string(gotAudio) != "RIFFdata" {
		t.Errorf("form not forwarded: model=%q name=%q audio=%q", gotModel, gotName, gotAudio)
	}
	if gotAuth != "Bearer k" {
		t.Errorf("auth = %q", gotAuth)
	}
}

func TestTranscribe_StatusErrorCarriesBody(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unsupported audio", http.StatusBadRequest)
	}))
	defer ts.Close()
	c := NewClient(ts.URL, "", "m")
	_, err := c.Transcribe(context.Background(), "stt", "a.wav", []byte("x"))
	if err == nil || !strings.Contains(err.Error(), "unsupported audio") {
		t.Fatalf("want the provider message in the error, got %v", err)
	}
}
