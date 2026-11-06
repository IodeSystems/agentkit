package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// Each media helper must marshal to the exact OpenAI chat-completions part a
// server parses; a field name off by one is a 400 from every server, or worse,
// a part silently ignored.
func TestMediaParts_MarshalToTheOpenAIWireShape(t *testing.T) {
	raw := []byte("hi") // base64 "aGk="
	cases := map[string]struct {
		part ContentPart
		want string
	}{
		// input_audio data is BARE base64, not a data: URI.
		"audio":      {AudioData("wav", raw), `{"type":"input_audio","input_audio":{"data":"aGk=","format":"wav"}}`},
		"video data": {VideoData("video/mp4", raw), `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,aGk="}}`},
		"video url":  {VideoPart("https://x/v.mp4"), `{"type":"video_url","video_url":{"url":"https://x/v.mp4"}}`},
		// file_data IS a data: URI.
		"pdf":  {PDFData("a.pdf", raw), `{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,aGk="}}`},
		"file": {FileData("text/plain", "", raw), `{"type":"file","file":{"file_data":"data:text/plain;base64,aGk="}}`},
	}
	for name, c := range cases {
		b, err := json.Marshal(c.part)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != c.want {
			t.Errorf("%s:\n got %s\nwant %s", name, b, c.want)
		}
	}
	// And inside a message: content is the array.
	b, _ := json.Marshal(Message{Role: "user", Parts: []ContentPart{TextPart("x"), AudioData("mp3", raw)}})
	if want := `{"role":"user","content":[{"type":"text","text":"x"},{"type":"input_audio","input_audio":{"data":"aGk=","format":"mp3"}}]}`; string(b) != want {
		t.Errorf("message:\n got %s\nwant %s", b, want)
	}
}

// On the Messages API path a part Claude cannot read is refused BEFORE anything
// is sent, with an error a host can test for — not a 400 from upstream, and not
// a request with the part dropped.
func TestMessagesAPI_UnsupportedMediaIsRefusedBeforeSending(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(500)
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL+"/v1", "k", "claude-opus-5-5")
	c.MessagesAPI = true
	for name, p := range map[string]ContentPart{
		"audio": AudioData("wav", []byte("RIFF")),
		"video": VideoData("video/mp4", []byte{0}),
		"docx":  FileData("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "a.docx", []byte{0}),
	} {
		_, err := c.ChatStream(context.Background(), []Message{{Role: "user", Parts: []ContentPart{TextPart("x"), p}}}, nil, nil)
		if !errors.Is(err, ErrUnsupportedContent) {
			t.Errorf("%s: err = %v, want ErrUnsupportedContent", name, err)
		}
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("%d requests reached the server", n)
	}
}

// A PDF crosses the Messages API path as a document block.
func TestMessagesAPI_PDFIsSentAsADocument(t *testing.T) {
	c, got := fakeMessages(t, evStart, evStop)
	msgChunks(t, c, []Message{{Role: "user", Parts: []ContentPart{TextPart("read"), PDFData("a.pdf", []byte("%PDF"))}}}, nil)
	msgs, _ := (*got)["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %v", *got)
	}
	content := msgs[0].(map[string]any)["content"].([]any)
	doc := content[1].(map[string]any)
	src, _ := doc["source"].(map[string]any)
	if doc["type"] != "document" || doc["title"] != "a.pdf" || src["media_type"] != "application/pdf" || src["data"] != "JVBERg==" {
		t.Errorf("document = %v", doc)
	}
}
