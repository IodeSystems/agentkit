# Connecting to a provider

agentkit talks to any OpenAI-compatible `/v1/chat/completions` endpoint. This
page covers the iode **corrallm** server the examples use, plus the
provider-specific behavior worth knowing.

## corrallm at `llm.iodesystems.com`

```go
client := llm.NewClient("https://llm.iodesystems.com", os.Getenv("AGENTKIT_API_KEY"), "Qwen3-6-27B-MPT")
```

`NewClient(baseURL, apiKey, model)` accepts a base URL with or without a
trailing `/v1` — both `https://host` and `http://host:11434/v1` resolve to the
same chat endpoint.

Discover models with a plain GET:

```
curl -s https://llm.iodesystems.com/v1/models
```

corrallm exposes chat, embeddings, and speech models. The chat model at time of
writing is **`Qwen3-6-27B-MPT`** (~220k context). It is a single shared,
capacity-limited backend — expect it to be busy.

## The API key is a scheduling identity

corrallm is a llama-swap–derived fair-share proxy. The key you pass becomes the
`Authorization: Bearer …` header, and the proxy maps it to a **scheduling
priority** — it is not merely authentication. An empty key means no auth and
default priority. Set it via `AGENTKIT_API_KEY` or the `--api-key` flag in the
demo.

`ChatOpts.TraceID` is unrelated to priority: it sets an `X-Trace-Id` header for
log correlation only (the demo/session use it to tie requests back to a thread).

## 429 backpressure — a first-class path

Because the backend is capacity-limited, 429 is routine, not exceptional. When
all slots are busy, corrallm queues your request and — if no slot frees in
time — returns a JSON backpressure body:

```json
{"error":{"capacity":1,"in_flight":1,"message":"backend at capacity; retry after backoff","reason":"queue-timeout","retry_after":10,"type":"backpressure","waiting":0}}
```

Note `retry_after` is in the **body**, not the `Retry-After` header. agentkit's
client handles both: it prefers the header, falls back to the body's
`retry_after` (top-level or nested under `error`), clamps to a safety ceiling,
and otherwise uses an exponential backoff (1s → 2s → … → 30s, then holds).

429 retries continue until a slot frees, bounded by **two** ceilings — whichever
comes first:

- **`Client.RetryBudget`** — total wall-clock the client will keep retrying one
  request before giving up with a clear `retry budget … exhausted` error.
  Default **5 minutes**; set it per client for a busier or more patient endpoint.
- **your `ctx` deadline** — the hard stop; wins if shorter than the budget.

```go
client := llm.NewClient(baseURL, apiKey, model)
client.RetryBudget = 5 * time.Minute   // default; raise/lower to taste

ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
defer cancel()
```

On a super-busy box, retrying for a bounded window (honoring each
`retry_after`) is usually better than failing fast — every wait is another shot
at a slot — but the budget guarantees the call can't hang forever. If it
elapses while the model is still saturated you get a clear error; that's the
expected outcome under sustained load, not a client bug. The demo's `chat`,
`tools`, and `grammar` subcommands set both ceilings from `--timeout` (default
5m) and print this explanation when it happens.

## Constrained decoding support

corrallm is llama.cpp-based, so `ChatOpts.Grammar` (GBNF) and
`ChatOpts.ResponseFormat` (`json_object` / `json_schema`) are honored — the
server constrains sampling. Other OpenAI-compatible providers that don't support
these fields simply ignore them; the request still succeeds. See
[features.md §4](features.md#4-server-side-constrained-decoding--grammar).

## Using a different provider

Point `NewClient` at any OpenAI-compatible base URL and set the model id. The
tool loop, shaper, injection, lifting, and validation are provider-agnostic.
Provider-specific fields (`grammar`, `response_format`, cache-token usage) are
best-effort: sent when set, ignored by providers that don't implement them.
