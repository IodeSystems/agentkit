# Test plan: st_1m7htn49484y53wgd2grta

# D3 (product): the test that proves each F1–F5 spec

Base: `main@db3c3c26c`, branch 0 commits ahead. The task is attribution only:
name the test that proves each spec, or state which promises have no
implementation. Attestation rows are a reviewer act and are not written here.

## F1 — `llm.Usage` names the model that served a response (f_1m7awtv9kcaeakg5bnpzqp)

**Implemented on base.** `llm.Usage.ServedModel` (`llm/client.go:388`) is set in
`responseMeta.stamp` (`llm/cost.go:348-355`): the proxy's served-model header,
else the response body's top-level `model` field (first one seen on a stream).

| Spec | Proving test(s) |
|---|---|
| sp_1m7ax0wa0bjvppn9mxxx65 — `Usage` carries the model field (stream chunk / body); empty when the response names none | `TestPlainEndpointAttribution` (no header + no body model → `ServedModel == ""`; body-only echo → the body model), `TestOpenRouterAttributionFromBodyModelOnly` (body model on stream and non-stream) |
| sp_1m7ax0wa04ksx0fvnntwam — `OnUsage` and the final stream chunk both carry the model | `TestServedModelAndRequestIDReachUsage` asserts `ServedModel` on both the last stream `chunk.Usage` and the `OnUsage` value for the same server; every test in `wantAttribution` does the same |
| sp_1m7ax0wa0091xwqq3msphx — httptest server naming a model the request did not, end-to-end | `TestOpenAIAttributionFromRequestIDHeaderAndBodyModel` (request says `asked-for`, body answers `gpt-4o-2024-08-06`), `TestAnthropicMessagesAttribution` (request `claude-opus-5-5`, body `claude-opus-5-5-20260101`), plus `TestCorrallmHeadersBeatBodyModelAndVendorRequestID` for the header-beats-body precedence |

Supporting, same file `llm/servertiming_wire_test.go`:
`TestLiteLLMAttributionFromCallIDHeader` (id only, no model).
All 7 listed pass on this branch (block below; the earlier full-suite run
recorded in the D1 sibling node put the suite at 617 tests, 0 failures).

## F4 — document blocks with citations (f_1m7h6zzcq5km65p1brh6en): **no implementation on base**

`plan/claude-api-plan.md` lists "Citations and document blocks" under **Not
exposed**. `llm.DocumentData` (named in sp_1m7hcrmcf1hp9je13e0ety) does not
exist anywhere in the tree; there is no citations field on any request option
or on `Usage`/the chat result. `TestMessagesAPI_PDFIsSentAsADocument`
(`llm/media_test.go:74`) proves a PDF crosses the Messages API as a document
block — a media part, not the F4 promise (no citations option, no citation
metadata surfaced).

| Spec | Status |
|---|---|
| sp_1m7hcrmcf1hp9je13e0ety — document part sent with citations enabled | **Unproven — no implementation on base.** No `DocumentData`, no citations option. |
| sp_1m7hcrmcf0czvar9m5rmnn — citation metadata surfaced from the response | **Unproven — no implementation on base.** No citation field on the result. |
| sp_1m7hcrmcffp4pw98vwx2xy — a request with no document parts is unchanged | Vacuously true for today's wire format, but there is no test written against the F4 promise and no F4 surface to assert it. Recorded unproven, not attributed. |

## F5 — batch completion path at the batch token price (f_1m7hent1qa84ka32fcd2tk): **no implementation on base**

`plan/claude-api-plan.md` lists "Batches" under **Not exposed**. No batch
endpoint, no completion window, no batch-rate accounting anywhere in `llm`
(`grep BatchEndpoint|/batches|CompletionWindow` → 0 matches; the "batching"
hits in `agent/` and `CLAUDE.md` are Turn message-batching, a different thing).

| Spec | Status |
|---|---|
| sp_1m7heqjcp38grtptrn52zy — a completion is submitted through the batch path | **Unproven — no implementation on base.** |
| sp_1m7heqjcpeavawnvp8bgsb — a batch completion is accounted at the batch rate | **Unproven — no implementation on base.** |
| sp_1m7heqjcp3ywjxspt5vtdb — a batch `Usage` is distinguishable from real-time | **Unproven — no implementation on base.** |
| sp_1m7heqjcp9sq7zbgbykyj1 — the real-time path is unchanged | Vacuously true for today's wire format; no F4/F5 surface exists, so there is nothing to assert against. Recorded unproven, not attributed. |

## F2 / F3

Both retired as duplicates of F1 (their own descriptions say so); nothing to
attribute.

## Bottom line for the reviewer

F1's three specs can be attested to the tests named above (satisfiable ids are
the `sp_*` rows, per the earlier D1/F1 correction). F4's three and F5's four
specs are **unproven because the features do not exist on base** — they need
implementation work, not attribution, before a reviewer can attest anything.

```sh produces=f1-tests.log inputs=llm/servertiming_wire_test.go,llm/cost.go,llm/client.go
go test ./llm -run 'TestServedModelAndRequestIDReachUsage|TestOpenAIAttributionFromRequestIDHeaderAndBodyModel|TestOpenRouterAttributionFromBodyModelOnly|TestLiteLLMAttributionFromCallIDHeader|TestCorrallmHeadersBeatBodyModelAndVendorRequestID|TestPlainEndpointAttribution|TestAnthropicMessagesAttribution' -v > "$OUT/f1-tests.log"
```
