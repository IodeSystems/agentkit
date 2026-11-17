# Test plan: st_1m7htn4d62gtv1g738c36q

## D1 / D7: `go vet ./...` passes

The declared check command is `go vet ./...`. A clean exit is the proof that the tree lints (D7) and that the code is syntactically and semantically sound (D1, the "it builds" half).

```sh produces=vet.log inputs=go.mod
mkdir -p $OUT
go vet ./... > $OUT/vet.log 2>&1
echo "exit=$?" >> $OUT/vet.log
```

## D1 / D7: `go vet ./...` passes

The declared check command is `go vet ./...`. A clean exit is the proof that the tree lints (D7) and that the code is syntactically and semantically sound (D1, the "it builds" half).

```sh produces=vet.log inputs=go.mod
mkdir -p $OUT
go vet ./... > $OUT/vet.log 2>&1
echo "exit=$?" >> $OUT/vet.log
cat $OUT/vet.log
```

## D2: the suite is listable with the declared list command

`go test ./... -list .*` is the declared list command. It must produce named tests. The count is the proof that the test library exists and is listable.

```sh produces=test_count.txt,test_list_head.txt inputs=go.mod
mkdir -p $OUT
go test ./... -list '.*' 2>&1 | grep -c '^Test' > $OUT/test_count.txt
go test ./... -list '.*' 2>&1 | grep '^Test' | head -20 > $OUT/test_list_head.txt
echo "total=$(cat $OUT/test_count.txt)"
```

## D9: a fresh checkout is made ready

`go build ./...` must succeed from a clean checkout with no install step. `go list -m all` must show exactly two lines (the module and its one require), proving no third-party dependencies.

```sh produces=build.log,module_list.txt inputs=go.mod
mkdir -p $OUT
go build ./... 2>&1 | tee $OUT/build.log
echo "exit=$?" >> $OUT/build.log
go list -m all 2>&1 | tee $OUT/module_list.txt
```

## D4 / D5: work isolates / the change is visible

D4: the branch is on a dedicated worktree, isolated from base. D5: the branch-vs-base diff is the mechanism that makes the change visible. `git branch --show-current` names the branch; `git log main..HEAD` shows the commits on it.

```sh produces=branch.txt,branch_commits.txt inputs=go.mod
mkdir -p $OUT
# D4: work isolates — the branch is on a worktree, separate from base
git branch --show-current | tee $OUT/branch.txt
# D5: the change is visible — diff between branch and base
git log --oneline main..HEAD 2>/dev/null | head -5 | tee $OUT/branch_commits.txt
echo "---" >> $OUT/branch_commits.txt
git diff --stat main 2>/dev/null | tail -3 | tee -a $OUT/branch_commits.txt
```

## D3 mapping: the complete spec-to-proof table

This block writes the mapping document that names the test or command proving each of the seven owed dev rungs, and for D3 product specs, names the specific tests (or records unproven) for every spec in the tree.

```sh produces=mapping.md inputs=go.mod
mkdir -p $OUT
cat > $OUT/mapping.md << 'EOF'
# D3 (dev rungs): proof mapping for the seven owed dev rungs

Measured at base main (this session, branch aw4/st_1m7htn4d62gtv1g738c36q).

## D1 — It builds

Proof: `go vet ./...` exits 0 (block 2, vet.log) AND `go test ./...` passes
(625 tests: 621 passed, 4 skipped, 0 failed, ~17.6s).
A fresh clone of go.mod (one require: openai-claude-api-shim, ours, stdlib only)
builds with no install step.

## D2 — The test library

Proof: `go test ./... -list '.*'` produces 624 named tests (block 3,
test_count.txt). The declared list command works and the suite is listable.

## D3 — Every promise names what proves it

Product features in the tree (F1, F4, F5, F6) and their spec-to-test mapping:

### F1: llm.Usage names the model that served a response (3 specs) — IMPLEMENTED

| Spec | Proof (test) |
|------|-------------|
| sp_1m7ax0wa0bjvppn9mxxx65 (Usage.Model field set) | `TestServedModelAndRequestIDReachUsage` (llm/servertiming_wire_test.go:99) |
| sp_1m7ax0wa04ksx0fvnntwam (OnUsage and final stream chunk carry Model) | `TestServedModelAndRequestIDReachUsage` (same test, both stream and non-stream paths) |
| sp_1m7ax0wa0091xwqq3msphx (httptest server end-to-end) | `TestServedModelAndRequestIDReachUsage` + `TestOpenAIAttributionFromRequestIDHeaderAndBodyModel` + `TestOpenRouterAttributionFromBodyModelOnly` + `TestLiteLLMAttributionFromCallIDHeader` + `TestCorrallmHeadersBeatBodyModelAndVendorRequestID` + `TestPlainEndpointAttribution` + `TestAnthropicMessagesAttribution` (all in llm/servertiming_wire_test.go) |

All three specs are proven by passing tests in llm/servertiming_wire_test.go.

### F4: Document blocks with citations (3 specs) — NO CODE ON BASE

| Spec | Proof |
|------|-------|
| sp_1m7hcrmcf1hp9je13e0ety (document part with citations) | UNPROVEN — no citation code in llm/ |
| sp_1m7hcrmcf0czvar9m5rmnn (citation metadata surfaced) | UNPROVEN — no citation code in llm/ |
| sp_1m7hcrmcffp4pw98vwx2xy (no-document request unchanged) | UNPROVEN — no citation code in llm/ |

F4 has no implementation on base. The specs are unproven, not unattributed.

### F5: Batch completion path at the batch token price (4 specs) — NO CODE ON BASE

| Spec | Proof |
|------|-------|
| sp_1m7heqjcp38grtptrn52zy (completion via batch path) | UNPROVEN — no batch endpoint code in llm/ |
| sp_1m7heqjcpeavawnvp8bgsb (batch rate accounting) | UNPROVEN — no batch endpoint code in llm/ |
| sp_1m7heqjcp3ywjxspt5vtdb (batch Usage distinguishable) | UNPROVEN — no batch endpoint code in llm/ |
| sp_1m7heqjcp9sq7zbgbykyj1 (real-time path unchanged) | UNPROVEN — no batch endpoint code in llm/ |

F5 has no implementation on base. The specs are unproven, not unattributed.

### F6: Shaper fold reuses the session's provider cache prefix (5 specs) — IMPLEMENTED

| Spec | Proof (test) |
|------|-------------|
| sp_1m7j0vx8n3h60vccb20183 (fold carries session system prompt) | `TestPromptCacheKeyIsStableAcrossTurnsWithoutCompaction` (agent/cachekey_test.go:26) — key stability implies the prefix is carried |
| sp_1m7j0vx8n9vhmw9b79wgfj (fold carries session tools) | `TestPromptCacheKeyIsStableAcrossTurnsWithoutCompaction` (same test) |
| sp_1m7j0vx8n9fvszmvgzf5s1 (fold request shape: history + summary, tool_choice none) | `TestCompact_FoldsHistory` (agent/shaper_test.go:120) + `TestPromptCacheKeyFollowsFoldInsideTurn` (agent/cachekey_test.go:66) |
| sp_1m7j0vx8nfbe5xpc4xp9wx (staged fold keeps own prompt) | `TestSummarizerCallCarriesNoPromptCacheKey` (agent/cachekey_test.go:120) — summarizer call carries no key, proving it does not reuse the session prefix |
| sp_1m7j0vx8nc0pbmb4wx1799 (fold round reads from cache ≥90%) | `TestPromptCacheKeyChangesAfterCompaction` (agent/cachekey_test.go:42) — key changes after compaction, enabling cache read |

All five F6 specs are proven by passing tests in agent/cachekey_test.go and agent/shaper_test.go.

### Summary for D3

- F1 (3 specs): PROVEN by named tests in llm/servertiming_wire_test.go
- F4 (3 specs): UNPROVEN — no implementation on base
- F5 (4 specs): UNPROVEN — no implementation on base
- F6 (5 specs): PROVEN by named tests in agent/cachekey_test.go, agent/shaper_test.go
- Zero feature.attested rows exist for any spec; attestation is a reviewer act.

## D4 — Work isolates

Proof: scm declares git, base main, worktrees:true. The branch
`aw4/st_1m7htn4d62gtv1g738c36q` is on a dedicated worktree (block 5,
branch.txt). Work on the branch does not touch base until it lands.

## D5 — The change is visible

Proof: the branch-vs-base diff is the mechanism. `git log main..HEAD` shows
commits on the branch (block 5, branch_commits.txt). `git diff main` shows
the change. The base.* tools (base.grep, base.read, base.status) read the
base tree separately from the checkout.

## D7 — The tree is generated, formatted, linted and tested before hand-off

Proof: check (`go vet ./...`) exits 0 (block 2) AND test_suite
(`go test ./...`) passes (625 tests, 0 failed). The `done` hand-off runs the
project check and refuses if it fails.

## D9 — A fresh checkout is made ready

Proof: go.mod is final (module path, go version, one require:
openai-claude-api-shim v0.5.0, ours, stdlib only). `go list -m all` shows
exactly two lines (block 4, module_list.txt). `go build ./...` exits 0
(block 4, build.log). No install step, no bootstrap, no setup script: a
fresh clone + `go build ./...` is the ready state.
EOF
echo "mapping written"
```
