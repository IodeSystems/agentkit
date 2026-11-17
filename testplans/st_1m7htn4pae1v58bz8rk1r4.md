# Test plan: st_1m7htn4pae1v58bz8rk1r4

D9 — a fresh checkout is made ready. agentkit declares no setup/bootstrap step because go.mod is final: one require (openai-claude-api-shim, ours, stdlib only), so a fresh clone needs no install. This block proves the declared commands alone make it ready: `go build ./...` succeeds and `go list -m all` lists exactly the module plus that single require (2 lines).

```sh produces=modlist.txt inputs=go.mod
echo "=== go build ./... ==="
go build ./... 2>&1 && echo "BUILD_OK"
echo "=== go list -m all ==="
go list -m all 2>&1 | tee "$OUT/modlist.txt"
echo "LINE_COUNT=$(go list -m all 2>&1 | wc -l)"
echo "=== go vet ./... ==="
go vet ./... 2>&1 && echo "VET_OK"
```
