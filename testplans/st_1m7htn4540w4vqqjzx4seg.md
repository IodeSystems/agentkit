# Test plan: st_1m7htn4540w4vqqjzx4seg

D2 — The test library is listable per-test with the declared list command.

The declared `test_suite` list command is `go test ./... -list .*`. Running it
prints, for every package, the names of the test functions it would run. The
named-test count is the number of `Test[A-Z]…` lines across all packages.

Measured: 624 named tests. The tree holds 625 `func Test` declarations
(`dir.grep '^func Test[A-Z]'` = 625 in 101 files); the declared list command
names 624 of them because `go test -list` does not list `TestMain` —
`mcpmgr/fakeserver_test.go:41 func TestMain(m *testing.M)` — which is a
test-package entry hook, not a test. 624 is the per-test list the D2 spec asks
for.

```sh produces=list.raw,named.tests inputs=go.mod
set -e
cd /home/nthalk/local/src/iodesystems/agentkit.wt0
echo "== declared list command: go test ./... -list .* =="
go test ./... -list '.*' 2>&1 | tee $OUT/list.raw
echo
echo "== count of named tests (lines that are a test name, not a package header) =="
# go test -list prints per package: a bare test-identifier line for each listed
# test, plus "ok  pkg  (cached)" header lines. Named tests are the bare
# Test[A-Z]... lines. (No name collides across packages, so sort changes
# nothing here; TestMain is the only decl the list omits.)
grep -E '^Test[A-Z][A-Za-z0-9_]*$' $OUT/list.raw | sort > $OUT/named.tests
wc -l < $OUT/named.tests
```

Diagnosis: run the declared check (`go vet ./...`) to see why the hand-off gate refused.

```sh produces=vet.out inputs=go.mod
set -e
cd /home/nthalk/local/src/iodesystems/agentkit.wt0
echo "== declared check: go vet ./... =="
go vet ./... 2>&1 | tee $OUT/vet.out
echo "vet exit: ${PIPESTATUS[0]}"
```
