# Test plan: st_1m7htn40h1psyncy0g6bcy

D1 spec 1 — the project's declared check (`go vet ./...`) succeeds on base main.

Runs the exact argv declared for the `check` property and captures its output. A non-zero exit fails the block.

```sh produces=vet.out inputs=go.mod
set -e
cd "$PWD"
go vet ./... 2> "$OUT/vet.err"
rc=$?
echo "go vet ./... exit=$rc (empty output = clean)" > "$OUT/vet.out"
cat "$OUT/vet.err" >> "$OUT/vet.out"
exit $rc
```



D1 spec 2 — the full test suite (`go test ./...`) passes on base main.

Runs the exact argv declared for the `test_suite` property. The artifact carries the exit line and the tail of the package summary (each package's `ok`/`?   ` line). A non-zero exit fails the block.

```sh produces=test.out inputs=go.mod
go test ./... > "$OUT/test.out" 2>&1
rc=$?
{ echo "go test ./... exit=$rc"; cat "$OUT/test.out"; } > "$OUT/test.all"
mv "$OUT/test.all" "$OUT/test.out"
exit $rc
```

D1 supporting count — the suite's named tests, via the declared list command `go test ./... -list .*`.

Lists every test the suite declares and counts them, so "the full test suite passes" names how many it passed.

```sh produces=list.out inputs=go.mod
go test ./... -list '.*' > "$OUT/list.raw" 2>&1
grep -E '^(Test|Benchmark|Example)' "$OUT/list.raw" > "$OUT/list.txt"
total=$(wc -l < "$OUT/list.txt")
{ echo "named tests (go test ./... -list .*): $total"; cat "$OUT/list.txt"; } > "$OUT/list.out"
```
