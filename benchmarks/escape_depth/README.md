# Escape depth

Does escape depth break LLM-authored tool calls? Claim under test: failures
appear at 3+ levels of nesting when driving code edits through a tool argument.

**Result: not reproduced.** 60 trials on Qwen3-6-27B across two transports, copy
and authored. One genuine escaping error, and it was at depth 1.

Uses mcpshell as the verifier: a generated script is EXECUTED and its output
compared, so a corrupted script is caught by behaviour rather than a byte diff.
Set `MCPSHELL` if it is not on PATH.

## The ladder

Each level nests one more layer of quoting, which is where backslashes double: a
string holding JSON holding JSON. Every level has a distinct output.

| depth | output |
|---|---|
| 1 | `hi!` |
| 2 | `8` |
| 3 | `10` |
| 4 | `19` |
| 5 | `36` |

`copy/` hands over the script and asks for it verbatim. `authored/` states in
prose what the string must contain, so the model derives the escaping itself —
the shape of a real edit, and the variant that actually matters.

Both compare the NATIVE tool path (arguments as an escaped JSON string) against
the HEREDOC path (raw body, grammar-constrained).

## Results

```
copy      native 3/3 every depth; heredoc 3/3 except d5 (one stray brace)
authored  native 3/3 and heredoc 3/3 at every case
```

## Two traps, both of which manufactured evidence FOR the hypothesis

**Hand-computed expected values.** Two of five were wrong (`say "hi"` is 8 not
10, `C:\Users\dev` is 12 not 16) — at depths 2 and 5, exactly the depths under
suspicion. Verify ground truth against the interpreter before trusting a table.

**A harness that omits the tool's own prompt.** The first authored run showed 9
of 11 failures and read as escaping collapse. Every one was `'s' is not defined`:
the harness declared a bare `eval(code)` schema and never sent
`mcpshell --prompt`, so the model did not know a declaration was required. With
the reference attached it writes `let` every time. A benchmark that skips the
tool's prompt measures the harness.

## Running

    go run ./copy
    go run ./authored

`BASE` and `MODEL` point at another endpoint. If the claim reproduces elsewhere
it is model-specific, not a property of the format.

## Untested

Realistic edit size (these are one-liners), depth beyond 5, and mcpshell's
`vars` / `r"..."` raw strings versus inline escaping — it ships both as
mitigations, which suggests someone hit this for real.
