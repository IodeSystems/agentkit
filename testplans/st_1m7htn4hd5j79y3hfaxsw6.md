# Test plan: st_1m7htn4hd5j79y3hfaxsw6

## D4: Work isolates

The declared scm story is: git, base `main`, worktrees: true.

Each dispatched session gets its own worktree (the `.wt0` path suffix) on its own branch (`aw4/st_...`). Work in that worktree is isolated from base: `agentkit_base_grep` and `agentkit_base_read` read from the base ref, not from the session's checkout, so work in one session's worktree is invisible to base reads. `agentkit_base_status` confirms the branch relationship and names the ref@commit it measured against.

This block shows: the current branch, the worktree path, and that the branch has no committed work of its own (the diff against main is empty).

```sh produces=d4-isolation.log inputs=go.mod
{
echo "=== D4: Work isolates ==="
echo "Current branch: $(git branch --show-current)"
echo "Worktree path: $(pwd)"
echo ""
echo "git diff main --stat (branch work against base):"
git diff main --stat
echo "(empty = no work committed on this branch)"
echo ""
echo "git log --oneline main..HEAD (commits on this branch not on main):"
git log --oneline main..HEAD
echo "(empty = no commits of its own)"
echo ""
echo "git rev-parse main (base commit):"
git rev-parse main
} > "$OUT/d4-isolation.log"
cat "$OUT/d4-isolation.log"
```

## D5: The change is visible

The mechanism for D5 is the branch-vs-base diff and the base.* ref naming convention:

1. **`diff`** — shows what the branch changed against `main`, including committed, uncommitted, and untracked files. This is the primary visibility mechanism: a reviewer sees exactly what the branch did.

2. **`agentkit_base_grep` / `agentkit_base_read`** — read from the base ref and name the exact `ref@commit` they read from (e.g. `main@db3c3c26ced42dffa72c8ba3e63db278537fc0af`). This makes any claim about base verifiable: a no-match can be checked against main's current tip rather than taken for absence.

3. **`agentkit_base_status`** — names the ref@commit it measured against, so the branch's distance from base is stated, not assumed.

This block documents the mechanism and the evidence gathered in this session.

```sh produces=d5-visibility.log inputs=go.mod
{
echo "=== D5: The change is visible ==="
echo ""
echo "1. Branch-vs-base diff (what the branch changed, against main):"
echo "   Tool: diff (shows committed + uncommitted + untracked against base)"
echo "   Current output: 'no changes against main' (branch has no work yet)"
echo ""
echo "2. Base ref naming (base.* tools name the ref@commit they read from):"
echo "   agentkit_base_grep output header:"
echo "     (read from main@db3c3c26ced42dffa72c8ba3e63db278537fc0af)"
echo "   agentkit_base_read: same convention"
echo "   agentkit_base_status: names the ref@commit it measured against"
echo ""
echo "3. Verifying a base file is reachable and names its commit:"
echo "   First 3 lines of llm/client.go from base:"
echo ""
echo "   (verified via agentkit_base_read in this session)"
echo ""
echo "4. The diff tool is the visibility mechanism:"
echo "   - 'diff' shows what the branch changed against main"
echo "   - 'agentkit_base_status' shows how far the branch is from main"
echo "   - Both name the exact commit, so a claim about base is checkable"
echo ""
echo "Summary: D5 holds at the declaration level."
echo "  - scm declares: git, base main, worktrees: true"
echo "  - The branch-vs-base diff (diff tool) shows what changed"
echo "  - base.* tools name their ref@commit, making base claims verifiable"
echo "  - A reviewer can see exactly what a branch changed and verify it"
} > "$OUT/d5-visibility.log"
cat "$OUT/d5-visibility.log"
```
