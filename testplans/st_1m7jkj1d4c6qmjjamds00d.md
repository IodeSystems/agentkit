# Test plan: st_1m7jkj1d4c6qmjjamds00d

Spike answer: does the v0.28.0 tag of agentkit contain `llm.TransientUpstream` / `TransientUpstreamReason`? Verified directly from the local agentkit checkout's git metadata: the tag exists, both functions are in the tag's tree, and the file's introducing commit (af850bf4) is an ancestor of the tag. Fails if any of the three is missing.

```sh
set -euo pipefail
A=/home/nthalk/local/src/iodesystems/agentkit
echo "=== v0.28.0 tag commit ==="
git -C "$A" rev-parse 'v0.28.0^{commit}'
echo "=== functions present in the tag's tree ==="
git -C "$A" show 'v0.28.0:llm/transient.go' | grep -n 'func TransientUpstream'
git -C "$A" show 'v0.28.0:llm/transient.go' | grep -n 'func TransientUpstreamReason'
echo "=== introducing commit af850bf4 ancestor of v0.28.0? ==="
git -C "$A" merge-base --is-ancestor af850bf4 'v0.28.0^{commit}' && echo ANCESTOR
```
