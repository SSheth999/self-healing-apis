# Verifier sandboxing choice

AGENTS.md Section 7 asks for the chosen sandboxing approach to be an
explicit, documented decision rather than an oversight. This is that
decision.

## Choice: temp directory + git worktree + subprocess pytest

`run_verifier()` (`sandbox.py`) creates a fresh temporary directory, adds a
detached `git worktree` pointed at the target repo's current `HEAD` inside
it, applies the candidate patch there with `git apply`, and runs
`python3 -m pytest tests/` as a subprocess with a wall-clock timeout. The
worktree is removed and pruned unconditionally in a `finally` block, even
if patch application or the test run raises.

This is Section 7's "Simplest" option:

- No isolation from the host beyond filesystem location - the sandboxed
  process runs as the same user, with the same network/filesystem access,
  as the rest of the pipeline.
- No separate virtualenv or container is created; the sandbox reuses
  whatever Python environment is already active (the same one with
  `requirements.txt` installed), which is why `demo_repo/requirements.txt`
  documents its dependencies as "must already be installed in the ambient
  environment" rather than something the Verifier installs itself.

## Why this is acceptable for the MVP

`target_repo` is `demo_repo/` - a fixture repo this project owns and
controls (AGENTS.md Section 3, rule 2: single target repo). There is no
untrusted third-party code involved, so the risk profile that would justify
Section 7's "Safer" (Docker-per-run) option doesn't apply yet.

## When to revisit

If this project ever points at a target repo it doesn't fully trust and
control - which Section 3 explicitly treats as a phase-2 change requiring
sign-off, not something to build quietly - the sandboxing approach must be
revisited to the Docker-per-run option before that happens, not after.
