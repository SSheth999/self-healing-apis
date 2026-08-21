"""One-time local setup helpers - not part of the pipeline's runtime logic.

`demo_repo/` (and any future `target_repo`) must be its own git repository
for the Verifier's `git worktree` sandboxing to work (AGENTS.md Section 7).
To keep this project a single, normal GitHub repo - rather than committing
a nested embedded git repo inside it, which `git` handles awkwardly as a
gitlink - `demo_repo/.git` is deliberately NOT committed to this repo's own
history. Instead, it's created on demand, once, the first time it's needed
locally (see conftest.py and graph.py's `_main()`). See verifier/README.md
for the fuller rationale.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_git_repo(repo_path: str | Path) -> None:
    """If `repo_path` isn't already a git repo, initialize one with a
    single commit of its current contents. Idempotent - a no-op if
    `repo_path/.git` already exists, so it's safe to call on every run."""

    path = Path(repo_path)
    if (path / ".git").exists():
        return

    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-c", "user.email=demo@example.com", "-c", "user.name=Demo Repo", *args],
        cwd=path,
        check=True,
        capture_output=True,
    )
    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "Initial commit (auto-created by devsetup.ensure_git_repo)")
