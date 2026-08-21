"""Session-wide pytest setup.

Ensures `demo_repo/` is a git repository before any test runs, since the
Verifier's sandboxing (`git worktree`) requires `target_repo` to be one -
see AGENTS.md Section 7. `demo_repo/.git` is intentionally not committed to
this repo's own history (see devsetup.py), so it's created here on demand
instead.
"""

from pathlib import Path

import pytest

from devsetup import ensure_git_repo

REPO_ROOT = Path(__file__).resolve().parent
DEMO_REPO_PATH = REPO_ROOT / "demo_repo"


@pytest.fixture(scope="session", autouse=True)
def _demo_repo_is_git_initialized() -> None:
    ensure_git_repo(DEMO_REPO_PATH)
