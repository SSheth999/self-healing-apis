"""Unit tests for verifier/sandbox.py (AGENTS.md Section 5.6 / Section 7).

Uses a throwaway git repo per test (tmp_path), never the real demo_repo/,
so these tests can't leave artifacts in or depend on the actual fixture
repo's state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from schemas import PatchResult
from verifier.sandbox import VerifierError, run_verifier

FIX_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # bug\n"
    "+    return a + b\n"
)

NOOP_BAD_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # bug\n"
    "+    return a - b  # still buggy, unrelated comment change\n"
)

INVALID_DIFF = "this is not a diff at all\n"


def _git(repo_dir: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "calc.py").write_text("def add(a, b):\n    return a - b  # bug\n", encoding="utf-8")
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    (repo_dir / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n",
        encoding="utf-8",
    )
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-q", "-m", "init")
    return repo_dir


def _patch_result(diff: str) -> PatchResult:
    return PatchResult(call_site_id="cs-1", diff=diff, rationale="fix", attempt_number=1, critic_rounds=0)


class TestRunVerifierOutcomes:
    def test_passing_patch_reports_passed_true(self, sample_repo: Path) -> None:
        result = run_verifier(_patch_result(FIX_DIFF), str(sample_repo))

        assert result["passed"] is True
        assert result["failing_tests"] == []
        assert result["failure_trace"] is None
        assert result["duration_seconds"] > 0

    def test_still_failing_patch_reports_failing_tests_and_trace(self, sample_repo: Path) -> None:
        result = run_verifier(_patch_result(NOOP_BAD_DIFF), str(sample_repo))

        assert result["passed"] is False
        assert any("test_add" in t for t in result["failing_tests"])
        assert result["failure_trace"] is not None

    def test_patch_that_does_not_apply_reports_passed_false(self, sample_repo: Path) -> None:
        result = run_verifier(_patch_result(INVALID_DIFF), str(sample_repo))

        assert result["passed"] is False
        assert result["failing_tests"] == []
        assert "apply" in result["failure_trace"].lower()

    def test_timeout_is_reported_as_a_failed_result_not_an_exception(self, sample_repo: Path) -> None:
        (sample_repo / "tests" / "test_slow.py").write_text(
            "import time\n\n\ndef test_slow():\n    time.sleep(5)\n", encoding="utf-8"
        )
        _git(sample_repo, "add", "-A")
        _git(sample_repo, "commit", "-q", "-m", "add slow test")

        result = run_verifier(_patch_result(FIX_DIFF), str(sample_repo), timeout_seconds=0.5)

        assert result["passed"] is False
        assert "timeout" in result["failure_trace"].lower()

    def test_failure_trace_is_truncated(self, sample_repo: Path) -> None:
        result = run_verifier(_patch_result(INVALID_DIFF), str(sample_repo))

        assert len(result["failure_trace"]) <= 2000


class TestRunVerifierCleanup:
    def test_no_worktrees_left_behind_after_success(self, sample_repo: Path) -> None:
        run_verifier(_patch_result(FIX_DIFF), str(sample_repo))

        listing = subprocess.run(
            ["git", "worktree", "list"], cwd=sample_repo, capture_output=True, text=True, check=True
        )
        assert len(listing.stdout.strip().splitlines()) == 1  # just the main worktree

    def test_no_worktrees_left_behind_after_patch_apply_failure(self, sample_repo: Path) -> None:
        run_verifier(_patch_result(INVALID_DIFF), str(sample_repo))

        listing = subprocess.run(
            ["git", "worktree", "list"], cwd=sample_repo, capture_output=True, text=True, check=True
        )
        assert len(listing.stdout.strip().splitlines()) == 1

    def test_no_worktrees_left_behind_after_timeout(self, sample_repo: Path) -> None:
        (sample_repo / "tests" / "test_slow.py").write_text(
            "import time\n\n\ndef test_slow():\n    time.sleep(5)\n", encoding="utf-8"
        )
        _git(sample_repo, "add", "-A")
        _git(sample_repo, "commit", "-q", "-m", "add slow test")

        run_verifier(_patch_result(FIX_DIFF), str(sample_repo), timeout_seconds=0.5)

        listing = subprocess.run(
            ["git", "worktree", "list"], cwd=sample_repo, capture_output=True, text=True, check=True
        )
        assert len(listing.stdout.strip().splitlines()) == 1


class TestRunVerifierInfrastructureError:
    def test_raises_when_target_repo_is_not_a_git_repo(self, tmp_path: Path) -> None:
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        with pytest.raises(VerifierError):
            run_verifier(_patch_result(FIX_DIFF), str(not_a_repo))
