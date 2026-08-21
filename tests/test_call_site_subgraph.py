"""Integration tests for subgraph.py (AGENTS.md Section 2.1's call-site
subgraph). Coder and Critic LLM calls are mocked; the Verifier is real,
running pytest against a throwaway git repo, so these tests exercise the
actual routing/retry/escalation logic end-to-end.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from coder.agent import CoderDraft
from schemas import CallSite, CriticVerdict, DriftItem
from subgraph import MAX_RETRIES, process_call_site

FIX_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # bug\n"
    "+    return a + b\n"
)

STILL_BROKEN_DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # bug\n"
    "+    return a - b  # still buggy\n"
)


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


def _drift_item() -> DriftItem:
    return DriftItem(
        id="drift-1",
        change_type="field_renamed",
        api_path="/v1/calc",
        field_or_param=None,
        old_value={},
        new_value={},
        changelog_url=None,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )


def _call_site() -> CallSite:
    return CallSite(
        id="cs-1",
        drift_item_id="drift-1",
        file_path="calc.py",
        line_start=1,
        line_end=2,
        snippet="def add(a, b):\n    return a - b  # bug",
        symbol="calc.add",
    )


class _ScriptedCoder:
    """Drop-in replacement for coder.agent.run_coder that returns a
    scripted sequence of drafts, one per call, and records what it was
    invoked with."""

    def __init__(self, diffs: list[str]) -> None:
        self._diffs = iter(diffs)
        self.calls: list[dict] = []

    def __call__(self, drift_item, call_site, target_repo, attempt_number, **kwargs):
        self.calls.append({"attempt_number": attempt_number, **kwargs})
        diff = next(self._diffs)
        return CoderDraft(diff=diff, rationale="fix"), []


class _ScriptedCritic:
    """Drop-in replacement for critic.agent.run_critic that returns a
    scripted sequence of verdicts, one per call."""

    def __init__(self, approvals: list[bool]) -> None:
        self._approvals = iter(approvals)
        self.calls = 0

    def __call__(self, drift_item, call_site, diff, *, attempt_number, revision_round):
        self.calls += 1
        approved = next(self._approvals)
        verdict = CriticVerdict(
            call_site_id=call_site["id"],
            attempt_number=attempt_number,
            revision_round=revision_round,
            approved=approved,
            feedback="" if approved else "please fix the bug for real this time",
        )
        return verdict, []


class TestFirstTrySuccess:
    def test_success_on_first_attempt(self, sample_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        coder = _ScriptedCoder([FIX_DIFF])
        critic = _ScriptedCritic([True])
        monkeypatch.setattr("subgraph.run_coder", coder)
        monkeypatch.setattr("subgraph.run_critic", critic)

        result = process_call_site(_drift_item(), _call_site(), str(sample_repo))

        assert result.outcome == "success"
        assert result.retry_count == 0
        assert result.patch_result["attempt_number"] == 1
        assert result.patch_result["critic_rounds"] == 1
        assert result.test_result["passed"] is True
        assert len(result.critic_verdicts) == 1


class TestCriticRejectionThenApprove:
    def test_coder_revises_after_critic_feedback_then_verifier_passes(
        self, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coder = _ScriptedCoder([STILL_BROKEN_DIFF, FIX_DIFF])
        critic = _ScriptedCritic([False, True])
        monkeypatch.setattr("subgraph.run_coder", coder)
        monkeypatch.setattr("subgraph.run_critic", critic)

        result = process_call_site(_drift_item(), _call_site(), str(sample_repo))

        assert result.outcome == "success"
        assert result.patch_result["diff"] == FIX_DIFF
        assert result.patch_result["critic_rounds"] == 2
        assert len(result.critic_verdicts) == 2
        assert result.critic_verdicts[0]["approved"] is False
        assert result.critic_verdicts[1]["approved"] is True
        # the revision round's Coder call should have received the rejection feedback
        assert coder.calls[1]["critic_feedback"] == "please fix the bug for real this time"
        assert coder.calls[1]["previous_diff"] == STILL_BROKEN_DIFF
        # still the same attempt - only one Verifier round-trip happened
        assert result.retry_count == 0


class TestVerifierFailThenRetryThenPass:
    def test_retries_with_failure_trace_after_verifier_failure(
        self, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coder = _ScriptedCoder([STILL_BROKEN_DIFF, FIX_DIFF])
        critic = _ScriptedCritic([True, True])  # critic approves both times; Verifier catches the real bug
        monkeypatch.setattr("subgraph.run_coder", coder)
        monkeypatch.setattr("subgraph.run_critic", critic)

        result = process_call_site(_drift_item(), _call_site(), str(sample_repo))

        assert result.outcome == "success"
        assert result.retry_count == 1
        assert result.patch_result["attempt_number"] == 2
        assert result.patch_result["diff"] == FIX_DIFF
        # second Coder call should have received the first attempt's failure trace
        assert coder.calls[1]["attempt_number"] == 2
        assert coder.calls[1]["failure_trace"] is not None
        assert "test_add" in coder.calls[1]["failure_trace"]


class TestFullEscalation:
    def test_escalates_after_max_retries_exhausted(
        self, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coder = _ScriptedCoder([STILL_BROKEN_DIFF] * MAX_RETRIES)
        critic = _ScriptedCritic([True] * MAX_RETRIES)
        monkeypatch.setattr("subgraph.run_coder", coder)
        monkeypatch.setattr("subgraph.run_critic", critic)

        result = process_call_site(_drift_item(), _call_site(), str(sample_repo))

        assert result.outcome == "escalated"
        assert result.retry_count == MAX_RETRIES
        assert result.test_result["passed"] is False
        assert len(coder.calls) == MAX_RETRIES
        assert critic.calls == MAX_RETRIES

    def test_no_worktrees_left_behind_after_escalation(
        self, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coder = _ScriptedCoder([STILL_BROKEN_DIFF] * MAX_RETRIES)
        critic = _ScriptedCritic([True] * MAX_RETRIES)
        monkeypatch.setattr("subgraph.run_coder", coder)
        monkeypatch.setattr("subgraph.run_critic", critic)

        process_call_site(_drift_item(), _call_site(), str(sample_repo))

        listing = subprocess.run(
            ["git", "worktree", "list"], cwd=sample_repo, capture_output=True, text=True, check=True
        )
        assert len(listing.stdout.strip().splitlines()) == 1
