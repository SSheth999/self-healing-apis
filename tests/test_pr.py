"""Unit tests for pr/github_client.py (AGENTS.md Section 5.7).

The GitHub API itself is always mocked here - these tests never make a
real network call. Section 3, rule 4 ("no merge endpoint anywhere in this
codebase") is checked structurally: nothing in pr/github_client.py calls
anything named merge, and the fake repo double below doesn't even expose
a merge method, so any accidental call would raise AttributeError.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pr.common import EscalatedItem, SuccessfulPatch, describe_drift
from pr.github_client import PRNodeError, open_or_update_pr
from schemas import CallSite, DriftItem, PatchResult, TestResult

DIFF = (
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b  # bug\n"
    "+    return a + b\n"
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
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-q", "-m", "init")
    return repo_dir


def _drift_item(**overrides: object) -> DriftItem:
    base = DriftItem(
        id="drift-1",
        change_type="field_renamed",
        api_path="/v1/charges",
        field_or_param="source -> payment_method",
        old_value={},
        new_value={},
        changelog_url="https://stripe.com/docs/changelog",
        detected_at=datetime.now(timezone.utc).isoformat(),
    )
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _call_site() -> CallSite:
    return CallSite(
        id="cs-1",
        drift_item_id="drift-1",
        file_path="calc.py",
        line_start=1,
        line_end=2,
        snippet="def add(a, b):\n    return a - b",
        symbol="calc.add",
    )


def _patch_result() -> PatchResult:
    return PatchResult(call_site_id="cs-1", diff=DIFF, rationale="fixed the sign error", attempt_number=1, critic_rounds=1)


def _test_result(passed: bool = True) -> TestResult:
    return TestResult(passed=passed, failing_tests=[], failure_trace=None if passed else "boom", duration_seconds=0.42)


def _successful_patch() -> SuccessfulPatch:
    return SuccessfulPatch(
        drift_item=_drift_item(), call_site=_call_site(), patch_result=_patch_result(), test_result=_test_result()
    )


class TestDescribeDrift:
    def test_includes_change_type_path_and_field(self) -> None:
        text = describe_drift(_drift_item())
        assert "renamed field" in text
        assert "/v1/charges" in text
        assert "source -> payment_method" in text


class TestOpenOrUpdatePrDryRun:
    def test_dry_run_never_touches_github_and_returns_no_url(self) -> None:
        result = open_or_update_pr(
            api_provider="stripe",
            target_repo="unused",
            github_repo="acme/demo",
            successful_patches=[_successful_patch()],
            escalated_items=[],
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.pr_url is None
        assert "calc.py" in result.body
        assert "fixed the sign error" in result.body

    def test_dry_run_mentions_escalated_items_as_not_automatically_fixed(self) -> None:
        escalated = EscalatedItem(
            drift_item=_drift_item(id="drift-2"), call_site=_call_site(), test_result=_test_result(passed=False), retry_count=3
        )

        result = open_or_update_pr(
            api_provider="stripe",
            target_repo="unused",
            github_repo="acme/demo",
            successful_patches=[_successful_patch()],
            escalated_items=[escalated],
            dry_run=True,
        )

        assert "Not automatically fixed" in result.body

    def test_raises_when_no_successful_patches(self) -> None:
        with pytest.raises(ValueError):
            open_or_update_pr(
                api_provider="stripe",
                target_repo="unused",
                github_repo="acme/demo",
                successful_patches=[],
                escalated_items=[],
                dry_run=True,
            )

    def test_raises_when_not_dry_run_and_no_client_given(self, sample_repo: Path) -> None:
        with pytest.raises(ValueError):
            open_or_update_pr(
                api_provider="stripe",
                target_repo=str(sample_repo),
                github_repo="acme/demo",
                successful_patches=[_successful_patch()],
                escalated_items=[],
                dry_run=False,
                github_client=None,
            )


class _FakeCommit:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class _FakeBranch:
    def __init__(self, sha: str) -> None:
        self.commit = _FakeCommit(sha)


class _FakeContentFile:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class _FakePR:
    def __init__(self, html_url: str, head_ref: str) -> None:
        self.html_url = html_url
        self._head_ref = head_ref
        self.edited: dict | None = None

    def edit(self, *, title: str, body: str) -> None:
        self.edited = {"title": title, "body": body}


class _FakeOwner:
    login = "acme"


class _FakeRepo:
    """No merge-related method exists on this double at all - if
    open_or_update_pr ever tried to call one, this would raise
    AttributeError, which is exactly the guardrail we want."""

    def __init__(self, *, existing_branch: bool = False, existing_pr: bool = False) -> None:
        self.default_branch = "main"
        self.owner = _FakeOwner()
        self._existing_branch = existing_branch
        self._existing_pr = _FakePR("https://github.com/acme/demo/pull/1", "self-healing/stripe") if existing_pr else None
        self.created_refs: list[str] = []
        self.updated_files: dict[str, str] = {}
        self.created_files: dict[str, str] = {}
        self.created_pr: dict | None = None

    def get_branch(self, name: str) -> _FakeBranch:
        return _FakeBranch("base-sha-123")

    def get_git_ref(self, ref: str):
        if not self._existing_branch:
            raise Exception("not found")
        return object()

    def create_git_ref(self, *, ref: str, sha: str) -> None:
        self.created_refs.append(ref)

    def get_contents(self, path: str, ref: str):
        if path in self.updated_files or self._existing_branch:
            return _FakeContentFile("file-sha-abc")
        raise Exception("not found")

    def update_file(self, path: str, *, message: str, content: str, sha: str, branch: str) -> None:
        self.updated_files[path] = content

    def create_file(self, path: str, *, message: str, content: str, branch: str) -> None:
        self.created_files[path] = content

    def get_pulls(self, *, state: str, head: str):
        if self._existing_pr is not None:
            return [self._existing_pr]
        return []

    def create_pull(self, *, title: str, body: str, head: str, base: str) -> _FakePR:
        self.created_pr = {"title": title, "body": body, "head": head, "base": base}
        return _FakePR("https://github.com/acme/demo/pull/2", head)


class _FakeGithub:
    def __init__(self, repo: _FakeRepo) -> None:
        self._repo = repo

    def get_repo(self, full_name: str) -> _FakeRepo:
        return self._repo


class TestOpenOrUpdatePrRealMode:
    def test_creates_new_branch_and_new_pr_when_none_exist(self, sample_repo: Path) -> None:
        repo = _FakeRepo(existing_branch=False, existing_pr=False)
        client = _FakeGithub(repo)

        result = open_or_update_pr(
            api_provider="stripe",
            target_repo=str(sample_repo),
            github_repo="acme/demo",
            successful_patches=[_successful_patch()],
            escalated_items=[],
            dry_run=False,
            github_client=client,
        )

        assert result.pr_url == "https://github.com/acme/demo/pull/2"
        assert repo.created_refs == ["refs/heads/self-healing/stripe"]
        assert "calc.py" in repo.created_files
        assert "return a + b" in repo.created_files["calc.py"]
        assert repo.created_pr is not None
        assert repo.created_pr["head"] == "self-healing/stripe"

    def test_updates_existing_branch_and_existing_pr_instead_of_duplicating(self, sample_repo: Path) -> None:
        repo = _FakeRepo(existing_branch=True, existing_pr=True)
        client = _FakeGithub(repo)

        result = open_or_update_pr(
            api_provider="stripe",
            target_repo=str(sample_repo),
            github_repo="acme/demo",
            successful_patches=[_successful_patch()],
            escalated_items=[],
            dry_run=False,
            github_client=client,
        )

        assert result.pr_url == "https://github.com/acme/demo/pull/1"
        assert repo.created_refs == []  # did not create a new branch
        assert repo.created_pr is None  # did not open a duplicate PR
        assert "calc.py" in repo.updated_files

    def test_raises_if_patch_cannot_be_reapplied(self, sample_repo: Path) -> None:
        bad_patch = SuccessfulPatch(
            drift_item=_drift_item(),
            call_site=_call_site(),
            patch_result=PatchResult(
                call_site_id="cs-1", diff="not a real diff", rationale="x", attempt_number=1, critic_rounds=1
            ),
            test_result=_test_result(),
        )
        repo = _FakeRepo()
        client = _FakeGithub(repo)

        with pytest.raises(PRNodeError):
            open_or_update_pr(
                api_provider="stripe",
                target_repo=str(sample_repo),
                github_repo="acme/demo",
                successful_patches=[bad_patch],
                escalated_items=[],
                dry_run=False,
                github_client=client,
            )
