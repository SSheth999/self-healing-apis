"""Unit tests for escalation/github_issues.py (AGENTS.md Section 5.8)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from escalation.github_issues import open_escalation_issue
from pr.common import EscalatedItem
from schemas import CallSite, DriftItem, TestResult


def _drift_item(item_id: str = "drift-1") -> DriftItem:
    return DriftItem(
        id=item_id,
        change_type="endpoint_moved",
        api_path="/v1/customers/{customer}/sources",
        field_or_param=None,
        old_value={},
        new_value={},
        changelog_url=None,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )


def _call_site(cs_id: str = "cs-1") -> CallSite:
    return CallSite(
        id=cs_id,
        drift_item_id="drift-1",
        file_path="billing.py",
        line_start=1,
        line_end=5,
        snippet="stripe.Customer.create_source(...)",
        symbol="stripe.Customer.create_source",
    )


def _escalated_item(**overrides: object) -> EscalatedItem:
    base = dict(
        drift_item=_drift_item(),
        call_site=_call_site(),
        test_result=TestResult(passed=False, failing_tests=["tests/test_billing.py::test_x"], failure_trace="AssertionError: boom", duration_seconds=1.0),
        retry_count=3,
    )
    base.update(overrides)
    return EscalatedItem(**base)  # type: ignore[arg-type]


class TestOpenEscalationIssueDryRun:
    def test_dry_run_never_touches_github(self) -> None:
        result = open_escalation_issue(
            api_provider="stripe", github_repo="acme/demo", escalated_items=[_escalated_item()], dry_run=True
        )

        assert result.dry_run is True
        assert result.issue_url is None
        assert "billing.py" in result.body
        assert "AssertionError: boom" in result.body

    def test_consolidates_multiple_escalated_items_into_one_issue(self) -> None:
        items = [_escalated_item(), _escalated_item(call_site=_call_site("cs-2"))]

        result = open_escalation_issue(
            api_provider="stripe", github_repo="acme/demo", escalated_items=items, dry_run=True
        )

        assert result.body.count("## `billing.py`") == 2
        assert "2 call site(s)" in result.title

    def test_raises_when_no_escalated_items(self) -> None:
        with pytest.raises(ValueError):
            open_escalation_issue(api_provider="stripe", github_repo="acme/demo", escalated_items=[], dry_run=True)

    def test_raises_when_not_dry_run_and_no_client(self) -> None:
        with pytest.raises(ValueError):
            open_escalation_issue(
                api_provider="stripe",
                github_repo="acme/demo",
                escalated_items=[_escalated_item()],
                dry_run=False,
                github_client=None,
            )

    def test_failure_trace_is_truncated_in_issue_body(self) -> None:
        huge_trace = "x" * 5000
        item = _escalated_item(
            test_result=TestResult(passed=False, failing_tests=[], failure_trace=huge_trace, duration_seconds=1.0)
        )

        result = open_escalation_issue(
            api_provider="stripe", github_repo="acme/demo", escalated_items=[item], dry_run=True
        )

        assert huge_trace not in result.body
        assert "x" * 2000 in result.body


class _FakeIssue:
    def __init__(self, html_url: str) -> None:
        self.html_url = html_url


class _FakeRepo:
    def __init__(self) -> None:
        self.created_issue: dict | None = None

    def create_issue(self, *, title: str, body: str) -> _FakeIssue:
        self.created_issue = {"title": title, "body": body}
        return _FakeIssue("https://github.com/acme/demo/issues/7")


class _FakeGithub:
    def __init__(self, repo: _FakeRepo) -> None:
        self._repo = repo

    def get_repo(self, full_name: str) -> _FakeRepo:
        return self._repo


class TestOpenEscalationIssueRealMode:
    def test_creates_issue_via_github_client(self) -> None:
        repo = _FakeRepo()
        client = _FakeGithub(repo)

        result = open_escalation_issue(
            api_provider="stripe",
            github_repo="acme/demo",
            escalated_items=[_escalated_item()],
            dry_run=False,
            github_client=client,
        )

        assert result.issue_url == "https://github.com/acme/demo/issues/7"
        assert repo.created_issue is not None
        assert "billing.py" in repo.created_issue["body"]
