"""Escalation Node: opens a GitHub issue for every call site that
exhausted its retry budget without a passing patch.

AGENTS.md Section 5.8. Pure code, no LLM. This node exists so failure is
always visible - a run that ends in "nothing happened, no error, no
output" is a bug in this system, full stop (AGENTS.md Section 5.8's
"Principle"). All escalated items in a single run are consolidated into
one issue rather than opened one-by-one, per Section 5.8's "or one
consolidated issue if several call sites in the same run escalated."
"""

from __future__ import annotations

import logging

from pr.common import EscalatedItem, describe_drift

logger = logging.getLogger(__name__)

MAX_FAILURE_TRACE_CHARS_IN_ISSUE = 2000  # AGENTS.md Section 6.3


class EscalationResult:
    def __init__(self, *, dry_run: bool, title: str, body: str, issue_url: str | None) -> None:
        self.dry_run = dry_run
        self.title = title
        self.body = body
        self.issue_url = issue_url


def _build_issue_body(items: list[EscalatedItem]) -> str:
    sections = [
        "The self-healing pipeline could not produce a passing patch for the "
        f"following {len(items)} call site(s) within its retry budget. A human needs "
        "to look at these.\n"
    ]

    for item in items:
        trace = (item.test_result["failure_trace"] or "(no failure trace captured)")[-MAX_FAILURE_TRACE_CHARS_IN_ISSUE:]
        sections.append(
            f"## `{item.call_site['file_path']}` ({item.call_site['symbol']})\n"
            f"- **What changed:** {describe_drift(item.drift_item)}\n"
            f"- **Attempts made:** {item.retry_count}\n"
            f"- **Last failure trace:**\n```\n{trace}\n```\n"
        )

    return "\n".join(sections)


def open_escalation_issue(
    *,
    api_provider: str,
    github_repo: str,
    escalated_items: list[EscalatedItem],
    dry_run: bool,
    github_client: object | None = None,
) -> EscalationResult:
    """Open (or, in dry-run mode, describe) a single consolidated GitHub
    issue covering every escalated call site from this run."""

    if not escalated_items:
        raise ValueError("open_escalation_issue called with no escalated items")

    title = f"Self-healing pipeline could not auto-fix {len(escalated_items)} call site(s) for {api_provider} API drift"
    body = _build_issue_body(escalated_items)

    if dry_run:
        logger.info("[dry-run] Would open escalation issue title=%r", title)
        return EscalationResult(dry_run=True, title=title, body=body, issue_url=None)

    if github_client is None:
        raise ValueError("github_client is required when dry_run=False")

    repo = github_client.get_repo(github_repo)
    issue = repo.create_issue(title=title, body=body)
    return EscalationResult(dry_run=False, title=title, body=body, issue_url=issue.html_url)
