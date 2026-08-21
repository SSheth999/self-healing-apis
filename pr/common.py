"""Shared types/helpers for the PR Node and Escalation Node.

Not part of AGENTS.md Section 4's canonical cross-node data model - these
are internal bundling structures for calling pr.github_client and
escalation.github_issues conveniently, kept separate from schemas.py so
that file stays a clean 1:1 mirror of AGENTS.md Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import CallSite, DriftItem, PatchResult, TestResult

_CHANGE_TYPE_ENGLISH = {
    "field_renamed": "renamed field",
    "field_removed": "removed field",
    "field_required_changed": "field became required",
    "endpoint_moved": "moved/renamed endpoint",
    "param_type_changed": "changed parameter type",
}


@dataclass
class SuccessfulPatch:
    drift_item: DriftItem
    call_site: CallSite
    patch_result: PatchResult
    test_result: TestResult


@dataclass
class EscalatedItem:
    drift_item: DriftItem
    call_site: CallSite
    test_result: TestResult
    retry_count: int


def describe_drift(drift_item: DriftItem) -> str:
    """Render a DriftItem as a short plain-English description, for PR/issue bodies."""

    change_type_text = _CHANGE_TYPE_ENGLISH.get(drift_item["change_type"], drift_item["change_type"])
    field_suffix = f" (`{drift_item['field_or_param']}`)" if drift_item["field_or_param"] else ""
    return f"{change_type_text} on `{drift_item['api_path']}`{field_suffix}"
