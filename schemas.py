"""Canonical cross-node data shapes for the self-healing pipeline.

This module is the executable form of AGENTS.md Section 4. If code and
AGENTS.md ever disagree about a shape, AGENTS.md wins and this file should be
updated to match - see AGENTS.md Section 4's own framing.

Every node/agent in graph.py takes and returns (slices of) `HealingState`.
Nothing crosses a node boundary as a raw string of prose; LLM-produced
content always lands in one of the typed shapes below before it re-enters
state (see AGENTS.md Section 2.2, "Structured data between every node").
"""

from __future__ import annotations

from typing import Literal, TypedDict


class DriftItem(TypedDict):
    """A single detected breaking change in a provider's API contract.

    AGENTS.md Section 4.1.
    """

    id: str  # stable id, e.g. hash of (path, change_type)
    change_type: Literal[
        "field_renamed",
        "field_removed",
        "field_required_changed",
        "endpoint_moved",
        "param_type_changed",
    ]
    api_path: str  # e.g. "/v1/charges"
    field_or_param: str | None  # e.g. "source -> payment_method"
    old_value: dict  # minimal relevant slice of old spec
    new_value: dict  # minimal relevant slice of new spec
    changelog_url: str | None
    detected_at: str  # ISO 8601 timestamp


class CallSite(TypedDict):
    """A single location in the target repo affected by a DriftItem.

    AGENTS.md Section 4.2.
    """

    id: str  # stable id, e.g. hash of (file_path, line_start)
    drift_item_id: str  # foreign key to DriftItem.id
    file_path: str
    line_start: int
    line_end: int
    snippet: str  # exact source text, nothing more
    symbol: str  # e.g. "stripe.Charge.create"


class SearchPlan(TypedDict):
    """The Planner's structured output: which symbols a DriftItem affects.

    AGENTS.md Section 4.3. Emitted by the Planner (planner/agent.py),
    consumed by the Locator (locator/ast_scanner.py) as input for which
    symbols to AST-scan for. The Locator never decides *what* to search for.
    """

    drift_item_id: str  # foreign key to DriftItem.id
    symbols: list[str]  # e.g. ["stripe.Charge.create", "stripe.Charge.modify"]
    rationale: str  # why these symbols, one/two sentences


class PatchResult(TypedDict):
    """The Coder's structured output for one attempt at one call site.

    AGENTS.md Section 4.4.
    """

    call_site_id: str
    diff: str  # unified diff format, single file
    rationale: str  # one sentence, for the PR description
    attempt_number: int  # 1-indexed, resets per call_site
    critic_rounds: int  # how many Coder<->Critic revisions this attempt took


class CriticVerdict(TypedDict):
    """The Critic's structured review of a Coder-proposed diff.

    AGENTS.md Section 4.5. A quality filter, not a hard gate - it never
    escalates a call site on its own. If revisions keep getting rejected,
    the last diff still goes to the Verifier; the real test suite is the
    ground truth, not the Critic's opinion (see AGENTS.md Section 2.2).
    """

    call_site_id: str
    attempt_number: int  # which PatchResult attempt this critiques
    revision_round: int  # 1-indexed within this attempt
    approved: bool
    feedback: str  # fed back to Coder verbatim if not approved


class AgentStep(TypedDict):
    """One logged step (LLM call or tool call) from any agent.

    AGENTS.md Section 4.6. This is the mechanism behind the "structured
    trace" auditability principle in Section 2.2 - every LLM call and every
    tool call, for every agent, in order, with nothing dropped.
    `tool_result_summary` and `output` are always the structured/truncated
    forms, never a raw dump (see Section 6.3 on size limits).
    """

    node: Literal["planner", "coder", "critic"]
    drift_item_id: str | None  # set for planner steps
    call_site_id: str | None  # set for coder/critic steps
    attempt_number: int | None  # set for coder/critic steps
    step_number: int  # 1-indexed within this single node invocation
    tool_called: str | None  # e.g. "expand_snippet", "search_repo"; None if a final-output step
    tool_args: dict | None
    tool_result_summary: str | None  # truncated, see Section 6.3
    output: dict | None  # the structured output if this is a final step
    timestamp: str  # ISO 8601


class TestResult(TypedDict):
    """The Verifier's structured output for one PatchResult.

    AGENTS.md Section 4.7.
    """

    passed: bool
    failing_tests: list[str]  # test node ids, empty if passed
    failure_trace: str | None  # truncated, see Section 6.3 on size limits
    duration_seconds: float


TestResult.__test__ = False  # tell pytest this isn't a test class despite the name


class HealingState(TypedDict):
    """The LangGraph shared state threaded through the whole pipeline.

    AGENTS.md Section 4.8. Processed one CallSite at a time through
    Coder<->Critic->Verifier, not batched - see the Rule immediately
    following this class in AGENTS.md.
    """

    api_provider: str
    spec_version_old: dict
    spec_version_new: dict
    drift_report: list[DriftItem]
    target_repo: str
    search_plans: dict[str, SearchPlan]  # keyed by drift_item_id
    affected_call_sites: list[CallSite]
    current_call_site_id: str | None  # which call site is being processed now
    patch_results: dict[str, PatchResult]  # keyed by call_site_id
    critic_verdicts: dict[str, list[CriticVerdict]]  # keyed by call_site_id
    agent_traces: dict[str, list[AgentStep]]  # keyed by call_site_id, or "planner:<drift_item_id>"
    test_results: dict[str, TestResult]  # keyed by call_site_id
    retry_counts: dict[str, int]  # keyed by call_site_id
    escalated_call_site_ids: list[str]
    pr_url: str | None
    dry_run: bool  # execution flag, not part of AGENTS.md 4.8's canonical list but needed by PR/Escalation nodes


def new_healing_state(
    *,
    api_provider: str,
    spec_version_old: dict,
    spec_version_new: dict,
    target_repo: str,
    dry_run: bool = False,
) -> HealingState:
    """Construct an initial HealingState with all accumulator fields empty.

    Centralizing this avoids every call site re-typing the same empty
    dict/list boilerplate and risking a typo in a key name.
    """

    return HealingState(
        api_provider=api_provider,
        spec_version_old=spec_version_old,
        spec_version_new=spec_version_new,
        drift_report=[],
        target_repo=target_repo,
        search_plans={},
        affected_call_sites=[],
        current_call_site_id=None,
        patch_results={},
        critic_verdicts={},
        agent_traces={},
        test_results={},
        retry_counts={},
        escalated_call_site_ids=[],
        pr_url=None,
        dry_run=dry_run,
    )
