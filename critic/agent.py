"""Critic node: reviews one Coder-proposed diff before it reaches the Verifier.

AGENTS.md Section 5.5. Exactly one LLM call per invocation, no tool loop.
A quality filter, not a hard gate: it never escalates a call site on its
own, and it can't modify the diff itself - only approve or give feedback.
Whatever the Coder's latest diff is when the revision exchange ends still
goes to the Verifier regardless of the Critic's verdict (AGENTS.md Section
2.2 and 4.5).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel

from llm.client import get_chat_model
from schemas import AgentStep, CallSite, CriticVerdict, DriftItem

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "critic_template.txt"
MAX_ATTEMPTS = 3


class CriticOutputError(RuntimeError):
    """Raised when the Critic can't produce valid structured output within MAX_ATTEMPTS."""


class _CriticLLMOutput(TypedDict):
    """The semantic-only shape the LLM produces. Bookkeeping fields
    (call_site_id, attempt_number, revision_round) are filled in by this
    module afterward, from values the caller already knows."""

    approved: bool
    feedback: str


def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_prompt(drift_item: DriftItem, call_site: CallSite, diff: str) -> str:
    template = _load_prompt_template()
    return template.format(
        change_type=drift_item["change_type"],
        api_path=drift_item["api_path"],
        field_or_param=drift_item["field_or_param"] or "(none)",
        old_value=json.dumps(drift_item["old_value"]),
        new_value=json.dumps(drift_item["new_value"]),
        file_path=call_site["file_path"],
        symbol=call_site["symbol"],
        snippet=call_site["snippet"],
        diff=diff,
    )


def _coerce_output(raw_output: object) -> _CriticLLMOutput:
    if isinstance(raw_output, dict):
        data = raw_output
    elif hasattr(raw_output, "model_dump"):
        data = raw_output.model_dump()
    else:
        raise ValueError(f"Unexpected structured output type: {type(raw_output)!r}")

    approved = data.get("approved")
    feedback = data.get("feedback", "")
    if not isinstance(approved, bool):
        raise ValueError(f"'approved' must be a bool, got {approved!r}")
    if not approved and not (isinstance(feedback, str) and feedback):
        raise ValueError("'feedback' must be a non-empty string when approved=False")
    if not isinstance(feedback, str):
        raise ValueError(f"'feedback' must be a string, got {feedback!r}")

    return _CriticLLMOutput(approved=approved, feedback=feedback)


def run_critic(
    drift_item: DriftItem,
    call_site: CallSite,
    diff: str,
    *,
    attempt_number: int,
    revision_round: int,
    model: BaseChatModel | None = None,
) -> tuple[CriticVerdict, list[AgentStep]]:
    """Run the Critic once against one proposed diff.

    Raises CriticOutputError if MAX_ATTEMPTS is exhausted without valid
    structured output - this must propagate, never be swallowed (AGENTS.md
    Section 6.2). This is a malformed-output retry only; it never becomes
    a de-facto rejection of the diff itself.
    """

    chat_model = model or get_chat_model("critic")
    structured_model = chat_model.with_structured_output(_CriticLLMOutput)
    prompt = _build_prompt(drift_item, call_site, diff)

    steps: list[AgentStep] = []
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw_output = structured_model.invoke(prompt)
            output = _coerce_output(raw_output)
        except Exception as exc:  # noqa: BLE001 - malformed output is a failed attempt, not a crash
            last_error = exc
            steps.append(
                AgentStep(
                    node="critic",
                    drift_item_id=None,
                    call_site_id=call_site["id"],
                    attempt_number=attempt_number,
                    step_number=attempt,
                    tool_called=None,
                    tool_args=None,
                    tool_result_summary=f"malformed structured output: {str(exc)[:200]}",
                    output=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            logger.warning(
                "Critic attempt %d/%d failed for call_site=%s: %s", attempt, MAX_ATTEMPTS, call_site["id"], exc
            )
            continue

        steps.append(
            AgentStep(
                node="critic",
                drift_item_id=None,
                call_site_id=call_site["id"],
                attempt_number=attempt_number,
                step_number=attempt,
                tool_called=None,
                tool_args=None,
                tool_result_summary=None,
                output=dict(output),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        verdict = CriticVerdict(
            call_site_id=call_site["id"],
            attempt_number=attempt_number,
            revision_round=revision_round,
            approved=output["approved"],
            feedback=output["feedback"],
        )
        return verdict, steps

    raise CriticOutputError(
        f"Critic failed to produce valid structured output for call_site={call_site['id']} "
        f"after {MAX_ATTEMPTS} attempts: {last_error}"
    )
