"""Planner node: turns one DriftItem into a SearchPlan.

AGENTS.md Section 5.2. Exactly one LLM call per invocation - no tool loop
(that's the Coder's job, Section 5.4). This is the one place genuine
judgment about "which symbols does this drift affect" lives; the Locator
that consumes this output stays a deterministic AST walker (Section 5.3).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel

from llm.client import get_chat_model
from schemas import AgentStep, DriftItem, SearchPlan

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "plan_template.txt"
MAX_ATTEMPTS = 3


class PlannerOutputError(RuntimeError):
    """Raised when the Planner can't produce valid structured output within MAX_ATTEMPTS."""


class _PlannerLLMOutput(TypedDict):
    """The semantic-only shape the LLM itself produces.

    Bookkeeping fields (drift_item_id) are filled in by this module after
    the call - the model never needs to invent an id it was already given
    verbatim in its own prompt.
    """

    symbols: list[str]
    rationale: str


def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_prompt(drift_item: DriftItem) -> str:
    template = _load_prompt_template()
    return template.format(
        change_type=drift_item["change_type"],
        api_path=drift_item["api_path"],
        field_or_param=drift_item["field_or_param"] or "(none)",
        old_value=json.dumps(drift_item["old_value"]),
        new_value=json.dumps(drift_item["new_value"]),
    )


def _coerce_output(raw_output: object) -> _PlannerLLMOutput:
    """Validate/normalize whatever with_structured_output() returned.

    LangChain's with_structured_output can return a dict (for TypedDict
    schemas) or a pydantic model depending on provider/method; normalize
    defensively rather than assuming one shape.
    """

    if isinstance(raw_output, dict):
        data = raw_output
    elif hasattr(raw_output, "model_dump"):
        data = raw_output.model_dump()
    else:
        raise ValueError(f"Unexpected structured output type: {type(raw_output)!r}")

    symbols = data.get("symbols")
    rationale = data.get("rationale")
    if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
        raise ValueError(f"'symbols' must be a non-empty list of strings, got {symbols!r}")
    if not isinstance(rationale, str) or not rationale:
        raise ValueError(f"'rationale' must be a non-empty string, got {rationale!r}")

    return _PlannerLLMOutput(symbols=symbols, rationale=rationale)


def run_planner(
    drift_item: DriftItem,
    *,
    model: BaseChatModel | None = None,
) -> tuple[SearchPlan, list[AgentStep]]:
    """Run the Planner once for a single DriftItem.

    Returns the resulting SearchPlan plus every AgentStep logged during
    this invocation, including failed-parse retries. A malformed structured
    output is treated as a failed attempt and retried, per AGENTS.md
    Section 5.2, bullet 3 - it never gets passed downstream as-is. This
    retry does not consume any per-CallSite retry budget, since no
    CallSite exists yet at this point in the pipeline.

    Raises PlannerOutputError if MAX_ATTEMPTS is exhausted without valid
    output - this must propagate, never be swallowed (Section 6.2).
    """

    chat_model = model or get_chat_model("planner")
    structured_model = chat_model.with_structured_output(_PlannerLLMOutput)
    prompt = _build_prompt(drift_item)

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
                    node="planner",
                    drift_item_id=drift_item["id"],
                    call_site_id=None,
                    attempt_number=None,
                    step_number=attempt,
                    tool_called=None,
                    tool_args=None,
                    tool_result_summary=f"malformed structured output: {str(exc)[:200]}",
                    output=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            logger.warning(
                "Planner attempt %d/%d failed for drift_item=%s: %s", attempt, MAX_ATTEMPTS, drift_item["id"], exc
            )
            continue

        steps.append(
            AgentStep(
                node="planner",
                drift_item_id=drift_item["id"],
                call_site_id=None,
                attempt_number=None,
                step_number=attempt,
                tool_called=None,
                tool_args=None,
                tool_result_summary=None,
                output=dict(output),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        search_plan = SearchPlan(
            drift_item_id=drift_item["id"],
            symbols=output["symbols"],
            rationale=output["rationale"],
        )
        return search_plan, steps

    raise PlannerOutputError(
        f"Planner failed to produce valid structured output for drift_item={drift_item['id']} "
        f"after {MAX_ATTEMPTS} attempts: {last_error}"
    )
