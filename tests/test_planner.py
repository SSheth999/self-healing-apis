"""Unit tests for planner/agent.py (AGENTS.md Section 5.2). LLM is mocked."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from planner.agent import MAX_ATTEMPTS, PlannerOutputError, run_planner
from schemas import DriftItem


def _drift_item(**overrides: object) -> DriftItem:
    base = DriftItem(
        id="drift-1",
        change_type="field_renamed",
        api_path="/v1/charges",
        field_or_param="source -> payment_method",
        old_value={"field": "source"},
        new_value={"field": "payment_method"},
        changelog_url=None,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class _FakeStructuredRunnable:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.prompts_seen: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts_seen.append(prompt)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(self, responses: list[object]) -> None:
        self._runnable = _FakeStructuredRunnable(responses)

    def with_structured_output(self, schema: object) -> _FakeStructuredRunnable:
        return self._runnable


class TestRunPlannerSuccess:
    def test_returns_search_plan_on_first_attempt(self) -> None:
        model = FakeChatModel([{"symbols": ["stripe.Charge.create"], "rationale": "renamed field on charges"}])

        plan, steps = run_planner(_drift_item(), model=model)  # type: ignore[arg-type]

        assert plan["drift_item_id"] == "drift-1"
        assert plan["symbols"] == ["stripe.Charge.create"]
        assert plan["rationale"] == "renamed field on charges"
        assert len(steps) == 1
        assert steps[0]["node"] == "planner"
        assert steps[0]["tool_called"] is None
        assert steps[0]["output"] == {"symbols": ["stripe.Charge.create"], "rationale": "renamed field on charges"}

    def test_prompt_includes_drift_fields_but_not_full_spec(self) -> None:
        model = FakeChatModel([{"symbols": ["stripe.Charge.create"], "rationale": "x"}])

        run_planner(_drift_item(), model=model)  # type: ignore[arg-type]

        prompt = model._runnable.prompts_seen[0]
        assert "/v1/charges" in prompt
        assert "field_renamed" in prompt

    def test_retries_once_on_malformed_output_then_succeeds(self) -> None:
        model = FakeChatModel(
            [
                {"symbols": [], "rationale": "empty list is invalid"},  # malformed: empty symbols
                {"symbols": ["stripe.Charge.create"], "rationale": "valid this time"},
            ]
        )

        plan, steps = run_planner(_drift_item(), model=model)  # type: ignore[arg-type]

        assert plan["symbols"] == ["stripe.Charge.create"]
        assert len(steps) == 2
        assert steps[0]["output"] is None
        assert steps[0]["tool_result_summary"] is not None
        assert steps[1]["output"] is not None


class TestRunPlannerFailure:
    def test_raises_after_exhausting_max_attempts(self) -> None:
        model = FakeChatModel([{"symbols": [], "rationale": "bad"} for _ in range(MAX_ATTEMPTS)])

        with pytest.raises(PlannerOutputError):
            run_planner(_drift_item(), model=model)  # type: ignore[arg-type]

    def test_llm_exception_is_treated_as_failed_attempt_not_a_crash(self) -> None:
        model = FakeChatModel(
            [RuntimeError("provider timeout"), {"symbols": ["stripe.Refund.create"], "rationale": "ok"}]
        )

        plan, steps = run_planner(_drift_item(), model=model)  # type: ignore[arg-type]

        assert plan["symbols"] == ["stripe.Refund.create"]
        assert len(steps) == 2
        assert "provider timeout" in steps[0]["tool_result_summary"]
