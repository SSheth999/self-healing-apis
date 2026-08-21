"""Unit tests for critic/agent.py (AGENTS.md Section 5.5). LLM is mocked."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from critic.agent import MAX_ATTEMPTS, CriticOutputError, run_critic
from schemas import CallSite, DriftItem


def _drift_item() -> DriftItem:
    return DriftItem(
        id="drift-1",
        change_type="field_renamed",
        api_path="/v1/charges",
        field_or_param="source -> payment_method",
        old_value={"field": "source"},
        new_value={"field": "payment_method"},
        changelog_url=None,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )


def _call_site() -> CallSite:
    return CallSite(
        id="cs-1",
        drift_item_id="drift-1",
        file_path="billing.py",
        line_start=1,
        line_end=10,
        snippet="stripe.Charge.create(source=source_token)",
        symbol="stripe.Charge.create",
    )


class _FakeStructuredRunnable:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)

    def invoke(self, prompt: str) -> object:
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(self, responses: list[object]) -> None:
        self._runnable = _FakeStructuredRunnable(responses)

    def with_structured_output(self, schema: object) -> _FakeStructuredRunnable:
        return self._runnable


class TestRunCriticApproval:
    def test_approves_a_good_diff(self) -> None:
        model = FakeChatModel([{"approved": True, "feedback": ""}])

        verdict, steps = run_critic(
            _drift_item(), _call_site(), "a valid diff", attempt_number=1, revision_round=1, model=model  # type: ignore[arg-type]
        )

        assert verdict["approved"] is True
        assert verdict["call_site_id"] == "cs-1"
        assert verdict["attempt_number"] == 1
        assert verdict["revision_round"] == 1
        assert len(steps) == 1
        assert steps[0]["node"] == "critic"


class TestRunCriticRejection:
    def test_rejects_with_feedback(self) -> None:
        model = FakeChatModel([{"approved": False, "feedback": "this touches an unrelated field"}])

        verdict, _ = run_critic(
            _drift_item(), _call_site(), "a bad diff", attempt_number=1, revision_round=1, model=model  # type: ignore[arg-type]
        )

        assert verdict["approved"] is False
        assert verdict["feedback"] == "this touches an unrelated field"

    def test_rejection_without_feedback_is_treated_as_malformed_and_retried(self) -> None:
        model = FakeChatModel(
            [
                {"approved": False, "feedback": ""},  # invalid: rejection needs feedback
                {"approved": False, "feedback": "now with feedback"},
            ]
        )

        verdict, steps = run_critic(
            _drift_item(), _call_site(), "a bad diff", attempt_number=1, revision_round=1, model=model  # type: ignore[arg-type]
        )

        assert verdict["approved"] is False
        assert verdict["feedback"] == "now with feedback"
        assert len(steps) == 2


class TestRunCriticFailure:
    def test_raises_after_exhausting_max_attempts(self) -> None:
        model = FakeChatModel([{"approved": False, "feedback": ""} for _ in range(MAX_ATTEMPTS)])

        with pytest.raises(CriticOutputError):
            run_critic(_drift_item(), _call_site(), "diff", attempt_number=1, revision_round=1, model=model)  # type: ignore[arg-type]

    def test_never_modifies_the_diff_itself(self) -> None:
        # The Critic's verdict schema has no diff field at all - this test
        # exists as a structural guardrail: if someone ever adds a way for
        # the Critic to return a modified diff, this should fail loudly.
        model = FakeChatModel([{"approved": True, "feedback": ""}])

        verdict, _ = run_critic(
            _drift_item(), _call_site(), "original diff", attempt_number=1, revision_round=1, model=model  # type: ignore[arg-type]
        )

        assert "diff" not in verdict
