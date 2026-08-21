"""Unit tests for coder/agent.py and coder/tools.py (AGENTS.md Section 5.4).

The LLM's tool-calling loop is exercised with a scripted fake model, per
AGENTS.md Section 8's guidance to test the loop logic itself (does it call
tools when expected, does it respect feedback on revision), not just the
final diff output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from coder.agent import run_coder
from coder.tools import MAX_TOTAL_WINDOW_LINES, expand_snippet, search_repo
from schemas import CallSite, DriftItem

VALID_DIFF = (
    "--- a/billing.py\n"
    "+++ b/billing.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-source=source_token,\n"
    "+payment_method=source_token,\n"
)


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
        snippet="stripe.Charge.create(amount=amount, currency=currency, source=source_token)",
        symbol="stripe.Charge.create",
    )


class _FakeBoundModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = iter(responses)
        self.invocations = 0

    def invoke(self, messages: list) -> AIMessage:
        self.invocations += 1
        return next(self._responses)


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._bound = _FakeBoundModel(responses)

    def bind_tools(self, tools: list) -> _FakeBoundModel:
        return self._bound


class TestRunCoderImmediateSubmit:
    def test_submits_valid_diff_on_first_turn(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(source=x)\n" * 1, encoding="utf-8")
        model = FakeChatModel(
            [AIMessage(content="", tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "1"}])]
        )

        draft, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=1, model=model)  # type: ignore[arg-type]

        assert draft.diff == VALID_DIFF
        assert draft.rationale == "rename"
        assert len(steps) == 1
        assert steps[0]["tool_called"] == "submit_patch"
        assert steps[0]["call_site_id"] == "cs-1"
        assert steps[0]["attempt_number"] == 1


class TestRunCoderToolLoop:
    def test_calls_expand_snippet_then_submits(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
        model = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[{"name": "expand_snippet", "args": {"extra_lines": 3}, "id": "1"}]),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "2"}],
                ),
            ]
        )

        draft, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=1, model=model)  # type: ignore[arg-type]

        assert draft.diff == VALID_DIFF
        assert len(steps) == 2
        assert steps[0]["tool_called"] == "expand_snippet"
        assert steps[0]["tool_args"] == {"extra_lines": 3}
        assert steps[1]["tool_called"] == "submit_patch"

    def test_calls_search_repo_then_submits(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(source=x)\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("stripe.Charge.create(source=y)\n", encoding="utf-8")
        model = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[{"name": "search_repo", "args": {"query": "stripe.Charge.create"}, "id": "1"}]),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "2"}],
                ),
            ]
        )

        draft, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=1, model=model)  # type: ignore[arg-type]

        assert draft.diff == VALID_DIFF
        assert steps[0]["tool_called"] == "search_repo"
        assert "other.py" in steps[0]["tool_result_summary"]

    def test_rejects_malformed_diff_and_lets_model_retry(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(source=x)\n", encoding="utf-8")
        model = FakeChatModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": "not a real diff", "rationale": "x"}, "id": "1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "2"}],
                ),
            ]
        )

        draft, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=1, model=model)  # type: ignore[arg-type]

        assert draft.diff == VALID_DIFF
        assert len(steps) == 2
        assert "rejected" in steps[0]["tool_result_summary"]

    def test_nudges_model_when_no_tool_call_returned(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(source=x)\n", encoding="utf-8")
        model = FakeChatModel(
            [
                AIMessage(content="I think I should just explain instead of calling a tool."),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "1"}],
                ),
            ]
        )

        draft, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=1, model=model)  # type: ignore[arg-type]

        assert draft.diff == VALID_DIFF
        assert steps[0]["tool_called"] is None
        assert "no tool call" in steps[0]["tool_result_summary"]

    def test_attempt_number_and_call_site_id_stamped_on_every_step(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(source=x)\n", encoding="utf-8")
        model = FakeChatModel(
            [
                AIMessage(content="", tool_calls=[{"name": "expand_snippet", "args": {"extra_lines": 2}, "id": "1"}]),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "submit_patch", "args": {"diff": VALID_DIFF, "rationale": "rename"}, "id": "2"}],
                ),
            ]
        )

        _, steps = run_coder(_drift_item(), _call_site(), str(tmp_path), attempt_number=3, model=model)  # type: ignore[arg-type]

        assert all(step["attempt_number"] == 3 for step in steps)
        assert all(step["call_site_id"] == "cs-1" for step in steps)
        assert all(step["node"] == "coder" for step in steps)


class TestExpandSnippetTool:
    def test_widens_window_by_extra_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("\n".join(f"line{i}" for i in range(1, 31)), encoding="utf-8")

        result = expand_snippet(str(tmp_path), "f.py", 10, 15, extra_lines=3)

        assert result.line_start == 7
        assert result.line_end == 18

    def test_caps_total_window_size(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("\n".join(f"line{i}" for i in range(1, 501)), encoding="utf-8")

        result = expand_snippet(str(tmp_path), "f.py", 200, 205, extra_lines=1000)

        assert (result.line_end - result.line_start + 1) <= MAX_TOTAL_WINDOW_LINES

    def test_does_not_go_below_line_1_or_past_file_end(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("\n".join(f"line{i}" for i in range(1, 6)), encoding="utf-8")

        result = expand_snippet(str(tmp_path), "f.py", 2, 3, extra_lines=100)

        assert result.line_start == 1
        assert result.line_end == 5


class TestSearchRepoTool:
    def test_finds_matches_across_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("stripe.Charge.create(x)\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("stripe.Charge.create(y)\n", encoding="utf-8")

        matches = search_repo(str(tmp_path), "stripe.Charge.create")

        assert {m.file_path for m in matches} == {"a.py", "b.py"}

    def test_excludes_test_directories(self, tmp_path: Path) -> None:
        (tmp_path / "billing.py").write_text("stripe.Charge.create(x)\n", encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_billing.py").write_text("stripe.Charge.create(x)\n", encoding="utf-8")

        matches = search_repo(str(tmp_path), "stripe.Charge.create")

        assert all("tests" not in m.file_path for m in matches)

    def test_caps_number_of_results(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"f{i}.py").write_text("needle\n", encoding="utf-8")

        matches = search_repo(str(tmp_path), "needle")

        assert len(matches) == 10
