"""Unit tests for locator/ast_scanner.py (AGENTS.md Section 5.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from locator.ast_scanner import MAX_CONTEXT_LINES, locate_call_sites
from schemas import SearchPlan

REPO_ROOT = Path(__file__).resolve().parent.parent


def _plan(drift_item_id: str, symbols: list[str]) -> SearchPlan:
    return SearchPlan(drift_item_id=drift_item_id, symbols=symbols, rationale="test")


class TestLocateCallSitesAgainstDemoRepo:
    def test_finds_all_three_hand_crafted_call_sites(self) -> None:
        search_plans = {
            "d1": _plan("d1", ["stripe.Charge.create"]),
            "d2": _plan("d2", ["stripe.Refund.create"]),
            "d3": _plan("d3", ["stripe.Customer.create_source"]),
        }

        call_sites = locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"))

        symbols_found = {cs["symbol"] for cs in call_sites}
        assert symbols_found == {"stripe.Charge.create", "stripe.Refund.create", "stripe.Customer.create_source"}
        assert len(call_sites) == 3

    def test_call_sites_link_back_to_correct_drift_item(self) -> None:
        search_plans = {
            "d1": _plan("drift-charges", ["stripe.Charge.create"]),
            "d2": _plan("drift-refunds", ["stripe.Refund.create"]),
        }

        call_sites = locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"))

        by_symbol = {cs["symbol"]: cs for cs in call_sites}
        assert by_symbol["stripe.Charge.create"]["drift_item_id"] == "drift-charges"
        assert by_symbol["stripe.Refund.create"]["drift_item_id"] == "drift-refunds"

    def test_snippet_contains_the_call_expression(self) -> None:
        search_plans = {"d1": _plan("d1", ["stripe.Charge.create"])}

        call_sites = locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"))

        assert len(call_sites) == 1
        assert "stripe.Charge.create(" in call_sites[0]["snippet"]
        assert call_sites[0]["file_path"] == "billing.py"

    def test_never_matches_unrelated_symbols(self) -> None:
        search_plans = {"d1": _plan("d1", ["stripe.PaymentIntent.create"])}

        call_sites = locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"))

        assert call_sites == []

    def test_excludes_test_directories_from_scan(self) -> None:
        # demo_repo/tests/test_billing.py references "stripe.Charge.create"
        # only as a @patch(...) string argument, never as a real call - but
        # this also verifies the tests/ directory itself is skipped
        # entirely, so it can never become a source of call sites to patch.
        search_plans = {"d1": _plan("d1", ["stripe.Charge.create"])}

        call_sites = locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"))

        assert all("tests" not in cs["file_path"] for cs in call_sites)

    def test_empty_search_plans_yields_no_call_sites(self) -> None:
        assert locate_call_sites({}, str(REPO_ROOT / "demo_repo")) == []

    def test_rejects_context_lines_beyond_max(self) -> None:
        search_plans = {"d1": _plan("d1", ["stripe.Charge.create"])}
        with pytest.raises(ValueError):
            locate_call_sites(search_plans, str(REPO_ROOT / "demo_repo"), context_lines=MAX_CONTEXT_LINES + 1)


class TestLocateCallSitesSkipsUnparsableFiles:
    def test_syntax_error_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
        (tmp_path / "good.py").write_text("import stripe\nstripe.Charge.create(amount=1)\n", encoding="utf-8")

        search_plans = {"d1": _plan("d1", ["stripe.Charge.create"])}
        call_sites = locate_call_sites(search_plans, str(tmp_path))

        assert len(call_sites) == 1
        assert call_sites[0]["file_path"] == "good.py"
