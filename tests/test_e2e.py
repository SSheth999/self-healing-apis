"""Mandatory end-to-end test (AGENTS.md Section 8): runs the full graph
against the real fixtures and the real demo_repo, with only the LLM calls
(Planner/Coder/Critic) mocked - deterministic, no real API key needed in
CI. Locator, Verifier, and the PR/Escalation dry-run logic all run for
real, so this proves the actual wiring works end to end, not just each
node in isolation.

Scenario: of the three hand-crafted drift items, two get a correct fix
(field_renamed on /v1/charges, endpoint_moved on the Sources API) and one
(field_required_changed on /v1/refunds) never gets a working fix from the
scripted Coder, so it must escalate after exhausting its retry budget.
This single run exercises both the PR path and the Escalation path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import graph
from coder.agent import CoderDraft
from schemas import CriticVerdict, SearchPlan, new_healing_state

REPO_ROOT = Path(__file__).resolve().parent.parent

SYMBOLS_BY_API_PATH = {
    "/v1/charges": ["stripe.Charge.create"],
    "/v1/refunds": ["stripe.Refund.create"],
    "/v1/customers/{customer}/sources": ["stripe.Customer.create_source"],
}

CHARGE_FIX_DIFF = (
    "--- a/billing.py\n"
    "+++ b/billing.py\n"
    "@@ -30,5 +30,5 @@\n"
    "     return stripe.Charge.create(\n"
    "         amount=amount,\n"
    "         currency=currency,\n"
    "-        source=source_token,\n"
    "+        payment_method=source_token,\n"
    "         customer=customer_id,\n"
)

REFUND_NOOP_DIFF = (
    "--- a/billing.py\n"
    "+++ b/billing.py\n"
    "@@ -44,3 +44,3 @@\n"
    " \n"
    "-    return stripe.Refund.create(charge=charge_id, amount=amount)\n"
    "+    return stripe.Refund.create(charge=charge_id, amount=amount)  # no-op\n"
    " \n"
)

ATTACH_FIX_DIFF = (
    "--- a/billing.py\n"
    "+++ b/billing.py\n"
    "@@ -55,2 +55,2 @@\n"
    " \n"
    "-    return stripe.Customer.create_source(customer_id, source=source_token)\n"
    "+    return stripe.PaymentMethod.attach(source_token, customer=customer_id)\n"
)

DIFFS_BY_SYMBOL = {
    "stripe.Charge.create": CHARGE_FIX_DIFF,
    "stripe.Refund.create": REFUND_NOOP_DIFF,  # deliberately never fixes the drift -> escalates
    "stripe.Customer.create_source": ATTACH_FIX_DIFF,
}


def _fake_run_planner(drift_item):
    symbols = SYMBOLS_BY_API_PATH[drift_item["api_path"]]
    plan = SearchPlan(drift_item_id=drift_item["id"], symbols=symbols, rationale="e2e test stub")
    return plan, []


def _fake_run_coder(drift_item, call_site, target_repo, attempt_number, **kwargs):
    diff = DIFFS_BY_SYMBOL[call_site["symbol"]]
    return CoderDraft(diff=diff, rationale=f"fix for {call_site['symbol']}"), []


def _fake_run_critic(drift_item, call_site, diff, *, attempt_number, revision_round):
    verdict = CriticVerdict(
        call_site_id=call_site["id"],
        attempt_number=attempt_number,
        revision_round=revision_round,
        approved=True,
        feedback="",
    )
    return verdict, []


@pytest.fixture(autouse=True)
def _mock_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("graph.run_planner", _fake_run_planner)
    monkeypatch.setattr("subgraph.run_coder", _fake_run_coder)
    monkeypatch.setattr("subgraph.run_critic", _fake_run_critic)


class TestEndToEndDryRun:
    def test_full_pipeline_against_fixtures_and_real_demo_repo(self) -> None:
        real_open_or_update_pr = graph.open_or_update_pr
        real_open_escalation_issue = graph.open_escalation_issue
        captured: dict = {}

        def _spy_pr(**kwargs):
            result = real_open_or_update_pr(**kwargs)
            captured["pr_result"] = result
            return result

        def _spy_escalation(**kwargs):
            result = real_open_escalation_issue(**kwargs)
            captured["escalation_result"] = result
            return result

        import pytest as _pytest  # local import to use monkeypatch context manager style

        mp = _pytest.MonkeyPatch()
        mp.setattr("graph.open_or_update_pr", _spy_pr)
        mp.setattr("graph.open_escalation_issue", _spy_escalation)

        try:
            compiled = graph.build_graph(simulate_drift=True, github_repo=None, github_client=None).compile()
            initial_state = new_healing_state(
                api_provider="stripe",
                spec_version_old={},
                spec_version_new={},
                target_repo=str(REPO_ROOT / "demo_repo"),
                dry_run=True,
            )

            final_state = compiled.invoke(initial_state, config={"recursion_limit": 1000})
        finally:
            mp.undo()

        # --- Watcher / Planner / Locator wiring ---
        assert len(final_state["drift_report"]) == 3
        assert len(final_state["search_plans"]) == 3
        assert len(final_state["affected_call_sites"]) == 3

        # --- Call-site outcomes: 2 successes, 1 escalation ---
        assert len(final_state["patch_results"]) == 3
        assert len(final_state["escalated_call_site_ids"]) == 1

        escalated_call_site_id = final_state["escalated_call_site_ids"][0]
        escalated_call_site = next(
            cs for cs in final_state["affected_call_sites"] if cs["id"] == escalated_call_site_id
        )
        assert escalated_call_site["symbol"] == "stripe.Refund.create"
        assert final_state["retry_counts"][escalated_call_site_id] == 3

        successful_symbols = {
            cs["symbol"]
            for cs in final_state["affected_call_sites"]
            if cs["id"] not in final_state["escalated_call_site_ids"]
        }
        assert successful_symbols == {"stripe.Charge.create", "stripe.Customer.create_source"}

        # --- Dry-run: no real GitHub calls, no pr_url ---
        assert final_state["pr_url"] is None

        # --- PR content reached the right shape ---
        pr_result = captured["pr_result"]
        assert pr_result.dry_run is True
        assert "billing.py" in pr_result.body
        assert "renamed field" in pr_result.body  # field_renamed drift description
        assert "moved/renamed endpoint" in pr_result.body  # endpoint_moved drift description
        assert "Not automatically fixed" in pr_result.body  # mentions the escalation

        # --- Escalation content reached the right shape ---
        escalation_result = captured["escalation_result"]
        assert escalation_result.dry_run is True
        assert "billing.py" in escalation_result.body
        assert "field became required" in escalation_result.body  # field_required_changed drift description
        assert "1 call site(s)" in escalation_result.title

    def test_real_demo_repo_is_left_untouched(self) -> None:
        billing_path = REPO_ROOT / "demo_repo" / "billing.py"
        original_content = billing_path.read_text(encoding="utf-8")

        compiled = graph.build_graph(simulate_drift=True, github_repo=None, github_client=None).compile()
        initial_state = new_healing_state(
            api_provider="stripe",
            spec_version_old={},
            spec_version_new={},
            target_repo=str(REPO_ROOT / "demo_repo"),
            dry_run=True,
        )
        compiled.invoke(initial_state, config={"recursion_limit": 1000})

        assert billing_path.read_text(encoding="utf-8") == original_content

        import subprocess

        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT / "demo_repo", capture_output=True, text=True, check=True
        )
        assert status.stdout.strip() == ""
