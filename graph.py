"""Top-level LangGraph wiring for the self-healing pipeline.

AGENTS.md Section 2.1 and Section 9 (CLI commands). Wires Watcher ->
Planner -> Locator -> call-site processing -> PR/Escalation output into a
single `StateGraph(HealingState)`, and exposes the CLI entrypoint.

Design notes on where two of AGENTS.md's diagram boxes actually live in
code, so this isn't read as a silent deviation:

- "One CallSite at a time" (Section 4.8) is implemented as a single graph
  node (`call_sites_node`) with a plain Python `for` loop inside it that
  invokes the compiled call-site subgraph (`subgraph.process_call_site`)
  once per CallSite, sequentially, to full completion (success or
  escalation) before moving to the next. This satisfies "not batched or
  fanned out" without needing LangGraph-level looping for a loop whose
  trip count is just "however many call sites the Locator found" - a
  deterministic count, not a conditional/agentic decision. The genuinely
  cyclic, conditional part of the pipeline is inside the call-site
  subgraph itself (Coder<->Critic<->Verifier) - see subgraph.py.
- The PR Node and Escalation Node are invoked from one `output_node`
  rather than as two separately-routed graph nodes, since a single run can
  need both (some call sites patched, others escalated) and calling them
  from one node avoids LangGraph parallel-branch state-merge complexity
  for what is, in the end, just two independent side-effecting API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from devsetup import ensure_git_repo
from escalation.github_issues import open_escalation_issue
from locator.ast_scanner import locate_call_sites
from planner.agent import run_planner
from pr.common import EscalatedItem, SuccessfulPatch
from pr.github_client import open_or_update_pr
from schemas import HealingState, new_healing_state
from subgraph import process_call_site
from watcher.diff_engine import commit_snapshot, detect_drift

logger = logging.getLogger(__name__)


def _log_node_boundary(node_name: str, *, duration_seconds: float, extra: dict | None = None) -> None:
    """Structured JSON-lines logging at each node boundary (AGENTS.md
    Section 6.4): node name, duration, and small pass/fail-style counters
    only - no secrets, tokens, or full LLM prompts at this default log
    level."""

    record = {
        "node": node_name,
        "duration_seconds": round(duration_seconds, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    logger.info(json.dumps(record))


def watcher_node(state: HealingState, *, simulate_drift: bool) -> dict:
    start = time.monotonic()
    result = detect_drift(state["api_provider"], simulate_drift=simulate_drift)
    _log_node_boundary("watcher", duration_seconds=time.monotonic() - start, extra={"drift_items": len(result.drift_items)})
    return {
        "drift_report": result.drift_items,
        "spec_version_old": result.old_spec,
        "spec_version_new": result.new_spec,
    }


def planner_node(state: HealingState) -> dict:
    start = time.monotonic()
    search_plans = dict(state["search_plans"])
    agent_traces = dict(state["agent_traces"])

    for drift_item in state["drift_report"]:
        plan, steps = run_planner(drift_item)
        search_plans[drift_item["id"]] = plan
        agent_traces[f"planner:{drift_item['id']}"] = steps

    _log_node_boundary("planner", duration_seconds=time.monotonic() - start, extra={"search_plans": len(search_plans)})
    return {"search_plans": search_plans, "agent_traces": agent_traces}


def locator_node(state: HealingState) -> dict:
    start = time.monotonic()
    call_sites = locate_call_sites(state["search_plans"], state["target_repo"])
    _log_node_boundary("locator", duration_seconds=time.monotonic() - start, extra={"call_sites": len(call_sites)})
    return {"affected_call_sites": call_sites}


def call_sites_node(state: HealingState) -> dict:
    start = time.monotonic()
    patch_results = dict(state["patch_results"])
    critic_verdicts = dict(state["critic_verdicts"])
    agent_traces = dict(state["agent_traces"])
    test_results = dict(state["test_results"])
    retry_counts = dict(state["retry_counts"])
    escalated_call_site_ids = list(state["escalated_call_site_ids"])

    drift_items_by_id = {item["id"]: item for item in state["drift_report"]}

    for call_site in state["affected_call_sites"]:
        drift_item = drift_items_by_id[call_site["drift_item_id"]]
        outcome = process_call_site(drift_item, call_site, state["target_repo"])

        patch_results[call_site["id"]] = outcome.patch_result
        critic_verdicts[call_site["id"]] = outcome.critic_verdicts
        agent_traces[call_site["id"]] = outcome.agent_steps
        test_results[call_site["id"]] = outcome.test_result
        retry_counts[call_site["id"]] = outcome.retry_count

        if outcome.outcome == "escalated":
            escalated_call_site_ids.append(call_site["id"])

    _log_node_boundary(
        "call_sites",
        duration_seconds=time.monotonic() - start,
        extra={"processed": len(state["affected_call_sites"]), "escalated": len(escalated_call_site_ids)},
    )
    return {
        "patch_results": patch_results,
        "critic_verdicts": critic_verdicts,
        "agent_traces": agent_traces,
        "test_results": test_results,
        "retry_counts": retry_counts,
        "escalated_call_site_ids": escalated_call_site_ids,
    }


def _successful_patches_and_escalated_items(state: HealingState) -> tuple[list[SuccessfulPatch], list[EscalatedItem]]:
    call_sites_by_id = {cs["id"]: cs for cs in state["affected_call_sites"]}
    drift_items_by_id = {item["id"]: item for item in state["drift_report"]}
    escalated = set(state["escalated_call_site_ids"])

    successful_patches = [
        SuccessfulPatch(
            drift_item=drift_items_by_id[call_sites_by_id[cs_id]["drift_item_id"]],
            call_site=call_sites_by_id[cs_id],
            patch_result=patch_result,
            test_result=state["test_results"][cs_id],
        )
        for cs_id, patch_result in state["patch_results"].items()
        if cs_id not in escalated
    ]
    escalated_items = [
        EscalatedItem(
            drift_item=drift_items_by_id[call_sites_by_id[cs_id]["drift_item_id"]],
            call_site=call_sites_by_id[cs_id],
            test_result=state["test_results"][cs_id],
            retry_count=state["retry_counts"][cs_id],
        )
        for cs_id in state["escalated_call_site_ids"]
    ]
    return successful_patches, escalated_items


def output_node(state: HealingState, *, github_repo: str | None, github_client: object | None) -> dict:
    start = time.monotonic()
    successful_patches, escalated_items = _successful_patches_and_escalated_items(state)

    update: dict = {}

    if successful_patches:
        pr_result = open_or_update_pr(
            api_provider=state["api_provider"],
            target_repo=state["target_repo"],
            github_repo=github_repo or "",
            successful_patches=successful_patches,
            escalated_items=escalated_items,
            dry_run=state["dry_run"],
            github_client=github_client,
        )
        update["pr_url"] = pr_result.pr_url

    if escalated_items:
        open_escalation_issue(
            api_provider=state["api_provider"],
            github_repo=github_repo or "",
            escalated_items=escalated_items,
            dry_run=state["dry_run"],
            github_client=github_client,
        )

    _log_node_boundary(
        "output",
        duration_seconds=time.monotonic() - start,
        extra={"successful_patches": len(successful_patches), "escalated_items": len(escalated_items)},
    )
    return update


def _route_after_watcher(state: HealingState) -> str:
    # AGENTS.md Section 5.1, bullet 4: no drift is the expected common
    # case and terminates cleanly, not an error.
    return "planner" if state["drift_report"] else "end"


def _route_after_locator(state: HealingState) -> str:
    return "call_sites" if state["affected_call_sites"] else "end"


def _route_after_call_sites(state: HealingState) -> str:
    return "output" if (state["patch_results"] or state["escalated_call_site_ids"]) else "end"


def build_graph(
    *,
    simulate_drift: bool,
    github_repo: str | None = None,
    github_client: object | None = None,
) -> StateGraph:
    graph = StateGraph(HealingState)

    graph.add_node("watcher", partial(watcher_node, simulate_drift=simulate_drift))
    graph.add_node("planner", planner_node)
    graph.add_node("locator", locator_node)
    graph.add_node("call_sites", call_sites_node)
    graph.add_node("output", partial(output_node, github_repo=github_repo, github_client=github_client))

    graph.set_entry_point("watcher")
    graph.add_conditional_edges("watcher", _route_after_watcher, {"planner": "planner", "end": END})
    graph.add_edge("planner", "locator")
    graph.add_conditional_edges("locator", _route_after_locator, {"call_sites": "call_sites", "end": END})
    graph.add_conditional_edges("call_sites", _route_after_call_sites, {"output": "output", "end": END})
    graph.add_edge("output", END)

    return graph


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the self-healing API integration pipeline (AGENTS.md Section 9).")
    parser.add_argument("--provider", default="stripe")
    parser.add_argument("--target-repo", default="demo_repo")
    parser.add_argument("--simulate-drift", action="store_true", help="Diff fixture specs instead of a live fetch.")
    parser.add_argument("--dry-run", action="store_true", help="Never call the GitHub API; print what would happen.")
    parser.add_argument("--github-repo", default=os.environ.get("GITHUB_TARGET_REPO"), help="owner/repo, required unless --dry-run.")
    args = parser.parse_args()

    # The Verifier's git-worktree sandboxing (AGENTS.md Section 7) requires
    # target_repo to be its own git repository; see devsetup.py for why
    # this is a runtime-created, not committed, precondition.
    ensure_git_repo(args.target_repo)

    github_client = None
    if not args.dry_run:
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            print("GITHUB_TOKEN is required unless --dry-run is set.", file=sys.stderr)
            sys.exit(1)
        if not args.github_repo:
            print("--github-repo (or GITHUB_TARGET_REPO) is required unless --dry-run is set.", file=sys.stderr)
            sys.exit(1)
        from github import Github

        github_client = Github(github_token)

    compiled = build_graph(
        simulate_drift=args.simulate_drift, github_repo=args.github_repo, github_client=github_client
    ).compile()

    initial_state = new_healing_state(
        api_provider=args.provider,
        spec_version_old={},
        spec_version_new={},
        target_repo=args.target_repo,
        dry_run=args.dry_run,
    )

    final_state = compiled.invoke(initial_state, config={"recursion_limit": 1000})

    print(
        json.dumps(
            {
                "drift_items": len(final_state["drift_report"]),
                "call_sites": len(final_state["affected_call_sites"]),
                "patched": len([cid for cid in final_state["patch_results"] if cid not in final_state["escalated_call_site_ids"]]),
                "escalated": len(final_state["escalated_call_site_ids"]),
                "pr_url": final_state["pr_url"],
            },
            indent=2,
        )
    )

    # AGENTS.md Section 5.1, bullet 5: only commit the new snapshot after a
    # successful, non-dry-run, non-simulated run - never immediately on
    # fetch, and never in --simulate-drift mode (the fixtures are the
    # fixed source of truth for demos; see watcher/diff_engine.py).
    if not args.dry_run and not args.simulate_drift and final_state["drift_report"]:
        commit_snapshot(args.provider, final_state["spec_version_new"])


if __name__ == "__main__":
    _main()
