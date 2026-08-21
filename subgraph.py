"""Call-site subgraph: wires Coder <-> Critic <-> Verifier for one CallSite.

AGENTS.md Section 2.1's "call-site subgraph" and Section 4.8's rule that
state is processed one CallSite at a time. This is the genuinely cyclic,
agentic part of the pipeline - the outer graph (graph.py) invokes
`process_call_site()` once per call site, sequentially, never batched or
fanned out.

Implementation note on where the "agentic loop" actually lives: the
Coder's own bounded tool-calling loop (propose -> call a tool -> observe ->
propose again) is implemented inside `coder.agent.run_coder()` as plain
Python - by the time this subgraph sees a "coder" node run, that inner
loop has already completed and produced a single diff+rationale. What this
subgraph adds on top is the *outer* cycle described in AGENTS.md Section
2.1's second diagram: Coder -> Critic -> (approve -> Verifier | reject ->
Coder again) -> (pass -> done | fail with retries left -> Coder again |
fail with retries exhausted -> escalate). That outer cycle is real
LangGraph conditional routing, which is what makes LangGraph load-bearing
here per Section 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from coder.agent import run_coder
from critic.agent import run_critic
from schemas import AgentStep, CallSite, CriticVerdict, DriftItem, PatchResult, TestResult
from verifier.sandbox import run_verifier

MAX_RETRIES = 3  # AGENTS.md Section 5.4: at retry_counts[call_site_id] >= 3, escalate

# LangGraph enforces some recursion limit; AGENTS.md Section 6.3 deliberately
# leaves the Coder<->Critic revision loop uncapped for the MVP. This is set
# very high (not infinite - Python/LangGraph can't truly do that) to honor
# that decision in practice rather than silently reintroducing a low cap.
_RECURSION_LIMIT = 10_000


class CallSiteSubgraphState(TypedDict):
    drift_item: DriftItem
    call_site: CallSite
    target_repo: str
    attempt_number: int
    critic_rounds: int
    current_diff: str | None
    current_rationale: str | None
    failure_trace: str | None
    critic_feedback: str | None
    critic_verdicts: list[CriticVerdict]
    agent_steps: list[AgentStep]
    retry_count: int
    outcome: Literal["pending", "success", "escalated"]
    final_patch_result: PatchResult | None
    final_test_result: TestResult | None


def coder_node(state: CallSiteSubgraphState) -> CallSiteSubgraphState:
    # previous_diff is only meaningful as "what the Critic just rejected" -
    # it must not leak across attempt boundaries (a fresh attempt after a
    # Verifier failure starts from a clean slate, informed by failure_trace
    # instead).
    previous_diff = state["current_diff"] if state["critic_feedback"] else None

    draft, steps = run_coder(
        state["drift_item"],
        state["call_site"],
        state["target_repo"],
        state["attempt_number"],
        failure_trace=state["failure_trace"],
        critic_feedback=state["critic_feedback"],
        previous_diff=previous_diff,
    )

    return {
        **state,
        "current_diff": draft.diff,
        "current_rationale": draft.rationale,
        "agent_steps": [*state["agent_steps"], *steps],
        "critic_feedback": None,  # consumed for this round
    }


def critic_node(state: CallSiteSubgraphState) -> CallSiteSubgraphState:
    assert state["current_diff"] is not None

    verdict, steps = run_critic(
        state["drift_item"],
        state["call_site"],
        state["current_diff"],
        attempt_number=state["attempt_number"],
        revision_round=state["critic_rounds"] + 1,
    )

    return {
        **state,
        "critic_rounds": state["critic_rounds"] + 1,
        "critic_verdicts": [*state["critic_verdicts"], verdict],
        "agent_steps": [*state["agent_steps"], *steps],
        "critic_feedback": None if verdict["approved"] else verdict["feedback"],
    }


def verifier_node(state: CallSiteSubgraphState) -> CallSiteSubgraphState:
    assert state["current_diff"] is not None and state["current_rationale"] is not None

    patch_result = PatchResult(
        call_site_id=state["call_site"]["id"],
        diff=state["current_diff"],
        rationale=state["current_rationale"],
        attempt_number=state["attempt_number"],
        critic_rounds=state["critic_rounds"],
    )
    test_result = run_verifier(patch_result, state["target_repo"])

    if test_result["passed"]:
        return {
            **state,
            "outcome": "success",
            "final_patch_result": patch_result,
            "final_test_result": test_result,
        }

    new_retry_count = state["retry_count"] + 1
    if new_retry_count >= MAX_RETRIES:
        return {
            **state,
            "outcome": "escalated",
            "retry_count": new_retry_count,
            "final_patch_result": patch_result,
            "final_test_result": test_result,
        }

    return {
        **state,
        "retry_count": new_retry_count,
        "attempt_number": state["attempt_number"] + 1,
        "critic_rounds": 0,
        "failure_trace": test_result["failure_trace"],
        "critic_feedback": None,
        "final_test_result": test_result,
    }


def _route_after_critic(state: CallSiteSubgraphState) -> str:
    return "revise" if state["critic_feedback"] else "verify"


def _route_after_verifier(state: CallSiteSubgraphState) -> str:
    return "retry" if state["outcome"] == "pending" else "done"


def build_call_site_subgraph():
    """Build (uncompiled) the Coder<->Critic<->Verifier subgraph. Exposed
    mainly so tests/graph.py can compile it with different configs."""

    graph = StateGraph(CallSiteSubgraphState)
    graph.add_node("coder", coder_node)
    graph.add_node("critic", critic_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("coder")
    graph.add_edge("coder", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {"revise": "coder", "verify": "verifier"})
    graph.add_conditional_edges("verifier", _route_after_verifier, {"retry": "coder", "done": END})

    return graph


@dataclass
class CallSiteOutcome:
    """What graph.py needs to fold this call site's result back into the
    top-level HealingState (AGENTS.md Section 4.8)."""

    outcome: Literal["success", "escalated"]
    patch_result: PatchResult
    test_result: TestResult
    critic_verdicts: list[CriticVerdict]
    agent_steps: list[AgentStep]
    retry_count: int


def process_call_site(drift_item: DriftItem, call_site: CallSite, target_repo: str) -> CallSiteOutcome:
    """Run the full Coder<->Critic<->Verifier cycle for exactly one
    CallSite, to completion (success or escalation)."""

    compiled = build_call_site_subgraph().compile()

    initial_state: CallSiteSubgraphState = {
        "drift_item": drift_item,
        "call_site": call_site,
        "target_repo": target_repo,
        "attempt_number": 1,
        "critic_rounds": 0,
        "current_diff": None,
        "current_rationale": None,
        "failure_trace": None,
        "critic_feedback": None,
        "critic_verdicts": [],
        "agent_steps": [],
        "retry_count": 0,
        "outcome": "pending",
        "final_patch_result": None,
        "final_test_result": None,
    }

    final_state = compiled.invoke(initial_state, config={"recursion_limit": _RECURSION_LIMIT})

    assert final_state["outcome"] in ("success", "escalated")
    assert final_state["final_patch_result"] is not None
    assert final_state["final_test_result"] is not None

    return CallSiteOutcome(
        outcome=final_state["outcome"],
        patch_result=final_state["final_patch_result"],
        test_result=final_state["final_test_result"],
        critic_verdicts=final_state["critic_verdicts"],
        agent_steps=final_state["agent_steps"],
        retry_count=final_state["retry_count"],
    )
