"""Coder node: a bounded, tool-calling agent that proposes a patch for one CallSite.

AGENTS.md Section 5.4. This is the one node in the graph allowed to make
more than one LLM call per invocation (see AGENTS.md Section 6.1) - it's a
real ReAct-style loop: propose -> optionally call a read-only tool ->
observe -> propose again, until it calls `submit_patch` to end its turn.

Per AGENTS.md Section 6.3, there is currently no hard cap on the number of
tool-call turns here - that is a deliberate, documented gap for the MVP,
not an oversight. Do not add a silent step limit without updating
AGENTS.md to match; if a cap is ever added, it belongs in this file's loop
condition, not e.g. an unrelated timeout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from coder import tools as coder_tools
from llm.client import get_chat_model
from schemas import AgentStep, CallSite, DriftItem

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "coder_template.txt"
_MAX_TOOL_RESULT_CHARS = 2000  # AGENTS.md Section 6.3-style truncation, applied defensively here too


@dataclass
class CoderDraft:
    """The Coder's raw output for one attempt.

    Bookkeeping fields that turn this into a full PatchResult
    (call_site_id, attempt_number, critic_rounds) are filled in by the
    call-site subgraph that orchestrates the Coder<->Critic<->Verifier
    loop (AGENTS.md Section 4.4) - the Coder itself only knows about this
    one attempt, not the attempt's position in the larger retry history.
    """

    diff: str
    rationale: str


def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_initial_prompt(
    drift_item: DriftItem,
    call_site: CallSite,
    window: dict,
    *,
    failure_trace: str | None,
    critic_feedback: str | None,
    previous_diff: str | None,
) -> str:
    template = _load_prompt_template()
    prompt = template.format(
        change_type=drift_item["change_type"],
        api_path=drift_item["api_path"],
        field_or_param=drift_item["field_or_param"] or "(none)",
        old_value=json.dumps(drift_item["old_value"]),
        new_value=json.dumps(drift_item["new_value"]),
        file_path=call_site["file_path"],
        symbol=call_site["symbol"],
        line_start=window["line_start"],
        line_end=window["line_end"],
        snippet=window["snippet"],
    )

    if previous_diff is not None:
        prompt += (
            "\n\n### Your previous attempt ###\n"
            f"{previous_diff}\n"
        )
    if critic_feedback is not None:
        prompt += (
            "\n\n### Critic feedback on your previous attempt (address this before resubmitting) ###\n"
            f"{critic_feedback}\n"
        )
    if failure_trace is not None:
        prompt += (
            "\n\n### Your previous patch failed the test suite. Failure trace (truncated) ###\n"
            f"{failure_trace[-2000:]}\n"
        )
    return prompt


def _looks_like_unified_diff(diff: str) -> bool:
    return "--- " in diff and "+++ " in diff and "@@" in diff


def _log_step(
    call_site: CallSite,
    attempt_number: int,
    step_number: int,
    tool_called: str | None,
    tool_args: dict | None,
    tool_result_summary: str | None,
    output: dict | None,
) -> AgentStep:
    return AgentStep(
        node="coder",
        drift_item_id=None,
        call_site_id=call_site["id"],
        attempt_number=attempt_number,
        step_number=step_number,
        tool_called=tool_called,
        tool_args=tool_args,
        tool_result_summary=tool_result_summary[:_MAX_TOOL_RESULT_CHARS] if tool_result_summary else None,
        output=output,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_coder(
    drift_item: DriftItem,
    call_site: CallSite,
    target_repo: str,
    attempt_number: int,
    *,
    failure_trace: str | None = None,
    critic_feedback: str | None = None,
    previous_diff: str | None = None,
    model: BaseChatModel | None = None,
) -> tuple[CoderDraft, list[AgentStep]]:
    """Run the Coder's bounded tool-calling loop for one attempt at one
    CallSite. Returns the final CoderDraft plus every AgentStep logged
    along the way (tool calls and the final submit_patch call)."""

    chat_model = model or get_chat_model("coder")

    window = {
        "line_start": call_site["line_start"],
        "line_end": call_site["line_end"],
        "snippet": call_site["snippet"],
    }

    @tool
    def expand_snippet(extra_lines: int) -> str:
        """Widen the call site's context window by this many lines on each side (bounded)."""

        result = coder_tools.expand_snippet(
            target_repo, call_site["file_path"], window["line_start"], window["line_end"], extra_lines
        )
        window["line_start"] = result.line_start
        window["line_end"] = result.line_end
        window["snippet"] = result.snippet
        return f"Lines {result.line_start}-{result.line_end}:\n{result.snippet}"

    @tool
    def search_repo(query: str) -> str:
        """Search the rest of the repo (read-only) for other usages of a symbol or term."""

        matches = coder_tools.search_repo(target_repo, query)
        if not matches:
            return "No matches found."
        return "\n---\n".join(f"{m.file_path}:{m.line_number}\n{m.context}" for m in matches)

    @tool
    def submit_patch(diff: str, rationale: str) -> str:
        """Submit your final unified diff and one-sentence rationale. Ends your turn."""

        return "submitted"

    tool_map = {"expand_snippet": expand_snippet, "search_repo": search_repo}
    bound_model = chat_model.bind_tools([expand_snippet, search_repo, submit_patch])

    messages: list[BaseMessage] = [
        HumanMessage(
            content=_build_initial_prompt(
                drift_item,
                call_site,
                window,
                failure_trace=failure_trace,
                critic_feedback=critic_feedback,
                previous_diff=previous_diff,
            )
        )
    ]

    steps: list[AgentStep] = []
    step_number = 0

    while True:
        step_number += 1
        ai_message = bound_model.invoke(messages)
        assert isinstance(ai_message, AIMessage)
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls or []
        if not tool_calls:
            steps.append(
                _log_step(
                    call_site,
                    attempt_number,
                    step_number,
                    None,
                    None,
                    f"no tool call in response, nudging: {str(ai_message.content)[:200]}",
                    None,
                )
            )
            messages.append(
                HumanMessage(
                    content="You must call one of the provided tools - expand_snippet, search_repo, or "
                    "submit_patch - in every response."
                )
            )
            continue

        submit_call = next((tc for tc in tool_calls if tc["name"] == "submit_patch"), None)
        if submit_call is not None:
            diff = submit_call["args"].get("diff", "")
            rationale = submit_call["args"].get("rationale", "")
            if not _looks_like_unified_diff(diff):
                steps.append(
                    _log_step(
                        call_site,
                        attempt_number,
                        step_number,
                        "submit_patch",
                        submit_call["args"],
                        "rejected: does not look like a unified diff, asking to retry",
                        None,
                    )
                )
                messages.append(
                    ToolMessage(
                        content=(
                            "That doesn't look like a valid unified diff (need '--- ', '+++ ', and '@@' hunk "
                            "markers). Please call submit_patch again with a proper unified diff."
                        ),
                        tool_call_id=submit_call["id"],
                    )
                )
                continue

            steps.append(
                _log_step(
                    call_site,
                    attempt_number,
                    step_number,
                    "submit_patch",
                    submit_call["args"],
                    None,
                    {"diff": diff, "rationale": rationale},
                )
            )
            return CoderDraft(diff=diff, rationale=rationale), steps

        for tool_call in tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn is None:
                result_content = f"Unknown tool: {tool_call['name']}"
            else:
                result_content = tool_fn.invoke(tool_call["args"])
            steps.append(
                _log_step(
                    call_site,
                    attempt_number,
                    step_number,
                    tool_call["name"],
                    tool_call["args"],
                    result_content,
                    None,
                )
            )
            messages.append(ToolMessage(content=result_content, tool_call_id=tool_call["id"]))
