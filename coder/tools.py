"""Read-only, bounded tools available to the Coder agent.

AGENTS.md Section 5.4. These exist for gathering context only - none of
them can write to the target repo, execute code, or run tests. That
boundary is enforced by these simply being the only tools that exist here,
not by a runtime check (see AGENTS.md Section 2.2: "sandboxed execution
stays LLM-free, no exceptions"). The Coder may still only ever propose a
diff for the single file its CallSite lives in, regardless of what
search_repo turns up elsewhere in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_TOTAL_WINDOW_LINES = 100  # AGENTS.md Section 6.3: cap total expansion, never grow to "whole file"
MAX_SEARCH_RESULTS = 10  # AGENTS.md Section 6.3
SEARCH_CONTEXT_LINES = 1  # "a line or two of context" per match

_EXCLUDED_DIR_NAMES = {"tests", "test", "__pycache__", ".venv", "venv", ".git"}


@dataclass
class SnippetWindow:
    file_path: str
    line_start: int
    line_end: int
    snippet: str


def expand_snippet(
    target_repo: str,
    file_path: str,
    current_line_start: int,
    current_line_end: int,
    extra_lines: int,
) -> SnippetWindow:
    """Widen a call site's context window by `extra_lines` on each side,
    capped so the total window never exceeds MAX_TOTAL_WINDOW_LINES."""

    if extra_lines < 0:
        raise ValueError("extra_lines must be non-negative")

    full_path = Path(target_repo) / file_path
    source_lines = full_path.read_text(encoding="utf-8").splitlines()

    new_start = max(1, current_line_start - extra_lines)
    new_end = min(len(source_lines), current_line_end + extra_lines)

    if (new_end - new_start + 1) > MAX_TOTAL_WINDOW_LINES:
        half = MAX_TOTAL_WINDOW_LINES // 2
        midpoint = (current_line_start + current_line_end) // 2
        new_start = max(1, midpoint - half)
        new_end = min(len(source_lines), new_start + MAX_TOTAL_WINDOW_LINES - 1)

    snippet = "\n".join(source_lines[new_start - 1 : new_end])
    return SnippetWindow(file_path=file_path, line_start=new_start, line_end=new_end, snippet=snippet)


@dataclass
class SearchMatch:
    file_path: str
    line_number: int
    context: str


def search_repo(target_repo: str, query: str) -> list[SearchMatch]:
    """Bounded, read-only, case-insensitive substring search across the
    target repo's .py files. This is context for the Coder, not a path to
    multi-file edits - it may only ever propose a diff for its one file."""

    repo_path = Path(target_repo)
    matches: list[SearchMatch] = []
    query_lower = query.lower()

    for py_file in sorted(repo_path.rglob("*.py")):
        relative_parts = py_file.relative_to(repo_path).parts[:-1]
        if any(part in _EXCLUDED_DIR_NAMES or part.startswith(".") for part in relative_parts):
            continue

        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, idx - SEARCH_CONTEXT_LINES)
                end = min(len(lines), idx + SEARCH_CONTEXT_LINES + 1)
                context = "\n".join(lines[start:end])
                matches.append(
                    SearchMatch(
                        file_path=str(py_file.relative_to(repo_path)),
                        line_number=idx + 1,
                        context=context,
                    )
                )
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return matches

    return matches
