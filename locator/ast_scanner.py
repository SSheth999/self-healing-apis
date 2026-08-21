"""Locator node: AST-based scan for call sites matching a SearchPlan.

AGENTS.md Section 5.3. Pure code, no LLM. Executes the Planner's
`SearchPlan.symbols` list as-is - this module never decides *what* to
search for, only *how* to find it. Never falls back to a blind full-repo
text grep as a substitute for AST parsing (Section 5.3's scoping rule):
false positives (e.g. a comment mentioning a field name) would waste
Coder/Critic calls downstream and pollute the eventual PR.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path

from schemas import CallSite, SearchPlan

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LINES = 10
MAX_CONTEXT_LINES = 50  # AGENTS.md Section 6.3: cap total expanded window, don't grow toward "whole file"

_EXCLUDED_DIR_NAMES = {"tests", "test", "__pycache__", ".venv", "venv", ".git"}


def _dotted_name(node: ast.expr) -> str | None:
    """Resolve an Attribute/Name chain to a dotted string.

    e.g. `stripe.Charge.create` parses to
    Attribute(value=Attribute(value=Name(id='stripe'), attr='Charge'), attr='create')
    and resolves to "stripe.Charge.create". Returns None for anything else
    (subscripts, calls-as-callee, etc.) - those aren't symbols a SearchPlan
    can name anyway.

    Known MVP limitation: this does not resolve import aliases (e.g.
    `import stripe as s; s.Charge.create(...)` would not match
    "stripe.Charge.create"). Not needed for the current fixtures; flag if
    the target repo ever aliases the SDK import.
    """

    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, symbols: set[str]) -> None:
        self._symbols = symbols
        self.matches: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _dotted_name(node.func) in self._symbols:
            self.matches.append(node)
        self.generic_visit(node)


def _find_python_files(target_repo: Path) -> list[Path]:
    files: list[Path] = []
    for path in target_repo.rglob("*.py"):
        relative_parts = path.relative_to(target_repo).parts[:-1]  # exclude the filename itself
        if any(part in _EXCLUDED_DIR_NAMES or part.startswith(".") for part in relative_parts):
            continue
        files.append(path)
    return sorted(files)


def _extract_snippet(source_lines: list[str], call: ast.Call, context_lines: int) -> tuple[int, int, str]:
    call_start = call.lineno
    call_end = call.end_lineno or call.lineno
    line_start = max(1, call_start - context_lines)
    line_end = min(len(source_lines), call_end + context_lines)
    snippet = "\n".join(source_lines[line_start - 1 : line_end])
    return line_start, line_end, snippet


def _make_call_site_id(file_path: str, line_start: int) -> str:
    stable_key = f"{file_path}:{line_start}"
    return hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]


def locate_call_sites(
    search_plans: dict[str, SearchPlan],
    target_repo: str,
    *,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> list[CallSite]:
    """AST-scan every .py file in `target_repo` (excluding tests/venv/etc.)
    for calls matching any symbol named across all `search_plans`.

    Files that fail to parse are logged and skipped rather than crashing
    the whole scan or being silently dropped without a trace (AGENTS.md
    Section 6.2).
    """

    if context_lines > MAX_CONTEXT_LINES:
        raise ValueError(f"context_lines={context_lines} exceeds MAX_CONTEXT_LINES={MAX_CONTEXT_LINES}")

    repo_path = Path(target_repo)
    call_sites: list[CallSite] = []

    symbol_to_drift_item: dict[str, str] = {}
    for plan in search_plans.values():
        for symbol in plan["symbols"]:
            symbol_to_drift_item[symbol] = plan["drift_item_id"]

    if not symbol_to_drift_item:
        return call_sites

    all_symbols = set(symbol_to_drift_item)

    for py_file in _find_python_files(repo_path):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unparsable file %s: %s", py_file, exc)
            continue

        visitor = _CallVisitor(all_symbols)
        visitor.visit(tree)
        if not visitor.matches:
            continue

        source_lines = source.splitlines()
        relative_path = str(py_file.relative_to(repo_path))
        for call in visitor.matches:
            symbol = _dotted_name(call.func)
            assert symbol is not None  # guaranteed by _CallVisitor's own filter
            line_start, line_end, snippet = _extract_snippet(source_lines, call, context_lines)
            call_sites.append(
                CallSite(
                    id=_make_call_site_id(relative_path, line_start),
                    drift_item_id=symbol_to_drift_item[symbol],
                    file_path=relative_path,
                    line_start=line_start,
                    line_end=line_end,
                    snippet=snippet,
                    symbol=symbol,
                )
            )

    return call_sites
