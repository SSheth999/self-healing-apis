"""Verifier node: applies a patch in an isolated sandbox and runs the
target repo's test suite.

AGENTS.md Section 5.6 (Verifier) and Section 7 (sandboxing options). Pure
code, no LLM - the Verifier is the only place in the whole pipeline that
actually executes target-repo code, and it always does so in a sandbox
(AGENTS.md Section 2.2: "sandboxed execution stays LLM-free, no
exceptions"). Never modifies, skips, or loosens the target repo's tests to
force a pass (AGENTS.md Section 3, rule 6).

Sandboxing approach: fresh temp directory + `git worktree add` + subprocess
`pytest`, per Section 7's "Simplest" option - see README.md in this
directory for the explicit rationale for that choice.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from schemas import PatchResult, TestResult

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_FAILURE_TRACE_CHARS = 2000  # AGENTS.md Section 6.3


class VerifierError(RuntimeError):
    """Raised for verifier-internal infrastructure failures (e.g. git
    worktree setup itself failing) - distinct from a normal patch/test
    failure, which is reported as TestResult(passed=False, ...) rather
    than raised, so the retry loop can act on it normally."""


class _PatchApplyError(Exception):
    """Internal: the diff didn't apply. Caught in run_verifier and turned
    into a failed TestResult, not raised further - a bad diff is exactly
    the kind of thing the retry loop exists to handle."""


class _VerifierTimeout(Exception):
    """Internal: a single pytest invocation exceeded timeout_seconds."""


def _run(cmd: list[str], cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _apply_patch(worktree_dir: Path, diff: str) -> None:
    """Try `git apply` at both common strip levels (diffs with an a/ b/
    prefix need -p1; bare-filename diffs need -p0) before giving up."""

    last_stderr = ""
    for strip_level in ("1", "0"):
        result = subprocess.run(
            ["git", "apply", f"-p{strip_level}", "--whitespace=nowarn", "-"],
            cwd=worktree_dir,
            input=diff,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        last_stderr = result.stderr
    raise _PatchApplyError(last_stderr)


def _parse_failing_tests(pytest_output: str) -> list[str]:
    """Extract failing test node ids from pytest's summary output lines
    (e.g. "FAILED tests/test_billing.py::TestFoo::test_bar")."""

    return [match.group(1) for line in pytest_output.splitlines() if (match := re.match(r"^FAILED (\S+)", line))]


def _run_pytest(worktree_dir: Path, timeout_seconds: float) -> tuple[set[str], str]:
    """Run pytest once and return (failing_test_ids, combined_output).
    Raises _VerifierTimeout if it exceeds timeout_seconds."""

    try:
        process = _run(["python3", "-m", "pytest", "tests/", "-v"], worktree_dir, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise _VerifierTimeout from exc

    combined_output = process.stdout + "\n" + process.stderr
    return set(_parse_failing_tests(combined_output)), combined_output


def run_verifier(
    patch_result: PatchResult,
    target_repo: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TestResult:
    """Apply patch_result["diff"] in an isolated git-worktree copy of
    target_repo and run its test suite, judging success by a
    FAIL_TO_PASS/PASS_TO_PASS-style comparison (AGENTS.md Section 8)
    rather than "does the whole suite pass."

    Why: call sites are processed one at a time (AGENTS.md Section 4.8),
    but they all live in the same target repo. While call site B is still
    unpatched, its tests are still failing - that must not make a
    correct, isolated patch for call site A look like a failure. So this
    runs pytest twice against the worktree: once before applying the
    diff (baseline) and once after. A patch is considered passed only if
    (a) no test that was passing at baseline now fails (no regressions -
    the PASS_TO_PASS half) and (b) at least one previously-failing test
    now passes (real progress - the FAIL_TO_PASS half). A no-op or
    ineffective diff has an empty "fixed" set and is correctly reported
    as passed=False even though it introduced no new failures either.

    Sandbox cleanup (worktree removal) happens unconditionally, even on
    exception, via try/finally - AGENTS.md Section 5.6's "must not leave
    sandbox artifacts on disk after the run" rule.

    Raises VerifierError only for infrastructure problems (e.g. target_repo
    isn't a git repo). A patch that fails to apply, or tests that fail, are
    both reported as TestResult(passed=False, ...) - never raised - so the
    Coder<->Critic<->Verifier retry loop can act on them normally.
    """

    repo_path = Path(target_repo).resolve()
    start_time = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="verifier-sandbox-") as tmp_dir_str:
        worktree_dir = Path(tmp_dir_str) / "worktree"

        add_result = _run(["git", "worktree", "add", "--detach", str(worktree_dir), "HEAD"], repo_path)
        if add_result.returncode != 0:
            raise VerifierError(f"Failed to create git worktree for {repo_path}: {add_result.stderr}")

        try:
            try:
                baseline_failing, _ = _run_pytest(worktree_dir, timeout_seconds)
            except _VerifierTimeout:
                duration = time.monotonic() - start_time
                return TestResult(
                    passed=False,
                    failing_tests=[],
                    failure_trace=f"Baseline test run exceeded the {timeout_seconds}s timeout and was killed.",
                    duration_seconds=duration,
                )

            try:
                _apply_patch(worktree_dir, patch_result["diff"])
            except _PatchApplyError as exc:
                duration = time.monotonic() - start_time
                return TestResult(
                    passed=False,
                    failing_tests=[],
                    failure_trace=f"Patch failed to apply:\n{str(exc)}"[-MAX_FAILURE_TRACE_CHARS:],
                    duration_seconds=duration,
                )

            try:
                after_failing, after_output = _run_pytest(worktree_dir, timeout_seconds)
            except _VerifierTimeout:
                duration = time.monotonic() - start_time
                return TestResult(
                    passed=False,
                    failing_tests=[],
                    failure_trace=f"Test run exceeded the {timeout_seconds}s timeout and was killed.",
                    duration_seconds=duration,
                )

            new_failures = after_failing - baseline_failing
            fixed_tests = baseline_failing - after_failing
            passed = not new_failures and bool(fixed_tests)

            duration = time.monotonic() - start_time
            return TestResult(
                passed=passed,
                failing_tests=sorted(after_failing),
                failure_trace=None if passed else after_output[-MAX_FAILURE_TRACE_CHARS:],
                duration_seconds=duration,
            )
        finally:
            _run(["git", "worktree", "remove", "--force", str(worktree_dir)], repo_path)
            _run(["git", "worktree", "prune"], repo_path)
