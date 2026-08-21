"""PR Node: bundles successful patches into a single GitHub pull request.

AGENTS.md Section 5.7. Pure code, no LLM. This module NEVER calls a merge,
auto-merge, or branch-protection-bypass endpoint anywhere - its job ends at
creating (or updating) a pull request via the GitHub REST API. This is
AGENTS.md Section 3, rule 4: "the single most important boundary in this
file." Do not add a merge call here even behind a flag, even for testing.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from pr.common import EscalatedItem, SuccessfulPatch, describe_drift

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "self-healing"


class PRNodeError(RuntimeError):
    """Raised for infrastructure failures re-applying already-verified patches."""


class PRResult:
    def __init__(self, *, dry_run: bool, branch_name: str, title: str, body: str, pr_url: str | None) -> None:
        self.dry_run = dry_run
        self.branch_name = branch_name
        self.title = title
        self.body = body
        self.pr_url = pr_url


def branch_name_for(api_provider: str) -> str:
    return f"{BRANCH_PREFIX}/{api_provider}"


def _apply_patches_and_collect_file_contents(target_repo: str, patches: list[SuccessfulPatch]) -> dict[str, str]:
    """Re-apply every successful patch's diff in a fresh worktree copy of
    target_repo (never the real target_repo itself) and return
    {file_path: new_full_content} for every changed file, for pushing to
    GitHub via the Contents API. Worktree is cleaned up unconditionally."""

    repo_path = Path(target_repo).resolve()
    file_contents: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="pr-node-worktree-") as tmp_dir:
        worktree_dir = Path(tmp_dir) / "worktree"
        add_result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_dir), "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if add_result.returncode != 0:
            raise PRNodeError(f"Failed to create git worktree for {repo_path}: {add_result.stderr}")

        try:
            for patch in patches:
                _apply_one_diff(worktree_dir, patch)

            for patch in patches:
                file_path = patch.call_site["file_path"]
                file_contents[file_path] = (worktree_dir / file_path).read_text(encoding="utf-8")
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_path, capture_output=True, text=True
            )
            subprocess.run(["git", "worktree", "prune"], cwd=repo_path, capture_output=True, text=True)

    return file_contents


def _apply_one_diff(worktree_dir: Path, patch: SuccessfulPatch) -> None:
    for strip_level in ("1", "0"):
        result = subprocess.run(
            ["git", "apply", f"-p{strip_level}", "--whitespace=nowarn", "-"],
            cwd=worktree_dir,
            input=patch.patch_result["diff"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
    raise PRNodeError(
        f"PR Node failed to re-apply an already-Verifier-passed patch for "
        f"call_site={patch.call_site['id']}: {result.stderr}"
    )


def _build_pr_body(successful: list[SuccessfulPatch], escalated: list[EscalatedItem]) -> str:
    sections = ["## Self-healing API integration patches\n"]

    for patch in successful:
        changelog = (
            f"[changelog]({patch.drift_item['changelog_url']})" if patch.drift_item["changelog_url"] else "(no changelog url)"
        )
        sections.append(
            f"### `{patch.call_site['file_path']}` ({patch.call_site['symbol']})\n"
            f"- **What changed:** {describe_drift(patch.drift_item)} - {changelog}\n"
            f"- **Fix:** {patch.patch_result['rationale']}\n"
            f"- **Tests:** the target repo's full test suite passed "
            f"({patch.test_result['duration_seconds']:.2f}s), after "
            f"{patch.patch_result['attempt_number']} attempt(s) and "
            f"{patch.patch_result['critic_rounds']} critic revision round(s)\n"
        )

    if escalated:
        sections.append("## Not automatically fixed\n")
        for item in escalated:
            sections.append(
                f"### `{item.call_site['file_path']}` ({item.call_site['symbol']})\n"
                f"- **What changed:** {describe_drift(item.drift_item)}\n"
                f"- Exhausted {item.retry_count} attempt(s) without a passing patch. "
                f"See the linked escalation issue for the failure trace.\n"
            )

    return "\n".join(sections)


def open_or_update_pr(
    *,
    api_provider: str,
    target_repo: str,
    github_repo: str,
    successful_patches: list[SuccessfulPatch],
    escalated_items: list[EscalatedItem],
    dry_run: bool,
    github_client: object | None = None,
) -> PRResult:
    """Bundle every successful patch into a single PR on a dedicated
    branch. `github_client` is expected to be a `github.Github` instance
    (or a test double with the same interface); typed as `object` here so
    this module doesn't require PyGithub to be importable just to read its
    signature in a dry-run-only caller.

    If dry_run is True, never touches the GitHub API - just returns what
    would have happened. Never calls a merge endpoint (see module
    docstring). If an open PR from a prior run already exists on this
    provider's branch, its branch is updated in place rather than a
    duplicate PR being opened (AGENTS.md Section 5.7).
    """

    if not successful_patches:
        raise ValueError("open_or_update_pr called with no successful patches - nothing to PR")

    branch_name = branch_name_for(api_provider)
    title = f"Self-healing patch: {api_provider} API drift ({len(successful_patches)} call site(s))"
    body = _build_pr_body(successful_patches, escalated_items)

    if dry_run:
        logger.info("[dry-run] Would open/update PR on branch=%s title=%r", branch_name, title)
        return PRResult(dry_run=True, branch_name=branch_name, title=title, body=body, pr_url=None)

    if github_client is None:
        raise ValueError("github_client is required when dry_run=False")

    file_contents = _apply_patches_and_collect_file_contents(target_repo, successful_patches)

    repo = github_client.get_repo(github_repo)
    default_branch = repo.default_branch
    base_sha = repo.get_branch(default_branch).commit.sha

    try:
        repo.get_git_ref(f"heads/{branch_name}")
        branch_existed = True
    except Exception:
        branch_existed = False

    if not branch_existed:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
    # If the branch already exists (an open PR from a prior run), it is
    # deliberately left where it is rather than force-reset to base_sha -
    # AGENTS.md Section 5.7: "prefer updating that PR's branch over
    # opening a duplicate," not force-pushing over it blindly.

    for file_path, new_content in file_contents.items():
        _upsert_file(repo, file_path, new_content, branch_name, api_provider)

    existing_pr = _find_open_pr(repo, branch_name)
    if existing_pr is not None:
        existing_pr.edit(title=title, body=body)
        pr_url = existing_pr.html_url
    else:
        pr = repo.create_pull(title=title, body=body, head=branch_name, base=default_branch)
        pr_url = pr.html_url

    return PRResult(dry_run=False, branch_name=branch_name, title=title, body=body, pr_url=pr_url)


def _upsert_file(repo: object, file_path: str, new_content: str, branch_name: str, api_provider: str) -> None:
    message = f"self-healing: patch {file_path} for {api_provider} API drift"
    try:
        existing_file = repo.get_contents(file_path, ref=branch_name)
        repo.update_file(file_path, message=message, content=new_content, sha=existing_file.sha, branch=branch_name)
    except Exception:
        repo.create_file(file_path, message=message, content=new_content, branch=branch_name)


def _find_open_pr(repo: object, branch_name: str) -> object | None:
    owner_login = repo.owner.login
    for pr in repo.get_pulls(state="open", head=f"{owner_login}:{branch_name}"):
        return pr
    return None
