# AGENTS.md

Comprehensive guidance for any coding agent (Claude Code, Devin, Cursor, etc.)
working in this repository. This file is the contract between the project
owner and any agent touching this code. When in doubt, an agent should follow
this file over its own priors about "good practice," and should stop and ask
rather than silently deviate.

---

## 1. Project summary

**Name (working title):** Self-Healing API Integrations
**One-liner:** Dependabot for API integrations — when a provider ships a
breaking spec change, the system autonomously detects the drift, locates
every affected call site in a target codebase, generates a fix, verifies it
against the existing test suite in a sandbox, and opens a GitHub PR. A human
merges it.

**Origin:** Scoped against YC's Fall 2026 "Self-Maintaining APIs" Request for
Startups (Harsha Gaddipati). Thesis: API providers shouldn't just announce
breaking changes, they should apply the fix themselves — either as a
per-provider agent, or as a neutral third-party service tracking changes
across vendors (this project takes the neutral third-party shape).

**Current phase:** MVP / proof of concept. One provider, one target repo, one
class of breaking change. The goal of this phase is to prove the full loop
works end-to-end and produces a trustworthy PR — not to generalize yet.

**Non-goals for this phase** (see Section 3 for the enforced list): multi-
provider support, multi-tenant onboarding, semantic/behavioral breaking-change
detection, auto-merge, any user-facing UI beyond a CLI.

---

## 2. Architecture

### 2.1 High-level flow

```
        ┌───────────┐
        │  Watcher  │  scheduled poll or webhook trigger, no LLM
        └─────┬─────┘
              │ drift_report: list[DriftItem]
              ▼
        ┌───────────┐
        │  Planner  │  LLM, 1 call per DriftItem, structured SearchPlan out
        └─────┬─────┘
              │ search_plan: SearchPlan (symbols to look for + rationale)
              ▼
        ┌───────────┐
        │  Locator  │  AST scan of target repo, executes SearchPlan, no LLM
        └─────┬─────┘
              │ affected_call_sites: list[CallSite]
              ▼
   ┌──────────────────────────────────────────┐
   │         Coder ⇄ Critic loop               │
   │  Coder: agentic, bounded tool calls        │
   │    (expand snippet, read-only repo search) │
   │    → proposes diff for ONE file            │
   │  Critic: LLM, 1 call, reviews diff against │
   │    drift + call site → approve / feedback  │
   │    (bounded revision rounds, then proceeds │
   │    to Verifier regardless — tests are the  │
   │    ground truth, not the Critic's opinion) │
   └────────────────────┬──────────────────────┘
                         │ patch_result: PatchResult
   ┌─────────────────────┼──────────────────────┐
   │                     ▼                       │
   │               ┌───────────┐                 │
   │               │ Verifier  │  sandbox, no LLM │
   │               └─────┬─────┘                 │
   │                     │                        │
   │                pass ┴ fail                   │
   │                 │        │                   │
   │                 ▼        └─(retry_count < 3)─► back to Coder⇄Critic
   │           ┌───────────┐        w/ failure trace
   │           │  PR Node  │  opens real GitHub PR, cites changelog
   │           └───────────┘
   │
   └──────────────────(retry_count >= 3)─► Escalation Node (visible failure)
```

Every LLM call in this graph — Planner, Coder (each tool-call turn), Critic —
is logged as a structured `AgentStep` (Section 4.6). Auditability now comes
from complete step-level logging, not from having a single LLM call site (see
2.2).

### 2.2 Design principles (why the graph looks like this)

- **This is a multi-agent system, not a single-LLM pipeline — that's a
  deliberate choice, not scope creep.** Planner, Coder, and Critic are three
  narrow, single-responsibility LLM roles. Watcher, Locator, Verifier, PR
  Node, and Escalation Node remain deterministic, LLM-free code. Auditability
  no longer comes from minimizing the *number* of LLM call sites — it comes
  from every LLM call and every tool invocation, at every step, producing a
  logged, structured `AgentStep` (Section 4.6). You can always reconstruct
  exactly what an agent saw, called, and decided, in order.
- **LangGraph is load-bearing here, not decorative.** The Coder is a real
  bounded agentic loop (propose → call a tool → observe → propose again),
  which needs conditional routing between "call a tool" and "emit final
  output" — this is a genuine graph/loop shape, not a linear script. If the
  Coder is ever simplified back to single-shot generation, revisit whether
  LangGraph is still justified (see the "when is LangGraph necessary"
  discussion this file's history is built on — don't keep the dependency out
  of inertia).
- **Sandboxed execution stays LLM-free, no exceptions.** No agent — Planner,
  Coder, or Critic — ever executes code, runs tests, or gets write access
  outside of the single target file it's patching. The Verifier is the only
  place code actually runs, and it runs in a sandbox (Section 7). This is a
  security boundary as much as a design one: agentic reasoning does not mean
  agentic code execution.
- **Autonomy stops at the PR.** The system is "self-healing" through the
  point of producing a verified, tested fix — but merging into a real
  codebase is a human decision, always, in every configuration, with no
  override flag. This is a hard boundary, not a default (see Section 3).
- **Context is scoped per agent invocation, not threaded globally.** No
  agent receives more information than it strictly needs to do its job:
  Planner sees one `DriftItem`; Coder sees one `DriftItem` + one `CallSite`
  + (bounded) tool results + (on retry) one failure trace; Critic sees one
  `DriftItem` + one `CallSite` + one proposed diff. None of them ever see the
  full spec diff or whole file contents dumped in. Smaller, scoped context
  produces more reliable output and is cheaper and faster.
- **Fail loudly, never silently.** If the system can't produce a passing
  patch within the retry budget, it must surface that clearly (Escalation
  Node), not drop the task or merge something untested.
- **Structured data between every node.** Nodes never pass raw strings of
  prose between each other — always typed objects (see Section 4). Free-text
  reasoning happens inside an agent's own LLM calls, but the *output* of
  every agent call must conform to a typed shape (`SearchPlan`, `PatchResult`,
  `CriticVerdict`), never prose, and every intermediate reasoning/tool-call
  step is captured in `AgentStep` rather than discarded.

---

## 3. Scope boundaries — hard rules

These are enforced boundaries, not suggestions. An agent must not cross them
even if a task seems to "naturally" extend in that direction, even if asked
by a vague or ambiguous instruction, and even if it seems like an
improvement. If a task requires crossing one of these lines, **stop and ask
the project owner explicitly before proceeding.**

1. **One API provider only for this phase (Stripe).** Do not add abstraction
   layers, plugin systems, or "provider adapters" in anticipation of future
   providers. Build the concrete Stripe case. Generalizing before the
   concrete case is proven is a known failure mode for this kind of project
   — resist it even when it looks like "clean code."
2. **One target repo (`demo_repo/`).** No OAuth flows, no GitHub App
   installation flows, no multi-tenant repo registration. If asked to "make
   it work for any repo," treat that as a phase-2 request and flag it rather
   than building it quietly.
3. **Schema-detectable breaking changes only.** Handle: renamed field,
   removed/deprecated field, changed required-vs-optional, moved/renamed
   endpoint, changed parameter type. Do NOT attempt to detect or fix
   semantic/behavioral changes (e.g. "this endpoint now rate-limits more
   aggressively" or "the meaning of this field subtly changed without a
   schema change"). These require product judgment this system does not
   have yet.
4. **No auto-merge, under any circumstance, in this codebase.** Do not:
   - add a `--auto-merge` flag, even disabled by default
   - add merge automation "behind a config flag for testing"
   - call the GitHub merge API anywhere in this codebase
   The PR Node's job ends at `POST /repos/{owner}/{repo}/pulls`. Nothing in
   this repository should ever call a merge endpoint. This is the single
   most important boundary in this file.
5. **No dashboard or web UI.** The interface is a CLI and a GitHub Action
   trigger. Do not add a web server, admin panel, or frontend framework to
   this repo in this phase.
6. **No test-weakening to make patches pass.** The Verifier must never
   modify, skip, or loosen the target repo's existing tests in order to make
   a patch appear to pass. If a patch legitimately requires a test update
   (e.g. the test itself hardcoded the old field name), that must be
   surfaced as part of the PR description for human review, never done
   silently.
7. **No telemetry/analytics exfiltration of target repo contents.** Code
   snippets pulled from the target repo (`CallSite.snippet`) must not be
   logged to any external service, sent to a third-party analytics
   provider, or persisted anywhere outside this system's own sandbox/storage.
   They may be sent to the LLM provider as part of the Coder's or Critic's
   prompt (that is the system's job), and nowhere else.

If genuinely unsure whether something falls inside or outside these
boundaries, the correct move is to ask, not to interpret generously.

---

## 4. Data model (canonical schemas)

All cross-node data must conform to these shapes. Treat this section as the
source of truth over any inline comments in code — if they ever disagree,
this file wins until deliberately updated.

### 4.1 `DriftItem`

```python
class DriftItem(TypedDict):
    id: str                      # stable id, e.g. hash of (path, change_type)
    change_type: Literal[
        "field_renamed",
        "field_removed",
        "field_required_changed",
        "endpoint_moved",
        "param_type_changed",
    ]
    api_path: str                # e.g. "/v1/charges"
    field_or_param: str | None    # e.g. "source" -> "payment_method"
    old_value: dict               # minimal relevant slice of old spec
    new_value: dict               # minimal relevant slice of new spec
    changelog_url: str | None
    detected_at: str              # ISO 8601 timestamp
```

### 4.2 `CallSite`

```python
class CallSite(TypedDict):
    id: str                       # stable id, e.g. hash of (file_path, line_start)
    drift_item_id: str             # foreign key to DriftItem.id
    file_path: str
    line_start: int
    line_end: int
    snippet: str                   # exact source text, nothing more
    symbol: str                    # e.g. "stripe.Charge.create"
```

### 4.3 `SearchPlan`

```python
class SearchPlan(TypedDict):
    drift_item_id: str             # foreign key to DriftItem.id
    symbols: list[str]             # e.g. ["stripe.Charge.create", "stripe.Charge.modify"]
    rationale: str                 # why these symbols, one/two sentences
```

Emitted by the Planner (Section 5.2), consumed by the Locator (Section 5.3)
as its input for which symbols to AST-scan for. The Locator itself never
decides *what* to search for — it only executes the plan.

### 4.4 `PatchResult`

```python
class PatchResult(TypedDict):
    call_site_id: str
    diff: str                      # unified diff format, single file
    rationale: str                 # one sentence, for the PR description
    attempt_number: int            # 1-indexed, resets per call_site
    critic_rounds: int             # how many Coder<->Critic revisions this attempt took
```

### 4.5 `CriticVerdict`

```python
class CriticVerdict(TypedDict):
    call_site_id: str
    attempt_number: int             # which PatchResult attempt this critiques
    revision_round: int             # 1-indexed within this attempt
    approved: bool
    feedback: str                   # fed back to Coder verbatim if not approved
```

The Critic is a quality filter, not a hard gate: it never escalates a call
site on its own. If revisions keep getting rejected, the last diff still
goes to the Verifier — the real test suite is the ground truth, not the
Critic's opinion of the diff (see 2.2).

### 4.6 `AgentStep`

```python
class AgentStep(TypedDict):
    node: Literal["planner", "coder", "critic"]
    drift_item_id: str | None      # set for planner steps
    call_site_id: str | None       # set for coder/critic steps
    attempt_number: int | None     # set for coder/critic steps
    step_number: int               # 1-indexed within this single node invocation
    tool_called: str | None        # e.g. "expand_snippet", "search_repo"; None if a final-output step
    tool_args: dict | None
    tool_result_summary: str | None  # truncated, see Section 6.3
    output: dict | None            # the structured output if this is a final step
    timestamp: str                 # ISO 8601
```

This is the mechanism behind the "structured trace" auditability principle
in 2.2 — every LLM call and every tool call, for every agent, in order,
with nothing dropped. Nothing here is prose-only; `tool_result_summary` and
`output` are always the structured/truncated forms, never a raw dump.

### 4.7 `TestResult`

```python
class TestResult(TypedDict):
    passed: bool
    failing_tests: list[str]       # test node ids, empty if passed
    failure_trace: str | None      # truncated, see Section 6.3 on size limits
    duration_seconds: float
```

### 4.8 `HealingState` (the LangGraph shared state)

```python
class HealingState(TypedDict):
    api_provider: str
    spec_version_old: dict
    spec_version_new: dict
    drift_report: list[DriftItem]
    target_repo: str
    search_plans: dict[str, SearchPlan]      # keyed by drift_item_id
    affected_call_sites: list[CallSite]
    current_call_site_id: str | None    # which call site is being processed now
    patch_results: dict[str, PatchResult]        # keyed by call_site_id
    critic_verdicts: dict[str, list[CriticVerdict]]  # keyed by call_site_id
    agent_traces: dict[str, list[AgentStep]]     # keyed by call_site_id, or "planner:<drift_item_id>"
    test_results: dict[str, TestResult]    # keyed by call_site_id
    retry_counts: dict[str, int]           # keyed by call_site_id
    escalated_call_site_ids: list[str]
    pr_url: str | None
```

**Rule:** state is processed **one `CallSite` at a time** through
Coder⇄Critic→Verifier, not batched. This keeps retry logic and context
scoping simple — a failure on one call site never contaminates the prompt,
agent trace, or retry budget of another. Only after every call site is
either patched-and-verified or escalated does the graph proceed to the PR
node, which bundles all successful patches into a single PR (or splits into
multiple PRs if the project owner has configured that — see Section 8).

---

## 5. Node specifications

### 5.1 Watcher (`watcher/`)

- **Type:** pure code, no LLM.
- **Input:** `api_provider: str`.
- **Behavior:**
  1. Fetch current spec (OpenAPI JSON/YAML if available; fall back to
     changelog scraping only if no machine-readable spec exists).
  2. Load last-stored spec snapshot from local storage (SQLite table
     `spec_snapshots`).
  3. Diff old vs. new at the level of paths, required params, field types.
  4. Emit `list[DriftItem]`. If no drift, emit empty list and the graph
     terminates cleanly (this is the expected common case — most polls find
     nothing).
  5. Store the new spec as the latest snapshot ONLY after a successful full
     graph run (or explicit dry-run flag) — not immediately on fetch, so a
     crashed mid-run doesn't cause a missed diff on next poll.
- **Fixtures for MVP:** since polling live Stripe continuously isn't
  necessary for a demo, ship `watcher/fixtures/stripe_v_old.json` and
  `watcher/fixtures/stripe_v_new.json` with 2–3 hand-crafted realistic
  breaking changes, and a `--simulate-drift` flag that uses these instead of
  a live fetch.
- **Must not:** call an LLM. Must not silently swallow a fetch error — a
  failed fetch should raise/log clearly, not be treated as "no drift."

### 5.2 Planner (`planner/`)

- **Type:** LLM-calling, exactly one call per invocation (no tool loop).
- **Input:** exactly one `DriftItem`.
- **Behavior:**
  1. Build a prompt containing ONLY the drift description (what changed,
     old value, new value, `change_type`).
  2. Call the LLM, requesting a structured `SearchPlan`: the list of
     symbols/call patterns this drift plausibly affects, plus a short
     rationale.
  3. Parse the response into a `SearchPlan`. If it isn't valid structured
     output, retry (same malformed-output-is-a-failed-attempt rule as the
     Coder, Section 5.4) — this does not consume the per-`CallSite` retry
     budget, since no `CallSite` exists yet at this point.
- **Why this exists as its own agent role:** deciding *which symbols* a
  schema change affects (e.g. "renamed field on `/v1/charges`" implies both
  `stripe.Charge.create` and `stripe.Charge.modify`, not just the literally
  named endpoint) requires domain knowledge about the Stripe SDK's shape.
  That's a judgment call, not a mechanical lookup — it belongs in an LLM
  step with structured output, not hardcoded in the Locator.
- **Must not:** touch the target repo, read any file contents, or execute
  anything. It only reasons about the drift description in the abstract.
- **Prompt template location:** `planner/prompts/plan_template.txt`.

### 5.3 Locator (`locator/`)

- **Type:** pure code, AST-based.
- **Input:** `search_plans: dict[str, SearchPlan]`, `target_repo: str`.
- **Behavior:**
  1. For each `SearchPlan`, take its `symbols` list as given — the Locator
     does not decide what to search for, only how to find it.
  2. Parse target repo files with Python's `ast` module (v1 is Python-only;
     do not add JS/TS parsing in this phase — see Section 3).
  3. Walk the AST for call expressions matching the plan's symbol(s).
  4. Emit one `CallSite` per match, each linked to its `drift_item_id`.
- **Scoping rule:** never do a blind full-repo text grep as a substitute for
  AST parsing — false positives (e.g. a comment mentioning the field name)
  produce garbage `CallSite`s that waste Coder/Critic calls and pollute the
  PR. This holds regardless of the Planner's output; the Planner supplies
  *what* to look for, the Locator still finds it precisely.
- **Must not:** call an LLM. Must not modify any files.

### 5.4 Coder (`coder/`)

- **Type:** LLM-calling agent, bounded tool-calling loop (the one node in
  this graph allowed to make more than one LLM call per invocation — see
  Section 6.1).
- **Input (per invocation):** exactly one `DriftItem`, one `CallSite`, and
  (only on retry) the most recent `failure_trace` from `TestResult` and any
  `CriticVerdict.feedback` from the current attempt's revision rounds.
- **Available tools (read-only, scoped):**
  - `expand_snippet(extra_lines: int)` — widen the call site's context
    window; still bounded, does not escalate to whole-file reads (Section
    6.3).
  - `search_repo(query: str)` — read-only, bounded-results grep across
    `target_repo` for other usages of a symbol, for consistency checking.
    This is for *context*, not editing — the Coder may still only propose a
    diff for the one file its `CallSite` lives in.
- **Behavior:**
  1. Reason over the drift description + call site snippet (+ any failure
     trace / critic feedback on retry), optionally calling tools above to
     gather more context.
  2. Each tool call + its result, and the final output, are logged as an
     `AgentStep` (Section 4.6).
  3. Produce a unified diff for the single file in question, plus a
     one-sentence rationale, as a `PatchResult`.
  4. If the Critic (Section 5.5) rejects the diff, the Coder gets the
     feedback and revises within the same attempt (`critic_rounds`
     increments); once the Critic approves (or the exchange has gone on
     long enough that continuing isn't productive), the diff goes to the
     Verifier.
  5. If the final output isn't a valid unified diff, treat as a failed
     attempt and retry (counts against the retry budget) rather than
     passing malformed output downstream.
- **Explicit prohibition:** do not pass the full old/new spec JSON into any
  prompt. Do not pass full file contents when only the call site is needed.
  Never execute code, run tests, or write to any file other than the one
  `CallSite.file_path` — that boundary is enforced by tooling, not just
  convention (see 2.2 on sandboxed execution staying LLM-free).
- **Retry behavior:** each Verifier round-trip failure increments
  `retry_counts[call_site_id]`. At `retry_counts[call_site_id] >= 3`, this
  call site is added to `escalated_call_site_ids` and the graph moves to the
  next call site (or to Escalation Node handling if this is the last one)
  instead of looping forever. (Note: as of this revision there is no hard
  cap on tool-call turns within a single Coder invocation or on
  Coder⇄Critic revision rounds within a single attempt — this is an
  explicit, deliberate gap for the MVP, not an oversight; see Section 6.3.
  Revisit if cost or runaway-loop behavior becomes a real problem.)
- **Prompt template location:** `coder/prompts/coder_template.txt`. Keep
  the template in a file, not inlined in code, so it can be iterated on
  without a code change.

### 5.5 Critic (`critic/`)

- **Type:** LLM-calling, exactly one call per invocation (no tool loop).
- **Input:** the `DriftItem`, the `CallSite`, and the Coder's current
  proposed diff.
- **Behavior:**
  1. Build a prompt containing ONLY the drift description, the call site
     snippet, and the proposed diff.
  2. Call the LLM, requesting a structured `CriticVerdict`: `approved: bool`
     plus `feedback: str` (required if not approved).
  3. If not approved, `feedback` is passed back to the Coder verbatim for
     the next revision round within the same attempt.
- **Principle:** the Critic is a cheap pre-filter to catch obviously bad
  diffs before they cost a full sandboxed Verifier run — it is never a hard
  gate and never escalates a call site on its own. Whatever the Coder's
  latest diff is when the revision exchange ends (approved or not) still
  goes to the Verifier; the real test suite is the actual ground truth (see
  2.2 and Section 4.5).
- **Must not:** call any tool, execute anything, or modify the diff itself —
  it only reviews and gives feedback in prose (captured as an `AgentStep`).
- **Prompt template location:** `critic/prompts/critic_template.txt`.

### 5.6 Verifier (`verifier/`)

- **Type:** pure code, sandboxed execution.
- **Input:** `PatchResult`, `target_repo`.
- **Behavior:**
  1. Create an isolated copy of the target repo (fresh git worktree or
     container — see Section 7 for sandboxing options).
  2. Apply the patch diff.
  3. Run the repo's existing test suite (`pytest` for the demo repo).
  4. Capture pass/fail, failing test ids, and a truncated failure trace.
  5. Emit `TestResult`.
- **Must not:** modify, skip, comment out, or otherwise weaken any test to
  force a pass (see Section 3, rule 6). Must not leave sandbox artifacts on
  disk after the run — clean up temp dirs/containers unconditionally, even
  on failure/exception (use `try/finally` or context managers).
- **Timeout:** enforce a hard wall-clock timeout on test execution (default
  120s, configurable) to prevent a hung test run from stalling the whole
  graph.

### 5.7 PR Node (`pr/`)

- **Type:** pure code, GitHub REST API.
- **Input:** all successful `PatchResult` + `TestResult` pairs for this run.
- **Behavior:**
  1. Create a new branch.
  2. Apply all successful patches to it.
  3. Open a PR via `POST /repos/{owner}/{repo}/pulls`.
  4. PR description must include, per patch: the originating changelog URL,
     the drift description in plain English, and confirmation that tests
     passed (with the specific test names).
  5. If any call sites were escalated, mention them explicitly in the PR
     description as "not automatically fixed" rather than omitting them.
- **Must not:** call any merge, auto-merge, or branch-protection-bypass
  endpoint. Must not force-push over an existing open PR from a prior run
  without checking first — if a prior PR from this system is still open for
  this provider, prefer updating that PR's branch over opening a duplicate.

### 5.8 Escalation Node

- **Type:** pure code.
- **Input:** `escalated_call_site_ids`, their `DriftItem`s and last
  `TestResult`/failure trace.
- **Behavior:** opens a GitHub issue (not a PR) per escalated item, or one
  consolidated issue if several call sites in the same run escalated,
  containing: what changed, which call site, what was attempted, and the
  last failure trace.
- **Principle:** this node exists so failure is always visible. A run that
  ends in "nothing happened, no error, no output" is a bug in this system,
  full stop.

---

## 6. Cross-cutting conventions

### 6.1 Code style

- Python 3.11+, fully type-hinted. `TypedDict` for state/data shapes,
  `dataclass` acceptable for internal helper objects that never cross a node
  boundary.
- Each node exposed as a pure function `(state: HealingState) ->
  HealingState` at its LangGraph entry point, even if internally it calls
  several helpers. This keeps the graph wiring in `graph.py` trivially
  readable.
- Planner and Critic make exactly one LLM call per invocation. The Coder is
  the sole exception: it may make multiple LLM calls (tool-call turns) per
  invocation as part of its bounded agentic loop (Section 5.4) — every turn
  must still be logged as an `AgentStep`. No node may call an LLM in a batch
  across multiple call sites or drift items at once; invocation is always
  scoped to exactly one `CallSite` or one `DriftItem`.
- Prefer explicit over clever. This is an auditability-first project; a
  slightly more verbose function that's easy to trace beats a compact one
  that isn't.

### 6.2 Error handling

- Every external call (spec fetch, GitHub API, LLM API, subprocess/test
  execution) must be wrapped and produce a typed result or a clearly logged
  exception — never a silent `except: pass`.
- Distinguish "no drift found" (success, empty result) from "failed to check
  for drift" (error) everywhere. These must never be conflated.
- Sandbox cleanup must happen even on exception (see 5.4).

### 6.3 Size / truncation limits

- Failure traces passed back into the Coder's retry prompt: truncate to
  the last 2000 characters (the most relevant part of a traceback is
  usually the end). Never pass an entire raw CI log into a prompt.
- `CallSite.snippet`: bounded to the matched call expression plus a small
  fixed window (default ±10 lines), not the whole file, unless the Coder
  explicitly requests more via `expand_snippet` (Section 5.4) — even then,
  cap the total expanded window (e.g. 100 lines) rather than allowing
  unbounded growth toward "just read the whole file."
- `search_repo` tool results (Section 5.4): cap the number of matches
  returned (e.g. 10) and truncate each match to a line or two of context.
  This is read-only context for the Coder, not a path to multi-file edits.
- **Known gap, deliberately left open for MVP:** there is currently no hard
  cap on the number of tool-call turns in a single Coder invocation, nor on
  the number of Coder⇄Critic revision rounds within a single attempt. The
  outer bound (`retry_counts[call_site_id] >= 3` → escalate) still limits
  total cost per call site across Verifier round-trips, but a single attempt
  could in principle loop for a while before reaching the Verifier. This was
  an explicit decision (not an oversight — see chat history around the
  agentic redesign), to be revisited if cost or hangs become a real problem
  in practice.

### 6.4 Logging

- Structured logging (JSON lines) at each node boundary: node name, state
  keys touched, duration, pass/fail. No secrets, tokens, or full LLM prompts
  in logs at default log level — those go behind a `--verbose`/debug flag
  only, and even then should be written to local files, not shipped
  anywhere.

### 6.5 Configuration

- All provider-specific config (spec URL, changelog URL, auth if needed) in
  `config/providers/stripe.yaml`. Do not hardcode provider details inline in
  node code — even though this is a single-provider MVP, keep provider
  *facts* in config so the eventual (future, not-now) generalization is a
  data change, not a code change. This does not contradict Section 3's
  "don't build multi-provider abstractions" — it just means don't hardcode
  a URL string three files deep.
- Secrets (GitHub token, LLM API key) via environment variables only, never
  committed, never logged. `.env.example` should list required variable
  names with no real values.

---

## 7. Sandboxing options (pick one for MVP, document the choice in code)

- **Simplest:** fresh temp directory + git worktree + subprocess `pytest`
  run. No isolation from the host beyond filesystem location. Fine for a
  personal demo repo with no untrusted code.
- **Safer:** Docker container per verification run, repo mounted read-write
  inside, tests run inside the container, container destroyed after.
  Preferred if there's any chance this demo repo will later be a stand-in
  for a less-trusted target.
- For MVP, the temp-directory approach is acceptable and faster to build;
  note in `verifier/README.md` which approach is in use and why, so it's an
  explicit decision, not an oversight.

---

## 8. Testing this project itself

(Distinct from the Verifier node, which tests the *target* repo — this
section is about testing the self-healing system's own code.)

- `demo_repo/` and its `tests/` directory function as the system's own
  integration fixture. Every seeded breaking change in
  `watcher/fixtures/stripe_v_new.json` must have:
  - a corresponding call site in `demo_repo/` written in the "old" style
  - a test in `demo_repo/tests/` that fails against the old call site and
    passes against the correctly patched one (FAIL_TO_PASS / PASS_TO_PASS
    pattern)
- Unit tests for each node in isolation (`tests/test_watcher.py`,
  `tests/test_locator.py`, `tests/test_planner.py`, `tests/test_coder.py`,
  `tests/test_critic.py`, etc.) using the fixtures, with every LLM call
  (Planner, Coder — including its tool-call turns, Critic) mocked for
  deterministic unit tests. For the Coder specifically, test the loop logic
  itself (does it call tools when expected, does it respect the Critic
  feedback on revision) with a scripted mock LLM, not just the final diff
  output.
- **End-to-end test is mandatory before any node is considered done:**
  `python graph.py --simulate-drift` must run the full graph against the
  fixtures and produce a real (or dry-run) PR. Passing a node's isolated
  unit test is necessary but not sufficient — confirm it works inside the
  full graph run too.

---

## 9. Commands

```bash
# Full pipeline against simulated/fixture drift (primary demo command)
python graph.py --simulate-drift

# Full pipeline in dry-run mode (no real PR/issue created, prints what would happen)
python graph.py --simulate-drift --dry-run

# Watcher only, against stored fixtures
python -m watcher.diff_engine --provider stripe --simulate-drift

# Run demo repo's own test suite directly (sanity check outside the pipeline)
pytest demo_repo/tests/

# Run this project's own unit tests
pytest tests/
```

(Keep this section synced with real entrypoints as they're built — an
AGENTS.md with stale commands is worse than one with none, because it
actively misleads. Update it in the same commit that changes a CLI surface.)

---

## 10. Glossary

- **Drift** — a detected difference between two versions of an API's
  contract (spec or changelog-derived) that could break existing integrations.
- **Call site** — a specific location in a target codebase where code
  invokes the part of the API that drifted.
- **Patch** — a minimal, scoped code change intended to make a call site
  compatible with the new API contract.
- **Escalation** — the explicit, visible failure path taken when the system
  cannot produce a passing patch within its retry budget.
- **Self-healing** — in this project's specific and limited sense: fully
  autonomous through PR creation, with merge always gated on a human. Not a
  claim of unattended production deployment.
- **Agent** — an LLM-calling node with a narrow, single responsibility
  (Planner, Coder, Critic). "Agentic" in this project means bounded,
  tool-using, and multi-step where useful (the Coder) — not unconstrained
  or unsupervised. Every agent's calls and tool use are logged as
  `AgentStep`s; see 2.2.
- **Search plan** — the Planner's structured output: which code symbols a
  given drift item plausibly affects, and why.
- **Critic verdict** — the Critic's structured review of a Coder-proposed
  diff; approval or rejection-with-feedback, never a hard gate on its own.

---

## 11. When instructions conflict

If a task instruction (from the project owner, in a PR description, in an
issue) appears to conflict with this file — especially anything touching
Section 3's hard rules — treat this file as authoritative and flag the
conflict rather than silently following the more permissive instruction.
This file should only be changed via an explicit, deliberate edit to
AGENTS.md itself, discussed as its own change — not as a side effect of
"just doing what the latest message asked for."