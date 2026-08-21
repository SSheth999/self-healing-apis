"""Watcher node: detects API drift by diffing two spec snapshots.

AGENTS.md Section 5.1. Pure code, no LLM. Diffs are computed at the level
of paths, required params, and field types (Section 3, rule 3 -
schema-detectable changes only). Renames and moved endpoints are detected
via explicit `x-renamed-from` / `x-moved-from` spec annotations rather than
name-similarity heuristics - this keeps the diff fully deterministic and
avoids the kind of guessing that belongs in the Planner (an LLM step), not
here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from schemas import DriftItem

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_PROVIDERS_CONFIG_DIR = _REPO_ROOT / "config" / "providers"
_DB_PATH = _REPO_ROOT / "data" / "spec_snapshots.db"


class SpecFetchError(RuntimeError):
    """Raised when fetching the live spec fails.

    Must never be swallowed or treated as "no drift" - AGENTS.md Section
    5.1's "Must not" clause and Section 6.2's "distinguish no drift from
    failed to check, never conflate" rule.
    """


class ProviderConfigError(RuntimeError):
    """Raised when a provider's config/providers/<name>.yaml is missing or malformed."""


@dataclass
class DriftDetectionResult:
    """Internal return value of detect_drift(); unpacked into HealingState
    fields by the graph node wrapper, so it never itself crosses a node
    boundary (AGENTS.md Section 6.1)."""

    drift_items: list[DriftItem]
    old_spec: dict
    new_spec: dict


def _load_provider_config(provider: str) -> dict[str, Any]:
    config_path = _PROVIDERS_CONFIG_DIR / f"{provider}.yaml"
    if not config_path.exists():
        raise ProviderConfigError(f"No provider config found at {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ProviderConfigError(f"{config_path} must contain a mapping")
    return config


def _load_fixture(filename: str) -> dict:
    fixture_path = _FIXTURES_DIR / filename
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


def fetch_live_spec(spec_url: str, timeout: float = 10.0) -> dict:
    """Fetch the current OpenAPI spec from a live URL.

    Raises SpecFetchError on any network/HTTP/parse failure - callers must
    let this propagate, never catch-and-treat-as-empty.
    """

    try:
        response = requests.get(spec_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SpecFetchError(f"Failed to fetch spec from {spec_url}: {exc}") from exc


def _get_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spec_snapshots (
            provider TEXT PRIMARY KEY,
            spec_json TEXT NOT NULL,
            stored_at TEXT NOT NULL
        )
        """
    )
    return conn


def load_last_snapshot(provider: str) -> dict | None:
    """Load the last-stored spec snapshot for a provider, or None if there
    has never been one (e.g. first-ever run)."""

    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT spec_json FROM spec_snapshots WHERE provider = ?", (provider,)
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def commit_snapshot(provider: str, spec: dict) -> None:
    """Store `spec` as the latest snapshot for `provider`.

    Callers (graph.py) must only call this after a successful, non-dry-run
    full graph run - never immediately on fetch - so a crashed mid-run
    doesn't cause a missed diff on the next poll (AGENTS.md Section 5.1,
    bullet 5). Not called at all in --simulate-drift mode, since the
    fixtures are the fixed source of truth for demos and re-running the
    demo command must stay repeatable.
    """

    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO spec_snapshots (provider, spec_json, stored_at)
            VALUES (?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                spec_json = excluded.spec_json,
                stored_at = excluded.stored_at
            """,
            (provider, json.dumps(spec), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _make_drift_item(
    *,
    change_type: str,
    api_path: str,
    field_or_param: str | None,
    old_value: dict,
    new_value: dict,
    changelog_url: str | None,
) -> DriftItem:
    stable_key = f"{api_path}:{change_type}:{field_or_param or ''}"
    item_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]
    return DriftItem(
        id=item_id,
        change_type=change_type,  # type: ignore[typeddict-item]
        api_path=api_path,
        field_or_param=field_or_param,
        old_value=old_value,
        new_value=new_value,
        changelog_url=changelog_url,
        detected_at=datetime.now(timezone.utc).isoformat(),
    )


def _extract_request_schema(operation: dict) -> dict | None:
    try:
        return operation["requestBody"]["content"]["application/x-www-form-urlencoded"]["schema"]
    except (KeyError, TypeError):
        return None


def _diff_schema(api_path: str, old_schema: dict, new_schema: dict, changelog_url: str | None) -> list[DriftItem]:
    items: list[DriftItem] = []
    old_props: dict = old_schema.get("properties", {})
    new_props: dict = new_schema.get("properties", {})
    old_required = set(old_schema.get("required", []))
    new_required = set(new_schema.get("required", []))

    renamed_from_map: dict[str, str] = {}
    for new_name, new_prop in new_props.items():
        renamed_from = new_prop.get("x-renamed-from")
        if renamed_from:
            renamed_from_map[renamed_from] = new_name
            items.append(
                _make_drift_item(
                    change_type="field_renamed",
                    api_path=api_path,
                    field_or_param=f"{renamed_from} -> {new_name}",
                    old_value={"field": renamed_from, **old_props.get(renamed_from, {})},
                    new_value={"field": new_name, **{k: v for k, v in new_prop.items() if k != "x-renamed-from"}},
                    changelog_url=changelog_url,
                )
            )

    for name, old_prop in old_props.items():
        if name in renamed_from_map:
            continue  # already captured as field_renamed above

        if name not in new_props:
            items.append(
                _make_drift_item(
                    change_type="field_removed",
                    api_path=api_path,
                    field_or_param=name,
                    old_value={"field": name, **old_prop},
                    new_value={},
                    changelog_url=changelog_url,
                )
            )
            continue

        new_prop = new_props[name]
        was_required = name in old_required
        is_required = name in new_required
        if not was_required and is_required:
            items.append(
                _make_drift_item(
                    change_type="field_required_changed",
                    api_path=api_path,
                    field_or_param=name,
                    old_value={"field": name, "required": False},
                    new_value={"field": name, "required": True},
                    changelog_url=changelog_url,
                )
            )

        if old_prop.get("type") and new_prop.get("type") and old_prop["type"] != new_prop["type"]:
            items.append(
                _make_drift_item(
                    change_type="param_type_changed",
                    api_path=api_path,
                    field_or_param=name,
                    old_value={"field": name, "type": old_prop["type"]},
                    new_value={"field": name, "type": new_prop["type"]},
                    changelog_url=changelog_url,
                )
            )

    return items


def diff_specs(old_spec: dict, new_spec: dict, *, changelog_url: str | None = None) -> list[DriftItem]:
    """Diff two OpenAPI-shaped specs at the level of paths, required params,
    and field types. Returns an empty list if there's no detectable drift -
    this is the expected common case, not an error (AGENTS.md Section
    5.1, bullet 4)."""

    items: list[DriftItem] = []
    old_paths: dict = old_spec.get("paths", {})
    new_paths: dict = new_spec.get("paths", {})

    moved_from_paths: set[str] = set()
    for new_path, new_path_item in new_paths.items():
        for _method, operation in new_path_item.items():
            if not isinstance(operation, dict):
                continue
            moved_from = operation.get("x-moved-from")
            if moved_from:
                moved_from_paths.add(moved_from)
                items.append(
                    _make_drift_item(
                        change_type="endpoint_moved",
                        api_path=moved_from,
                        field_or_param=None,
                        old_value={"path": moved_from},
                        new_value={"path": new_path},
                        changelog_url=changelog_url,
                    )
                )

    for path, old_path_item in old_paths.items():
        if path in moved_from_paths:
            # Already captured as endpoint_moved; don't also try to diff
            # request-body properties on a path that no longer exists.
            continue
        new_path_item = new_paths.get(path)
        if new_path_item is None:
            continue

        for method, old_operation in old_path_item.items():
            new_operation = new_path_item.get(method)
            if new_operation is None:
                continue
            old_schema = _extract_request_schema(old_operation)
            new_schema = _extract_request_schema(new_operation)
            if old_schema is None or new_schema is None:
                continue
            items.extend(_diff_schema(path, old_schema, new_schema, changelog_url))

    return items


def detect_drift(provider: str, *, simulate_drift: bool = False) -> DriftDetectionResult:
    """Top-level Watcher entry point.

    In --simulate-drift mode, diffs the two hand-crafted fixtures directly
    and never touches the SQLite snapshot table - the fixtures are the
    fixed source of truth for demos, and this keeps `--simulate-drift`
    repeatable across runs. In live mode, fetches the current spec and
    diffs it against the last committed snapshot (see commit_snapshot).
    """

    config = _load_provider_config(provider)
    changelog_url = config.get("changelog_url")

    if simulate_drift:
        old_spec = _load_fixture(f"{provider}_v_old.json")
        new_spec = _load_fixture(f"{provider}_v_new.json")
        drift_items = diff_specs(old_spec, new_spec, changelog_url=changelog_url)
        return DriftDetectionResult(drift_items=drift_items, old_spec=old_spec, new_spec=new_spec)

    new_spec = fetch_live_spec(config["spec_url"])
    old_spec = load_last_snapshot(provider)
    if old_spec is None:
        logger.info(
            "No prior spec snapshot found for provider=%s; treating this run as "
            "establishing the baseline, not as zero drift.",
            provider,
        )
        return DriftDetectionResult(drift_items=[], old_spec=new_spec, new_spec=new_spec)

    drift_items = diff_specs(old_spec, new_spec, changelog_url=changelog_url)
    return DriftDetectionResult(drift_items=drift_items, old_spec=old_spec, new_spec=new_spec)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run the Watcher in isolation against stored fixtures or a live spec.")
    parser.add_argument("--provider", required=True, help="Provider name, e.g. 'stripe'")
    parser.add_argument("--simulate-drift", action="store_true", help="Diff the fixture specs instead of fetching live")
    args = parser.parse_args()

    result = detect_drift(args.provider, simulate_drift=args.simulate_drift)
    print(json.dumps(result.drift_items, indent=2))
    logger.info("Detected %d drift item(s) for provider=%s", len(result.drift_items), args.provider)


if __name__ == "__main__":
    _main()
