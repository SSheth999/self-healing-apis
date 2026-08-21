"""Unit tests for watcher/diff_engine.py (AGENTS.md Section 5.1)."""

from __future__ import annotations

import pytest

from watcher.diff_engine import (
    DriftDetectionResult,
    SpecFetchError,
    commit_snapshot,
    detect_drift,
    diff_specs,
    load_last_snapshot,
)


class TestDiffSpecs:
    def test_no_drift_on_identical_specs(self) -> None:
        spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["amount"],
                                        "properties": {"amount": {"type": "integer"}},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert diff_specs(spec, spec) == []

    def test_detects_field_renamed(self) -> None:
        old_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "required": [],
                                        "properties": {"source": {"type": "string"}},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        new_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "required": [],
                                        "properties": {
                                            "payment_method": {"type": "string", "x-renamed-from": "source"}
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        items = diff_specs(old_spec, new_spec)
        assert len(items) == 1
        assert items[0]["change_type"] == "field_renamed"
        assert items[0]["api_path"] == "/v1/charges"
        assert items[0]["field_or_param"] == "source -> payment_method"

    def test_detects_field_required_changed(self) -> None:
        def spec_with_required(required: list[str]) -> dict:
            return {
                "paths": {
                    "/v1/refunds": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/x-www-form-urlencoded": {
                                        "schema": {
                                            "type": "object",
                                            "required": required,
                                            "properties": {"reason": {"type": "string"}},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        items = diff_specs(spec_with_required([]), spec_with_required(["reason"]))
        assert len(items) == 1
        assert items[0]["change_type"] == "field_required_changed"
        assert items[0]["field_or_param"] == "reason"

    def test_detects_endpoint_moved(self) -> None:
        old_spec = {"paths": {"/v1/customers/{customer}/sources": {"post": {}}}}
        new_spec = {
            "paths": {
                "/v1/payment_methods/{payment_method}/attach": {
                    "post": {"x-moved-from": "/v1/customers/{customer}/sources"}
                }
            }
        }
        items = diff_specs(old_spec, new_spec)
        assert len(items) == 1
        assert items[0]["change_type"] == "endpoint_moved"
        assert items[0]["api_path"] == "/v1/customers/{customer}/sources"
        assert items[0]["new_value"]["path"] == "/v1/payment_methods/{payment_method}/attach"

    def test_detects_field_removed(self) -> None:
        old_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "required": [],
                                        "properties": {"legacy_field": {"type": "string"}},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        new_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {"type": "object", "required": [], "properties": {}}
                                }
                            }
                        }
                    }
                }
            }
        }
        items = diff_specs(old_spec, new_spec)
        assert len(items) == 1
        assert items[0]["change_type"] == "field_removed"
        assert items[0]["field_or_param"] == "legacy_field"

    def test_detects_param_type_changed(self) -> None:
        def spec_with_type(type_name: str) -> dict:
            return {
                "paths": {
                    "/v1/charges": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/x-www-form-urlencoded": {
                                        "schema": {
                                            "type": "object",
                                            "required": [],
                                            "properties": {"amount": {"type": type_name}},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        items = diff_specs(spec_with_type("integer"), spec_with_type("string"))
        assert len(items) == 1
        assert items[0]["change_type"] == "param_type_changed"


class TestDetectDriftSimulateMode:
    def test_finds_three_hand_crafted_drift_items(self) -> None:
        result = detect_drift("stripe", simulate_drift=True)
        assert isinstance(result, DriftDetectionResult)
        change_types = {item["change_type"] for item in result.drift_items}
        assert change_types == {"field_renamed", "field_required_changed", "endpoint_moved"}

    def test_simulate_mode_never_touches_sqlite_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr("watcher.diff_engine.load_last_snapshot", lambda provider: calls.append("load") or None)
        monkeypatch.setattr("watcher.diff_engine.commit_snapshot", lambda provider, spec: calls.append("commit"))

        detect_drift("stripe", simulate_drift=True)

        assert calls == []

    def test_is_repeatable_across_runs(self) -> None:
        first = detect_drift("stripe", simulate_drift=True)
        second = detect_drift("stripe", simulate_drift=True)
        assert len(first.drift_items) == len(second.drift_items) == 3


class TestDetectDriftLiveMode:
    def test_first_ever_run_establishes_baseline_with_no_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("watcher.diff_engine.load_last_snapshot", lambda provider: None)
        monkeypatch.setattr("watcher.diff_engine.fetch_live_spec", lambda spec_url, timeout=10.0: {"paths": {}})

        result = detect_drift("stripe", simulate_drift=False)

        assert result.drift_items == []

    def test_fetch_failure_raises_and_is_never_silently_treated_as_no_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_fetch_error(spec_url: str, timeout: float = 10.0) -> dict:
            raise SpecFetchError("boom")

        monkeypatch.setattr("watcher.diff_engine.fetch_live_spec", _raise_fetch_error)

        with pytest.raises(SpecFetchError):
            detect_drift("stripe", simulate_drift=False)

    def test_diffs_against_last_committed_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        old_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {"type": "object", "required": [], "properties": {"amount": {"type": "integer"}}}
                                }
                            }
                        }
                    }
                }
            }
        }
        new_spec = {
            "paths": {
                "/v1/charges": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {"type": "object", "required": [], "properties": {"amount": {"type": "string"}}}
                                }
                            }
                        }
                    }
                }
            }
        }
        monkeypatch.setattr("watcher.diff_engine.load_last_snapshot", lambda provider: old_spec)
        monkeypatch.setattr("watcher.diff_engine.fetch_live_spec", lambda spec_url, timeout=10.0: new_spec)

        result = detect_drift("stripe", simulate_drift=False)

        assert len(result.drift_items) == 1
        assert result.drift_items[0]["change_type"] == "param_type_changed"


class TestSnapshotPersistence:
    def test_commit_then_load_roundtrip(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("watcher.diff_engine._DB_PATH", tmp_path / "spec_snapshots.db")

        assert load_last_snapshot("test-provider") is None

        spec = {"paths": {"/v1/foo": {}}}
        commit_snapshot("test-provider", spec)

        assert load_last_snapshot("test-provider") == spec

    def test_commit_overwrites_previous_snapshot(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("watcher.diff_engine._DB_PATH", tmp_path / "spec_snapshots.db")

        commit_snapshot("test-provider", {"version": 1})
        commit_snapshot("test-provider", {"version": 2})

        assert load_last_snapshot("test-provider") == {"version": 2}
