"""Single seam for constructing chat models for the three agent roles.

AGENTS.md doesn't pin an LLM provider; this module is where that choice
lives, as a one-function lookup, not a router. A model router that picks a
model per-invocation based on task difficulty was explicitly called out as
future/out-of-scope work when this was designed - if that gets built later,
it plugs in here (config/models.yaml's shape already supports it), rather
than being invented preemptively now.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

AgentRole = Literal["planner", "coder", "critic"]

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


class ModelConfigError(RuntimeError):
    """Raised when config/models.yaml is missing, malformed, or missing a role."""


@lru_cache(maxsize=1)
def _load_model_config() -> dict[str, str]:
    if not _CONFIG_PATH.exists():
        raise ModelConfigError(f"Model config not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ModelConfigError(f"{_CONFIG_PATH} must contain a mapping of role -> model id")
    return config


def get_chat_model(role: AgentRole, *, temperature: float = 0.0, **kwargs: object) -> BaseChatModel:
    """Return a configured chat model for the given agent role.

    Raises ModelConfigError if the role has no entry in config/models.yaml,
    and RuntimeError if ANTHROPIC_API_KEY isn't set - callers should let
    these propagate (AGENTS.md Section 6.2: never a silent except: pass).
    """

    config = _load_model_config()
    if role not in config:
        raise ModelConfigError(f"No model configured for role '{role}' in {_CONFIG_PATH}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export it directly."
        )

    model_id = config[role]
    return ChatAnthropic(model=model_id, api_key=api_key, temperature=temperature, **kwargs)
