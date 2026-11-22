"""OpenAI-compatible LLM providers (OpenAI / DeepSeek / Groq) selected by env vars.

All three expose the OpenAI Chat Completions API, so one SDK (``openai``) is used with a
different ``base_url``. Per-provider overrides:

==========  ===================  ==============================  ===========================
provider    key env var          default base_url                default model (``*_MODEL``)
==========  ===================  ==============================  ===========================
openai      OPENAI_API_KEY       (SDK default)                   gpt-4o-mini
deepseek    DEEPSEEK_API_KEY     https://api.deepseek.com        deepseek-chat
groq        GROQ_API_KEY         https://api.groq.com/openai/v1  llama-3.3-70b-versatile
==========  ===================  ==============================  ===========================

``<PROVIDER>_BASE_URL``, ``<PROVIDER>_MODEL`` and ``<PROVIDER>_RESPONSE_MODE``
(``parse`` | ``json_schema`` | ``json_object``) override the defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

RESPONSE_MODES = ("parse", "json_schema", "json_object")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: str
    base_url: str | None
    default_model: str
    response_mode: str


PROVIDERS: dict[str, ProviderSpec] = {
    # OpenAI supports strict Structured Outputs -> SDK .parse() with a Pydantic model.
    "openai": ProviderSpec("openai", "OPENAI_API_KEY", None, "gpt-4o-mini", "parse"),
    # DeepSeek supports JSON mode (json_object); the schema goes into the system prompt.
    "deepseek": ProviderSpec("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat", "json_object"),
    # Groq: JSON mode works for all chat models; json_schema only for some.
    "groq": ProviderSpec("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "json_object"),
}


class ProviderUnavailable(RuntimeError):
    """Raised when a provider is requested but its API key is not configured."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str | None
    model: str
    response_mode: str


def load_dotenv(path: str | Path = ".env") -> int:
    """Minimal ``.env`` loader (KEY=VALUE lines); never overrides existing variables."""
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    with open(p, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                n += 1
    return n


def available_providers() -> list[str]:
    """Providers whose API key is present in the environment."""
    return [name for name, spec in PROVIDERS.items() if os.environ.get(spec.key_env)]


def resolve_provider(name: str) -> ProviderConfig:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    spec = PROVIDERS[name]
    key = os.environ.get(spec.key_env)
    if not key:
        raise ProviderUnavailable(f"{spec.key_env} is not set; cannot use provider '{name}'")
    prefix = name.upper()
    mode = os.environ.get(f"{prefix}_RESPONSE_MODE", spec.response_mode)
    if mode not in RESPONSE_MODES:
        raise ValueError(f"{prefix}_RESPONSE_MODE must be one of {RESPONSE_MODES}")
    return ProviderConfig(
        name=name,
        api_key=key,
        base_url=os.environ.get(f"{prefix}_BASE_URL", spec.base_url),
        model=os.environ.get(f"{prefix}_MODEL", spec.default_model),
        response_mode=mode,
    )


def make_client(cfg: ProviderConfig, timeout: float = 60.0):
    """Instantiate an ``openai.OpenAI`` client pointed at the provider's endpoint."""
    from openai import OpenAI

    kwargs: dict = {"api_key": cfg.api_key, "max_retries": 2, "timeout": timeout}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    LOG.info("LLM provider=%s model=%s mode=%s", cfg.name, cfg.model, cfg.response_mode)
    return OpenAI(**kwargs)
