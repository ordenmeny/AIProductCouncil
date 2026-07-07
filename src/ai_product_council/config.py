from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    timeout_seconds: float = 15.0
    max_tokens: int = 220
    question_max_tokens: int = 160
    turn_max_tokens: int = 260
    enable_repair: bool = True


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        model=os.getenv("LM_STUDIO_MODEL", "local-model-name"),
        timeout_seconds=float(os.getenv("LM_STUDIO_TIMEOUT_SECONDS", "15")),
        max_tokens=int(os.getenv("LM_STUDIO_MAX_TOKENS", "220")),
        question_max_tokens=int(os.getenv("LM_STUDIO_QUESTION_MAX_TOKENS", "160")),
        turn_max_tokens=int(os.getenv("LM_STUDIO_TURN_MAX_TOKENS", "260")),
        enable_repair=_env_bool("LM_STUDIO_ENABLE_REPAIR", True),
    )
