from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Runtime settings for local Hail Mary commands."""

    data_dir: Path = Field(default=Path("./data"))
    local_only: bool = True
    log_level: str = "INFO"
    capital_budget: int = 100_000
    min_check: int = 1_000
    max_check: int = 10_000
    meridian_profile_dir: Path = Field(default=Path("./data/browser-profiles/meridian"))
    enable_web_research: bool = False

    @property
    def config_dir(self) -> Path:
        return Path(".hailmary")

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"


class InitResult(BaseModel):
    data_dir: Path
    config_path: Path
    config_created: bool


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def load_config(data_dir: Path | None = None) -> AppConfig:
    """Load config from environment variables and command options."""

    resolved_data_dir = data_dir or Path(os.getenv("HAILMARY_DATA_DIR", "./data"))
    local_only = _env_bool("HAILMARY_LOCAL_ONLY", True)

    return AppConfig(
        data_dir=resolved_data_dir,
        local_only=local_only,
        log_level=os.getenv("HAILMARY_LOG_LEVEL", "INFO"),
        capital_budget=_env_int("HAILMARY_CAPITAL_BUDGET", 100_000),
        min_check=_env_int("HAILMARY_MIN_CHECK", 1_000),
        max_check=_env_int("HAILMARY_MAX_CHECK", 10_000),
        meridian_profile_dir=Path(
            os.getenv("HAILMARY_MERIDIAN_PROFILE_DIR", "./data/browser-profiles/meridian")
        ),
        enable_web_research=_env_bool("HAILMARY_ENABLE_WEB_RESEARCH", False),
    )


def create_local_state(config: AppConfig, *, force: bool) -> InitResult:
    """Create local folders used for generated output."""

    folders = [
        config.data_dir,
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.data_dir / "reports",
        config.data_dir / "browser-profiles",
        config.config_dir,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    config_created = force or not config.config_path.exists()
    if config_created:
        config.config_path.write_text(_default_config_text(config), encoding="utf-8")

    return InitResult(
        data_dir=config.data_dir,
        config_path=config.config_path,
        config_created=config_created,
    )


def _default_config_text(config: AppConfig) -> str:
    local_only = "true" if config.local_only else "false"
    web_research = "true" if config.enable_web_research else "false"

    return f"""# Local Hail Mary settings. Do not commit this file.
data_dir: {config.data_dir.as_posix()}
local_only: {local_only}
log_level: {config.log_level}
capital_budget: {config.capital_budget}
min_check: {config.min_check}
max_check: {config.max_check}
meridian_profile_dir: {config.meridian_profile_dir.as_posix()}
enable_web_research: {web_research}
"""
