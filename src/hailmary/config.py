from __future__ import annotations

import json
import os
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ConfigError(ValueError):
    """Configuration could not be used safely."""


CHECK_SIZE_TIERS = (0, 1_000, 2_500, 5_000, 7_500, 10_000)
MAX_CHECK_SIZE = max(CHECK_SIZE_TIERS)
CHECK_SIZE_TIER_TEXT = "$0, $1K, $2.5K, $5K, $7.5K, or $10K"
MAX_PORTFOLIO_DECIMAL_FIXED_CHARS = 120
MERIDIAN_PROFILE_MARKER = ".hailmary-profile"


class AppConfig(BaseModel):
    """Runtime settings for local Hail Mary commands."""

    data_dir: Path = Field(default=Path("./data"))
    local_only: bool = True
    log_level: str = "INFO"
    capital_budget: int = 100_000
    min_check: int = 1_000
    max_check: int = 10_000
    reserve_percent: Decimal = Field(default=Decimal("0"))
    reserve_dollars: int = 0
    estimated_dilution_percent: Decimal = Field(default=Decimal("0"))
    platform_fee_percent: Decimal = Field(default=Decimal("0"))
    carry_percent: Decimal = Field(default=Decimal("0"))
    gross_return_multiple: Decimal = Field(default=Decimal("5"))
    meridian_profile_dir: Path = Field(default=Path("./data/browser-profiles/meridian"))
    enable_ocr: bool = False
    enable_web_research: bool = False
    mock_llm: bool = True

    @property
    def config_dir(self) -> Path:
        project_config_dir = _project_root() / ".hailmary"
        return _display_path_from_cwd(project_config_dir)

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
    return _parse_bool(value, source=name)


def _config_bool(values: dict[str, str], name: str, default: bool) -> bool:
    value = values.get(name)
    if value is None:
        return default
    return _parse_bool(value, source=f".hailmary/config.yaml field {name}")


def _parse_bool(value: str, *, source: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(
        f"{source} must be true or false. Got {value!r}. "
        "Hail Mary did not guess because privacy settings should fail closed."
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return _parse_int(value, source=name)


def _config_int(values: dict[str, str], name: str, default: int) -> int:
    value = values.get(name)
    if value is None or value.strip() == "":
        return default
    return _parse_int(value, source=f".hailmary/config.yaml field {name}")


def _setting_int(
    *,
    env_name: str,
    config_values: dict[str, str],
    config_name: str,
    default: int,
) -> int:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip() != "":
        return _env_int(env_name, default)
    return _config_int(config_values, config_name, default)


def _env_decimal(name: str, default: Decimal) -> Decimal:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return _parse_decimal(value, source=name)


def _config_decimal(values: dict[str, str], name: str, default: Decimal) -> Decimal:
    value = values.get(name)
    if value is None or value.strip() == "":
        return default
    return _parse_decimal(value, source=f".hailmary/config.yaml field {name}")


def _setting_decimal(
    *,
    env_name: str,
    config_values: dict[str, str],
    config_name: str,
    default: Decimal,
) -> Decimal:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip() != "":
        return _env_decimal(env_name, default)
    return _config_decimal(config_values, config_name, default)


def _parse_int(value: str, *, source: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{source} must be a whole number. Got {value!r}. "
            "Hail Mary did not guess because investment limits should be explicit."
        ) from exc


def _parse_decimal(value: str, *, source: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ConfigError(
            f"{source} must be a number. Got {value!r}. "
            "Hail Mary did not guess because portfolio assumptions should be explicit."
        ) from exc
    if not parsed.is_finite():
        raise ConfigError(
            f"{source} must be a finite number. Got {value!r}. "
            "Hail Mary did not guess because portfolio assumptions should be explicit."
        )
    return parsed


def load_config(data_dir: Path | None = None, *, ignore_saved: bool = False) -> AppConfig:
    """Load config from environment variables and command options."""

    saved_values = {} if ignore_saved else _read_local_config(AppConfig().config_path)
    env_data_dir = os.getenv("HAILMARY_DATA_DIR")
    data_dir_was_overridden = data_dir is not None or (
        env_data_dir is not None and env_data_dir.strip() != ""
    )
    if data_dir is not None:
        raw_data_dir = data_dir
    elif env_data_dir is not None and env_data_dir.strip() != "":
        raw_data_dir = Path(env_data_dir)
    else:
        raw_data_dir = Path(saved_values.get("data_dir", "./data"))
    resolved_data_dir = _expand_project_path(raw_data_dir)
    local_only = (
        _env_bool("HAILMARY_LOCAL_ONLY", True)
        if "HAILMARY_LOCAL_ONLY" in os.environ
        else _config_bool(saved_values, "local_only", True)
    )
    enable_web_research = (
        _env_bool("HAILMARY_ENABLE_WEB_RESEARCH", False)
        if "HAILMARY_ENABLE_WEB_RESEARCH" in os.environ
        else _config_bool(saved_values, "enable_web_research", False)
    )
    enable_ocr = (
        _env_bool("HAILMARY_ENABLE_OCR", False)
        if "HAILMARY_ENABLE_OCR" in os.environ
        else _config_bool(saved_values, "enable_ocr", False)
    )
    if local_only:
        enable_web_research = False
    mock_llm = (
        _env_bool("HAILMARY_MOCK_LLM", True)
        if "HAILMARY_MOCK_LLM" in os.environ
        else _config_bool(saved_values, "mock_llm", True)
    )

    config = AppConfig(
        data_dir=resolved_data_dir,
        local_only=local_only,
        log_level=os.getenv("HAILMARY_LOG_LEVEL", saved_values.get("log_level", "INFO")),
        capital_budget=_setting_int(
            env_name="HAILMARY_CAPITAL_BUDGET",
            config_values=saved_values,
            config_name="capital_budget",
            default=100_000,
        ),
        min_check=_setting_int(
            env_name="HAILMARY_MIN_CHECK",
            config_values=saved_values,
            config_name="min_check",
            default=1_000,
        ),
        max_check=_setting_int(
            env_name="HAILMARY_MAX_CHECK",
            config_values=saved_values,
            config_name="max_check",
            default=10_000,
        ),
        reserve_percent=_setting_decimal(
            env_name="HAILMARY_RESERVE_PERCENT",
            config_values=saved_values,
            config_name="reserve_percent",
            default=Decimal("0"),
        ),
        reserve_dollars=_setting_int(
            env_name="HAILMARY_RESERVE_DOLLARS",
            config_values=saved_values,
            config_name="reserve_dollars",
            default=0,
        ),
        estimated_dilution_percent=_setting_decimal(
            env_name="HAILMARY_ESTIMATED_DILUTION_PERCENT",
            config_values=saved_values,
            config_name="estimated_dilution_percent",
            default=Decimal("0"),
        ),
        platform_fee_percent=_setting_decimal(
            env_name="HAILMARY_PLATFORM_FEE_PERCENT",
            config_values=saved_values,
            config_name="platform_fee_percent",
            default=Decimal("0"),
        ),
        carry_percent=_setting_decimal(
            env_name="HAILMARY_CARRY_PERCENT",
            config_values=saved_values,
            config_name="carry_percent",
            default=Decimal("0"),
        ),
        gross_return_multiple=_setting_decimal(
            env_name="HAILMARY_GROSS_RETURN_MULTIPLE",
            config_values=saved_values,
            config_name="gross_return_multiple",
            default=Decimal("5"),
        ),
        meridian_profile_dir=_meridian_profile_dir(
            resolved_data_dir,
            saved_values,
            data_dir_was_overridden=data_dir_was_overridden,
        ),
        enable_ocr=enable_ocr,
        enable_web_research=enable_web_research,
        mock_llm=mock_llm,
    )
    return validate_investment_settings(config)


def validate_investment_settings(config: AppConfig) -> AppConfig:
    """Validate deterministic investment and portfolio assumptions."""

    _ensure_investment_limits(config)
    return config


def validate_local_state(config: AppConfig) -> AppConfig:
    """Validate generated-output paths without creating local folders."""

    config = _expand_config_paths(config)
    validate_investment_settings(config)
    _ensure_generated_path_not_current_or_root(config.data_dir, purpose="data directory")
    _ensure_data_dir_not_config_dir(config)
    _ensure_dedicated_data_dir(config.data_dir)
    _ensure_generated_path_not_current_or_root(
        config.meridian_profile_dir, purpose="Meridian browser profile directory"
    )
    _ensure_meridian_profile_not_reserved_data_path(config)
    _ensure_meridian_profile_not_config_path(config)
    _ensure_dedicated_meridian_profile_dir(config.meridian_profile_dir)
    _ensure_folder_path(config.data_dir)
    _ensure_folder_path(config.meridian_profile_dir)
    _ensure_folder_path(config.config_dir)
    _ensure_config_file_path(config.config_path)
    _ensure_repo_local_path_ignored(config.data_dir, purpose="data directory")
    _ensure_repo_local_path_ignored(config.config_dir, purpose="local config directory")
    _ensure_repo_local_path_ignored(
        config.meridian_profile_dir, purpose="Meridian browser profile directory"
    )
    return config


def create_local_state(config: AppConfig, *, force: bool) -> InitResult:
    """Create local folders used for generated output."""

    config = validate_local_state(config)

    folders = [
        config.data_dir,
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.data_dir / "reports",
        config.data_dir / "agent-packets",
        config.data_dir / "agent-outputs",
        config.data_dir / "research-plans",
        config.data_dir / "research-results",
        config.data_dir / "browser-profiles",
        config.data_dir / "meridian-workflows",
        config.meridian_profile_dir,
        config.config_dir,
    ]
    for folder in folders:
        _ensure_folder_path(folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            folder.chmod(0o700)
        except OSError as exc:
            raise ConfigError(f"Could not create folder at {folder}: {exc}") from exc

    _write_meridian_profile_marker(config.meridian_profile_dir)
    config_created = force or not config.config_path.exists()
    if config_created:
        try:
            config.config_path.write_text(_default_config_text(config), encoding="utf-8")
            config.config_path.chmod(0o600)
        except OSError as exc:
            raise ConfigError(
                f"Could not write local config at {config.config_path}: {exc}"
            ) from exc

    return InitResult(
        data_dir=config.data_dir,
        config_path=config.config_path,
        config_created=config_created,
    )


def _meridian_profile_dir(
    data_dir: Path,
    saved_values: dict[str, str],
    *,
    data_dir_was_overridden: bool,
) -> Path:
    env_value = os.getenv("HAILMARY_MERIDIAN_PROFILE_DIR")
    if env_value is not None and env_value.strip():
        return _expand_project_path(Path(env_value))

    saved_value = saved_values.get("meridian_profile_dir")
    if saved_value:
        saved_profile_dir = Path(saved_value)
        if data_dir_was_overridden and _is_saved_default_profile_dir(
            saved_profile_dir,
            saved_values,
        ):
            return data_dir / "browser-profiles" / "meridian"
        return _expand_project_path(saved_profile_dir)

    return data_dir / "browser-profiles" / "meridian"


def _is_saved_default_profile_dir(
    saved_profile_dir: Path,
    saved_values: dict[str, str],
) -> bool:
    saved_data_dir = Path(saved_values.get("data_dir", "./data"))
    return saved_profile_dir == saved_data_dir / "browser-profiles" / "meridian"


def _expand_config_paths(config: AppConfig) -> AppConfig:
    default_config = AppConfig()
    meridian_profile_dir = config.meridian_profile_dir
    if (
        config.data_dir != default_config.data_dir
        and config.meridian_profile_dir == default_config.meridian_profile_dir
    ):
        meridian_profile_dir = config.data_dir / "browser-profiles" / "meridian"

    return config.model_copy(
        update={
            "data_dir": _expand_project_path(config.data_dir),
            "meridian_profile_dir": _expand_project_path(meridian_profile_dir),
            "enable_web_research": (
                False if config.local_only else config.enable_web_research
            ),
        }
    )


def _expand_project_path(path: Path) -> Path:
    expanded_path = path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path

    project_root = _project_root()
    current_dir = Path.cwd().resolve(strict=False)
    if project_root == current_dir:
        return expanded_path
    return project_root / expanded_path


def _project_root() -> Path:
    return (
        _find_git_root(Path.cwd())
        or _find_config_root(Path.cwd())
        or Path.cwd().resolve(strict=False)
    )


def _find_config_root(start: Path) -> Path | None:
    current = start.resolve(strict=False)
    for candidate in [current, *current.parents]:
        config_path = candidate / ".hailmary" / "config.yaml"
        if config_path.exists() or config_path.is_symlink():
            return candidate
    return None


def _display_path_from_cwd(path: Path) -> Path:
    current_dir = Path.cwd().resolve(strict=False)
    try:
        return Path(os.path.relpath(path, current_dir))
    except ValueError:
        return path


def _ensure_investment_limits(config: AppConfig) -> None:
    if config.capital_budget < 0:
        raise ConfigError("The capital budget cannot be negative.")
    if config.reserve_dollars < 0:
        raise ConfigError("The reserve dollars cannot be negative.")
    if config.reserve_dollars > config.capital_budget:
        raise ConfigError("The reserve dollars cannot be higher than the capital budget.")
    if config.max_check > MAX_CHECK_SIZE:
        raise ConfigError("The maximum check size cannot be above $10K.")
    if config.max_check not in CHECK_SIZE_TIERS:
        raise ConfigError(f"The maximum check size must be one of {CHECK_SIZE_TIER_TEXT}.")
    if config.min_check not in CHECK_SIZE_TIERS:
        raise ConfigError(f"The minimum check size must be one of {CHECK_SIZE_TIER_TEXT}.")
    if config.min_check > config.max_check:
        raise ConfigError("The minimum check size cannot be higher than the maximum check size.")
    if config.reserve_percent > 0 and config.reserve_dollars > 0:
        raise ConfigError(
            "Use either reserve percent or reserve dollars, not both. "
            "Hail Mary did not guess which reserve should control the portfolio budget."
        )
    _ensure_percent(config.reserve_percent, name="reserve percent")
    _ensure_percent(config.estimated_dilution_percent, name="estimated dilution percent")
    _ensure_percent(config.platform_fee_percent, name="platform fee percent")
    _ensure_percent(config.carry_percent, name="carry percent")
    if not config.gross_return_multiple.is_finite():
        raise ConfigError("The gross return multiple must be a finite number.")
    _ensure_bounded_decimal(
        config.gross_return_multiple,
        name="gross return multiple",
    )
    if config.gross_return_multiple < 0:
        raise ConfigError("The gross return multiple cannot be negative.")


def _ensure_percent(value: Decimal, *, name: str) -> None:
    if not value.is_finite():
        raise ConfigError(f"The {name} must be a finite number.")
    _ensure_bounded_decimal(value, name=name)
    if value < 0 or value > 100:
        raise ConfigError(f"The {name} must be between 0 and 100.")


def _ensure_bounded_decimal(value: Decimal, *, name: str) -> None:
    if _fixed_decimal_text_length(value) <= MAX_PORTFOLIO_DECIMAL_FIXED_CHARS:
        return
    raise ConfigError(
        f"The {name} is too long to use as a portfolio assumption. "
        "Use a simpler number with at most "
        f"{MAX_PORTFOLIO_DECIMAL_FIXED_CHARS} fixed-point digits."
    )


def _fixed_decimal_text_length(value: Decimal) -> int:
    if value == 0:
        return 1
    value_tuple = value.as_tuple()
    exponent = value_tuple.exponent
    if not isinstance(exponent, int):
        return MAX_PORTFOLIO_DECIMAL_FIXED_CHARS + 1
    digit_count = len(value_tuple.digits)
    if exponent >= 0:
        return digit_count + exponent
    integer_digits = max(value.adjusted() + 1, 1)
    return integer_digits + 1 + abs(exponent)


def _read_local_config(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ConfigError(f"Hail Mary needs {path} to be a real file, not a symlink.")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigError(f"Hail Mary needs {path} to be a file, but it is a folder.")

    values: dict[str, str] = {}
    try:
        config_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read local config at {path}: {exc}") from exc

    try:
        raw_values = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse local config at {path}: {exc}") from exc

    if raw_values is None:
        return {}
    if not isinstance(raw_values, dict):
        raise ConfigError(
            f"Hail Mary needs {path} to contain simple key-value settings."
        )

    for key, value in raw_values.items():
        if not isinstance(key, str):
            raise ConfigError(f"Hail Mary needs {path} setting names to be plain text.")
        normalized_value = _config_scalar_to_string(value, path=path, key=key)
        if normalized_value is None:
            continue
        values[key.strip()] = normalized_value
    return values


def _config_scalar_to_string(value: Any, *, path: Path, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    raise ConfigError(
        f"Hail Mary needs {path} field {key} to be a simple text, number, or true/false value."
    )


def _ensure_generated_path_not_current_or_root(path: Path, *, purpose: str) -> None:
    resolved_path = path if path.is_absolute() else Path.cwd() / path
    resolved_path = resolved_path.resolve(strict=False)
    current_dir = Path.cwd().resolve(strict=False)

    if resolved_path == current_dir:
        raise ConfigError(
            f"The {purpose} cannot be the current folder. Choose a generated-data folder."
        )
    if resolved_path.parent == resolved_path:
        raise ConfigError(
            f"The {purpose} cannot be the filesystem root. Choose a generated-data folder."
        )


def _ensure_data_dir_not_config_dir(config: AppConfig) -> None:
    data_dir = _absolute_resolved_path(config.data_dir)
    config_dir = _absolute_resolved_path(config.config_dir)
    config_path = _absolute_resolved_path(config.config_path)
    if _paths_overlap(data_dir, config_dir) or _paths_overlap(data_dir, config_path):
        raise ConfigError(
            "The data directory cannot overlap the local config directory. "
            "Choose a separate generated-data folder."
        )


def _ensure_dedicated_data_dir(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError("The data directory cannot be a symlink. Choose a real folder.")
    if not path.exists() or not path.is_dir():
        return

    allowed_entries = {
        "raw",
        "processed",
        "reports",
        "agent-packets",
        "agent-outputs",
        "research-plans",
        "research-results",
        "research-results-templates",
        "browser-profiles",
        "meridian-workflows",
    }
    harmless_metadata_entries = {".DS_Store"}
    try:
        unknown_entries = (
            {child.name for child in path.iterdir()}
            - allowed_entries
            - harmless_metadata_entries
        )
    except OSError as exc:
        raise ConfigError(f"Could not inspect data directory at {path}: {exc}") from exc
    if unknown_entries:
        raise ConfigError(
            f"The data directory at {path} already contains other files. "
            "Choose a new generated-data folder or an existing Hail Mary data folder."
        )


def _ensure_meridian_profile_not_reserved_data_path(config: AppConfig) -> None:
    data_dir = _absolute_resolved_path(config.data_dir)
    profile_dir = _absolute_resolved_path(config.meridian_profile_dir)
    reserved_paths = {
        data_dir,
        data_dir / "raw",
        data_dir / "processed",
        data_dir / "reports",
        data_dir / "agent-packets",
        data_dir / "agent-outputs",
        data_dir / "research-plans",
        data_dir / "research-results",
        data_dir / "research-results-templates",
        data_dir / "meridian-workflows",
    }
    allowed_profile_root = data_dir / "browser-profiles"

    try:
        data_dir.relative_to(profile_dir)
    except ValueError:
        pass
    else:
        raise ConfigError(
            "The Meridian browser profile directory cannot contain the data directory. "
            "Choose a separate generated-data folder."
        )

    for reserved_path in reserved_paths:
        if profile_dir == reserved_path:
            raise ConfigError(
                "The Meridian browser profile directory cannot be the data, raw, "
                "processed, reports, agent-packets, agent-outputs, research-plans, "
                "research-results, research-results-templates, or meridian-workflows folder. "
                "Choose a separate generated-data folder."
            )
        try:
            profile_dir.relative_to(reserved_path)
        except ValueError:
            continue
        try:
            profile_dir.relative_to(allowed_profile_root)
        except ValueError:
            pass
        else:
            continue
        raise ConfigError(
            "The Meridian browser profile directory cannot be inside a reserved Hail Mary "
            "data folder such as raw, processed, reports, agent-packets, agent-outputs, "
            "research-plans, research-results, research-results-templates, or "
            "meridian-workflows. "
            "Choose a separate generated-data folder."
        )


def _ensure_meridian_profile_not_config_path(config: AppConfig) -> None:
    profile_dir = _absolute_resolved_path(config.meridian_profile_dir)
    config_dir = _absolute_resolved_path(config.config_dir)
    if _paths_overlap(profile_dir, config_dir):
        raise ConfigError(
            "The Meridian browser profile directory cannot overlap the local config directory. "
            "Choose a separate generated-data folder."
        )


def _paths_overlap(first_path: Path, second_path: Path) -> bool:
    if first_path == second_path:
        return True
    try:
        first_path.relative_to(second_path)
    except ValueError:
        pass
    else:
        return True
    try:
        second_path.relative_to(first_path)
    except ValueError:
        return False
    return True


def _absolute_resolved_path(path: Path) -> Path:
    return (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)


def _ensure_dedicated_meridian_profile_dir(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError(
            "The Meridian browser profile directory cannot be a symlink. Choose a real folder."
        )
    if not path.exists():
        return
    if not path.is_dir():
        raise ConfigError(
            f"Hail Mary needs {path} to be a folder, but it is a file."
        )

    marker_path = path / MERIDIAN_PROFILE_MARKER
    try:
        existing_entries = {child.name for child in path.iterdir()}
    except OSError as exc:
        raise ConfigError(
            f"Could not inspect Meridian browser profile directory at {path}: {exc}"
        ) from exc

    if existing_entries and MERIDIAN_PROFILE_MARKER not in existing_entries:
        raise ConfigError(
            f"The Meridian browser profile directory at {path} already contains other files. "
            "Choose a new generated-data folder or an existing Hail Mary profile folder."
        )
    if marker_path.exists() and not marker_path.is_file():
        raise ConfigError(
            f"Hail Mary needs {marker_path} to be a real file, not a folder."
        )
    if marker_path.is_symlink():
        raise ConfigError(
            f"Hail Mary needs {marker_path} to be a real file, not a symlink."
        )


def _write_meridian_profile_marker(profile_dir: Path) -> None:
    marker_path = profile_dir / MERIDIAN_PROFILE_MARKER
    if marker_path.is_symlink():
        raise ConfigError(
            f"Hail Mary needs {marker_path} to be a real file, not a symlink."
        )

    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(marker_path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write("Hail Mary Meridian browser profile directory.\n")
        marker_path.chmod(0o600)
    except OSError as exc:
        raise ConfigError(
            f"Could not write Meridian browser profile marker at {marker_path}: {exc}"
        ) from exc


def _ensure_repo_local_path_ignored(path: Path, *, purpose: str) -> None:
    resolved_path = _absolute_resolved_path(path)
    git_root = _find_git_root(resolved_path) or _find_git_root(Path.cwd())
    if git_root is None:
        return

    try:
        relative_path = resolved_path.relative_to(git_root)
    except ValueError:
        return

    if relative_path == Path("."):
        raise ConfigError(
            f"The {purpose} cannot be the repository root. Choose a generated-data folder."
        )
    if ".git" in {part.lower() for part in relative_path.parts}:
        raise ConfigError(
            f"The {purpose} cannot be inside .git. Choose a generated-data folder."
        )

    relative_text = relative_path.as_posix().rstrip("/")
    if _path_has_tracked_files(git_root, relative_text):
        raise ConfigError(
            f"The {purpose} overlaps tracked project files at {relative_text}. "
            "Choose a separate generated-data folder."
        )

    _append_local_git_exclude(git_root, f"{relative_text}/")


def _find_git_root(start: Path) -> Path | None:
    current = start.resolve(strict=False)
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _path_has_tracked_files(git_root: Path, relative_text: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "ls-files", "--", relative_text],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ConfigError(
            f"Could not check whether {relative_text} overlaps tracked project files: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f" Git said: {detail}" if detail else ""
        raise ConfigError(
            f"Could not check whether {relative_text} overlaps tracked project files.{suffix}"
        )
    return bool(result.stdout.strip())


def _append_local_git_exclude(git_root: Path, pattern: str) -> None:
    exclude_path = _local_git_exclude_path(git_root)
    if exclude_path is None:
        return
    if exclude_path.is_symlink():
        raise ConfigError(
            f"The local Git exclude file at {exclude_path} cannot be a symlink."
        )
    for parent in exclude_path.parents:
        if parent == git_root.parent:
            break
        if parent.is_symlink():
            raise ConfigError(
                f"The local Git exclude path cannot use a symlinked parent folder at {parent}."
            )

    escaped_pattern = _escape_git_exclude_pattern(pattern)
    try:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    except OSError as exc:
        raise ConfigError(
            f"Could not read local Git exclude file at {exclude_path}: {exc}"
        ) from exc

    existing_patterns = {line.strip() for line in existing.splitlines()}
    if escaped_pattern in existing_patterns:
        return

    newline = "" if existing.endswith("\n") or not existing else "\n"
    try:
        exclude_path.write_text(f"{existing}{newline}{escaped_pattern}\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Could not update local Git exclude file at {exclude_path}: {exc}"
        ) from exc


def _ensure_folder_path(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError(f"Hail Mary needs {path} to be a real folder, not a symlink.")
    if path.exists() and not path.is_dir():
        raise ConfigError(f"Hail Mary needs {path} to be a folder, but it is a file.")

    for parent in path.parents:
        if parent.is_symlink():
            raise ConfigError(
                f"Hail Mary cannot create {path} because {parent} is a symlinked parent folder."
            )
        if parent.exists() and not parent.is_dir():
            raise ConfigError(
                f"Hail Mary cannot create {path} because {parent} is a file."
            )


def _ensure_config_file_path(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError(f"Hail Mary needs {path} to be a real file, not a symlink.")
    if path.exists() and not path.is_file():
        raise ConfigError(f"Hail Mary needs {path} to be a file, but it is a folder.")


def _escape_git_exclude_pattern(pattern: str) -> str:
    escaped = pattern.replace("\\", "\\\\")
    for character in ["*", "?", "[", "]"]:
        escaped = escaped.replace(character, f"\\{character}")
    if escaped.startswith(("#", "!")):
        escaped = f"\\{escaped}"
    return escaped


def _local_git_exclude_path(git_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "rev-parse", "--git-path", "info/exclude"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        result = None

    if result is not None and result.returncode == 0 and result.stdout.strip():
        path = Path(result.stdout.strip())
        return path if path.is_absolute() else git_root / path

    dot_git = git_root / ".git"
    if dot_git.is_dir():
        return dot_git / "info" / "exclude"
    return None


def _default_config_text(config: AppConfig) -> str:
    local_only = "true" if config.local_only else "false"
    enable_ocr = "true" if config.enable_ocr else "false"
    web_research = "true" if config.enable_web_research else "false"
    mock_llm = "true" if config.mock_llm else "false"

    return f"""# Local Hail Mary settings. Do not commit this file.
data_dir: {_yaml_string(config.data_dir.as_posix())}
local_only: {local_only}
log_level: {_yaml_string(config.log_level)}
capital_budget: {config.capital_budget}
min_check: {config.min_check}
max_check: {config.max_check}
reserve_percent: {_decimal_text(config.reserve_percent)}
reserve_dollars: {config.reserve_dollars}
estimated_dilution_percent: {_decimal_text(config.estimated_dilution_percent)}
platform_fee_percent: {_decimal_text(config.platform_fee_percent)}
carry_percent: {_decimal_text(config.carry_percent)}
gross_return_multiple: {_decimal_text(config.gross_return_multiple)}
meridian_profile_dir: {_yaml_string(config.meridian_profile_dir.as_posix())}
enable_ocr: {enable_ocr}
enable_web_research: {web_research}
mock_llm: {mock_llm}
"""


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _yaml_string(value: str) -> str:
    return json.dumps(value)
