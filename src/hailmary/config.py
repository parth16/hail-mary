from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class ConfigError(ValueError):
    """Configuration could not be used safely."""


CHECK_SIZE_TIERS = (0, 1_000, 2_500, 5_000, 7_500, 10_000)
MAX_CHECK_SIZE = max(CHECK_SIZE_TIERS)
CHECK_SIZE_TIER_TEXT = "$0, $1K, $2.5K, $5K, $7.5K, or $10K"
MERIDIAN_PROFILE_MARKER = ".hailmary-profile"


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
    if env_name in os.environ:
        return _env_int(env_name, default)
    return _config_int(config_values, config_name, default)


def _parse_int(value: str, *, source: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigError(
            f"{source} must be a whole number. Got {value!r}. "
            "Hail Mary did not guess because investment limits should be explicit."
        ) from exc


def load_config(data_dir: Path | None = None, *, ignore_saved: bool = False) -> AppConfig:
    """Load config from environment variables and command options."""

    saved_values = {} if ignore_saved else _read_local_config(AppConfig().config_path)
    data_dir_is_explicit = data_dir is not None or "HAILMARY_DATA_DIR" in os.environ
    raw_data_dir = data_dir or Path(
        os.getenv("HAILMARY_DATA_DIR") or saved_values.get("data_dir", "./data")
    )
    resolved_data_dir = (
        _expand_path(raw_data_dir)
        if data_dir_is_explicit
        else _expand_project_path(raw_data_dir)
    )
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
    if local_only:
        enable_web_research = False

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
        meridian_profile_dir=_meridian_profile_dir(resolved_data_dir, saved_values),
        enable_web_research=enable_web_research,
    )
    _ensure_investment_limits(config)
    return config


def create_local_state(config: AppConfig, *, force: bool) -> InitResult:
    """Create local folders used for generated output."""

    config = _expand_config_paths(config)
    _ensure_investment_limits(config)
    _ensure_generated_path_not_current_or_root(config.data_dir, purpose="data directory")
    _ensure_dedicated_data_dir(config.data_dir)
    _ensure_generated_path_not_current_or_root(
        config.meridian_profile_dir, purpose="Meridian browser profile directory"
    )
    _ensure_meridian_profile_not_reserved_data_path(config)
    _ensure_dedicated_meridian_profile_dir(config.meridian_profile_dir)
    _ensure_folder_path(config.data_dir)
    _ensure_folder_path(config.meridian_profile_dir)
    _ensure_folder_path(config.config_dir)
    _ensure_repo_local_path_ignored(config.data_dir, purpose="data directory")
    _ensure_repo_local_path_ignored(config.config_dir, purpose="local config directory")
    _ensure_repo_local_path_ignored(
        config.meridian_profile_dir, purpose="Meridian browser profile directory"
    )

    folders = [
        config.data_dir,
        config.data_dir / "raw",
        config.data_dir / "processed",
        config.data_dir / "reports",
        config.data_dir / "browser-profiles",
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
    _ensure_config_file_path(config.config_path)
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


def _meridian_profile_dir(data_dir: Path, saved_values: dict[str, str]) -> Path:
    env_value = os.getenv("HAILMARY_MERIDIAN_PROFILE_DIR")
    if env_value is not None and env_value.strip():
        return _expand_path(Path(env_value))

    saved_value = saved_values.get("meridian_profile_dir")
    if saved_value:
        return _expand_project_path(Path(saved_value))

    return data_dir / "browser-profiles" / "meridian"


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
            "data_dir": _expand_path(config.data_dir),
            "meridian_profile_dir": _expand_path(meridian_profile_dir),
            "enable_web_research": (
                False if config.local_only else config.enable_web_research
            ),
        }
    )


def _expand_path(path: Path) -> Path:
    return path.expanduser()


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
    return _find_git_root(Path.cwd()) or Path.cwd().resolve(strict=False)


def _display_path_from_cwd(path: Path) -> Path:
    current_dir = Path.cwd().resolve(strict=False)
    try:
        return Path(os.path.relpath(path, current_dir))
    except ValueError:
        return path


def _ensure_investment_limits(config: AppConfig) -> None:
    if config.max_check > MAX_CHECK_SIZE:
        raise ConfigError("The maximum check size cannot be above $10K.")
    if config.max_check not in CHECK_SIZE_TIERS:
        raise ConfigError(f"The maximum check size must be one of {CHECK_SIZE_TIER_TEXT}.")
    if config.min_check not in CHECK_SIZE_TIERS:
        raise ConfigError(f"The minimum check size must be one of {CHECK_SIZE_TIER_TEXT}.")
    if config.min_check > config.max_check:
        raise ConfigError("The minimum check size cannot be higher than the maximum check size.")


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

    for line in config_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip()
    return values


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


def _ensure_dedicated_data_dir(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError("The data directory cannot be a symlink. Choose a real folder.")
    if not path.exists() or not path.is_dir():
        return

    allowed_entries = {"raw", "processed", "reports", "browser-profiles"}
    try:
        unknown_entries = {child.name for child in path.iterdir()} - allowed_entries
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
    }
    allowed_profile_root = data_dir / "browser-profiles"

    for reserved_path in reserved_paths:
        if profile_dir == reserved_path:
            raise ConfigError(
                "The Meridian browser profile directory cannot be the data, raw, "
                "processed, or reports folder. Choose a separate generated-data folder."
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
            "data folder. Choose a separate generated-data folder."
        )


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
    except OSError:
        return False
    return bool(result.stdout.strip())


def _append_local_git_exclude(git_root: Path, pattern: str) -> None:
    exclude_path = _local_git_exclude_path(git_root)
    if exclude_path is None:
        return

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
        if parent.exists():
            if not parent.is_dir():
                raise ConfigError(
                    f"Hail Mary cannot create {path} because {parent} is a file."
                )
            return


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
