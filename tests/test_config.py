from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from hailmary.config import AppConfig, ConfigError, create_local_state, load_config


def test_init_adds_repo_local_custom_data_dir_to_local_git_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    git_info = tmp_path / ".git" / "info"
    git_info.mkdir(parents=True)
    exclude_path = git_info / "exclude"
    exclude_path.write_text("# local excludes\n", encoding="utf-8")

    create_local_state(AppConfig(data_dir=Path("local-data")), force=True)

    exclude_text = exclude_path.read_text(encoding="utf-8")
    assert "local-data/" in exclude_text


def test_local_state_uses_owner_only_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    create_local_state(AppConfig(data_dir=Path("local-data")), force=True)

    assert stat.S_IMODE((tmp_path / "local-data").stat().st_mode) == 0o700
    assert (tmp_path / "local-data" / "browser-profiles" / "meridian").is_dir()
    assert stat.S_IMODE((tmp_path / ".hailmary").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / ".hailmary" / "config.yaml").stat().st_mode) == 0o600


def test_local_state_rejects_data_dir_inside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)

    with pytest.raises(ConfigError, match="cannot be inside .git"):
        create_local_state(AppConfig(data_dir=Path(".git/hailmary")), force=True)


def test_local_state_rejects_current_folder_outside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="cannot be the current folder"):
        create_local_state(AppConfig(data_dir=Path(".")), force=True)


def test_local_state_rejects_existing_shared_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    (shared_dir / "other-file.txt").write_text("not Hail Mary state", encoding="utf-8")

    with pytest.raises(ConfigError, match="already contains other files"):
        create_local_state(AppConfig(data_dir=shared_dir), force=True)


def test_git_exclude_patterns_are_escaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    git_info = tmp_path / ".git" / "info"
    git_info.mkdir(parents=True)
    exclude_path = git_info / "exclude"
    exclude_path.write_text("# local excludes\n", encoding="utf-8")

    create_local_state(AppConfig(data_dir=Path("#data")), force=True)
    create_local_state(AppConfig(data_dir=Path("!data")), force=True)

    exclude_text = exclude_path.read_text(encoding="utf-8")
    assert "\\#data/" in exclude_text
    assert "\\!data/" in exclude_text


def test_git_exclude_read_error_has_clear_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    exclude_path = tmp_path / ".git" / "info" / "exclude"
    exclude_path.mkdir(parents=True)

    with pytest.raises(ConfigError, match="Could not read local Git exclude file"):
        create_local_state(AppConfig(data_dir=Path("local-data")), force=True)


def test_init_uses_git_exclude_path_in_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    git_dir = tmp_path / "actual-gitdir"
    git_info = git_dir / "info"
    git_info.mkdir(parents=True)
    exclude_path = git_info / "exclude"
    exclude_path.write_text("# local excludes\n", encoding="utf-8")
    (tmp_path / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    def fake_run(
        args: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        if args[3:] == ["rev-parse", "--git-path", "info/exclude"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{exclude_path}\n", stderr="")
        if args[3] == "ls-files":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected command")

    monkeypatch.setattr(subprocess, "run", fake_run)

    create_local_state(AppConfig(data_dir=Path("local-data")), force=True)

    exclude_text = exclude_path.read_text(encoding="utf-8")
    assert "local-data/" in exclude_text


def test_load_config_reads_saved_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: local-data",
                "local_only: true",
                "log_level: INFO",
                "capital_budget: 100000",
                "min_check: 1000",
                "max_check: 10000",
                "meridian_profile_dir: local-data/browser-profiles/meridian",
                "enable_web_research: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.data_dir == Path("local-data")
    assert config.meridian_profile_dir == Path("local-data/browser-profiles/meridian")


def test_custom_data_dir_derives_default_meridian_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config(data_dir=Path("local-data"))

    assert config.meridian_profile_dir == Path("local-data/browser-profiles/meridian")


def test_configured_paths_expand_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    config = load_config(data_dir=Path("~/hm-data"))

    assert config.data_dir == home / "hm-data"
    assert config.meridian_profile_dir == home / "hm-data" / "browser-profiles" / "meridian"


def test_init_expands_home_directory_before_creating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    result = create_local_state(
        AppConfig(
            data_dir=Path("~/hm-data"),
            meridian_profile_dir=Path("~/hm-profile"),
        ),
        force=True,
    )

    assert result.data_dir == home / "hm-data"
    assert (home / "hm-data").is_dir()
    assert (home / "hm-profile").is_dir()


def test_init_creates_custom_meridian_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    create_local_state(
        AppConfig(data_dir=Path("local-data"), meridian_profile_dir=Path("local-profile")),
        force=True,
    )

    assert (tmp_path / "local-profile").is_dir()
    assert stat.S_IMODE((tmp_path / "local-profile").stat().st_mode) == 0o700


def test_max_check_above_allowed_tier_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "25000")

    with pytest.raises(ConfigError, match="maximum check size cannot be above \\$10K"):
        load_config()


def test_init_rejects_max_check_above_allowed_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="maximum check size cannot be above \\$10K"):
        create_local_state(AppConfig(max_check=25_000), force=True)


def test_non_tier_check_sizes_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "6000")

    with pytest.raises(ConfigError, match="maximum check size must be one of"):
        load_config()


def test_min_check_cannot_exceed_max_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="minimum check size cannot be higher"):
        create_local_state(AppConfig(min_check=10_000, max_check=1_000), force=True)


def test_invalid_boolean_env_value_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "treu")

    with pytest.raises(ConfigError, match="must be true or false"):
        load_config()


def test_invalid_numeric_env_value_has_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "ten")

    with pytest.raises(ConfigError, match="must be a whole number"):
        load_config()


def test_numeric_env_value_overrides_invalid_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("max_check: ten\n", encoding="utf-8")
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "7500")

    config = load_config()

    assert config.max_check == 7500


def test_local_state_rejects_file_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local-data").write_text("not a folder", encoding="utf-8")

    with pytest.raises(ConfigError, match="needs local-data to be a folder"):
        create_local_state(AppConfig(data_dir=Path("local-data")), force=True)


def test_local_state_rejects_config_path_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".hailmary" / "config.yaml"
    config_path.mkdir(parents=True)

    with pytest.raises(ConfigError, match="needs .hailmary/config.yaml to be a file"):
        create_local_state(AppConfig(data_dir=Path("data")), force=True)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_local_state_rejects_symlinked_config_file_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    target = tmp_path / "README.md"
    target.write_text("do not overwrite", encoding="utf-8")
    (config_dir / "config.yaml").symlink_to(target)

    with pytest.raises(ConfigError, match="real file, not a symlink"):
        create_local_state(AppConfig(data_dir=Path("data")), force=True)

    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_load_config_rejects_config_path_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".hailmary" / "config.yaml"
    config_path.mkdir(parents=True)

    with pytest.raises(ConfigError, match="needs .hailmary/config.yaml to be a file"):
        load_config()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_load_config_rejects_symlinked_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    target = tmp_path / "other-config.yaml"
    target.write_text("data_dir: local-data\n", encoding="utf-8")
    (config_dir / "config.yaml").symlink_to(target)

    with pytest.raises(ConfigError, match="real file, not a symlink"):
        load_config()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_load_config_rejects_broken_symlinked_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").symlink_to(tmp_path / "missing-config.yaml")

    with pytest.raises(ConfigError, match="real file, not a symlink"):
        load_config()
