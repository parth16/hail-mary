from __future__ import annotations

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
