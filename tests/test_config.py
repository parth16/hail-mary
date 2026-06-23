from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from hailmary.config import (
    MERIDIAN_PROFILE_MARKER,
    AppConfig,
    ConfigError,
    create_local_state,
    load_config,
)


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_init_adds_repo_local_custom_data_dir_to_local_git_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _init_git_repo(tmp_path)
    exclude_path = tmp_path / ".git" / "info" / "exclude"
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
    assert (
        tmp_path / "local-data" / "browser-profiles" / "meridian" / MERIDIAN_PROFILE_MARKER
    ).is_file()
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
    _init_git_repo(tmp_path)
    exclude_path = tmp_path / ".git" / "info" / "exclude"
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
    _init_git_repo(tmp_path)
    exclude_path = tmp_path / ".git" / "info" / "exclude"
    exclude_path.unlink()
    exclude_path.mkdir(parents=True)

    with pytest.raises(ConfigError, match="Could not read local Git exclude file"):
        create_local_state(AppConfig(data_dir=Path("local-data")), force=True)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_git_exclude_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _init_git_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("do not edit\n", encoding="utf-8")
    exclude_path = tmp_path / ".git" / "info" / "exclude"
    exclude_path.unlink()
    exclude_path.symlink_to(readme)

    with pytest.raises(ConfigError, match="local Git exclude file.*cannot be a symlink"):
        create_local_state(AppConfig(data_dir=Path("local-data")), force=True)

    assert readme.read_text(encoding="utf-8") == "do not edit\n"


def test_tracked_file_check_error_has_clear_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)

    def fake_run(
        args: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        if args[3] == "ls-files":
            return subprocess.CompletedProcess(
                args,
                128,
                stdout="",
                stderr="Git could not inspect tracked files.",
            )
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected command")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ConfigError, match="Could not check whether local-data overlaps"):
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


def test_load_config_uses_repo_root_from_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (repo_root / ".git" / "info").mkdir(parents=True)
    config_dir = repo_root / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: local-data",
                "local_only: true",
                "meridian_profile_dir: local-data/browser-profiles/meridian",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(src_dir)

    config = load_config()

    assert config.data_dir == repo_root / "local-data"
    assert config.meridian_profile_dir == repo_root / "local-data/browser-profiles/meridian"


def test_load_config_finds_saved_config_from_subdirectory_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deals_root = tmp_path / "deals"
    company_dir = deals_root / "Acme"
    company_dir.mkdir(parents=True)
    config_dir = deals_root / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: local-data",
                "local_only: true",
                "meridian_profile_dir: local-data/browser-profiles/meridian",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(company_dir)

    config = load_config()

    assert config.data_dir == deals_root / "local-data"
    assert config.meridian_profile_dir == deals_root / "local-data/browser-profiles/meridian"


def test_blank_data_dir_env_uses_saved_project_path_from_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    (repo_root / ".git" / "info").mkdir(parents=True)
    config_dir = repo_root / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: local-data",
                "local_only: true",
                "meridian_profile_dir: local-data/browser-profiles/meridian",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(src_dir)
    monkeypatch.setenv("HAILMARY_DATA_DIR", "")

    config = load_config()

    assert config.data_dir == repo_root / "local-data"
    assert config.meridian_profile_dir == repo_root / "local-data/browser-profiles/meridian"


def test_blank_numeric_env_uses_saved_config_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: local-data",
                "local_only: true",
                "capital_budget: 25000",
                "min_check: 1000",
                "max_check: 5000",
                "meridian_profile_dir: local-data/browser-profiles/meridian",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "")

    config = load_config()

    assert config.max_check == 5_000


def test_data_dir_override_rebases_saved_default_meridian_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    secure_data = tmp_path / "secure-data"
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                "data_dir: data",
                "local_only: true",
                "meridian_profile_dir: data/browser-profiles/meridian",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(data_dir=secure_data)

    assert config.data_dir == secure_data
    assert config.meridian_profile_dir == secure_data / "browser-profiles" / "meridian"


def test_init_from_subdirectory_anchors_explicit_relative_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True)
    _init_git_repo(repo_root)
    monkeypatch.chdir(src_dir)

    config = load_config(data_dir=Path("local-data"), ignore_saved=True)
    create_local_state(config, force=True)

    assert (repo_root / "local-data" / "processed").is_dir()
    assert not (src_dir / "local-data").exists()
    config_text = (repo_root / ".hailmary" / "config.yaml").read_text(encoding="utf-8")
    assert f"data_dir: {(repo_root / 'local-data').as_posix()}" in config_text


def test_init_ignores_data_dir_in_target_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_repo = tmp_path / "current"
    target_repo = tmp_path / "target"
    current_repo.mkdir()
    target_repo.mkdir()
    _init_git_repo(current_repo)
    _init_git_repo(target_repo)
    target_exclude = target_repo / ".git" / "info" / "exclude"
    target_exclude.write_text("# target local excludes\n", encoding="utf-8")
    monkeypatch.chdir(current_repo)

    create_local_state(AppConfig(data_dir=target_repo / "local-data"), force=True)

    assert "local-data/" in target_exclude.read_text(encoding="utf-8")


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
    assert (tmp_path / "local-profile" / MERIDIAN_PROFILE_MARKER).is_file()
    assert stat.S_IMODE((tmp_path / "local-profile").stat().st_mode) == 0o700


def test_init_rejects_existing_shared_meridian_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shared_dir = tmp_path / "shared-profile"
    shared_dir.mkdir()
    shared_dir.chmod(0o755)
    (shared_dir / "other-file.txt").write_text("not Hail Mary state", encoding="utf-8")

    with pytest.raises(ConfigError, match="already contains other files"):
        create_local_state(
            AppConfig(
                data_dir=Path("local-data"),
                meridian_profile_dir=shared_dir,
            ),
            force=True,
        )

    assert stat.S_IMODE(shared_dir.stat().st_mode) == 0o755


def test_init_allows_existing_marked_meridian_profile_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_dir = tmp_path / "existing-profile"
    profile_dir.mkdir()
    (profile_dir / MERIDIAN_PROFILE_MARKER).write_text("marker", encoding="utf-8")
    (profile_dir / "browser-cookie-store").write_text("local browser state", encoding="utf-8")

    create_local_state(
        AppConfig(data_dir=Path("local-data"), meridian_profile_dir=profile_dir),
        force=True,
    )

    assert (profile_dir / "browser-cookie-store").exists()
    assert stat.S_IMODE(profile_dir.stat().st_mode) == 0o700


def test_init_rejects_meridian_profile_reserved_data_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    for profile_dir in [Path("local-data"), Path("local-data/raw")]:
        with pytest.raises(ConfigError, match="Meridian browser profile directory cannot"):
            create_local_state(
                AppConfig(data_dir=Path("local-data"), meridian_profile_dir=profile_dir),
                force=True,
            )


def test_init_rejects_meridian_profile_that_contains_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="cannot contain the data directory"):
        create_local_state(
            AppConfig(
                data_dir=Path("local-state/data"),
                meridian_profile_dir=Path("local-state"),
            ),
            force=True,
        )


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


def test_local_only_disables_web_research_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "true")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")

    config = load_config()

    assert config.local_only is True
    assert config.enable_web_research is False


def test_local_only_disables_web_research_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "local_only: true\nenable_web_research: true\n",
        encoding="utf-8",
    )

    config = load_config()

    assert config.local_only is True
    assert config.enable_web_research is False


def test_init_persists_web_research_off_in_local_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    create_local_state(
        AppConfig(
            data_dir=Path("local-data"),
            local_only=True,
            enable_web_research=True,
        ),
        force=True,
    )

    config_text = (tmp_path / ".hailmary" / "config.yaml").read_text(encoding="utf-8")
    assert "enable_web_research: false" in config_text


def test_local_state_rejects_file_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local-data").write_text("not a folder", encoding="utf-8")

    with pytest.raises(ConfigError, match="needs local-data to be a folder"):
        create_local_state(AppConfig(data_dir=Path("local-data")), force=True)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_local_state_rejects_symlinked_parent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    outside_parent = tmp_path / "outside-parent"
    outside_parent.mkdir()
    (tmp_path / "link").symlink_to(outside_parent, target_is_directory=True)

    with pytest.raises(ConfigError, match="symlinked parent folder"):
        create_local_state(AppConfig(data_dir=Path("link/local-data")), force=True)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_local_state_rejects_nested_symlinked_parent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    outside_parent = tmp_path / "outside-parent"
    (outside_parent / "existing").mkdir(parents=True)
    (tmp_path / "link").symlink_to(outside_parent, target_is_directory=True)

    with pytest.raises(ConfigError, match="symlinked parent folder"):
        create_local_state(AppConfig(data_dir=Path("link/existing/local-data")), force=True)


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
