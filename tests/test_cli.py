from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from hailmary.cli import app

runner = CliRunner()


def test_init_creates_local_state(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "local-data"

    result = runner.invoke(app, ["init", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Created Hail Mary local folders" in result.output
    assert (data_dir / "raw").is_dir()
    assert (data_dir / "processed").is_dir()
    assert (data_dir / "reports").is_dir()
    assert (tmp_path / ".hailmary" / "config.yaml").is_file()


def test_ingest_folder_command_writes_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "Acme memo.txt").write_text("Memo about Acme customer traction.", encoding="utf-8")
    data_dir = tmp_path / "data"

    result = runner.invoke(app, ["ingest-folder", str(source.parent), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Found 1 deal and 1 document" in result.output

    summary_path = data_dir / "processed" / "ingestion_summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["deals"][0]["company_name"] == "Acme"


def test_ingest_folder_missing_folder_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    missing_folder = tmp_path / "missing"

    result = runner.invoke(app, ["ingest-folder", str(missing_folder)])

    assert result.exit_code != 0
    assert "The folder does not exist" in result.output
    assert "Traceback" not in result.output


def test_invalid_boolean_env_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "treu")

    result = runner.invoke(app, ["init", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "must be true or false" in result.output
    assert "privacy settings should fail closed" in result.output
    assert "Traceback" not in result.output


def test_invalid_numeric_env_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "ten")

    result = runner.invoke(app, ["init", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "must be a whole number" in result.output
    assert "Traceback" not in result.output


def test_init_unsafe_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)

    result = runner.invoke(app, ["init", "--data-dir", "."])

    assert result.exit_code != 0
    assert "cannot be the repository root" in result.output
    assert "Traceback" not in result.output


def test_init_file_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local-data").write_text("not a folder", encoding="utf-8")

    result = runner.invoke(app, ["init", "--data-dir", "local-data"])

    assert result.exit_code != 0
    assert "needs local-data to be a folder" in result.output
    assert "Traceback" not in result.output


def test_ingest_folder_unsafe_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")

    result = runner.invoke(app, ["ingest-folder", str(source.parent), "--data-dir", "."])

    assert result.exit_code != 0
    assert "cannot be the repository root" in result.output
    assert "Traceback" not in result.output
