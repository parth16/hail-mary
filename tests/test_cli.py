from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from hailmary.cli import app
from hailmary.ingest import folder_loader
from hailmary.ingest.extractors import ExtractionResult
from hailmary.schemas.documents import ExtractedPage, ExtractionQuality

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
    assert "source-linked evidence" in result.output

    summary_path = data_dir / "processed" / "ingestion_summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["deals"][0]["company_name"] == "Acme"


def test_ingest_folder_warns_when_documents_need_image_text_reading(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "ScanCo"
    source.mkdir(parents=True)
    (source / "ScanCo pitch deck.pdf").write_bytes(b"%PDF-1.4")

    def fake_extract_document(path: Path) -> ExtractionResult:
        assert path.name == "ScanCo pitch deck.pdf"
        page = ExtractedPage(
            page_number=1,
            raw_text="",
            clean_text="",
            needs_ocr=True,
            vision_recommended=True,
        )
        return ExtractionResult(
            pages=[page],
            page_count=1,
            extraction_quality=ExtractionQuality.LOW,
            ocr_recommended=True,
            vision_recommended=True,
        )

    monkeypatch.setattr(folder_loader, "extract_document", fake_extract_document)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "may need image-based text reading (OCR)" in normalized_output
    assert "before Hail Mary can use all of their content" in normalized_output


def test_ingest_folder_warns_when_no_usable_evidence_is_built(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "EmptyCo"
    source.mkdir(parents=True)
    (source / "empty.pdf").write_bytes(b"%PDF-1.4")

    def fake_extract_document(path: Path) -> ExtractionResult:
        assert path.name == "empty.pdf"
        return ExtractionResult(
            pages=[],
            page_count=1,
            extraction_quality=ExtractionQuality.LOW,
        )

    monkeypatch.setattr(folder_loader, "extract_document", fake_extract_document)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code == 0, result.output
    assert "No usable evidence text was built" in result.output
    assert "cannot use their text yet" in result.output


def test_ingest_folder_unreadable_path_has_plain_english_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "pitch-decks"
    company = root / "HiddenCo"
    secret_dir = company / "Secret"
    company.mkdir(parents=True)
    secret_dir.mkdir()
    (company / "memo.txt").write_text("Memo about HiddenCo.", encoding="utf-8")

    def fake_walk(
        top: Path,
        topdown: bool,
        onerror: Callable[[OSError], None] | None,
        followlinks: bool,
    ) -> Iterator[tuple[Path, list[str], list[str]]]:
        assert top == root.resolve()
        assert topdown is True
        assert followlinks is False
        yield root.resolve(), ["HiddenCo"], []
        if callable(onerror):
            onerror(PermissionError(13, "Permission denied", str(secret_dir)))
        yield company.resolve(), ["Secret"], ["memo.txt"]

    monkeypatch.setattr(os, "walk", fake_walk)

    result = runner.invoke(app, ["ingest-folder", str(root), "--data-dir", str(tmp_path / "data")])

    assert result.exit_code == 0, result.output
    assert "Could not read 1 path" in result.output
    assert "documents may be missing" in result.output
    assert "unsupported or ignored" not in result.output


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
    assert "cannot be the current folder" in result.output
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


def test_init_bad_config_path_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".hailmary" / "config.yaml"
    config_path.mkdir(parents=True)

    result = runner.invoke(app, ["init", "--force"])

    assert result.exit_code != 0
    assert "needs .hailmary/config.yaml to be a file" in result.output
    assert "Traceback" not in result.output


def test_init_force_recovers_invalid_saved_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("local_only: treu\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--force", "--data-dir", "local-data"])

    assert result.exit_code == 0, result.output
    assert "Created local config" in result.output
    config_text = (config_dir / "config.yaml").read_text(encoding="utf-8")
    assert 'data_dir: "local-data"' in config_text
    assert "local_only: true" in config_text


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
    assert "cannot be the current folder" in result.output
    assert "Traceback" not in result.output


def test_ingest_folder_summary_write_error_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")
    summary_path = tmp_path / "data" / "processed" / "ingestion_summary.json"
    summary_path.mkdir(parents=True)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code != 0
    assert "Could not write scan summary" in result.output
    assert "Traceback" not in result.output
