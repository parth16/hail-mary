from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from docx import Document

from hailmary.config import AppConfig
from hailmary.ingest import folder_loader
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.schemas.documents import DocumentType, ExtractionQuality, FileType


def _write_docx(path: Path, lines: list[str]) -> None:
    document = Document()
    for line in lines:
        document.add_paragraph(line)
    document.save(str(path))


def test_ingest_folder_groups_documents_and_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "ExampleCo"
    closing = company / "Closing documents"
    closing.mkdir(parents=True)
    (company / ".DS_Store").write_text("noise", encoding="utf-8")
    (company / "ExampleCo _ AngelList.pdf").write_text("not a real pdf", encoding="utf-8")
    (company / "notes.txt").write_text("Memo about traction and team.", encoding="utf-8")
    _write_docx(
        closing / "ExampleCo - PPM.docx",
        ["PRIVATE PLACEMENT MEMORANDUM", "ExampleCo confidential investment material."],
    )

    config = AppConfig(data_dir=tmp_path / "data")

    summary = ingest_folder(root, config=config)

    assert summary.document_count == 3
    assert len(summary.deals) == 1
    assert summary.deals[0].company_name == "ExampleCo"
    assert ".DS_Store" in " ".join(summary.skipped_files)
    assert summary.summary_path.exists()

    document_types = {doc.source.document_type for doc in summary.deals[0].documents}
    assert DocumentType.PLATFORM_DEAL_PAGE in document_types
    assert DocumentType.LEGAL_DOCUMENT in document_types

    output_paths = [doc.output_path for doc in summary.deals[0].documents]
    assert all(path.exists() for path in output_paths)

    saved_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert saved_summary["deals"][0]["id"] == "exampleco"


def test_invalid_pdf_is_recorded_without_crashing(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "BadPdfCo"
    company.mkdir(parents=True)
    (company / "BadPdfCo _ AngelList.pdf").write_text("not a real pdf", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    document = summary.deals[0].documents[0].source
    assert document.file_type == FileType.PDF
    assert document.document_type == DocumentType.PLATFORM_DEAL_PAGE
    assert document.extraction_quality == ExtractionQuality.LOW
    assert document.notes is not None


def test_duplicate_files_get_distinct_output_paths(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "DuplicateCo"
    company.mkdir(parents=True)
    (company / "one.txt").write_text("Same diligence note.", encoding="utf-8")
    (company / "two.txt").write_text("Same diligence note.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))
    output_paths = [doc.output_path for doc in summary.deals[0].documents]

    assert len(output_paths) == 2
    assert len(set(output_paths)) == 2
    assert all(path.exists() for path in output_paths)


def test_top_level_deal_named_data_is_ingested(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Data"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Memo about the Data company.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "generated-data"))

    assert summary.document_count == 1
    assert len(summary.deals) == 1
    assert summary.deals[0].company_name == "Data"


def test_real_nested_data_and_reports_folders_are_ingested(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    data_folder = root / "Acme" / "Data"
    reports_folder = root / "Acme" / "Reports"
    data_folder.mkdir(parents=True)
    reports_folder.mkdir(parents=True)
    (data_folder / "deck.pdf").write_text("not a real pdf", encoding="utf-8")
    (reports_folder / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "generated-data"))

    assert summary.document_count == 2


def test_configured_output_folder_inside_scan_root_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme"
    output_folder = root / "generated-data"
    company.mkdir(parents=True)
    output_folder.mkdir(parents=True)
    (company / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")
    (output_folder / "old-output.txt").write_text("This is generated output.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=output_folder))

    assert summary.document_count == 1
    assert "generated-data/old-output.txt" in summary.skipped_files


def test_generated_outputs_are_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "PrivateCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Memo about PrivateCo.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    document_path = summary.deals[0].documents[0].output_path
    assert stat.S_IMODE(document_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(summary.summary_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "data" / "processed").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "data" / "processed" / "deals").stat().st_mode) == 0o700
    assert stat.S_IMODE(document_path.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(document_path.parent.stat().st_mode) == 0o700


def test_colliding_deal_folder_slugs_stay_separate(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    first = root / "Acme.AI"
    second = root / "Acme AI"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "memo.txt").write_text("Memo about Acme.AI.", encoding="utf-8")
    (second / "memo.txt").write_text("Memo about Acme AI.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal_ids = {deal.id for deal in summary.deals}
    company_names = {deal.company_name for deal in summary.deals}
    assert summary.document_count == 2
    assert len(summary.deals) == 2
    assert len(deal_ids) == 2
    assert company_names == {"Acme.AI", "Acme AI"}


def test_confidentiality_detection_uses_raw_text_before_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "ConfidentialCo"
    company.mkdir(parents=True)
    (company / "ConfidentialCo memo.txt").write_text(
        "\n".join(
            [
                "Confidential: Parth Shah",
                "Business details.",
                "Confidential: Parth Shah",
                "More business details.",
                "Confidential: Parth Shah",
            ]
        ),
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    source = summary.deals[0].documents[0].source
    assert source.confidentiality_detected is True


def test_docx_header_confidentiality_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "HeaderCo"
    company.mkdir(parents=True)
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Confidential"
    document.add_paragraph("Business details.")
    document.save(str(company / "memo.docx"))

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    source = summary.deals[0].documents[0].source
    assert source.confidentiality_detected is True


def test_hash_failure_is_recorded_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "HashFailCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Memo about HashFailCo.", encoding="utf-8")

    def fail_sha256(path: Path) -> str:
        raise PermissionError("permission changed")

    monkeypatch.setattr(folder_loader, "_sha256", fail_sha256)

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    source = summary.deals[0].documents[0].source
    assert summary.document_count == 1
    assert source.sha256 is None
    assert source.notes is not None
    assert "Could not calculate the file fingerprint" in source.notes
    assert summary.deals[0].documents[0].output_path.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_symlinked_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "SymlinkCo"
    company.mkdir(parents=True)
    outside_file = tmp_path / "outside-secret.txt"
    outside_file.write_text("This should not be read.", encoding="utf-8")
    (company / "deck.txt").symlink_to(outside_file)

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    assert summary.document_count == 0
    assert summary.skipped_files == ["SymlinkCo/deck.txt"]
