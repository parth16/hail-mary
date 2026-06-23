from __future__ import annotations

import json
from pathlib import Path

from docx import Document

from hailmary.config import AppConfig
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
