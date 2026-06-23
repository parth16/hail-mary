from __future__ import annotations

from pathlib import Path

from hailmary.ingest.document_classifier import (
    classify_document,
    classify_file_type,
    is_ignored_path,
)
from hailmary.schemas.documents import DocumentType, FileType


def test_classifies_platform_deal_page_from_angellist_pdf_name() -> None:
    path = Path("H3X/H3X _ AngelList.pdf")

    assert classify_file_type(path) == FileType.PDF
    assert classify_document(path) == DocumentType.PLATFORM_DEAL_PAGE


def test_classifies_legal_docx_from_content() -> None:
    path = Path("Example/closing.docx")

    document_type = classify_document(path, text_sample="PRIVATE PLACEMENT MEMORANDUM")

    assert document_type == DocumentType.LEGAL_DOCUMENT


def test_classifies_pitch_deck_from_investor_overview_name() -> None:
    path = Path("Wild West/Wild_West_Systems_Investor_Overview_June26.pdf")

    assert classify_document(path) == DocumentType.PITCH_DECK


def test_ignores_local_noise_paths() -> None:
    assert is_ignored_path(Path(".DS_Store"))
    assert is_ignored_path(Path("Company/.DS_Store"))
    assert is_ignored_path(Path("data/raw/file.pdf"))
