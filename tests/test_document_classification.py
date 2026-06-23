from __future__ import annotations

from pathlib import Path

from hailmary.ingest.document_classifier import (
    SUPPORTED_SUFFIXES,
    classify_document,
    classify_file_type,
    is_ignored_path,
)
from hailmary.schemas.documents import DocumentType, FileType


def test_classifies_platform_deal_page_from_angellist_pdf_name() -> None:
    path = Path("H3X/H3X _ AngelList.pdf")

    assert classify_file_type(path) == FileType.PDF
    assert classify_document(path) == DocumentType.PLATFORM_DEAL_PAGE


def test_classifies_saved_platform_html_as_platform_deal_page() -> None:
    path = Path("H3X/H3X _ AngelList.html")

    assert classify_file_type(path) == FileType.HTML
    assert classify_document(path) == DocumentType.PLATFORM_DEAL_PAGE


def test_does_not_classify_company_named_meridian_as_platform_page() -> None:
    path = Path("Meridian Robotics/Meridian Robotics Pitch Deck.pdf")

    assert classify_document(path) == DocumentType.PITCH_DECK


def test_meridian_company_text_does_not_trigger_platform_page() -> None:
    path = Path("Meridian Robotics/Meridian Robotics Pitch Deck.pdf")

    document_type = classify_document(
        path,
        text_sample="Meridian Robotics investment opportunity for seed investors.",
    )

    assert document_type == DocumentType.PITCH_DECK


def test_angellist_text_requires_page_marker_for_platform_page() -> None:
    path = Path("Example/research.pdf")

    document_type = classify_document(
        path,
        text_sample="The company may invest in go-to-market after AngelList outreach.",
    )

    assert document_type == DocumentType.UNKNOWN


def test_angellist_source_name_does_not_override_pitch_deck() -> None:
    path = Path("Acme/Acme AngelList Pitch Deck.pdf")

    assert classify_document(path) == DocumentType.PITCH_DECK


def test_angellist_source_name_does_not_override_legal_document() -> None:
    path = Path("Acme/Acme AngelList Subscription Agreement.pdf")

    assert classify_document(path) == DocumentType.LEGAL_DOCUMENT


def test_classifies_legal_docx_from_content() -> None:
    path = Path("Example/closing.docx")

    document_type = classify_document(path, text_sample="PRIVATE PLACEMENT MEMORANDUM")

    assert document_type == DocumentType.LEGAL_DOCUMENT


def test_classifies_legal_pdf_from_content() -> None:
    path = Path("Example/closing.pdf")

    document_type = classify_document(path, text_sample="PRIVATE PLACEMENT MEMORANDUM")

    assert document_type == DocumentType.LEGAL_DOCUMENT


def test_classifies_legal_pdf_from_line_broken_content() -> None:
    path = Path("Example/closing.pdf")

    document_type = classify_document(path, text_sample="PRIVATE PLACEMENT\nMEMORANDUM")

    assert document_type == DocumentType.LEGAL_DOCUMENT


def test_classifies_standalone_legal_abbreviations_from_filename() -> None:
    assert classify_document(Path("Example/PPM.pdf")) == DocumentType.LEGAL_DOCUMENT
    assert classify_document(Path("Example/LPA.docx")) == DocumentType.LEGAL_DOCUMENT


def test_company_name_legal_abbreviations_do_not_override_pitch_deck() -> None:
    path = Path("PPM Labs/PPM Labs Pitch Deck.pdf")

    assert classify_document(path) == DocumentType.PITCH_DECK


def test_classifies_pitch_deck_from_investor_overview_name() -> None:
    path = Path("Wild West/Wild_West_Systems_Investor_Overview_June26.pdf")

    assert classify_document(path) == DocumentType.PITCH_DECK


def test_pitch_deck_markers_must_be_words() -> None:
    assert classify_document(Path("PitchBook/terms.pdf")) == DocumentType.UNKNOWN
    assert classify_document(Path("Deckard/customer.pdf")) == DocumentType.UNKNOWN


def test_images_are_recognized_but_not_supported_for_ingestion() -> None:
    assert classify_file_type(Path("scan.png")) == FileType.PNG
    assert classify_file_type(Path("photo.jpg")) == FileType.JPG
    assert ".png" not in SUPPORTED_SUFFIXES
    assert ".jpg" not in SUPPORTED_SUFFIXES
    assert ".jpeg" not in SUPPORTED_SUFFIXES


def test_classifies_customer_diligence_documents() -> None:
    assert (
        classify_document(Path("Acme/customer-reference-notes.pdf"))
        == DocumentType.CUSTOMER_DOCUMENT
    )
    assert (
        classify_document(Path("Acme/contracts.docx"), text_sample="Customer contract")
        == DocumentType.CUSTOMER_DOCUMENT
    )


def test_classifies_pdf_and_docx_investment_memos() -> None:
    assert classify_document(Path("Acme/Investment Memo.pdf")) == DocumentType.MEMO
    assert classify_document(Path("Acme/Investment Committee Memorandum.docx")) == DocumentType.MEMO


def test_series_stock_purchase_agreement_is_legal_document() -> None:
    path = Path("Acme/Series Seed Preferred Stock Purchase Agreement.pdf")

    assert classify_document(path) == DocumentType.LEGAL_DOCUMENT


def test_classifies_financials_and_projection_spreadsheets() -> None:
    assert classify_document(Path("Acme/Acme Financials.xlsx")) == DocumentType.FINANCIAL_MODEL
    assert classify_document(Path("Acme/2026 projections.csv")) == DocumentType.FINANCIAL_MODEL


def test_ignores_local_noise_paths() -> None:
    assert is_ignored_path(Path(".DS_Store"))
    assert is_ignored_path(Path("Company/.DS_Store"))
    assert is_ignored_path(Path("Company/.git/file.pdf"))


def test_does_not_ignore_top_level_deal_named_data_or_reports() -> None:
    assert not is_ignored_path(Path("Data/deck.pdf"))
    assert not is_ignored_path(Path("Reports/memo.txt"))


def test_does_not_ignore_real_nested_data_or_reports_folders() -> None:
    assert not is_ignored_path(Path("Acme/Data/deck.pdf"))
    assert not is_ignored_path(Path("Acme/Reports/customer.pdf"))
