from __future__ import annotations

import re
from pathlib import Path

from hailmary.schemas.documents import DocumentType, FileType

FILE_TYPE_BY_SUFFIX: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".xlsx": FileType.XLSX,
    ".csv": FileType.CSV,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".txt": FileType.TXT,
    ".md": FileType.MD,
    ".png": FileType.PNG,
    ".jpg": FileType.JPG,
    ".jpeg": FileType.JPG,
}


SUPPORTED_SUFFIXES = {
    suffix
    for suffix, file_type in FILE_TYPE_BY_SUFFIX.items()
    if file_type not in {FileType.PNG, FileType.JPG}
}


def classify_file_type(path: Path) -> FileType:
    return FILE_TYPE_BY_SUFFIX.get(path.suffix.lower(), FileType.UNKNOWN)


def classify_document(path: Path, text_sample: str = "") -> DocumentType:
    """Classify a diligence document using path hints and extracted text."""

    file_type = classify_file_type(path)
    haystack = f"{path.as_posix()} {text_sample}".lower()
    normalized_haystack = haystack.replace("_", " ").replace("-", " ")
    word_haystack = _word_haystack(path, text_sample)
    filename_word_haystack = _word_haystack(Path(path.name), "")

    if _is_text_confirmed_platform_deal_page(text_sample, file_type):
        return DocumentType.PLATFORM_DEAL_PAGE

    if _is_filename_platform_deal_page(path, file_type):
        return DocumentType.PLATFORM_DEAL_PAGE

    legal_markers = [
        "convertible note",
        "convertible notes",
        "limited partnership agreement",
        "private placement memorandum",
        "safe agreement",
        "safe financing",
        "simple agreement for future equity",
        "subscription agreement",
        "subscription documents",
        "stock purchase agreement",
        "stock purchase agreements",
    ]
    legal_abbreviations = ["lpa", "ppm", "safe"]
    if file_type == FileType.PDF and (
        any(_contains_phrase(filename_word_haystack, marker) for marker in ["pitch", "deck"])
        or any(
            _contains_phrase(filename_word_haystack, marker)
            for marker in ["investor overview", "series deck"]
        )
    ):
        return DocumentType.PITCH_DECK

    if file_type in {FileType.DOCX, FileType.PDF} and any(
        _contains_phrase(word_haystack, marker) for marker in legal_markers
    ):
        return DocumentType.LEGAL_DOCUMENT

    if file_type == FileType.PDF and (
        any(_contains_phrase(word_haystack, marker) for marker in ["pitch", "deck"])
        or any(
            _contains_phrase(word_haystack, marker)
            for marker in ["investor overview", "series deck"]
        )
    ):
        return DocumentType.PITCH_DECK

    if file_type in {FileType.DOCX, FileType.PDF} and any(
        _contains_phrase(filename_word_haystack, marker) for marker in legal_abbreviations
    ):
        return DocumentType.LEGAL_DOCUMENT

    if file_type in {FileType.XLSX, FileType.CSV} and any(
        _contains_phrase(word_haystack, marker)
        for marker in [
            "model",
            "forecast",
            "financial",
            "financials",
            "projection",
            "projections",
            "revenue",
        ]
    ):
        return DocumentType.FINANCIAL_MODEL

    if file_type in {FileType.DOCX, FileType.PDF} and any(
        _contains_phrase(word_haystack, marker)
        for marker in [
            "investment memo",
            "investment memorandum",
            "investment committee memo",
            "investment committee memorandum",
            "diligence memo",
        ]
    ):
        return DocumentType.MEMO

    if any(
        _contains_phrase(word_haystack, marker)
        for marker in [
            "customer reference",
            "customer references",
            "customer call",
            "customer calls",
            "customer contract",
            "customer contracts",
            "customer diligence",
            "case study",
            "case studies",
        ]
    ):
        return DocumentType.CUSTOMER_DOCUMENT

    if file_type == FileType.HTML:
        return DocumentType.WEB_PAGE

    if file_type in {FileType.TXT, FileType.MD} and any(
        marker in normalized_haystack for marker in ["memo", "note", "diligence"]
    ):
        return DocumentType.MEMO

    return DocumentType.UNKNOWN


def _word_haystack(path: Path, text_sample: str) -> str:
    return f" {re.sub(r'[^a-z0-9]+', ' ', f'{path.as_posix()} {text_sample}'.lower()).strip()} "


def _contains_phrase(word_haystack: str, phrase: str) -> bool:
    phrase_words = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
    return f" {phrase_words} " in word_haystack


def _is_filename_platform_deal_page(path: Path, file_type: FileType) -> bool:
    if file_type not in {FileType.PDF, FileType.HTML}:
        return False

    filename = path.name.lower().replace("_", " ").replace("-", " ")
    filename_stem = path.stem.lower().replace("_", " ").replace("-", " ")
    platform_filename_markers = [
        " angellist",
        "angel list",
        "angellist deal",
        "angellist investment",
        "angellist profile",
        "meridian deal",
        "meridian investment",
        "meridian profile",
    ]

    if filename_stem in {"angellist", "angel list", "meridian"}:
        return True
    if _filename_has_document_type_marker(filename_stem):
        return False
    return any(marker in filename for marker in platform_filename_markers)


def _is_text_confirmed_platform_deal_page(text_sample: str, file_type: FileType) -> bool:
    if file_type not in {FileType.PDF, FileType.HTML}:
        return False

    text = text_sample.lower().replace("_", " ").replace("-", " ")
    has_angellist_text = any(
        marker in text for marker in ["angellist", "angel list", "portal.angellist.com"]
    )
    has_platform_page_text = any(
        marker in text
        for marker in [
            "investment opportunity",
            "deal page",
            "company profile",
            "view deal",
            "invest now",
        ]
    )
    has_meridian_page_text = any(
        marker in text
        for marker in [
            "meridian deal",
            "meridian deal page",
            "meridian investment page",
            "meridian profile",
            "meridian portal",
        ]
    )
    return (has_angellist_text and has_platform_page_text) or has_meridian_page_text


def _filename_has_document_type_marker(filename_stem: str) -> bool:
    filename_words = f" {re.sub(r'[^a-z0-9]+', ' ', filename_stem).strip()} "
    document_markers = [
        "pitch",
        "deck",
        "investor overview",
        "series deck",
        "limited partnership agreement",
        "private placement memorandum",
        "safe agreement",
        "safe financing",
        "simple agreement for future equity",
        "subscription agreement",
        "subscription documents",
        "stock purchase agreement",
        "stock purchase agreements",
        "lpa",
        "ppm",
        "safe",
        "convertible note",
        "convertible notes",
        "model",
        "forecast",
        "financial",
        "financials",
        "projection",
        "projections",
        "revenue",
        "customer reference",
        "customer references",
        "customer call",
        "customer calls",
        "customer contract",
        "customer contracts",
        "customer diligence",
        "case study",
        "case studies",
        "memo",
        "investment memo",
        "investment memorandum",
        "investment committee memo",
        "investment committee memorandum",
        "diligence memo",
        "note",
        "diligence",
    ]
    return any(_contains_phrase(filename_words, marker) for marker in document_markers)


def is_ignored_path(path: Path) -> bool:
    ignored_anywhere = {
        ".ds_store",
        "__pycache__",
        ".git",
        ".hailmary",
    }

    parts = [part.lower() for part in path.parts]
    return any(part in ignored_anywhere for part in parts)
