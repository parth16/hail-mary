from __future__ import annotations

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


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


def classify_file_type(path: Path) -> FileType:
    return FILE_TYPE_BY_SUFFIX.get(path.suffix.lower(), FileType.UNKNOWN)


def classify_document(path: Path, text_sample: str = "") -> DocumentType:
    """Classify a diligence document using path hints and extracted text."""

    file_type = classify_file_type(path)
    haystack = f"{path.as_posix()} {text_sample}".lower()
    normalized_haystack = haystack.replace("_", " ").replace("-", " ")

    if _is_platform_deal_page(path, text_sample, file_type):
        return DocumentType.PLATFORM_DEAL_PAGE

    legal_markers = [
        "limited partnership agreement",
        "private placement memorandum",
        "subscription agreement",
        "subscription documents",
        " lpa",
        " ppm",
    ]
    if file_type in {FileType.DOCX, FileType.PDF} and any(
        marker in normalized_haystack for marker in legal_markers
    ):
        return DocumentType.LEGAL_DOCUMENT

    if file_type in {FileType.XLSX, FileType.CSV} and any(
        marker in normalized_haystack for marker in ["model", "forecast", "financial", "revenue"]
    ):
        return DocumentType.FINANCIAL_MODEL

    if file_type == FileType.PDF and any(
        marker in normalized_haystack
        for marker in ["pitch", "deck", "investor overview", "series deck"]
    ):
        return DocumentType.PITCH_DECK

    if file_type == FileType.HTML:
        return DocumentType.WEB_PAGE

    if file_type in {FileType.TXT, FileType.MD} and any(
        marker in normalized_haystack for marker in ["memo", "note", "diligence"]
    ):
        return DocumentType.MEMO

    return DocumentType.UNKNOWN


def _is_platform_deal_page(path: Path, text_sample: str, file_type: FileType) -> bool:
    if file_type not in {FileType.PDF, FileType.HTML}:
        return False

    filename = path.name.lower().replace("_", " ").replace("-", " ")
    filename_stem = path.stem.lower().replace("_", " ").replace("-", " ")
    text = text_sample.lower().replace("_", " ").replace("-", " ")
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
    if any(marker in filename for marker in platform_filename_markers):
        return True

    has_angellist_text = any(
        marker in text for marker in ["angellist", "angel list", "portal.angellist.com"]
    )
    has_platform_page_text = any(
        marker in text
        for marker in ["invest", "investment opportunity", "deal page", "company profile"]
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


def is_ignored_path(path: Path) -> bool:
    ignored_anywhere = {
        ".ds_store",
        "__pycache__",
        ".git",
        ".hailmary",
    }

    parts = [part.lower() for part in path.parts]
    return any(part in ignored_anywhere for part in parts)
