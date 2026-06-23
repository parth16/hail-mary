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


SUPPORTED_SUFFIXES = set(FILE_TYPE_BY_SUFFIX)


def classify_file_type(path: Path) -> FileType:
    return FILE_TYPE_BY_SUFFIX.get(path.suffix.lower(), FileType.UNKNOWN)


def classify_document(path: Path, text_sample: str = "") -> DocumentType:
    """Classify a diligence document using path hints and extracted text."""

    file_type = classify_file_type(path)
    haystack = f"{path.as_posix()} {text_sample}".lower()
    normalized_haystack = haystack.replace("_", " ").replace("-", " ")

    if file_type == FileType.PDF and (
        "angellist" in normalized_haystack or "meridian" in normalized_haystack
    ):
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
        marker in normalized_haystack for marker in ["pitch", "deck", "investor overview", "series"]
    ):
        return DocumentType.PITCH_DECK

    if file_type == FileType.HTML:
        return DocumentType.WEB_PAGE

    if file_type in {FileType.TXT, FileType.MD} and any(
        marker in normalized_haystack for marker in ["memo", "note", "diligence"]
    ):
        return DocumentType.MEMO

    return DocumentType.UNKNOWN


def is_ignored_path(path: Path) -> bool:
    ignored_anywhere = {
        ".ds_store",
        "__pycache__",
        ".git",
        ".hailmary",
    }
    generated_dirs = {
        "data",
        "reports",
        "browser-profiles",
    }

    parts = [part.lower() for part in path.parts]
    if any(part in ignored_anywhere for part in parts):
        return True

    return any(part in generated_dirs for part in parts[1:])
