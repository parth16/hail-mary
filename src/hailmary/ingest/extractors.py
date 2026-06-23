from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pydantic import BaseModel
from pypdf import PdfReader

from hailmary.ingest.document_classifier import classify_file_type
from hailmary.schemas.documents import ExtractedPage, ExtractionQuality, FileType
from hailmary.utils.text_cleaning import clean_extracted_text


class ExtractionResult(BaseModel):
    pages: list[ExtractedPage]
    page_count: int | None
    extraction_quality: ExtractionQuality
    notes: str | None = None

    @property
    def combined_text(self) -> str:
        return "\n\n".join(page.clean_text for page in self.pages if page.clean_text)

    @property
    def combined_raw_text(self) -> str:
        return "\n\n".join(page.raw_text for page in self.pages if page.raw_text)


def extract_document(path: Path) -> ExtractionResult:
    file_type = classify_file_type(path)

    if file_type == FileType.PDF:
        return _extract_pdf(path)
    if file_type == FileType.DOCX:
        return _extract_docx(path)
    if file_type == FileType.HTML:
        return _extract_html(path)
    if file_type in {FileType.TXT, FileType.MD}:
        return _extract_text_file(path)

    return ExtractionResult(
        pages=[],
        page_count=None,
        extraction_quality=ExtractionQuality.LOW,
        notes="Text extraction is not implemented for this file type yet.",
    )


def _quality_from_pages(pages: list[ExtractedPage]) -> ExtractionQuality:
    if not pages:
        return ExtractionQuality.LOW

    word_count = sum(page.word_count for page in pages)
    words_per_page = word_count / len(pages)
    if words_per_page >= 50:
        return ExtractionQuality.HIGH
    if word_count > 0:
        return ExtractionQuality.MEDIUM
    return ExtractionQuality.LOW


def _extract_pdf(path: Path) -> ExtractionResult:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the PDF: {exc}",
        )

    try:
        pages_proxy = reader.pages
        page_count = len(pages_proxy)
    except Exception as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read pages from the PDF: {exc}",
        )

    pages: list[ExtractedPage] = []
    for index in range(page_count):
        try:
            page = pages_proxy[index]
            raw_text = page.extract_text() or ""
            notes = None
        except Exception as exc:
            raw_text = ""
            notes = f"Could not extract text from page {index + 1}: {exc}"

        clean_text = clean_extracted_text(raw_text)
        pages.append(
            ExtractedPage(
                page_number=index + 1,
                raw_text=raw_text,
                clean_text=clean_text,
                word_count=len(clean_text.split()),
                needs_ocr=len(clean_text.split()) < 10,
                notes=notes,
            )
        )

    return ExtractionResult(
        pages=pages,
        page_count=page_count,
        extraction_quality=_quality_from_pages(pages),
    )


def _extract_docx(path: Path) -> ExtractionResult:
    try:
        document = Document(str(path))
    except Exception as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the DOCX file: {exc}",
        )

    paragraphs = [
        paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
    ]
    table_text: list[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                table_text.append(" | ".join(cells))

    raw_text = "\n".join([*paragraphs, *table_text])
    clean_text = clean_extracted_text(raw_text)
    page = ExtractedPage(
        page_number=None,
        raw_text=raw_text,
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        needs_ocr=False,
        notes=None,
    )

    return ExtractionResult(
        pages=[page] if clean_text else [],
        page_count=None,
        extraction_quality=_quality_from_pages([page] if clean_text else []),
    )


def _extract_text_file(path: Path) -> ExtractionResult:
    try:
        raw_text = path.read_text(encoding="utf-8")
        notes = None
    except UnicodeDecodeError:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        notes = "Some characters could not be read and were replaced."
    except OSError as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the text file: {exc}",
        )

    clean_text = clean_extracted_text(raw_text)
    page = ExtractedPage(
        page_number=None,
        raw_text=raw_text,
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        needs_ocr=False,
        notes=notes,
    )

    return ExtractionResult(
        pages=[page] if clean_text else [],
        page_count=1,
        extraction_quality=_quality_from_pages([page] if clean_text else []),
        notes=notes,
    )


def _extract_html(path: Path) -> ExtractionResult:
    try:
        html = path.read_text(encoding="utf-8")
        notes = None
    except UnicodeDecodeError:
        html = path.read_text(encoding="utf-8", errors="replace")
        notes = "Some characters could not be read and were replaced."
    except OSError as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the HTML file: {exc}",
        )

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    raw_text = soup.get_text(separator="\n")
    clean_text = clean_extracted_text(raw_text)
    page = ExtractedPage(
        page_number=None,
        raw_text=raw_text,
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        needs_ocr=False,
        notes=notes,
    )

    return ExtractionResult(
        pages=[page] if clean_text else [],
        page_count=1,
        extraction_quality=_quality_from_pages([page] if clean_text else []),
        notes=notes,
    )
