from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

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
    if file_type == FileType.CSV:
        return _extract_csv(path)
    if file_type == FileType.XLSX:
        return _extract_xlsx(path)
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
    ocr_needed_pages = sum(1 for page in pages if page.needs_ocr)
    if words_per_page >= 50 and ocr_needed_pages * 2 < len(pages):
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
        notes=_page_failure_notes(pages),
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

    try:
        raw_text = "\n".join(_docx_text_parts(document))
    except Exception as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not extract text from the DOCX file: {exc}",
        )

    clean_text = clean_extracted_text(raw_text)
    page = ExtractedPage(
        page_number=None,
        raw_text=raw_text,
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        needs_ocr=False,
        notes=None,
    )

    pages = _pages_with_raw_text(page)
    return ExtractionResult(
        pages=pages,
        page_count=None,
        extraction_quality=_quality_from_pages(pages),
    )


def _docx_text_parts(document: Any) -> list[str]:
    text_parts: list[str] = []
    text_parts.extend(_paragraph_text(document.paragraphs))
    text_parts.extend(_table_text(document.tables))

    for section in document.sections:
        containers = [
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ]
        for container in containers:
            text_parts.extend(_paragraph_text(container.paragraphs))
            text_parts.extend(_table_text(container.tables))

    return text_parts


def _paragraph_text(paragraphs: Iterable[Any]) -> list[str]:
    return [
        paragraph.text.strip()
        for paragraph in paragraphs
        if getattr(paragraph, "text", "").strip()
    ]


def _table_text(tables: Iterable[Any]) -> list[str]:
    table_text: list[str] = []
    for table in tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                table_text.append(" | ".join(cells))
    return table_text


def _page_failure_notes(pages: list[ExtractedPage]) -> str | None:
    page_notes = [page.notes for page in pages if page.notes]
    if not page_notes:
        return None
    return " ".join(page_notes)


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

    pages = _pages_with_raw_text(page)
    return ExtractionResult(
        pages=pages,
        page_count=1,
        extraction_quality=_quality_from_pages(pages),
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

    pages = _pages_with_raw_text(page)
    return ExtractionResult(
        pages=pages,
        page_count=1,
        extraction_quality=_quality_from_pages(pages),
        notes=notes,
    )


def _extract_csv(path: Path) -> ExtractionResult:
    try:
        csv_text = path.read_text(encoding="utf-8")
        notes = None
    except UnicodeDecodeError:
        csv_text = path.read_text(encoding="utf-8", errors="replace")
        notes = "Some characters could not be read and were replaced."
    except OSError as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the CSV file: {exc}",
        )

    rows = csv.reader(io.StringIO(csv_text))
    try:
        raw_text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
    except csv.Error as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not parse the CSV file: {exc}",
        )
    return _single_page_result(raw_text, page_count=1, notes=notes)


def _extract_xlsx(path: Path) -> ExtractionResult:
    try:
        with zipfile.ZipFile(path) as workbook:
            shared_strings = _xlsx_shared_strings(workbook)
            sheet_names = sorted(
                name
                for name in workbook.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            sheet_text = [
                _xlsx_sheet_text(workbook.read(sheet_name), shared_strings)
                for sheet_name in sheet_names
            ]
    except (OSError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the XLSX file: {exc}",
        )

    raw_text = "\n".join(text for text in sheet_text if text)
    return _single_page_result(raw_text, page_count=len(sheet_names), notes=None)


def _xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        shared_strings_xml = workbook.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ElementTree.fromstring(shared_strings_xml)
    strings: list[str] = []
    for item in root.iter():
        if _xml_local_name(item.tag) != "si":
            continue
        text_parts = [
            text_element.text or ""
            for text_element in item.iter()
            if _xml_local_name(text_element.tag) == "t"
        ]
        strings.append("".join(text_parts).strip())
    return strings


def _xlsx_sheet_text(sheet_xml: bytes, shared_strings: list[str]) -> str:
    root = ElementTree.fromstring(sheet_xml)
    row_lines: list[str] = []
    for row in root.iter():
        if _xml_local_name(row.tag) != "row":
            continue
        values = [
            _xlsx_cell_value(cell, shared_strings)
            for cell in row
            if _xml_local_name(cell.tag) == "c"
        ]
        stripped_values = [value for value in values if value]
        if stripped_values:
            row_lines.append(" | ".join(stripped_values))
    return "\n".join(row_lines)


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return " ".join(
            text_element.text or ""
            for text_element in cell.iter()
            if _xml_local_name(text_element.tag) == "t"
        ).strip()

    value = ""
    for child in cell:
        if _xml_local_name(child.tag) == "v":
            value = child.text or ""
            break

    if cell_type == "s":
        try:
            shared_string_index = int(value)
        except ValueError:
            return value
        if shared_string_index < 0:
            return value
        try:
            return shared_strings[shared_string_index]
        except IndexError:
            return value
    return value.strip()


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _single_page_result(
    raw_text: str,
    *,
    page_count: int | None,
    notes: str | None,
) -> ExtractionResult:
    clean_text = clean_extracted_text(raw_text)
    page = ExtractedPage(
        page_number=None,
        raw_text=raw_text,
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        needs_ocr=False,
        notes=notes,
    )

    pages = _pages_with_raw_text(page)
    return ExtractionResult(
        pages=pages,
        page_count=page_count,
        extraction_quality=_quality_from_pages(pages),
        notes=notes,
    )


def _pages_with_raw_text(page: ExtractedPage) -> list[ExtractedPage]:
    if page.clean_text or page.raw_text.strip():
        return [page]
    return []
