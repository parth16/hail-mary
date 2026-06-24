from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from bs4 import BeautifulSoup
from docx import Document
from pydantic import BaseModel, Field
from pypdf import PdfReader

from hailmary.ingest.document_classifier import classify_file_type
from hailmary.schemas.documents import (
    ExtractedPage,
    ExtractedTable,
    ExtractionQuality,
    FileType,
)
from hailmary.utils.text_cleaning import clean_extracted_text_with_metadata


class ExtractionResult(BaseModel):
    pages: list[ExtractedPage]
    tables: list[ExtractedTable] = Field(default_factory=list)
    page_count: int | None
    extraction_quality: ExtractionQuality
    ocr_recommended: bool = False
    vision_recommended: bool = False
    notes: str | None = None

    @property
    def combined_text(self) -> str:
        return "\n\n".join(page.clean_text for page in self.pages if page.clean_text)

    @property
    def combined_raw_text(self) -> str:
        return "\n\n".join(page.raw_text for page in self.pages if page.raw_text)

    @property
    def table_count(self) -> int:
        return len(self.tables)


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


def _make_page(
    raw_text: str,
    *,
    page_number: int | None,
    needs_ocr: bool | None,
    vision_recommended: bool | None = False,
    source_span_start: int | None = None,
    notes: str | None = None,
) -> ExtractedPage:
    cleaning = clean_extracted_text_with_metadata(raw_text)
    word_count = len(cleaning.clean_text.split())
    page_needs_ocr = not cleaning.clean_text.strip() if needs_ocr is None else needs_ocr
    page_needs_vision = page_needs_ocr if vision_recommended is None else vision_recommended
    source_span_end = (
        source_span_start + len(raw_text) if source_span_start is not None else None
    )
    return ExtractedPage(
        page_number=page_number,
        raw_text=raw_text,
        clean_text=cleaning.clean_text,
        word_count=word_count,
        needs_ocr=page_needs_ocr,
        vision_recommended=page_needs_vision,
        source_span_start=source_span_start,
        source_span_end=source_span_end,
        removed_boilerplate_lines=cleaning.removed_boilerplate_lines,
        notes=notes,
    )


def _result_from_pages(
    pages: list[ExtractedPage],
    *,
    page_count: int | None,
    tables: list[ExtractedTable] | None = None,
    notes: str | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        pages=pages,
        tables=tables or [],
        page_count=page_count,
        extraction_quality=_quality_from_pages(pages),
        ocr_recommended=any(page.needs_ocr for page in pages),
        vision_recommended=any(page.vision_recommended for page in pages),
        notes=notes,
    )


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
    source_offset = 0
    for index in range(page_count):
        try:
            page = pages_proxy[index]
            raw_text = page.extract_text() or ""
            notes = None
        except Exception as exc:
            raw_text = ""
            notes = f"Could not extract text from page {index + 1}: {exc}"

        pages.append(
            _make_page(
                raw_text,
                page_number=index + 1,
                needs_ocr=None,
                vision_recommended=None,
                source_span_start=source_offset,
                notes=notes,
            )
        )
        if raw_text:
            source_offset += len(raw_text) + 2

    return _result_from_pages(
        pages,
        page_count=page_count,
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
        text_parts = _docx_text_parts(document)
        tables = _docx_tables(document)
    except Exception as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not extract text from the DOCX file: {exc}",
        )

    table_text = [table.clean_text for table in tables if table.clean_text]
    raw_text = "\n".join([*text_parts, *table_text])
    page = _make_page(
        raw_text,
        page_number=None,
        needs_ocr=False,
        source_span_start=0,
    )

    pages = _pages_with_raw_text(page)
    return _result_from_pages(
        pages,
        tables=tables,
        page_count=None,
    )


def _docx_text_parts(document: Any) -> list[str]:
    text_parts: list[str] = []
    text_parts.extend(_paragraph_text(document.paragraphs))

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

    return text_parts


def _paragraph_text(paragraphs: Iterable[Any]) -> list[str]:
    return [
        paragraph.text.strip()
        for paragraph in paragraphs
        if getattr(paragraph, "text", "").strip()
    ]


def _docx_tables(document: Any) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    table_index = 1
    for table in _iter_docx_tables(document):
        rows = _docx_table_rows(table)
        if not rows:
            continue
        tables.append(_make_table(rows, table_index=table_index, page_number=None))
        table_index += 1
    return tables


def _iter_docx_tables(document: Any) -> Iterable[Any]:
    yield from document.tables
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
            yield from container.tables


def _docx_table_rows(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [" ".join(cell.text.split()) for cell in row.cells]
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            rows.append(cells)
    return rows


def _page_failure_notes(pages: list[ExtractedPage]) -> str | None:
    page_notes = [page.notes for page in pages if page.notes]
    if not page_notes:
        return None
    return " ".join(page_notes)


def _make_table(
    rows: list[list[str]],
    *,
    table_index: int,
    page_number: int | None = None,
    source_span_start: int | None = None,
    notes: str | None = None,
) -> ExtractedTable:
    normalized_rows = [[cell.strip() for cell in row] for row in rows]
    text = "\n".join(_table_rows_as_text(normalized_rows))
    source_span_end = source_span_start + len(text) if source_span_start is not None else None
    return ExtractedTable(
        page_number=page_number,
        table_index=table_index,
        rows=normalized_rows,
        clean_text=text,
        row_count=len(normalized_rows),
        column_count=max((len(row) for row in normalized_rows), default=0),
        source_span_start=source_span_start,
        source_span_end=source_span_end,
        notes=notes,
    )


def _table_rows_as_text(rows: list[list[str]]) -> list[str]:
    return [" | ".join(cell for cell in row if cell) for row in rows if any(row)]


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

    page = _make_page(
        raw_text,
        page_number=None,
        needs_ocr=False,
        source_span_start=0,
        notes=notes,
    )

    pages = _pages_with_raw_text(page)
    return _result_from_pages(
        pages,
        page_count=1,
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

    tables = _html_tables(soup)
    raw_text = soup.get_text(separator="\n")
    page = _make_page(
        raw_text,
        page_number=None,
        needs_ocr=False,
        source_span_start=0,
        notes=notes,
    )

    pages = _pages_with_raw_text(page)
    return _result_from_pages(
        pages,
        tables=tables,
        page_count=1,
        notes=notes,
    )


def _html_tables(soup: BeautifulSoup) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    for table_index, table in enumerate(soup.find_all("table"), start=1):
        rows: list[list[str]] = []
        for row in _html_direct_table_rows(table):
            cells = [_html_cell_text(cell) for cell in row.find_all(["th", "td"], recursive=False)]
            while cells and not cells[-1]:
                cells.pop()
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(_make_table(rows, table_index=table_index))
    return tables


def _html_direct_table_rows(table: Any) -> list[Any]:
    rows: list[Any] = []
    for child in table.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
        if child.name == "tr":
            rows.append(child)
            continue
        rows.extend(child.find_all("tr", recursive=False))
    return rows


def _html_cell_text(cell: Any) -> str:
    copied_cell_soup = BeautifulSoup(str(cell), "html.parser")
    copied_cell = copied_cell_soup.find(["th", "td"])
    if copied_cell is None:
        return ""
    for nested_table in copied_cell.find_all("table"):
        nested_table.decompose()
    return " ".join(copied_cell.get_text(" ").split())


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

    try:
        rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(csv_text))]
    except csv.Error as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not parse the CSV file: {exc}",
        )
    raw_text = "\n".join(_table_rows_as_text(rows))
    tables = [_make_table(rows, table_index=1, source_span_start=0)] if rows else []
    return _single_page_result(raw_text, page_count=1, notes=notes, tables=tables)


def _extract_xlsx(path: Path) -> ExtractionResult:
    try:
        with zipfile.ZipFile(path) as workbook:
            shared_strings = _xlsx_shared_strings(workbook)
            sheet_names = sorted(
                name
                for name in workbook.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            sheet_rows = [
                _xlsx_sheet_rows(workbook.read(sheet_name), shared_strings)
                for sheet_name in sheet_names
            ]
    except (OSError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return ExtractionResult(
            pages=[],
            page_count=None,
            extraction_quality=ExtractionQuality.LOW,
            notes=f"Could not read the XLSX file: {exc}",
        )

    tables = [
        _make_table(rows, table_index=index)
        for index, rows in enumerate(sheet_rows, start=1)
        if rows
    ]
    raw_text = "\n".join(table.clean_text for table in tables if table.clean_text)
    return _single_page_result(
        raw_text,
        page_count=len(sheet_names),
        notes=None,
        tables=tables,
    )


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


def _xlsx_sheet_rows(sheet_xml: bytes, shared_strings: list[str]) -> list[list[str]]:
    root = ElementTree.fromstring(sheet_xml)
    rows: list[list[str]] = []
    for row in root.iter():
        if _xml_local_name(row.tag) != "row":
            continue
        values: list[str] = []
        for cell in row:
            if _xml_local_name(cell.tag) != "c":
                continue
            cell_index = _xlsx_cell_index(cell)
            if cell_index is not None:
                while len(values) < cell_index:
                    values.append("")
            values.append(_xlsx_cell_value(cell, shared_strings))
        while values and not values[-1]:
            values.pop()
        if any(values):
            rows.append(values)
    return rows


def _xlsx_cell_index(cell: ElementTree.Element) -> int | None:
    cell_reference = cell.attrib.get("r")
    if not cell_reference:
        return None
    match = re.match(r"([A-Za-z]+)", cell_reference)
    if not match:
        return None

    index = 0
    for character in match.group(1).upper():
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


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
    tables: list[ExtractedTable] | None = None,
) -> ExtractionResult:
    page = _make_page(
        raw_text,
        page_number=None,
        needs_ocr=False,
        source_span_start=0,
        notes=notes,
    )

    pages = _pages_with_raw_text(page)
    return _result_from_pages(
        pages,
        tables=tables,
        page_count=page_count,
        notes=notes,
    )


def _pages_with_raw_text(page: ExtractedPage) -> list[ExtractedPage]:
    if page.clean_text or page.raw_text.strip():
        return [page]
    return []
