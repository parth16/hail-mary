from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from hailmary.ingest import extractors
from hailmary.ingest.extractors import extract_document
from hailmary.schemas.documents import ExtractedPage, ExtractionQuality


def test_html_extraction_removes_markup_and_scripts(tmp_path: Path) -> None:
    html_path = tmp_path / "deal.html"
    html_path.write_text(
        """
        <html>
          <head><script>window.secret = "ignore";</script></head>
          <body><h1>Acme Deal</h1><p>Customer traction is growing.</p></body>
        </html>
        """,
        encoding="utf-8",
    )

    result = extract_document(html_path)

    assert result.pages
    assert "Acme Deal" in result.combined_text
    assert "Customer traction is growing." in result.combined_text
    assert "script" not in result.combined_text.lower()
    assert "window.secret" not in result.combined_text


def test_pdf_page_access_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "locked.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class BrokenReader:
        @property
        def pages(self) -> object:
            raise RuntimeError("PDF is locked")

    monkeypatch.setattr(extractors, "PdfReader", lambda _: BrokenReader())

    result = extract_document(pdf_path)

    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not read pages" in result.notes


def test_pdf_quality_is_capped_when_most_pages_need_ocr() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            raw_text=" ".join(["traction"] * 220),
            clean_text=" ".join(["traction"] * 220),
            word_count=220,
            needs_ocr=False,
        ),
        ExtractedPage(page_number=2, raw_text="", clean_text="", word_count=0, needs_ocr=True),
        ExtractedPage(page_number=3, raw_text="", clean_text="", word_count=0, needs_ocr=True),
        ExtractedPage(page_number=4, raw_text="", clean_text="", word_count=0, needs_ocr=True),
    ]

    assert extractors._quality_from_pages(pages) == ExtractionQuality.MEDIUM


def test_docx_text_traversal_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docx_path = tmp_path / "broken.docx"
    docx_path.write_bytes(b"placeholder")

    def fail_text_parts(document: object) -> list[str]:
        raise RuntimeError("document part is malformed")

    monkeypatch.setattr(extractors, "Document", lambda _: object())
    monkeypatch.setattr(extractors, "_docx_text_parts", fail_text_parts)

    result = extract_document(docx_path)

    assert result.pages == []
    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not extract text from the DOCX file" in result.notes


def test_missing_text_file_is_recorded_without_crashing(tmp_path: Path) -> None:
    result = extract_document(tmp_path / "missing.txt")

    assert result.pages == []
    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not read the text file" in result.notes


def test_csv_extraction_records_table_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "model.csv"
    csv_path.write_text("year,revenue\n2026,100\n", encoding="utf-8")

    result = extract_document(csv_path)

    assert result.pages
    assert "year | revenue" in result.combined_text
    assert "2026 | 100" in result.combined_text


def test_csv_parser_error_is_recorded_without_crashing(tmp_path: Path) -> None:
    csv_path = tmp_path / "model.csv"
    csv_path.write_text(f"notes\n{'A' * 32}\n", encoding="utf-8")
    original_limit = csv.field_size_limit()
    csv.field_size_limit(10)
    try:
        result = extract_document(csv_path)
    finally:
        csv.field_size_limit(original_limit)

    assert result.pages == []
    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not parse the CSV file" in result.notes


def test_xlsx_extraction_records_table_text(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "model.xlsx"
    with zipfile.ZipFile(xlsx_path, "w") as workbook:
        workbook.writestr(
            "xl/sharedStrings.xml",
            """
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Revenue</t></si>
            </sst>
            """,
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row><c t="s"><v>0</v></c><c><v>100</v></c></row>
              </sheetData>
            </worksheet>
            """,
        )

    result = extract_document(xlsx_path)

    assert result.pages
    assert "Revenue | 100" in result.combined_text


def test_xlsx_negative_shared_string_index_is_left_raw(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "model.xlsx"
    with zipfile.ZipFile(xlsx_path, "w") as workbook:
        workbook.writestr(
            "xl/sharedStrings.xml",
            """
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Do not use this string</t></si>
            </sst>
            """,
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            """
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row><c t="s"><v>-1</v></c></row>
              </sheetData>
            </worksheet>
            """,
        )

    result = extract_document(xlsx_path)

    assert result.pages
    assert "-1" in result.combined_text
    assert "Do not use this string" not in result.combined_text


def test_xlsx_worksheet_read_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xlsx_path = tmp_path / "encrypted.xlsx"
    xlsx_path.write_bytes(b"placeholder")

    class BrokenWorkbook:
        def __enter__(self) -> BrokenWorkbook:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def namelist(self) -> list[str]:
            return ["xl/worksheets/sheet1.xml"]

        def read(self, name: str) -> bytes:
            raise RuntimeError(f"{name} is encrypted")

    monkeypatch.setattr(zipfile, "ZipFile", lambda _: BrokenWorkbook())

    result = extract_document(xlsx_path)

    assert result.pages == []
    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not read the XLSX file" in result.notes
