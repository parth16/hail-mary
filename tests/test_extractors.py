from __future__ import annotations

from pathlib import Path

import pytest

from hailmary.ingest import extractors
from hailmary.ingest.extractors import extract_document
from hailmary.schemas.documents import ExtractionQuality


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


def test_missing_text_file_is_recorded_without_crashing(tmp_path: Path) -> None:
    result = extract_document(tmp_path / "missing.txt")

    assert result.pages == []
    assert result.extraction_quality == ExtractionQuality.LOW
    assert result.notes is not None
    assert "Could not read the text file" in result.notes
