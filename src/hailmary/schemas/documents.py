from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    MERIDIAN = "meridian"
    WEB = "web"
    MANUAL_NOTE = "manual_note"


class DocumentType(StrEnum):
    PLATFORM_DEAL_PAGE = "platform_deal_page"
    PITCH_DECK = "pitch_deck"
    LEGAL_DOCUMENT = "legal_document"
    FINANCIAL_MODEL = "financial_model"
    CUSTOMER_DOCUMENT = "customer_document"
    MEMO = "memo"
    WEB_PAGE = "web_page"
    UNKNOWN = "unknown"


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    PNG = "png"
    JPG = "jpg"
    UNKNOWN = "unknown"


class ExtractionQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceDocument(BaseModel):
    id: str
    deal_id: str
    path: Path
    source_url: str | None = None
    source_kind: SourceKind
    document_type: DocumentType
    file_type: FileType
    title: str
    company_name: str | None = None
    created_at: datetime | None = None
    ingested_at: datetime
    retrieved_at: datetime | None = None
    page_count: int | None = None
    sha256: str
    confidentiality_detected: bool = False
    extraction_quality: ExtractionQuality
    notes: str | None = None


class ExtractedPage(BaseModel):
    page_number: int | None
    raw_text: str
    clean_text: str
    word_count: int = 0
    needs_ocr: bool = False
    notes: str | None = None


class IngestedDocument(BaseModel):
    source: SourceDocument
    pages: list[ExtractedPage] = Field(default_factory=list)
    output_path: Path


class IngestedDeal(BaseModel):
    id: str
    company_name: str
    documents: list[IngestedDocument] = Field(default_factory=list)


class IngestionSummary(BaseModel):
    root_path: Path
    scanned_at: datetime
    deals: list[IngestedDeal] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    summary_path: Path

    @property
    def document_count(self) -> int:
        return sum(len(deal.documents) for deal in self.deals)
