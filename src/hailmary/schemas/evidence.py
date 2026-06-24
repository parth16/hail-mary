from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from hailmary.schemas.documents import DocumentType, FileType, SourceKind


class EvidenceKind(StrEnum):
    PAGE_TEXT = "page_text"
    TABLE_TEXT = "table_text"


class ClaimType(StrEnum):
    DEAL_TERM = "deal_term"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    MISSING_CITATION = "missing_citation"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    SPAN_MISMATCH = "span_mismatch"
    QUOTE_MISMATCH = "quote_mismatch"
    CONFLICTED = "conflicted"


class SourceFreshness(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class EvidenceQuality(BaseModel):
    claim_type: ClaimType
    source_type: SourceKind
    verification_status: VerificationStatus
    recency: SourceFreshness = SourceFreshness.UNKNOWN
    reliability: str
    confidence: float
    materiality: str
    score_impact: str = "not_scored_yet"


class EvidenceRecord(BaseModel):
    id: str
    deal_id: str
    document_id: str
    document_path: Path
    evidence_kind: EvidenceKind
    source_kind: SourceKind
    document_type: DocumentType
    file_type: FileType
    text: str
    page_number: int | None = None
    table_index: int | None = None
    source_span_start: int | None = None
    source_span_end: int | None = None
    source_freshness: SourceFreshness = SourceFreshness.UNKNOWN


class EvidenceCitation(BaseModel):
    evidence_id: str
    quote: str
    source_span_start: int
    source_span_end: int
    verification_status: VerificationStatus = VerificationStatus.MISSING_CITATION


class ClaimRecord(BaseModel):
    id: str
    deal_id: str
    claim_type: ClaimType
    label: str
    value: str
    normalized_value: str
    unit: str | None = None
    raw_text: str
    citations: list[EvidenceCitation] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.MISSING_CITATION
    quality: EvidenceQuality


class ClaimConflict(BaseModel):
    id: str
    deal_id: str
    claim_type: ClaimType
    label: str
    normalized_values: list[str]
    claim_ids: list[str]
    notes: str


class EvidenceStore(BaseModel):
    version: str = "1"
    deal_id: str
    company_name: str
    created_at: datetime
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)
    conflicts: list[ClaimConflict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)
