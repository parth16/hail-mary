from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from hailmary.schemas.documents import IngestedDeal, IngestedDocument, SourceDocument
from hailmary.schemas.evidence import (
    ClaimConflict,
    ClaimRecord,
    ClaimType,
    EvidenceCitation,
    EvidenceKind,
    EvidenceQuality,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)
from hailmary.utils.slug import slugify

STALE_SOURCE_DAYS = 365
MONEY_PATTERN = (
    r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s?(?:thousand|million|billion|k|m|b))?"
)
PERCENT_PATTERN = r"\d+(?:\.\d+)?\s?%"
TERM_SEPARATOR = r"\s*(?:(?:is|of|at|:|\|)\s*)?"


@dataclass(frozen=True)
class DealTermPattern:
    label: str
    unit: str
    regex: re.Pattern[str]


DEAL_TERM_PATTERNS = (
    DealTermPattern(
        label="valuation cap",
        unit="usd",
        regex=re.compile(
            rf"\bvaluation cap\b{TERM_SEPARATOR}(?P<value>{MONEY_PATTERN})",
            re.IGNORECASE,
        ),
    ),
    DealTermPattern(
        label="post-money valuation",
        unit="usd",
        regex=re.compile(
            rf"\bpost[-\s]?money valuation\b{TERM_SEPARATOR}(?P<value>{MONEY_PATTERN})",
            re.IGNORECASE,
        ),
    ),
    DealTermPattern(
        label="pre-money valuation",
        unit="usd",
        regex=re.compile(
            rf"\bpre[-\s]?money valuation\b{TERM_SEPARATOR}(?P<value>{MONEY_PATTERN})",
            re.IGNORECASE,
        ),
    ),
    DealTermPattern(
        label="round size",
        unit="usd",
        regex=re.compile(
            rf"\b(?:round size|target raise|raising|raise)\b"
            rf"{TERM_SEPARATOR}(?P<value>{MONEY_PATTERN})",
            re.IGNORECASE,
        ),
    ),
    DealTermPattern(
        label="minimum investment",
        unit="usd",
        regex=re.compile(
            rf"\b(?:minimum investment|minimum check|min check)\b"
            rf"{TERM_SEPARATOR}(?P<value>{MONEY_PATTERN})",
            re.IGNORECASE,
        ),
    ),
    DealTermPattern(
        label="discount",
        unit="percent",
        regex=re.compile(
            rf"\bdiscount\b{TERM_SEPARATOR}(?P<value>{PERCENT_PATTERN})|"
            rf"(?P<value_before>{PERCENT_PATTERN})\s*\bdiscount\b",
            re.IGNORECASE,
        ),
    ),
)


def build_evidence_store(
    deal: IngestedDeal,
    *,
    created_at: datetime | None = None,
) -> EvidenceStore:
    """Build source-linked evidence and deterministic claims for one deal."""

    built_at = created_at or datetime.now(UTC)
    evidence_records: list[EvidenceRecord] = []
    for document in deal.documents:
        evidence_records.extend(_document_evidence_records(document, built_at=built_at))

    claims = _extract_deal_term_claims(
        deal_id=deal.id,
        evidence_records=evidence_records,
    )
    conflicts = _find_conflicts(deal.id, claims)
    conflicted_claim_ids = {claim_id for conflict in conflicts for claim_id in conflict.claim_ids}
    if conflicted_claim_ids:
        claims = [
            claim.model_copy(
                update={
                    "verification_status": VerificationStatus.CONFLICTED,
                    "quality": claim.quality.model_copy(
                        update={
                            "verification_status": VerificationStatus.CONFLICTED,
                            "confidence": 0.2,
                            "score_impact": "excluded_until_conflict_is_resolved",
                        }
                    ),
                }
            )
            if claim.id in conflicted_claim_ids
            else claim
            for claim in claims
        ]

    notes = []
    if not evidence_records:
        notes.append("No usable extracted text was available for evidence records.")
    if conflicts:
        notes.append("Some extracted deal terms conflict and need human review.")

    return EvidenceStore(
        deal_id=deal.id,
        company_name=deal.company_name,
        created_at=built_at,
        evidence=evidence_records,
        claims=claims,
        conflicts=conflicts,
        notes=notes,
    )


def _document_evidence_records(
    document: IngestedDocument,
    *,
    built_at: datetime,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    source = document.source
    source_freshness = _source_freshness(source, now=built_at)
    table_texts = [
        table.clean_text.strip()
        for table in document.tables
        if table.clean_text.strip()
    ]

    for page in document.pages:
        text = _page_text_without_table_evidence(page.clean_text.strip(), table_texts)
        if not text:
            continue
        span_start, span_end = _safe_clean_page_span(
            page_raw_text=page.raw_text,
            text=text,
            source_span_start=page.source_span_start,
        )
        records.append(
            EvidenceRecord(
                id=_evidence_id(
                    source.id,
                    "page",
                    str(page.page_number or 0),
                    text,
                ),
                deal_id=source.deal_id,
                document_id=source.id,
                document_path=source.path,
                evidence_kind=EvidenceKind.PAGE_TEXT,
                source_kind=source.source_kind,
                document_type=source.document_type,
                file_type=source.file_type,
                text=text,
                page_number=page.page_number,
                source_span_start=span_start,
                source_span_end=span_end,
                source_freshness=source_freshness,
            )
        )

    for table in document.tables:
        text = table.clean_text.strip()
        if not text:
            continue
        records.append(
            EvidenceRecord(
                id=_evidence_id(
                    source.id,
                    "table",
                    str(table.table_index),
                    text,
                ),
                deal_id=source.deal_id,
                document_id=source.id,
                document_path=source.path,
                evidence_kind=EvidenceKind.TABLE_TEXT,
                source_kind=source.source_kind,
                document_type=source.document_type,
                file_type=source.file_type,
                text=text,
                page_number=table.page_number,
                table_index=table.table_index,
                source_span_start=table.source_span_start,
                source_span_end=table.source_span_end,
                source_freshness=source_freshness,
            )
        )

    return records


def _page_text_without_table_evidence(page_text: str, table_texts: list[str]) -> str:
    remaining_text = page_text
    for table_text in table_texts:
        remaining_text = remaining_text.replace(table_text, "", 1)
    return remaining_text.strip()


def _safe_clean_page_span(
    *,
    page_raw_text: str,
    text: str,
    source_span_start: int | None,
) -> tuple[int | None, int | None]:
    if text != page_raw_text.strip():
        return None, None
    if source_span_start is None:
        return None, None
    local_start = len(page_raw_text) - len(page_raw_text.lstrip())
    start = source_span_start + local_start
    return start, start + len(text)


def _source_freshness(source: SourceDocument, *, now: datetime) -> SourceFreshness:
    source_date = source.created_at or source.retrieved_at
    if source_date is None:
        return SourceFreshness.UNKNOWN
    if source_date.tzinfo is None:
        source_date = source_date.replace(tzinfo=UTC)
    age_days = (now - source_date).days
    if age_days > STALE_SOURCE_DAYS:
        return SourceFreshness.STALE
    return SourceFreshness.CURRENT


def _extract_deal_term_claims(
    *,
    deal_id: str,
    evidence_records: list[EvidenceRecord],
) -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    seen_claims: set[tuple[str, str, str, str]] = set()

    evidence_by_id = {evidence.id: evidence for evidence in evidence_records}
    for evidence in evidence_records:
        for pattern in DEAL_TERM_PATTERNS:
            for match in pattern.regex.finditer(evidence.text):
                raw_value = match.groupdict().get("value") or match.groupdict().get(
                    "value_before"
                )
                if not raw_value:
                    continue
                normalized_value = _normalize_value(raw_value, unit=pattern.unit)
                citation = _citation_from_match(evidence, match)
                if citation is None:
                    continue
                quote = citation.quote
                if not quote:
                    continue
                claim_key = (pattern.label, normalized_value, evidence.id, quote)
                if claim_key in seen_claims:
                    continue
                seen_claims.add(claim_key)

                citation_status = verify_citation(citation, evidence_by_id)
                citation = citation.model_copy(
                    update={"verification_status": citation_status}
                )
                claim_status = _claim_verification_status([citation])
                claim_id = _claim_id(
                    deal_id=deal_id,
                    label=pattern.label,
                    normalized_value=normalized_value,
                    evidence_id=evidence.id,
                    source_span_start=citation.source_span_start,
                )
                quality = EvidenceQuality(
                    claim_type=ClaimType.DEAL_TERM,
                    source_type=evidence.source_kind,
                    verification_status=claim_status,
                    recency=evidence.source_freshness,
                    reliability="extracted_from_source_document",
                    confidence=0.72 if claim_status == VerificationStatus.VERIFIED else 0.35,
                    materiality="high",
                )
                claims.append(
                    ClaimRecord(
                        id=claim_id,
                        deal_id=deal_id,
                        claim_type=ClaimType.DEAL_TERM,
                        label=pattern.label,
                        value=raw_value.strip(),
                        normalized_value=normalized_value,
                        unit=pattern.unit,
                        raw_text=quote,
                        citations=[citation],
                        verification_status=claim_status,
                        quality=quality,
                    )
                )

    return claims


def _citation_from_match(
    evidence: EvidenceRecord,
    match: re.Match[str],
) -> EvidenceCitation | None:
    start = match.start()
    end = match.end()
    while start < end and evidence.text[start].isspace():
        start += 1
    while end > start and evidence.text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return EvidenceCitation(
        evidence_id=evidence.id,
        quote=evidence.text[start:end],
        source_span_start=start,
        source_span_end=end,
    )


def verify_citation(
    citation: EvidenceCitation,
    evidence_by_id: dict[str, EvidenceRecord],
) -> VerificationStatus:
    evidence = evidence_by_id.get(citation.evidence_id)
    if evidence is None:
        return VerificationStatus.EVIDENCE_NOT_FOUND
    if citation.source_span_start < 0 or citation.source_span_end > len(evidence.text):
        return VerificationStatus.SPAN_MISMATCH
    if citation.source_span_start >= citation.source_span_end:
        return VerificationStatus.SPAN_MISMATCH
    if evidence.text[citation.source_span_start : citation.source_span_end] != citation.quote:
        return VerificationStatus.QUOTE_MISMATCH
    return VerificationStatus.VERIFIED


def _claim_verification_status(
    citations: list[EvidenceCitation],
) -> VerificationStatus:
    if not citations:
        return VerificationStatus.MISSING_CITATION
    for citation in citations:
        if citation.verification_status != VerificationStatus.VERIFIED:
            return citation.verification_status
    return VerificationStatus.VERIFIED


def _find_conflicts(deal_id: str, claims: list[ClaimRecord]) -> list[ClaimConflict]:
    grouped: dict[tuple[ClaimType, str], dict[str, list[str]]] = {}
    for claim in claims:
        key = (claim.claim_type, claim.label)
        grouped.setdefault(key, {}).setdefault(claim.normalized_value, []).append(claim.id)

    conflicts: list[ClaimConflict] = []
    for (claim_type, label), values in grouped.items():
        if len(values) <= 1:
            continue
        normalized_values = sorted(values)
        claim_ids = sorted(claim_id for claim_ids in values.values() for claim_id in claim_ids)
        conflicts.append(
            ClaimConflict(
                id=_conflict_id(deal_id, claim_type, label, normalized_values),
                deal_id=deal_id,
                claim_type=claim_type,
                label=label,
                normalized_values=normalized_values,
                claim_ids=claim_ids,
                notes=f"Multiple values were extracted for {label}. Review the cited evidence.",
            )
        )
    return conflicts


def _normalize_value(raw_value: str, *, unit: str) -> str:
    if unit == "usd":
        cents = _money_to_cents(raw_value)
        return f"usd_cents:{cents}" if cents is not None else f"usd_text:{raw_value.strip()}"
    if unit == "percent":
        basis_points = _percent_to_basis_points(raw_value)
        return (
            f"basis_points:{basis_points}"
            if basis_points is not None
            else f"percent_text:{raw_value.strip()}"
        )
    return raw_value.strip()


def _money_to_cents(raw_value: str) -> int | None:
    normalized = raw_value.lower().replace("$", "").replace(",", "").strip()
    multiplier = Decimal("1")
    suffixes = {
        "thousand": Decimal("1000"),
        "million": Decimal("1000000"),
        "billion": Decimal("1000000000"),
        "k": Decimal("1000"),
        "m": Decimal("1000000"),
        "b": Decimal("1000000000"),
    }
    for suffix, suffix_multiplier in suffixes.items():
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            multiplier = suffix_multiplier
            break
    try:
        return int((Decimal(normalized) * multiplier * 100).to_integral_value())
    except InvalidOperation:
        return None


def _percent_to_basis_points(raw_value: str) -> int | None:
    normalized = raw_value.lower().replace("%", "").strip()
    try:
        return int((Decimal(normalized) * 100).to_integral_value())
    except InvalidOperation:
        return None


def _evidence_id(document_id: str, kind: str, locator: str, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{kind}:{locator}:{text}".encode()).hexdigest()
    return f"ev_{digest[:16]}"


def _claim_id(
    *,
    deal_id: str,
    label: str,
    normalized_value: str,
    evidence_id: str,
    source_span_start: int,
) -> str:
    digest = hashlib.sha256(
        f"{deal_id}:{label}:{normalized_value}:{evidence_id}:{source_span_start}".encode()
    ).hexdigest()
    return f"claim_{slugify(label)}_{digest[:12]}"


def _conflict_id(
    deal_id: str,
    claim_type: ClaimType,
    label: str,
    normalized_values: list[str],
) -> str:
    digest = hashlib.sha256(
        f"{deal_id}:{claim_type}:{label}:{'|'.join(normalized_values)}".encode()
    ).hexdigest()
    return f"conflict_{slugify(label)}_{digest[:12]}"
