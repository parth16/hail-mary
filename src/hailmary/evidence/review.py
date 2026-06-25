from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from hailmary.config import AppConfig
from hailmary.evidence.store import verify_citation
from hailmary.ingest.ocr import LOW_OCR_CONFIDENCE_THRESHOLD
from hailmary.schemas.documents import IngestedDeal, IngestionSummary, SourceKind
from hailmary.schemas.evidence import (
    ClaimConflict,
    ClaimRecord,
    EvidenceCitation,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)

LOW_CLAIM_CONFIDENCE_THRESHOLD = 0.5


class EvidenceReviewError(RuntimeError):
    """Evidence review could not safely read local generated evidence."""


@dataclass(frozen=True)
class SourceDocumentEvidenceSummary:
    document_id: str
    document_path: Path
    source_kind: SourceKind
    evidence_count: int
    missing_source_span_count: int
    ocr_applied_count: int
    stale_count: int
    unknown_freshness_count: int


@dataclass(frozen=True)
class ClaimStatusSummary:
    label: str
    verification_status: VerificationStatus
    count: int


@dataclass(frozen=True)
class ConflictReviewSummary:
    label: str
    status: Literal["active", "stale"]
    normalized_values: list[str]
    claim_count: int
    notes: str
    why_it_matters: str


@dataclass(frozen=True)
class ReviewIssueSummary:
    issue: str
    count: int
    guidance: str


@dataclass(frozen=True)
class DealEvidenceReview:
    deal_id: str
    company_name: str
    evidence_store_path: Path
    evidence_count: int
    claim_count: int
    conflict_count: int
    source_documents: list[SourceDocumentEvidenceSummary]
    claim_statuses: list[ClaimStatusSummary]
    conflicts: list[ConflictReviewSummary]
    issues: list[ReviewIssueSummary]
    evidence_records: list[EvidenceRecord]


@dataclass(frozen=True)
class EvidenceReviewResult:
    data_dir: Path
    summary_path: Path
    deals: list[DealEvidenceReview]


def review_evidence(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
    all_deals: bool = False,
    evidence_id: str | None = None,
) -> EvidenceReviewResult:
    """Read local generated evidence stores and return operator review summaries."""

    _validate_selector(deal_id=deal_id, company_name=company_name, all_deals=all_deals)
    if evidence_id is not None and not evidence_id.strip():
        raise EvidenceReviewError("The --evidence-id value cannot be blank.")
    evidence_id = evidence_id.strip() if evidence_id is not None else None
    configured_data_dir = _absolute_path(config.data_dir)
    _ensure_local_data(configured_data_dir)
    data_dir = configured_data_dir.resolve(strict=False)
    summary_path = data_dir / "processed" / "ingestion_summary.json"
    if not summary_path.exists():
        raise EvidenceReviewError(
            "No ingestion summary found. Run `hailmary ingest-folder` before reviewing "
            "evidence."
        )

    summary = _load_ingestion_summary(summary_path)
    selected_deals = _select_deals(
        summary.deals,
        deal_id=deal_id,
        company_name=company_name,
        all_deals=all_deals,
    )
    deal_reviews: list[DealEvidenceReview] = []
    evidence_id_found = evidence_id is None
    for deal in selected_deals:
        store_path = _evidence_store_path_for_deal(
            deal,
            data_dir=data_dir,
            summary_path=summary_path,
        )
        store = _load_evidence_store(store_path, company_name=deal.company_name)
        if evidence_id is not None and any(
            evidence.id == evidence_id for evidence in store.evidence
        ):
            evidence_id_found = True
        deal_reviews.append(
            _deal_review(
                store,
                evidence_store_path=store_path,
                evidence_id=evidence_id,
            )
        )

    if not evidence_id_found:
        raise EvidenceReviewError(
            f"No selected deal has evidence record {evidence_id}. Check the evidence ID "
            "or choose a different deal."
        )

    return EvidenceReviewResult(
        data_dir=data_dir,
        summary_path=summary_path,
        deals=deal_reviews,
    )


def _validate_selector(
    *,
    deal_id: str | None,
    company_name: str | None,
    all_deals: bool,
) -> None:
    selector_count = sum(
        [
            deal_id is not None and deal_id.strip() != "",
            company_name is not None and company_name.strip() != "",
            all_deals,
        ]
    )
    if selector_count > 1:
        raise EvidenceReviewError(
            "Choose only one deal selector: --deal-id, --company, or --all."
        )
    if deal_id is not None and not deal_id.strip():
        raise EvidenceReviewError("The --deal-id value cannot be blank.")
    if company_name is not None and not company_name.strip():
        raise EvidenceReviewError("The --company value cannot be blank.")


def _ensure_local_data(data_dir: Path) -> None:
    if data_dir.is_symlink():
        raise EvidenceReviewError(
            f"The local data directory at {data_dir} is a symlink. Choose the real "
            "Hail Mary data folder."
        )
    for parent in data_dir.parents:
        if parent.is_symlink():
            raise EvidenceReviewError(
                f"Hail Mary cannot review evidence at {data_dir} because {parent} is "
                "a symlinked parent folder."
            )
    if not data_dir.exists():
        raise EvidenceReviewError(
            "No local data initialized. Run `hailmary init` and `hailmary ingest-folder` "
            "before reviewing evidence."
        )
    if not data_dir.is_dir():
        raise EvidenceReviewError(
            f"Hail Mary local data at {data_dir} is not a folder. Choose the data "
            "folder created by `hailmary init`."
        )


def _load_ingestion_summary(summary_path: Path) -> IngestionSummary:
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceReviewError(
            "The ingestion summary is not plain text. Run `hailmary ingest-folder` again "
            "before reviewing evidence."
        ) from exc
    except OSError as exc:
        raise EvidenceReviewError(
            f"Could not read the ingestion summary at {summary_path}: {exc}"
        ) from exc
    try:
        return IngestionSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise EvidenceReviewError(
            "The ingestion summary could not be read. Run `hailmary ingest-folder` again "
            "before reviewing evidence."
        ) from exc


def _select_deals(
    deals: list[IngestedDeal],
    *,
    deal_id: str | None,
    company_name: str | None,
    all_deals: bool,
) -> list[IngestedDeal]:
    if not deals:
        raise EvidenceReviewError(
            "The latest ingestion summary has no deals. Run `hailmary ingest-folder` "
            "with deal documents before reviewing evidence."
        )
    if all_deals:
        return deals
    if deal_id is not None:
        normalized_deal_id = deal_id.strip()
        matches = [deal for deal in deals if deal.id == normalized_deal_id]
        if not matches:
            raise EvidenceReviewError(
                f"No ingested deal has deal ID {normalized_deal_id}."
            )
        return matches
    if company_name is not None:
        normalized_name = company_name.strip().casefold()
        matches = [deal for deal in deals if deal.company_name.casefold() == normalized_name]
        if not matches:
            raise EvidenceReviewError(
                f"No ingested deal has exact company name {company_name.strip()}."
            )
        if len(matches) > 1:
            raise EvidenceReviewError(
                f"Company name {company_name.strip()} matches more than one ingested deal. "
                "Use --deal-id instead."
            )
        return matches
    if len(deals) > 1:
        raise EvidenceReviewError(
            f"The latest ingestion summary has {len(deals)} deals. Use --deal-id, "
            "--company, or --all to choose which evidence to review."
        )
    return deals


def _evidence_store_path_for_deal(
    deal: IngestedDeal,
    *,
    data_dir: Path,
    summary_path: Path,
) -> Path:
    if deal.evidence_store_path is None:
        raise EvidenceReviewError(
            f"No evidence store was found for {deal.company_name}. Run "
            "`hailmary ingest-folder` again before reviewing evidence."
        )
    store_path = _resolve_saved_path(
        deal.evidence_store_path,
        data_dir=data_dir,
        summary_path=summary_path,
    )
    if not store_path.exists():
        raise EvidenceReviewError(
            f"The evidence store for {deal.company_name} is missing at {store_path}. "
            "Run `hailmary ingest-folder` again before reviewing evidence."
        )
    return store_path


def _load_evidence_store(path: Path, *, company_name: str) -> EvidenceStore:
    try:
        raw_store = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceReviewError(
            f"The evidence store for {company_name} is not plain text. Run "
            "`hailmary ingest-folder` again before reviewing evidence."
        ) from exc
    except OSError as exc:
        raise EvidenceReviewError(
            f"Could not read the evidence store for {company_name} at {path}: {exc}"
        ) from exc
    try:
        return EvidenceStore.model_validate_json(raw_store)
    except ValidationError as exc:
        raise EvidenceReviewError(
            f"The evidence store for {company_name} could not be read. Run "
            "`hailmary ingest-folder` again before reviewing evidence."
        ) from exc


def _resolve_saved_path(path: Path, *, data_dir: Path, summary_path: Path) -> Path:
    absolute_data_dir = _absolute_path(data_dir).resolve(strict=False)
    if path.is_absolute():
        resolved_path = path.resolve(strict=False)
        if not _is_relative_to(resolved_path, absolute_data_dir):
            raise EvidenceReviewError(
                f"The evidence store path {path} is outside the private data directory."
            )
        return resolved_path

    absolute_summary_path = _absolute_path(summary_path).resolve(strict=False)
    candidate_roots = [
        Path.cwd().resolve(strict=False),
        *absolute_data_dir.parents,
        absolute_data_dir,
        absolute_summary_path.parent,
    ]
    candidates = [root / path for root in candidate_roots]
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir) and candidate.exists():
            return resolved_candidate
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir):
            return resolved_candidate
    return (absolute_data_dir / path.name).resolve(strict=False)


def _deal_review(
    store: EvidenceStore,
    *,
    evidence_store_path: Path,
    evidence_id: str | None,
) -> DealEvidenceReview:
    evidence_records = [
        evidence for evidence in store.evidence if evidence_id is None or evidence.id == evidence_id
    ]
    return DealEvidenceReview(
        deal_id=store.deal_id,
        company_name=store.company_name,
        evidence_store_path=evidence_store_path,
        evidence_count=store.evidence_count,
        claim_count=store.claim_count,
        conflict_count=store.conflict_count,
        source_documents=_source_document_summaries(store.evidence),
        claim_statuses=_claim_status_summaries(store),
        conflicts=_conflict_summaries(store),
        issues=_issue_summaries(store),
        evidence_records=evidence_records,
    )


def _source_document_summaries(
    evidence_records: list[EvidenceRecord],
) -> list[SourceDocumentEvidenceSummary]:
    grouped: dict[str, list[EvidenceRecord]] = {}
    for evidence in evidence_records:
        grouped.setdefault(evidence.document_id, []).append(evidence)

    summaries = []
    for document_id, records in grouped.items():
        summaries.append(
            SourceDocumentEvidenceSummary(
                document_id=document_id,
                document_path=records[0].document_path,
                source_kind=records[0].source_kind,
                evidence_count=len(records),
                missing_source_span_count=sum(
                    1 for evidence in records if _missing_source_span(evidence)
                ),
                ocr_applied_count=sum(1 for evidence in records if evidence.ocr_applied),
                stale_count=sum(
                    1
                    for evidence in records
                    if evidence.source_freshness == SourceFreshness.STALE
                ),
                unknown_freshness_count=sum(
                    1
                    for evidence in records
                    if evidence.source_freshness == SourceFreshness.UNKNOWN
                ),
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (
            summary.document_path.as_posix().casefold(),
            summary.document_id.casefold(),
        ),
    )


def _claim_status_summaries(store: EvidenceStore) -> list[ClaimStatusSummary]:
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    counts = Counter(
        (claim.label, _claim_review_status(claim, evidence_by_id))
        for claim in store.claims
    )
    return [
        ClaimStatusSummary(label=label, verification_status=status, count=count)
        for (label, status), count in sorted(
            counts.items(),
            key=lambda item: (item[0][0].casefold(), item[0][1].value),
        )
    ]


def _claim_review_status(
    claim: ClaimRecord,
    evidence_by_id: dict[str, EvidenceRecord],
) -> VerificationStatus:
    if not claim.citations:
        return VerificationStatus.MISSING_CITATION
    for citation in claim.citations:
        if citation.verification_status != VerificationStatus.VERIFIED:
            return citation.verification_status
        live_status = verify_citation(citation, evidence_by_id)
        if live_status != VerificationStatus.VERIFIED:
            return live_status
    if claim.verification_status == VerificationStatus.CONFLICTED:
        return VerificationStatus.CONFLICTED
    return VerificationStatus.VERIFIED


def _conflict_summaries(store: EvidenceStore) -> list[ConflictReviewSummary]:
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    claim_by_id = {claim.id: claim for claim in store.claims}
    summaries: list[ConflictReviewSummary] = []
    for conflict in store.conflicts:
        valid_claims = [
            claim_by_id[claim_id]
            for claim_id in conflict.claim_ids
            if claim_id in claim_by_id
            and _claim_citations_are_valid(claim_by_id[claim_id], evidence_by_id)
            and claim_by_id[claim_id].verification_status
            in {VerificationStatus.VERIFIED, VerificationStatus.CONFLICTED}
        ]
        valid_values = sorted({claim.normalized_value for claim in valid_claims})
        active = len(valid_claims) >= 2 and len(valid_values) >= 2
        summaries.append(
            ConflictReviewSummary(
                label=conflict.label,
                status="active" if active else "stale",
                normalized_values=valid_values if active else conflict.normalized_values,
                claim_count=len(valid_claims) if active else len(conflict.claim_ids),
                notes=conflict.notes,
                why_it_matters=_conflict_guidance(conflict, active=active),
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (summary.status != "active", summary.label.casefold()),
    )


def _conflict_guidance(conflict: ClaimConflict, *, active: bool) -> str:
    if active:
        return (
            f"Multiple still-valid values were extracted for {conflict.label}. "
            "Resolve this before scoring or model review."
        )
    return (
        f"This stored {conflict.label} conflict no longer has enough valid citation "
        "support. Re-ingest or refresh claims if it still matters."
    )


def _issue_summaries(store: EvidenceStore) -> list[ReviewIssueSummary]:
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    issues = [
        _issue(
            "No usable evidence",
            1 if not store.evidence else 0,
            "Re-run ingestion after adding readable source documents or enabling needed OCR.",
        ),
        _issue(
            "OCR-applied evidence",
            sum(1 for evidence in store.evidence if evidence.ocr_applied),
            "Review image-read text against the source document before relying on it.",
        ),
        _issue(
            "Low-confidence OCR",
            sum(
                1
                for evidence in store.evidence
                if evidence.ocr_applied
                and evidence.ocr_confidence is not None
                and evidence.ocr_confidence < LOW_OCR_CONFIDENCE_THRESHOLD
            ),
            "Treat low-confidence image-read text as needing source review.",
        ),
        _issue(
            "Stale source freshness",
            sum(
                1
                for evidence in store.evidence
                if evidence.source_freshness == SourceFreshness.STALE
            ),
            "Find newer evidence or confirm that the old source is still accurate.",
        ),
        _issue(
            "Unknown source freshness",
            sum(
                1
                for evidence in store.evidence
                if evidence.source_freshness == SourceFreshness.UNKNOWN
            ),
            "Confirm when the source was created or retrieved.",
        ),
        _issue(
            "Missing source spans",
            sum(1 for evidence in store.evidence if _missing_source_span(evidence)),
            "The evidence can be reviewed, but its exact source-text position is missing.",
        ),
        _issue(
            "Missing claim citations",
            sum(1 for claim in store.claims if not claim.citations),
            "Claims without citations should not be relied on until source evidence is linked.",
        ),
        _issue(
            "Invalid claim citations",
            sum(
                1
                for claim in store.claims
                if claim.citations and not _claim_citations_are_valid(claim, evidence_by_id)
            ),
            "Citation spans no longer match the stored evidence text.",
        ),
        _issue(
            "Low-confidence claims",
            sum(
                1
                for claim in store.claims
                if claim.quality.confidence < LOW_CLAIM_CONFIDENCE_THRESHOLD
            ),
            "Review these claims before scoring or model review.",
        ),
        _issue(
            "External evidence missing confidence notes",
            sum(
                1
                for evidence in store.evidence
                if evidence.source_kind != SourceKind.LOCAL_FILE
                and not (evidence.external_confidence or "").strip()
            ),
            "External research evidence should include an operator confidence note.",
        ),
    ]
    return [issue for issue in issues if issue.count > 0]


def _issue(issue: str, count: int, guidance: str) -> ReviewIssueSummary:
    return ReviewIssueSummary(issue=issue, count=count, guidance=guidance)


def _claim_citations_are_valid(
    claim: ClaimRecord,
    evidence_by_id: dict[str, EvidenceRecord],
) -> bool:
    if not claim.citations:
        return False
    return all(_citation_is_valid(citation, evidence_by_id) for citation in claim.citations)


def _citation_is_valid(
    citation: EvidenceCitation,
    evidence_by_id: dict[str, EvidenceRecord],
) -> bool:
    if citation.verification_status != VerificationStatus.VERIFIED:
        return False
    return verify_citation(citation, evidence_by_id) == VerificationStatus.VERIFIED


def _missing_source_span(evidence: EvidenceRecord) -> bool:
    start = evidence.source_span_start
    end = evidence.source_span_end
    return start is None or end is None or start < 0 or end <= start


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
