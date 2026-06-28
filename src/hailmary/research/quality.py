from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from hailmary.schemas.documents import DocumentType, SourceKind
from hailmary.schemas.evidence import (
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    SourceReliability,
)

from .matching import CompanyMatch, CompanyMatchKind, classify_company_match
from .schemas import ResearchResultInput

STALE_SOURCE_DAYS = 365

ResearchDuplicateKey = tuple[str, str, str]

_PROVIDER_RELIABILITY_RANKS = {
    "sec_form_d": 0,
    "sam_gov": 0,
    "usaspending": 0,
    "sbir": 0,
    "uspto": 0,
    "company_website": 1,
    "github": 2,
    "public_web": 3,
}
_PROVIDER_RELIABILITY_TAGS = {
    "company_website": SourceReliability.OFFICIAL_COMPANY,
    "sec_form_d": SourceReliability.GOVERNMENT_FILING,
    "sam_gov": SourceReliability.PUBLIC_DATABASE,
    "usaspending": SourceReliability.PUBLIC_DATABASE,
    "sbir": SourceReliability.PUBLIC_DATABASE,
    "uspto": SourceReliability.PUBLIC_DATABASE,
    "github": SourceReliability.REPOSITORY_METADATA,
    "meridian": SourceReliability.MANUAL_PORTAL_ENTRY,
    "newsapi": SourceReliability.REPUTABLE_PRESS,
    "crunchbase": SourceReliability.PUBLIC_DATABASE,
    "people_data_labs": SourceReliability.PUBLIC_DATABASE,
    "similarweb": SourceReliability.PUBLIC_DATABASE,
    "sensor_tower": SourceReliability.PUBLIC_DATABASE,
    "pitchbook": SourceReliability.PUBLIC_DATABASE,
    "cb_insights": SourceReliability.PUBLIC_DATABASE,
}


class ResearchQualityMetric(BaseModel):
    label: str
    count: int = Field(ge=0)


class ResearchQualityStatus(BaseModel):
    status: str
    imported_record_count: int = Field(ge=0)
    current_record_count: int = Field(ge=0)
    stale_record_count: int = Field(ge=0)
    unknown_freshness_record_count: int = Field(ge=0)
    stale_only: bool = False
    unknown_reliability_record_count: int = Field(ge=0)
    ambiguous_or_related_match_count: int = Field(ge=0)
    identity_mismatch_count: int = Field(ge=0)
    source_reliability: list[ResearchQualityMetric] = Field(default_factory=list)
    freshness: list[ResearchQualityMetric] = Field(default_factory=list)
    identity_matches: list[ResearchQualityMetric] = Field(default_factory=list)
    skipped_identity_matches: list[ResearchQualityMetric] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    refresh_guidance: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _RankedResearchResult:
    result: ResearchResultInput
    index: int
    duplicate_keys: frozenset[ResearchDuplicateKey]


def ranked_public_research_results(
    results: Sequence[ResearchResultInput],
    *,
    company_name: str,
    ranked_at: datetime,
) -> list[ResearchResultInput]:
    """Collapse duplicates and return public results in deterministic quality order."""
    ranked_results = [
        _RankedResearchResult(
            result=result,
            index=index,
            duplicate_keys=frozenset(
                research_result_duplicate_keys(result, deal_key=company_name)
            ),
        )
        for index, result in enumerate(results)
    ]
    groups = _duplicate_groups(ranked_results)
    winners = [
        min(
            group,
            key=lambda ranked: _public_result_rank(
                ranked.result,
                company_name=company_name,
                ranked_at=ranked_at,
                duplicate=len(group) > 1,
                index=ranked.index,
            ),
        )
        for group in groups
    ]
    group_sizes = {
        ranked.index: len(group)
        for group in groups
        for ranked in group
    }
    return [
        winner.result
        for winner in sorted(
            winners,
            key=lambda ranked: _public_result_rank(
                ranked.result,
                company_name=company_name,
                ranked_at=ranked_at,
                duplicate=group_sizes.get(ranked.index, 1) > 1,
                index=ranked.index,
            ),
        )
    ]


def source_reliability_for_result(result: ResearchResultInput) -> SourceReliability:
    if result.source_reliability is not None:
        return result.source_reliability
    return source_reliability_for_provider(
        result.provider_id,
        source_kind=result.source_kind,
        document_type=result.document_type,
    )


def source_reliability_for_provider(
    provider_id: str | None,
    *,
    source_kind: SourceKind,
    document_type: DocumentType,
) -> SourceReliability:
    if provider_id is not None:
        tag = _PROVIDER_RELIABILITY_TAGS.get(provider_id)
        if tag is not None:
            return tag
    if source_kind == SourceKind.MERIDIAN:
        return SourceReliability.MANUAL_PORTAL_ENTRY
    if source_kind == SourceKind.WEB and document_type == DocumentType.WEB_PAGE:
        return SourceReliability.UNKNOWN
    return SourceReliability.UNKNOWN


def source_freshness_for_retrieved_at(
    retrieved_at: datetime,
    *,
    now: datetime,
) -> SourceFreshness:
    age_days = (_as_utc(now) - _as_utc(retrieved_at)).days
    if age_days > STALE_SOURCE_DAYS:
        return SourceFreshness.STALE
    return SourceFreshness.CURRENT


def research_quality_status(
    store: EvidenceStore,
    *,
    match_details: Sequence[CompanyMatch] = (),
) -> ResearchQualityStatus:
    research_evidence = _external_research_evidence(store)
    imported_count = len(research_evidence)
    current_count = sum(
        1
        for evidence in research_evidence
        if evidence.source_freshness == SourceFreshness.CURRENT
    )
    stale_count = sum(
        1
        for evidence in research_evidence
        if evidence.source_freshness == SourceFreshness.STALE
    )
    unknown_freshness_count = sum(
        1
        for evidence in research_evidence
        if evidence.source_freshness == SourceFreshness.UNKNOWN
    )
    unknown_reliability_count = sum(
        1
        for evidence in research_evidence
        if evidence.source_reliability == SourceReliability.UNKNOWN
    )
    skipped_match_counts = _metric_count_values(
        match.kind.value for match in match_details if not match.import_ready
    )
    ambiguous_or_related_count = sum(
        count
        for kind, count in skipped_match_counts.items()
        if kind != CompanyMatchKind.REJECTED.value
    )
    identity_mismatch_count = skipped_match_counts.get(CompanyMatchKind.REJECTED.value, 0)
    limitations = _research_quality_limitations(
        imported_count=imported_count,
        stale_count=stale_count,
        unknown_freshness_count=unknown_freshness_count,
        unknown_reliability_count=unknown_reliability_count,
        ambiguous_or_related_count=ambiguous_or_related_count,
        identity_mismatch_count=identity_mismatch_count,
    )
    return ResearchQualityStatus(
        status=_research_quality_status_label(
            imported_count=imported_count,
            limitations=limitations,
        ),
        imported_record_count=imported_count,
        current_record_count=current_count,
        stale_record_count=stale_count,
        unknown_freshness_record_count=unknown_freshness_count,
        stale_only=imported_count > 0 and stale_count == imported_count,
        unknown_reliability_record_count=unknown_reliability_count,
        ambiguous_or_related_match_count=ambiguous_or_related_count,
        identity_mismatch_count=identity_mismatch_count,
        source_reliability=_metrics_for_counts(
            _metric_count_values(
                evidence.source_reliability.value for evidence in research_evidence
            )
        ),
        freshness=_metrics_for_counts(
            _metric_count_values(
                evidence.source_freshness.value for evidence in research_evidence
            )
        ),
        identity_matches=_metrics_for_counts(
            _metric_count_values(
                evidence.identity_match_kind or "unknown" for evidence in research_evidence
            )
        ),
        skipped_identity_matches=_metrics_for_counts(skipped_match_counts),
        limitations=limitations,
        refresh_guidance=_research_quality_refresh_guidance(
            imported_count=imported_count,
            stale_count=stale_count,
            unknown_freshness_count=unknown_freshness_count,
            unknown_reliability_count=unknown_reliability_count,
            ambiguous_or_related_count=ambiguous_or_related_count,
            identity_mismatch_count=identity_mismatch_count,
        ),
    )


def research_result_duplicate_keys(
    result: ResearchResultInput,
    *,
    deal_key: str,
) -> set[ResearchDuplicateKey]:
    return _duplicate_keys(
        deal_key=deal_key,
        source_url=result.source_url,
        source_api=result.source_api,
        text=result.text,
    )


def evidence_duplicate_keys(
    evidence: EvidenceRecord,
    *,
    deal_key: str,
) -> set[ResearchDuplicateKey]:
    return _duplicate_keys(
        deal_key=deal_key,
        source_url=evidence.source_url,
        source_api=evidence.source_api,
        text=evidence.text,
    )


def _external_research_evidence(store: EvidenceStore) -> list[EvidenceRecord]:
    return [evidence for evidence in store.evidence if evidence.provider_id is not None]


def _research_quality_status_label(
    *,
    imported_count: int,
    limitations: Sequence[str],
) -> str:
    if imported_count == 0:
        return "missing"
    if limitations:
        return "limited"
    return "usable"


def _research_quality_limitations(
    *,
    imported_count: int,
    stale_count: int,
    unknown_freshness_count: int,
    unknown_reliability_count: int,
    ambiguous_or_related_count: int,
    identity_mismatch_count: int,
) -> list[str]:
    limitations: list[str] = []
    if imported_count == 0:
        limitations.append("No external research evidence was imported before scoring.")
    elif stale_count == imported_count:
        limitations.append(
            "All imported external research records were stale, so they should not be "
            "treated as strong support until refreshed."
        )
    elif stale_count:
        limitations.append(
            "Some imported external research records were stale, so those records are "
            "limited support until refreshed."
        )
    if unknown_freshness_count:
        limitations.append(
            "Some imported external research records did not have known freshness."
        )
    if unknown_reliability_count:
        limitations.append(
            "Some imported external research records had unknown source reliability."
        )
    if ambiguous_or_related_count:
        limitations.append(
            "Some external research results were skipped because their company identity "
            "was related, product-like, founder-related, or ambiguous."
        )
    if identity_mismatch_count:
        limitations.append(
            "Some external research results were skipped because their company identity "
            "did not match the requested company."
        )
    return limitations


def _research_quality_refresh_guidance(
    *,
    imported_count: int,
    stale_count: int,
    unknown_freshness_count: int,
    unknown_reliability_count: int,
    ambiguous_or_related_count: int,
    identity_mismatch_count: int,
) -> list[str]:
    guidance: list[str] = []
    if imported_count == 0:
        guidance.append(
            "Prepare exact-match external research results before relying on public-source gaps."
        )
    if stale_count:
        guidance.append(
            "Refresh stale external research from the original public source before treating it "
            "as strong support."
        )
    if unknown_freshness_count:
        guidance.append(
            "Add a real retrieved_at timestamp for undated external research before relying on it."
        )
    if unknown_reliability_count:
        guidance.append(
            "Verify unknown-reliability research against an official, government, reputable "
            "press, repository, or reviewed portal source."
        )
    if ambiguous_or_related_count or identity_mismatch_count:
        guidance.append(
            "Resolve skipped identity matches manually before treating missing public evidence "
            "as meaningful."
        )
    return guidance


def _metric_count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _metrics_for_counts(counts: dict[str, int]) -> list[ResearchQualityMetric]:
    return [
        ResearchQualityMetric(label=label, count=count)
        for label, count in sorted(counts.items())
        if count > 0
    ]


def normalize_research_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def canonical_source_references(
    *,
    source_url: str | None,
    source_api: str | None,
) -> set[str]:
    references: set[str] = set()
    for value in (source_url, source_api):
        if value is None:
            continue
        canonical = _canonical_source_reference(value)
        if canonical:
            references.add(canonical)
    return references


def _duplicate_keys(
    *,
    deal_key: str,
    source_url: str | None,
    source_api: str | None,
    text: str,
) -> set[ResearchDuplicateKey]:
    normalized_deal = _normalized_deal_key(deal_key)
    normalized_text = normalize_research_text(text)
    if not normalized_deal or not normalized_text:
        return set()
    return {
        (normalized_deal, source_reference, normalized_text)
        for source_reference in canonical_source_references(
            source_url=source_url,
            source_api=source_api,
        )
    }


def _duplicate_groups(
    ranked_results: list[_RankedResearchResult],
) -> list[list[_RankedResearchResult]]:
    groups: list[list[_RankedResearchResult]] = []
    group_keys: list[set[ResearchDuplicateKey]] = []
    for ranked in ranked_results:
        if not ranked.duplicate_keys:
            groups.append([ranked])
            group_keys.append(set())
            continue
        matching_indexes = [
            index
            for index, keys in enumerate(group_keys)
            if keys and not keys.isdisjoint(ranked.duplicate_keys)
        ]
        if not matching_indexes:
            groups.append([ranked])
            group_keys.append(set(ranked.duplicate_keys))
            continue
        primary_index = matching_indexes[0]
        groups[primary_index].append(ranked)
        group_keys[primary_index].update(ranked.duplicate_keys)
        for duplicate_index in reversed(matching_indexes[1:]):
            groups[primary_index].extend(groups.pop(duplicate_index))
            group_keys[primary_index].update(group_keys.pop(duplicate_index))
    return groups


def _public_result_rank(
    result: ResearchResultInput,
    *,
    company_name: str,
    ranked_at: datetime,
    duplicate: bool,
    index: int,
) -> tuple[int, int, int, int, int, int, int]:
    match = classify_company_match(company_name, result.company_name or "")
    retrieved_at = _as_utc(result.retrieved_at)
    return (
        _company_match_rank(match.kind),
        int(duplicate),
        _provider_reliability_rank(result.provider_id),
        _freshness_rank(retrieved_at, ranked_at=ranked_at),
        _source_type_rank(result.source_kind, result.document_type),
        -int(retrieved_at.timestamp()),
        index,
    )


def _company_match_rank(kind: CompanyMatchKind) -> int:
    return {
        CompanyMatchKind.EXACT: 0,
        CompanyMatchKind.LEGAL_ENTITY: 1,
        CompanyMatchKind.LIKELY: 2,
        CompanyMatchKind.PRODUCT_NAME: 3,
        CompanyMatchKind.FOUNDER_RELATED: 4,
        CompanyMatchKind.RELATED: 5,
        CompanyMatchKind.AMBIGUOUS: 6,
        CompanyMatchKind.REJECTED: 7,
    }[kind]


def _provider_reliability_rank(provider_id: str) -> int:
    return _PROVIDER_RELIABILITY_RANKS.get(provider_id, 4)


def _freshness_rank(retrieved_at: datetime, *, ranked_at: datetime) -> int:
    return int((_as_utc(ranked_at) - retrieved_at).days > STALE_SOURCE_DAYS)


def _source_type_rank(source_kind: SourceKind, document_type: DocumentType) -> int:
    if source_kind == SourceKind.WEB and document_type == DocumentType.WEB_PAGE:
        return 0
    if source_kind == SourceKind.MERIDIAN:
        return 1
    if source_kind == SourceKind.MANUAL_NOTE:
        return 2
    return 3


def _canonical_source_reference(value: str) -> str | None:
    stripped = re.sub(r"\s+", " ", value).strip()
    if not stripped:
        return None
    parsed = urlparse(stripped)
    if parsed.scheme and parsed.netloc:
        try:
            port = parsed.port
        except ValueError:
            port = None
        host = (parsed.hostname or "").casefold()
        netloc = f"{host}:{port}" if port is not None else host
        path = re.sub(r"/+", "/", parsed.path).rstrip("/")
        if not path:
            path = "/"
        reference = f"{parsed.scheme.casefold()}://{netloc}{path}"
        if parsed.query:
            reference = f"{reference}?{parsed.query}"
        return reference
    return stripped.casefold()


def _normalized_deal_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
