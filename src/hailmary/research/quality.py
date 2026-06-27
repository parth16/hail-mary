from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from hailmary.schemas.documents import DocumentType, SourceKind
from hailmary.schemas.evidence import EvidenceRecord

from .matching import CompanyMatchKind, classify_company_match
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
        CompanyMatchKind.LIKELY: 1,
        CompanyMatchKind.RELATED: 2,
        CompanyMatchKind.REJECTED: 3,
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
