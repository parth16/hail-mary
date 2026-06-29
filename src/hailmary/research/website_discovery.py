from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from hailmary.schemas.documents import DocumentType, SourceKind
from hailmary.schemas.evidence import EvidenceRecord, EvidenceStore, SourceReliability

from .source_urls import validate_provider_source_url

URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
TRAILING_PUNCTUATION = ".,;:!?)\"]}'"
AUTHENTICATED_PORTAL_HOST_PARTS = (
    "angellist",
    "meridian",
    "dealroom",
    "portal",
    "app.",
    "investor",
    "private",
)
SEARCH_HOSTS = (
    "google.",
    "bing.com",
    "brave.com",
    "search.yahoo.",
    "duckduckgo.com",
)
HIGH_SIGNAL_TERMS = (
    "official website",
    "company website",
    "website:",
    "site:",
    "homepage",
    "home page",
)


@dataclass(frozen=True)
class OfficialWebsiteDiscovery:
    selected_url: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _WebsiteCandidate:
    url: str
    score: int

    @property
    def host(self) -> str:
        return (urlparse(self.url).hostname or "").casefold()


def discover_official_website_url(store: EvidenceStore) -> OfficialWebsiteDiscovery:
    """Find one safe official website URL from already ingested evidence."""

    candidates: dict[str, _WebsiteCandidate] = {}
    rejected_count = 0
    for evidence in store.evidence:
        for raw_url, base_score in _evidence_url_candidates(evidence):
            cleaned_url = _clean_candidate_url(raw_url)
            if cleaned_url is None:
                rejected_count += 1
                continue
            safety_reason = _unsafe_official_website_reason(cleaned_url)
            if safety_reason is not None:
                rejected_count += 1
                continue
            score = base_score + _evidence_score(evidence) + _url_score(cleaned_url, store)
            existing = candidates.get(cleaned_url)
            if existing is None or score > existing.score:
                candidates[cleaned_url] = _WebsiteCandidate(cleaned_url, score)

    warnings: list[str] = []
    if rejected_count:
        warnings.append(
            "Ignored one or more unsafe website URLs found in stored evidence. "
            "Hail Mary did not print those URLs because they may contain private "
            "portal paths, credentials, tokens, or local-network addresses."
        )
    if not candidates:
        return OfficialWebsiteDiscovery(selected_url=None, warnings=tuple(warnings))

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.score, _url_path_depth(candidate.url), candidate.url),
    )
    selected = ranked[0]
    plausible = [
        candidate
        for candidate in ranked
        if candidate.url != selected.url and candidate.score >= selected.score - 20
    ]
    if plausible:
        safe_choices = ", ".join(candidate.url for candidate in plausible[:4])
        warnings.append(
            "Multiple safe official website URLs were found in stored evidence, so "
            f"Hail Mary chose {selected.url}. Other plausible URLs were: {safe_choices}."
        )
    return OfficialWebsiteDiscovery(selected_url=selected.url, warnings=tuple(warnings))


def _evidence_url_candidates(evidence: EvidenceRecord) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    if evidence.source_url and _source_url_is_official_candidate(evidence):
        candidates.append((evidence.source_url, 80))
    for match in URL_PATTERN.finditer(evidence.text):
        surrounding = evidence.text[max(0, match.start() - 80) : match.end() + 80]
        score = 25
        if any(term in surrounding.casefold() for term in HIGH_SIGNAL_TERMS):
            score += 45
        candidates.append((match.group(0), score))
    return candidates


def _source_url_is_official_candidate(evidence: EvidenceRecord) -> bool:
    return (
        evidence.provider_id == "company_website"
        or evidence.source_reliability == SourceReliability.OFFICIAL_COMPANY
    )


def _evidence_score(evidence: EvidenceRecord) -> int:
    score = 0
    if evidence.source_kind == SourceKind.MERIDIAN:
        score += 40
    if evidence.document_type == DocumentType.PLATFORM_DEAL_PAGE:
        score += 35
    if evidence.document_type == DocumentType.MEMO:
        score += 20
    if evidence.provider_id == "company_website":
        score += 40
    if evidence.source_reliability == SourceReliability.OFFICIAL_COMPANY:
        score += 35
    return score


def _url_score(url: str, store: EvidenceStore) -> int:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.strip("/")
    normalized_company = _slugish(store.company_name)
    score = 0
    if normalized_company and normalized_company in _slugish(host):
        score += 35
    if not path:
        score += 25
    elif _url_path_depth(url) == 1:
        score += 10
    if host.startswith("www."):
        score += 5
    return score


def _clean_candidate_url(raw_url: str) -> str | None:
    cleaned = raw_url.strip().rstrip(TRAILING_PUNCTUATION)
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc or parsed.hostname is None:
        return None
    return cleaned


def _unsafe_official_website_reason(url: str) -> str | None:
    try:
        validate_provider_source_url("company_website", url, field_name="website URL")
    except ValueError as exc:
        return str(exc)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.query or parsed.fragment:
        return "website URL cannot include query strings or fragments"
    if _is_local_or_private_host(host):
        return "website URL cannot use localhost or a private network address"
    if _looks_like_authenticated_portal_host(host):
        return "website URL looks like an authenticated portal"
    if _looks_like_search_page(host, parsed.path):
        return "website URL looks like a search-results page"
    return None


def _looks_like_authenticated_portal_host(host: str) -> bool:
    return any(part in host for part in AUTHENTICATED_PORTAL_HOST_PARTS)


def _looks_like_search_page(host: str, path: str) -> bool:
    normalized_path = path.casefold()
    if any(search_host in host for search_host in SEARCH_HOSTS):
        return True
    return normalized_path.startswith(("/search", "/results"))


def _is_local_or_private_host(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global


def _url_path_depth(url: str) -> int:
    path = urlparse(url).path.strip("/")
    if not path:
        return 0
    return len([part for part in path.split("/") if part])


def _slugish(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
