from __future__ import annotations

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self
from urllib.parse import quote, urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import DocumentType

from .matching import (
    CompanyMatch,
    CompanyMatchKind,
    best_company_match,
    classify_company_match,
    normalize_company_name,
    normalize_company_slug,
)
from .providers import builtin_research_providers
from .quality import ranked_public_research_results, source_reliability_for_provider
from .schemas import ResearchProvider, ResearchResultInput, ResearchResultsFile
from .source_urls import (
    source_reference_looks_like_url,
    validate_http_url,
    validate_provider_source_url,
)
from .web import (
    WebResearchFetchError,
    _ensure_resolved_public_host,
    _GuardedHTTPHandler,
    _GuardedHTTPSHandler,
)


class ResearchCollectionError(RuntimeError):
    """Public research results could not be prepared safely."""


class UsaspendingApiError(RuntimeError):
    """USAspending public API results could not be collected safely."""


class SecFormDApiError(RuntimeError):
    """SEC Form D public results could not be collected safely."""


USASPENDING_AWARDS_ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
SEC_FORM_D_ATOM_ENDPOINT = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_FORM_D_MAX_PAGES = 20
SEC_FORM_D_REQUEST_INTERVAL_SECONDS = 0.11
SEC_FORM_D_USER_AGENT_ENV_VAR = "HAILMARY_SEC_USER_AGENT"
PUBLIC_API_MAX_BYTES = 2_000_000
USASPENDING_AWARD_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Recipient UEI",
    "Start Date",
    "End Date",
    "Award Amount",
    "Award Type",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Funding Agency",
    "Funding Sub Agency",
    "Description",
    "generated_internal_id",
]
USASPENDING_AWARD_TYPE_CODES = [
    "-1",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "A",
    "B",
    "C",
    "D",
    "IDV_A",
    "IDV_B",
    "IDV_B_A",
    "IDV_B_B",
    "IDV_B_C",
    "IDV_C",
    "IDV_D",
    "IDV_E",
]
USASPENDING_MAX_PAGES = 20
SBIR_AWARDS_ENDPOINT = "https://api.www.sbir.gov/public/api/awards"
SBIR_MAX_PAGES = 20


class SbirApiError(RuntimeError):
    """SBIR/STTR public API results could not be collected safely."""


class GitHubApiError(RuntimeError):
    """GitHub public API results could not be collected safely."""


GITHUB_REPOSITORY_SEARCH_ENDPOINT = "https://api.github.com/search/repositories"
GITHUB_MAX_PAGES = 5
GITHUB_SEARCH_REQUEST_INTERVAL_SECONDS = 6.1
MAX_IDENTITY_WARNING_DETAILS = 5


@dataclass(frozen=True)
class PublicSourceAdapterConfig:
    provider_id: str
    description: str

    def confidence(self, provider: ResearchProvider) -> str:
        return (
            "high: exact company name match from a local "
            f"{provider.name} source file"
        )


PUBLIC_SOURCE_ADAPTERS: dict[str, PublicSourceAdapterConfig] = {
    "sec_form_d": PublicSourceAdapterConfig("sec_form_d", "SEC Form D results"),
    "sam_gov": PublicSourceAdapterConfig("sam_gov", "SAM.gov results"),
    "usaspending": PublicSourceAdapterConfig("usaspending", "USAspending results"),
    "sbir": PublicSourceAdapterConfig("sbir", "SBIR/STTR results"),
    "uspto": PublicSourceAdapterConfig("uspto", "USPTO results"),
    "github": PublicSourceAdapterConfig("github", "GitHub results"),
}


class PublicSourceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    title: str
    text: str
    source_url: str | None = None
    source_api: str | None = None
    retrieved_at: datetime
    filed_at: datetime | None = None
    accession_number: str | None = None

    @field_validator(
        "company_name",
        "title",
        "text",
        "source_url",
        "source_api",
        "accession_number",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "company_name",
        "title",
        "text",
        "source_url",
        "source_api",
        "accession_number",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("company_name", "title", "text")
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_source_reference(self) -> Self:
        if self.source_url is None and self.source_api is None:
            raise ValueError("Each public source result needs a source_url or source_api.")
        if self.source_url is not None:
            validate_http_url(self.source_url, field_name="source_url")
        if self.source_api is not None and source_reference_looks_like_url(self.source_api):
            validate_http_url(self.source_api, field_name="source_api")
        return self


class SecFormDSearchResult(PublicSourceSearchResult):
    @model_validator(mode="after")
    def validate_sec_source_url(self) -> Self:
        if self.source_url is not None:
            validate_provider_source_url("sec_form_d", self.source_url)
        if self.source_api is not None and source_reference_looks_like_url(self.source_api):
            validate_provider_source_url(
                "sec_form_d",
                self.source_api,
                field_name="source_api",
            )
        return self


class PublicSourceSearchResultsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[PublicSourceSearchResult]


class UsaspendingAwardRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    award_id: str | None = Field(default=None, alias="Award ID")
    recipient_name: str = Field(alias="Recipient Name")
    generated_internal_id: str | None = None
    award_amount: float | None = Field(default=None, alias="Award Amount")
    award_type: str | None = Field(default=None, alias="Award Type")
    awarding_agency: str | None = Field(default=None, alias="Awarding Agency")
    awarding_sub_agency: str | None = Field(default=None, alias="Awarding Sub Agency")
    funding_agency: str | None = Field(default=None, alias="Funding Agency")
    funding_sub_agency: str | None = Field(default=None, alias="Funding Sub Agency")
    start_date: str | None = Field(default=None, alias="Start Date")
    end_date: str | None = Field(default=None, alias="End Date")
    description: str | None = Field(default=None, alias="Description")
    recipient_uei: str | None = Field(default=None, alias="Recipient UEI")

    @field_validator(
        "award_id",
        "recipient_name",
        "generated_internal_id",
        "award_type",
        "awarding_agency",
        "awarding_sub_agency",
        "funding_agency",
        "funding_sub_agency",
        "start_date",
        "end_date",
        "description",
        "recipient_uei",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "award_id",
        "recipient_name",
        "generated_internal_id",
        "award_type",
        "awarding_agency",
        "awarding_sub_agency",
        "funding_agency",
        "funding_sub_agency",
        "start_date",
        "end_date",
        "description",
        "recipient_uei",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("recipient_name")
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value


class UsaspendingPageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    page: int | None = None
    has_next: bool = Field(alias="hasNext")


class UsaspendingAwardsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[dict[str, Any]]
    page_metadata: UsaspendingPageMetadata

    @property
    def has_next_page(self) -> bool:
        return self.page_metadata.has_next


class SbirAwardRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    firm: str
    award_title: str | None = None
    agency: str | None = None
    branch: str | None = None
    phase: str | None = None
    program: str | None = None
    agency_tracking_number: str | None = None
    contract: str | None = None
    proposal_award_date: str | None = None
    contract_end_date: str | None = None
    solicitation_number: str | None = None
    solicitation_year: str | None = None
    topic_code: str | None = None
    award_year: str | None = None
    award_amount: float | None = None
    uei: str | None = None
    research_area_keywords: str | None = None
    abstract: str | None = None
    award_link: str | None = None

    @field_validator(
        "firm",
        "award_title",
        "agency",
        "branch",
        "phase",
        "program",
        "agency_tracking_number",
        "contract",
        "proposal_award_date",
        "contract_end_date",
        "solicitation_number",
        "solicitation_year",
        "topic_code",
        "award_year",
        "uei",
        "research_area_keywords",
        "abstract",
        "award_link",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "firm",
        "award_title",
        "agency",
        "branch",
        "phase",
        "program",
        "agency_tracking_number",
        "contract",
        "proposal_award_date",
        "contract_end_date",
        "solicitation_number",
        "solicitation_year",
        "topic_code",
        "award_year",
        "uei",
        "research_area_keywords",
        "abstract",
        "award_link",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("award_amount", mode="before")
    @classmethod
    def clean_award_amount(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            if not cleaned:
                return None
            return cleaned
        return value

    @field_validator("firm")
    @classmethod
    def require_nonblank_firm(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def require_award_title(self) -> Self:
        if self.award_title is None:
            raise ValueError("award_title must not be blank")
        if self.award_link is not None:
            validate_provider_source_url("sbir", self.award_link)
        return self


class SbirAwardsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]]

    @property
    def result_count(self) -> int:
        return len(self.results)


class SecFormDFilingRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    issuer_name: str
    filing_type: str
    accession_number: str
    source_url: str
    source_api: str
    filing_date: str | None = None
    form_name: str | None = None
    total_offering_amount: str | None = None
    total_amount_sold: str | None = None
    total_remaining: str | None = None
    minimum_investment_accepted: str | None = None
    total_investors: str | None = None
    industry_group: str | None = None
    revenue_range: str | None = None
    federal_exemptions: list[str] = Field(default_factory=list)

    @field_validator(
        "issuer_name",
        "filing_type",
        "accession_number",
        "source_url",
        "source_api",
        "filing_date",
        "form_name",
        "total_offering_amount",
        "total_amount_sold",
        "total_remaining",
        "minimum_investment_accepted",
        "total_investors",
        "industry_group",
        "revenue_range",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "issuer_name",
        "filing_type",
        "accession_number",
        "source_url",
        "source_api",
        "filing_date",
        "form_name",
        "total_offering_amount",
        "total_amount_sold",
        "total_remaining",
        "minimum_investment_accepted",
        "total_investors",
        "industry_group",
        "revenue_range",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator(
        "issuer_name",
        "filing_type",
        "accession_number",
        "source_url",
        "source_api",
    )
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("filing_type")
    @classmethod
    def require_form_d_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"D", "D/A"}:
            raise ValueError("must be SEC filing type D or D/A")
        return normalized

    @field_validator("federal_exemptions", mode="before")
    @classmethod
    def normalize_federal_exemptions(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("federal_exemptions")
    @classmethod
    def strip_federal_exemptions(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_sec_urls(self) -> Self:
        validate_provider_source_url("sec_form_d", self.source_url)
        validate_provider_source_url(
            "sec_form_d",
            self.source_api,
            field_name="source_api",
        )
        if urlparse(self.source_url).scheme != "https":
            raise ValueError("source_url must stay on HTTPS")
        if urlparse(self.source_api).scheme != "https":
            raise ValueError("source_api must stay on HTTPS")
        return self


class SecFormDFilingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]]
    has_next: bool = False
    warnings: list[str] = Field(default_factory=list)


class GitHubRepositoryRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    full_name: str
    owner_login: str
    html_url: str
    url: str
    description: str | None = None
    language: str | None = None
    stargazers_count: int | None = None
    forks_count: int | None = None
    open_issues_count: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    license_name: str | None = None
    private: bool = False
    fork: bool = False
    archived: bool = False
    disabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def extract_nested_github_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        updated = dict(data)
        owner = updated.get("owner")
        if isinstance(owner, dict):
            owner_login = owner.get("login")
            if isinstance(owner_login, str):
                updated.setdefault("owner_login", owner_login)
        license_data = updated.get("license")
        if isinstance(license_data, dict):
            license_name = license_data.get("name")
            if isinstance(license_name, str):
                updated.setdefault("license_name", license_name)
        return updated

    @field_validator(
        "name",
        "full_name",
        "owner_login",
        "html_url",
        "url",
        "description",
        "language",
        "license_name",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "name",
        "full_name",
        "owner_login",
        "html_url",
        "url",
        "description",
        "language",
        "license_name",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("name", "full_name", "owner_login", "html_url", "url")
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_github_urls(self) -> Self:
        validate_provider_source_url("github", self.html_url)
        validate_provider_source_url("github", self.url, field_name="source_api")
        if urlparse(self.html_url).scheme != "https":
            raise ValueError("source_url must stay on HTTPS")
        if urlparse(self.url).scheme != "https":
            raise ValueError("source_api must stay on HTTPS")
        if self.private:
            raise ValueError("GitHub repository results must be public.")
        return self


class GitHubRepositorySearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    results: list[dict[str, Any]] = Field(alias="items")
    incomplete_results: bool
    total_count: int
    has_next: bool = False

    @property
    def result_count(self) -> int:
        return len(self.results)


SecFormDSearchResultsFile = PublicSourceSearchResultsFile


class ResearchCollectionDealSummary(BaseModel):
    company_name: str
    result_count: int = 0


class ResearchCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    provider_ids: list[str]
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    provider_result_counts: dict[str, int] = Field(default_factory=dict)
    provider_company_result_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    skipped_non_exact_company_names: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)

    @property
    def skipped_non_exact_count(self) -> int:
        return len(self.skipped_non_exact_company_names)


class UsaspendingCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    dry_run: bool = False
    endpoint: str = USASPENDING_AWARDS_ENDPOINT
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)


class SbirCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    dry_run: bool = False
    endpoint: str = SBIR_AWARDS_ENDPOINT
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)


class SecFormDCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    dry_run: bool = False
    endpoint: str = SEC_FORM_D_ATOM_ENDPOINT
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)


class GitHubRepositoryCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    dry_run: bool = False
    endpoint: str = GITHUB_REPOSITORY_SEARCH_ENDPOINT
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)


@dataclass(frozen=True)
class ResearchCollectionDeal:
    company_name: str


class PublicSourceSearchClient(Protocol):
    def search(self, company_name: str) -> Iterable[PublicSourceSearchResult]:
        """Return locally available public-source results for one company."""


class UsaspendingAwardsClient(Protocol):
    def search_awards(
        self,
        company_name: str,
        *,
        limit: int,
        page: int,
        timeout_seconds: float,
    ) -> UsaspendingAwardsResponse:
        """Return one USAspending public API award-results page for one company."""


class SbirAwardsClient(Protocol):
    def search_awards(
        self,
        company_name: str,
        *,
        rows: int,
        start: int,
        timeout_seconds: float,
    ) -> SbirAwardsResponse:
        """Return one SBIR/STTR public API award-results page for one company."""


class SecFormDFilingsClient(Protocol):
    def search_filings(
        self,
        company_name: str,
        *,
        count: int,
        start: int,
        timeout_seconds: float,
    ) -> SecFormDFilingsResponse:
        """Return one SEC EDGAR company-search filing page for one company."""


class GitHubRepositorySearchClient(Protocol):
    def search_repositories(
        self,
        company_name: str,
        *,
        per_page: int,
        page: int,
        timeout_seconds: float,
    ) -> GitHubRepositorySearchResponse:
        """Return one GitHub public repository search page for one company."""


@dataclass(frozen=True)
class LocalPublicSourceSearchClient:
    results: tuple[PublicSourceSearchResult, ...]

    def search(self, company_name: str) -> Iterable[PublicSourceSearchResult]:
        for result in self.results:
            if _exact_company_name_match(company_name, result.company_name):
                yield result


@dataclass(frozen=True)
class PublicSourceFileAdapter:
    client: PublicSourceSearchClient
    provider_id: str = "sec_form_d"

    def collect(
        self,
        deal: ResearchCollectionDeal,
        *,
        collected_at: datetime,
    ) -> list[ResearchResultInput]:
        provider = _provider_by_id(self.provider_id)
        adapter_config = _public_source_adapter_config(self.provider_id)
        results: list[ResearchResultInput] = []
        for search_result in self.client.search(deal.company_name):
            match = classify_company_match(deal.company_name, search_result.company_name)
            if not match.import_ready:
                continue
            if search_result.source_url is not None:
                validate_provider_source_url(provider.id, search_result.source_url)
            if (
                search_result.source_api is not None
                and source_reference_looks_like_url(search_result.source_api)
            ):
                validate_provider_source_url(
                    provider.id,
                    search_result.source_api,
                    field_name="source_api",
                )
            try:
                result = ResearchResultInput(
                    company_name=deal.company_name,
                    provider_id=provider.id,
                    provider_name=provider.name,
                    title=search_result.title,
                    text=search_result.text,
                    retrieved_at=_as_utc(search_result.retrieved_at),
                    source_url=search_result.source_url,
                    source_api=search_result.source_api,
                    confidence=adapter_config.confidence(provider),
                    licensing_notes=provider.licensing_notes,
                    source_kind=provider.source_kind,
                    source_reliability=source_reliability_for_provider(
                        provider.id,
                        source_kind=provider.source_kind,
                        document_type=DocumentType.WEB_PAGE,
                    ),
                    identity_match_kind=match.kind,
                    identity_match_reason=match.reason,
                )
            except ValidationError as exc:
                detail = _first_validation_detail(exc)
                raise ResearchCollectionError(
                    f"{provider.name} result for {deal.company_name} is incomplete: "
                    f"{detail}"
                ) from exc
            results.append(result)
        return results


SecFormDSearchClient = PublicSourceSearchClient
LocalSecFormDSearchClient = LocalPublicSourceSearchClient
SecFormDPublicAdapter = PublicSourceFileAdapter


@dataclass(frozen=True)
class UrlLibUsaspendingAwardsClient:
    user_agent: str = "HailMary/0.1 USAspending public API research"

    def search_awards(
        self,
        company_name: str,
        *,
        limit: int,
        page: int,
        timeout_seconds: float,
    ) -> UsaspendingAwardsResponse:
        _validate_usaspending_api_url(USASPENDING_AWARDS_ENDPOINT)
        _ensure_usaspending_resolved_public_endpoint(USASPENDING_AWARDS_ENDPOINT)
        payload = _usaspending_awards_payload(company_name, limit=limit, page=page)
        request = urllib.request.Request(
            USASPENDING_AWARDS_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = _build_usaspending_api_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_usaspending_api_url(final_url)
                raw_content = response.read(2_000_001)
                if len(raw_content) > 2_000_000:
                    raise UsaspendingApiError(
                        "The USAspending response was larger than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise UsaspendingApiError(
                f"USAspending returned HTTP {exc.code}."
            ) from exc
        except (OSError, TimeoutError, urllib.error.URLError, WebResearchFetchError) as exc:
            raise UsaspendingApiError(f"Could not reach USAspending: {exc}") from exc
        if status_code != 200:
            raise UsaspendingApiError(f"USAspending returned HTTP {status_code}.")
        try:
            decoded_text = raw_content.decode(charset, errors="replace")
        except LookupError:
            decoded_text = raw_content.decode("utf-8", errors="replace")
        try:
            response_payload = json.loads(decoded_text)
        except json.JSONDecodeError as exc:
            raise UsaspendingApiError(
                "USAspending returned a response that was not valid JSON."
            ) from exc
        try:
            parsed_response = UsaspendingAwardsResponse.model_validate(response_payload)
        except ValidationError as exc:
            detail = _first_validation_detail(exc)
            raise UsaspendingApiError(
                f"USAspending returned an unexpected response: {detail}"
            ) from exc
        return parsed_response


class _UsaspendingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_usaspending_api_url(newurl)
        _ensure_usaspending_resolved_public_endpoint(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not isinstance(redirected, urllib.request.Request):
            raise UsaspendingApiError("USAspending returned an unsupported redirect.")
        return redirected


def _build_usaspending_api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler("usaspending"),
        _GuardedHTTPSHandler("usaspending"),
        _UsaspendingRedirectHandler(),
    )


@dataclass(frozen=True)
class UrlLibSbirAwardsClient:
    user_agent: str = "HailMary/0.1 SBIR public API research"

    def search_awards(
        self,
        company_name: str,
        *,
        rows: int,
        start: int,
        timeout_seconds: float,
    ) -> SbirAwardsResponse:
        request_url = _sbir_awards_api_url(company_name, rows=rows, start=start)
        _validate_sbir_api_url(request_url)
        _ensure_sbir_resolved_public_endpoint(request_url)
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        opener = _build_sbir_api_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_sbir_api_url(final_url)
                _ensure_sbir_resolved_public_endpoint(final_url)
                raw_content = response.read(2_000_001)
                if len(raw_content) > 2_000_000:
                    raise SbirApiError(
                        "The SBIR/STTR response was larger than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise SbirApiError(f"SBIR/STTR returned HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError, WebResearchFetchError) as exc:
            raise SbirApiError(f"Could not reach SBIR/STTR: {exc}") from exc
        if status_code != 200:
            raise SbirApiError(f"SBIR/STTR returned HTTP {status_code}.")
        try:
            decoded_text = raw_content.decode(charset, errors="replace")
        except LookupError:
            decoded_text = raw_content.decode("utf-8", errors="replace")
        try:
            response_payload = json.loads(decoded_text)
        except json.JSONDecodeError as exc:
            raise SbirApiError(
                "SBIR/STTR returned a response that was not valid JSON."
            ) from exc
        try:
            return _validate_sbir_awards_response(response_payload)
        except ValidationError as exc:
            detail = _first_validation_detail(exc)
            raise SbirApiError(
                f"SBIR/STTR returned an unexpected response: {detail}"
            ) from exc


class _SbirRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_sbir_api_url(newurl)
        _ensure_sbir_resolved_public_endpoint(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not isinstance(redirected, urllib.request.Request):
            raise SbirApiError("SBIR/STTR returned an unsupported redirect.")
        return redirected


def _build_sbir_api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler("sbir"),
        _GuardedHTTPSHandler("sbir"),
        _SbirRedirectHandler(),
    )


@dataclass(frozen=True)
class UrlLibSecFormDFilingsClient:
    user_agent: str | None = None
    request_interval_seconds: float = SEC_FORM_D_REQUEST_INTERVAL_SECONDS

    def search_filings(
        self,
        company_name: str,
        *,
        count: int,
        start: int,
        timeout_seconds: float,
    ) -> SecFormDFilingsResponse:
        request_url = _sec_form_d_atom_api_url(
            company_name,
            count=count,
            start=start,
        )
        _validate_sec_form_d_atom_url(request_url)
        _ensure_sec_form_d_resolved_public_endpoint(request_url)
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": self._user_agent(),
                "Accept": "application/atom+xml, application/xml;q=0.9, text/xml;q=0.8",
            },
        )
        opener = _build_sec_form_d_api_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_sec_form_d_atom_url(final_url)
                _ensure_sec_form_d_resolved_public_endpoint(final_url)
                raw_content = response.read(PUBLIC_API_MAX_BYTES + 1)
                if len(raw_content) > PUBLIC_API_MAX_BYTES:
                    raise SecFormDApiError(
                        "The SEC EDGAR response was larger than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise SecFormDApiError(f"SEC EDGAR returned HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError, WebResearchFetchError) as exc:
            raise SecFormDApiError(f"Could not reach SEC EDGAR: {exc}") from exc
        if status_code != 200:
            raise SecFormDApiError(f"SEC EDGAR returned HTTP {status_code}.")
        decoded_text = _decode_public_api_text(
            raw_content,
            charset=charset,
            source_name="SEC EDGAR",
        )
        return _parse_sec_form_d_atom_response(
            decoded_text,
            requested_company_name=company_name,
            source_api=request_url,
            fetch_submission=lambda source_url: self._fetch_submission_details(
                source_url,
                timeout_seconds=timeout_seconds,
            ),
        )

    def _fetch_submission_details(
        self,
        source_url: str,
        *,
        timeout_seconds: float,
    ) -> dict[str, str | list[str] | None]:
        _validate_sec_form_d_archive_url(source_url, field_name="SEC Form D source URL")
        _ensure_sec_form_d_resolved_public_endpoint(source_url)
        if self.request_interval_seconds > 0:
            time.sleep(self.request_interval_seconds)
        request = urllib.request.Request(
            source_url,
            headers={
                "User-Agent": self._user_agent(),
                "Accept": "text/plain, application/xml;q=0.9, text/xml;q=0.8",
            },
        )
        opener = _build_sec_form_d_api_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_sec_form_d_archive_url(
                    final_url,
                    field_name="SEC Form D source URL",
                )
                _ensure_sec_form_d_resolved_public_endpoint(final_url)
                raw_content = response.read(PUBLIC_API_MAX_BYTES + 1)
                if len(raw_content) > PUBLIC_API_MAX_BYTES:
                    raise SecFormDApiError(
                        "The SEC Form D filing was larger than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise SecFormDApiError(f"SEC EDGAR returned HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError, WebResearchFetchError) as exc:
            raise SecFormDApiError(f"Could not reach SEC EDGAR: {exc}") from exc
        if status_code != 200:
            raise SecFormDApiError(f"SEC EDGAR returned HTTP {status_code}.")
        decoded_text = _decode_public_api_text(
            raw_content,
            charset=charset,
            source_name="SEC Form D filing",
        )
        return _sec_form_d_details_from_submission(decoded_text, source_url=final_url)

    def _user_agent(self) -> str:
        if self.user_agent is None:
            return _sec_form_d_user_agent_from_env()
        return _validate_sec_form_d_user_agent(self.user_agent)


class _SecFormDRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        request_url = req.full_url
        if _sec_form_d_url_is_atom_endpoint(request_url):
            _validate_sec_form_d_atom_url(newurl)
        else:
            _validate_sec_form_d_archive_url(newurl, field_name="SEC redirect URL")
        _ensure_sec_form_d_resolved_public_endpoint(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not isinstance(redirected, urllib.request.Request):
            raise SecFormDApiError("SEC EDGAR returned an unsupported redirect.")
        return redirected


def _build_sec_form_d_api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler("sec_form_d"),
        _GuardedHTTPSHandler("sec_form_d"),
        _SecFormDRedirectHandler(),
    )


@dataclass(frozen=True)
class UrlLibGitHubRepositorySearchClient:
    user_agent: str = "HailMary/0.1 GitHub public repository research"
    request_interval_seconds: float = GITHUB_SEARCH_REQUEST_INTERVAL_SECONDS

    def search_repositories(
        self,
        company_name: str,
        *,
        per_page: int,
        page: int,
        timeout_seconds: float,
    ) -> GitHubRepositorySearchResponse:
        responses = [
            self._search_repository_url(
                request_url,
                per_page=per_page,
                page=page,
                timeout_seconds=timeout_seconds,
            )
            for request_url in _github_repository_search_api_urls(
                company_name,
                per_page=per_page,
                page=page,
            )
        ]
        combined_results: list[dict[str, Any]] = []
        for response in responses:
            combined_results.extend(response.results)
        return GitHubRepositorySearchResponse.model_validate(
            {
                "items": combined_results,
                "incomplete_results": any(
                    response.incomplete_results for response in responses
                ),
                "total_count": sum(response.total_count for response in responses),
                "has_next": any(response.has_next for response in responses),
            }
        )

    def _search_repository_url(
        self,
        request_url: str,
        *,
        per_page: int,
        page: int,
        timeout_seconds: float,
    ) -> GitHubRepositorySearchResponse:
        _validate_github_repository_search_url(request_url)
        _ensure_github_resolved_public_endpoint(request_url)
        if self.request_interval_seconds > 0:
            time.sleep(self.request_interval_seconds)
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        opener = _build_github_api_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_github_repository_search_url(final_url)
                _ensure_github_resolved_public_endpoint(final_url)
                raw_content = response.read(PUBLIC_API_MAX_BYTES + 1)
                if len(raw_content) > PUBLIC_API_MAX_BYTES:
                    raise GitHubApiError(
                        "The GitHub response was larger than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise GitHubApiError(f"GitHub returned HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError, WebResearchFetchError) as exc:
            raise GitHubApiError(f"Could not reach GitHub: {exc}") from exc
        if status_code != 200:
            raise GitHubApiError(f"GitHub returned HTTP {status_code}.")
        decoded_text = _decode_public_api_text(
            raw_content,
            charset=charset,
            source_name="GitHub",
        )
        try:
            response_payload = json.loads(decoded_text)
        except json.JSONDecodeError as exc:
            raise GitHubApiError(
                "GitHub returned a response that was not valid JSON."
            ) from exc
        if not isinstance(response_payload, dict):
            raise GitHubApiError(
                "GitHub returned an unexpected response: response was not a JSON object."
            )
        try:
            parsed_response = GitHubRepositorySearchResponse.model_validate(
                response_payload
            )
        except ValidationError as exc:
            detail = _first_validation_detail(exc)
            raise GitHubApiError(
                f"GitHub returned an unexpected response: {detail}"
            ) from exc
        response_payload["has_next"] = page * per_page < parsed_response.total_count
        try:
            return GitHubRepositorySearchResponse.model_validate(response_payload)
        except ValidationError as exc:
            detail = _first_validation_detail(exc)
            raise GitHubApiError(
                f"GitHub returned an unexpected response: {detail}"
            ) from exc


class _GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_github_repository_search_url(newurl, field_name="GitHub redirect URL")
        _ensure_github_resolved_public_endpoint(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not isinstance(redirected, urllib.request.Request):
            raise GitHubApiError("GitHub returned an unsupported redirect.")
        return redirected


def _build_github_api_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler("github"),
        _GuardedHTTPSHandler("github"),
        _GitHubRedirectHandler(),
    )


@dataclass(frozen=True)
class PublicSourceFile:
    provider_id: str
    path: Path
    description: str


def prepare_public_research_results(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    sec_form_d_results_path: Path | None = None,
    sam_gov_results_path: Path | None = None,
    usaspending_results_path: Path | None = None,
    sbir_results_path: Path | None = None,
    uspto_results_path: Path | None = None,
    github_results_path: Path | None = None,
    collected_at: datetime | None = None,
) -> ResearchCollectionRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchCollectionError(str(exc)) from exc

    collected_at = _as_utc(collected_at or datetime.now(UTC))
    companies = _clean_company_names(company_names or [])
    if not companies:
        raise ResearchCollectionError(
            "Pass at least one --company value. Hail Mary will only prepare public "
            "research results for companies you name explicitly."
        )
    source_files = _public_source_files(
        sec_form_d_results_path=sec_form_d_results_path,
        sam_gov_results_path=sam_gov_results_path,
        usaspending_results_path=usaspending_results_path,
        sbir_results_path=sbir_results_path,
        uspto_results_path=uspto_results_path,
        github_results_path=github_results_path,
    )
    if not source_files:
        raise ResearchCollectionError(
            "Add at least one local public-source file, such as --sec-form-d-results. "
            "Hail Mary will not fetch websites or software data feeds in this command."
        )

    source_results = [
        (
            source_file,
            _load_public_source_search_results(
                source_file.path,
                provider_id=source_file.provider_id,
                description=source_file.description,
            ),
        )
        for source_file in source_files
    ]
    adapters = [
        PublicSourceFileAdapter(
            provider_id=source_file.provider_id,
            client=LocalPublicSourceSearchClient(tuple(results_file.results)),
        )
        for source_file, results_file in source_results
    ]
    provider_ids = [adapter.provider_id for adapter in adapters]
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    skipped_non_exact_company_names = _skipped_non_exact_company_names(
        requested_company_names=companies,
        source_results=[
            result
            for _source_file, results_file in source_results
            for result in results_file.results
        ],
    )
    match_details = _source_result_match_details(
        requested_company_names=companies,
        source_results=[
            result
            for _source_file, results_file in source_results
            for result in results_file.results
        ],
    )
    results: list[ResearchResultInput] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    provider_result_counts: dict[str, int] = {provider_id: 0 for provider_id in provider_ids}
    provider_company_result_counts: dict[str, dict[str, int]] = {
        provider_id: {company_name: 0 for company_name in companies}
        for provider_id in provider_ids
    }
    for deal in deals:
        deal_results = _rank_public_results(
            [
                result
                for adapter in adapters
                for result in adapter.collect(deal, collected_at=collected_at)
            ],
            deal=deal,
            collected_at=collected_at,
        )
        results.extend(deal_results)
        for result in deal_results:
            provider_result_counts[result.provider_id] = (
                provider_result_counts.get(result.provider_id, 0) + 1
            )
            provider_company_counts = provider_company_result_counts.setdefault(
                result.provider_id,
                {},
            )
            provider_company_counts[result.company_name or deal.company_name] = (
                provider_company_counts.get(result.company_name or deal.company_name, 0)
                + 1
            )
        deal_summaries.append(
            ResearchCollectionDealSummary(
                company_name=deal.company_name,
                result_count=len(deal_results),
            )
        )

    if not results:
        return ResearchCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            provider_ids=provider_ids,
            deals=deal_summaries,
            provider_result_counts=provider_result_counts,
            provider_company_result_counts=provider_company_result_counts,
            skipped_non_exact_company_names=skipped_non_exact_company_names,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Prepared public research results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="public research results",
    )
    return ResearchCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        provider_ids=provider_ids,
        deals=deal_summaries,
        provider_result_counts=provider_result_counts,
        provider_company_result_counts=provider_company_result_counts,
        skipped_non_exact_company_names=skipped_non_exact_company_names,
        match_details=match_details,
    )


def collect_usaspending_awards(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    limit: int = 10,
    dry_run: bool = False,
    client: UsaspendingAwardsClient | None = None,
    collected_at: datetime | None = None,
    timeout_seconds: float = 20.0,
) -> UsaspendingCollectionRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchCollectionError(str(exc)) from exc
    _ensure_live_public_research_enabled(config, source_name="USAspending")
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    companies = _clean_company_names(company_names or [])
    if not companies:
        raise ResearchCollectionError(
            "Pass at least one --company value. Hail Mary will only send company "
            "names you list to USAspending."
        )
    _validate_usaspending_limit(limit)
    provider = _provider_by_id("usaspending")
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    if dry_run:
        return UsaspendingCollectionRunSummary(
            collected_at=collected_at,
            dry_run=True,
            deals=[
                ResearchCollectionDealSummary(company_name=deal.company_name)
                for deal in deals
            ],
        )

    awards_client = client or UrlLibUsaspendingAwardsClient()
    results: list[ResearchResultInput] = []
    run_warnings: list[str] = []
    match_details: list[CompanyMatch] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    for deal in deals:
        deal_results: list[ResearchResultInput] = []
        page = 1
        seen_awards: set[str] = set()
        try:
            while len(deal_results) < limit:
                awards_response = awards_client.search_awards(
                    deal.company_name,
                    limit=limit,
                    page=page,
                    timeout_seconds=timeout_seconds,
                )
                for raw_award in awards_response.results:
                    recipient_name = _usaspending_raw_recipient_name(raw_award)
                    if recipient_name is None:
                        continue
                    match = classify_company_match(deal.company_name, recipient_name)
                    match_details.append(match)
                    if not match.import_ready:
                        continue
                    award = _validate_usaspending_exact_award(raw_award, deal=deal)
                    dedupe_key = award.generated_internal_id or award.award_id
                    if dedupe_key is not None:
                        if dedupe_key in seen_awards:
                            continue
                        seen_awards.add(dedupe_key)
                    deal_results.append(
                        _research_result_from_usaspending_award(
                            deal,
                            award,
                            provider=provider,
                            collected_at=collected_at,
                            identity_match=match,
                        )
                    )
                    if len(deal_results) >= limit:
                        break
                if len(deal_results) >= limit or not awards_response.has_next_page:
                    break
                page += 1
                if page > USASPENDING_MAX_PAGES:
                    match_count = len(deal_results)
                    if match_count == 1:
                        match_text = (
                            "saved 1 exact recipient-name match it already validated"
                        )
                    elif match_count > 1:
                        match_text = (
                            f"saved {match_count} exact recipient-name matches it "
                            "already validated"
                        )
                    else:
                        match_text = "did not find exact recipient-name matches"
                    run_warnings.append(
                        "USAspending still had more fuzzy result pages for "
                        f"{deal.company_name} after Hail Mary checked "
                        f"{USASPENDING_MAX_PAGES} pages. Hail Mary {match_text}, "
                        "but more USAspending results may exist."
                    )
                    break
        except UsaspendingApiError as exc:
            raise ResearchCollectionError(str(exc)) from exc
        deal_results = _rank_public_results(
            deal_results,
            deal=deal,
            collected_at=collected_at,
        )
        results.extend(deal_results)
        deal_summaries.append(
            ResearchCollectionDealSummary(
                company_name=deal.company_name,
                result_count=len(deal_results),
            )
        )

    if not results:
        return UsaspendingCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            deals=deal_summaries,
            warnings=run_warnings,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Collected USAspending results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_usaspending_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="USAspending research results",
    )
    return UsaspendingCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        deals=deal_summaries,
        warnings=run_warnings,
        match_details=match_details,
    )


def collect_sbir_awards(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    limit: int = 10,
    dry_run: bool = False,
    client: SbirAwardsClient | None = None,
    collected_at: datetime | None = None,
    timeout_seconds: float = 20.0,
) -> SbirCollectionRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchCollectionError(str(exc)) from exc
    _ensure_live_public_research_enabled(config, source_name="SBIR/STTR")
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    companies = _clean_company_names(company_names or [])
    if not companies:
        raise ResearchCollectionError(
            "Pass at least one --company value. Hail Mary will only send company "
            "names you list to the SBIR/STTR public API."
        )
    _validate_sbir_limit(limit)
    provider = _provider_by_id("sbir")
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    if dry_run:
        return SbirCollectionRunSummary(
            collected_at=collected_at,
            dry_run=True,
            deals=[
                ResearchCollectionDealSummary(company_name=deal.company_name)
                for deal in deals
            ],
        )

    awards_client = client or UrlLibSbirAwardsClient()
    results: list[ResearchResultInput] = []
    run_warnings: list[str] = []
    match_details: list[CompanyMatch] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    for deal in deals:
        deal_results: list[ResearchResultInput] = []
        page = 1
        seen_awards: set[str] = set()
        try:
            while len(deal_results) < limit:
                start = (page - 1) * limit
                awards_response = awards_client.search_awards(
                    deal.company_name,
                    rows=limit,
                    start=start,
                    timeout_seconds=timeout_seconds,
                )
                source_api = _sbir_awards_api_url(
                    deal.company_name,
                    rows=limit,
                    start=start,
                )
                for raw_award in awards_response.results:
                    firm = _sbir_raw_firm(raw_award)
                    if firm is None:
                        continue
                    match = classify_company_match(deal.company_name, firm)
                    match_details.append(match)
                    if not match.import_ready:
                        continue
                    award = _validate_sbir_exact_award(raw_award, deal=deal)
                    dedupe_key = _sbir_award_dedupe_key(award)
                    if dedupe_key in seen_awards:
                        continue
                    seen_awards.add(dedupe_key)
                    deal_results.append(
                        _research_result_from_sbir_award(
                            deal,
                            award,
                            provider=provider,
                            collected_at=collected_at,
                            source_api=source_api,
                            identity_match=match,
                        )
                    )
                    if len(deal_results) >= limit:
                        break
                if len(deal_results) >= limit or awards_response.result_count < limit:
                    break
                if page >= SBIR_MAX_PAGES:
                    match_count = len(deal_results)
                    if match_count == 1:
                        match_text = (
                            "saved 1 exact firm-name match it already validated"
                        )
                    elif match_count > 1:
                        match_text = (
                            f"saved {match_count} exact firm-name matches it "
                            "already validated"
                        )
                    else:
                        match_text = "did not find exact firm-name matches"
                    run_warnings.append(
                        "SBIR/STTR still returned full fuzzy result pages for "
                        f"{deal.company_name} after Hail Mary checked "
                        f"{SBIR_MAX_PAGES} pages. Hail Mary {match_text}, "
                        "but more SBIR/STTR results may exist."
                    )
                    break
                page += 1
        except SbirApiError as exc:
            raise ResearchCollectionError(str(exc)) from exc
        deal_results = _rank_public_results(
            deal_results,
            deal=deal,
            collected_at=collected_at,
        )
        results.extend(deal_results)
        deal_summaries.append(
            ResearchCollectionDealSummary(
                company_name=deal.company_name,
                result_count=len(deal_results),
            )
        )

    if not results:
        return SbirCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            deals=deal_summaries,
            warnings=run_warnings,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Collected SBIR/STTR results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_sbir_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="SBIR/STTR research results",
    )
    return SbirCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        deals=deal_summaries,
        warnings=run_warnings,
        match_details=match_details,
    )


def collect_sec_form_d_filings(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    limit: int = 10,
    dry_run: bool = False,
    client: SecFormDFilingsClient | None = None,
    collected_at: datetime | None = None,
    timeout_seconds: float = 20.0,
) -> SecFormDCollectionRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchCollectionError(str(exc)) from exc
    _ensure_live_public_research_enabled(config, source_name="SEC Form D")
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    companies = _clean_company_names(company_names or [])
    if not companies:
        raise ResearchCollectionError(
            "Pass at least one --company value. Hail Mary will only send company "
            "names you list to SEC EDGAR."
        )
    _validate_sec_form_d_limit(limit)
    provider = _provider_by_id("sec_form_d")
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    if dry_run:
        return SecFormDCollectionRunSummary(
            collected_at=collected_at,
            dry_run=True,
            deals=[
                ResearchCollectionDealSummary(company_name=deal.company_name)
                for deal in deals
            ],
        )

    filings_client = client or UrlLibSecFormDFilingsClient(
        user_agent=_sec_form_d_user_agent_from_env()
    )
    results: list[ResearchResultInput] = []
    run_warnings: list[str] = []
    match_details: list[CompanyMatch] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    for deal in deals:
        deal_results: list[ResearchResultInput] = []
        seen_filings: set[str] = set()
        page = 1
        try:
            while len(deal_results) < limit:
                start = (page - 1) * limit
                filings_response = filings_client.search_filings(
                    deal.company_name,
                    count=limit,
                    start=start,
                    timeout_seconds=timeout_seconds,
                )
                run_warnings.extend(filings_response.warnings)
                for raw_filing in filings_response.results:
                    issuer_name = _sec_form_d_raw_issuer_name(raw_filing)
                    if issuer_name is None:
                        continue
                    match = classify_company_match(deal.company_name, issuer_name)
                    match_details.append(match)
                    if not match.import_ready:
                        continue
                    filing = _validate_sec_form_d_exact_filing(raw_filing, deal=deal)
                    dedupe_key = filing.accession_number
                    if dedupe_key in seen_filings:
                        continue
                    seen_filings.add(dedupe_key)
                    deal_results.append(
                        _research_result_from_sec_form_d_filing(
                            deal,
                            filing,
                            provider=provider,
                            collected_at=collected_at,
                            identity_match=match,
                        )
                    )
                    if len(deal_results) >= limit:
                        break
                if len(deal_results) >= limit or not filings_response.has_next:
                    break
                if page >= SEC_FORM_D_MAX_PAGES:
                    match_count = len(deal_results)
                    if match_count == 1:
                        match_text = (
                            "saved 1 exact issuer-name match it already validated"
                        )
                    elif match_count > 1:
                        match_text = (
                            f"saved {match_count} exact issuer-name matches it "
                            "already validated"
                        )
                    else:
                        match_text = "did not find exact issuer-name matches"
                    run_warnings.append(
                        "SEC EDGAR still had more company-search result pages for "
                        f"{deal.company_name} after Hail Mary checked "
                        f"{SEC_FORM_D_MAX_PAGES} pages. Hail Mary {match_text}, "
                        "but more SEC Form D filings may exist."
                    )
                    break
                page += 1
        except SecFormDApiError as exc:
            raise ResearchCollectionError(str(exc)) from exc
        deal_results = _rank_public_results(
            deal_results,
            deal=deal,
            collected_at=collected_at,
        )
        results.extend(deal_results)
        deal_summaries.append(
            ResearchCollectionDealSummary(
                company_name=deal.company_name,
                result_count=len(deal_results),
            )
        )

    if not results:
        return SecFormDCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            deals=deal_summaries,
            warnings=run_warnings,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Collected SEC Form D results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_sec_form_d_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="SEC Form D research results",
    )
    return SecFormDCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        deals=deal_summaries,
        warnings=run_warnings,
        match_details=match_details,
    )


def collect_github_repositories(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    limit: int = 10,
    dry_run: bool = False,
    client: GitHubRepositorySearchClient | None = None,
    collected_at: datetime | None = None,
    timeout_seconds: float = 20.0,
) -> GitHubRepositoryCollectionRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchCollectionError(str(exc)) from exc
    _ensure_live_public_research_enabled(config, source_name="GitHub")
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    companies = _clean_company_names(company_names or [])
    if not companies:
        raise ResearchCollectionError(
            "Pass at least one --company value. Hail Mary will only send company "
            "names you list to the GitHub public repository search API."
        )
    _validate_github_repository_limit(limit)
    provider = _provider_by_id("github")
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    if dry_run:
        return GitHubRepositoryCollectionRunSummary(
            collected_at=collected_at,
            dry_run=True,
            deals=[
                ResearchCollectionDealSummary(company_name=deal.company_name)
                for deal in deals
            ],
        )

    repository_client = client or UrlLibGitHubRepositorySearchClient()
    results: list[ResearchResultInput] = []
    run_warnings: list[str] = []
    match_details: list[CompanyMatch] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    for deal in deals:
        deal_results: list[ResearchResultInput] = []
        seen_repositories: set[str] = set()
        page = 1
        try:
            while len(deal_results) < limit:
                repositories_response = repository_client.search_repositories(
                    deal.company_name,
                    per_page=limit,
                    page=page,
                    timeout_seconds=timeout_seconds,
                )
                for raw_repository in _prioritize_github_repository_matches(
                    repositories_response.results,
                    deal.company_name,
                ):
                    match = _github_repository_company_match(
                        raw_repository,
                        deal.company_name,
                    )
                    match_details.append(match)
                    if not match.import_ready:
                        continue
                    repository = _validate_github_exact_repository(
                        raw_repository,
                        deal=deal,
                    )
                    dedupe_key = repository.full_name.casefold()
                    if dedupe_key in seen_repositories:
                        continue
                    seen_repositories.add(dedupe_key)
                    deal_results.append(
                        _research_result_from_github_repository(
                            deal,
                            repository,
                            provider=provider,
                            collected_at=collected_at,
                            identity_match=match,
                        )
                    )
                    if len(deal_results) >= limit:
                        break
                if repositories_response.incomplete_results:
                    run_warnings.append(
                        "GitHub marked repository search results for "
                        f"{deal.company_name} as incomplete. Hail Mary saved exact "
                        "owner matches it already validated, but "
                        "more GitHub results may exist."
                    )
                    break
                if len(deal_results) >= limit or not repositories_response.has_next:
                    break
                if page >= GITHUB_MAX_PAGES:
                    match_count = len(deal_results)
                    if match_count == 1:
                        match_text = "saved 1 exact owner match"
                    elif match_count > 1:
                        match_text = f"saved {match_count} exact owner matches"
                    else:
                        match_text = "did not find exact owner matches"
                    run_warnings.append(
                        "GitHub still returned full repository-search pages for "
                        f"{deal.company_name} after Hail Mary checked "
                        f"{GITHUB_MAX_PAGES} pages. Hail Mary {match_text}, but "
                        "more GitHub results may exist."
                    )
                    break
                page += 1
        except GitHubApiError as exc:
            raise ResearchCollectionError(str(exc)) from exc
        deal_results = _rank_public_results(
            deal_results,
            deal=deal,
            collected_at=collected_at,
        )
        results.extend(deal_results)
        deal_summaries.append(
            ResearchCollectionDealSummary(
                company_name=deal.company_name,
                result_count=len(deal_results),
            )
        )

    if not results:
        return GitHubRepositoryCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            deals=deal_summaries,
            warnings=run_warnings,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Collected GitHub results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_github_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="GitHub research results",
    )
    return GitHubRepositoryCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        deals=deal_summaries,
        warnings=run_warnings,
        match_details=match_details,
    )


def _public_source_files(
    *,
    sec_form_d_results_path: Path | None,
    sam_gov_results_path: Path | None,
    usaspending_results_path: Path | None,
    sbir_results_path: Path | None,
    uspto_results_path: Path | None,
    github_results_path: Path | None,
) -> list[PublicSourceFile]:
    candidates = [
        (sec_form_d_results_path, "sec_form_d"),
        (sam_gov_results_path, "sam_gov"),
        (usaspending_results_path, "usaspending"),
        (sbir_results_path, "sbir"),
        (uspto_results_path, "uspto"),
        (github_results_path, "github"),
    ]
    return [
        PublicSourceFile(
            provider_id=provider_id,
            path=path,
            description=_public_source_adapter_config(provider_id).description,
        )
        for path, provider_id in candidates
        if path is not None
    ]


def _load_public_source_search_results(
    path: Path,
    *,
    provider_id: str,
    description: str,
) -> PublicSourceSearchResultsFile:
    provider = _provider_by_id(provider_id)
    input_path = _resolve_input_file(path, description=description)
    try:
        raw_text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchCollectionError(
            f"The {description} file is not plain UTF-8 text. Save it as JSON text "
            "and try again."
        ) from exc
    except OSError as exc:
        raise ResearchCollectionError(
            f"Could not read the {description} file at {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResearchCollectionError(
            f"The {description} file is not valid JSON: {exc.msg}."
        ) from exc
    if not isinstance(payload, dict):
        raise ResearchCollectionError(
            f"The {description} file must be a JSON object with a `results` list."
        )
    try:
        results_file = PublicSourceSearchResultsFile.model_validate(payload)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"The {description} file is incomplete: {detail}"
        ) from exc
    for index, result in enumerate(results_file.results, start=1):
        if result.source_url is not None:
            try:
                validate_provider_source_url(provider_id, result.source_url)
            except ValueError as exc:
                raise ResearchCollectionError(
                    f"{provider.name} result {index} has an invalid source_url: {exc}"
                ) from exc
        if result.source_api is not None and source_reference_looks_like_url(result.source_api):
            try:
                validate_provider_source_url(
                    provider_id,
                    result.source_api,
                    field_name="source_api",
                )
            except ValueError as exc:
                raise ResearchCollectionError(
                    f"{provider.name} result {index} has an invalid source_api: {exc}"
                ) from exc
    return results_file


def _load_sec_form_d_search_results(path: Path) -> SecFormDSearchResultsFile:
    return _load_public_source_search_results(
        path,
        provider_id="sec_form_d",
        description="SEC Form D results",
    )


def _ensure_live_public_research_enabled(
    config: AppConfig,
    *,
    source_name: str,
) -> None:
    _ = (config, source_name)


def _validate_usaspending_limit(limit: int) -> None:
    if limit < 1 or limit > 25:
        raise ResearchCollectionError(
            "USAspending result limit must be between 1 and 25 per company."
        )


def _validate_sbir_limit(limit: int) -> None:
    if limit < 1 or limit > 25:
        raise ResearchCollectionError(
            "SBIR/STTR result limit must be between 1 and 25 per company."
        )


def _validate_sec_form_d_limit(limit: int) -> None:
    if limit < 1 or limit > 25:
        raise ResearchCollectionError(
            "SEC Form D result limit must be between 1 and 25 per company."
        )


def _validate_github_repository_limit(limit: int) -> None:
    if limit < 1 or limit > 25:
        raise ResearchCollectionError(
            "GitHub repository result limit must be between 1 and 25 per company."
        )


def _sec_form_d_user_agent_from_env() -> str:
    value = os.environ.get(SEC_FORM_D_USER_AGENT_ENV_VAR, "").strip()
    return _validate_sec_form_d_user_agent(value)


def _validate_sec_form_d_user_agent(value: str) -> str:
    value = value.strip()
    if not value:
        raise ResearchCollectionError(
            f"Set {SEC_FORM_D_USER_AGENT_ENV_VAR} before live SEC Form D collection. "
            "SEC EDGAR asks automated clients to send a User-Agent that includes "
            "an application or company name and a contact email address."
        )
    if not re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
        raise ResearchCollectionError(
            f"{SEC_FORM_D_USER_AGENT_ENV_VAR} must include a contact email address "
            "for SEC EDGAR fair-access identification."
        )
    return value


def _validate_usaspending_api_url(url: str) -> None:
    try:
        validate_provider_source_url(
            "usaspending",
            url,
            field_name="USAspending API URL",
        )
    except ValueError as exc:
        raise UsaspendingApiError(str(exc)) from exc
    if url != USASPENDING_AWARDS_ENDPOINT:
        raise UsaspendingApiError(
            "USAspending redirected the request away from the expected public API endpoint."
        )


def _validate_sbir_api_url(url: str) -> None:
    try:
        validate_provider_source_url(
            "sbir",
            url,
            field_name="SBIR/STTR API URL",
        )
    except ValueError as exc:
        raise SbirApiError(str(exc)) from exc
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or host != "api.www.sbir.gov"
        or path != "/public/api/awards"
    ):
        raise SbirApiError(
            "SBIR/STTR redirected the request away from the expected public API endpoint."
        )


def _validate_sec_form_d_public_url(url: str, *, field_name: str) -> None:
    try:
        validate_provider_source_url("sec_form_d", url, field_name=field_name)
    except ValueError as exc:
        raise SecFormDApiError(str(exc)) from exc
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SecFormDApiError(f"{field_name} must stay on HTTPS.")


def _validate_sec_form_d_atom_url(url: str) -> None:
    _validate_sec_form_d_public_url(url, field_name="SEC EDGAR API URL")
    if not _sec_form_d_url_is_atom_endpoint(url):
        raise SecFormDApiError(
            "SEC EDGAR redirected the request away from the expected public API endpoint."
        )


def _sec_form_d_url_is_atom_endpoint(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    return (
        parsed.scheme == "https"
        and host == "www.sec.gov"
        and path == "/cgi-bin/browse-edgar"
    )


def _validate_sec_form_d_archive_url(url: str, *, field_name: str) -> None:
    _validate_sec_form_d_public_url(url, field_name=field_name)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if host != "www.sec.gov" or not parsed.path.startswith("/Archives/edgar/data/"):
        raise SecFormDApiError(
            f"{field_name} must stay under the SEC EDGAR archive path."
        )


def _validate_github_public_url(url: str, *, field_name: str) -> None:
    try:
        validate_provider_source_url("github", url, field_name=field_name)
    except ValueError as exc:
        raise GitHubApiError(str(exc)) from exc
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise GitHubApiError(f"{field_name} must stay on HTTPS.")


def _validate_github_repository_search_url(
    url: str,
    *,
    field_name: str = "GitHub API URL",
) -> None:
    _validate_github_public_url(url, field_name=field_name)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    if host != "api.github.com" or path != "/search/repositories":
        raise GitHubApiError(
            "GitHub redirected the request away from the expected public API endpoint."
        )


def _ensure_usaspending_resolved_public_endpoint(url: str) -> None:
    _validate_usaspending_api_url(url)
    try:
        _ensure_resolved_public_host(url)
    except WebResearchFetchError as exc:
        message = str(exc)
        message = message.replace("The page URL", "The USAspending API URL")
        message = message.replace("Website host", "USAspending host")
        raise UsaspendingApiError(message) from exc


def _ensure_sbir_resolved_public_endpoint(url: str) -> None:
    _validate_sbir_api_url(url)
    try:
        _ensure_resolved_public_host(url)
    except WebResearchFetchError as exc:
        message = str(exc)
        message = message.replace("The page URL", "The SBIR/STTR API URL")
        message = message.replace("Website host", "SBIR/STTR host")
        raise SbirApiError(message) from exc


def _ensure_sec_form_d_resolved_public_endpoint(url: str) -> None:
    _validate_sec_form_d_public_url(url, field_name="SEC URL")
    try:
        _ensure_resolved_public_host(url)
    except WebResearchFetchError as exc:
        message = str(exc)
        message = message.replace("The page URL", "The SEC URL")
        message = message.replace("Website host", "SEC host")
        raise SecFormDApiError(message) from exc


def _ensure_github_resolved_public_endpoint(url: str) -> None:
    _validate_github_public_url(url, field_name="GitHub URL")
    try:
        _ensure_resolved_public_host(url)
    except WebResearchFetchError as exc:
        message = str(exc)
        message = message.replace("The page URL", "The GitHub URL")
        message = message.replace("Website host", "GitHub host")
        raise GitHubApiError(message) from exc


def _usaspending_awards_payload(
    company_name: str,
    *,
    limit: int,
    page: int,
) -> dict[str, object]:
    return {
        "subawards": False,
        "limit": limit,
        "page": page,
        "sort": "Award Amount",
        "order": "desc",
        "filters": {
            "recipient_search_text": [company_name],
            "award_type_codes": USASPENDING_AWARD_TYPE_CODES,
        },
        "fields": USASPENDING_AWARD_FIELDS,
    }


def _sbir_awards_api_url(
    company_name: str,
    *,
    rows: int,
    start: int,
) -> str:
    query = urlencode({"firm": company_name, "rows": rows, "start": start})
    return f"{SBIR_AWARDS_ENDPOINT}?{query}"


def _sec_form_d_atom_api_url(
    company_name: str,
    *,
    count: int,
    start: int,
) -> str:
    query = urlencode(
        {
            "action": "getcompany",
            "company": company_name,
            "type": "D",
            "owner": "exclude",
            "output": "atom",
            "count": count,
            "start": start,
        }
    )
    return f"{SEC_FORM_D_ATOM_ENDPOINT}?{query}"


def _github_repository_search_api_url(
    company_name: str,
    *,
    per_page: int,
    page: int,
    query: str | None = None,
) -> str:
    query = urlencode(
        {
            "q": query or f"{company_name} in:name fork:false",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }
    )
    return f"{GITHUB_REPOSITORY_SEARCH_ENDPOINT}?{query}"


def _github_repository_search_api_urls(
    company_name: str,
    *,
    per_page: int,
    page: int,
) -> list[str]:
    requested_slug = _normalize_company_slug(company_name)
    queries: list[str] = []
    if requested_slug:
        queries.extend(
            [
                f"user:{requested_slug} fork:false",
                f"org:{requested_slug} fork:false",
            ]
        )
    queries.append(f"{company_name} in:name fork:false")
    return [
        _github_repository_search_api_url(
            company_name,
            per_page=per_page,
            page=page,
            query=query,
        )
        for query in queries
    ]


def _validate_sbir_awards_response(response_payload: object) -> SbirAwardsResponse:
    if not isinstance(response_payload, list):
        return SbirAwardsResponse.model_validate(response_payload)
    for index, raw_result in enumerate(response_payload, start=1):
        if not isinstance(raw_result, dict):
            raise SbirApiError(
                "SBIR/STTR returned an unexpected response: "
                f"result {index} was not a JSON object."
            )
    return SbirAwardsResponse.model_validate({"results": response_payload})


def _usaspending_raw_recipient_name(raw_award: dict[str, Any]) -> str | None:
    value = raw_award.get("Recipient Name")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sbir_raw_firm(raw_award: dict[str, Any]) -> str | None:
    value = raw_award.get("firm")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sec_form_d_raw_issuer_name(raw_filing: dict[str, Any]) -> str | None:
    value = raw_filing.get("issuer_name")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _github_raw_repository_matches_company(
    raw_repository: dict[str, Any],
    company_name: str,
) -> bool:
    return _github_repository_company_match(raw_repository, company_name).import_ready


def _prioritize_github_repository_matches(
    raw_repositories: list[dict[str, Any]],
    company_name: str,
) -> list[dict[str, Any]]:
    return [
        raw_repository
        for _, raw_repository in sorted(
            enumerate(raw_repositories),
            key=lambda indexed_repository: (
                _github_repository_match_priority(
                    indexed_repository[1],
                    company_name,
                ),
                indexed_repository[0],
            ),
        )
    ]


def _github_repository_match_priority(
    raw_repository: dict[str, Any],
    company_name: str,
) -> int:
    requested_slug = _normalize_company_slug(company_name)
    if not requested_slug:
        return 3
    owner = raw_repository.get("owner")
    if isinstance(owner, dict):
        owner_login = owner.get("login")
        if (
            isinstance(owner_login, str)
            and _normalize_company_slug(owner_login) == requested_slug
        ):
            return 0
    owner_login = raw_repository.get("owner_login")
    if (
        isinstance(owner_login, str)
        and _normalize_company_slug(owner_login) == requested_slug
    ):
        return 0
    name = raw_repository.get("name")
    if isinstance(name, str) and _normalize_company_slug(name) == requested_slug:
        return 1
    match = _github_repository_company_match(raw_repository, company_name)
    if match.kind == CompanyMatchKind.RELATED:
        return 2
    return 3


def _github_repository_company_match(
    raw_repository: dict[str, Any],
    company_name: str,
) -> CompanyMatch:
    requested_slug = _normalize_company_slug(company_name)
    owner_login = _github_raw_owner_login(raw_repository)
    if owner_login is not None and _normalize_company_slug(owner_login) == requested_slug:
        return CompanyMatch(
            requested_name=company_name,
            candidate_name=owner_login,
            kind=CompanyMatchKind.EXACT,
            reason="GitHub owner slug matches the normalized company name exactly.",
            normalized_requested=_normalize_company_name(company_name),
            normalized_candidate=_normalize_company_name(owner_login),
        )

    repository_name = raw_repository.get("name")
    if (
        isinstance(repository_name, str)
        and _normalize_company_slug(repository_name) == requested_slug
    ):
        full_name = raw_repository.get("full_name")
        display_name = full_name if isinstance(full_name, str) and full_name else repository_name
        return CompanyMatch(
            requested_name=company_name,
            candidate_name=display_name,
            kind=CompanyMatchKind.LIKELY,
            reason=(
                "GitHub repository name matches, but the owner does not. Operator "
                "validation is required before import."
            ),
            normalized_requested=_normalize_company_name(company_name),
            normalized_candidate=_normalize_company_name(repository_name),
        )

    candidate_name = raw_repository.get("full_name")
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        candidate_name = repository_name if isinstance(repository_name, str) else ""
    return classify_company_match(company_name, candidate_name)


def _github_raw_owner_login(raw_repository: dict[str, Any]) -> str | None:
    owner = raw_repository.get("owner")
    if isinstance(owner, dict):
        owner_login = owner.get("login")
        if isinstance(owner_login, str) and owner_login.strip():
            return owner_login.strip()
    owner_login = raw_repository.get("owner_login")
    if isinstance(owner_login, str) and owner_login.strip():
        return owner_login.strip()
    return None


def _validate_usaspending_exact_award(
    raw_award: dict[str, Any],
    *,
    deal: ResearchCollectionDeal,
) -> UsaspendingAwardRecord:
    try:
        return UsaspendingAwardRecord.model_validate(raw_award)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"An exact USAspending result for {deal.company_name} is incomplete: "
            f"{detail}"
        ) from exc


def _validate_sbir_exact_award(
    raw_award: dict[str, Any],
    *,
    deal: ResearchCollectionDeal,
) -> SbirAwardRecord:
    try:
        return SbirAwardRecord.model_validate(raw_award)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"An exact SBIR/STTR result for {deal.company_name} is incomplete: "
            f"{detail}"
        ) from exc


def _validate_sec_form_d_exact_filing(
    raw_filing: dict[str, Any],
    *,
    deal: ResearchCollectionDeal,
) -> SecFormDFilingRecord:
    try:
        return SecFormDFilingRecord.model_validate(raw_filing)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"An exact SEC Form D result for {deal.company_name} is incomplete: "
            f"{detail}"
        ) from exc


def _validate_github_exact_repository(
    raw_repository: dict[str, Any],
    *,
    deal: ResearchCollectionDeal,
) -> GitHubRepositoryRecord:
    try:
        return GitHubRepositoryRecord.model_validate(raw_repository)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"An exact GitHub repository result for {deal.company_name} is incomplete: "
            f"{detail}"
        ) from exc


def _research_result_from_usaspending_award(
    deal: ResearchCollectionDeal,
    award: UsaspendingAwardRecord,
    *,
    provider: ResearchProvider,
    collected_at: datetime,
    identity_match: CompanyMatch,
) -> ResearchResultInput:
    award_id = award.award_id
    generated_internal_id = award.generated_internal_id
    if not award_id or not generated_internal_id:
        missing_fields = []
        if not award_id:
            missing_fields.append("Award ID")
        if not generated_internal_id:
            missing_fields.append("generated_internal_id")
        raise ResearchCollectionError(
            "An exact USAspending result for "
            f"{deal.company_name} is missing {', '.join(missing_fields)}. "
            "Hail Mary cannot save it as evidence until USAspending returns the "
            "award identifier fields."
        )
    source_url = _usaspending_award_url(generated_internal_id)
    validate_provider_source_url(provider.id, source_url)
    try:
        return ResearchResultInput(
            company_name=deal.company_name,
            provider_id=provider.id,
            provider_name=provider.name,
            title=f"USAspending award {award_id} for {award.recipient_name}",
            text=_usaspending_award_text(award),
            retrieved_at=collected_at,
            source_url=source_url,
            confidence=(
                "medium: exact recipient name match from the USAspending public API; "
                "Hail Mary did not verify entity identity"
            ),
            licensing_notes=(
                f"{provider.licensing_notes} Automatically fetched from the public "
                "USAspending API. Confirm recipient identity before relying on it."
            ),
            source_kind=provider.source_kind,
            source_reliability=source_reliability_for_provider(
                provider.id,
                source_kind=provider.source_kind,
                document_type=DocumentType.WEB_PAGE,
            ),
            identity_match_kind=identity_match.kind,
            identity_match_reason=identity_match.reason,
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"USAspending result for {deal.company_name} is incomplete: {detail}"
        ) from exc


def _research_result_from_sbir_award(
    deal: ResearchCollectionDeal,
    award: SbirAwardRecord,
    *,
    provider: ResearchProvider,
    collected_at: datetime,
    source_api: str,
    identity_match: CompanyMatch,
) -> ResearchResultInput:
    validate_provider_source_url(provider.id, source_api, field_name="source_api")
    if award.award_link is not None:
        validate_provider_source_url(provider.id, award.award_link)
    try:
        return ResearchResultInput(
            company_name=deal.company_name,
            provider_id=provider.id,
            provider_name=provider.name,
            title=f"SBIR/STTR award {award.award_title} for {award.firm}",
            text=_sbir_award_text(award),
            retrieved_at=collected_at,
            source_url=award.award_link,
            source_api=source_api,
            confidence=(
                "medium: exact firm name match from the SBIR/STTR public API; "
                "Hail Mary did not verify entity identity"
            ),
            licensing_notes=(
                f"{provider.licensing_notes} Automatically fetched from the public "
                "SBIR/STTR API. Confirm firm identity before relying on it."
            ),
            source_kind=provider.source_kind,
            source_reliability=source_reliability_for_provider(
                provider.id,
                source_kind=provider.source_kind,
                document_type=DocumentType.WEB_PAGE,
            ),
            identity_match_kind=identity_match.kind,
            identity_match_reason=identity_match.reason,
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"SBIR/STTR result for {deal.company_name} is incomplete: {detail}"
        ) from exc


def _research_result_from_sec_form_d_filing(
    deal: ResearchCollectionDeal,
    filing: SecFormDFilingRecord,
    *,
    provider: ResearchProvider,
    collected_at: datetime,
    identity_match: CompanyMatch,
) -> ResearchResultInput:
    try:
        return ResearchResultInput(
            company_name=deal.company_name,
            provider_id=provider.id,
            provider_name=provider.name,
            title=f"SEC Form {filing.filing_type} filing for {filing.issuer_name}",
            text=_sec_form_d_filing_text(filing),
            retrieved_at=collected_at,
            source_url=filing.source_url,
            source_api=filing.source_api,
            confidence=(
                "medium: exact issuer name match from SEC EDGAR public filing "
                "metadata; Hail Mary did not verify entity identity"
            ),
            licensing_notes=(
                f"{provider.licensing_notes} Automatically fetched from public SEC "
                "EDGAR metadata. Hail Mary saved parsed filing fields only, not raw "
                "filings or contact details."
            ),
            source_kind=provider.source_kind,
            source_reliability=source_reliability_for_provider(
                provider.id,
                source_kind=provider.source_kind,
                document_type=DocumentType.WEB_PAGE,
            ),
            identity_match_kind=identity_match.kind,
            identity_match_reason=identity_match.reason,
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"SEC Form D result for {deal.company_name} is incomplete: {detail}"
        ) from exc


def _research_result_from_github_repository(
    deal: ResearchCollectionDeal,
    repository: GitHubRepositoryRecord,
    *,
    provider: ResearchProvider,
    collected_at: datetime,
    identity_match: CompanyMatch,
) -> ResearchResultInput:
    try:
        return ResearchResultInput(
            company_name=deal.company_name,
            provider_id=provider.id,
            provider_name=provider.name,
            title=f"GitHub repository {repository.full_name}",
            text=_github_repository_text(repository),
            retrieved_at=collected_at,
            source_url=repository.html_url,
            source_api=repository.url,
            confidence=(
                "medium: exact GitHub owner match from the "
                "public repository search API; Hail Mary did not verify entity identity"
            ),
            licensing_notes=(
                f"{provider.licensing_notes} Automatically fetched repository "
                "metadata from the public GitHub API. Hail Mary did not clone code, "
                "fetch README files, or save repository contents."
            ),
            source_kind=provider.source_kind,
            source_reliability=source_reliability_for_provider(
                provider.id,
                source_kind=provider.source_kind,
                document_type=DocumentType.WEB_PAGE,
            ),
            identity_match_kind=identity_match.kind,
            identity_match_reason=identity_match.reason,
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"GitHub result for {deal.company_name} is incomplete: {detail}"
        ) from exc


def _usaspending_award_url(generated_internal_id: str) -> str:
    return f"https://www.usaspending.gov/award/{quote(generated_internal_id, safe='')}"


def _sbir_award_dedupe_key(award: SbirAwardRecord) -> str:
    for value in (
        award.award_link,
        award.agency_tracking_number,
        award.contract,
    ):
        if value:
            return value
    return "\0".join(
        [
            award.firm,
            award.award_title or "",
            award.agency or "",
            award.proposal_award_date or "",
        ]
    )


def _usaspending_award_text(award: UsaspendingAwardRecord) -> str:
    parts = [
        f"Award ID: {award.award_id}.",
        f"Recipient: {award.recipient_name}.",
    ]
    if award.recipient_uei:
        parts.append(f"Recipient UEI: {award.recipient_uei}.")
    if award.award_amount is not None:
        parts.append(f"Award amount: {_format_money(award.award_amount)}.")
    if award.award_type:
        parts.append(f"Award type: {award.award_type}.")
    if award.start_date or award.end_date:
        date_range = " to ".join(
            date for date in [award.start_date, award.end_date] if date
        )
        parts.append(f"Period: {date_range}.")
    if award.awarding_agency:
        parts.append(f"Awarding agency: {award.awarding_agency}.")
    if award.awarding_sub_agency:
        parts.append(f"Awarding sub-agency: {award.awarding_sub_agency}.")
    if award.funding_agency:
        parts.append(f"Funding agency: {award.funding_agency}.")
    if award.funding_sub_agency:
        parts.append(f"Funding sub-agency: {award.funding_sub_agency}.")
    if award.description:
        parts.append(f"Description: {award.description}.")
    return " ".join(parts)


def _sbir_award_text(award: SbirAwardRecord) -> str:
    parts = [
        f"Firm: {award.firm}.",
        f"Award title: {award.award_title}.",
    ]
    if award.agency:
        parts.append(f"Agency: {award.agency}.")
    if award.branch:
        parts.append(f"Branch: {award.branch}.")
    if award.phase:
        parts.append(f"Phase: {award.phase}.")
    if award.program:
        parts.append(f"Program: {award.program}.")
    if award.award_amount is not None:
        parts.append(f"Award amount: {_format_money(award.award_amount)}.")
    if award.proposal_award_date or award.contract_end_date:
        date_range = " to ".join(
            date
            for date in [award.proposal_award_date, award.contract_end_date]
            if date
        )
        parts.append(f"Period: {date_range}.")
    if award.agency_tracking_number:
        parts.append(f"Agency tracking number: {award.agency_tracking_number}.")
    if award.contract:
        parts.append(f"Contract: {award.contract}.")
    if award.solicitation_number:
        parts.append(f"Solicitation number: {award.solicitation_number}.")
    if award.solicitation_year:
        parts.append(f"Solicitation year: {award.solicitation_year}.")
    if award.topic_code:
        parts.append(f"Topic code: {award.topic_code}.")
    if award.award_year:
        parts.append(f"Award year: {award.award_year}.")
    if award.uei:
        parts.append(f"UEI: {award.uei}.")
    if award.research_area_keywords:
        parts.append(f"Research area keywords: {award.research_area_keywords}.")
    if award.abstract:
        parts.append(f"Abstract: {award.abstract}.")
    return " ".join(parts)


def _sec_form_d_filing_text(filing: SecFormDFilingRecord) -> str:
    parts = [
        f"Issuer: {filing.issuer_name}.",
        f"Filing type: {filing.filing_type}.",
        f"Accession number: {filing.accession_number}.",
    ]
    if filing.filing_date:
        parts.append(f"Filing date: {filing.filing_date}.")
    if filing.form_name:
        parts.append(f"Form name: {filing.form_name}.")
    if filing.total_offering_amount:
        parts.append(f"Total offering amount: {filing.total_offering_amount}.")
    if filing.total_amount_sold:
        parts.append(f"Total amount sold: {filing.total_amount_sold}.")
    if filing.total_remaining:
        parts.append(f"Total remaining: {filing.total_remaining}.")
    if filing.minimum_investment_accepted:
        parts.append(
            f"Minimum investment accepted: {filing.minimum_investment_accepted}."
        )
    if filing.total_investors:
        parts.append(f"Total investors: {filing.total_investors}.")
    if filing.industry_group:
        parts.append(f"Industry group: {filing.industry_group}.")
    if filing.revenue_range:
        parts.append(f"Revenue range: {filing.revenue_range}.")
    if filing.federal_exemptions:
        exemptions = ", ".join(filing.federal_exemptions)
        parts.append(f"Federal exemptions listed: {exemptions}.")
    return " ".join(parts)


def _github_repository_text(repository: GitHubRepositoryRecord) -> str:
    parts = [
        f"Repository: {repository.full_name}.",
        f"Owner: {repository.owner_login}.",
        f"Repository name: {repository.name}.",
    ]
    if repository.description:
        parts.append(f"Description: {repository.description}.")
    if repository.language:
        parts.append(f"Primary language: {repository.language}.")
    if repository.stargazers_count is not None:
        parts.append(f"Stars: {repository.stargazers_count}.")
    if repository.forks_count is not None:
        parts.append(f"Forks: {repository.forks_count}.")
    if repository.open_issues_count is not None:
        parts.append(f"Open issues: {repository.open_issues_count}.")
    if repository.created_at is not None:
        parts.append(f"Created at: {repository.created_at.date().isoformat()}.")
    if repository.updated_at is not None:
        parts.append(f"Updated at: {repository.updated_at.date().isoformat()}.")
    if repository.pushed_at is not None:
        parts.append(f"Last pushed at: {repository.pushed_at.date().isoformat()}.")
    if repository.license_name:
        parts.append(f"License: {repository.license_name}.")
    status_flags = []
    if repository.fork:
        status_flags.append("fork")
    if repository.archived:
        status_flags.append("archived")
    if repository.disabled:
        status_flags.append("disabled")
    if status_flags:
        parts.append(f"Repository status: {', '.join(status_flags)}.")
    return " ".join(parts)


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _decode_public_api_text(
    raw_content: bytes,
    *,
    charset: str,
    source_name: str,
) -> str:
    try:
        return raw_content.decode(charset, errors="replace")
    except LookupError:
        return raw_content.decode("utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise ResearchCollectionError(
            f"{source_name} returned text that could not be decoded."
        ) from exc


def _parse_sec_form_d_atom_response(
    xml_text: str,
    *,
    requested_company_name: str,
    source_api: str,
    fetch_submission: Callable[[str], dict[str, str | list[str] | None]],
) -> SecFormDFilingsResponse:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SecFormDApiError(
            "SEC EDGAR returned a response that was not valid XML."
        ) from exc
    total_results = _required_sec_atom_int(root, "totalResults")
    start_index = _required_sec_atom_int(root, "startIndex")
    items_per_page = _required_sec_atom_int(root, "itemsPerPage")
    has_next = start_index + items_per_page < total_results

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in _sec_atom_entries(root):
        filing_type = _sec_form_d_entry_filing_type(entry)
        if filing_type not in {"D", "D/A"}:
            continue
        entry_issuer_name = _sec_form_d_entry_issuer_name(entry)
        is_exact_entry = (
            entry_issuer_name is not None
            and _exact_company_name_match(requested_company_name, entry_issuer_name)
        )
        if entry_issuer_name is not None and not is_exact_entry:
            continue
        filing_href = _sec_form_d_entry_filing_href(entry)
        accession_number = _sec_form_d_entry_accession_number(entry, filing_href)
        if filing_href is None:
            if not is_exact_entry:
                warnings.append(
                    "SEC EDGAR returned a fuzzy Form D result without a filing URL; "
                    "Hail Mary skipped it before preparing evidence."
                )
                continue
            raise SecFormDApiError(
                "SEC EDGAR returned a Form D result without a filing URL."
            )
        if accession_number is None:
            if not is_exact_entry:
                warnings.append(
                    "SEC EDGAR returned a fuzzy Form D result without an accession "
                    "number; Hail Mary skipped it before preparing evidence."
                )
                continue
            raise SecFormDApiError(
                "SEC EDGAR returned a Form D result without an accession number."
            )
        source_url = _sec_form_d_complete_submission_url(
            filing_href,
            accession_number=accession_number,
        )
        try:
            filing_details = fetch_submission(source_url)
        except SecFormDApiError as exc:
            if is_exact_entry:
                raise
            warnings.append(
                "SEC EDGAR returned a fuzzy Form D result whose filing metadata "
                f"could not be parsed; Hail Mary skipped it. Reason: {exc}"
            )
            continue
        results.append(
            {
                **filing_details,
                "filing_type": filing_type,
                "accession_number": accession_number,
                "source_url": source_url,
                "source_api": source_api,
                "filing_date": _sec_form_d_entry_text(entry, "filing-date"),
                "form_name": _sec_form_d_entry_text(entry, "form-name"),
            }
        )
    try:
        return SecFormDFilingsResponse.model_validate(
            {"results": results, "has_next": has_next, "warnings": warnings}
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise SecFormDApiError(
            f"SEC EDGAR returned an unexpected response: {detail}"
        ) from exc


def _required_sec_atom_int(root: ET.Element, local_name: str) -> int:
    value = _first_descendant_text(root, local_name)
    if value is None:
        raise SecFormDApiError(
            "SEC EDGAR returned an unexpected response: missing pagination metadata."
        )
    try:
        return int(value)
    except ValueError as exc:
        raise SecFormDApiError(
            "SEC EDGAR returned an unexpected response: pagination metadata was not a number."
        ) from exc


def _sec_atom_entries(root: ET.Element) -> list[ET.Element]:
    return [child for child in root if _local_name(child.tag).casefold() == "entry"]


def _sec_form_d_entry_text(entry: ET.Element, local_name: str) -> str | None:
    for child in entry.iter():
        if _local_name(child.tag).casefold() == local_name.casefold():
            return _collapse_sec_text(child.text)
    return None


def _sec_form_d_entry_filing_type(entry: ET.Element) -> str | None:
    filing_type = _sec_form_d_entry_text(entry, "filing-type")
    if filing_type:
        return filing_type.upper()
    for child in entry.iter():
        if _local_name(child.tag).casefold() == "category":
            term = child.attrib.get("term")
            if isinstance(term, str) and term.strip():
                return term.strip().upper()
    title = _sec_form_d_entry_text(entry, "title")
    if title:
        match = re.match(r"\s*(D(?:/A)?)\b", title, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _sec_form_d_entry_filing_href(entry: ET.Element) -> str | None:
    filing_href = _sec_form_d_entry_text(entry, "filing-href")
    if filing_href:
        return filing_href
    for child in entry.iter():
        if _local_name(child.tag).casefold() == "link":
            href = child.attrib.get("href")
            if isinstance(href, str) and href.strip():
                return href.strip()
    return None


def _sec_form_d_entry_issuer_name(entry: ET.Element) -> str | None:
    for local_name in ("company-name", "companyName", "issuer-name", "issuerName"):
        value = _sec_form_d_entry_text(entry, local_name)
        if value:
            return value
    title = _sec_form_d_entry_text(entry, "title")
    if title is None:
        return None
    title_without_type = re.sub(r"^\s*D(?:/A)?\s*-\s*", "", title, flags=re.IGNORECASE)
    title_without_cik = re.sub(r"\s*\(\d{1,10}\).*$", "", title_without_type)
    return _collapse_sec_text(title_without_cik)


def _sec_form_d_entry_accession_number(
    entry: ET.Element,
    filing_href: str | None,
) -> str | None:
    accession_number = _sec_form_d_entry_text(entry, "accession-number")
    if accession_number:
        return accession_number
    if filing_href is None:
        return None
    match = re.search(r"(\d{10}-\d{2}-\d{6})", filing_href)
    if match:
        return match.group(1)
    match = re.search(r"/(\d{18})(?:/|\.txt|$)", filing_href)
    if match:
        raw = match.group(1)
        return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
    return None


def _sec_form_d_complete_submission_url(
    filing_href: str,
    *,
    accession_number: str,
) -> str:
    _validate_sec_form_d_public_url(
        filing_href,
        field_name="SEC Form D filing URL",
    )
    parsed = urlparse(filing_href)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() != "www.sec.gov":
        raise SecFormDApiError(
            "SEC Form D filing URLs must use the expected SEC host."
        )
    path = parsed.path
    if path.endswith(".txt"):
        source_url = f"https://{parsed.netloc}{path}"
        _validate_sec_form_d_archive_url(
            source_url,
            field_name="SEC Form D source URL",
        )
        return source_url
    _validate_sec_form_d_archive_url(
        filing_href,
        field_name="SEC Form D filing URL",
    )
    accession_filename = _sec_accession_filename(accession_number)
    if accession_filename is None:
        raise SecFormDApiError(
            "SEC EDGAR returned a Form D result with an unusable accession number."
        )
    accession_digits = accession_filename.replace("-", "")
    folder_path = _sec_accession_folder_path(
        path,
        accession_digits=accession_digits,
    )
    source_url = f"https://{parsed.netloc}{folder_path}/{accession_filename}.txt"
    _validate_sec_form_d_archive_url(source_url, field_name="SEC Form D source URL")
    return source_url


def _sec_accession_folder_path(path: str, *, accession_digits: str) -> str:
    path_parts = path.split("/")
    for index, path_part in enumerate(path_parts):
        if path_part == accession_digits:
            return "/".join(path_parts[: index + 1])
    return path.rsplit("/", 1)[0]


def _sec_accession_filename(accession_number: str) -> str | None:
    stripped = accession_number.strip()
    if re.fullmatch(r"\d{10}-\d{2}-\d{6}", stripped):
        return stripped
    accession_digits = re.sub(r"[^0-9]", "", stripped)
    if re.fullmatch(r"\d{18}", accession_digits):
        return (
            f"{accession_digits[:10]}-{accession_digits[10:12]}-"
            f"{accession_digits[12:]}"
        )
    return None


def _sec_form_d_details_from_submission(
    submission_text: str,
    *,
    source_url: str,
) -> dict[str, str | list[str] | None]:
    parse_errors = 0
    for xml_block in _sec_xml_blocks(submission_text):
        try:
            root = ET.fromstring(xml_block)
        except ET.ParseError:
            parse_errors += 1
            continue
        issuer_name = _first_path_text(
            root,
            ("primaryIssuer", "entityName"),
            ("primaryIssuer", "issuerName"),
        )
        if issuer_name is None:
            continue
        exemptions_parent = _first_path_element(
            root,
            ("offeringData", "federalExemptionsExclusions"),
        )
        return {
            "issuer_name": issuer_name,
            "total_offering_amount": _first_path_text(
                root,
                ("offeringData", "offeringSalesAmounts", "totalOfferingAmount"),
            ),
            "total_amount_sold": _first_path_text(
                root,
                ("offeringData", "offeringSalesAmounts", "totalAmountSold"),
            ),
            "total_remaining": _first_path_text(
                root,
                ("offeringData", "offeringSalesAmounts", "totalRemaining"),
            ),
            "minimum_investment_accepted": _first_path_text(
                root,
                ("offeringData", "minimumInvestmentAccepted"),
            ),
            "total_investors": _first_path_text(
                root,
                ("offeringData", "investors", "totalNumberAlreadyInvested"),
            ),
            "industry_group": _first_path_text(
                root,
                ("offeringData", "industryGroup", "industryGroupType"),
            ),
            "revenue_range": _first_path_text(
                root,
                ("offeringData", "issuerSize", "revenueRange"),
            ),
            "federal_exemptions": _descendant_texts(exemptions_parent, "item"),
        }
    if parse_errors:
        raise SecFormDApiError(
            "SEC Form D filing metadata could not be parsed as XML."
        )
    raise SecFormDApiError(
        f"SEC Form D filing at {source_url} did not include issuer-name metadata."
    )


def _sec_xml_blocks(submission_text: str) -> list[str]:
    blocks = [
        match.group(1).strip()
        for match in re.finditer(
            r"<XML>\s*(.*?)\s*</XML>",
            submission_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match.group(1).strip()
    ]
    if blocks:
        return blocks
    stripped = submission_text.strip()
    if stripped.startswith("<?xml") or stripped.startswith("<edgarSubmission"):
        return [stripped]
    return []


def _first_descendant_text(root: ET.Element, local_name: str) -> str | None:
    for element in root.iter():
        if _local_name(element.tag).casefold() == local_name.casefold():
            collapsed = _collapse_sec_text(element.text)
            if collapsed is not None:
                return collapsed
    return None


def _first_path_text(root: ET.Element, *paths: tuple[str, ...]) -> str | None:
    for path in paths:
        element = _first_path_element(root, path)
        if element is not None:
            collapsed = _collapse_sec_text(element.text)
            if collapsed is not None:
                return collapsed
    return None


def _first_path_element(root: ET.Element, path: tuple[str, ...]) -> ET.Element | None:
    current = root
    for expected_local_name in path:
        next_child = None
        for child in current:
            if _local_name(child.tag).casefold() == expected_local_name.casefold():
                next_child = child
                break
        if next_child is None:
            return None
        current = next_child
    return current


def _descendant_texts(root: ET.Element | None, local_name: str) -> list[str]:
    if root is None:
        return []
    values = []
    for element in root.iter():
        if _local_name(element.tag).casefold() != local_name.casefold():
            continue
        collapsed = _collapse_sec_text(element.text)
        if collapsed is not None:
            values.append(collapsed)
    return values


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _collapse_sec_text(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed or None


def _clean_company_names(company_names: list[str]) -> list[str]:
    cleaned = [re.sub(r"\s+", " ", company_name).strip() for company_name in company_names]
    if any(not company_name for company_name in cleaned):
        raise ResearchCollectionError("Company names cannot be blank.")
    seen: set[str] = set()
    deduped: list[str] = []
    for company_name in cleaned:
        normalized = company_name.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(company_name)
    return deduped


def _rank_public_results(
    results: list[ResearchResultInput],
    *,
    deal: ResearchCollectionDeal,
    collected_at: datetime,
) -> list[ResearchResultInput]:
    return ranked_public_research_results(
        results,
        company_name=deal.company_name,
        ranked_at=collected_at,
    )


def _exact_company_name_match(query_company_name: str, result_company_name: str) -> bool:
    return classify_company_match(query_company_name, result_company_name).import_ready


def _skipped_non_exact_company_names(
    *,
    requested_company_names: list[str],
    source_results: list[PublicSourceSearchResult],
) -> list[str]:
    skipped: list[str] = []
    seen: set[str] = set()
    for result in source_results:
        match = best_company_match(requested_company_names, result.company_name)
        if match.import_ready:
            continue
        normalized = _normalize_company_name(result.company_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        skipped.append(result.company_name)
    return skipped


def _source_result_match_details(
    *,
    requested_company_names: list[str],
    source_results: list[PublicSourceSearchResult],
) -> list[CompanyMatch]:
    details: list[CompanyMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for result in source_results:
        match = best_company_match(requested_company_names, result.company_name)
        key = (match.requested_name, match.candidate_name, match.kind.value)
        if key in seen:
            continue
        seen.add(key)
        details.append(match)
    return details


def _normalize_company_name(value: str) -> str:
    return normalize_company_name(value)


def _normalize_company_slug(value: str) -> str:
    return normalize_company_slug(value)


def _provider_by_id(provider_id: str) -> ResearchProvider:
    providers = {provider.id: provider for provider in builtin_research_providers()}
    provider = providers.get(provider_id)
    if provider is None:
        raise ResearchCollectionError(
            f"Built-in research provider {provider_id} is not configured."
        )
    return provider


def _public_source_adapter_config(provider_id: str) -> PublicSourceAdapterConfig:
    adapter_config = PUBLIC_SOURCE_ADAPTERS.get(provider_id)
    if adapter_config is None:
        raise ResearchCollectionError(
            f"Public source adapter {provider_id} is not configured."
        )
    return adapter_config


def _resolve_input_file(path: Path, *, description: str) -> Path:
    expanded_path = path.expanduser()
    absolute_path = expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path
    if absolute_path.is_symlink():
        raise ResearchCollectionError(f"The {description} file cannot be a symlink.")
    for parent in absolute_path.parents:
        if parent.is_symlink():
            raise ResearchCollectionError(
                f"Hail Mary cannot read {path} because {parent} is a symlinked parent folder."
            )
    resolved_path = absolute_path.resolve(strict=False)
    if not resolved_path.exists():
        raise ResearchCollectionError(f"The {description} file does not exist: {path}")
    if not resolved_path.is_file():
        raise ResearchCollectionError(f"The {description} path is not a file: {path}")
    return resolved_path


def _validate_sec_source_url(url: str) -> None:
    validate_provider_source_url("sec_form_d", url)


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ResearchCollectionError(
            f"Public research results folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ResearchCollectionError(f"Public research results folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise ResearchCollectionError(
            f"Could not create public research results folder at {path}: {exc}"
        ) from exc


def _unique_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"public-research-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _unique_usaspending_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"usaspending-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _unique_sbir_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"sbir-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _unique_sec_form_d_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"sec-form-d-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _unique_github_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"github-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ResearchCollectionError(
            f"Could not write {description} at {path}: output file is a symlink."
        )
    token = secrets.token_hex(8)
    temp_path = path.with_name(f".{path.name}.{token}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(temp_path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        temp_path.chmod(0o600)
        os.replace(temp_path, path)
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise ResearchCollectionError(
            f"Could not write {description} at {path}: the results contain text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ResearchCollectionError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        with suppress(OSError):
            temp_path.unlink()


def _first_validation_detail(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location_parts = tuple(first_error.get("loc", ()))
    location = ".".join(str(part) for part in location_parts)
    message = str(first_error.get("msg", "invalid value"))
    error_type = str(first_error.get("type", ""))
    field_name = str(location_parts[-1]) if location_parts else ""
    plain_message = _plain_public_source_validation_message(
        field_name=field_name,
        message=message,
        error_type=error_type,
    )
    return f"{location}: {plain_message}" if location else plain_message


def _plain_public_source_validation_message(
    *,
    field_name: str,
    message: str,
    error_type: str,
) -> str:
    if field_name == "retrieved_at":
        if error_type != "missing":
            return (
                "retrieved_at is invalid. Use an ISO 8601 timestamp for when the "
                "source was retrieved or viewed, such as 2026-01-01T12:00:00Z."
            )
        return (
            "retrieved_at is required. Enter the time the source was retrieved or "
            "viewed, such as 2026-01-01T12:00:00Z."
        )
    if field_name == "source_url":
        return (
            "source_url is incomplete. Use the exact source URL, or use source_api "
            "for an API source reference."
        )
    if field_name == "source_api":
        return (
            "source_api is incomplete. Use the exact API source reference, or use "
            "source_url for a web page."
        )
    if field_name == "licensing_notes":
        return (
            "licensing_notes is required. Explain why this source or short excerpt "
            "can be saved and used for diligence."
        )
    if field_name == "company_name":
        return "company_name is required and must exactly match the requested company."
    return message


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
