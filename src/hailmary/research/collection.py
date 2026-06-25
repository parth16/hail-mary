from __future__ import annotations

import json
import os
import re
import secrets
import urllib.error
import urllib.request
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self
from urllib.parse import quote, urlencode, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state

from .providers import builtin_research_providers
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


USASPENDING_AWARDS_ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
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


SecFormDSearchResultsFile = PublicSourceSearchResultsFile


class ResearchCollectionDealSummary(BaseModel):
    company_name: str
    result_count: int = 0


class ResearchCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    provider_ids: list[str]
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def result_count(self) -> int:
        return sum(deal.result_count for deal in self.deals)


class UsaspendingCollectionRunSummary(BaseModel):
    output_path: Path | None = None
    collected_at: datetime
    dry_run: bool = False
    endpoint: str = USASPENDING_AWARDS_ENDPOINT
    deals: list[ResearchCollectionDealSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

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
            if not _exact_company_name_match(deal.company_name, search_result.company_name):
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

    adapters = [
        PublicSourceFileAdapter(
            provider_id=source_file.provider_id,
            client=LocalPublicSourceSearchClient(
                tuple(
                    _load_public_source_search_results(
                        source_file.path,
                        provider_id=source_file.provider_id,
                        description=source_file.description,
                    ).results
                )
            ),
        )
        for source_file in source_files
    ]
    provider_ids = [adapter.provider_id for adapter in adapters]
    deals = [ResearchCollectionDeal(company_name=company_name) for company_name in companies]
    results: list[ResearchResultInput] = []
    deal_summaries: list[ResearchCollectionDealSummary] = []
    for deal in deals:
        deal_results = [
            result
            for adapter in adapters
            for result in adapter.collect(deal, collected_at=collected_at)
        ]
        results.extend(deal_results)
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
    _ensure_live_public_research_enabled(config)
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
                    if not _exact_company_name_match(deal.company_name, recipient_name):
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
    _ensure_live_public_research_enabled(config)
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
                    if not _exact_company_name_match(deal.company_name, firm):
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


def _ensure_live_public_research_enabled(config: AppConfig) -> None:
    if config.local_only:
        raise ResearchCollectionError(
            "Local-only mode is on. Set HAILMARY_LOCAL_ONLY=false before collecting "
            "USAspending results."
        )
    if not config.enable_web_research:
        raise ResearchCollectionError(
            "Web research is disabled. Set HAILMARY_ENABLE_WEB_RESEARCH=true before "
            "collecting USAspending results."
        )


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
    if host != "api.www.sbir.gov" or path != "/public/api/awards":
        raise SbirApiError(
            "SBIR/STTR redirected the request away from the expected public API endpoint."
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


def _research_result_from_usaspending_award(
    deal: ResearchCollectionDeal,
    award: UsaspendingAwardRecord,
    *,
    provider: ResearchProvider,
    collected_at: datetime,
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
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"SBIR/STTR result for {deal.company_name} is incomplete: {detail}"
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


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _clean_company_names(company_names: list[str]) -> list[str]:
    cleaned = [re.sub(r"\s+", " ", company_name).strip() for company_name in company_names]
    if any(not company_name for company_name in cleaned):
        raise ResearchCollectionError("Company names cannot be blank.")
    seen: set[str] = set()
    deduped: list[str] = []
    for company_name in cleaned:
        normalized = _normalize_company_name(company_name)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(company_name)
    return deduped


def _exact_company_name_match(query_company_name: str, result_company_name: str) -> bool:
    query = _normalize_company_name(query_company_name)
    result = _normalize_company_name(result_company_name)
    if not query or not result:
        return False
    return query == result


def _normalize_company_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


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
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
