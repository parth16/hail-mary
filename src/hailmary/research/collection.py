from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state

from .providers import builtin_research_providers
from .schemas import ResearchProvider, ResearchResultInput, ResearchResultsFile


class ResearchCollectionError(RuntimeError):
    """Public research results could not be prepared safely."""


PUBLIC_SOURCE_HOSTS: dict[str, tuple[str, str]] = {
    "sec_form_d": ("sec.gov", "an SEC website host such as www.sec.gov or data.sec.gov"),
    "sam_gov": ("sam.gov", "a SAM.gov website host such as sam.gov or www.sam.gov"),
    "usaspending": (
        "usaspending.gov",
        "a USAspending website host such as www.usaspending.gov",
    ),
    "sbir": ("sbir.gov", "an SBIR website host such as www.sbir.gov"),
    "uspto": ("uspto.gov", "a USPTO website host such as tmsearch.uspto.gov"),
    "github": ("github.com", "the GitHub website host github.com"),
}


class PublicSourceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    title: str
    text: str
    source_url: str
    retrieved_at: datetime
    filed_at: datetime | None = None
    accession_number: str | None = None

    @field_validator(
        "company_name",
        "title",
        "text",
        "source_url",
        "accession_number",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("company_name", "title", "text", "source_url", "accession_number")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("company_name", "title", "text", "source_url")
    @classmethod
    def require_nonblank_text(cls, value: str | None) -> str:
        if value is None or not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_source_url(self) -> Self:
        _validate_http_url(self.source_url, field_name="source_url")
        return self


class SecFormDSearchResult(PublicSourceSearchResult):
    @model_validator(mode="after")
    def validate_sec_source_url(self) -> Self:
        _validate_provider_source_url("sec_form_d", self.source_url)
        return self


class PublicSourceSearchResultsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[PublicSourceSearchResult]


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


@dataclass(frozen=True)
class ResearchCollectionDeal:
    company_name: str


class PublicSourceSearchClient(Protocol):
    def search(self, company_name: str) -> Iterable[PublicSourceSearchResult]:
        """Return locally available public-source results for one company."""


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
        results: list[ResearchResultInput] = []
        for search_result in self.client.search(deal.company_name):
            if not _exact_company_name_match(deal.company_name, search_result.company_name):
                continue
            _validate_provider_source_url(provider.id, search_result.source_url)
            try:
                result = ResearchResultInput(
                    company_name=deal.company_name,
                    provider_id=provider.id,
                    provider_name=provider.name,
                    title=search_result.title,
                    text=search_result.text,
                    retrieved_at=_as_utc(search_result.retrieved_at),
                    source_url=search_result.source_url,
                    confidence=(
                        "high: exact company name match from a local "
                        f"{provider.name} source file"
                    ),
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
        (sec_form_d_results_path, "sec_form_d", "SEC Form D results"),
        (sam_gov_results_path, "sam_gov", "SAM.gov results"),
        (usaspending_results_path, "usaspending", "USAspending results"),
        (sbir_results_path, "sbir", "SBIR/STTR results"),
        (uspto_results_path, "uspto", "USPTO results"),
        (github_results_path, "github", "GitHub results"),
    ]
    return [
        PublicSourceFile(provider_id=provider_id, path=path, description=description)
        for path, provider_id, description in candidates
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
        try:
            _validate_provider_source_url(provider_id, result.source_url)
        except ValueError as exc:
            raise ResearchCollectionError(
                f"{provider.name} result {index} has an invalid source_url: {exc}"
            ) from exc
    return results_file


def _load_sec_form_d_search_results(path: Path) -> SecFormDSearchResultsFile:
    return _load_public_source_search_results(
        path,
        provider_id="sec_form_d",
        description="SEC Form D results",
    )


def _clean_company_names(company_names: list[str]) -> list[str]:
    cleaned = [company_name.strip() for company_name in company_names]
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


def _validate_http_url(url: str, *, field_name: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{field_name} must start with http:// or https://")
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid URL") from exc
    if not parsed.netloc or host is None:
        raise ValueError(f"{field_name} must include a website host")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} cannot include a username or password")
    if any(character.isspace() for character in url):
        raise ValueError(f"{field_name} cannot contain spaces")


def _validate_sec_source_url(url: str) -> None:
    _validate_provider_source_url("sec_form_d", url)


def _validate_provider_source_url(provider_id: str, url: str) -> None:
    _validate_http_url(url, field_name="source_url")
    host_rule = PUBLIC_SOURCE_HOSTS.get(provider_id)
    if host_rule is None:
        return
    allowed_suffix, description = host_rule
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    allowed_host = allowed_suffix.casefold()
    if host == allowed_host or host.endswith(f".{allowed_host}"):
        return
    raise ValueError(
        f"source_url must use {description}"
    )


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
