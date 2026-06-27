from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state

from .collection import (
    ResearchCollectionDealSummary,
    ResearchCollectionError,
    ResearchCollectionRunSummary,
    _as_utc,
    _ensure_private_directory,
    _first_validation_detail,
    _write_private_json,
)
from .matching import CompanyMatch, best_company_match, normalize_company_name
from .providers import builtin_research_providers
from .schemas import (
    ResearchProvider,
    ResearchProviderCategory,
    ResearchResultInput,
    ResearchResultsFile,
)
from .source_urls import source_reference_looks_like_url, validate_provider_source_url

PLACEHOLDER_LICENSING_NOTES = {
    "n/a",
    "na",
    "none",
    "unknown",
    "todo",
    "tbd",
    "placeholder",
}


class PaidProviderSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    provider_name: str
    company_name: str

    @field_validator("provider_id", "provider_name", "company_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class PaidProviderFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    title: str
    text: str
    retrieved_at: datetime
    source_url: str | None = None
    source_api: str | None = None
    confidence: str
    licensing_notes: str

    @field_validator("source_url", "source_api", mode="before")
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
        "confidence",
        "licensing_notes",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("company_name", "title", "text", "confidence", "licensing_notes")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def require_source_reference(self) -> Self:
        if self.source_url is None and self.source_api is None:
            raise ValueError("Each paid provider fact needs a source_url or source_api.")
        return self


class PaidProviderSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    results: list[PaidProviderFact]

    @field_validator("provider_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class PaidProviderClient(Protocol):
    def search_company(
        self,
        request: PaidProviderSearchRequest,
    ) -> PaidProviderSearchResponse:
        """Return licensed paid-provider facts for one requested company."""


def collect_paid_research_results(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    provider_ids: list[str] | None = None,
    clients: Mapping[str, PaidProviderClient] | None = None,
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
            "Pass at least one --company value. Hail Mary will only query paid "
            "provider clients for companies you name explicitly."
        )

    selected_providers = _selected_paid_providers(config, provider_ids=provider_ids)
    deal_summaries = [
        ResearchCollectionDealSummary(company_name=company_name)
        for company_name in companies
    ]
    if not selected_providers:
        return ResearchCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            provider_ids=[],
            deals=deal_summaries,
        )

    _ensure_paid_credentials(selected_providers)
    clients_by_provider = dict(clients or {})
    _ensure_paid_clients(selected_providers, clients_by_provider)

    provider_ids_for_summary = [provider.id for provider in selected_providers]
    provider_result_counts: dict[str, int] = {
        provider.id: 0 for provider in selected_providers
    }
    provider_company_result_counts: dict[str, dict[str, int]] = {
        provider.id: {company_name: 0 for company_name in companies}
        for provider in selected_providers
    }
    skipped_non_exact_company_names: list[str] = []
    skipped_seen: set[str] = set()
    match_details: list[CompanyMatch] = []
    match_seen: set[tuple[str, str, str]] = set()
    research_results: list[ResearchResultInput] = []
    deal_counts: dict[str, int] = {company_name: 0 for company_name in companies}

    for company_name in companies:
        for provider in selected_providers:
            client = clients_by_provider[provider.id]
            response = _search_paid_provider(
                provider=provider,
                client=client,
                company_name=company_name,
            )
            for fact in response.results:
                match = best_company_match([company_name], fact.company_name)
                _append_match_detail(match, match_details=match_details, seen=match_seen)
                if not match.import_ready:
                    _append_skipped_company(
                        fact.company_name,
                        skipped_non_exact_company_names,
                        seen=skipped_seen,
                    )
                    continue
                result = _research_result_for_fact(
                    provider=provider,
                    requested_company_name=company_name,
                    fact=fact,
                )
                research_results.append(result)
                provider_result_counts[provider.id] += 1
                provider_company_result_counts[provider.id][company_name] += 1
                deal_counts[company_name] += 1

    deal_summaries = [
        ResearchCollectionDealSummary(
            company_name=company_name,
            result_count=deal_counts[company_name],
        )
        for company_name in companies
    ]
    if not research_results:
        return ResearchCollectionRunSummary(
            output_path=None,
            collected_at=collected_at,
            provider_ids=provider_ids_for_summary,
            deals=deal_summaries,
            provider_result_counts=provider_result_counts,
            provider_company_result_counts=provider_company_result_counts,
            skipped_non_exact_company_names=skipped_non_exact_company_names,
            match_details=match_details,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": research_results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"Prepared paid research results did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_paid_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="paid research results",
    )
    return ResearchCollectionRunSummary(
        output_path=output_path,
        collected_at=collected_at,
        provider_ids=provider_ids_for_summary,
        deals=deal_summaries,
        provider_result_counts=provider_result_counts,
        provider_company_result_counts=provider_company_result_counts,
        skipped_non_exact_company_names=skipped_non_exact_company_names,
        match_details=match_details,
    )


def _selected_paid_providers(
    config: AppConfig,
    *,
    provider_ids: list[str] | None,
) -> list[ResearchProvider]:
    providers_by_id = _paid_providers_by_id()
    requested_provider_ids = (
        _clean_provider_ids(provider_ids)
        if provider_ids is not None
        else list(config.enabled_paid_providers)
    )
    unknown_provider_ids = [
        provider_id
        for provider_id in requested_provider_ids
        if provider_id not in providers_by_id
    ]
    if unknown_provider_ids:
        raise ResearchCollectionError(
            "Unknown paid provider "
            f"{', '.join(sorted(unknown_provider_ids))}. Available paid providers: "
            f"{', '.join(sorted(providers_by_id))}."
        )

    if provider_ids is not None:
        enabled_provider_ids = set(config.enabled_paid_providers)
        disabled_provider_ids = [
            provider_id
            for provider_id in requested_provider_ids
            if provider_id not in enabled_provider_ids
        ]
        if disabled_provider_ids:
            raise ResearchCollectionError(
                "Paid provider "
                f"{', '.join(sorted(disabled_provider_ids))} is disabled. Add it to "
                "HAILMARY_ENABLED_PAID_PROVIDERS before using it."
            )

    return [providers_by_id[provider_id] for provider_id in requested_provider_ids]


def _paid_providers_by_id() -> dict[str, ResearchProvider]:
    return {
        provider.id: provider
        for provider in builtin_research_providers(include_paid=True)
        if provider.category == ResearchProviderCategory.PAID_OPTIONAL
    }


def _clean_provider_ids(provider_ids: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for provider_id in provider_ids:
        normalized = provider_id.strip().casefold().replace("-", "_")
        if not normalized:
            raise ResearchCollectionError("Paid provider IDs cannot be blank.")
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _ensure_paid_credentials(providers: list[ResearchProvider]) -> None:
    for provider in providers:
        credential_env_var = provider.credential_env_var
        if credential_env_var is None:
            raise ResearchCollectionError(
                f"{provider.name} does not define a credential environment variable."
            )
        credential_value = os.getenv(credential_env_var)
        if credential_value is None or not credential_value.strip():
            raise ResearchCollectionError(
                f"{provider.name} is enabled, but {credential_env_var} is missing. "
                f"Set {credential_env_var} in the environment for a licensed account; "
                "do not commit credentials."
            )


def _ensure_paid_clients(
    providers: list[ResearchProvider],
    clients: dict[str, PaidProviderClient],
) -> None:
    missing_client_providers = [
        provider.name for provider in providers if provider.id not in clients
    ]
    if not missing_client_providers:
        return
    raise ResearchCollectionError(
        f"{', '.join(sorted(missing_client_providers))} is enabled and credentialed, "
        "but no paid provider client was provided. This scaffold does not make paid "
        "API calls by itself."
    )


def _search_paid_provider(
    *,
    provider: ResearchProvider,
    client: PaidProviderClient,
    company_name: str,
) -> PaidProviderSearchResponse:
    request = PaidProviderSearchRequest(
        provider_id=provider.id,
        provider_name=provider.name,
        company_name=company_name,
    )
    try:
        response = client.search_company(request)
    except Exception as exc:
        raise ResearchCollectionError(
            f"{provider.name} paid provider client failed for {company_name}: {exc}"
        ) from exc
    if response.provider_id != provider.id:
        raise ResearchCollectionError(
            f"{provider.name} paid provider client returned provider_id "
            f"{response.provider_id!r}, but {provider.id!r} was requested."
        )
    return response


def _research_result_for_fact(
    *,
    provider: ResearchProvider,
    requested_company_name: str,
    fact: PaidProviderFact,
) -> ResearchResultInput:
    _validate_paid_fact_source(provider=provider, fact=fact)
    _validate_paid_licensing_notes(provider=provider, fact=fact)
    try:
        return ResearchResultInput(
            company_name=requested_company_name,
            provider_id=provider.id,
            provider_name=provider.name,
            title=fact.title,
            text=fact.text,
            retrieved_at=_as_utc(fact.retrieved_at),
            source_url=fact.source_url,
            source_api=fact.source_api,
            confidence=fact.confidence,
            licensing_notes=fact.licensing_notes,
            source_kind=provider.source_kind,
        )
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchCollectionError(
            f"{provider.name} result for {requested_company_name} is incomplete: {detail}"
        ) from exc


def _validate_paid_fact_source(
    *,
    provider: ResearchProvider,
    fact: PaidProviderFact,
) -> None:
    try:
        if fact.source_url is not None:
            validate_provider_source_url(provider.id, fact.source_url)
        if fact.source_api is not None and source_reference_looks_like_url(fact.source_api):
            validate_provider_source_url(
                provider.id,
                fact.source_api,
                field_name="source_api",
            )
    except ValueError as exc:
        raise ResearchCollectionError(
            f"{provider.name} returned an unsafe source reference: {exc}."
        ) from exc


def _validate_paid_licensing_notes(
    *,
    provider: ResearchProvider,
    fact: PaidProviderFact,
) -> None:
    notes = fact.licensing_notes.strip()
    compact = re.sub(r"[\W_]+", "", notes).casefold()
    if notes.casefold() in PLACEHOLDER_LICENSING_NOTES or compact in {
        "na",
        "none",
        "unknown",
        "todo",
        "tbd",
        "placeholder",
    }:
        raise ResearchCollectionError(
            f"{provider.name} returned licensing_notes that are only a placeholder. "
            "Add a plain-English note explaining why the short fact can be saved."
        )
    if re.fullmatch(r"https?://\S+", notes):
        raise ResearchCollectionError(
            f"{provider.name} returned licensing_notes that only contain a URL. "
            "Add a plain-English note explaining source permissions."
        )


def _append_match_detail(
    match: CompanyMatch,
    *,
    match_details: list[CompanyMatch],
    seen: set[tuple[str, str, str]],
) -> None:
    key = (match.requested_name, match.candidate_name, match.kind.value)
    if key in seen:
        return
    seen.add(key)
    match_details.append(match)


def _append_skipped_company(
    company_name: str,
    skipped: list[str],
    *,
    seen: set[str],
) -> None:
    normalized = normalize_company_name(company_name)
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    skipped.append(company_name)


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


def _unique_paid_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"paid-research-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate
