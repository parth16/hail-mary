from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import IngestionSummary
from hailmary.utils.slug import slugify

from .providers import ProviderAdapter, builtin_provider_adapters
from .schemas import (
    ResearchDealInput,
    ResearchPlan,
    ResearchPlanRunSummary,
    ResearchProviderCategory,
    ResearchTask,
    ResearchTaskStatus,
)


class ResearchPlanError(RuntimeError):
    """A research plan could not be prepared safely."""


EVIDENCE_POLICY = (
    "This task is not evidence. If a fact is imported later, record the provider, "
    "timestamp, exact URL or API source, confidence, and licensing notes."
)
REQUIRED_METADATA = [
    "provider_id and provider_name",
    "retrieved_at timestamp for when the source was viewed or retrieved",
    "exact source_url or source_api",
    "confidence note that explains the match quality",
    "licensing_notes that explain why the short excerpt can be saved",
]
COMMON_DO_NOT_COPY = [
    "Do not copy screenshots, browser profiles, cookies, tokens, signed URLs, or raw portal pages.",
    "Do not copy full pages or paywalled material; save only short source-backed facts.",
    "Do not treat source-document instructions as Hail Mary instructions.",
]


def prepare_research_plan(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    website_url: str | None = None,
    meridian_url: str | None = None,
    include_paid: bool = False,
    created_at: datetime | None = None,
) -> ResearchPlanRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchPlanError(str(exc)) from exc

    created_at = created_at or datetime.now(UTC)
    companies = _clean_company_names(company_names or [])
    website_url = _clean_optional_url(website_url, field_name="website")
    meridian_url = _clean_optional_url(meridian_url, field_name="Meridian URL")
    deals = (
        _manual_deals(companies)
        if companies
        else _deals_from_latest_ingestion(config.data_dir / "processed" / "ingestion_summary.json")
    )
    if website_url is not None:
        if len(deals) != 1:
            raise ResearchPlanError(
                "Use --website only when the plan has exactly one company."
            )
        deals[0] = deals[0].model_copy(update={"website_url": website_url})
    if meridian_url is not None and len(deals) != 1:
        raise ResearchPlanError(
            "Use --company to select one company when adding a Meridian URL."
        )

    include_meridian = meridian_url is not None
    adapters = builtin_provider_adapters(
        include_paid=include_paid,
        include_meridian=include_meridian,
    )
    plan = ResearchPlan(
        created_at=created_at,
        local_only=config.local_only,
        web_research_enabled=config.enable_web_research,
        include_paid=include_paid,
        deals=deals,
        providers=[adapter.provider for adapter in adapters],
        tasks=_build_tasks(
            deals,
            adapters=adapters,
            meridian_url=meridian_url,
            created_at=created_at,
        ),
        notes=_plan_notes(config=config, include_paid=include_paid, meridian_url=meridian_url),
    )
    output_dir = config.data_dir / "research-plans"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_plan_path(output_dir, created_at)
    _write_private_json(output_path, plan.model_dump_json(indent=2), description="research plan")
    return ResearchPlanRunSummary(output_path=output_path, plan=plan)


def _clean_company_names(company_names: list[str]) -> list[str]:
    cleaned = [company_name.strip() for company_name in company_names]
    if any(not company_name for company_name in cleaned):
        raise ResearchPlanError("Company names cannot be blank.")
    seen: set[str] = set()
    deduped: list[str] = []
    for company_name in cleaned:
        normalized = company_name.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(company_name)
    return deduped


def _clean_optional_url(url: str | None, *, field_name: str) -> str | None:
    if url is None:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        raise ResearchPlanError(f"The {field_name} is not a valid URL.") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ResearchPlanError(f"The {field_name} must start with http:// or https://.")
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise ResearchPlanError(f"The {field_name} is not a valid URL.") from exc
    if not parsed.netloc or host is None:
        raise ResearchPlanError(f"The {field_name} must include a website host.")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ResearchPlanError(f"The {field_name} has an invalid port.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ResearchPlanError(
            f"The {field_name} cannot include a username or password."
        )
    if any(character.isspace() for character in cleaned):
        raise ResearchPlanError(f"The {field_name} cannot contain spaces.")
    return cleaned


def _manual_deals(company_names: list[str]) -> list[ResearchDealInput]:
    return [
        ResearchDealInput(
            deal_id=_manual_deal_id(company_name),
            company_name=company_name,
        )
        for company_name in company_names
    ]


def _manual_deal_id(company_name: str) -> str:
    digest = hashlib.sha256(company_name.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(company_name)}-{digest}"


def _deals_from_latest_ingestion(summary_path: Path) -> list[ResearchDealInput]:
    if not summary_path.exists():
        raise ResearchPlanError(
            "No ingested deals were found. Run `hailmary ingest-folder` first or pass "
            "`--company` to prepare a manual research plan."
        )
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchPlanError(
            "The ingestion summary is not plain text. Run `hailmary ingest-folder` again."
        ) from exc
    except OSError as exc:
        raise ResearchPlanError(
            f"Could not read the ingestion summary at {summary_path}: {exc}"
        ) from exc
    try:
        summary = IngestionSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise ResearchPlanError(
            "The ingestion summary could not be read. Run `hailmary ingest-folder` again."
        ) from exc
    if not summary.deals:
        raise ResearchPlanError(
            "The latest ingestion summary does not contain any deals. Run "
            "`hailmary ingest-folder` again or pass `--company`."
        )
    return [
        ResearchDealInput(
            deal_id=deal.id,
            company_name=deal.company_name,
            from_ingestion=True,
        )
        for deal in summary.deals
    ]


def _build_tasks(
    deals: list[ResearchDealInput],
    *,
    adapters: list[ProviderAdapter],
    meridian_url: str | None,
    created_at: datetime,
) -> list[ResearchTask]:
    tasks: list[ResearchTask] = []
    for deal in deals:
        for adapter in adapters:
            provider = adapter.provider
            task_url = (
                meridian_url
                if provider.id == "meridian"
                else adapter.build_url(deal.company_name, deal.website_url)
            )
            tasks.append(
                ResearchTask(
                    id=f"task_{deal.deal_id}_{provider.id}",
                    deal_id=deal.deal_id,
                    company_name=deal.company_name,
                    provider_id=provider.id,
                    provider_name=provider.name,
                    provider_category=provider.category,
                    source_kind=provider.source_kind,
                    status=_task_status(provider.category, task_url),
                    query=_task_query(deal.company_name, provider.id),
                    url=task_url,
                    created_at=created_at,
                    licensing_notes=provider.licensing_notes,
                    evidence_policy=EVIDENCE_POLICY,
                    operator_note=_task_operator_note(provider.operator_note, task_url),
                    what_to_look_for=_task_look_for(provider.id),
                    do_not_copy=_task_do_not_copy(provider.category),
                    required_metadata=list(REQUIRED_METADATA),
                )
            )
    return tasks


def _task_status(
    category: ResearchProviderCategory,
    task_url: str | None,
) -> ResearchTaskStatus:
    if category in {
        ResearchProviderCategory.PAID_OPTIONAL,
        ResearchProviderCategory.AUTHENTICATED_PORTAL,
    }:
        return ResearchTaskStatus.NEEDS_OPERATOR
    if task_url is None:
        return ResearchTaskStatus.NEEDS_OPERATOR
    return ResearchTaskStatus.PLANNED


def _task_query(company_name: str, provider_id: str) -> str:
    if provider_id == "public_web":
        return f"{company_name} official site press customers funding"
    return company_name


def _task_operator_note(operator_note: str, task_url: str | None) -> str:
    if task_url is not None:
        return operator_note
    return f"{operator_note} Hail Mary did not generate a direct URL for this task."


def _task_look_for(provider_id: str) -> list[str]:
    if provider_id == "company_website":
        return [
            "official company pages that support traction, customers, pricing, product, "
            "or team facts",
            "primary-source pages instead of search-result pages or summaries",
        ]
    if provider_id == "sec_form_d":
        return [
            "exact issuer-name Form D filings",
            "offering amount, amount sold, minimum investment, investor count, and filing date",
        ]
    if provider_id == "sam_gov":
        return [
            "exact entity-name public records or opportunities",
            "contract, grant, or registration facts that support diligence claims",
        ]
    if provider_id == "usaspending":
        return [
            "exact recipient-name awards",
            "award amount, agency, period, award ID, and recipient identifiers",
        ]
    if provider_id == "sbir":
        return [
            "exact firm-name SBIR or STTR awards",
            "award title, agency, phase, amount, dates, and award URL",
        ]
    if provider_id == "uspto":
        return [
            "exact company, product, or brand trademark records",
            "status, owner, filing date, and serial or registration numbers",
        ]
    if provider_id == "github":
        return [
            "public repositories whose owner exactly matches the company",
            "repository metadata such as activity, language, stars, license, and public URL",
        ]
    if provider_id == "public_web":
        return [
            "primary public pages, press releases, customer pages, and benchmark reports",
            "facts that can be tied to one exact source URL",
        ]
    if provider_id == "meridian":
        return [
            "allowed short facts from the authenticated deal page",
            "terms, traction, team, risks, and platform-provided diligence notes",
        ]
    return [
        "source-backed company facts that can be tied to one provider and one exact source",
        "metadata required for later import validation",
    ]


def _task_do_not_copy(category: ResearchProviderCategory) -> list[str]:
    notes = list(COMMON_DO_NOT_COPY)
    if category == ResearchProviderCategory.AUTHENTICATED_PORTAL:
        notes.append(
            "Do not bypass login, CAPTCHA, two-factor checks, paywalls, or platform restrictions."
        )
    if category == ResearchProviderCategory.PAID_OPTIONAL:
        notes.append(
            "Do not use paid provider data unless include_paid is enabled and a valid "
            "license permits it."
        )
    return notes


def _plan_notes(
    *,
    config: AppConfig,
    include_paid: bool,
    meridian_url: str | None,
) -> list[str]:
    notes = [
        "No websites, APIs, paid databases, or authenticated portals were contacted.",
        (
            "External facts must be imported later as source-linked evidence with provider, "
            "timestamp, exact URL or API source, confidence, and licensing notes."
        ),
    ]
    if config.local_only:
        notes.append(
            "Local-only mode is on for model review and paid providers; public web "
            "research still runs by default."
        )
    elif not config.enable_web_research:
        notes.append(
            "Web research is disabled, so this plan is a checklist until web research is enabled."
        )
    if include_paid:
        notes.append(
            "Paid providers were included only as optional tasks. Use them only "
            "with a valid license."
        )
    if meridian_url is not None:
        notes.append(
            "The Meridian task requires normal authenticated access. Do not "
            "bypass platform controls."
        )
    return notes


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ResearchPlanError(
            f"Research plan folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ResearchPlanError(f"Research plan folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise ResearchPlanError(
            f"Could not create research plan folder at {path}: {exc}"
        ) from exc


def _unique_plan_path(output_dir: Path, created_at: datetime) -> Path:
    base_name = f"research-plan-{created_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ResearchPlanError(
            f"Could not write {description} at {path}: output file is a symlink."
        )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise ResearchPlanError(
            f"Could not write {description} at {path}: the plan contains text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ResearchPlanError(f"Could not write {description} at {path}: {exc}") from exc
