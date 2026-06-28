from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from hailmary.config import AppConfig

from .collection import (
    SEC_FORM_D_USER_AGENT_ENV_VAR,
    GitHubRepositorySearchClient,
    SbirAwardsClient,
    SecFormDFilingsClient,
    UsaspendingAwardsClient,
    collect_github_repositories,
    collect_sbir_awards,
    collect_sec_form_d_filings,
    collect_usaspending_awards,
    prepare_public_research_results,
)
from .importer import (
    ResearchImportError,
    import_research_results,
    preview_meridian_results_file,
)
from .matching import CompanyMatch
from .meridian import clean_meridian_url, prepare_meridian_workflow
from .paid import PaidProviderClient, collect_paid_research_results
from .planner import prepare_research_plan
from .providers import builtin_research_providers
from .schemas import (
    MeridianImportPreview,
    MeridianUnresolvedField,
    ResearchImportDealSummary,
    ResearchImportRunSummary,
    ResearchPlan,
    ResearchProviderCategory,
    ResearchResultsFile,
    ResearchTask,
    ResearchTaskStatus,
)
from .templates import prepare_research_results_template
from .web import WebResearchClient, WebResearchTaskSummary, collect_web_research


class ResearchWorkflowError(RuntimeError):
    """The higher-level research workflow could not be prepared safely."""


IssueSeverity = Literal["warning", "error"]
CollectionKind = Literal["local_public", "live_public", "web", "paid_optional"]

LIVE_PROVIDER_IDS = {"company_website", "sec_form_d", "usaspending", "sbir", "github"}
MANUAL_OR_LOCAL_PROVIDER_IDS = {"sam_gov", "uspto", "public_web"}
NO_PAID_OUTPUTS_PRIVACY_NOTE = (
    "No screenshots, cookies, browser profiles, raw portal HTML, hidden authenticated data, "
    "signed URLs, or paid-source outputs are saved by this workflow."
)
PAID_OUTPUTS_PRIVACY_NOTE = (
    "No screenshots, cookies, browser profiles, raw portal HTML, hidden authenticated data, "
    "or signed URLs are saved by this workflow. When explicit paid clients are supplied, "
    "paid-provider facts may be saved only as generated research-result JSON for import preview."
)
PRIVACY_NOTES = [
    NO_PAID_OUTPUTS_PRIVACY_NOTE,
    "Every imported external fact still needs provider, retrieval time, exact URL or API source, "
    "confidence, and licensing notes.",
]
MAX_IDENTITY_WARNING_DETAILS = 5


class ResearchWorkflowIssue(BaseModel):
    severity: IssueSeverity
    source: str
    message: str


class ResearchWorkflowArtifact(BaseModel):
    kind: str
    path: Path


class ResearchProviderRunStatus(StrEnum):
    NOT_RUN = "not_run"
    PLANNED = "planned"
    MANUAL_NEEDED = "manual_needed"
    IMPORTED = "imported"
    NO_EXACT_RESULTS = "no_exact_results"
    INCOMPLETE_SEARCH = "incomplete_search"
    FAILED = "failed"


class ResearchWorkflowSourceSummary(BaseModel):
    provider_id: str
    provider_name: str
    planned_count: int = 0
    manual_count: int = 0
    live_collectable_count: int = 0


class ResearchWorkflowCollectionSummary(BaseModel):
    kind: CollectionKind
    source_id: str
    source_name: str
    status: ResearchProviderRunStatus = ResearchProviderRunStatus.PLANNED
    output_path: Path | None = None
    result_count: int = 0
    deal_count: int = 0
    no_result_companies: list[str] = Field(default_factory=list)
    skipped_non_exact_company_names: list[str] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)
    provider_ids: list[str] = Field(default_factory=list)
    provider_result_counts: dict[str, int] = Field(default_factory=dict)
    provider_company_result_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    provider_statuses: list[ResearchProviderStatusSummary] = Field(default_factory=list)
    incomplete_search: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchProviderStatusSummary(BaseModel):
    provider_id: str
    provider_name: str
    status: ResearchProviderRunStatus
    planned_count: int = 0
    collected_count: int = 0
    imported_count: int = 0
    warning_count: int = 0
    no_exact_result_companies: list[str] = Field(default_factory=list)
    incomplete_search: bool = False
    failure: str | None = None


class ResearchWorkflowSummary(BaseModel):
    planned_task_count: int = 0
    imported_record_count: int = 0
    stale_record_count: int = 0
    failed_provider_count: int = 0
    incomplete_search_count: int = 0
    no_exact_result_provider_count: int = 0
    manual_needed_provider_count: int = 0
    not_run_provider_count: int = 0
    warning_count: int = 0
    provider_statuses: list[ResearchProviderStatusSummary] = Field(default_factory=list)


class ResearchWorkflowImportPreview(BaseModel):
    input_path: Path
    imported_count: int = 0
    skipped_duplicate_count: int = 0
    skipped_blank_template_row_count: int = 0
    stale_count: int = 0
    meridian_preview: MeridianImportPreview | None = None
    deals: list[ResearchImportDealSummary] = Field(default_factory=list)
    error: str | None = None


class ResearchWorkflowRunSummary(BaseModel):
    created_at: datetime
    plan: ResearchPlan
    plan_path: Path
    result_template_path: Path
    manual_task_queue_path: Path | None = None
    meridian_workflow_path: Path | None = None
    meridian_result_template_path: Path | None = None
    source_summaries: list[ResearchWorkflowSourceSummary] = Field(default_factory=list)
    collections: list[ResearchWorkflowCollectionSummary] = Field(default_factory=list)
    import_previews: list[ResearchWorkflowImportPreview] = Field(default_factory=list)
    issues: list[ResearchWorkflowIssue] = Field(default_factory=list)
    privacy_notes: list[str] = Field(default_factory=lambda: list(PRIVACY_NOTES))
    live_collection_enabled: bool = False

    @property
    def summary(self) -> ResearchWorkflowSummary:
        return _research_workflow_summary(self)

    @property
    def planned_source_count(self) -> int:
        return len(self.source_summaries)

    @property
    def manual_task_count(self) -> int:
        return sum(source.manual_count for source in self.source_summaries)

    @property
    def unresolved_manual_task_count(self) -> int:
        return len(_unresolved_manual_tasks(self))

    @property
    def live_collectable_task_count(self) -> int:
        return sum(source.live_collectable_count for source in self.source_summaries)

    @property
    def ready_to_import_count(self) -> int:
        return sum(preview.imported_count for preview in self.import_previews)

    @property
    def blocking_issue_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def no_prepared_result_companies(self) -> list[str]:
        prepared_counts = {deal.company_name: 0 for deal in self.plan.deals}
        for collection in self.collections:
            if collection.error is not None:
                continue
            for company_name in prepared_counts:
                if company_name not in collection.no_result_companies:
                    prepared_counts[company_name] += collection.result_count
        for preview in self.import_previews:
            for deal in preview.deals:
                prepared_counts[deal.company_name] = (
                    prepared_counts.get(deal.company_name, 0) + deal.imported_count
                )
        return [
            company_name
            for company_name, result_count in prepared_counts.items()
            if result_count == 0
        ]

    @property
    def meridian_unresolved_fields(self) -> list[MeridianUnresolvedField]:
        unresolved_by_id: dict[str, MeridianUnresolvedField] = {}
        resolved_field_ids: set[str] = set()
        for preview in self.import_previews:
            if preview.meridian_preview is None:
                continue
            if preview.error is None:
                if preview.meridian_preview.source_url is not None:
                    resolved_field_ids.add("deal_url")
                for row in preview.meridian_preview.rows:
                    if row.status == "import_ready" and row.field_id is not None:
                        resolved_field_ids.add(row.field_id)
            for field in preview.meridian_preview.unresolved_required_fields:
                unresolved_by_id.setdefault(field.field_id, field)
        return [
            field
            for field in unresolved_by_id.values()
            if field.field_id not in resolved_field_ids
        ]

    @property
    def artifacts(self) -> list[ResearchWorkflowArtifact]:
        artifacts = [
            ResearchWorkflowArtifact(kind="research_plan", path=self.plan_path),
            ResearchWorkflowArtifact(
                kind="research_results_template",
                path=self.result_template_path,
            ),
        ]
        if self.manual_task_queue_path is not None:
            artifacts.append(
                ResearchWorkflowArtifact(
                    kind="manual_task_queue",
                    path=self.manual_task_queue_path,
                )
            )
        if self.meridian_workflow_path is not None:
            artifacts.append(
                ResearchWorkflowArtifact(
                    kind="meridian_workflow",
                    path=self.meridian_workflow_path,
                )
            )
        if self.meridian_result_template_path is not None:
            artifacts.append(
                ResearchWorkflowArtifact(
                    kind="meridian_results_template",
                    path=self.meridian_result_template_path,
                )
            )
        for collection in self.collections:
            if collection.output_path is not None:
                artifacts.append(
                    ResearchWorkflowArtifact(
                        kind=f"{collection.source_id}_results",
                        path=collection.output_path,
                    )
                )
        return artifacts


def run_research_workflow(
    *,
    config: AppConfig,
    company_names: list[str] | None = None,
    website_url: str | None = None,
    meridian_url: str | None = None,
    include_paid: bool = False,
    sec_form_d_results_path: Path | None = None,
    sam_gov_results_path: Path | None = None,
    usaspending_results_path: Path | None = None,
    sbir_results_path: Path | None = None,
    uspto_results_path: Path | None = None,
    github_results_path: Path | None = None,
    results_files: list[Path] | None = None,
    created_at: datetime | None = None,
    web_client: WebResearchClient | None = None,
    usaspending_client: UsaspendingAwardsClient | None = None,
    sbir_client: SbirAwardsClient | None = None,
    sec_form_d_client: SecFormDFilingsClient | None = None,
    github_client: GitHubRepositorySearchClient | None = None,
    paid_clients: Mapping[str, PaidProviderClient] | None = None,
) -> ResearchWorkflowRunSummary:
    created_at = _as_utc(created_at or datetime.now(UTC))
    try:
        cleaned_website_url = _clean_workflow_website_url(website_url)
        cleaned_meridian_url = (
            clean_meridian_url(meridian_url) if meridian_url is not None else None
        )
        plan_result = prepare_research_plan(
            config=config,
            company_names=company_names or [],
            website_url=cleaned_website_url,
            meridian_url=cleaned_meridian_url,
            include_paid=include_paid,
            created_at=created_at,
        )
        template_result = prepare_research_results_template(
            config=config,
            plan_path=plan_result.output_path,
            created_at=created_at,
        )
    except Exception as exc:
        raise ResearchWorkflowError(str(exc)) from exc

    issues: list[ResearchWorkflowIssue] = []
    collections: list[ResearchWorkflowCollectionSummary] = []
    result_paths: list[Path] = []
    meridian_workflow_path: Path | None = None
    meridian_result_template_path: Path | None = None

    if cleaned_meridian_url is not None:
        try:
            meridian_result = prepare_meridian_workflow(
                config=config,
                company_name=plan_result.plan.deals[0].company_name,
                meridian_url=cleaned_meridian_url,
                created_at=created_at,
            )
            meridian_workflow_path = meridian_result.output_path
            meridian_result_template_path = meridian_result.result_template_path
            result_paths.append(meridian_result.result_template_path)
            issues.append(
                ResearchWorkflowIssue(
                    severity="warning",
                    source="meridian",
                    message=(
                        "Meridian is a manual authenticated workflow. Complete the "
                        "Meridian results template with short source-backed facts and "
                        "run the import dry run before relying on Meridian evidence."
                    ),
                )
            )
        except Exception as exc:
            issues.append(
                ResearchWorkflowIssue(
                    severity="error",
                    source="meridian",
                    message=str(exc),
                )
            )

    local_public_summary = _prepare_local_public_sources(
        config=config,
        company_names=[deal.company_name for deal in plan_result.plan.deals],
        sec_form_d_results_path=sec_form_d_results_path,
        sam_gov_results_path=sam_gov_results_path,
        usaspending_results_path=usaspending_results_path,
        sbir_results_path=sbir_results_path,
        uspto_results_path=uspto_results_path,
        github_results_path=github_results_path,
        collected_at=created_at,
    )
    if local_public_summary is not None:
        collections.append(local_public_summary)
        if local_public_summary.output_path is not None:
            result_paths.append(local_public_summary.output_path)
        if local_public_summary.error is not None:
            issues.append(
                ResearchWorkflowIssue(
                    severity="error",
                    source="local public sources",
                    message=local_public_summary.error,
                )
            )

    paid_company_names = list(company_names or [])
    if include_paid and paid_company_names:
        paid_summary = _prepare_paid_optional_sources(
            config=config,
            company_names=paid_company_names,
            clients=paid_clients,
            collected_at=created_at,
        )
        if paid_summary is not None:
            collections.append(paid_summary)
            if paid_summary.output_path is not None:
                result_paths.append(paid_summary.output_path)
            if paid_summary.error is not None:
                issues.append(
                    ResearchWorkflowIssue(
                        severity="error",
                        source="optional paid providers",
                        message=paid_summary.error,
                    )
                )

    live_collection_enabled = not config.local_only and config.enable_web_research
    if live_collection_enabled:
        live_summaries = _run_live_collectors(
            config=config,
            plan_path=plan_result.output_path,
            company_names=[deal.company_name for deal in plan_result.plan.deals],
            collected_at=created_at,
            web_client=web_client,
            usaspending_client=usaspending_client,
            sbir_client=sbir_client,
            sec_form_d_client=sec_form_d_client,
            github_client=github_client,
        )
        collections.extend(live_summaries)
        for collection in live_summaries:
            if collection.output_path is not None:
                result_paths.append(collection.output_path)
            if collection.error is not None:
                issues.append(
                    ResearchWorkflowIssue(
                        severity="error",
                        source=collection.source_name,
                        message=collection.error,
                    )
                )
            if collection.kind == "web":
                issues.extend(
                    ResearchWorkflowIssue(
                        severity="error",
                        source=collection.source_name,
                        message=warning,
                    )
                    for warning in collection.warnings
                )
    else:
        issues.append(
            ResearchWorkflowIssue(
                severity="warning",
                source="live public collection",
                message=(
                    "Live collection was not run because local-only mode is on or web "
                    "research is disabled."
                ),
            )
        )

    for path in results_files or []:
        result_paths.append(path)

    import_previews, import_issues = _preview_imports(
        config=config,
        result_paths=_dedupe_paths(result_paths),
        imported_at=created_at,
        supplied_result_paths=set(results_files or []),
    )
    issues.extend(import_issues)
    try:
        manual_task_queue_path = _write_manual_task_queue(
            config=config,
            manual_tasks=_unresolved_manual_tasks_from_plan(
                plan_result.plan,
                import_previews,
            ),
            created_at=created_at,
        )
    except Exception as exc:
        raise ResearchWorkflowError(str(exc)) from exc

    return ResearchWorkflowRunSummary(
        created_at=created_at,
        plan=plan_result.plan,
        plan_path=plan_result.output_path,
        result_template_path=template_result.output_path,
        manual_task_queue_path=manual_task_queue_path,
        meridian_workflow_path=meridian_workflow_path,
        meridian_result_template_path=meridian_result_template_path,
        source_summaries=_source_summaries(plan_result.plan),
        collections=collections,
        import_previews=import_previews,
        issues=issues,
        privacy_notes=_privacy_notes_for_collections(collections),
        live_collection_enabled=live_collection_enabled,
    )


def _write_manual_task_queue(
    *,
    config: AppConfig,
    manual_tasks: list[ResearchTask],
    created_at: datetime,
) -> Path | None:
    if not manual_tasks:
        return None
    output_dir = config.data_dir / "research-manual-tasks"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_manual_task_queue_path(output_dir, created_at)
    payload = {
        "version": "1",
        "created_at": _as_utc(created_at).isoformat(),
        "task_count": len(manual_tasks),
        "tasks": [task.model_dump(mode="json") for task in manual_tasks],
        "privacy_notes": list(PRIVACY_NOTES),
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="manual research task queue",
    )
    return output_path


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ResearchWorkflowError(
            f"Manual research task queue folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ResearchWorkflowError(f"Manual research task queue folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise ResearchWorkflowError(
            f"Could not create manual research task queue folder at {path}: {exc}"
        ) from exc


def _unique_manual_task_queue_path(output_dir: Path, created_at: datetime) -> Path:
    base_name = f"research-manual-tasks-{created_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ResearchWorkflowError(
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
        raise ResearchWorkflowError(
            f"Could not write {description} at {path}: the queue contains text that "
            "cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ResearchWorkflowError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        with suppress(OSError):
            temp_path.unlink()


def _prepare_local_public_sources(
    *,
    config: AppConfig,
    company_names: list[str],
    sec_form_d_results_path: Path | None,
    sam_gov_results_path: Path | None,
    usaspending_results_path: Path | None,
    sbir_results_path: Path | None,
    uspto_results_path: Path | None,
    github_results_path: Path | None,
    collected_at: datetime,
) -> ResearchWorkflowCollectionSummary | None:
    if not any(
        (
            sec_form_d_results_path,
            sam_gov_results_path,
            usaspending_results_path,
            sbir_results_path,
            uspto_results_path,
            github_results_path,
        )
    ):
        return None
    try:
        result = prepare_public_research_results(
            config=config,
            company_names=company_names,
            sec_form_d_results_path=sec_form_d_results_path,
            sam_gov_results_path=sam_gov_results_path,
            usaspending_results_path=usaspending_results_path,
            sbir_results_path=sbir_results_path,
            uspto_results_path=uspto_results_path,
            github_results_path=github_results_path,
            collected_at=collected_at,
        )
    except Exception as exc:
        return ResearchWorkflowCollectionSummary(
            kind="local_public",
            source_id="local_public",
            source_name="Local public-source files",
            status=ResearchProviderRunStatus.FAILED,
            error=str(exc),
        )
    return _collection_summary_from_result(
        kind="local_public",
        source_id="local_public",
        source_name="Local public-source files",
        result=result,
        skipped_non_exact_company_names=result.skipped_non_exact_company_names,
    )


def _prepare_paid_optional_sources(
    *,
    config: AppConfig,
    company_names: list[str],
    clients: Mapping[str, PaidProviderClient] | None,
    collected_at: datetime,
) -> ResearchWorkflowCollectionSummary | None:
    if (
        not config.enabled_paid_providers
        or not clients
        or config.local_only
        or not config.enable_web_research
    ):
        return None
    try:
        result = collect_paid_research_results(
            config=config,
            company_names=company_names,
            clients=clients,
            collected_at=collected_at,
        )
    except Exception as exc:
        return ResearchWorkflowCollectionSummary(
            kind="paid_optional",
            source_id="paid_optional",
            source_name="Optional paid providers",
            status=ResearchProviderRunStatus.FAILED,
            error=str(exc),
        )
    if not result.provider_ids:
        return None
    return _collection_summary_from_result(
        kind="paid_optional",
        source_id="paid_optional",
        source_name="Optional paid providers",
        result=result,
        skipped_non_exact_company_names=result.skipped_non_exact_company_names,
    )


def _privacy_notes_for_collections(
    collections: list[ResearchWorkflowCollectionSummary],
) -> list[str]:
    notes = list(PRIVACY_NOTES)
    if any(
        collection.kind == "paid_optional" and collection.output_path is not None
        for collection in collections
    ):
        notes[0] = PAID_OUTPUTS_PRIVACY_NOTE
    return notes


def _run_live_collectors(
    *,
    config: AppConfig,
    plan_path: Path,
    company_names: list[str],
    collected_at: datetime,
    web_client: WebResearchClient | None,
    usaspending_client: UsaspendingAwardsClient | None,
    sbir_client: SbirAwardsClient | None,
    sec_form_d_client: SecFormDFilingsClient | None,
    github_client: GitHubRepositorySearchClient | None,
) -> list[ResearchWorkflowCollectionSummary]:
    summaries = [
        _run_web_collection(
            config=config,
            plan_path=plan_path,
            collected_at=collected_at,
            client=web_client,
        )
    ]
    live_collectors: list[tuple[str, str, bool, Callable[[], object]]] = [
        (
            "sec_form_d",
            "SEC Form D",
            sec_form_d_client is not None
            or bool(os.environ.get(SEC_FORM_D_USER_AGENT_ENV_VAR, "").strip()),
            lambda: collect_sec_form_d_filings(
                config=config,
                company_names=company_names,
                client=sec_form_d_client,
                collected_at=collected_at,
            ),
        ),
        (
            "usaspending",
            "USAspending",
            usaspending_client is not None,
            lambda: collect_usaspending_awards(
                config=config,
                company_names=company_names,
                client=usaspending_client,
                collected_at=collected_at,
            ),
        ),
        (
            "sbir",
            "SBIR/STTR",
            sbir_client is not None,
            lambda: collect_sbir_awards(
                config=config,
                company_names=company_names,
                client=sbir_client,
                collected_at=collected_at,
            ),
        ),
        (
            "github",
            "GitHub",
            github_client is not None,
            lambda: collect_github_repositories(
                config=config,
                company_names=company_names,
                client=github_client,
                collected_at=collected_at,
            ),
        ),
    ]
    for source_id, source_name, should_run, collector in live_collectors:
        if not should_run:
            summaries.append(
                ResearchWorkflowCollectionSummary(
                    kind="live_public",
                    source_id=source_id,
                    source_name=source_name,
                    status=ResearchProviderRunStatus.NOT_RUN,
                )
            )
            continue
        try:
            result = collector()
        except Exception as exc:
            summaries.append(
                ResearchWorkflowCollectionSummary(
                    kind="live_public",
                    source_id=source_id,
                    source_name=source_name,
                    status=ResearchProviderRunStatus.FAILED,
                    error=str(exc),
                )
            )
            continue
        summaries.append(
            _collection_summary_from_result(
                kind="live_public",
                source_id=source_id,
                source_name=source_name,
                result=result,
            )
        )
    return summaries


def _run_web_collection(
    *,
    config: AppConfig,
    plan_path: Path,
    collected_at: datetime,
    client: WebResearchClient | None,
) -> ResearchWorkflowCollectionSummary:
    try:
        result = collect_web_research(
            config=config,
            plan_path=plan_path,
            client=client,
            collected_at=collected_at,
        )
    except Exception as exc:
        return ResearchWorkflowCollectionSummary(
            kind="web",
            source_id="public_web_pages",
            source_name="Direct public web pages",
            status=ResearchProviderRunStatus.FAILED,
            error=str(exc),
        )
    warnings = [
        f"{task.company_name} / {task.provider_id}: {task.reason}"
        for task in result.tasks
        if task.status == "failed"
    ]
    fetched_by_company = {
        task.company_name for task in result.tasks if task.status == "fetched"
    }
    no_result_companies = (
        sorted(
            {
                task.company_name
                for task in result.tasks
                if task.company_name not in fetched_by_company
            }
        )
        if fetched_by_company
        else []
    )
    status = (
        ResearchProviderRunStatus.FAILED
        if warnings
        else _collection_status(
            result_count=result.fetched_count,
            no_result_companies=no_result_companies,
            warnings=warnings,
            error=None,
        )
    )
    return ResearchWorkflowCollectionSummary(
        kind="web",
        source_id="public_web_pages",
        source_name="Direct public web pages",
        status=status,
        output_path=result.output_path,
        result_count=result.fetched_count,
        deal_count=len({task.company_name for task in result.tasks}),
        no_result_companies=sorted(no_result_companies),
        provider_statuses=_web_provider_statuses(result.tasks),
        incomplete_search=_warnings_indicate_incomplete_search(warnings),
        warnings=warnings,
    )


def _collection_summary_from_result(
    *,
    kind: CollectionKind,
    source_id: str,
    source_name: str,
    result: object,
    skipped_non_exact_company_names: list[str] | None = None,
) -> ResearchWorkflowCollectionSummary:
    output_path = getattr(result, "output_path", None)
    result_count = int(getattr(result, "result_count", 0))
    deals = list(getattr(result, "deals", []))
    match_details = list(getattr(result, "match_details", []))
    warnings = [
        *list(getattr(result, "warnings", [])),
        *_identity_match_warnings(match_details),
    ]
    no_result_companies = [
        deal.company_name for deal in deals if getattr(deal, "result_count", 0) == 0
    ]
    incomplete_search = _warnings_indicate_incomplete_search(warnings)
    provider_ids = list(getattr(result, "provider_ids", []))
    provider_result_counts = dict(getattr(result, "provider_result_counts", {}))
    provider_company_result_counts = {
        str(provider_id): dict(company_counts)
        for provider_id, company_counts in dict(
            getattr(result, "provider_company_result_counts", {})
        ).items()
    }
    return ResearchWorkflowCollectionSummary(
        kind=kind,
        source_id=source_id,
        source_name=source_name,
        status=_collection_status(
            result_count=result_count,
            no_result_companies=no_result_companies,
            warnings=warnings,
            error=None,
        ),
        output_path=output_path,
        result_count=result_count,
        deal_count=len(deals),
        no_result_companies=no_result_companies,
        skipped_non_exact_company_names=skipped_non_exact_company_names or [],
        match_details=match_details,
        provider_ids=provider_ids,
        provider_result_counts=provider_result_counts,
        provider_company_result_counts=provider_company_result_counts,
        provider_statuses=(
            _local_public_provider_statuses(
                provider_ids=provider_ids,
                provider_result_counts=provider_result_counts,
                provider_company_result_counts=provider_company_result_counts,
            )
            if kind in {"local_public", "paid_optional"}
            else []
        ),
        incomplete_search=incomplete_search,
        warnings=warnings,
    )


def _identity_match_warnings(match_details: list[CompanyMatch]) -> list[str]:
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    skipped_count = 0
    for match in match_details:
        if match.import_ready:
            continue
        skipped_count += 1
        key = (match.requested_name, match.candidate_name, match.kind.value)
        if key in seen:
            continue
        seen.add(key)
        if len(warnings) >= MAX_IDENTITY_WARNING_DETAILS:
            continue
        kind = match.kind.value.replace("_", " ")
        warnings.append(
            "Skipped external research result for "
            f"{match.requested_name}: {match.candidate_name} was classified as "
            f"{kind}. {match.reason}"
        )
    remaining = skipped_count - len(warnings)
    if remaining > 0:
        warning_word = "result" if remaining == 1 else "results"
        warnings.append(
            f"Skipped {remaining} additional external research {warning_word} because "
            "the company identity was not an import-ready match."
        )
    return warnings


def _preview_imports(
    *,
    config: AppConfig,
    result_paths: list[Path],
    imported_at: datetime,
    supplied_result_paths: set[Path],
) -> tuple[list[ResearchWorkflowImportPreview], list[ResearchWorkflowIssue]]:
    previews: list[ResearchWorkflowImportPreview] = []
    issues: list[ResearchWorkflowIssue] = []
    for path in result_paths:
        try:
            result = import_research_results(
                config=config,
                results_path=path,
                imported_at=imported_at,
                dry_run=True,
            )
        except ResearchImportError as exc:
            message = str(exc)
            previews.append(
                ResearchWorkflowImportPreview(
                    input_path=path,
                    meridian_preview=_safe_meridian_preview(
                        path,
                        imported_at=imported_at,
                    ),
                    error=message,
                )
            )
            severity: IssueSeverity = (
                "error" if path in supplied_result_paths else _import_issue_severity(message)
            )
            issues.append(
                ResearchWorkflowIssue(
                    severity=severity,
                    source=f"import dry run for {path}",
                    message=message,
                )
            )
            continue
        previews.append(_import_preview_from_result(result))
    return previews, issues


def _import_preview_from_result(
    result: ResearchImportRunSummary,
) -> ResearchWorkflowImportPreview:
    return ResearchWorkflowImportPreview(
        input_path=result.input_path,
        imported_count=result.imported_count,
        skipped_duplicate_count=result.skipped_duplicate_count,
        skipped_blank_template_row_count=result.skipped_blank_template_row_count,
        stale_count=result.stale_count,
        meridian_preview=result.meridian_preview,
        deals=result.deals,
    )


def _safe_meridian_preview(
    path: Path,
    *,
    imported_at: datetime,
) -> MeridianImportPreview | None:
    try:
        return preview_meridian_results_file(
            results_path=path,
            imported_at=imported_at,
        )
    except ResearchImportError:
        return None


def _import_issue_severity(message: str) -> IssueSeverity:
    if message.startswith("No ingested deals were found."):
        return "warning"
    return "error"


def _collection_status(
    *,
    result_count: int,
    no_result_companies: list[str],
    warnings: list[str],
    error: str | None,
) -> ResearchProviderRunStatus:
    if error is not None:
        return ResearchProviderRunStatus.FAILED
    if _warnings_indicate_incomplete_search(warnings):
        return ResearchProviderRunStatus.INCOMPLETE_SEARCH
    if result_count > 0:
        return ResearchProviderRunStatus.PLANNED
    if no_result_companies:
        return ResearchProviderRunStatus.NO_EXACT_RESULTS
    return ResearchProviderRunStatus.NOT_RUN


def _local_public_provider_statuses(
    *,
    provider_ids: list[str],
    provider_result_counts: dict[str, int],
    provider_company_result_counts: dict[str, dict[str, int]],
) -> list[ResearchProviderStatusSummary]:
    statuses: list[ResearchProviderStatusSummary] = []
    for provider_id in provider_ids:
        result_count = provider_result_counts.get(provider_id, 0)
        company_counts = provider_company_result_counts.get(provider_id, {})
        no_exact_result_companies = sorted(
            company_name
            for company_name, company_result_count in company_counts.items()
            if company_result_count == 0
        )
        statuses.append(
            ResearchProviderStatusSummary(
                provider_id=provider_id,
                provider_name=_provider_display_name(provider_id),
                status=(
                    ResearchProviderRunStatus.PLANNED
                    if result_count > 0
                    else ResearchProviderRunStatus.NO_EXACT_RESULTS
                ),
                collected_count=result_count,
                no_exact_result_companies=no_exact_result_companies,
            )
        )
    return statuses


def _web_provider_statuses(
    tasks: list[WebResearchTaskSummary],
) -> list[ResearchProviderStatusSummary]:
    by_provider: dict[str, list[WebResearchTaskSummary]] = {}
    for task in tasks:
        by_provider.setdefault(task.provider_id, []).append(task)

    statuses: list[ResearchProviderStatusSummary] = []
    for provider_id, provider_tasks in by_provider.items():
        provider_name = provider_tasks[0].provider_name
        fetched_count = sum(1 for task in provider_tasks if task.status == "fetched")
        planned_count = sum(1 for task in provider_tasks if task.status == "planned")
        skipped_count = sum(1 for task in provider_tasks if task.status == "skipped")
        warnings = [
            f"{task.company_name}: {task.reason}"
            for task in provider_tasks
            if task.status == "failed"
        ]
        no_result_companies: list[str] = []
        if warnings:
            status = ResearchProviderRunStatus.FAILED
        elif fetched_count > 0 or planned_count > 0:
            status = ResearchProviderRunStatus.PLANNED
        elif skipped_count > 0:
            status = (
                ResearchProviderRunStatus.MANUAL_NEEDED
                if any(_web_task_needs_manual_work(task) for task in provider_tasks)
                else ResearchProviderRunStatus.NOT_RUN
            )
        else:
            status = ResearchProviderRunStatus.NOT_RUN
        statuses.append(
            ResearchProviderStatusSummary(
                provider_id=provider_id,
                provider_name=provider_name,
                status=status,
                collected_count=fetched_count,
                warning_count=len(warnings),
                no_exact_result_companies=no_result_companies,
                incomplete_search=_warnings_indicate_incomplete_search(warnings),
                failure="; ".join(warnings) if warnings else None,
            )
        )
    return statuses


def _provider_display_name(provider_id: str) -> str:
    provider = {
        provider.id: provider
        for provider in builtin_research_providers(include_paid=True)
    }.get(provider_id)
    return provider.name if provider is not None else provider_id


def _import_preview_provider_statuses(
    preview: ResearchWorkflowImportPreview,
    *,
    plan: ResearchPlan,
) -> list[ResearchProviderStatusSummary]:
    if preview.error is not None or preview.imported_count <= 0:
        return []
    try:
        results_file = ResearchResultsFile.model_validate_json(
            preview.input_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    deal_company_names = {
        deal.deal_id: deal.company_name
        for deal in plan.deals
    }
    provider_counts: dict[str, int] = {}
    for result in results_file.results:
        company_name = result.company_name
        if company_name is None and result.deal_id is not None:
            company_name = deal_company_names.get(result.deal_id)
        if company_name is None:
            continue
        provider_counts[result.provider_id] = provider_counts.get(result.provider_id, 0) + 1
    return [
        ResearchProviderStatusSummary(
            provider_id=provider_id,
            provider_name=_provider_display_name(provider_id),
            status=ResearchProviderRunStatus.PLANNED,
            collected_count=result_count,
        )
        for provider_id, result_count in sorted(provider_counts.items())
        if result_count > 0
    ]


def _warnings_indicate_incomplete_search(warnings: list[str]) -> bool:
    return any(
        "incomplete" in warning.casefold()
        or "more " in warning.casefold()
        and "may exist" in warning.casefold()
        for warning in warnings
    )


def _web_task_needs_manual_work(task: WebResearchTaskSummary) -> bool:
    reason = task.reason.casefold()
    return "manual" in reason or "paid" in reason or "authenticated" in reason


def _research_workflow_summary(
    workflow: ResearchWorkflowRunSummary,
) -> ResearchWorkflowSummary:
    provider_ids_with_ready_results = _provider_ids_with_ready_results(workflow)
    statuses = {
        source.provider_id: ResearchProviderStatusSummary(
            provider_id=source.provider_id,
            provider_name=source.provider_name,
            status=_initial_provider_status(
                source,
                workflow,
                provider_ids_with_ready_results=provider_ids_with_ready_results,
            ),
            planned_count=source.planned_count,
        )
        for source in workflow.source_summaries
    }
    collection_output_paths = {
        collection.output_path.resolve(strict=False)
        for collection in workflow.collections
        if collection.output_path is not None
    }

    for collection in workflow.collections:
        for provider_status in collection.provider_statuses:
            existing_status = statuses.get(provider_status.provider_id)
            if existing_status is None:
                existing_status = ResearchProviderStatusSummary(
                    provider_id=provider_status.provider_id,
                    provider_name=provider_status.provider_name,
                    status=provider_status.status,
                )
            statuses[provider_status.provider_id] = existing_status.model_copy(
                update={
                    "provider_name": provider_status.provider_name,
                    "status": _merge_provider_status_summary(
                        existing=existing_status,
                        incoming=provider_status,
                    ),
                    "collected_count": (
                        existing_status.collected_count
                        + provider_status.collected_count
                    ),
                    "warning_count": (
                        existing_status.warning_count + provider_status.warning_count
                    ),
                    "no_exact_result_companies": sorted(
                        set(existing_status.no_exact_result_companies)
                        | set(provider_status.no_exact_result_companies)
                    ),
                    "incomplete_search": (
                        existing_status.incomplete_search
                        or provider_status.incomplete_search
                    ),
                    "failure": provider_status.failure or existing_status.failure,
                }
            )
        if collection.provider_statuses:
            continue
        status = statuses.get(collection.source_id)
        if status is None:
            status = ResearchProviderStatusSummary(
                provider_id=collection.source_id,
                provider_name=collection.source_name,
                status=collection.status,
            )
        statuses[collection.source_id] = status.model_copy(
            update={
                "provider_name": collection.source_name,
                "status": _merge_provider_status_summary(
                    existing=status,
                    incoming=ResearchProviderStatusSummary(
                        provider_id=collection.source_id,
                        provider_name=collection.source_name,
                        status=collection.status,
                        collected_count=collection.result_count,
                    ),
                ),
                "collected_count": status.collected_count + collection.result_count,
                "warning_count": status.warning_count + len(collection.warnings),
                "no_exact_result_companies": sorted(
                    set(status.no_exact_result_companies)
                    | set(collection.no_result_companies)
                ),
                "incomplete_search": status.incomplete_search
                or collection.incomplete_search,
                "failure": collection.error or status.failure,
            }
        )

    for preview in workflow.import_previews:
        preview_path = preview.input_path.resolve(strict=False)
        if preview.imported_count <= 0 or preview_path in collection_output_paths:
            continue
        for provider_status in _import_preview_provider_statuses(
            preview,
            plan=workflow.plan,
        ):
            existing_status = statuses.get(provider_status.provider_id)
            if existing_status is None:
                existing_status = ResearchProviderStatusSummary(
                    provider_id=provider_status.provider_id,
                    provider_name=provider_status.provider_name,
                    status=provider_status.status,
                )
            statuses[provider_status.provider_id] = existing_status.model_copy(
                update={
                    "provider_name": provider_status.provider_name,
                    "status": _merge_provider_status_summary(
                        existing=existing_status,
                        incoming=provider_status,
                    ),
                    "collected_count": (
                        existing_status.collected_count
                        + provider_status.collected_count
                    ),
                    "no_exact_result_companies": sorted(
                        set(existing_status.no_exact_result_companies)
                        | set(provider_status.no_exact_result_companies)
                    ),
                }
            )

    provider_statuses = list(statuses.values())
    warning_count = sum(1 for issue in workflow.issues if issue.severity == "warning")
    warning_count += sum(len(collection.warnings) for collection in workflow.collections)
    return ResearchWorkflowSummary(
        planned_task_count=workflow.plan.task_count,
        imported_record_count=sum(
            preview.imported_count for preview in workflow.import_previews
        ),
        stale_record_count=sum(preview.stale_count for preview in workflow.import_previews),
        failed_provider_count=sum(
            1
            for status in provider_statuses
            if status.status == ResearchProviderRunStatus.FAILED
        ),
        incomplete_search_count=sum(
            1 for status in provider_statuses if status.incomplete_search
        ),
        no_exact_result_provider_count=sum(
            1
            for status in provider_statuses
            if status.status == ResearchProviderRunStatus.NO_EXACT_RESULTS
        ),
        manual_needed_provider_count=sum(
            1
            for status in provider_statuses
            if status.status == ResearchProviderRunStatus.MANUAL_NEEDED
        ),
        not_run_provider_count=sum(
            1
            for status in provider_statuses
            if status.status == ResearchProviderRunStatus.NOT_RUN
        ),
        warning_count=warning_count,
        provider_statuses=provider_statuses,
    )


def _merge_provider_status(
    existing: ResearchProviderRunStatus,
    incoming: ResearchProviderRunStatus,
) -> ResearchProviderRunStatus:
    rank = {
        ResearchProviderRunStatus.FAILED: 0,
        ResearchProviderRunStatus.INCOMPLETE_SEARCH: 1,
        ResearchProviderRunStatus.IMPORTED: 2,
        ResearchProviderRunStatus.NO_EXACT_RESULTS: 3,
        ResearchProviderRunStatus.MANUAL_NEEDED: 4,
        ResearchProviderRunStatus.PLANNED: 5,
        ResearchProviderRunStatus.NOT_RUN: 6,
    }
    return existing if rank[existing] <= rank[incoming] else incoming


def _merge_provider_status_summary(
    *,
    existing: ResearchProviderStatusSummary,
    incoming: ResearchProviderStatusSummary,
) -> ResearchProviderRunStatus:
    if _ready_result_clears_manual_needed(existing=existing, incoming=incoming):
        return ResearchProviderRunStatus.PLANNED
    if _has_ready_results(existing) or _has_ready_results(incoming):
        higher_priority_statuses = {
            ResearchProviderRunStatus.FAILED,
            ResearchProviderRunStatus.INCOMPLETE_SEARCH,
            ResearchProviderRunStatus.IMPORTED,
        }
        if (
            existing.status not in higher_priority_statuses
            and incoming.status not in higher_priority_statuses
        ):
            return ResearchProviderRunStatus.PLANNED
    if (
        ResearchProviderRunStatus.NO_EXACT_RESULTS
        in {existing.status, incoming.status}
        and ResearchProviderRunStatus.PLANNED in {existing.status, incoming.status}
        and (existing.collected_count > 0 or incoming.collected_count > 0)
    ):
        return ResearchProviderRunStatus.PLANNED
    return _merge_provider_status(existing.status, incoming.status)


def _ready_result_clears_manual_needed(
    *,
    existing: ResearchProviderStatusSummary,
    incoming: ResearchProviderStatusSummary,
) -> bool:
    return (
        existing.status == ResearchProviderRunStatus.MANUAL_NEEDED
        and _has_ready_results(incoming)
    ) or (
        incoming.status == ResearchProviderRunStatus.MANUAL_NEEDED
        and _has_ready_results(existing)
    )


def _has_ready_results(status: ResearchProviderStatusSummary) -> bool:
    return (
        status.status == ResearchProviderRunStatus.PLANNED
        and status.collected_count > 0
    )


def _provider_ids_with_ready_results(
    workflow: ResearchWorkflowRunSummary,
) -> set[str]:
    provider_ids = {
        provider_status.provider_id
        for collection in workflow.collections
        for provider_status in collection.provider_statuses
        if provider_status.collected_count > 0
    }
    collection_output_paths = {
        collection.output_path.resolve(strict=False)
        for collection in workflow.collections
        if collection.output_path is not None
    }
    for preview in workflow.import_previews:
        if preview.input_path.resolve(strict=False) in collection_output_paths:
            continue
        provider_ids.update(
            provider_status.provider_id
            for provider_status in _import_preview_provider_statuses(
                preview,
                plan=workflow.plan,
            )
            if provider_status.collected_count > 0
        )
    return provider_ids


def _initial_provider_status(
    source: ResearchWorkflowSourceSummary,
    workflow: ResearchWorkflowRunSummary,
    *,
    provider_ids_with_ready_results: set[str],
) -> ResearchProviderRunStatus:
    if source.provider_id in provider_ids_with_ready_results:
        return ResearchProviderRunStatus.PLANNED
    if source.manual_count:
        return ResearchProviderRunStatus.MANUAL_NEEDED
    if _source_not_run_by_live_gate(source, workflow):
        return ResearchProviderRunStatus.NOT_RUN
    return ResearchProviderRunStatus.PLANNED


def _source_not_run_by_live_gate(
    source: ResearchWorkflowSourceSummary,
    workflow: ResearchWorkflowRunSummary,
) -> bool:
    return (
        not workflow.live_collection_enabled
        and source.live_collectable_count > 0
        and source.provider_id in LIVE_PROVIDER_IDS
    )


def _source_summaries(plan: ResearchPlan) -> list[ResearchWorkflowSourceSummary]:
    summaries: dict[str, ResearchWorkflowSourceSummary] = {}
    for task in plan.tasks:
        summary = summaries.setdefault(
            task.provider_id,
            ResearchWorkflowSourceSummary(
                provider_id=task.provider_id,
                provider_name=task.provider_name,
            ),
        )
        summaries[task.provider_id] = summary.model_copy(
            update={
                "planned_count": summary.planned_count + 1,
                "manual_count": summary.manual_count + int(_task_needs_manual_work(task)),
                "live_collectable_count": (
                    summary.live_collectable_count + int(_task_can_collect_live(task))
                ),
            }
        )
    return list(summaries.values())


def _unresolved_manual_tasks(workflow: ResearchWorkflowRunSummary) -> list[ResearchTask]:
    return _unresolved_manual_tasks_from_plan(
        workflow.plan,
        workflow.import_previews,
    )


def _unresolved_manual_tasks_from_plan(
    plan: ResearchPlan,
    import_previews: list[ResearchWorkflowImportPreview],
) -> list[ResearchTask]:
    resolved_result_keys = _resolved_research_result_keys(plan, import_previews)
    return [
        task
        for task in plan.tasks
        if _task_needs_manual_work(task)
        and _research_result_key(task.company_name, task.provider_id)
        not in resolved_result_keys
    ]


def _resolved_research_result_keys(
    plan: ResearchPlan,
    import_previews: list[ResearchWorkflowImportPreview],
) -> set[tuple[str, str]]:
    deal_company_names = {
        deal.deal_id: deal.company_name
        for deal in plan.deals
    }
    result_keys: set[tuple[str, str]] = set()
    for preview in import_previews:
        if preview.error is not None:
            continue
        try:
            results_file = ResearchResultsFile.model_validate_json(
                preview.input_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        for result in results_file.results:
            company_name = result.company_name
            if company_name is None and result.deal_id is not None:
                company_name = deal_company_names.get(result.deal_id)
            if company_name is None:
                continue
            result_keys.add(_research_result_key(company_name, result.provider_id))
    return result_keys


def _research_result_key(company_name: str, provider_id: str) -> tuple[str, str]:
    return (company_name.strip().casefold(), provider_id.strip())


def _task_needs_manual_work(task: ResearchTask) -> bool:
    if task.status == ResearchTaskStatus.NEEDS_OPERATOR:
        return True
    if task.provider_category in {
        ResearchProviderCategory.AUTHENTICATED_PORTAL,
        ResearchProviderCategory.PAID_OPTIONAL,
    }:
        return True
    return task.provider_id in MANUAL_OR_LOCAL_PROVIDER_IDS


def _task_can_collect_live(task: ResearchTask) -> bool:
    if task.provider_id not in LIVE_PROVIDER_IDS:
        return False
    if task.provider_id == "company_website":
        return task.url is not None and task.status == ResearchTaskStatus.PLANNED
    return True


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        marker = path.expanduser()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(path)
    return deduped


def _clean_workflow_website_url(url: str | None) -> str | None:
    if url is None:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        raise ResearchWorkflowError("The website URL is not a valid URL.") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ResearchWorkflowError("The website URL must start with http:// or https://.")
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise ResearchWorkflowError("The website URL is not a valid URL.") from exc
    if not parsed.netloc or host is None:
        raise ResearchWorkflowError("The website URL must include a website host.")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ResearchWorkflowError("The website URL has an invalid port.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ResearchWorkflowError(
            "The website URL cannot include a username or password."
        )
    if any(character.isspace() for character in cleaned):
        raise ResearchWorkflowError("The website URL cannot contain spaces.")
    try:
        decoded_path = parsed.path
        for _ in range(len(parsed.path) + 1):
            next_decoded_path = unquote(decoded_path, errors="strict")
            if next_decoded_path == decoded_path:
                break
            decoded_path = next_decoded_path
            if any(delimiter in decoded_path for delimiter in ("?", "#", ";")):
                break
        else:
            raise ResearchWorkflowError("The website URL is not a valid URL.")
    except UnicodeDecodeError as exc:
        raise ResearchWorkflowError("The website URL is not a valid URL.") from exc
    if (
        ";" in parsed.path
        or any(delimiter in decoded_path for delimiter in ("?", "#", ";"))
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ResearchWorkflowError(
            "The website URL cannot include query strings, fragments, or extra "
            "parameter text. Use the base public page URL."
        )
    return cleaned


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
