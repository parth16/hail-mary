from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from hailmary.config import AppConfig

from .collection import (
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
from .importer import ResearchImportError, import_research_results
from .meridian import clean_meridian_url, prepare_meridian_workflow
from .planner import prepare_research_plan
from .schemas import (
    ResearchImportDealSummary,
    ResearchImportRunSummary,
    ResearchPlan,
    ResearchProviderCategory,
    ResearchTask,
    ResearchTaskStatus,
)
from .templates import prepare_research_results_template
from .web import WebResearchClient, collect_web_research


class ResearchWorkflowError(RuntimeError):
    """The higher-level research workflow could not be prepared safely."""


IssueSeverity = Literal["warning", "error"]
CollectionKind = Literal["local_public", "live_public", "web"]

LIVE_PROVIDER_IDS = {"company_website", "sec_form_d", "usaspending", "sbir", "github"}
MANUAL_OR_LOCAL_PROVIDER_IDS = {"sam_gov", "uspto", "public_web"}
PRIVACY_NOTES = [
    "No screenshots, cookies, browser profiles, raw portal HTML, hidden authenticated data, "
    "signed URLs, or paid-source outputs are saved by this workflow.",
    "Every imported external fact still needs provider, retrieval time, exact URL or API source, "
    "confidence, and licensing notes.",
]


class ResearchWorkflowIssue(BaseModel):
    severity: IssueSeverity
    source: str
    message: str


class ResearchWorkflowArtifact(BaseModel):
    kind: str
    path: Path


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
    output_path: Path | None = None
    result_count: int = 0
    deal_count: int = 0
    no_result_companies: list[str] = Field(default_factory=list)
    skipped_non_exact_company_names: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchWorkflowImportPreview(BaseModel):
    input_path: Path
    imported_count: int = 0
    skipped_duplicate_count: int = 0
    skipped_blank_template_row_count: int = 0
    deals: list[ResearchImportDealSummary] = Field(default_factory=list)
    error: str | None = None


class ResearchWorkflowRunSummary(BaseModel):
    created_at: datetime
    plan: ResearchPlan
    plan_path: Path
    result_template_path: Path
    meridian_workflow_path: Path | None = None
    meridian_result_template_path: Path | None = None
    source_summaries: list[ResearchWorkflowSourceSummary] = Field(default_factory=list)
    collections: list[ResearchWorkflowCollectionSummary] = Field(default_factory=list)
    import_previews: list[ResearchWorkflowImportPreview] = Field(default_factory=list)
    issues: list[ResearchWorkflowIssue] = Field(default_factory=list)
    privacy_notes: list[str] = Field(default_factory=lambda: list(PRIVACY_NOTES))
    live_collection_enabled: bool = False

    @property
    def planned_source_count(self) -> int:
        return len(self.source_summaries)

    @property
    def manual_task_count(self) -> int:
        return sum(source.manual_count for source in self.source_summaries)

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
    def artifacts(self) -> list[ResearchWorkflowArtifact]:
        artifacts = [
            ResearchWorkflowArtifact(kind="research_plan", path=self.plan_path),
            ResearchWorkflowArtifact(
                kind="research_results_template",
                path=self.result_template_path,
            ),
        ]
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
) -> ResearchWorkflowRunSummary:
    created_at = _as_utc(created_at or datetime.now(UTC))
    try:
        cleaned_meridian_url = (
            clean_meridian_url(meridian_url) if meridian_url is not None else None
        )
        plan_result = prepare_research_plan(
            config=config,
            company_names=company_names or [],
            website_url=website_url,
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

    return ResearchWorkflowRunSummary(
        created_at=created_at,
        plan=plan_result.plan,
        plan_path=plan_result.output_path,
        result_template_path=template_result.output_path,
        meridian_workflow_path=meridian_workflow_path,
        meridian_result_template_path=meridian_result_template_path,
        source_summaries=_source_summaries(plan_result.plan),
        collections=collections,
        import_previews=import_previews,
        issues=issues,
        live_collection_enabled=live_collection_enabled,
    )


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
            error=str(exc),
        )
    return _collection_summary_from_result(
        kind="local_public",
        source_id="local_public",
        source_name="Local public-source files",
        result=result,
        skipped_non_exact_company_names=result.skipped_non_exact_company_names,
    )


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
    live_collectors: list[tuple[str, str, Callable[[], object]]] = [
        (
            "sec_form_d",
            "SEC Form D",
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
            lambda: collect_github_repositories(
                config=config,
                company_names=company_names,
                client=github_client,
                collected_at=collected_at,
            ),
        ),
    ]
    for source_id, source_name, collector in live_collectors:
        try:
            result = collector()
        except Exception as exc:
            summaries.append(
                ResearchWorkflowCollectionSummary(
                    kind="live_public",
                    source_id=source_id,
                    source_name=source_name,
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
            error=str(exc),
        )
    fetched_by_company: dict[str, int] = {}
    for task in result.tasks:
        if task.status == "fetched":
            fetched_by_company[task.company_name] = (
                fetched_by_company.get(task.company_name, 0) + 1
            )
    no_result_companies = [
        company_name
        for company_name in {task.company_name for task in result.tasks}
        if fetched_by_company.get(company_name, 0) == 0
    ]
    warnings = [
        f"{task.company_name} / {task.provider_id}: {task.reason}"
        for task in result.tasks
        if task.status == "failed"
    ]
    return ResearchWorkflowCollectionSummary(
        kind="web",
        source_id="public_web_pages",
        source_name="Direct public web pages",
        output_path=result.output_path,
        result_count=result.fetched_count,
        deal_count=len({task.company_name for task in result.tasks}),
        no_result_companies=sorted(no_result_companies),
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
    warnings = list(getattr(result, "warnings", []))
    return ResearchWorkflowCollectionSummary(
        kind=kind,
        source_id=source_id,
        source_name=source_name,
        output_path=output_path,
        result_count=result_count,
        deal_count=len(deals),
        no_result_companies=[
            deal.company_name for deal in deals if getattr(deal, "result_count", 0) == 0
        ],
        skipped_non_exact_company_names=skipped_non_exact_company_names or [],
        warnings=warnings,
    )


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
            previews.append(ResearchWorkflowImportPreview(input_path=path, error=message))
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
        deals=result.deals,
    )


def _import_issue_severity(message: str) -> IssueSeverity:
    if "No ingested deals" in message or "Run `hailmary ingest-folder`" in message:
        return "warning"
    return "error"


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


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
