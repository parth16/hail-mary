from __future__ import annotations

import html
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from hailmary.config import (
    AppConfig,
    ConfigError,
    create_local_state,
    validate_local_state,
)
from hailmary.evaluation import (
    DealEvaluationResult,
    EvaluationError,
    evaluate_deal_folder,
)
from hailmary.evidence.actions import EvidenceActionError, apply_evidence_actions
from hailmary.portfolio import PortfolioError, portfolio_status
from hailmary.portfolio.scenario import portfolio_scenario
from hailmary.schemas.evidence import EvidenceStore
from hailmary.schemas.scoring import Recommendation, ScoredDeal
from hailmary.scoring.portfolio import (
    portfolio_exposure_state_after_score,
    portfolio_exposure_state_from_ledger,
    portfolio_rank_key,
    skip_reason,
)
from hailmary.scoring.scorer import score_evidence_store
from hailmary.utils.slug import slugify


class BatchEvaluationError(RuntimeError):
    """Batch evaluation could not continue safely."""


@dataclass(frozen=True)
class BatchDealOutcome:
    folder: Path
    company_name: str
    result: DealEvaluationResult | None = None
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class BatchAllocationRow:
    company_name: str
    deal_id: str | None
    folder: Path
    evaluation_status: str
    final_recommendation: Recommendation
    single_deal_check_size: int
    batch_check_size: int
    score: int | None
    max_score: int | None
    confidence: str | None
    key_blockers: list[str]
    memo_path: Path | None = None
    json_path: Path | None = None
    allocation_sequence: int | None = None
    portfolio_rank: int | None = None
    capital_before: int | None = None
    capital_after: int | None = None
    skipped_reason: str | None = None
    failure_reason: str | None = None

    @property
    def allocated(self) -> bool:
        return self.batch_check_size > 0


@dataclass(frozen=True)
class BatchPortfolioConstraints:
    starting_capital: int
    reserve_amount: int
    allocatable_capital: int
    existing_invested_capital: int
    available_capital_before_batch: int
    new_allocated_capital: int
    remaining_allocatable_capital: int
    allowed_check_sizes: list[int]
    configured_min_check: int
    configured_max_check: int
    max_company_exposure_percent: str
    max_category_exposure_percent: str
    max_stage_exposure_percent: str
    max_low_confidence_exposure_percent: str
    max_medium_confidence_exposure_percent: str
    max_high_confidence_exposure_percent: str


@dataclass(frozen=True)
class BatchEvaluationResult:
    root_folder: Path
    created_at: datetime
    report_path: Path
    json_path: Path
    outcomes: list[BatchDealOutcome]
    allocation_rows: list[BatchAllocationRow]
    constraints: BatchPortfolioConstraints
    warnings: list[str] = field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.outcomes)

    @property
    def evaluated_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.succeeded)

    @property
    def allocated_count(self) -> int:
        return sum(1 for row in self.allocation_rows if row.allocated)

    @property
    def skipped_count(self) -> int:
        return sum(
            1
            for row in self.allocation_rows
            if not row.allocated and row.evaluation_status != "failed"
        )


def batch_evaluate_folder(
    root_folder: Path,
    *,
    config: AppConfig,
    max_concurrency: int = 3,
    run_research: bool = True,
    include_paid_research: bool = False,
    created_at: datetime | None = None,
    deal_stage_callback: Callable[[str, str], None] | None = None,
) -> BatchEvaluationResult:
    if max_concurrency < 1:
        raise BatchEvaluationError("--max-concurrency must be at least 1.")

    root = _resolve_batch_root(root_folder)
    try:
        create_local_state(config, force=False)
        config = validate_local_state(config)
    except ConfigError as exc:
        raise BatchEvaluationError(f"Local generated-data setup failed: {exc}") from exc

    deal_folders = _discover_deal_folders(root, config=config)
    if not deal_folders:
        raise BatchEvaluationError(
            f"No company folders were found in {root}. Put each deal in its own "
            "child folder and run batch-evaluate again."
        )

    run_created_at = created_at or datetime.now(UTC)
    outcomes: list[BatchDealOutcome] = []
    for folder in deal_folders:
        if folder.is_symlink():
            outcomes.append(
                BatchDealOutcome(
                    folder=folder,
                    company_name=folder.name,
                    failure_reason=(
                        "The deal folder is a symlink. Choose a real folder so "
                        "Hail Mary can keep source lineage local."
                    ),
                )
            )
            continue
        stage_callback: Callable[[str], None] | None = None
        if deal_stage_callback is not None:
            deal_name = folder.name

            def _stage_callback(stage: str, *, deal_name: str = deal_name) -> None:
                deal_stage_callback(deal_name, stage)

            stage_callback = _stage_callback

        try:
            deal_result = evaluate_deal_folder(
                folder,
                config=config,
                max_concurrency=max_concurrency,
                run_research=run_research,
                include_paid_research=include_paid_research,
                stage_callback=stage_callback,
                created_at=run_created_at,
            )
        except EvaluationError as exc:
            outcomes.append(
                BatchDealOutcome(
                    folder=folder,
                    company_name=folder.name,
                    failure_reason=str(exc),
                )
            )
            continue
        outcomes.append(
            BatchDealOutcome(
                folder=folder,
                company_name=deal_result.company_name,
                result=deal_result,
            )
        )
    if not any(outcome.succeeded for outcome in outcomes):
        raise BatchEvaluationError(
            "No deal could be evaluated successfully. Fix the per-deal failures "
            "and run batch-evaluate again."
        )

    rows, constraints = _allocate_batch(outcomes, config=config)
    if not any(row.evaluation_status != "failed" for row in rows):
        raise BatchEvaluationError(
            "No deal could be evaluated successfully. Fix the per-deal failures "
            "and run batch-evaluate again."
        )
    report_dir = config.data_dir / "reports"
    _ensure_private_directory(report_dir, private_root=config.data_dir, description="batch report")
    run_slug = slugify(root.name) or "batch"
    timestamp = run_created_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"{run_slug}-{timestamp}-batch-portfolio-report.md"
    json_path = report_dir / f"{run_slug}-{timestamp}-batch-portfolio.json"
    batch_result = BatchEvaluationResult(
        root_folder=root,
        created_at=run_created_at,
        report_path=report_path,
        json_path=json_path,
        outcomes=outcomes,
        allocation_rows=rows,
        constraints=constraints,
    )
    _write_private_text(
        report_path,
        render_batch_portfolio_report(batch_result),
        description="batch portfolio report",
    )
    _write_private_text(
        json_path,
        json.dumps(build_batch_evaluation_export(batch_result), indent=2, sort_keys=True),
        description="batch portfolio JSON export",
    )
    return batch_result


def build_batch_evaluation_export(result: BatchEvaluationResult) -> dict[str, object]:
    return {
        "schema_version": "1",
        "batch": {
            "root_folder": str(result.root_folder),
            "created_at": result.created_at.isoformat(),
            "deal_count": result.deal_count,
            "evaluated_count": result.evaluated_count,
            "failed_count": result.failed_count,
            "allocated_count": result.allocated_count,
            "skipped_count": result.skipped_count,
        },
        "portfolio_constraints": {
            "starting_capital": result.constraints.starting_capital,
            "reserve_amount": result.constraints.reserve_amount,
            "allocatable_capital": result.constraints.allocatable_capital,
            "existing_invested_capital": result.constraints.existing_invested_capital,
            "available_capital_before_batch": (
                result.constraints.available_capital_before_batch
            ),
            "new_allocated_capital": result.constraints.new_allocated_capital,
            "remaining_allocatable_capital": (
                result.constraints.remaining_allocatable_capital
            ),
            "allowed_check_sizes": result.constraints.allowed_check_sizes,
            "configured_min_check": result.constraints.configured_min_check,
            "configured_max_check": result.constraints.configured_max_check,
            "max_company_exposure_percent": (
                result.constraints.max_company_exposure_percent
            ),
            "max_category_exposure_percent": (
                result.constraints.max_category_exposure_percent
            ),
            "max_stage_exposure_percent": result.constraints.max_stage_exposure_percent,
            "max_low_confidence_exposure_percent": (
                result.constraints.max_low_confidence_exposure_percent
            ),
            "max_medium_confidence_exposure_percent": (
                result.constraints.max_medium_confidence_exposure_percent
            ),
            "max_high_confidence_exposure_percent": (
                result.constraints.max_high_confidence_exposure_percent
            ),
        },
        "deals": [
            {
                "company_name": row.company_name,
                "deal_id": row.deal_id,
                "evaluation_status": row.evaluation_status,
                "final_recommendation": row.final_recommendation.value,
                "single_deal_check_size": row.single_deal_check_size,
                "batch_check_size": row.batch_check_size,
                "score": row.score,
                "max_score": row.max_score,
                "confidence": row.confidence,
                "portfolio_rank": row.portfolio_rank,
                "allocation_sequence": row.allocation_sequence,
                "capital_before": row.capital_before,
                "capital_after": row.capital_after,
                "key_blockers": list(row.key_blockers),
                "skipped_reason": row.skipped_reason,
                "failure_reason": row.failure_reason,
                "artifacts": {
                    "memo_path": str(row.memo_path) if row.memo_path is not None else None,
                    "json_path": str(row.json_path) if row.json_path is not None else None,
                },
            }
            for row in result.allocation_rows
        ],
        "artifacts": {
            "batch_report_path": str(result.report_path),
            "batch_json_path": str(result.json_path),
        },
        "privacy": {
            "contains_raw_evidence_text": False,
            "contains_model_excerpts": False,
            "evidence_lineage": (
                "Batch artifacts include child artifact paths and evidence IDs only "
                "through child exports. They do not copy source evidence text."
            ),
        },
    }


def render_batch_portfolio_report(result: BatchEvaluationResult) -> str:
    constraints = result.constraints
    lines = [
        "# Hail Mary Batch Portfolio Evaluation",
        "",
        "## Batch Summary",
        "",
        f"- Root folder: {_markdown_text(str(result.root_folder))}",
        f"- Deals found: {result.deal_count}",
        f"- Deals evaluated: {result.evaluated_count}",
        f"- Deal-level failures: {result.failed_count}",
        f"- New allocated checks: {_format_dollars(constraints.new_allocated_capital)}",
        "- Remaining allocatable capital: "
        f"{_format_dollars(constraints.remaining_allocatable_capital)}",
        "",
        "## Portfolio Constraints",
        "",
        f"- Starting capital budget: {_format_dollars(constraints.starting_capital)}",
        f"- Reserve: {_format_dollars(constraints.reserve_amount)}",
        f"- Allocatable capital after reserve: {_format_dollars(constraints.allocatable_capital)}",
        "- Recorded existing investments: "
        f"{_format_dollars(constraints.existing_invested_capital)}",
        "- Available capital before batch: "
        f"{_format_dollars(constraints.available_capital_before_batch)}",
        f"- Allowed check sizes: {_check_size_list(constraints.allowed_check_sizes)}",
        f"- Configured minimum check: {_format_check_size(constraints.configured_min_check)}",
        f"- Configured maximum check: {_format_check_size(constraints.configured_max_check)}",
        f"- Company exposure cap: {constraints.max_company_exposure_percent}%",
        f"- Category exposure cap: {constraints.max_category_exposure_percent}%",
        f"- Stage exposure cap: {constraints.max_stage_exposure_percent}%",
        f"- Low-confidence exposure cap: {constraints.max_low_confidence_exposure_percent}%",
        f"- Medium-confidence exposure cap: {constraints.max_medium_confidence_exposure_percent}%",
        f"- High-confidence exposure cap: {constraints.max_high_confidence_exposure_percent}%",
        "",
        "## Allocation View",
        "",
    ]
    if not result.allocation_rows:
        lines.append("No deals were available for allocation.")
    else:
        lines.extend(
            [
                "| Rank | Company | Status | Final recommendation | Single-deal check | "
                "Batch check | Score | Confidence | Key blockers | Memo | JSON |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in result.allocation_rows:
            lines.append(
                "| "
                f"{row.portfolio_rank or ''} | "
                f"{_markdown_text(row.company_name)} | "
                f"{_markdown_text(row.evaluation_status)} | "
                f"{row.final_recommendation.value} | "
                f"{_format_check_size(row.single_deal_check_size)} | "
                f"{_format_check_size(row.batch_check_size)} | "
                f"{_score_text(row)} | "
                f"{_markdown_text(row.confidence or 'unknown')} | "
                f"{_markdown_text(_blocker_text(row.key_blockers))} | "
                f"{_markdown_text(str(row.memo_path) if row.memo_path else 'not written')} | "
                f"{_markdown_text(str(row.json_path) if row.json_path else 'not written')} |"
            )

    lines.extend(["", "## Skipped Or Failed Deals", ""])
    skipped_rows = [row for row in result.allocation_rows if not row.allocated]
    if not skipped_rows:
        lines.append("No deals were skipped or failed.")
    else:
        lines.extend(
            [
                "| Company | Status | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for row in skipped_rows:
            reason = (
                row.failure_reason
                or row.skipped_reason
                or "No nonzero batch check was allocated."
            )
            lines.append(
                "| "
                f"{_markdown_text(row.company_name)} | "
                f"{_markdown_text(row.evaluation_status)} | "
                f"{_markdown_text(reason)} |"
            )

    lines.extend(
        [
            "",
            "This report is a diligence aid, not legal, tax, financial, or investment advice.",
            "",
        ]
    )
    return "\n".join(lines)


def _allocate_batch(
    outcomes: Sequence[BatchDealOutcome],
    *,
    config: AppConfig,
) -> tuple[list[BatchAllocationRow], BatchPortfolioConstraints]:
    try:
        status = portfolio_status(config)
    except PortfolioError as exc:
        raise BatchEvaluationError(
            f"Could not read the private portfolio ledger: {exc}"
        ) from exc

    remaining_capital = status.available_capital
    base_exposure_state = portfolio_exposure_state_from_ledger(status.ledger, config=config)
    exposure_state = base_exposure_state
    rows_by_folder: dict[Path, BatchAllocationRow] = {}
    successful_outcomes = [outcome for outcome in outcomes if outcome.result is not None]
    ranking_scores: dict[Path, ScoredDeal] = {}
    stores_by_folder: dict[Path, EvidenceStore] = {}
    failed_success_rows: dict[Path, BatchAllocationRow] = {}
    ranking_capital = max(status.available_capital, config.max_check)
    for outcome in successful_outcomes:
        result = outcome.result
        if result is None:
            continue
        try:
            store = _load_actioned_store_for_result(result, config=config)
        except BatchEvaluationError as exc:
            failed_success_rows[outcome.folder] = BatchAllocationRow(
                company_name=result.company_name,
                deal_id=result.deal_id,
                folder=outcome.folder,
                evaluation_status="failed",
                final_recommendation=Recommendation.PASS,
                single_deal_check_size=0,
                batch_check_size=0,
                score=result.deterministic_score.total_score,
                max_score=result.deterministic_score.max_score,
                confidence=result.deterministic_score.confidence.value,
                key_blockers=[str(exc)],
                memo_path=result.final_memo_path,
                json_path=result.final_json_path,
                skipped_reason=None,
                failure_reason=str(exc),
            )
            continue
        stores_by_folder[outcome.folder] = store
        ranking_scores[outcome.folder] = score_evidence_store(
            store,
            config=config,
            capital_remaining=ranking_capital,
            exposure_state=base_exposure_state,
        )
    ranked_successes = sorted(
        [outcome for outcome in successful_outcomes if outcome.folder in ranking_scores],
        key=lambda outcome: portfolio_rank_key(ranking_scores[outcome.folder]),
    )
    portfolio_ranks = {
        outcome.folder: rank for rank, outcome in enumerate(ranked_successes, start=1)
    }
    for folder, row in failed_success_rows.items():
        rows_by_folder[folder] = row

    allocation_sequence = 0
    for outcome in ranked_successes:
        result = outcome.result
        if result is None:
            continue
        base_row = _row_from_success(
            outcome,
            portfolio_rank=portfolio_ranks[outcome.folder],
            ranking_score=ranking_scores[outcome.folder],
        )
        if not _eligible_for_batch_allocation(result):
            rows_by_folder[outcome.folder] = base_row
            continue
        store = stores_by_folder[outcome.folder]

        reallocated_score = score_evidence_store(
            store,
            config=config,
            capital_remaining=remaining_capital,
            exposure_state=exposure_state,
        )
        if reallocated_score.recommendation == Recommendation.INVEST:
            allocation_sequence += 1
            remaining_capital = reallocated_score.capital_remaining_after or 0
            exposure_state = portfolio_exposure_state_after_score(
                exposure_state,
                reallocated_score,
            )
            rows_by_folder[outcome.folder] = BatchAllocationRow(
                company_name=result.company_name,
                deal_id=result.deal_id,
                folder=outcome.folder,
                evaluation_status="allocated",
                final_recommendation=result.final_recommendation.recommendation,
                single_deal_check_size=result.final_recommendation.check_size,
                batch_check_size=reallocated_score.check_size,
                score=reallocated_score.total_score,
                max_score=reallocated_score.max_score,
                confidence=reallocated_score.confidence.value,
                key_blockers=_key_blockers(reallocated_score),
                memo_path=result.final_memo_path,
                json_path=result.final_json_path,
                allocation_sequence=allocation_sequence,
                portfolio_rank=portfolio_ranks[outcome.folder],
                capital_before=reallocated_score.capital_remaining_before,
                capital_after=reallocated_score.capital_remaining_after,
            )
            continue

        rows_by_folder[outcome.folder] = BatchAllocationRow(
            company_name=result.company_name,
            deal_id=result.deal_id,
            folder=outcome.folder,
            evaluation_status="skipped",
            final_recommendation=result.final_recommendation.recommendation,
            single_deal_check_size=result.final_recommendation.check_size,
            batch_check_size=0,
            score=reallocated_score.total_score,
            max_score=reallocated_score.max_score,
            confidence=reallocated_score.confidence.value,
            key_blockers=_key_blockers(reallocated_score),
            memo_path=result.final_memo_path,
            json_path=result.final_json_path,
            portfolio_rank=portfolio_ranks[outcome.folder],
            capital_before=reallocated_score.capital_remaining_before,
            capital_after=reallocated_score.capital_remaining_after,
            skipped_reason=skip_reason(reallocated_score),
        )

    for outcome in outcomes:
        if outcome.result is not None:
            continue
        rows_by_folder[outcome.folder] = BatchAllocationRow(
            company_name=outcome.company_name,
            deal_id=None,
            folder=outcome.folder,
            evaluation_status="failed",
            final_recommendation=Recommendation.PASS,
            single_deal_check_size=0,
            batch_check_size=0,
            score=None,
            max_score=None,
            confidence=None,
            key_blockers=[outcome.failure_reason or "The deal evaluation failed."],
            failure_reason=outcome.failure_reason or "The deal evaluation failed.",
        )

    ordered_rows = sorted(
        rows_by_folder.values(),
        key=lambda row: (
            1 if row.portfolio_rank is None else 0,
            row.portfolio_rank or 0,
            row.company_name.casefold(),
        ),
    )
    allocated_capital = sum(row.batch_check_size for row in ordered_rows)
    scenario = portfolio_scenario(config)
    constraints = BatchPortfolioConstraints(
        starting_capital=scenario.starting_capital,
        reserve_amount=scenario.reserve_amount,
        allocatable_capital=scenario.allocatable_capital,
        existing_invested_capital=status.invested_amount,
        available_capital_before_batch=status.available_capital,
        new_allocated_capital=allocated_capital,
        remaining_allocatable_capital=max(0, status.available_capital - allocated_capital),
        allowed_check_sizes=list(status.allowed_check_tiers),
        configured_min_check=config.min_check,
        configured_max_check=config.max_check,
        max_company_exposure_percent=str(config.max_company_exposure_percent),
        max_category_exposure_percent=str(config.max_category_exposure_percent),
        max_stage_exposure_percent=str(config.max_stage_exposure_percent),
        max_low_confidence_exposure_percent=str(config.max_low_confidence_exposure_percent),
        max_medium_confidence_exposure_percent=str(
            config.max_medium_confidence_exposure_percent
        ),
        max_high_confidence_exposure_percent=str(config.max_high_confidence_exposure_percent),
    )
    return ordered_rows, constraints


def _row_from_success(
    outcome: BatchDealOutcome,
    *,
    portfolio_rank: int,
    ranking_score: ScoredDeal,
) -> BatchAllocationRow:
    result = outcome.result
    if result is None:
        raise BatchEvaluationError("Internal error: successful row was missing a result.")
    scored = result.deterministic_score
    skipped_reason = None
    if scored.recommendation == Recommendation.PASS:
        skipped_reason = skip_reason(scored)
    elif result.final_recommendation.recommendation == Recommendation.PASS:
        skipped_reason = "Final recommendation was PASS."
    elif result.final_recommendation.check_size <= 0:
        skipped_reason = "Final recommendation did not include a nonzero check."
    return BatchAllocationRow(
        company_name=result.company_name,
        deal_id=result.deal_id,
        folder=outcome.folder,
        evaluation_status="skipped",
        final_recommendation=result.final_recommendation.recommendation,
        single_deal_check_size=result.final_recommendation.check_size,
        batch_check_size=0,
        score=ranking_score.total_score,
        max_score=ranking_score.max_score,
        confidence=ranking_score.confidence.value,
        key_blockers=_key_blockers(scored),
        memo_path=result.final_memo_path,
        json_path=result.final_json_path,
        portfolio_rank=portfolio_rank,
        capital_before=scored.capital_remaining_before,
        capital_after=scored.capital_remaining_after,
        skipped_reason=skipped_reason or "Batch allocation did not select this deal.",
    )


def _eligible_for_batch_allocation(result: DealEvaluationResult) -> bool:
    return (
        result.final_recommendation.recommendation == Recommendation.INVEST
        and result.final_recommendation.check_size > 0
        and result.deterministic_score.recommendation == Recommendation.INVEST
        and result.deterministic_score.check_size > 0
    )


def _load_actioned_store_for_result(
    result: DealEvaluationResult,
    *,
    config: AppConfig,
) -> EvidenceStore:
    store_path = config.data_dir / "processed" / "deals" / result.deal_id / "evidence_store.json"
    resolved_data_dir = _absolute_path(config.data_dir).resolve(strict=False)
    resolved_store_path = _absolute_path(store_path).resolve(strict=False)
    try:
        resolved_store_path.relative_to(resolved_data_dir)
    except ValueError:
        raise BatchEvaluationError(
            f"The evidence store path for {result.company_name} is outside the "
            "private data directory."
        ) from None
    try:
        raw_store = resolved_store_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BatchEvaluationError(
            f"The evidence store for {result.company_name} is not plain text."
        ) from exc
    except OSError as exc:
        raise BatchEvaluationError(
            f"Could not read the evidence store for {result.company_name}: {exc}"
        ) from exc
    try:
        store = EvidenceStore.model_validate_json(raw_store)
    except ValidationError as exc:
        detail = _validation_error_detail(exc)
        raise BatchEvaluationError(
            f"The evidence store for {result.company_name} could not be read. "
            f"First problem: {detail}"
        ) from exc
    try:
        return apply_evidence_actions(config=config, store=store).store
    except EvidenceActionError as exc:
        raise BatchEvaluationError(
            f"Could not apply local evidence actions for {result.company_name}: {exc}"
        ) from exc


def _resolve_batch_root(root_folder: Path) -> Path:
    expanded = root_folder.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise BatchEvaluationError("The batch root folder cannot be a symlink.")
    resolved = absolute.resolve(strict=False)
    if not resolved.exists():
        raise BatchEvaluationError(f"The batch root folder does not exist: {root_folder}")
    if not resolved.is_dir():
        raise BatchEvaluationError(f"This is not a folder: {root_folder}")
    return resolved


def _discover_deal_folders(root: Path, *, config: AppConfig) -> list[Path]:
    resolved_data_dir = _absolute_path(config.data_dir).resolve(strict=False)
    resolved_raw_dir = (resolved_data_dir / "raw").resolve(strict=False)
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.casefold())
    except OSError as exc:
        raise BatchEvaluationError(f"Could not list company folders in {root}: {exc}") from exc
    deal_folders: list[Path] = []
    for child in children:
        if child.name.startswith("."):
            continue
        resolved_child = child.resolve(strict=False)
        under_data_dir = _is_relative_to(resolved_child, resolved_data_dir)
        under_raw_child = (
            _is_relative_to(resolved_child, resolved_raw_dir)
            and resolved_child != resolved_raw_dir
        )
        if under_data_dir and not under_raw_child:
            continue
        if child.is_symlink():
            deal_folders.append(child)
            continue
        if not child.is_dir():
            continue
        deal_folders.append(child)
    return deal_folders


def _key_blockers(scored: ScoredDeal) -> list[str]:
    blockers: list[str] = []
    for gate in scored.triggered_kill_gates:
        _append_unique(blockers, f"Guardrail: {gate.name}")
    if scored.recommendation == Recommendation.PASS and scored.total_score < 75:
        _append_unique(blockers, "Score below INVEST threshold")
    for factor in scored.score_factors:
        for missing_input in factor.missing_inputs:
            _append_unique(blockers, f"Missing input: {missing_input}")
            if len(blockers) >= 3:
                return blockers
    for missing_input in scored.net_return.missing_inputs:
        _append_unique(blockers, f"Missing input: {missing_input}")
        if len(blockers) >= 3:
            return blockers
    if not blockers and scored.recommendation == Recommendation.INVEST:
        blockers.append("No blocking portfolio guardrails")
    return blockers[:3]


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _ensure_private_directory(path: Path, *, private_root: Path, description: str) -> None:
    resolved_root = _absolute_path(private_root).resolve(strict=False)
    _reject_output_symlink_escape(path, resolved_root=resolved_root, description=description)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BatchEvaluationError(
            f"Could not create private {description} folder at {path}: {exc}"
        ) from exc

    resolved_path = path.resolve(strict=False)
    relative_parts = resolved_path.relative_to(resolved_root).parts
    directories = [resolved_root]
    current = resolved_root
    for part in relative_parts:
        current = current / part
        directories.append(current)

    for directory in directories:
        try:
            directory.chmod(0o700)
        except OSError as exc:
            raise BatchEvaluationError(
                f"Could not make private {description} folder at {directory}: {exc}"
            ) from exc


def _reject_output_symlink_escape(
    path: Path,
    *,
    resolved_root: Path,
    description: str,
) -> None:
    absolute_path = _absolute_path(path).resolve(strict=False)
    try:
        relative_parts = absolute_path.relative_to(resolved_root).parts
    except ValueError:
        raise BatchEvaluationError(
            f"The private {description} folder {path} is outside the data directory."
        ) from None

    current = resolved_root
    if current.is_symlink():
        raise BatchEvaluationError(
            f"The private {description} folder {path} uses a symlinked data directory."
        )

    for part in relative_parts:
        current = current / part
        if not current.is_symlink():
            continue
        try:
            current.resolve(strict=True).relative_to(resolved_root)
        except ValueError:
            raise BatchEvaluationError(
                f"The private {description} folder {path} resolves outside the data directory."
            ) from None


def _write_private_text(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise BatchEvaluationError(
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
        raise BatchEvaluationError(
            f"Could not write {description} at {path}: the text cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise BatchEvaluationError(f"Could not write {description} at {path}: {exc}") from exc


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validation_error_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "document: Invalid structured JSON."
    first_error = errors[0]
    location = first_error.get("loc", ())
    location_text = ".".join(str(part) for part in location) if location else "document"
    message = first_error.get("msg", "Invalid value.")
    return f"{location_text}: {message}"


def _format_check_size(check_size: int) -> str:
    if check_size == 0:
        return "$0"
    if check_size % 1_000 == 0:
        return f"${check_size // 1_000}K"
    return f"${check_size / 1_000:g}K"


def _format_dollars(amount: int) -> str:
    return f"${amount:,}"


def _check_size_list(values: Sequence[int]) -> str:
    return ", ".join(_format_check_size(value) for value in values)


def _score_text(row: BatchAllocationRow) -> str:
    if row.score is None or row.max_score is None:
        return "unknown"
    return f"{row.score}/{row.max_score}"


def _blocker_text(blockers: Sequence[str]) -> str:
    if not blockers:
        return "none"
    return "; ".join(blockers)


def _markdown_text(value: str) -> str:
    collapsed = " ".join(value.split())
    markdown_characters = "\\`*_{}[]()#+!|>"
    markdown_escaped = "".join(
        f"\\{character}" if character in markdown_characters else character
        for character in collapsed
    )
    return html.escape(markdown_escaped, quote=True)
