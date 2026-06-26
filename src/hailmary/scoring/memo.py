from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from pathlib import Path

from pydantic import ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.portfolio import portfolio_status
from hailmary.portfolio.scenario import (
    allowed_check_tiers_for_available_capital,
    portfolio_scenario,
)
from hailmary.schemas.documents import IngestionSummary
from hailmary.schemas.evidence import ClaimRecord, EvidenceRecord, EvidenceStore
from hailmary.schemas.scoring import MemoRunSummary, ScoredDeal
from hailmary.scoring.portfolio import (
    allowed_check_tiers,
    portfolio_rank_key,
    portfolio_return_cases,
    skipped_deals,
)
from hailmary.scoring.portfolio import (
    ranked_deals as ranked_portfolio_deals,
)
from hailmary.scoring.scorer import (
    score_evidence_store,
    validated_conflicts,
    validated_verified_claims,
)
from hailmary.utils.slug import slugify


class ScoringError(RuntimeError):
    """Scoring could not continue safely."""


PORTFOLIO_REPORT_FILENAME = "portfolio-comparison-report.md"
MAX_FIXED_DECIMAL_REPORT_CHARS = 120


def score_latest_ingestion(*, config: AppConfig) -> MemoRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ScoringError(str(exc)) from exc

    summary_path = config.data_dir / "processed" / "ingestion_summary.json"
    if not summary_path.exists():
        raise ScoringError(
            "No ingested deals were found. Run `hailmary ingest-folder` before scoring."
        )

    summary = _load_ingestion_summary(summary_path)
    report_dir = config.data_dir / "reports"
    _ensure_private_directory(report_dir, private_root=config.data_dir)

    status = portfolio_status(config)
    scoring_inputs: list[tuple[EvidenceStore, ScoredDeal, Path]] = []
    for deal in summary.deals:
        if deal.evidence_store_path is None:
            raise ScoringError(
                f"No evidence store was found for {deal.company_name}. "
                "Run `hailmary ingest-folder` again before scoring."
            )
        evidence_store_path = _resolve_saved_path(
            deal.evidence_store_path,
            data_dir=config.data_dir,
            summary_path=summary_path,
        )
        if not evidence_store_path.exists():
            raise ScoringError(
                f"The evidence store for {deal.company_name} is missing at "
                f"{evidence_store_path}. Run `hailmary ingest-folder` again."
            )
        store = _load_evidence_store(evidence_store_path, company_name=deal.company_name)
        ranking_scored_deal = score_evidence_store(
            store,
            config=config,
            capital_remaining=max(status.available_capital, config.max_check),
        )
        memo_path = report_dir / f"{slugify(deal.company_name)}-{deal.id}-memo.md"
        scoring_inputs.append((store, ranking_scored_deal, memo_path))

    remaining_capital = status.available_capital
    scored_by_index: dict[int, ScoredDeal] = {}
    portfolio_rank_by_index: dict[int, int] = {}
    ranked_inputs = sorted(
        enumerate(scoring_inputs),
        key=lambda item: portfolio_rank_key(item[1][1]),
    )
    for portfolio_rank, (index, (store, _, _)) in enumerate(ranked_inputs, start=1):
        scored_deal = score_evidence_store(
            store,
            config=config,
            capital_remaining=remaining_capital,
        )
        remaining_capital = scored_deal.capital_remaining_after or 0
        scored_by_index[index] = scored_deal
        portfolio_rank_by_index[index] = portfolio_rank

    scored_deals: list[ScoredDeal] = []
    for index, (store, _, memo_path) in enumerate(scoring_inputs):
        scored_deal = scored_by_index[index]
        _write_private_text(
            memo_path,
            render_markdown_memo(scored_deal, store),
            description="Markdown memo",
        )
        scored_deals.append(
            scored_deal.model_copy(
                update={
                    "memo_path": memo_path,
                    "portfolio_rank": portfolio_rank_by_index[index],
                }
            )
        )

    portfolio_report_path = report_dir / PORTFOLIO_REPORT_FILENAME
    _write_private_text(
        portfolio_report_path,
        render_portfolio_report(
            scored_deals,
            config=config,
            existing_invested_capital=status.invested_amount,
        ),
        description="portfolio comparison report",
    )

    return MemoRunSummary(
        report_dir=report_dir,
        scored_deals=scored_deals,
        portfolio_report_path=portfolio_report_path,
    )


def render_markdown_memo(scored_deal: ScoredDeal, store: EvidenceStore) -> str:
    verified_claims = _verified_claims(store)
    round_summary = _memo_metadata_value(_round_summary(verified_claims))
    valuation_summary = _memo_metadata_value(_valuation_summary(verified_claims))
    lines = [
        f"# Hail Mary Investment Memo: {_memo_metadata_value(scored_deal.company_name)}",
        "",
        "## Decision",
        "",
        f"**Recommendation:** {scored_deal.recommendation}",
        f"**Suggested check:** {_format_check_size(scored_deal.check_size)}",
        f"**Score:** {scored_deal.total_score}/{scored_deal.max_score}",
        f"**Confidence:** {scored_deal.confidence}",
        f"**One-line reason:** {_memo_metadata_value(scored_deal.one_line_reason)}",
        "**Deadline:** unknown",
        f"**Round / Instrument:** {round_summary} / unknown",
        f"**Valuation / Cap:** {valuation_summary}",
        f"**Stage:** {_memo_metadata_value(str(scored_deal.company_stage))}",
        f"**Product-market fit:** {_memo_metadata_value(str(scored_deal.pmf_level))}",
        f"**Fundability risk:** {_memo_metadata_value(str(scored_deal.fundability_risk))}",
        f"**Valuation risk:** {_memo_metadata_value(str(scored_deal.valuation_risk))}",
        f"**Net return math:** {_memo_metadata_value(_net_return_summary(scored_deal))}",
        "",
        "## Kill Gates",
    ]
    for gate in scored_deal.kill_gates:
        status = "TRIGGERED" if gate.triggered else "Clear"
        lines.append(
            f"- {status}: {_memo_metadata_value(gate.name)}. "
            f"{_memo_metadata_value(gate.reason)}"
            f"{_support_text(gate.support_status)}"
            f"{_evidence_reference_text(gate.evidence_ids)}"
        )

    lines.extend(["", "## Score Factors"])
    for factor in scored_deal.score_factors:
        evidence_text = _evidence_reference_text(factor.evidence_ids)
        missing_text = _missing_input_text(factor.missing_inputs)
        lines.append(
            f"- {_memo_metadata_value(factor.name)}: {factor.score}/{factor.max_score}. "
            f"{_memo_metadata_value(factor.explanation)}"
            f"{_support_text(factor.support_status)}"
            f"{missing_text}{evidence_text}"
        )

    lines.extend(["", "## Verified Deal Terms"])
    if verified_claims:
        for claim in verified_claims:
            citation_ids = ", ".join(
                _memo_metadata_value(citation.evidence_id)
                for citation in claim.citations
            )
            lines.append(
                f"- {_memo_metadata_value(claim.label)}: {_memo_metadata_value(claim.value)} "
                f"(evidence: {citation_ids or 'none'})."
            )
    else:
        lines.append("- No verified deal-term claims were available.")

    lines.extend(["", "## Diligence Questions"])
    for question in scored_deal.diligence_questions:
        lines.append(
            f"{question.priority}. {_memo_metadata_value(question.question)} "
            f"Reason: {_memo_metadata_value(question.reason)}"
            f"{_support_text(question.support_status)}"
            f"{_evidence_reference_text(question.evidence_ids)}"
        )

    lines.extend(["", "## Evidence Used"])
    if store.evidence:
        for evidence in _memo_evidence_records(store, scored_deal, verified_claims):
            lines.append(_evidence_line(evidence))
    else:
        lines.append("- No source-linked evidence records were available.")

    lines.extend(
        [
            "",
            "This memo is a diligence aid, not legal, tax, financial, or investment advice.",
            "",
        ]
    )
    return "\n".join(lines)


def render_portfolio_report(
    scored_deals: list[ScoredDeal],
    *,
    config: AppConfig,
    existing_invested_capital: int = 0,
) -> str:
    scenario = portfolio_scenario(config)
    ranked_deals = ranked_portfolio_deals(scored_deals)
    new_allocated_capital = sum(deal.check_size for deal in scored_deals)
    available_for_new_checks = max(0, scenario.allocatable_capital - existing_invested_capital)
    remaining_capital = max(0, available_for_new_checks - new_allocated_capital)
    return_math_capital = existing_invested_capital + new_allocated_capital
    lines = [
        "# Hail Mary Portfolio Comparison Report",
        "",
        "## Portfolio Scenario And Constraints",
        "",
        f"- Starting capital budget: {_format_dollars(scenario.starting_capital)}",
        f"- Reserve: {_format_dollars(scenario.reserve_amount)} ({_reserve_source_text(config)})",
        f"- Allocatable capital after reserve: {_format_dollars(scenario.allocatable_capital)}",
        f"- Recorded existing investments: {_format_dollars(existing_invested_capital)}",
        f"- Allocatable capital for new checks: {_format_dollars(available_for_new_checks)}",
        f"- Allocated capital: {_format_dollars(new_allocated_capital)}",
        f"- Remaining allocatable capital after allocation: {_format_dollars(remaining_capital)}",
        f"- Allowed check sizes: {_check_tier_text(config, available_for_new_checks)}",
        f"- Configured minimum check: {_format_check_size(scenario.min_check)}",
        f"- Configured maximum check: {_format_check_size(scenario.max_check)}",
        f"- Estimated dilution: {_format_percent(scenario.estimated_dilution_percent)}",
        f"- Platform fee: {_format_percent(scenario.platform_fee_percent)}",
        f"- Carry: {_format_percent(scenario.carry_percent)}",
        f"- Gross return multiple: {_format_multiple(scenario.gross_return_multiple)}",
        "- Carry means the share of profits paid to the fund manager or platform.",
        "- Dilution means ownership reduction from future fundraising.",
        "",
        "## Ranked Deals",
        "",
    ]

    if not ranked_deals:
        lines.append("No scored deals were available.")
    else:
        lines.extend(
            [
                "| Rank | Company | Recommendation | Check size | Score | Confidence | "
                "Triggered kill gates | Diligence questions | Budget before | Budget after |",
                "| ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, deal in enumerate(ranked_deals, start=1):
            lines.append(
                "| "
                f"{rank} | "
                f"{_memo_metadata_value(deal.company_name)} | "
                f"{_memo_metadata_value(str(deal.recommendation))} | "
                f"{_format_check_size(deal.check_size)} | "
                f"{deal.total_score}/{deal.max_score} | "
                f"{_memo_metadata_value(str(deal.confidence))} | "
                f"{_triggered_gate_summary(deal)} | "
                f"{len(deal.diligence_questions)} | "
                f"{_optional_check_size(deal.capital_remaining_before)} | "
                f"{_optional_check_size(deal.capital_remaining_after)} |"
            )

    lines.extend(["", "## Skipped Deals", ""])
    skipped = skipped_deals(scored_deals)
    if not skipped:
        lines.append("No deals were skipped.")
    else:
        lines.extend(
            [
                "| Rank | Company | Score | Reason |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for skipped_deal in skipped:
            lines.append(
                "| "
                f"{skipped_deal.rank} | "
                f"{_memo_metadata_value(skipped_deal.company_name)} | "
                f"{skipped_deal.score_text} | "
                f"{_memo_metadata_value(skipped_deal.reason)} |"
            )

    lines.extend(["", "## Net Return Math", ""])
    lines.extend(
        _portfolio_return_lines(
            return_math_capital,
            config=config,
            includes_existing=existing_invested_capital > 0,
        )
    )

    lines.extend(["", "## Deal Details"])
    if not ranked_deals:
        lines.append("")
        lines.append("No scored deal details were available.")
    for deal in ranked_deals:
        lines.extend(
            [
                "",
                f"### {_memo_metadata_value(deal.company_name)}",
                "",
                f"- Deal ID: {_memo_metadata_value(deal.deal_id)}",
                f"- Recommendation: {_memo_metadata_value(str(deal.recommendation))}",
                f"- Suggested check: {_format_check_size(deal.check_size)}",
                f"- Score: {deal.total_score}/{deal.max_score}",
                f"- Confidence: {_memo_metadata_value(str(deal.confidence))}",
                f"- One-line reason: {_memo_metadata_value(deal.one_line_reason)}",
                f"- Memo path: {_memo_metadata_value(str(deal.memo_path or 'not written'))}",
                "- Key risks:",
            ]
        )
        for risk in _portfolio_key_risk_lines(deal):
            lines.append(f"  - {risk}")

        lines.append("- Kill gates:")
        for gate in deal.kill_gates:
            status = "TRIGGERED" if gate.triggered else "Clear"
            lineage_label = "NEEDS_DILIGENCE" if gate.triggered else "INFERRED"
            lines.append(
                "  - "
                f"{status} ({lineage_label}): {_memo_metadata_value(gate.name)}. "
                f"{_memo_metadata_value(gate.reason)}"
            )

        lines.append("- Diligence questions:")
        for question in deal.diligence_questions:
            lines.append(
                "  - "
                f"Priority {question.priority}: {_memo_metadata_value(question.question)} "
                f"Reason: {_memo_metadata_value(question.reason)}"
                f"{_portfolio_evidence_text(question.evidence_ids)}"
            )

    lines.extend(
        [
            "",
            "This report is a diligence aid, not legal, tax, financial, or investment advice.",
            "",
        ]
    )
    return "\n".join(lines)


def _check_tier_text(config: AppConfig, available_capital: int | None = None) -> str:
    allowed_tiers = (
        allowed_check_tiers(config)
        if available_capital is None
        else allowed_check_tiers_for_available_capital(
            config,
            available_capital=available_capital,
        )
    )
    return ", ".join(_format_check_size(tier) for tier in allowed_tiers)


def _reserve_source_text(config: AppConfig) -> str:
    if config.reserve_dollars > 0:
        return "configured reserve dollars"
    if config.reserve_percent > 0:
        return f"{_format_percent(config.reserve_percent)} reserve"
    return "no reserve"


def _portfolio_return_lines(
    invested_capital: int,
    *,
    config: AppConfig,
    includes_existing: bool = False,
) -> list[str]:
    scope_text = (
        "Net return math uses recorded investments plus newly allocated checks and "
        "excludes unallocated reserve capital."
        if includes_existing
        else "Net return math uses allocated checks only and excludes unallocated reserve capital."
    )
    lines = [
        scope_text,
        "",
        "| Case | Gross multiple | Invested checks | Gross value before dilution | "
        "Value after dilution | Platform fee | Carry | Net cash returned | "
        "Net profit after fees | Net multiple |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for return_case in portfolio_return_cases(
        invested_capital=invested_capital,
        config=config,
    ):
        lines.append(
            "| "
            f"{_memo_metadata_value(return_case.label)} | "
            f"{_format_multiple(return_case.gross_return_multiple)} | "
            f"{_format_dollars(return_case.invested_capital)} | "
            f"{_format_dollars(return_case.gross_value_before_dilution)} | "
            f"{_format_dollars(return_case.value_after_dilution)} | "
            f"{_format_dollars(return_case.platform_fee)} | "
            f"{_format_dollars(return_case.carry)} | "
            f"{_format_dollars(return_case.net_cash_returned)} | "
            f"{_format_dollars(return_case.net_profit_after_fees)} | "
            f"{_format_net_multiple(return_case.net_multiple)} |"
        )
    if invested_capital == 0:
        lines.append("")
        lines.append("No capital was allocated, so every return case starts from $0 invested.")
    return lines


def _optional_check_size(check_size: int | None) -> str:
    if check_size is None:
        return "unknown"
    return _format_check_size(check_size)


def _triggered_gate_summary(deal: ScoredDeal) -> str:
    if not deal.triggered_kill_gates:
        return "None"
    return "; ".join(_memo_metadata_value(gate.name) for gate in deal.triggered_kill_gates)


def _portfolio_key_risk_lines(deal: ScoredDeal) -> list[str]:
    risks: list[str] = []
    for gate in deal.triggered_kill_gates:
        risks.append(
            "NEEDS_DILIGENCE: "
            f"{_memo_metadata_value(gate.name)}. {_memo_metadata_value(gate.reason)}"
            f"{_portfolio_evidence_text(gate.evidence_ids)}"
        )

    for factor in deal.score_factors:
        if factor.score >= factor.max_score:
            continue
        evidence_text = _portfolio_evidence_text(factor.evidence_ids)
        label = _support_label(factor.support_status)
        missing_text = _portfolio_missing_input_text(factor.missing_inputs)
        risks.append(
            f"{label}: {_memo_metadata_value(factor.name)} scored "
            f"{factor.score}/{factor.max_score}. "
            f"{_memo_metadata_value(factor.explanation)}{missing_text}{evidence_text}"
        )

    for question in deal.diligence_questions:
        risks.append(
            "NEEDS_DILIGENCE: "
            f"{_memo_metadata_value(question.question)} Reason: "
            f"{_memo_metadata_value(question.reason)}"
            f"{_portfolio_evidence_text(question.evidence_ids)}"
        )

    if not risks:
        risks.append(
            "INFERRED: No blocking deterministic risks were found; "
            "verify the source-linked deal memo before committing capital."
        )
    return risks


def _portfolio_evidence_text(evidence_ids: list[str]) -> str:
    if not evidence_ids:
        return ""
    formatted_evidence_ids = ", ".join(
        _memo_metadata_value(evidence_id) for evidence_id in evidence_ids
    )
    return f" Evidence: {formatted_evidence_ids}."


def _load_ingestion_summary(summary_path: Path) -> IngestionSummary:
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ScoringError(
            "The ingestion summary is not plain text. "
            "Run `hailmary ingest-folder` again before scoring."
        ) from exc
    except OSError as exc:
        raise ScoringError(
            f"Could not read the ingestion summary at {summary_path}: {exc}"
        ) from exc
    try:
        return IngestionSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise ScoringError(
            "The ingestion summary could not be read. "
            "Run `hailmary ingest-folder` again before scoring."
        ) from exc


def _load_evidence_store(path: Path, *, company_name: str) -> EvidenceStore:
    try:
        raw_store = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ScoringError(
            f"The evidence store for {company_name} is not plain text. "
            "Run `hailmary ingest-folder` again before scoring."
        ) from exc
    except OSError as exc:
        raise ScoringError(
            f"Could not read the evidence store for {company_name} at {path}: {exc}"
        ) from exc
    try:
        return EvidenceStore.model_validate_json(raw_store)
    except ValidationError as exc:
        raise ScoringError(
            f"The evidence store for {company_name} could not be read. "
            "Run `hailmary ingest-folder` again before scoring."
        ) from exc


def _resolve_saved_path(path: Path, *, data_dir: Path, summary_path: Path) -> Path:
    absolute_data_dir = _absolute_path(data_dir).resolve(strict=False)
    if path.is_absolute():
        resolved_path = path.resolve(strict=False)
        if not _is_relative_to(resolved_path, absolute_data_dir):
            raise ScoringError(
                f"The evidence store path {path} is outside the private data directory."
            )
        return resolved_path

    absolute_summary_path = _absolute_path(summary_path).resolve(strict=False)
    candidate_roots = [
        Path.cwd().resolve(strict=False),
        *absolute_data_dir.parents,
        absolute_data_dir,
        absolute_summary_path.parent,
    ]
    candidates = [root / path for root in candidate_roots]
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir) and candidate.exists():
            return resolved_candidate
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir):
            return resolved_candidate
    return (absolute_data_dir / path.name).resolve(strict=False)


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verified_claims(store: EvidenceStore) -> list[ClaimRecord]:
    return validated_verified_claims(store)


def _memo_evidence_records(
    store: EvidenceStore,
    scored_deal: ScoredDeal,
    verified_claims: list[ClaimRecord],
) -> list[EvidenceRecord]:
    cited_ids = _cited_evidence_ids(scored_deal, verified_claims, store)
    included_ids: set[str] = set()
    selected: list[EvidenceRecord] = []

    for evidence in store.evidence[:25]:
        selected.append(evidence)
        included_ids.add(evidence.id)
    for evidence in store.evidence:
        if evidence.id in cited_ids and evidence.id not in included_ids:
            selected.append(evidence)
            included_ids.add(evidence.id)
    return selected


def _cited_evidence_ids(
    scored_deal: ScoredDeal,
    verified_claims: list[ClaimRecord],
    store: EvidenceStore,
) -> set[str]:
    cited_ids = {
        evidence_id
        for factor in scored_deal.score_factors
        for evidence_id in factor.evidence_ids
    }
    cited_ids.update(
        citation.evidence_id
        for claim in verified_claims
        for citation in claim.citations
    )
    conflict_claim_ids = {
        claim_id for conflict in validated_conflicts(store) for claim_id in conflict.claim_ids
    }
    cited_ids.update(
        citation.evidence_id
        for claim in store.claims
        if claim.id in conflict_claim_ids
        for citation in claim.citations
    )
    cited_ids.update(
        evidence_id
        for question in scored_deal.diligence_questions
        for evidence_id in question.evidence_ids
    )
    return cited_ids


def _evidence_line(evidence: EvidenceRecord) -> str:
    locator = (
        f"page {evidence.page_number}"
        if evidence.page_number is not None
        else f"table {evidence.table_index}"
        if evidence.table_index is not None
        else "document"
    )
    external_details = _external_source_details(evidence)
    detail_text = f"; {external_details}" if external_details else ""
    document_path = _memo_metadata_value(str(evidence.document_path))
    return (
        f"- {evidence.id}: {document_path} "
        f"({locator}, {evidence.evidence_kind}{detail_text})."
    )


def _external_source_details(evidence: EvidenceRecord) -> str:
    details: list[str] = []
    if evidence.provider_name:
        details.append(f"provider: {_memo_metadata_value(evidence.provider_name)}")
    if evidence.source_url:
        details.append(f"source page: {_memo_metadata_value(evidence.source_url)}")
    if evidence.source_api:
        details.append(f"data service source: {_memo_metadata_value(evidence.source_api)}")
    if evidence.retrieved_at:
        details.append(f"retrieved at: {evidence.retrieved_at.isoformat()}")
    if evidence.external_confidence:
        details.append(f"confidence: {_memo_metadata_value(evidence.external_confidence)}")
    if evidence.licensing_notes:
        details.append(f"licensing: {_memo_metadata_value(evidence.licensing_notes)}")
    if evidence.ocr_applied:
        details.append(
            "text source: image-based text reading (OCR; OCR means reading text from images)"
        )
        if evidence.ocr_confidence is not None:
            ocr_confidence = _format_ocr_confidence(evidence.ocr_confidence)
            details.append(
                f"OCR confidence: {_memo_metadata_value(ocr_confidence)}"
            )
    return "; ".join(details)


def _format_ocr_confidence(confidence: float) -> str:
    return f"{confidence:.0%}"


def _memo_metadata_value(value: str) -> str:
    collapsed = " ".join(value.split())
    markdown_characters = "\\`*_{}[]()#+!|>"
    return "".join(
        f"\\{character}" if character in markdown_characters else character
        for character in collapsed
    )


def _round_summary(verified_claims: list[ClaimRecord]) -> str:
    round_size = _first_claim_value(verified_claims, labels={"round size"})
    if round_size is None:
        return "unknown"
    return f"round size {round_size}"


def _valuation_summary(verified_claims: list[ClaimRecord]) -> str:
    for label in ("valuation cap", "post-money valuation", "pre-money valuation"):
        value = _first_claim_value(verified_claims, labels={label})
        if value is not None:
            return f"{label} {value}"
    return "unknown"


def _first_claim_value(
    verified_claims: list[ClaimRecord],
    *,
    labels: set[str],
) -> str | None:
    for claim in verified_claims:
        if claim.label in labels:
            return claim.value
    return None


def _evidence_reference_text(evidence_ids: list[str]) -> str:
    if not evidence_ids:
        return ""
    formatted_ids = ", ".join(
        _memo_metadata_value(evidence_id) for evidence_id in evidence_ids
    )
    return f" Evidence: {formatted_ids}."


def _support_text(status: object) -> str:
    return f" Support: {_support_label(status)}."


def _support_label(status: object) -> str:
    return str(status).upper()


def _missing_input_text(missing_inputs: list[str]) -> str:
    if not missing_inputs:
        return ""
    return f" Missing inputs: {', '.join(missing_inputs)}."


def _portfolio_missing_input_text(missing_inputs: list[str]) -> str:
    if not missing_inputs:
        return ""
    return f" Missing inputs: {_memo_metadata_value(', '.join(missing_inputs))}."


def _net_return_summary(scored_deal: ScoredDeal) -> str:
    net_return = scored_deal.net_return
    if net_return.net_return_multiple is not None:
        return f"{net_return.net_return_multiple:g}x estimated net return"
    if net_return.entry_valuation is not None:
        return (
            f"{_format_check_size_like_money(net_return.entry_valuation)} entry valuation; "
            f"missing {', '.join(net_return.missing_inputs) or 'return assumptions'}"
        )
    return "missing verified valuation inputs"


def _format_check_size_like_money(value: int) -> str:
    if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
        return f"${value // 1_000_000_000}B"
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"${value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"${value // 1_000}K"
    return f"${value:,}"


def _format_check_size(check_size: int) -> str:
    if check_size == 0:
        return "$0"
    if check_size % 1_000 == 0:
        return f"${check_size // 1_000}K"
    return f"${check_size / 1_000:g}K"


def _format_dollars(value: int | Decimal) -> str:
    decimal_value = Decimal(value)
    if _fixed_decimal_report_chars(decimal_value) > MAX_FIXED_DECIMAL_REPORT_CHARS:
        prefix = "-" if decimal_value < 0 else ""
        return f"{prefix}${_format_scientific_decimal(decimal_value.copy_abs())}"
    amount = _quantize_decimal(decimal_value, Decimal("0.01"))
    prefix = "-" if amount < 0 else ""
    absolute_amount = abs(amount)
    if absolute_amount == absolute_amount.to_integral_value():
        return f"{prefix}${int(absolute_amount):,}"
    return f"{prefix}${absolute_amount:,.2f}"


def _format_percent(value: Decimal) -> str:
    return f"{_format_decimal(value)}%"


def _format_multiple(value: Decimal) -> str:
    return f"{_format_decimal(value)}x"


def _format_net_multiple(value: Decimal) -> str:
    if _fixed_decimal_report_chars(value) > MAX_FIXED_DECIMAL_REPORT_CHARS:
        return f"{_format_decimal(value)}x"
    rounded = _quantize_decimal(value, Decimal("0.01"))
    return f"{_format_decimal(rounded)}x"


def _quantize_decimal(value: Decimal, quantizer: Decimal) -> Decimal:
    digits_before_decimal = max(value.adjusted() + 1, 1)
    precision = max(len(value.as_tuple().digits), digits_before_decimal) + 4
    with localcontext() as context:
        context.prec = precision
        return value.quantize(quantizer, rounding=ROUND_HALF_UP)


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    if _fixed_decimal_report_chars(value) > MAX_FIXED_DECIMAL_REPORT_CHARS:
        return _format_scientific_decimal(value)
    try:
        normalized = value.normalize()
    except DecimalException:
        return _format_scientific_decimal(value)
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _fixed_decimal_report_chars(value: Decimal) -> int:
    if value == 0:
        return 1
    value_tuple = value.as_tuple()
    digit_count = len(value_tuple.digits)
    exponent = value_tuple.exponent
    if not isinstance(exponent, int):
        return MAX_FIXED_DECIMAL_REPORT_CHARS + 1
    if exponent >= 0:
        return digit_count + exponent
    integer_digits = max(value.adjusted() + 1, 1)
    return integer_digits + 1 + abs(exponent)


def _format_scientific_decimal(value: Decimal) -> str:
    mantissa, exponent = format(value, ".6E").split("E", maxsplit=1)
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_sign = exponent[0] if exponent and exponent[0] in "+-" else "+"
    exponent_digits = exponent[1:] if exponent and exponent[0] in "+-" else exponent
    exponent_digits = exponent_digits.lstrip("0") or "0"
    return f"{mantissa}E{exponent_sign}{exponent_digits}"


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ScoringError(
            f"Report folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ScoringError(f"Report folder {path} is a symlink.")
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise ScoringError(f"Could not create report folder at {path}: {exc}") from exc


def _write_private_text(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ScoringError(f"Could not write {description} at {path}: output file is a symlink.")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise ScoringError(
            f"Could not write {description} at {path}: the memo contains text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ScoringError(f"Could not write {description} at {path}: {exc}") from exc
