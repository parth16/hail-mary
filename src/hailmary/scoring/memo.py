from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import IngestionSummary
from hailmary.schemas.evidence import ClaimRecord, EvidenceRecord, EvidenceStore
from hailmary.schemas.scoring import MemoRunSummary, Recommendation, ScoredDeal
from hailmary.scoring.scorer import score_evidence_store, validated_verified_claims
from hailmary.utils.slug import slugify


class ScoringError(RuntimeError):
    """Scoring could not continue safely."""


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
            capital_remaining=max(config.capital_budget, config.max_check),
        )
        memo_path = report_dir / f"{slugify(deal.company_name)}-{deal.id}-memo.md"
        scoring_inputs.append((store, ranking_scored_deal, memo_path))

    remaining_capital = config.capital_budget
    scored_by_index: dict[int, ScoredDeal] = {}
    ranked_inputs = sorted(
        enumerate(scoring_inputs),
        key=lambda item: (
            item[1][1].recommendation == Recommendation.INVEST,
            item[1][1].total_score,
        ),
        reverse=True,
    )
    for index, (store, _, _) in ranked_inputs:
        scored_deal = score_evidence_store(
            store,
            config=config,
            capital_remaining=remaining_capital,
        )
        remaining_capital = scored_deal.capital_remaining_after or 0
        scored_by_index[index] = scored_deal

    scored_deals: list[ScoredDeal] = []
    for index, (store, _, memo_path) in enumerate(scoring_inputs):
        scored_deal = scored_by_index[index]
        _write_private_text(
            memo_path,
            render_markdown_memo(scored_deal, store),
            description="Markdown memo",
        )
        scored_deals.append(scored_deal.model_copy(update={"memo_path": memo_path}))

    return MemoRunSummary(report_dir=report_dir, scored_deals=scored_deals)


def render_markdown_memo(scored_deal: ScoredDeal, store: EvidenceStore) -> str:
    verified_claims = _verified_claims(store)
    lines = [
        f"# Hail Mary Investment Memo: {scored_deal.company_name}",
        "",
        "## Decision",
        "",
        f"**Recommendation:** {scored_deal.recommendation}",
        f"**Suggested check:** {_format_check_size(scored_deal.check_size)}",
        f"**Score:** {scored_deal.total_score}/{scored_deal.max_score}",
        f"**Confidence:** {scored_deal.confidence}",
        f"**One-line reason:** {scored_deal.one_line_reason}",
        "**Deadline:** unknown",
        f"**Round / Instrument:** {_round_summary(verified_claims)} / unknown",
        f"**Valuation / Cap:** {_valuation_summary(verified_claims)}",
        "",
        "## Kill Gates",
    ]
    for gate in scored_deal.kill_gates:
        status = "TRIGGERED" if gate.triggered else "Clear"
        lines.append(f"- {status}: {gate.name}. {gate.reason}")

    lines.extend(["", "## Score Factors"])
    for factor in scored_deal.score_factors:
        evidence_text = _evidence_reference_text(factor.evidence_ids)
        lines.append(
            f"- {factor.name}: {factor.score}/{factor.max_score}. "
            f"{factor.explanation}{evidence_text}"
        )

    lines.extend(["", "## Verified Deal Terms"])
    if verified_claims:
        for claim in verified_claims:
            citation_ids = ", ".join(
                citation.evidence_id for citation in claim.citations
            )
            lines.append(
                f"- {claim.label}: {claim.value} "
                f"(evidence: {citation_ids or 'none'})."
            )
    else:
        lines.append("- No verified deal-term claims were available.")

    lines.extend(["", "## Diligence Questions"])
    for question in scored_deal.diligence_questions:
        lines.append(
            f"{question.priority}. {question.question} "
            f"Reason: {question.reason}"
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
    cited_ids = _cited_evidence_ids(scored_deal, verified_claims)
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
    return (
        f"- {evidence.id}: {evidence.document_path} "
        f"({locator}, {evidence.evidence_kind})."
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
    return f" Evidence: {', '.join(evidence_ids)}."


def _format_check_size(check_size: int) -> str:
    if check_size == 0:
        return "$0"
    if check_size % 1_000 == 0:
        return f"${check_size // 1_000}K"
    return f"${check_size / 1_000:g}K"


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
