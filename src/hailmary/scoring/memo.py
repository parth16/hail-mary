from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from hailmary.config import AppConfig
from hailmary.schemas.documents import IngestionSummary
from hailmary.schemas.evidence import ClaimRecord, EvidenceStore, VerificationStatus
from hailmary.schemas.scoring import MemoRunSummary, ScoredDeal
from hailmary.scoring.scorer import score_evidence_store
from hailmary.utils.slug import slugify


class ScoringError(RuntimeError):
    """Scoring could not continue safely."""


def score_latest_ingestion(*, config: AppConfig) -> MemoRunSummary:
    summary_path = config.data_dir / "processed" / "ingestion_summary.json"
    if not summary_path.exists():
        raise ScoringError(
            "No ingested deals were found. Run `hailmary ingest-folder` before scoring."
        )

    summary = _load_ingestion_summary(summary_path)
    report_dir = config.data_dir / "reports"
    _ensure_private_directory(report_dir, private_root=config.data_dir)

    scored_deals: list[ScoredDeal] = []
    remaining_capital = config.capital_budget
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
        scored_deal = score_evidence_store(
            store,
            config=config,
            capital_remaining=remaining_capital,
        )
        remaining_capital = scored_deal.capital_remaining_after or 0
        memo_path = report_dir / f"{slugify(deal.company_name)}-{deal.id}-memo.md"
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
        for evidence in store.evidence[:25]:
            locator = (
                f"page {evidence.page_number}"
                if evidence.page_number is not None
                else f"table {evidence.table_index}"
                if evidence.table_index is not None
                else "document"
            )
            lines.append(
                f"- {evidence.id}: {evidence.document_path} ({locator}, "
                f"{evidence.evidence_kind})."
            )
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
    if path.is_absolute():
        return path

    absolute_data_dir = _absolute_path(data_dir).resolve(strict=False)
    absolute_summary_path = _absolute_path(summary_path).resolve(strict=False)
    candidate_roots = [
        absolute_data_dir.parent,
        absolute_data_dir,
        absolute_summary_path.parent,
    ]
    candidates = [root / path for root in candidate_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _verified_claims(store: EvidenceStore) -> list[ClaimRecord]:
    return [
        claim
        for claim in store.claims
        if claim.verification_status == VerificationStatus.VERIFIED
    ]


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
    except OSError as exc:
        raise ScoringError(f"Could not write {description} at {path}: {exc}") from exc
