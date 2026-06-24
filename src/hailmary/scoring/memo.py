from __future__ import annotations

import os
from pathlib import Path

from hailmary.config import AppConfig
from hailmary.schemas.documents import IngestionSummary
from hailmary.schemas.evidence import EvidenceStore
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

    summary = IngestionSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    report_dir = config.data_dir / "reports"
    _ensure_private_directory(report_dir, private_root=config.data_dir)

    scored_deals: list[ScoredDeal] = []
    for deal in summary.deals:
        if deal.evidence_store_path is None:
            raise ScoringError(
                f"No evidence store was found for {deal.company_name}. "
                "Run `hailmary ingest-folder` again before scoring."
            )
        if not deal.evidence_store_path.exists():
            raise ScoringError(
                f"The evidence store for {deal.company_name} is missing at "
                f"{deal.evidence_store_path}. Run `hailmary ingest-folder` again."
            )
        store = EvidenceStore.model_validate_json(
            deal.evidence_store_path.read_text(encoding="utf-8")
        )
        scored_deal = score_evidence_store(store, config=config)
        memo_path = report_dir / f"{slugify(deal.company_name)}-{deal.id}-memo.md"
        _write_private_text(
            memo_path,
            render_markdown_memo(scored_deal, store),
            description="Markdown memo",
        )
        scored_deals.append(scored_deal.model_copy(update={"memo_path": memo_path}))

    return MemoRunSummary(report_dir=report_dir, scored_deals=scored_deals)


def render_markdown_memo(scored_deal: ScoredDeal, store: EvidenceStore) -> str:
    lines = [
        f"# {scored_deal.company_name} Hail Mary Memo",
        "",
        f"Recommendation: **{scored_deal.recommendation}**",
        f"Check size: **{_format_check_size(scored_deal.check_size)}**",
        f"Score: **{scored_deal.total_score}/{scored_deal.max_score}**",
        f"Product-market fit level: **{scored_deal.pmf_level}**",
        f"Next-round fundability risk: **{scored_deal.fundability_risk}**",
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
    verified_claims = [
        claim for claim in store.claims if claim.verification_status == "verified"
    ]
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
