from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.schemas.documents import DocumentType, FileType, SourceKind
from hailmary.schemas.evidence import (
    ClaimConflict,
    ClaimRecord,
    ClaimType,
    EvidenceCitation,
    EvidenceKind,
    EvidenceQuality,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)
from hailmary.schemas.scoring import Recommendation
from hailmary.scoring import render_markdown_memo, score_evidence_store, score_latest_ingestion

runner = CliRunner()


def _evidence(record_id: str, text: str) -> EvidenceRecord:
    return EvidenceRecord(
        id=record_id,
        deal_id="deal_test",
        document_id="doc_test",
        document_path=Path("memo.txt"),
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        text=text,
        source_freshness=SourceFreshness.CURRENT,
    )


def _claim(label: str, value: str, evidence_id: str) -> ClaimRecord:
    return ClaimRecord(
        id=f"claim_{label.replace(' ', '_')}",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label=label,
        value=value,
        normalized_value=f"{label}:{value}",
        unit="text",
        raw_text=f"{label} {value}",
        citations=[
            EvidenceCitation(
                evidence_id=evidence_id,
                quote=value,
                source_span_start=0,
                source_span_end=len(value),
                verification_status=VerificationStatus.VERIFIED,
            )
        ],
        verification_status=VerificationStatus.VERIFIED,
        quality=EvidenceQuality(
            claim_type=ClaimType.DEAL_TERM,
            source_type=SourceKind.LOCAL_FILE,
            verification_status=VerificationStatus.VERIFIED,
            recency=SourceFreshness.CURRENT,
            reliability="test",
            confidence=0.9,
            materiality="high",
        ),
    )


def _store(
    *,
    evidence: list[EvidenceRecord],
    claims: list[ClaimRecord],
    conflicts: list[ClaimConflict] | None = None,
) -> EvidenceStore:
    return EvidenceStore(
        deal_id="deal_test",
        company_name="ScoreCo",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        evidence=evidence,
        claims=claims,
        conflicts=conflicts or [],
    )


def test_score_evidence_store_passes_with_no_evidence() -> None:
    scored = score_evidence_store(
        _store(evidence=[], claims=[]),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.triggered_kill_gates[0].name == "No usable source-linked evidence"


def test_score_evidence_store_invests_when_verified_evidence_is_strong() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size == 7_500
    assert scored.total_score >= 70
    assert not scored.triggered_kill_gates


def test_score_evidence_store_passes_when_terms_conflict() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M. Valuation cap $10M.")]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms"),
        _claim("valuation cap", "$10M", "ev_terms"),
    ]
    conflict = ClaimConflict(
        id="conflict_valuation",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap",
        normalized_values=["$10M", "$8M"],
        claim_ids=[claim.id for claim in claims],
        notes="Conflicting valuation caps.",
    )

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims, conflicts=[conflict]),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert any(
        gate.name == "Conflicting material deal terms"
        for gate in scored.triggered_kill_gates
    )


def test_render_markdown_memo_includes_fixed_outputs_and_evidence_ids() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M.")]
    claim = _claim("valuation cap", "$8M", "ev_terms")
    store = _store(evidence=evidence, claims=[claim])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "Recommendation: **PASS**" in markdown
    assert "Check size: **$0**" in markdown
    assert "ev_terms" in markdown
    assert "not legal, tax, financial, or investment advice" in markdown


def test_score_latest_ingestion_writes_private_markdown_memos(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "MemoCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap $8M. ARR revenue growth with paid customers.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / "data")
    ingest_folder(root, config=config)

    result = score_latest_ingestion(config=config)

    assert result.deal_count == 1
    memo_path = result.scored_deals[0].memo_path
    assert memo_path is not None
    assert memo_path.exists()
    assert "Recommendation:" in memo_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(memo_path.stat().st_mode) == 0o600


def test_score_deals_missing_ingestion_summary_has_plain_english_error(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "No ingested deals were found" in result.output
    assert "Traceback" not in result.output
