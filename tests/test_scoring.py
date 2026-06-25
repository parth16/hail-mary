from __future__ import annotations

import stat
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.schemas.documents import (
    DocumentType,
    FileType,
    IngestedDeal,
    IngestionSummary,
    SourceKind,
)
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
from hailmary.schemas.scoring import (
    ConfidenceLevel,
    DiligenceQuestion,
    FundabilityRisk,
    KillGate,
    PMFLevel,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
)
from hailmary.scoring import (
    render_markdown_memo,
    render_portfolio_report,
    score_evidence_store,
    score_latest_ingestion,
)
from hailmary.scoring.memo import ScoringError, _write_private_text

runner = CliRunner()
_TEST_EVIDENCE_TEXT_BY_ID: dict[str, str] = {}


def _evidence(
    record_id: str,
    text: str,
    *,
    deal_id: str = "deal_test",
    document_id: str = "doc_test",
) -> EvidenceRecord:
    _TEST_EVIDENCE_TEXT_BY_ID[record_id] = text
    return EvidenceRecord(
        id=record_id,
        deal_id=deal_id,
        document_id=document_id,
        document_path=Path("memo.txt"),
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        text=text,
        source_freshness=SourceFreshness.CURRENT,
    )


def _claim(
    label: str,
    value: str,
    evidence_id: str,
    *,
    deal_id: str = "deal_test",
) -> ClaimRecord:
    evidence_text = _TEST_EVIDENCE_TEXT_BY_ID.get(evidence_id, "")
    source_span_start = evidence_text.find(value)
    if source_span_start == -1:
        source_span_start = 0
    source_span_end = source_span_start + len(value)
    normalized_id_value = (
        value.lower()
        .replace("$", "usd")
        .replace("%", "pct")
        .replace(".", "")
        .replace(" ", "_")
    )
    return ClaimRecord(
        id=f"claim_{label.replace(' ', '_')}_{normalized_id_value}",
        deal_id=deal_id,
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
                source_span_start=source_span_start,
                source_span_end=source_span_end,
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
    deal_id: str = "deal_test",
    company_name: str = "ScoreCo",
) -> EvidenceStore:
    return EvidenceStore(
        deal_id=deal_id,
        company_name=company_name,
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


def test_scored_deal_rejects_invalid_check_size_contracts() -> None:
    def make_scored_deal(
        *,
        recommendation: Recommendation,
        check_size: int,
    ) -> ScoredDeal:
        return ScoredDeal(
            deal_id="deal_contract",
            company_name="ContractCo",
            recommendation=recommendation,
            check_size=check_size,
            total_score=80,
            one_line_reason="Synthetic contract test.",
        )

    with pytest.raises(ValueError, match="check_size must be one of"):
        make_scored_deal(
            recommendation=Recommendation.PASS,
            check_size=3_000,
        )
    with pytest.raises(ValueError, match="PASS recommendations must use"):
        make_scored_deal(
            recommendation=Recommendation.PASS,
            check_size=1_000,
        )
    with pytest.raises(ValueError, match="INVEST recommendations must use"):
        make_scored_deal(
            recommendation=Recommendation.INVEST,
            check_size=0,
        )


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
    assert scored.check_size == 5_000
    assert scored.total_score >= 70
    assert not scored.triggered_kill_gates


def test_score_evidence_store_keeps_65_to_74_as_pass() -> None:
    evidence = [
        _evidence(
            "ev_all",
            "Valuation cap $8M. Discount 20%. Round size $1M. One paid customer.",
        )
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_all"),
        _claim("discount", "20%", "ev_all"),
        _claim("round size", "$1M", "ev_all"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert 65 <= scored.total_score <= 74
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0


def test_portfolio_report_explains_score_below_threshold_skip() -> None:
    evidence = [
        _evidence(
            "ev_all",
            "Valuation cap $8M. Discount 20%. Round size $1M. One paid customer.",
        )
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_all"),
        _claim("discount", "20%", "ev_all"),
        _claim("round size", "$1M", "ev_all"),
    ]
    config = AppConfig(data_dir=Path("data"))
    scored = score_evidence_store(_store(evidence=evidence, claims=claims), config=config)

    report = render_portfolio_report([scored], config=config)

    assert "Score below the 75/100 INVEST threshold." in report


def test_portfolio_report_prefers_score_reason_over_budget_for_low_scoring_skip() -> None:
    evidence = [
        _evidence(
            "ev_all",
            "Valuation cap $8M. Discount 20%. Round size $1M. One paid customer.",
        )
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_all"),
        _claim("discount", "20%", "ev_all"),
        _claim("round size", "$1M", "ev_all"),
    ]
    config = AppConfig(data_dir=Path("data"), capital_budget=0)
    scored = score_evidence_store(_store(evidence=evidence, claims=claims), config=config)

    report = render_portfolio_report([scored], config=config)

    assert "Score below the 75/100 INVEST threshold." in report
    assert "No allocatable capital remained for an allowed nonzero check." not in report


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


def test_score_evidence_store_reuses_valid_claim_from_stale_conflict() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    valid_claim = _claim("valuation cap", "$8M", "ev_terms").model_copy(
        update={"verification_status": VerificationStatus.CONFLICTED}
    )
    stale_claim = _claim("valuation cap", "$10M", "ev_terms").model_copy(
        update={"verification_status": VerificationStatus.CONFLICTED}
    )
    claims = [
        valid_claim,
        stale_claim,
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]
    conflict = ClaimConflict(
        id="conflict_valuation",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap",
        normalized_values=["$10M", "$8M"],
        claim_ids=[valid_claim.id, stale_claim.id],
        notes="One side of this stored conflict is stale.",
    )

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims, conflicts=[conflict]),
        config=AppConfig(data_dir=Path("data")),
    )

    assert not any(
        gate.name == "Conflicting material deal terms"
        for gate in scored.triggered_kill_gates
    )
    assert "valuation cap" in _score_factor(scored, "Deal-term clarity").explanation


def test_score_evidence_store_passes_when_platform_minimum_exceeds_max_check() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Minimum check $25K."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("minimum investment", "$25K", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data"), max_check=10_000),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert any(
        gate.name == "Platform minimum above maximum check"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_rounds_fractional_platform_minimum_up() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Minimum investment $2,500.50.",
        ),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    minimum_claim = _claim("minimum investment", "$2,500.50", "ev_terms").model_copy(
        update={"normalized_value": "usd_cents:250050"}
    )
    claims = [
        _claim("valuation cap", "$8M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        minimum_claim,
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(
            data_dir=Path("data"),
            capital_budget=2_500,
            max_check=2_500,
        ),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert any(
        gate.name == "Platform minimum above maximum check"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_passes_when_no_nonzero_check_fits() -> None:
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
        config=AppConfig(data_dir=Path("data"), capital_budget=500),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert any(
        gate.name == "No available check size"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_matches_traction_keywords_as_words() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. Barrier analysis.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_negated_traction_phrases() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "The company is pre-revenue with no customers yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_coordinated_negated_traction() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "The company has no customers or revenue yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_qualified_negated_traction() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_negative_traction",
            "The company has no meaningful revenue or customers yet.",
        ),
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert scored.recommendation == Recommendation.PASS
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


@pytest.mark.parametrize(
    "traction_text",
    [
        "The company is operating without any customers yet.",
        "The company is operating without any revenue yet.",
        "The company has no actual customers yet.",
        "The company has no customer revenue yet.",
        "The company lacks customers and revenue.",
    ],
)
def test_score_evidence_store_ignores_common_negative_traction_phrases(
    traction_text: str,
) -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M.",
        ),
        _evidence("ev_negative_traction", traction_text),
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert scored.recommendation == Recommendation.PASS
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


@pytest.mark.parametrize(
    "traction_text",
    [
        "No churn among paid customers.",
        "No retention issues among enterprise customers.",
    ],
)
def test_score_evidence_store_keeps_benign_no_phrases_as_positive_traction(
    traction_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", traction_text),
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

    assert scored.pmf_level == PMFLevel.DEVELOPING
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_traction"
    ]


def test_score_evidence_store_keeps_mixed_current_traction_evidence() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "Pre-revenue last year; now $500K ARR with paid customers.",
        )
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

    assert scored.pmf_level == PMFLevel.DEVELOPING
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_terms"
    ]


def test_score_evidence_store_ignores_negated_funding_language() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_funding", "There is no lead investor and no institutional follow-on yet."),
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

    assert scored.fundability_risk == FundabilityRisk.MEDIUM
    assert _score_factor(scored, "Next-round fundability").evidence_ids == []


def test_score_evidence_store_ignores_qualified_negated_funding_language() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_funding", "There is no committed lead investor yet."),
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

    assert scored.fundability_risk == FundabilityRisk.MEDIUM
    assert _score_factor(scored, "Next-round fundability").evidence_ids == []


@pytest.mark.parametrize(
    "funding_text",
    [
        "The round lacks a lead investor.",
        "The company lacks institutional investors.",
        "The round has not secured a lead investor.",
        "The round does not have a lead investor.",
    ],
)
def test_score_evidence_store_ignores_common_negative_funding_phrases(
    funding_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_negative_funding", funding_text),
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

    assert scored.fundability_risk == FundabilityRisk.MEDIUM
    assert _score_factor(scored, "Next-round fundability").evidence_ids == []


@pytest.mark.parametrize(
    "funding_text",
    [
        "No concerns from the lead investor.",
        "No lead investor concerns.",
        "No lead investor issues.",
    ],
)
def test_score_evidence_store_keeps_benign_lead_investor_concern_phrases(
    funding_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_funding", funding_text),
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

    assert scored.fundability_risk == FundabilityRisk.LOW
    assert _score_factor(scored, "Next-round fundability").evidence_ids == [
        "ev_funding"
    ]


def test_score_evidence_store_cites_early_pmf_evidence() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. Pilot with a design partner.",
        )
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

    assert scored.pmf_level == PMFLevel.EARLY
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_terms"
    ]


def test_score_evidence_store_ignores_negated_early_pmf_language() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "There are no pilots, no usage, and no retention yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_qualified_negated_early_pmf_language() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "The company has no signed pilot yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_coordinated_negated_pmf_language() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "The company has no usage or retention yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_ignores_comma_separated_negated_pmf_language() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "The company has no usage, retention, or growth yet.",
        )
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

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == []


def test_score_evidence_store_excludes_negated_early_pmf_from_citations() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. "
            "There are no pilots, no usage, and no retention yet.",
        ),
        _evidence("ev_positive_pmf", "Beta with a design partner."),
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

    assert scored.pmf_level == PMFLevel.EARLY
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_positive_pmf"
    ]


def test_score_evidence_store_does_not_mark_single_source_high_confidence() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_single", company_name="Single Source"),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.confidence == ConfidenceLevel.MEDIUM


def test_score_evidence_store_ignores_unrelated_sources_for_high_confidence() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_unrelated",
            "ARR revenue growth with paid customers and retention.",
            document_id="doc_unrelated",
        ),
        _evidence(
            "ev_funding",
            "Lead investor committed and seed round is active.",
            document_id="doc_unrelated_two",
        ),
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

    assert scored.confidence == ConfidenceLevel.MEDIUM


def test_score_evidence_store_can_mark_multiple_claim_sources_high_confidence() -> None:
    evidence = [
        _evidence("ev_valuation", "Valuation cap $8M.", document_id="doc_valuation"),
        _evidence("ev_discount", "Discount 20%.", document_id="doc_discount"),
        _evidence("ev_round", "Round size $1M.", document_id="doc_round"),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_valuation"),
        _claim("discount", "20%", "ev_discount"),
        _claim("round size", "$1M", "ev_round"),
    ]

    store = _store(evidence=evidence, claims=claims)
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    assert scored.confidence == ConfidenceLevel.HIGH


def test_score_evidence_store_requires_verified_pricing_terms_to_invest() -> None:
    evidence = [
        _evidence("ev_terms", "Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.PASS
    assert any(
        gate.name == "Missing key investment terms"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_revalidates_claim_citations() -> None:
    evidence = [_evidence("ev_terms", "Discount 20%. Round size $1M.")]
    claims = [_claim("valuation cap", "$8M", "ev_terms")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.PASS
    assert any(
        gate.name == "No verified deal terms"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_requires_citation_span_to_match_quote() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M. Revenue is also $8M.")]
    claim = _claim("valuation cap", "$8M", "ev_terms")
    stale_claim = claim.model_copy(
        update={
            "citations": [
                claim.citations[0].model_copy(
                    update={
                        "source_span_start": 0,
                        "source_span_end": len("$8M"),
                    }
                )
            ]
        }
    )

    scored = score_evidence_store(
        _store(evidence=evidence, claims=[stale_claim]),
        config=AppConfig(data_dir=Path("data")),
    )

    assert any(
        gate.name == "No verified deal terms"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_ignores_conflicts_with_invalid_citations() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M.")]
    valid_claim = _claim("valuation cap", "$8M", "ev_terms")
    invalid_claim = _claim("valuation cap", "$10M", "ev_terms")
    conflict = ClaimConflict(
        id="conflict_valuation",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap",
        normalized_values=["$10M", "$8M"],
        claim_ids=[valid_claim.id, invalid_claim.id],
        notes="One stale citation no longer maps to evidence.",
    )

    scored = score_evidence_store(
        _store(evidence=evidence, claims=[valid_claim, invalid_claim], conflicts=[conflict]),
        config=AppConfig(data_dir=Path("data")),
    )

    assert not any(
        gate.name == "Conflicting material deal terms"
        for gate in scored.triggered_kill_gates
    )


def test_render_markdown_memo_includes_fixed_outputs_and_evidence_ids() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M.")]
    claim = _claim("valuation cap", "$8M", "ev_terms")
    store = _store(evidence=evidence, claims=[claim])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert markdown.startswith("# Hail Mary Investment Memo: ScoreCo\n\n## Decision")
    assert "**Recommendation:** PASS" in markdown
    assert "**Suggested check:** $0" in markdown
    assert "**Confidence:** medium" in markdown
    assert "ev\\_terms" in markdown
    assert "not legal, tax, financial, or investment advice" in markdown


def test_render_markdown_memo_escapes_untrusted_company_and_claim_text() -> None:
    evidence = [_evidence("ev_terms", "Valuation cap $8M.")]
    claim = _claim("valuation cap", "$8M", "ev_terms").model_copy(
        update={
            "value": "$8M\n**Recommendation:** INVEST\n# Fake Claim",
        }
    )
    company_name = "SafeCo\n**Recommendation:** INVEST\n# Fake Heading | [link](x)"
    store = _store(
        evidence=evidence,
        claims=[claim],
        company_name=company_name,
    )
    scored = score_evidence_store(
        store,
        config=AppConfig(data_dir=Path("data")),
    ).model_copy(
        update={
            "company_name": company_name,
            "one_line_reason": "Synthetic\n# Fake Reason",
        }
    )

    markdown = render_markdown_memo(scored, store)
    title_block = markdown.split("\n## Decision", 1)[0].rstrip("\n")

    assert title_block == (
        "# Hail Mary Investment Memo: "
        "SafeCo \\*\\*Recommendation:\\*\\* INVEST \\# Fake Heading \\| "
        "\\[link\\]\\(x\\)"
    )
    assert "\n**Recommendation:** INVEST" not in markdown
    assert "\n# Fake Heading" not in markdown
    assert "\n# Fake Claim" not in markdown
    assert "\\# Fake Reason" in markdown
    assert "\\*\\*Recommendation:\\*\\* INVEST" in markdown


def test_render_markdown_memo_escapes_untrusted_document_paths() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M.").model_copy(
            update={
                "document_path": Path(
                    "docs/[fake](example)\n"
                    "# Fake Heading\n"
                    "- ev_fake: injected.md"
                )
            }
        )
    ]
    claim = _claim("valuation cap", "$8M", "ev_terms")
    store = _store(evidence=evidence, claims=[claim])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "\\[fake\\]\\(example\\)" in markdown
    assert "\\# Fake Heading" in markdown
    assert "\n# Fake Heading" not in markdown
    assert "\n- ev_fake:" not in markdown


def test_render_markdown_memo_includes_ocr_lineage_for_cited_evidence() -> None:
    evidence = [
        _evidence("ev_ocr", "Valuation cap $8M.").model_copy(
            update={
                "file_type": FileType.PNG,
                "document_path": Path("scan.png"),
                "ocr_applied": True,
                "ocr_confidence": 0.86,
            }
        )
    ]
    claim = _claim("valuation cap", "$8M", "ev_ocr")
    store = _store(evidence=evidence, claims=[claim])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "image-based text reading (OCR" in markdown
    assert "OCR means reading text from images" in markdown
    assert "OCR confidence: 86%" in markdown


def test_render_markdown_memo_includes_cited_evidence_beyond_first_25() -> None:
    evidence = [
        _evidence(f"ev_{index}", f"Background evidence {index}.")
        for index in range(29)
    ]
    evidence.append(_evidence("ev_29", "Valuation cap $8M."))
    claim = _claim("valuation cap", "$8M", "ev_29")
    store = _store(evidence=evidence, claims=[claim])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "- ev_29:" in markdown


def test_render_markdown_memo_includes_conflict_evidence_beyond_first_25() -> None:
    evidence = [
        _evidence(f"ev_{index}", f"Background evidence {index}.")
        for index in range(29)
    ]
    evidence.extend(
        [
            _evidence("ev_29", "Valuation cap $8M."),
            _evidence("ev_30", "Valuation cap $10M."),
        ]
    )
    claims = [
        _claim("valuation cap", "$8M", "ev_29"),
        _claim("valuation cap", "$10M", "ev_30"),
    ]
    conflicted_claims = [
        claim.model_copy(update={"verification_status": VerificationStatus.CONFLICTED})
        for claim in claims
    ]
    conflict = ClaimConflict(
        id="conflict_valuation",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap",
        normalized_values=["valuation cap:$10M", "valuation cap:$8M"],
        claim_ids=[claim.id for claim in conflicted_claims],
        notes="Conflicting valuation caps.",
    )
    store = _store(evidence=evidence, claims=conflicted_claims, conflicts=[conflict])
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "- ev_29:" in markdown
    assert "- ev_30:" in markdown


def test_render_markdown_memo_excludes_stale_conflict_evidence_beyond_first_25() -> None:
    evidence = [
        _evidence(f"ev_{index}", f"Background evidence {index}.")
        for index in range(25)
    ]
    evidence.extend(
        [
            _evidence("ev_valid", "Valuation cap $8M."),
            _evidence("ev_stale", "Stale background with no matching valuation."),
        ]
    )
    valid_claim = _claim("valuation cap", "$8M", "ev_valid").model_copy(
        update={"verification_status": VerificationStatus.CONFLICTED}
    )
    stale_claim = _claim("valuation cap", "$10M", "ev_stale").model_copy(
        update={"verification_status": VerificationStatus.CONFLICTED}
    )
    conflict = ClaimConflict(
        id="conflict_valuation",
        deal_id="deal_test",
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap",
        normalized_values=["valuation cap:$10M", "valuation cap:$8M"],
        claim_ids=[valid_claim.id, stale_claim.id],
        notes="One conflict side is stale.",
    )
    store = _store(
        evidence=evidence,
        claims=[valid_claim, stale_claim],
        conflicts=[conflict],
    )
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "- ev_valid:" in markdown
    assert "- ev_stale:" not in markdown


def test_score_latest_ingestion_tracks_remaining_capital(tmp_path: Path) -> None:
    stores = [
        _strong_store(deal_id="deal_one", company_name="Deal One"),
        _strong_store(deal_id="deal_two", company_name="Deal Two"),
    ]
    _write_ingestion_summary(tmp_path, stores)

    result = score_latest_ingestion(
        config=AppConfig(data_dir=tmp_path / "data", capital_budget=2_500)
    )

    assert result.scored_deals[0].recommendation == Recommendation.INVEST
    assert result.scored_deals[0].check_size == 2_500
    assert result.scored_deals[1].recommendation == Recommendation.PASS
    assert result.scored_deals[1].check_size == 0
    assert any(
        gate.name == "No available check size"
        for gate in result.scored_deals[1].triggered_kill_gates
    )


def test_score_latest_ingestion_allocates_scarce_capital_by_score(
    tmp_path: Path,
) -> None:
    lower_score_store = _store_without_funding_signal(
        deal_id="deal_lower",
        company_name="A Lower Score",
    )
    higher_score_store = _strong_store(
        deal_id="deal_higher",
        company_name="B Higher Score",
    )
    _write_ingestion_summary(tmp_path, [lower_score_store, higher_score_store])

    result = score_latest_ingestion(
        config=AppConfig(data_dir=tmp_path / "data", capital_budget=2_500)
    )

    scored_by_company = {deal.company_name: deal for deal in result.scored_deals}
    assert scored_by_company["B Higher Score"].recommendation == Recommendation.INVEST
    assert scored_by_company["B Higher Score"].check_size == 2_500
    assert scored_by_company["A Lower Score"].recommendation == Recommendation.PASS
    assert scored_by_company["A Lower Score"].check_size == 0


def test_score_latest_ingestion_allocates_after_reserve_percent(
    tmp_path: Path,
) -> None:
    stores = [
        _strong_store(deal_id="deal_alpha", company_name="Alpha Reserve"),
        _strong_store(deal_id="deal_zeta", company_name="Zeta Reserve"),
    ]
    _write_ingestion_summary(tmp_path, stores)

    result = score_latest_ingestion(
        config=AppConfig(
            data_dir=tmp_path / "data",
            capital_budget=5_000,
            reserve_percent=Decimal("50"),
        )
    )

    scored_by_company = {deal.company_name: deal for deal in result.scored_deals}
    assert scored_by_company["Alpha Reserve"].recommendation == Recommendation.INVEST
    assert scored_by_company["Alpha Reserve"].check_size == 2_500
    assert scored_by_company["Zeta Reserve"].recommendation == Recommendation.PASS
    assert scored_by_company["Zeta Reserve"].check_size == 0

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    assert "Reserve: $2,500 (50% reserve)" in report
    assert "Allocatable capital after reserve: $2,500" in report
    assert "No allocatable capital remained for an allowed nonzero check." in report


def test_score_latest_ingestion_respects_max_check_in_portfolio_allocation(
    tmp_path: Path,
) -> None:
    store = _strong_store(deal_id="deal_max", company_name="Max Check")
    _write_ingestion_summary(tmp_path, [store])

    result = score_latest_ingestion(
        config=AppConfig(
            data_dir=tmp_path / "data",
            max_check=1_000,
        )
    )

    assert result.scored_deals[0].recommendation == Recommendation.INVEST
    assert result.scored_deals[0].check_size == 1_000


def test_score_latest_ingestion_does_not_penalize_fresh_local_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Fresh Local"
    company.mkdir(parents=True)
    (company / "terms.txt").write_text(
        "Valuation cap $8M. Discount 20%. Round size $1M.",
        encoding="utf-8",
    )
    (company / "traction.txt").write_text(
        "ARR revenue growth with paid customers and retention.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / "data")
    ingest_folder(root, config=config)

    result = score_latest_ingestion(config=config)

    scored = result.scored_deals[0]
    assert scored.total_score == 75
    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size == 2_500


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


def test_score_latest_ingestion_writes_private_portfolio_report(tmp_path: Path) -> None:
    stores = [
        _strong_store(deal_id="deal_one", company_name="Deal One"),
        _store_without_funding_signal(
            deal_id="deal_two",
            company_name="Deal Two",
        ),
    ]
    _write_ingestion_summary(tmp_path, stores)

    result = score_latest_ingestion(config=AppConfig(data_dir=tmp_path / "data"))

    portfolio_report_path = result.portfolio_report_path
    assert portfolio_report_path == tmp_path / "data" / "reports" / (
        "portfolio-comparison-report.md"
    )
    assert portfolio_report_path.exists()
    assert stat.S_IMODE(portfolio_report_path.stat().st_mode) == 0o600
    report = portfolio_report_path.read_text(encoding="utf-8")
    assert report.startswith("# Hail Mary Portfolio Comparison Report")
    assert "## Portfolio Scenario And Constraints" in report
    assert "Starting capital budget: $100,000" in report
    assert "Reserve: $0 (no reserve)" in report
    assert "Allocatable capital after reserve: $100,000" in report
    assert "Allowed check sizes: $0, $1K, $2.5K, $5K, $7.5K, $10K" in report
    assert "Configured minimum check: $1K" in report
    assert "Configured maximum check: $10K" in report
    assert "## Ranked Deals" in report
    assert "## Skipped Deals" in report
    assert "## Net Return Math" in report
    assert "## Deal Details" in report
    assert "Memo path:" in report
    assert "Carry means the share of profits paid to the fund manager or platform." in report
    assert "Dilution means ownership reduction from future fundraising." in report


def test_render_portfolio_report_ranks_final_scored_deals_deterministically(
    tmp_path: Path,
) -> None:
    lower_score_store = _store_without_funding_signal(
        deal_id="deal_lower",
        company_name="A Lower Score",
    )
    higher_score_store = _strong_store(
        deal_id="deal_higher",
        company_name="B Higher Score",
    )
    _write_ingestion_summary(tmp_path, [lower_score_store, higher_score_store])

    result = score_latest_ingestion(
        config=AppConfig(data_dir=tmp_path / "data", capital_budget=2_500)
    )

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    higher_row = "| 1 | B Higher Score | INVEST | $2.5K |"
    lower_row = "| 2 | A Lower Score | PASS | $0 |"
    assert higher_row in report
    assert lower_row in report
    assert report.index(higher_row) < report.index(lower_row)
    assert "No available check size" in report


def test_score_latest_ingestion_allocates_tied_deals_in_report_rank_order(
    tmp_path: Path,
) -> None:
    later_company_store = _strong_store(
        deal_id="deal_zeta",
        company_name="Zeta Score",
    )
    earlier_company_store = _strong_store(
        deal_id="deal_alpha",
        company_name="Alpha Score",
    )
    _write_ingestion_summary(tmp_path, [later_company_store, earlier_company_store])

    result = score_latest_ingestion(
        config=AppConfig(data_dir=tmp_path / "data", capital_budget=2_500)
    )

    scored_by_company = {deal.company_name: deal for deal in result.scored_deals}
    assert scored_by_company["Alpha Score"].recommendation == Recommendation.INVEST
    assert scored_by_company["Alpha Score"].check_size == 2_500
    assert scored_by_company["Zeta Score"].recommendation == Recommendation.PASS
    assert scored_by_company["Zeta Score"].check_size == 0

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    alpha_row = "| 1 | Alpha Score | INVEST | $2.5K |"
    zeta_row = "| 2 | Zeta Score | PASS | $0 |"
    assert alpha_row in report
    assert zeta_row in report
    assert report.index(alpha_row) < report.index(zeta_row)
    assert "| $2.5K | $0 |" in report


def test_portfolio_report_preserves_allocation_order_when_early_pass_spends_no_capital(
    tmp_path: Path,
) -> None:
    high_minimum_evidence = [
        _evidence(
            "ev_high_min_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M. Minimum investment $5K.",
            deal_id="deal_high_min",
        ),
        _evidence(
            "ev_high_min_traction",
            "ARR revenue growth with paid customers and retention.",
            deal_id="deal_high_min",
        ),
        _evidence(
            "ev_high_min_funding",
            "Lead investor committed and seed round is active.",
            deal_id="deal_high_min",
        ),
    ]
    high_minimum_store = _store(
        evidence=high_minimum_evidence,
        claims=[
            _claim("valuation cap", "$8M", "ev_high_min_terms", deal_id="deal_high_min"),
            _claim("discount", "20%", "ev_high_min_terms", deal_id="deal_high_min"),
            _claim("round size", "$1M", "ev_high_min_terms", deal_id="deal_high_min"),
            _claim(
                "minimum investment",
                "$5K",
                "ev_high_min_terms",
                deal_id="deal_high_min",
            ),
        ],
        deal_id="deal_high_min",
        company_name="High Minimum",
    )
    affordable_store = _strong_store(
        deal_id="deal_affordable",
        company_name="Affordable",
    )
    _write_ingestion_summary(tmp_path, [high_minimum_store, affordable_store])

    result = score_latest_ingestion(
        config=AppConfig(data_dir=tmp_path / "data", capital_budget=2_500)
    )

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    high_minimum_row = "| 1 | High Minimum | PASS | $0 |"
    affordable_row = "| 2 | Affordable | INVEST | $2.5K |"
    assert high_minimum_row in report
    assert affordable_row in report
    assert report.index(high_minimum_row) < report.index(affordable_row)
    assert "| $2.5K | $2.5K |" in report
    assert "| $2.5K | $0 |" in report


def test_portfolio_report_handles_all_pass_portfolio(tmp_path: Path) -> None:
    stores = [
        _store(evidence=[], claims=[], deal_id="deal_empty_a", company_name="Empty A"),
        _store(evidence=[], claims=[], deal_id="deal_empty_b", company_name="Empty B"),
    ]
    _write_ingestion_summary(tmp_path, stores)

    result = score_latest_ingestion(config=AppConfig(data_dir=tmp_path / "data"))

    assert all(deal.recommendation == Recommendation.PASS for deal in result.scored_deals)
    assert all(deal.check_size == 0 for deal in result.scored_deals)
    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    assert "No capital was allocated, so every return case starts from $0 invested." in report
    assert "| 1 | Empty A | 9/100 | No usable source-linked evidence was available. |" in report
    assert "| 2 | Empty B | 9/100 | No usable source-linked evidence was available. |" in report


def test_render_portfolio_report_includes_net_return_math_after_fees_carry_and_dilution(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        estimated_dilution_percent=Decimal("10"),
        platform_fee_percent=Decimal("2"),
        carry_percent=Decimal("20"),
        gross_return_multiple=Decimal("3"),
    )
    scored = score_evidence_store(
        _strong_store(deal_id="deal_return", company_name="Return Math"),
        config=config,
    )

    report = render_portfolio_report([scored], config=config)

    assert "Carry means the share of profits paid to the fund manager or platform." in report
    assert "Dilution means ownership reduction from future fundraising." in report
    assert (
        "| Configured | 3x | $5,000 | $15,000 | $13,500 | $100 | $1,700 | "
        "$11,800 | $6,700 | 2.31x |"
    ) in report
    assert "| Sensitivity 1x | 1x |" in report
    assert "| Sensitivity 3x | 3x |" in report
    assert "| Sensitivity 10x | 10x |" in report


def test_render_portfolio_report_formats_large_return_assumptions_without_crashing(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        capital_budget=10**40,
        gross_return_multiple=Decimal("1e20"),
    )
    scored = score_evidence_store(
        _strong_store(deal_id="deal_large", company_name="Large Math"),
        config=config,
    )

    report = render_portfolio_report([scored], config=config)

    expected_budget = (
        "Starting capital budget: "
        "$10,000,000,000,000,000,000,000,000,000,000,000,000,000"
    )
    assert expected_budget in report
    assert "| Configured | 100000000000000000000x |" in report


def test_render_portfolio_report_labels_risks_with_evidence_or_uncertainty() -> None:
    strong_scored = score_evidence_store(
        _strong_store(deal_id="deal_strong", company_name="StrongCo"),
        config=AppConfig(data_dir=Path("data")),
    ).model_copy(update={"memo_path": Path("data/reports/strong-memo.md")})
    no_evidence_scored = score_evidence_store(
        _store(evidence=[], claims=[], deal_id="deal_empty", company_name="EmptyCo"),
        config=AppConfig(data_dir=Path("data")),
    ).model_copy(update={"memo_path": Path("data/reports/empty-memo.md")})

    report = render_portfolio_report(
        [strong_scored, no_evidence_scored],
        config=AppConfig(data_dir=Path("data")),
    )

    assert "INFERRED: Product-market fit evidence scored" in report
    assert "Evidence: ev\\_traction." in report
    assert "NEEDS_DILIGENCE: No usable source-linked evidence" in report
    assert (
        "NEEDS_DILIGENCE: Find concrete customer, revenue, retention, or usage evidence"
        in report
    )


def test_render_portfolio_report_filters_allowed_check_sizes_by_config() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_strong", company_name="StrongCo"),
        config=AppConfig(data_dir=Path("data"), min_check=2_500, max_check=5_000),
    )

    report = render_portfolio_report(
        [scored],
        config=AppConfig(data_dir=Path("data"), min_check=2_500, max_check=5_000),
    )

    allowed_line = next(
        line for line in report.splitlines() if line.startswith("- Allowed check sizes:")
    )
    assert allowed_line == "- Allowed check sizes: $0, $2.5K, $5K"
    assert "$1K" not in allowed_line
    assert "$7.5K" not in allowed_line
    assert "$10K" not in allowed_line


def test_render_portfolio_report_caps_allowed_check_sizes_by_capital_budget() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_strong", company_name="StrongCo"),
        config=AppConfig(data_dir=Path("data"), capital_budget=2_500),
    )

    report = render_portfolio_report(
        [scored],
        config=AppConfig(data_dir=Path("data"), capital_budget=2_500),
    )

    allowed_line = next(
        line for line in report.splitlines() if line.startswith("- Allowed check sizes:")
    )
    assert allowed_line == "- Allowed check sizes: $0, $1K, $2.5K"
    assert "$5K" not in allowed_line
    assert "$7.5K" not in allowed_line
    assert "$10K" not in allowed_line


def test_render_portfolio_report_escapes_dynamic_markdown() -> None:
    scored_deal = ScoredDeal(
        deal_id="deal|bad\n# Fake Deal",
        company_name="Bad|Co\n# Fake Heading",
        recommendation=Recommendation.PASS,
        check_size=0,
        total_score=42,
        confidence=ConfidenceLevel.LOW,
        one_line_reason="Injected\n# Bad Reason",
        pmf_level=PMFLevel.UNKNOWN,
        fundability_risk=FundabilityRisk.HIGH,
        kill_gates=[
            KillGate(
                name="Gate|Name\n# Bad Gate",
                triggered=True,
                reason="Reason with [fake](https://example.com)\n# Bad Reason",
            )
        ],
        score_factors=[
            ScoreFactor(
                name="Factor|Name",
                score=1,
                max_score=10,
                explanation="Explanation with | pipe\n# Bad Factor",
                evidence_ids=["ev|bad"],
            )
        ],
        diligence_questions=[
            DiligenceQuestion(
                priority=1,
                question="Question with | pipe\n# Bad Question",
                reason="Reason with `code`\n# Bad Diligence",
            )
        ],
        memo_path=Path("data/reports/[bad](memo).md"),
    )

    report = render_portfolio_report(
        [scored_deal],
        config=AppConfig(data_dir=Path("data")),
    )

    assert "\n# Fake Heading" not in report
    assert "\n# Bad Reason" not in report
    assert "\n# Bad Factor" not in report
    assert "\n# Bad Question" not in report
    assert "Bad\\|Co \\# Fake Heading" in report
    assert "Gate\\|Name \\# Bad Gate" in report
    assert "\\[bad\\]\\(memo\\).md" in report
    assert "ev\\|bad" in report


def test_score_latest_ingestion_rebases_relative_evidence_store_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "PathCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap $8M. Discount 20%. Round size $1M.",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    ingest_folder(root, config=AppConfig(data_dir=Path("data")))

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    result = score_latest_ingestion(config=AppConfig(data_dir=tmp_path / "data"))

    assert result.deal_count == 1
    assert result.scored_deals[0].memo_path is not None


def test_score_latest_ingestion_resolves_nested_relative_data_dir_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "NestedPathCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap $8M. Discount 20%. Round size $1M.",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config = AppConfig(data_dir=Path("local/data"))
    ingest_folder(root, config=config)

    result = score_latest_ingestion(config=config)

    assert result.deal_count == 1
    assert result.scored_deals[0].memo_path is not None


def test_score_latest_ingestion_rejects_absolute_evidence_path_outside_data_dir(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    store = _strong_store(deal_id="deal_outside", company_name="Outside Store")
    outside_store_path = outside_dir / "store.json"
    outside_store_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    summary = IngestionSummary(
        root_path=tmp_path / "pitch-decks",
        scanned_at=datetime(2026, 1, 1, tzinfo=UTC),
        deals=[
            IngestedDeal(
                id=store.deal_id,
                company_name=store.company_name,
                documents=[],
                evidence_store_path=outside_store_path,
                evidence_count=store.evidence_count,
                claim_count=store.claim_count,
                conflict_count=store.conflict_count,
            )
        ],
        summary_path=processed_dir / "ingestion_summary.json",
    )
    summary.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ScoringError, match="outside the private data directory"):
        score_latest_ingestion(config=AppConfig(data_dir=data_dir))


def test_score_deals_command_has_rich_success_output(tmp_path: Path) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code == 0, result.output
    assert "Scoring complete" in result.output
    assert "Company" in result.output
    assert "Recommendation" in result.output
    assert "Scored 1 deal." in result.output
    assert "Saved the portfolio comparison report to" in result.output
    assert "portfolio-comparison-report.md" in result.output


def test_score_deals_command_accepts_portfolio_scenario_overrides(tmp_path: Path) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "score-deals",
            "--data-dir",
            str(tmp_path / "data"),
            "--capital-budget",
            "2500",
            "--reserve-dollars",
            "1500",
            "--estimated-dilution-percent",
            "10",
            "--platform-fee-percent",
            "5",
            "--carry-percent",
            "20",
            "--gross-return-multiple",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "data" / "reports" / "portfolio-comparison-report.md"
    report = report_path.read_text(encoding="utf-8")
    assert "Starting capital budget: $2,500" in report
    assert "Reserve: $1,500 (configured reserve dollars)" in report
    assert "Allocatable capital after reserve: $1,000" in report
    assert "Estimated dilution: 10%" in report
    assert "Platform fee: 5%" in report
    assert "Carry: 20%" in report
    assert "Gross return multiple: 8x" in report
    assert "| 1 | Deal One | INVEST | $1K |" in report


def test_score_deals_command_rejects_conflicting_reserve_overrides(tmp_path: Path) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "score-deals",
            "--data-dir",
            str(tmp_path / "data"),
            "--reserve-percent",
            "10",
            "--reserve-dollars",
            "1000",
        ],
    )

    assert result.exit_code != 0
    assert "Use either reserve percent or reserve dollars" in result.output
    assert "Traceback" not in result.output


def test_score_deals_missing_ingestion_summary_has_plain_english_error(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "No ingested deals were found" in result.output
    assert "Traceback" not in result.output


def test_score_deals_bad_ingestion_summary_has_plain_english_error(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "ingestion_summary.json").write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "The ingestion summary could not be read" in result.output
    assert "Traceback" not in result.output


def test_score_deals_non_utf8_summary_has_plain_english_error(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "ingestion_summary.json").write_bytes(b"\xff\xfe\x00")

    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "ingestion summary is not plain text" in result.output
    assert "Traceback" not in result.output


def test_score_deals_rejects_current_folder_as_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["score-deals", "--data-dir", "."])

    assert result.exit_code != 0
    assert "data directory cannot be the current folder" in result.output
    assert not (tmp_path / "reports").exists()


def test_score_latest_ingestion_non_utf8_evidence_store_has_plain_english_error(
    tmp_path: Path,
) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])
    store_path = tmp_path / "data" / "processed" / f"{store.deal_id}.json"
    store_path.write_bytes(b"\xff\xfe\x00")

    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "evidence store for Deal One is not plain text" in result.output
    assert "Traceback" not in result.output


def test_write_private_text_wraps_unicode_encode_errors(tmp_path: Path) -> None:
    with pytest.raises(ScoringError, match="cannot be saved as UTF-8"):
        _write_private_text(
            tmp_path / "memo.md",
            "bad surrogate \udcff",
            description="Markdown memo",
        )


def _strong_store(*, deal_id: str, company_name: str) -> EvidenceStore:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M.",
            deal_id=deal_id,
        ),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            deal_id=deal_id,
        ),
        _evidence(
            "ev_funding",
            "Lead investor committed and seed round is active.",
            deal_id=deal_id,
        ),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms", deal_id=deal_id),
        _claim("discount", "20%", "ev_terms", deal_id=deal_id),
        _claim("round size", "$1M", "ev_terms", deal_id=deal_id),
    ]
    return _store(
        evidence=evidence,
        claims=claims,
        deal_id=deal_id,
        company_name=company_name,
    )


def _store_without_funding_signal(*, deal_id: str, company_name: str) -> EvidenceStore:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M.",
            deal_id=deal_id,
        ),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            deal_id=deal_id,
        ),
        _evidence(
            "ev_other",
            "No lead investor is committed yet.",
            deal_id=deal_id,
        ),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms", deal_id=deal_id),
        _claim("discount", "20%", "ev_terms", deal_id=deal_id),
        _claim("round size", "$1M", "ev_terms", deal_id=deal_id),
    ]
    return _store(
        evidence=evidence,
        claims=claims,
        deal_id=deal_id,
        company_name=company_name,
    )


def _write_ingestion_summary(tmp_path: Path, stores: list[EvidenceStore]) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    deals: list[IngestedDeal] = []
    for store in stores:
        store_path = processed_dir / f"{store.deal_id}.json"
        store_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
        deals.append(
            IngestedDeal(
                id=store.deal_id,
                company_name=store.company_name,
                documents=[],
                evidence_store_path=store_path,
                evidence_count=store.evidence_count,
                claim_count=store.claim_count,
                conflict_count=store.conflict_count,
            )
        )

    summary = IngestionSummary(
        root_path=tmp_path / "pitch-decks",
        scanned_at=datetime(2026, 1, 1, tzinfo=UTC),
        deals=deals,
        summary_path=processed_dir / "ingestion_summary.json",
    )
    summary.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def _score_factor(scored_deal: ScoredDeal, name: str) -> ScoreFactor:
    for factor in scored_deal.score_factors:
        if factor.name == name:
            return factor
    raise AssertionError(f"Missing score factor: {name}")
