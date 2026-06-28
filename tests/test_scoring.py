from __future__ import annotations

import stat
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.portfolio import add_portfolio_investment
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
    CompanyStage,
    ConfidenceLevel,
    DiligenceQuestion,
    DiligenceQuestionCategory,
    DiligenceResearchContext,
    FundabilityRisk,
    KillGate,
    PMFLevel,
    PortfolioExposureDimension,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
    ScoreSupportStatus,
    ValuationRisk,
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
    source_freshness: SourceFreshness = SourceFreshness.CURRENT,
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
        source_freshness=source_freshness,
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
    no_terms_gate = next(
        gate
        for gate in scored.kill_gates
        if gate.name == "No verified deal terms"
    )
    assert no_terms_gate.support_status == ScoreSupportStatus.NEEDS_DILIGENCE


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
    assert scored.check_size == 2_500
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


def test_stage_aware_score_changes_are_deterministic_and_evidence_linked() -> None:
    pre_seed_evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_stage", "Pre-seed beta with a design partner."),
    ]
    series_a_evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_stage", "Series A beta with a design partner."),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    pre_seed = score_evidence_store(
        _store(evidence=pre_seed_evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )
    series_a = score_evidence_store(
        _store(evidence=series_a_evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert pre_seed.company_stage == CompanyStage.PRE_SEED
    assert series_a.company_stage == CompanyStage.SERIES_A
    assert pre_seed.pmf_level == PMFLevel.EARLY
    assert series_a.pmf_level == PMFLevel.EARLY
    assert _score_factor(pre_seed, "Stage and product-market fit").score > _score_factor(
        series_a,
        "Stage and product-market fit",
    ).score
    assert _score_factor(pre_seed, "Stage and product-market fit").evidence_ids == [
        "ev_stage"
    ]


def test_stage_classification_ignores_negated_future_stage_mentions() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed stage. Not ready for Series A. Valuation cap $80M. "
            "Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$80M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == CompanyStage.SEED
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


@pytest.mark.parametrize(
    "stage_text",
    [
        "Pre-seed company plans to raise Series A next year.",
        "Pre-seed company will raise Series A next year.",
        "Not a defense company; pre-seed company.",
        "Not hard tech; pre-seed company.",
        "Not a defense or aerospace company; pre-seed company.",
        "Non-defense pre-seed company.",
        "Non-aerospace pre-seed company.",
    ],
)
def test_stage_classification_ignores_future_and_negated_category_mentions(
    stage_text: str,
) -> None:
    evidence = [
        _evidence(
            "ev_terms",
            f"{stage_text} Valuation cap $80M. Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$80M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == CompanyStage.PRE_SEED
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


def test_stage_classification_prefers_current_stage_over_series_a_investor_reference() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed stage; Series A investors are interested. Valuation cap $80M. "
            "Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$80M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == CompanyStage.SEED
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


def test_stage_classification_ignores_absent_seed_funding_mentions() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "No seed funding yet. Valuation cap $60M. Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$60M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == CompanyStage.UNKNOWN
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


def test_score_evidence_store_gates_valuation_far_ahead_of_evidence() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Pre-seed company. Valuation cap $80M. Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$80M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.company_stage == CompanyStage.PRE_SEED
    assert scored.valuation_risk == ValuationRisk.HIGH
    valuation_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "Valuation far ahead of evidence"
    )
    assert valuation_gate.evidence_ids == ["ev_terms"]


def test_valuation_far_ahead_gate_cites_stage_evidence() -> None:
    evidence = [
        _evidence("ev_stage", "Pre-seed company."),
        _evidence("ev_terms", "Valuation cap $80M. Discount 20%. Round size $1M."),
    ]
    claims = [
        _claim("valuation cap", "$80M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    valuation_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "Valuation far ahead of evidence"
    )
    assert valuation_gate.evidence_ids == ["ev_terms", "ev_stage"]
    valuation_question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Confirm why the valuation is justified by current evidence."
    )
    assert valuation_question.evidence_ids == ["ev_terms", "ev_stage"]


def test_missing_valuation_creates_ranked_financing_question() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed company. Discount 20%. Round size $1M. Beta users are active.",
        )
    ]
    claims = [
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    question = next(
        question
        for question in scored.diligence_questions
        if question.category == DiligenceQuestionCategory.FINANCING_TERMS
        and "valuation" in question.question.casefold()
    )

    assert question.rank_score > 0
    assert question.materiality_score == 5
    assert question.decision_impact_score == 5
    assert question.evidence_ids == ["ev_terms"]
    assert question.supporting_evidence_ids == ["ev_terms"]
    assert "verified valuation or valuation cap" in question.missing_evidence
    _assert_questions_have_lineage(scored.diligence_questions)


def test_conflicting_traction_creates_high_priority_customer_question() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_positive_traction",
            "ARR revenue growth with paid customers and retention.",
        ),
        _evidence("ev_negative_traction", "No customers, no revenue, or usage yet."),
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

    question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Resolve the conflicting customer traction signals."
    )

    assert question.priority <= 2
    assert question.category == DiligenceQuestionCategory.CUSTOMERS
    assert question.supporting_evidence_ids == ["ev_positive_traction"]
    assert question.conflicting_evidence_ids == ["ev_negative_traction"]
    assert question.evidence_ids == ["ev_positive_traction", "ev_negative_traction"]
    assert "resolved customer traction evidence" in question.missing_evidence
    _assert_questions_have_lineage(scored.diligence_questions)


def test_without_negative_traction_creates_customer_conflict_question() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_positive_traction",
            "ARR revenue growth with paid customers and retention.",
        ),
        _evidence("ev_negative_traction", "The company operates without customers."),
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

    question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Resolve the conflicting customer traction signals."
    )

    assert question.supporting_evidence_ids == ["ev_positive_traction"]
    assert question.conflicting_evidence_ids == ["ev_negative_traction"]
    assert question.evidence_ids == ["ev_positive_traction", "ev_negative_traction"]


def test_conflicting_traction_question_preserves_conflicting_ids_when_support_is_capped() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        *[
            _evidence(
                f"ev_positive_{index}",
                f"ARR revenue growth with paid customers in cohort {index}.",
            )
            for index in range(1, 7)
        ],
        _evidence("ev_negative_traction", "No customers or revenue yet."),
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

    question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Resolve the conflicting customer traction signals."
    )

    assert len(question.supporting_evidence_ids) == 5
    assert question.conflicting_evidence_ids == ["ev_negative_traction"]
    assert "ev_negative_traction" in question.evidence_ids
    assert len(question.evidence_ids) <= 5


def test_projected_growth_does_not_create_traction_conflict() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_current", "Current ARR revenue with paid customers."),
        _evidence("ev_projection", "Projected growth is expected next year."),
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

    assert all(
        question.question != "Resolve the conflicting customer traction signals."
        for question in scored.diligence_questions
    )


def test_weak_evidence_creates_verification_question() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed company. Valuation cap $8M. Discount 20%. Round size $1M.",
            source_freshness=SourceFreshness.STALE,
        ),
        _evidence("ev_traction", "Paid customers with revenue growth."),
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

    question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Verify stale or undated evidence before relying on it."
    )

    assert question.category == DiligenceQuestionCategory.COMPANY
    assert question.supporting_evidence_ids == ["ev_terms"]
    assert question.evidence_ids == ["ev_terms"]
    assert question.confidence_gap_score == 4
    assert "current source dates" in question.missing_evidence
    _assert_questions_have_lineage(scored.diligence_questions)


def test_pre_money_without_round_size_asks_for_round_size() -> None:
    evidence = [_evidence("ev_valuation", "Pre-money valuation $40M.")]
    claims = [_claim("pre-money valuation", "$40M", "ev_valuation")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    question = next(
        question
        for question in scored.diligence_questions
        if question.question == "Confirm the round size for the verified pre-money valuation."
    )

    assert question.category == DiligenceQuestionCategory.FINANCING_TERMS
    assert question.evidence_ids == ["ev_valuation"]
    assert question.missing_evidence == ["verified round size for pre-money valuation"]


def test_empty_evidence_creates_compact_baseline_question_set() -> None:
    scored = score_evidence_store(
        _store(evidence=[], claims=[]),
        config=AppConfig(data_dir=Path("data")),
    )

    categories = {question.category for question in scored.diligence_questions}

    assert len(scored.diligence_questions) == 4
    assert categories == {
        DiligenceQuestionCategory.COMPANY,
        DiligenceQuestionCategory.FINANCING_TERMS,
        DiligenceQuestionCategory.CUSTOMERS,
        DiligenceQuestionCategory.LEGAL_COMPLIANCE,
    }
    assert all(question.rank_score > 0 for question in scored.diligence_questions)
    assert all(question.missing_evidence for question in scored.diligence_questions)
    _assert_questions_have_lineage(scored.diligence_questions)


def test_empty_evidence_includes_research_gap_question() -> None:
    scored = score_evidence_store(
        _store(evidence=[], claims=[]),
        config=AppConfig(data_dir=Path("data")),
        research_context=DiligenceResearchContext(
            planned_task_count=2,
            incomplete_search_count=1,
            no_prepared_result_companies=["ResearchCo"],
        ),
    )

    question = next(
        question
        for question in scored.diligence_questions
        if question.category == DiligenceQuestionCategory.MARKET
    )

    assert "completed external research searches" in question.missing_evidence
    assert "prepared external research for requested companies" in question.missing_evidence
    _assert_questions_have_lineage(scored.diligence_questions)


def test_warning_only_research_context_creates_research_question() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_research", company_name="ResearchCo"),
        config=AppConfig(data_dir=Path("data")),
        research_context=DiligenceResearchContext(
            planned_task_count=1,
            imported_record_count=1,
            warning_count=1,
        ),
    )

    question = next(
        question
        for question in scored.diligence_questions
        if question.category == DiligenceQuestionCategory.MARKET
    )

    assert question.missing_evidence == ["resolved external research warnings"]
    _assert_questions_have_lineage(scored.diligence_questions)


def test_unknown_stage_valuation_risk_stays_conservative_with_traction() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $500M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
    ]
    claims = [
        _claim("valuation cap", "$500M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == CompanyStage.UNKNOWN
    assert scored.pmf_level == PMFLevel.DEVELOPING
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


@pytest.mark.parametrize(
    ("stage_text", "valuation_text", "expected_stage", "expected_risk", "triggers_gate"),
    [
        (
            "Pre-seed company with ARR revenue growth and paid customers.",
            "$25M",
            CompanyStage.PRE_SEED,
            ValuationRisk.MEDIUM,
            False,
        ),
        (
            "Seed company with a beta design partner.",
            "$60M",
            CompanyStage.SEED,
            ValuationRisk.HIGH,
            True,
        ),
        (
            "Series A company with ARR revenue growth and paid customers.",
            "$150M",
            CompanyStage.SERIES_A,
            ValuationRisk.MEDIUM,
            False,
        ),
        (
            "ARR revenue growth with paid customers and retention.",
            "$40M",
            CompanyStage.UNKNOWN,
            ValuationRisk.HIGH,
            True,
        ),
    ],
)
def test_balanced_stage_valuation_thresholds(
    stage_text: str,
    valuation_text: str,
    expected_stage: CompanyStage,
    expected_risk: ValuationRisk,
    triggers_gate: bool,
) -> None:
    evidence = [
        _evidence(
            "ev_terms",
            f"{stage_text} Valuation cap {valuation_text}. Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", valuation_text, "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.company_stage == expected_stage
    assert scored.valuation_risk == expected_risk
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    ) is triggers_gate


@pytest.mark.parametrize(
    ("stage_text", "valuation_text"),
    [
        ("Series B stage with paid customers and retention.", "$2B"),
        ("Defense pilot with design partner.", "$500M"),
    ],
)
def test_late_stage_valuation_risk_keeps_absolute_upper_threshold(
    stage_text: str,
    valuation_text: str,
) -> None:
    evidence = [
        _evidence(
            "ev_terms",
            f"{stage_text} Valuation cap {valuation_text}. Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", valuation_text, "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_reports_missing_inputs_without_inventing_data() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_strong", company_name="StrongCo"),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.entry_valuation == 8_000_000
    assert scored.net_return.net_return_multiple is None
    assert scored.net_return.missing_inputs == [
        "ownership",
        "dilution",
        "fees or carry",
        "gross exit scenario",
    ]
    assert "did not invent a net return" in scored.net_return.explanation
    assert _score_factor(scored, "Valuation and net return").evidence_ids == [
        "ev_terms",
        "ev_funding",
        "ev_traction",
    ]


def test_net_return_math_uses_cited_inputs_when_available() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_return",
            "Investor ownership 20%. Estimated dilution 20%. SPV expenses 5%. Carry 20%. "
            "Exit value $1B.",
        ),
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

    assert scored.net_return.net_return_multiple == 76.38
    assert scored.net_return.estimated_ownership_percent == 20
    assert scored.net_return.missing_inputs == []
    assert scored.net_return.evidence_ids == ["ev_terms", "ev_return"]
    assert _score_factor(scored, "Valuation and net return").score == 20


def test_net_return_math_requires_complete_fees_and_carry_inputs() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_return", "Estimated dilution 20%. Carry 20%. Exit value $1B."),
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

    assert scored.net_return.net_return_multiple is None
    assert "fees or carry" in scored.net_return.missing_inputs


def test_net_return_math_ignores_negated_exit_scenarios() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_return",
            "Estimated dilution 20%. SPV expenses 5%. Carry 20%. "
            "No exit value $1B was provided.",
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

    assert scored.net_return.net_return_multiple is None
    assert "gross exit scenario" in scored.net_return.missing_inputs


def test_net_return_math_adds_round_size_to_pre_money_valuation() -> None:
    evidence = [
        _evidence("ev_valuation", "Pre-money valuation $40M."),
        _evidence("ev_round", "Round size $20M."),
        _evidence(
            "ev_return",
            "Investor ownership 1%. Estimated dilution 20%. SPV expenses 5%. Carry 20%. "
            "Exit value $1B.",
        ),
    ]
    claims = [
        _claim("pre-money valuation", "$40M", "ev_valuation"),
        _claim("round size", "$20M", "ev_round"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.entry_valuation == 60_000_000
    assert scored.net_return.net_return_multiple == 10.35
    assert scored.net_return.evidence_ids == ["ev_valuation", "ev_round", "ev_return"]


def test_net_return_math_uses_implied_invested_capital() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_return",
            "Investor ownership 0.1%. Estimated dilution 20%. SPV expenses 5%. "
            "Carry 20%. Exit value $1B.",
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

    assert scored.net_return.net_return_multiple == 76.38
    assert scored.net_return.estimated_ownership_percent == 0.1
    assert _score_factor(scored, "Valuation and net return").score == 16


def test_net_return_math_does_not_understate_small_ownership_returns() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed stage. Post-money valuation $1M. ARR revenue growth with paid "
            "customers and retention. Lead investor committed.",
        ),
        _evidence(
            "ev_return",
            "Investor ownership 1%. Estimated dilution 0%. SPV expenses 0%. "
            "Carry 0%. Exit value $10M.",
        ),
    ]
    claims = [_claim("post-money valuation", "$1M", "ev_terms")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.net_return_multiple == 10
    assert scored.recommendation == Recommendation.INVEST
    assert not any(
        gate.name == "Verified return below capital back"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_treats_zero_ownership_as_verified_downside() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M. ARR "
            "revenue growth with paid customers and retention. Lead investor committed.",
        ),
        _evidence(
            "ev_return",
            "Investor ownership 0%. Estimated dilution 0%. SPV expenses 0%. "
            "Carry 0%. Exit value $1B.",
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

    assert scored.net_return.net_return_multiple == 0
    assert scored.net_return.estimated_ownership_percent == 0
    assert scored.net_return.missing_inputs == []
    assert scored.recommendation == Recommendation.PASS
    assert any(
        gate.name == "Verified return below capital back"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_treats_zero_ownership_as_complete_without_fees() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M. ARR "
            "revenue growth with paid customers and retention. Lead investor committed.",
        ),
        _evidence(
            "ev_return",
            "Investor ownership 0%. Estimated dilution 50%. Exit value $1B.",
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

    assert scored.net_return.net_return_multiple == 0
    assert scored.net_return.missing_inputs == []
    assert "fees or carry" not in scored.net_return.explanation
    assert any(
        gate.name == "Verified return below capital back"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_prefers_itemized_fees_over_summary_total() -> None:
    evidence = [
        _evidence("ev_terms", "Series A stage. Post-money valuation $100M."),
        _evidence(
            "ev_return",
            "Investor ownership 100%. Estimated dilution 0%. SPV expenses 10%. "
            "Carry 20%. Total fees and carry 30%. Exit value $130M.",
        ),
    ]
    claims = [_claim("post-money valuation", "$100M", "ev_terms")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.net_return_multiple == 1.13
    assert scored.net_return.estimated_fees_and_carry_percent == 30
    assert not any(
        gate.name == "Verified return below capital back"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_does_not_treat_summary_fees_as_itemized_carry() -> None:
    evidence = [
        _evidence("ev_terms", "Series A stage. Post-money valuation $100M."),
        _evidence(
            "ev_return",
            "Investor ownership 100%. Estimated dilution 0%. SPV expenses 10%. "
            "Total fees and carry 30%. Exit value $130M.",
        ),
    ]
    claims = [_claim("post-money valuation", "$100M", "ev_terms")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.net_return_multiple == 0.91
    assert scored.net_return.estimated_fees_and_carry_percent == 30
    assert any(
        gate.name == "Verified return below capital back"
        for gate in scored.triggered_kill_gates
    )


def test_net_return_math_rejects_itemized_carry_above_100_percent() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Post-money valuation $1M."),
        _evidence(
            "ev_return",
            "Investor ownership 100%. Estimated dilution 0%. SPV expenses 5%. "
            "Carry 150%. Exit value $10M.",
        ),
    ]
    claims = [_claim("post-money valuation", "$1M", "ev_terms")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.net_return_multiple is None
    assert scored.net_return.estimated_fees_and_carry_percent is None
    assert "fees or carry" in scored.net_return.missing_inputs


def test_net_return_math_rejects_impossible_ownership_percentages() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_return",
            "Investor ownership 120%. Estimated dilution 20%. SPV expenses 5%. "
            "Carry 20%. Exit value $1B.",
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

    assert scored.net_return.estimated_ownership_percent is None
    assert scored.net_return.net_return_multiple is None
    assert "ownership" in scored.net_return.missing_inputs


@pytest.mark.parametrize(
    "ownership_text",
    [
        "Founder ownership 60%.",
        "Employee estimated ownership 10%.",
    ],
)
def test_net_return_math_ignores_cap_table_ownership_as_check_input(
    ownership_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_return",
            f"{ownership_text} Estimated dilution 20%. SPV expenses 5%. "
            "Carry 20%. Exit value $1B.",
        ),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
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

    assert scored.net_return.estimated_ownership_percent is None
    assert scored.net_return.net_return_multiple is None
    assert "ownership" in scored.net_return.missing_inputs
    assert _score_factor(scored, "Valuation and net return").score == 13


def test_net_return_math_needs_round_size_for_pre_money_valuation() -> None:
    evidence = [_evidence("ev_valuation", "Pre-money valuation $40M.")]
    claims = [_claim("pre-money valuation", "$40M", "ev_valuation")]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.net_return.entry_valuation is None
    assert scored.net_return.net_return_multiple is None
    assert scored.net_return.missing_inputs == [
        "verified round size for pre-money valuation"
    ]
    assert "verified round size" in scored.net_return.explanation
    missing_terms_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert missing_terms_gate.evidence_ids == ["ev_valuation"]
    assert missing_terms_gate.reason == (
        "A verified pre-money valuation needs a verified round size before "
        "Hail Mary can calculate the entry valuation."
    )


def test_readable_pricing_is_required_before_clearing_key_terms() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap TBD. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("valuation cap", "TBD", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    missing_terms_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert missing_terms_gate.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert scored.net_return.missing_inputs == ["machine-readable entry valuation"]


def test_readable_pricing_skips_unreadable_higher_priority_candidates() -> None:
    evidence = [
        _evidence("ev_post_money", "Post-money valuation TBD."),
        _evidence("ev_cap", "Valuation cap $8M."),
        _evidence("ev_round", "Round size $1M."),
        _evidence("ev_discount", "Discount 20%."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("post-money valuation", "TBD", "ev_post_money"),
        _claim("valuation cap", "$8M", "ev_cap"),
        _claim("round size", "$1M", "ev_round"),
        _claim("discount", "20%", "ev_discount"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    missing_terms_gate = next(
        gate
        for gate in scored.kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert not missing_terms_gate.triggered
    assert missing_terms_gate.support_status == ScoreSupportStatus.VERIFIED
    assert missing_terms_gate.evidence_ids == ["ev_cap"]
    assert scored.net_return.entry_valuation == 8_000_000
    assert "machine-readable entry valuation" not in scored.net_return.missing_inputs


def test_key_terms_gate_cites_readable_pricing_evidence_when_claims_are_crowded() -> None:
    evidence = [
        _evidence("ev_discount", "Discount 20%."),
        _evidence("ev_minimum", "Minimum investment $1K."),
        _evidence("ev_round", "Round size $1M."),
        _evidence("ev_security", "Security is a SAFE."),
        _evidence("ev_pro_rata", "Pro rata rights are included."),
        _evidence("ev_cap", "Valuation cap $8M."),
    ]
    claims = [
        _claim("discount", "20%", "ev_discount"),
        _claim("minimum investment", "$1K", "ev_minimum"),
        _claim("round size", "$1M", "ev_round"),
        _claim("security", "SAFE", "ev_security"),
        _claim("pro rata", "included", "ev_pro_rata"),
        _claim("valuation cap", "$8M", "ev_cap"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    missing_terms_gate = next(
        gate
        for gate in scored.kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert not missing_terms_gate.triggered
    assert missing_terms_gate.evidence_ids == ["ev_cap"]


def test_valuation_factor_cites_stage_and_traction_evidence() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_stage", "Seed stage."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
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

    valuation_factor = _score_factor(scored, "Valuation and net return")
    assert valuation_factor.evidence_ids == ["ev_terms", "ev_stage", "ev_traction"]


def test_non_positive_entry_valuation_does_not_clear_pricing_gate() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $0. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed and seed round is active."),
    ]
    claims = [
        _claim("valuation cap", "$0", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    missing_terms_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert missing_terms_gate.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.net_return.entry_valuation is None


def test_does_not_have_customers_does_not_count_as_positive_pmf() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Pre-seed company does not have customers. Valuation cap $30M. "
            "Discount 20%. Round size $1M.",
        )
    ]
    claims = [
        _claim("valuation cap", "$30M", "ev_terms"),
        _claim("discount", "20%", "ev_terms"),
        _claim("round size", "$1M", "ev_terms"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    assert scored.pmf_level == PMFLevel.UNKNOWN
    assert scored.valuation_risk == ValuationRisk.HIGH
    assert any(
        gate.name == "Valuation far ahead of evidence"
        for gate in scored.triggered_kill_gates
    )


@pytest.mark.parametrize(
    "return_text",
    [
        "Estimated dilution 20%. Customer fees are 5%. Exit value $1B.",
        "Estimated dilution 20%. Management expenses are 5%. Exit value $1B.",
    ],
)
def test_net_return_math_ignores_non_investment_fees_without_context(
    return_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_return", return_text),
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

    assert scored.net_return.net_return_multiple is None
    assert "fees or carry" in scored.net_return.missing_inputs


def test_missing_key_terms_gate_is_not_verified_without_verified_claims() -> None:
    scored = score_evidence_store(
        _store(
            evidence=[_evidence("ev_notes", "Current source text exists.")],
            claims=[],
        ),
        config=AppConfig(data_dir=Path("data")),
    )

    missing_terms_gate = next(
        gate
        for gate in scored.kill_gates
        if gate.name == "Missing key investment terms"
    )
    assert not missing_terms_gate.triggered
    assert missing_terms_gate.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert missing_terms_gate.reason == "No verified valuation or valuation-cap term was found."


def test_stage_pmf_factor_marks_partial_support_needs_diligence() -> None:
    scored = score_evidence_store(
        _store(
            evidence=[_evidence("ev_stage", "Pre-seed company.")],
            claims=[],
        ),
        config=AppConfig(data_dir=Path("data")),
    )

    stage_factor = _score_factor(scored, "Stage and product-market fit")
    assert stage_factor.evidence_ids == ["ev_stage"]
    assert stage_factor.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert stage_factor.missing_inputs == [
        "customer, revenue, retention, usage, pilot, or design-partner proof"
    ]


def test_stage_pmf_factor_preserves_stage_evidence_with_many_pmf_records() -> None:
    evidence = [
        *[
            _evidence(
                f"ev_pmf_{index}",
                f"ARR revenue growth with paid customers and retention cohort {index}.",
            )
            for index in range(6)
        ],
        _evidence("ev_stage", "Seed stage."),
    ]
    scored = score_evidence_store(
        _store(evidence=evidence, claims=[]),
        config=AppConfig(data_dir=Path("data")),
    )

    stage_factor = _score_factor(scored, "Stage and product-market fit")
    assert stage_factor.support_status == ScoreSupportStatus.VERIFIED
    assert "ev_stage" in stage_factor.evidence_ids
    assert "ev_pmf_0" in stage_factor.evidence_ids


def test_evidence_authority_with_unknown_freshness_needs_diligence() -> None:
    scored = score_evidence_store(
        _store(
            evidence=[
                _evidence(
                    "ev_unknown",
                    "Valuation cap $8M.",
                    source_freshness=SourceFreshness.UNKNOWN,
                )
            ],
            claims=[_claim("valuation cap", "$8M", "ev_unknown")],
        ),
        config=AppConfig(data_dir=Path("data")),
    )

    authority_factor = _score_factor(scored, "Evidence authority and freshness")
    assert authority_factor.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert authority_factor.missing_inputs == ["current source dates"]


def test_evidence_authority_cites_stale_or_unknown_freshness_records_first() -> None:
    evidence = [
        *[
            _evidence(f"ev_current_{index}", f"Current evidence {index}.")
            for index in range(6)
        ],
        _evidence(
            "ev_stale",
            "Stale evidence requiring freshness penalty.",
            source_freshness=SourceFreshness.STALE,
        ),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=[]),
        config=AppConfig(data_dir=Path("data")),
    )

    authority_factor = _score_factor(scored, "Evidence authority and freshness")
    assert authority_factor.evidence_ids[0] == "ev_stale"


def test_fundability_factor_does_not_ask_for_funding_already_present() -> None:
    scored = score_evidence_store(
        _store(
            evidence=[_evidence("ev_funding", "Lead investor committed.")],
            claims=[],
        ),
        config=AppConfig(data_dir=Path("data")),
    )

    fundability_factor = _score_factor(scored, "Fundability and next-round risk")
    assert fundability_factor.evidence_ids == ["ev_funding"]
    assert "lead investor, institutional investor, or follow-on financing evidence" not in (
        fundability_factor.missing_inputs
    )
    assert fundability_factor.missing_inputs == [
        "verified deal terms",
        "customer, revenue, retention, or usage evidence",
    ]


def test_valuation_net_return_factor_keeps_all_return_input_evidence_ids() -> None:
    evidence = [
        _evidence("ev_valuation", "Pre-money valuation $40M."),
        _evidence("ev_round", "Round size $20M."),
        _evidence("ev_ownership", "Investor ownership 1%."),
        _evidence("ev_dilution", "Estimated dilution 20%."),
        _evidence("ev_fees", "SPV expenses 5%."),
        _evidence("ev_carry", "Carry 20%."),
        _evidence("ev_exit", "Exit value $1B."),
    ]
    claims = [
        _claim("pre-money valuation", "$40M", "ev_valuation"),
        _claim("round size", "$20M", "ev_round"),
    ]

    scored = score_evidence_store(
        _store(evidence=evidence, claims=claims),
        config=AppConfig(data_dir=Path("data")),
    )

    factor = _score_factor(scored, "Valuation and net return")
    assert factor.evidence_ids == [
        "ev_valuation",
        "ev_round",
        "ev_ownership",
        "ev_dilution",
        "ev_fees",
        "ev_carry",
        "ev_exit",
    ]


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


def test_score_evidence_store_explains_risk_cap_below_minimum_check() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_strong", company_name="StrongCo"),
        config=AppConfig(data_dir=Path("data"), min_check=5_000),
    )

    no_check_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "No available check size"
    )
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert "Risk caps" in no_check_gate.reason
    assert "minimum check" in no_check_gate.reason


def test_score_evidence_store_lowers_check_for_company_exposure_limit() -> None:
    scored = score_evidence_store(
        _high_confidence_store(deal_id="deal_exposure", company_name="ExposureCo"),
        config=AppConfig(
            data_dir=Path("data"),
            capital_budget=100_000,
            max_company_exposure_percent=Decimal("5"),
        ),
    )

    company_check = next(
        check
        for check in scored.check_sizing.exposure_checks
        if check.dimension == PortfolioExposureDimension.COMPANY
    )
    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size == 5_000
    assert scored.check_size in {0, 1_000, 2_500, 5_000, 7_500, 10_000}
    assert scored.check_sizing.exposure_cap == 5_000
    assert "exposure_limit_applied" in scored.check_sizing.reason_codes
    assert company_check.key == "exposureco"
    assert company_check.available_capacity == 5_000


def test_score_evidence_store_blocks_when_exposure_limit_below_minimum() -> None:
    scored = score_evidence_store(
        _high_confidence_store(deal_id="deal_blocked", company_name="BlockedCo"),
        config=AppConfig(
            data_dir=Path("data"),
            capital_budget=100_000,
            max_company_exposure_percent=Decimal("0.5"),
        ),
    )

    no_check_gate = next(
        gate
        for gate in scored.triggered_kill_gates
        if gate.name == "No available check size"
    )
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.check_sizing.exposure_cap == 500
    assert "exposure_cap_below_minimum" in scored.check_sizing.reason_codes
    assert scored.check_sizing.exposure_checks[0].blocking is True
    assert "exposure limits" in no_check_gate.reason


def test_score_evidence_store_treats_missing_exposure_key_as_unknown_bucket() -> None:
    scored = score_evidence_store(
        _strong_store(deal_id="deal_unknown_category", company_name="Unknown Category"),
        config=AppConfig(
            data_dir=Path("data"),
            capital_budget=100_000,
            max_category_exposure_percent=Decimal("0.5"),
        ),
    )

    category_check = next(
        check
        for check in scored.check_sizing.exposure_checks
        if check.dimension == PortfolioExposureDimension.CATEGORY
    )
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert category_check.key == "unknown"
    assert category_check.applied is True
    assert category_check.blocking is True
    assert "exposure_cap_below_minimum" in scored.check_sizing.reason_codes


def test_score_evidence_store_caps_high_fundability_risk_check_size() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_positive_funding", "Lead investor committed."),
        _evidence("ev_negative_funding", "No follow-on financing is confirmed."),
        _evidence(
            "ev_return",
            "Investor ownership 100%. Estimated dilution 20%. SPV expenses 5%. "
            "Carry 20%. Exit value $1B.",
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

    assert scored.total_score >= 75
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size == 2_500


def test_score_evidence_store_gates_verified_sub_1x_return_math() -> None:
    evidence = [
        _evidence("ev_terms", "Seed stage. Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers and retention."),
        _evidence("ev_funding", "Lead investor committed."),
        _evidence(
            "ev_return",
            "Investor ownership 0.1%. Estimated dilution 20%. SPV expenses 5%. "
            "Carry 20%. Exit value $1M.",
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

    assert scored.net_return.net_return_multiple == 0.1
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert any(
        gate.name == "Verified return below capital back"
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
        "The company is operating without customers now.",
        "A company without revenue is not fundable.",
        "A company without customers or revenue is not fundable.",
        "The company operates without customers, revenue, or retention now.",
        "The company operates without customers, revenue, or retention is not fundable.",
        "The company has no actual customers yet.",
        "The company has no customer revenue yet.",
        "The company lacks customers and revenue.",
        "The company operates without any customers or revenue and lacks usage and retention.",
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


def test_score_evidence_store_keeps_positive_comma_clause_after_without_negation() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_traction",
            "The company launched without customers, revenue is now $500K.",
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

    assert scored.pmf_level == PMFLevel.DEVELOPING
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_traction"
    ]


def test_score_evidence_store_keeps_positive_coordinated_clause_after_without_negation() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_traction",
            "The company launched without customers, revenue and retention are now strong.",
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

    assert scored.pmf_level == PMFLevel.DEVELOPING
    assert _score_factor(scored, "Product-market fit evidence").evidence_ids == [
        "ev_traction"
    ]


@pytest.mark.parametrize(
    "traction_text",
    [
        "The company launched without customers and revenue is now $500K.",
        "The company launched without customers, revenue and retention metrics are now strong.",
    ],
)
def test_score_evidence_store_keeps_positive_follow_on_clause_after_without_negation(
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


@pytest.mark.parametrize(
    "traction_text",
    [
        "The plan shows projected revenue next year.",
        "The plan shows projected ARR growth next year.",
        "The roadmap targets future customers after launch.",
        "The company has not-yet customers.",
    ],
)
def test_score_evidence_store_ignores_planned_or_not_yet_traction(
    traction_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_planned_traction", traction_text),
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

    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert _score_factor(scored, "Next-round fundability").evidence_ids == ["ev_funding"]


def test_score_evidence_store_treats_mixed_funding_as_high_risk() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_positive_funding", "Lead investor committed."),
        _evidence("ev_negative_funding", "No institutional follow-on is confirmed."),
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert fundability_factor.evidence_ids == [
        "ev_positive_funding",
        "ev_negative_funding",
    ]
    assert fundability_factor.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert (
        "resolved lead investor, institutional investor, or follow-on financing conflict"
        in fundability_factor.missing_inputs
    )


def test_fundability_factor_cites_stale_traction_when_it_drives_risk() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            source_freshness=SourceFreshness.STALE,
        ),
        _evidence("ev_funding", "Lead investor committed."),
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert fundability_factor.evidence_ids == ["ev_funding", "ev_traction"]
    assert "current customer, revenue, retention, or usage evidence" in (
        fundability_factor.missing_inputs
    )


def test_fundability_risk_is_high_for_stale_traction_without_funding() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            source_freshness=SourceFreshness.STALE,
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert fundability_factor.evidence_ids == ["ev_traction"]
    assert "lead investor, institutional investor, or follow-on financing evidence" in (
        fundability_factor.missing_inputs
    )
    assert "current customer, revenue, retention, or usage evidence" in (
        fundability_factor.missing_inputs
    )


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

    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert _score_factor(scored, "Next-round fundability").evidence_ids == ["ev_funding"]


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

    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert _score_factor(scored, "Next-round fundability").evidence_ids == [
        "ev_negative_funding"
    ]


@pytest.mark.parametrize(
    "funding_text",
    [
        "The company expects a future lead investor.",
        "The round has planned institutional investors.",
        "The company has not-yet secured follow-on financing.",
    ],
)
def test_score_evidence_store_treats_planned_funding_as_missing(
    funding_text: str,
) -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_planned_funding", funding_text),
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

    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert _score_factor(scored, "Next-round fundability").evidence_ids == [
        "ev_planned_funding"
    ]


def test_score_evidence_store_does_not_treat_future_round_plan_as_missing_funding() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence("ev_funding", "Lead investor committed; future Series A planned."),
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


def test_score_evidence_store_keeps_benign_institutional_concern_phrases() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence("ev_traction", "ARR revenue growth with paid customers."),
        _evidence(
            "ev_funding",
            "Institutional investor committed. No institutional investor concerns.",
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.LOW
    assert fundability_factor.evidence_ids == ["ev_funding"]
    assert (
        "resolved lead investor, institutional investor, or follow-on financing conflict"
        not in fundability_factor.missing_inputs
    )


@pytest.mark.parametrize(
    "funding_text",
    [
        "Lead investor committed. No follow-on concerns.",
        "Seed round active. No seed funding concerns.",
    ],
)
def test_score_evidence_store_keeps_benign_seed_or_follow_on_concern_phrases(
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.LOW
    assert fundability_factor.evidence_ids == ["ev_funding"]
    assert (
        "resolved lead investor, institutional investor, or follow-on financing conflict"
        not in fundability_factor.missing_inputs
    )


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


def test_score_evidence_store_passes_when_support_is_stale_only() -> None:
    evidence = [
        _evidence(
            "ev_terms",
            "Valuation cap $8M. Discount 20%. Round size $1M.",
            source_freshness=SourceFreshness.STALE,
        ),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            source_freshness=SourceFreshness.STALE,
        ),
        _evidence(
            "ev_funding",
            "Lead investor committed and seed round is active.",
            source_freshness=SourceFreshness.STALE,
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

    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert _score_factor(scored, "Evidence authority and freshness").evidence_ids[0] == (
        "ev_terms"
    )
    assert "current source dates" in _score_factor(
        scored,
        "Evidence authority and freshness",
    ).missing_inputs


def test_score_evidence_store_treats_unknown_freshness_support_as_not_current() -> None:
    evidence = [
        _evidence("ev_terms", "Valuation cap $8M. Discount 20%. Round size $1M."),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            source_freshness=SourceFreshness.UNKNOWN,
        ),
        _evidence(
            "ev_funding",
            "Lead investor committed and seed round is active.",
            source_freshness=SourceFreshness.UNKNOWN,
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

    fundability_factor = _score_factor(scored, "Next-round fundability")
    assert scored.fundability_risk == FundabilityRisk.HIGH
    assert fundability_factor.evidence_ids == ["ev_funding", "ev_traction"]
    assert "current customer, revenue, retention, or usage evidence" in (
        fundability_factor.missing_inputs
    )


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
    terms_factor = _score_factor(scored, "Deal terms and platform access")
    assert terms_factor.support_status == ScoreSupportStatus.NEEDS_DILIGENCE
    assert terms_factor.missing_inputs == ["verified valuation or valuation cap"]


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


def test_render_markdown_memo_includes_v2_score_context_and_citations() -> None:
    store = _strong_store(deal_id="deal_strong", company_name="StrongCo")
    scored = score_evidence_store(store, config=AppConfig(data_dir=Path("data")))

    markdown = render_markdown_memo(scored, store)

    assert "**Stage:** seed" in markdown
    assert "**Valuation risk:** low" in markdown
    assert (
        "**Net return math:** $8M entry valuation; missing ownership, dilution, fees or carry, "
        "gross exit scenario"
    ) in markdown
    assert "- Valuation and net return: 13/20." in markdown
    assert "Support: NEEDS_DILIGENCE." in markdown
    assert "Evidence: ev\\_terms." in markdown


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
    assert scored_by_company["Alpha Reserve"].allocation_scenario.follow_on_reserve == 2_500
    assert scored_by_company["Alpha Reserve"].allocation_scenario.capital_before_check == 2_500
    assert scored_by_company["Alpha Reserve"].allocation_scenario.capital_after_check == 0


def test_score_latest_ingestion_subtracts_recorded_portfolio_investments(
    tmp_path: Path,
) -> None:
    stores = [
        _strong_store(deal_id="deal_alpha", company_name="Alpha Ledger"),
        _strong_store(deal_id="deal_zeta", company_name="Zeta Ledger"),
    ]
    _write_ingestion_summary(tmp_path, stores)
    config = AppConfig(data_dir=tmp_path / "data", capital_budget=5_000)
    add_portfolio_investment(
        config=config,
        company_name="PriorCo",
        amount=2_500,
        invested_on=date(2026, 6, 23),
    )

    result = score_latest_ingestion(config=config)

    scored_by_company = {deal.company_name: deal for deal in result.scored_deals}
    assert scored_by_company["Alpha Ledger"].recommendation == Recommendation.INVEST
    assert scored_by_company["Alpha Ledger"].check_size == 2_500
    assert scored_by_company["Zeta Ledger"].recommendation == Recommendation.PASS
    assert scored_by_company["Zeta Ledger"].check_size == 0

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    assert "Recorded existing investments: $2,500" in report
    assert "Allocatable capital for new checks: $2,500" in report
    assert "Net return math uses recorded investments plus newly allocated checks" in report


def test_score_latest_ingestion_uses_prior_investments_for_company_exposure(
    tmp_path: Path,
) -> None:
    store = _high_confidence_store(deal_id="deal_same_company", company_name="SameCo")
    _write_ingestion_summary(tmp_path, [store])
    config = AppConfig(
        data_dir=tmp_path / "data",
        capital_budget=100_000,
        max_company_exposure_percent=Decimal("5"),
        max_category_exposure_percent=Decimal("5"),
    )
    add_portfolio_investment(
        config=config,
        company_name="SameCo",
        amount=2_500,
        invested_on=date(2026, 6, 23),
    )

    result = score_latest_ingestion(config=config)

    scored = result.scored_deals[0]
    company_check = next(
        check
        for check in scored.check_sizing.exposure_checks
        if check.dimension == PortfolioExposureDimension.COMPANY
    )
    category_check = next(
        check
        for check in scored.check_sizing.exposure_checks
        if check.dimension == PortfolioExposureDimension.CATEGORY
    )
    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size == 2_500
    assert scored.capital_remaining_before == 97_500
    assert company_check.exposure_before == 2_500
    assert company_check.available_capacity == 2_500
    assert category_check.key == "fintech"
    assert category_check.exposure_before == 2_500
    assert category_check.available_capacity == 2_500


def test_score_latest_ingestion_applies_category_stage_and_confidence_exposure(
    tmp_path: Path,
) -> None:
    stores = [
        _category_stage_store(deal_id="deal_alpha_group", company_name="Alpha Group"),
        _category_stage_store(deal_id="deal_zeta_group", company_name="Zeta Group"),
    ]
    _write_ingestion_summary(tmp_path, stores)

    result = score_latest_ingestion(
        config=AppConfig(
            data_dir=tmp_path / "data",
            capital_budget=100_000,
            max_category_exposure_percent=Decimal("2.5"),
            max_stage_exposure_percent=Decimal("2.5"),
            max_medium_confidence_exposure_percent=Decimal("2.5"),
        )
    )

    scored_by_company = {deal.company_name: deal for deal in result.scored_deals}
    alpha = scored_by_company["Alpha Group"]
    zeta = scored_by_company["Zeta Group"]
    assert alpha.recommendation == Recommendation.INVEST
    assert alpha.check_size == 2_500
    assert zeta.recommendation == Recommendation.PASS
    assert zeta.check_size == 0
    assert zeta.check_size in {0, 1_000, 2_500, 5_000, 7_500, 10_000}

    zeta_checks = {
        check.dimension: check
        for check in zeta.check_sizing.exposure_checks
        if check.applied
    }
    assert zeta_checks[PortfolioExposureDimension.CATEGORY].key == "fintech"
    assert zeta_checks[PortfolioExposureDimension.CATEGORY].exposure_before == 2_500
    assert zeta_checks[PortfolioExposureDimension.CATEGORY].available_capacity == 0
    assert zeta_checks[PortfolioExposureDimension.STAGE].key == "pre_seed"
    assert zeta_checks[PortfolioExposureDimension.STAGE].exposure_before == 2_500
    assert zeta_checks[PortfolioExposureDimension.SOURCE_CONFIDENCE].key == "medium"
    assert zeta_checks[
        PortfolioExposureDimension.SOURCE_CONFIDENCE
    ].available_capacity == 0
    assert "exposure_cap_below_minimum" in zeta.check_sizing.reason_codes
    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    assert "Exposure limits left no room for an allowed nonzero check." in report


def test_score_latest_ingestion_ceils_high_precision_reserve_percent(
    tmp_path: Path,
) -> None:
    store = _strong_store(deal_id="deal_precise_reserve", company_name="Precise Reserve")
    _write_ingestion_summary(tmp_path, [store])

    result = score_latest_ingestion(
        config=AppConfig(
            data_dir=tmp_path / "data",
            capital_budget=100_000,
            min_check=5_000,
            max_check=5_000,
            reserve_percent=Decimal("95.000000000000000000000000001"),
        )
    )

    scored = result.scored_deals[0]
    assert scored.recommendation == Recommendation.PASS
    assert scored.check_size == 0
    assert scored.capital_remaining_before == 4_999

    assert result.portfolio_report_path is not None
    report = result.portfolio_report_path.read_text(encoding="utf-8")
    assert "Reserve: $95,001 (95% reserve)" in report
    assert "Allocatable capital after reserve: $4,999" in report


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
        "Seed stage ARR revenue growth with paid customers and retention. "
        "Lead investor committed.",
        encoding="utf-8",
    )
    config = AppConfig(data_dir=tmp_path / "data")
    ingest_folder(root, config=config)

    result = score_latest_ingestion(config=config)

    scored = result.scored_deals[0]
    assert scored.recommendation == Recommendation.INVEST
    assert scored.check_size > 0
    assert _score_factor(scored, "Evidence authority and freshness").score == 11


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
    assert "| 1 | Empty A | 6/100 | No usable source-linked evidence was available. |" in report
    assert "| 2 | Empty B | 6/100 | No usable source-linked evidence was available. |" in report


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
        "| Configured | 3x | $2,500 | $7,500 | $6,750 | $50 | $850 | "
        "$5,900 | $3,350 | 2.31x |"
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


def test_render_portfolio_report_bounds_extreme_return_assumption_formatting(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        gross_return_multiple=Decimal("1e1000000"),
    )
    scored = score_evidence_store(
        _strong_store(deal_id="deal_extreme", company_name="Extreme Math"),
        config=config,
    )

    report = render_portfolio_report([scored], config=config)

    assert "Gross return multiple: 1E+1000000x" in report
    assert "| Configured | 1E+1000000x | $2,500 | $2.5E+1000003 |" in report
    assert len(report) < 20_000


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

    assert "VERIFIED: Stage and product-market fit scored" in report
    assert "Evidence: ev\\_funding, ev\\_traction." in report
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


def test_score_deals_run_only_reserve_percent_replaces_saved_reserve_dollars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("reserve_dollars: 1500\n", encoding="utf-8")
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "score-deals",
            "--data-dir",
            str(tmp_path / "data"),
            "--capital-budget",
            "10000",
            "--reserve-percent",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    report = (tmp_path / "data" / "reports" / "portfolio-comparison-report.md").read_text(
        encoding="utf-8"
    )
    assert "Reserve: $1,000 (10% reserve)" in report


def test_score_deals_run_only_reserve_dollars_replaces_saved_reserve_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("reserve_percent: 10\n", encoding="utf-8")
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "score-deals",
            "--data-dir",
            str(tmp_path / "data"),
            "--capital-budget",
            "10000",
            "--reserve-dollars",
            "1500",
        ],
    )

    assert result.exit_code == 0, result.output
    report = (tmp_path / "data" / "reports" / "portfolio-comparison-report.md").read_text(
        encoding="utf-8"
    )
    assert "Reserve: $1,500 (configured reserve dollars)" in report


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


def test_score_deals_command_rejects_oversized_decimal_override(tmp_path: Path) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "score-deals",
            "--data-dir",
            str(tmp_path / "data"),
            "--gross-return-multiple",
            "1e999999999999999999",
        ],
    )

    assert result.exit_code != 0
    assert "gross return multiple is too long" in result.output
    assert "Traceback" not in result.output


def test_score_deals_malformed_portfolio_ledger_has_plain_english_error(
    tmp_path: Path,
) -> None:
    store = _strong_store(deal_id="deal_one", company_name="Deal One")
    _write_ingestion_summary(tmp_path, [store])
    ledger_path = tmp_path / "data" / "portfolio" / "ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("{bad json", encoding="utf-8")

    result = runner.invoke(app, ["score-deals", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "private portfolio ledger" in result.output
    assert "not valid JSON" in result.output
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


def _high_confidence_store(*, deal_id: str, company_name: str) -> EvidenceStore:
    evidence = [
        _evidence(
            "ev_terms_a",
            "Seed stage. Valuation cap $8M. Investor ownership 100%.",
            deal_id=deal_id,
            document_id="doc_terms_a",
        ),
        _evidence(
            "ev_terms_b",
            "Discount 20%. Round size $1M. Estimated dilution 20%. "
            "SPV expenses 5%. Carry 20%. Exit value $1B.",
            deal_id=deal_id,
            document_id="doc_terms_b",
        ),
        _evidence(
            "ev_traction",
            "ARR revenue growth with paid customers and retention.",
            deal_id=deal_id,
            document_id="doc_traction",
        ),
        _evidence(
            "ev_funding",
            "Lead investor committed and institutional seed round is active.",
            deal_id=deal_id,
            document_id="doc_funding",
        ),
        _evidence(
            "ev_category",
            "Category: fintech.",
            deal_id=deal_id,
            document_id="doc_category",
        ),
    ]
    claims = [
        _claim("valuation cap", "$8M", "ev_terms_a", deal_id=deal_id),
        _claim("discount", "20%", "ev_terms_b", deal_id=deal_id),
        _claim("round size", "$1M", "ev_terms_b", deal_id=deal_id),
    ]
    return _store(
        evidence=evidence,
        claims=claims,
        deal_id=deal_id,
        company_name=company_name,
    )


def _category_stage_store(*, deal_id: str, company_name: str) -> EvidenceStore:
    store = _strong_store(deal_id=deal_id, company_name=company_name)
    category_stage_evidence = _evidence(
        f"ev_category_{deal_id}",
        "Pre-seed company. Category: fintech.",
        deal_id=deal_id,
        document_id=f"doc_category_{deal_id}",
    )
    return store.model_copy(update={"evidence": [*store.evidence, category_stage_evidence]})


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


def _assert_questions_have_lineage(questions: list[DiligenceQuestion]) -> None:
    assert questions
    for question in questions:
        assert (
            question.evidence_ids
            or question.supporting_evidence_ids
            or question.conflicting_evidence_ids
            or question.missing_evidence
        )


def _score_factor(scored_deal: ScoredDeal, name: str) -> ScoreFactor:
    aliases = {
        "Deal-term clarity": "Deal terms and platform access",
        "Product-market fit evidence": "Stage and product-market fit",
        "Next-round fundability": "Fundability and next-round risk",
        "Evidence quality": "Evidence authority and freshness",
    }
    resolved_name = aliases.get(name, name)
    for factor in scored_deal.score_factors:
        if factor.name == resolved_name:
            return factor
    raise AssertionError(f"Missing score factor: {name}")
