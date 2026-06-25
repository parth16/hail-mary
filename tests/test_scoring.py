from __future__ import annotations

import stat
from datetime import UTC, datetime
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
    FundabilityRisk,
    PMFLevel,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
)
from hailmary.scoring import render_markdown_memo, score_evidence_store, score_latest_ingestion
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
    assert "ev_terms" in markdown
    assert "not legal, tax, financial, or investment advice" in markdown


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
