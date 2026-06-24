from __future__ import annotations

from hailmary.config import CHECK_SIZE_TIERS, AppConfig
from hailmary.schemas.evidence import (
    ClaimRecord,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)
from hailmary.schemas.scoring import (
    DiligenceQuestion,
    FundabilityRisk,
    KillGate,
    PMFLevel,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
)

TRACTION_KEYWORDS = (
    "arr",
    "revenue",
    "customer",
    "customers",
    "retention",
    "growth",
    "usage",
    "paid",
)
EARLY_PMF_KEYWORDS = ("pilot", "beta", "loi", "waitlist", "design partner")
FUNDABILITY_KEYWORDS = ("lead investor", "institutional", "series a", "seed", "follow-on")


def score_evidence_store(store: EvidenceStore, *, config: AppConfig) -> ScoredDeal:
    """Score one deal using only validated evidence-store records."""

    verified_claims = [
        claim
        for claim in store.claims
        if claim.verification_status == VerificationStatus.VERIFIED
    ]
    text_index = _text_index(store.evidence)
    pmf_level = _pmf_level(text_index)
    fundability_risk = _fundability_risk(store, verified_claims, text_index)
    kill_gates = _kill_gates(store, verified_claims)
    score_factors = _score_factors(store, verified_claims, pmf_level, fundability_risk)
    total_score = sum(factor.score for factor in score_factors)
    has_kill_gate = any(gate.triggered for gate in kill_gates)
    recommendation = (
        Recommendation.PASS
        if has_kill_gate or total_score < 70
        else Recommendation.INVEST
    )
    check_size = (
        0
        if recommendation == Recommendation.PASS
        else _check_size_for_score(total_score, config=config)
    )

    return ScoredDeal(
        deal_id=store.deal_id,
        company_name=store.company_name,
        recommendation=recommendation,
        check_size=check_size,
        total_score=total_score,
        pmf_level=pmf_level,
        fundability_risk=fundability_risk,
        kill_gates=kill_gates,
        score_factors=score_factors,
        diligence_questions=_diligence_questions(
            store,
            verified_claims,
            pmf_level=pmf_level,
            fundability_risk=fundability_risk,
        ),
    )


def _kill_gates(store: EvidenceStore, verified_claims: list[ClaimRecord]) -> list[KillGate]:
    return [
        KillGate(
            name="No usable source-linked evidence",
            triggered=not store.evidence,
            reason=(
                "No usable extracted text was available."
                if not store.evidence
                else "At least one source-linked evidence record is available."
            ),
        ),
        KillGate(
            name="Conflicting material deal terms",
            triggered=bool(store.conflicts),
            reason=(
                "One or more extracted deal terms conflict."
                if store.conflicts
                else "No conflicting deal-term claims were detected."
            ),
        ),
        KillGate(
            name="No verified deal terms",
            triggered=bool(store.evidence) and not verified_claims,
            reason=(
                "Evidence exists, but no deal-term claim was verified."
                if store.evidence and not verified_claims
                else "At least one deal-term claim has a verified citation."
            ),
        ),
    ]


def _score_factors(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
) -> list[ScoreFactor]:
    return [
        _evidence_coverage_factor(store),
        _deal_terms_factor(verified_claims, store),
        _pmf_factor(store, pmf_level),
        _fundability_factor(store, fundability_risk),
        _evidence_quality_factor(store),
    ]


def _evidence_coverage_factor(store: EvidenceStore) -> ScoreFactor:
    score = min(20, len(store.evidence) * 4)
    explanation = (
        f"Hail Mary found {len(store.evidence)} source-linked evidence records."
        if store.evidence
        else "Hail Mary found no usable source-linked evidence."
    )
    return ScoreFactor(
        name="Evidence coverage",
        score=score,
        max_score=20,
        explanation=explanation,
        evidence_ids=[evidence.id for evidence in store.evidence[:5]],
    )


def _deal_terms_factor(
    verified_claims: list[ClaimRecord],
    store: EvidenceStore,
) -> ScoreFactor:
    unique_labels = sorted({claim.label for claim in verified_claims})
    score = min(25, len(unique_labels) * 7)
    if store.conflicts:
        score = min(score, 8)
    return ScoreFactor(
        name="Deal-term clarity",
        score=score,
        max_score=25,
        explanation=(
            f"Verified deal-term labels: {', '.join(unique_labels)}."
            if unique_labels
            else "No verified deal-term labels were found."
        ),
        evidence_ids=_claim_evidence_ids(verified_claims),
    )


def _pmf_factor(store: EvidenceStore, pmf_level: PMFLevel) -> ScoreFactor:
    score_by_level = {
        PMFLevel.UNKNOWN: 4,
        PMFLevel.EARLY: 10,
        PMFLevel.DEVELOPING: 18,
    }
    matched_evidence = _evidence_matching_keywords(store.evidence, TRACTION_KEYWORDS)
    return ScoreFactor(
        name="Product-market fit evidence",
        score=score_by_level[pmf_level],
        max_score=20,
        explanation=f"Product-market fit level is {pmf_level}.",
        evidence_ids=[evidence.id for evidence in matched_evidence[:5]],
    )


def _fundability_factor(
    store: EvidenceStore,
    fundability_risk: FundabilityRisk,
) -> ScoreFactor:
    score_by_risk = {
        FundabilityRisk.UNKNOWN: 5,
        FundabilityRisk.HIGH: 6,
        FundabilityRisk.MEDIUM: 13,
        FundabilityRisk.LOW: 18,
    }
    matched_evidence = _evidence_matching_keywords(store.evidence, FUNDABILITY_KEYWORDS)
    return ScoreFactor(
        name="Next-round fundability",
        score=score_by_risk[fundability_risk],
        max_score=20,
        explanation=f"Next-round fundability risk is {fundability_risk}.",
        evidence_ids=[evidence.id for evidence in matched_evidence[:5]],
    )


def _evidence_quality_factor(store: EvidenceStore) -> ScoreFactor:
    if not store.evidence:
        score = 0
        explanation = "No evidence quality could be assessed."
    else:
        stale_count = sum(
            1 for evidence in store.evidence if evidence.source_freshness == SourceFreshness.STALE
        )
        unknown_count = sum(
            1
            for evidence in store.evidence
            if evidence.source_freshness == SourceFreshness.UNKNOWN
        )
        score = 15
        score -= min(8, stale_count * 3)
        score -= min(4, unknown_count)
        if store.conflicts:
            score = min(score, 6)
        score = max(score, 0)
        explanation = (
            f"{stale_count} stale and {unknown_count} unknown-freshness evidence records."
        )
    return ScoreFactor(
        name="Evidence quality",
        score=score,
        max_score=15,
        explanation=explanation,
        evidence_ids=[evidence.id for evidence in store.evidence[:5]],
    )


def _pmf_level(text_index: str) -> PMFLevel:
    if any(keyword in text_index for keyword in TRACTION_KEYWORDS):
        return PMFLevel.DEVELOPING
    if any(keyword in text_index for keyword in EARLY_PMF_KEYWORDS):
        return PMFLevel.EARLY
    return PMFLevel.UNKNOWN


def _fundability_risk(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    text_index: str,
) -> FundabilityRisk:
    if not store.evidence:
        return FundabilityRisk.UNKNOWN
    has_terms = bool(verified_claims)
    has_traction = any(keyword in text_index for keyword in TRACTION_KEYWORDS)
    has_funding_signal = any(keyword in text_index for keyword in FUNDABILITY_KEYWORDS)
    if has_terms and has_traction and has_funding_signal:
        return FundabilityRisk.LOW
    if has_terms and has_traction:
        return FundabilityRisk.MEDIUM
    return FundabilityRisk.HIGH


def _diligence_questions(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    *,
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    if store.conflicts:
        questions.append(
            DiligenceQuestion(
                priority=1,
                question="Resolve the conflicting deal terms in the original documents.",
                reason="The evidence store has conflicting extracted values.",
            )
        )
    if not verified_claims:
        questions.append(
            DiligenceQuestion(
                priority=2,
                question="Confirm valuation, round size, discount, and minimum check.",
                reason="No verified deal-term claims were available.",
            )
        )
    if pmf_level == PMFLevel.UNKNOWN:
        questions.append(
            DiligenceQuestion(
                priority=3,
                question="Find concrete customer, revenue, retention, or usage evidence.",
                reason="The extracted evidence did not show product-market fit signals.",
            )
        )
    if fundability_risk in {FundabilityRisk.HIGH, FundabilityRisk.UNKNOWN}:
        questions.append(
            DiligenceQuestion(
                priority=4,
                question="Check whether the company can raise the next round.",
                reason="The evidence has limited investor or growth signals.",
            )
        )
    if not questions:
        questions.append(
            DiligenceQuestion(
                priority=1,
                question="Verify that the strongest claims remain true in current materials.",
                reason="The deterministic checks did not find a blocking gap.",
            )
        )
    return sorted(questions, key=lambda question: question.priority)


def _check_size_for_score(total_score: int, *, config: AppConfig) -> int:
    if total_score >= 85:
        target = 10_000
    elif total_score >= 78:
        target = 7_500
    else:
        target = 5_000
    allowed_tiers = [
        tier
        for tier in CHECK_SIZE_TIERS
        if tier == 0 or (config.min_check <= tier <= config.max_check)
    ]
    allowed_tiers = [tier for tier in allowed_tiers if tier <= config.capital_budget]
    if not allowed_tiers:
        return 0
    nonzero_tiers = [tier for tier in allowed_tiers if tier > 0]
    if not nonzero_tiers:
        return 0
    tiers_at_or_below_target = [tier for tier in nonzero_tiers if tier <= target]
    if tiers_at_or_below_target:
        return max(tiers_at_or_below_target)
    return min(nonzero_tiers)


def _text_index(evidence: list[EvidenceRecord]) -> str:
    return "\n".join(record.text.lower() for record in evidence)


def _claim_evidence_ids(claims: list[ClaimRecord]) -> list[str]:
    evidence_ids: list[str] = []
    for claim in claims:
        for citation in claim.citations:
            if citation.evidence_id not in evidence_ids:
                evidence_ids.append(citation.evidence_id)
    return evidence_ids[:5]


def _evidence_matching_keywords(
    evidence: list[EvidenceRecord],
    keywords: tuple[str, ...],
) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if any(keyword in record.text.lower() for keyword in keywords)
    ]
