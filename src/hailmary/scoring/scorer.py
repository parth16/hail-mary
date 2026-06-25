from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from hailmary.config import CHECK_SIZE_TIERS, AppConfig
from hailmary.schemas.evidence import (
    ClaimConflict,
    ClaimRecord,
    EvidenceCitation,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)
from hailmary.schemas.scoring import (
    CompanyStage,
    ConfidenceLevel,
    DiligenceQuestion,
    FundabilityRisk,
    KillGate,
    NetReturnEstimate,
    PMFLevel,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
    ScoreSupportStatus,
    ValuationRisk,
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
INVEST_MINIMUM_SCORE = 75
HARD_MAX_CHECK = max(CHECK_SIZE_TIERS)
SCORING_MONEY_PATTERN = (
    r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s?(?:thousand|million|billion|k|m|b))?"
)
STAGE_KEYWORDS: tuple[tuple[CompanyStage, tuple[str, ...]], ...] = (
    (CompanyStage.HARD_TECH_DEFENSE, ("hard tech", "hard-tech", "defense", "aerospace")),
    (CompanyStage.SERIES_B_PLUS, ("series b", "series c", "growth stage")),
    (CompanyStage.SERIES_A, ("series a",)),
    (CompanyStage.PRE_SEED, ("pre-seed", "pre seed", "preseed")),
    (CompanyStage.SEED, ("seed",)),
)
STAGE_NEGATED_SIGNAL = (
    r"(?:pre[-\s]?seed|seed|series\s+a|series\s+b|series\s+c|growth\s+stage|"
    r"hard[-\s]?tech|defense|aerospace)"
)
STAGE_NEGATED_PATTERNS = (
    re.compile(
        rf"\bnon[-\s]?{STAGE_NEGATED_SIGNAL}"
        rf"(?:\s*(?:,|or|and)\s*(?:non[-\s]?|a\s+|an\s+)?"
        rf"{STAGE_NEGATED_SIGNAL})*\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bnot\s+(?:yet\s+)?(?:ready\s+for\s+|at\s+|a\s+|an\s+)?"
        rf"{STAGE_NEGATED_SIGNAL}"
        rf"(?:\s*(?:,|or|and)\s*(?:a\s+|an\s+)?{STAGE_NEGATED_SIGNAL})*\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:before|prior\s+to|ahead\s+of)\s+(?:the\s+)?{STAGE_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:next|future|upcoming|planned|target)\s+(?:round\s+)?"
        rf"(?:is\s+|as\s+)?(?:a\s+|an\s+)?{STAGE_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:plans?|planned|planning|targets?|targeting|intends?|expects?)\s+"
        rf"(?:to\s+)?(?:raise|pursue|seek|close)\s+"
        rf"(?:a\s+|an\s+|the\s+)?{STAGE_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:will|would|could|may)\s+(?:raise|pursue|seek|close)\s+"
        rf"(?:a\s+|an\s+|the\s+)?{STAGE_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:plans?|planned|planning|targets?|targeting|intends?|expects?)\s+"
        rf"(?:for|toward|towards)\s+(?:a\s+|an\s+|the\s+)?{STAGE_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{STAGE_NEGATED_SIGNAL}\s+"
        rf"(?:investors?|funds?|backers?)\b(?:\s+\w+){{0,4}}",
        re.IGNORECASE,
    ),
)
RETURN_INPUT_PATTERNS = {
    "dilution": re.compile(
        r"\b(?:estimated\s+)?dilution\b\s*(?:is|of|at|:)?\s*(?P<value>\d+(?:\.\d+)?)\s?%",
        re.IGNORECASE,
    ),
    "fees": re.compile(
        r"\b(?:fund|platform|spv|investment|investor|wrapper)\s+"
        r"(?:management\s+)?"
        r"(?:fees?|expenses?)\b\s*(?:are|is|of|at|:)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s?%",
        re.IGNORECASE,
    ),
    "carry": re.compile(
        r"\bcarry\b\s*(?:is|of|at|:)?\s*(?P<value>\d+(?:\.\d+)?)\s?%",
        re.IGNORECASE,
    ),
    "combined_fees_and_carry": re.compile(
        r"\b(?:total\s+)?(?:fees?\s+(?:and|&|/)\s+carry|carry\s+(?:and|&|/)\s+fees?)\b"
        r"\s*(?:are|is|of|at|:)?\s*(?P<value>\d+(?:\.\d+)?)\s?%",
        re.IGNORECASE,
    ),
    "gross_exit_value": re.compile(
        rf"\b(?:gross\s+)?(?:exit value|exit)\b\s*(?:is|of|at|:)?\s*"
        rf"(?P<value>{SCORING_MONEY_PATTERN})",
        re.IGNORECASE,
    ),
}
TRACTION_NEGATED_SIGNAL = (
    r"(?:customers?|revenue|usage|retention|growth|pilots?|beta|lois?|waitlist)"
)
TRACTION_NEGATED_QUALIFIERS = (
    r"(?:(?:meaningful|material|measurable|real|recurring|commercial|signed|"
    r"active|current|clear|validated|paying|paid|confirmed|contracted|"
    r"production|live)\s+){0,3}"
)
BENIGN_NEGATED_TRACTION_NOUNS = r"(?:issues?|concerns?|problems?|churn|complaints?)"
NEGATED_TRACTION_PATTERNS = (
    re.compile(r"\bpre[-\s]?revenue\b", re.IGNORECASE),
    re.compile(
        rf"\bno\s+{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b)"
        rf"(?:(?:(?:\s*,\s*(?:(?:or|and)\s+)?)|\s+(?:or|and)\s+)"
        rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b))*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bwithout\s+{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot\s+(?:yet\s+)?(?:generating\s+)?revenue\b", re.IGNORECASE),
    re.compile(
        r"\bnot\s+(?:yet\s+)?(?:showing\s+)?"
        r"(?:usage|retention|growth|pilots?|beta|lois?|waitlist)\b",
        re.IGNORECASE,
    ),
)
BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS = r"(?:concerns?|issues?|problems?|complaints?)"
NEGATED_FUNDING_PATTERNS = (
    re.compile(
        r"\bno\s+(?:(?:committed|identified|confirmed|named|signed|"
        r"secured|current|active)\s+)?lead\s+investor\b"
        rf"(?!\s+{BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout\s+(?:a\s+)?(?:(?:committed|identified|confirmed|named|"
        r"signed|secured|current|active)\s+)?lead\s+investor\b"
        rf"(?!\s+{BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bno\s+institutional(?:\s+(?:investors?|follow[-\s]?on))?\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+institutional(?:\s+investors?)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:seed|follow[-\s]?on)(?:\s+\w+){0,3}\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:\w+\s+){0,3}follow[-\s]?on\b", re.IGNORECASE),
)
PRICING_TERM_LABELS = {
    "post-money valuation",
    "pre-money valuation",
    "valuation cap",
}


def score_evidence_store(
    store: EvidenceStore,
    *,
    config: AppConfig,
    capital_remaining: int | None = None,
) -> ScoredDeal:
    """Score one deal using only validated evidence-store records."""

    valid_conflicts = validated_conflicts(store)
    verified_claims = validated_verified_claims(store)
    available_capital = config.capital_budget if capital_remaining is None else capital_remaining
    platform_minimum_check = _platform_minimum_check(verified_claims)
    pmf_level = _pmf_level(store.evidence)
    company_stage = _company_stage(store.evidence)
    fundability_risk = _fundability_risk(store, verified_claims)
    net_return = _net_return_estimate(store, verified_claims)
    valuation_risk = _valuation_risk(
        net_return.entry_valuation,
        company_stage=company_stage,
        pmf_level=pmf_level,
    )
    confidence = _confidence_level(store, verified_claims, valid_conflicts)
    kill_gates = _kill_gates(
        store,
        verified_claims,
        valid_conflicts,
        config=config,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=available_capital,
        valuation_risk=valuation_risk,
        valuation_evidence_ids=_valuation_gate_evidence_ids(
            store,
            verified_claims,
            company_stage,
            pmf_level,
        ),
    )
    score_factors = _score_factors(
        store,
        verified_claims,
        valid_conflicts,
        company_stage,
        pmf_level,
        fundability_risk,
        valuation_risk,
        net_return,
    )
    total_score = sum(factor.score for factor in score_factors)
    has_kill_gate = any(gate.triggered for gate in kill_gates)
    recommendation = (
        Recommendation.PASS
        if has_kill_gate or total_score < INVEST_MINIMUM_SCORE
        else Recommendation.INVEST
    )
    check_size = (
        0
        if recommendation == Recommendation.PASS
        else _check_size_for_score(
            total_score,
            config=config,
            confidence=confidence,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=available_capital,
        )
    )
    if recommendation == Recommendation.INVEST and check_size == 0:
        recommendation = Recommendation.PASS
        kill_gates.append(
            KillGate(
                name="No available check size",
                triggered=True,
                reason="No configured check size fits the remaining capital.",
            )
        )

    return ScoredDeal(
        deal_id=store.deal_id,
        company_name=store.company_name,
        recommendation=recommendation,
        check_size=check_size,
        total_score=total_score,
        confidence=confidence,
        one_line_reason=_one_line_reason(
            recommendation,
            total_score=total_score,
            confidence=confidence,
            kill_gates=kill_gates,
        ),
        pmf_level=pmf_level,
        fundability_risk=fundability_risk,
        company_stage=company_stage,
        valuation_risk=valuation_risk,
        net_return=net_return,
        kill_gates=kill_gates,
        score_factors=score_factors,
        diligence_questions=_diligence_questions(
            store,
            verified_claims,
            company_stage=company_stage,
            pmf_level=pmf_level,
            fundability_risk=fundability_risk,
            valuation_risk=valuation_risk,
            net_return=net_return,
            valid_conflicts=valid_conflicts,
        ),
        capital_remaining_before=available_capital,
        capital_remaining_after=max(0, available_capital - check_size),
    )


def validated_verified_claims(store: EvidenceStore) -> list[ClaimRecord]:
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    valid_conflict_claim_ids = {
        claim_id for conflict in validated_conflicts(store) for claim_id in conflict.claim_ids
    }
    return [
        claim
        for claim in store.claims
        if claim.id not in valid_conflict_claim_ids
        and claim.verification_status
        in {VerificationStatus.VERIFIED, VerificationStatus.CONFLICTED}
        and _claim_citations_are_valid(claim, evidence_by_id)
    ]


def validated_conflicts(store: EvidenceStore) -> list[ClaimConflict]:
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    claim_by_id = {claim.id: claim for claim in store.claims}
    valid_claim_ids = {
        claim.id
        for claim in store.claims
        if claim.verification_status
        in {VerificationStatus.VERIFIED, VerificationStatus.CONFLICTED}
        and _claim_citations_are_valid(claim, evidence_by_id)
    }
    valid_conflicts: list[ClaimConflict] = []
    for conflict in store.conflicts:
        valid_ids = [
            claim_id
            for claim_id in conflict.claim_ids
            if claim_id in valid_claim_ids and claim_id in claim_by_id
        ]
        valid_values = sorted(
            {claim_by_id[claim_id].normalized_value for claim_id in valid_ids}
        )
        if len(valid_ids) >= 2 and len(valid_values) >= 2:
            valid_conflicts.append(
                conflict.model_copy(
                    update={
                        "claim_ids": valid_ids,
                        "normalized_values": valid_values,
                    }
                )
            )
    return valid_conflicts


def _claim_citations_are_valid(
    claim: ClaimRecord,
    evidence_by_id: dict[str, EvidenceRecord],
) -> bool:
    if not claim.citations:
        return False
    return all(_citation_is_valid(citation, evidence_by_id) for citation in claim.citations)


def _citation_is_valid(
    citation: EvidenceCitation,
    evidence_by_id: dict[str, EvidenceRecord],
) -> bool:
    if citation.verification_status != VerificationStatus.VERIFIED:
        return False
    evidence = evidence_by_id.get(citation.evidence_id)
    if evidence is None:
        return False

    start = citation.source_span_start
    end = citation.source_span_end
    return (
        0 <= start < end <= len(evidence.text)
        and evidence.text[start:end] == citation.quote
    )


def _kill_gates(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
    *,
    config: AppConfig,
    platform_minimum_check: int | None,
    capital_remaining: int,
    valuation_risk: ValuationRisk,
    valuation_evidence_ids: list[str],
) -> list[KillGate]:
    minimum_above_maximum = (
        platform_minimum_check is not None
        and (platform_minimum_check > config.max_check or platform_minimum_check > HARD_MAX_CHECK)
    )
    has_scorable_deal = bool(store.evidence) and bool(verified_claims)
    no_check_available = (
        has_scorable_deal
        and not minimum_above_maximum
        and not _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
        )
    )
    has_verified_claims = bool(verified_claims)
    has_pricing_term = _has_readable_pricing_term(verified_claims)
    missing_key_terms = has_scorable_deal and not has_pricing_term
    valuation_too_high = has_scorable_deal and valuation_risk == ValuationRisk.HIGH
    conflict_evidence_ids = _conflict_evidence_ids(store, valid_conflicts)
    return [
        KillGate(
            name="No usable source-linked evidence",
            triggered=not store.evidence,
            reason=(
                "No usable extracted text was available."
                if not store.evidence
                else "At least one source-linked evidence record is available."
            ),
            support_status=(
                ScoreSupportStatus.NEEDS_DILIGENCE
                if not store.evidence
                else ScoreSupportStatus.INFERRED
            ),
        ),
        KillGate(
            name="Conflicting material deal terms",
            triggered=bool(valid_conflicts),
            reason=(
                "One or more extracted deal terms conflict and still have valid citations."
                if valid_conflicts
                else "No conflicting deal-term claims were detected."
            ),
            evidence_ids=conflict_evidence_ids,
            support_status=(
                ScoreSupportStatus.VERIFIED
                if valid_conflicts
                else ScoreSupportStatus.INFERRED
            ),
        ),
        KillGate(
            name="No verified deal terms",
            triggered=bool(store.evidence) and not verified_claims,
            reason=(
                "Evidence exists, but no deal-term claim was verified."
                if store.evidence and not verified_claims
                else (
                    "At least one deal-term claim has a verified citation."
                    if has_verified_claims
                    else "No evidence was available to verify deal terms."
                )
            ),
            support_status=(
                ScoreSupportStatus.VERIFIED
                if has_verified_claims
                else ScoreSupportStatus.NEEDS_DILIGENCE
            ),
        ),
        KillGate(
            name="Missing key investment terms",
            triggered=missing_key_terms,
            reason=(
                "No verified valuation or valuation-cap term was found."
                if not has_pricing_term
                else "A verified valuation or valuation-cap term is available."
            ),
            evidence_ids=_claim_evidence_ids(verified_claims),
            support_status=(
                ScoreSupportStatus.VERIFIED
                if has_pricing_term
                else ScoreSupportStatus.NEEDS_DILIGENCE
            ),
        ),
        KillGate(
            name="Valuation far ahead of evidence",
            triggered=valuation_too_high,
            reason=(
                "The verified valuation is far ahead of the current stage and traction evidence."
                if valuation_too_high
                else "No verified valuation appears far ahead of the stage and traction evidence."
            ),
            evidence_ids=valuation_evidence_ids,
            support_status=(
                ScoreSupportStatus.VERIFIED
                if valuation_too_high
                else ScoreSupportStatus.INFERRED
            ),
        ),
        KillGate(
            name="Platform minimum above maximum check",
            triggered=minimum_above_maximum,
            reason=(
                "The platform minimum check is above the configured maximum check size."
                if minimum_above_maximum
                else "No verified platform minimum exceeds the configured maximum check size."
            ),
            evidence_ids=[
                evidence_id
                for claim in verified_claims
                if claim.label == "minimum investment"
                for evidence_id in _claim_evidence_ids([claim])
            ],
            support_status=(
                ScoreSupportStatus.VERIFIED
                if minimum_above_maximum
                else ScoreSupportStatus.INFERRED
            ),
        ),
        KillGate(
            name="No available check size",
            triggered=no_check_available,
            reason=(
                "No configured check size fits the platform minimum and remaining capital."
                if no_check_available
                else "At least one configured check size fits the remaining capital."
            ),
            support_status=(
                ScoreSupportStatus.NEEDS_DILIGENCE
                if no_check_available
                else ScoreSupportStatus.INFERRED
            ),
        ),
    ]


def _score_factors(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
    valuation_risk: ValuationRisk,
    net_return: NetReturnEstimate,
) -> list[ScoreFactor]:
    return [
        _evidence_authority_factor(store),
        _deal_terms_factor(verified_claims, valid_conflicts),
        _stage_pmf_factor(store, company_stage, pmf_level),
        _fundability_factor(store, verified_claims, pmf_level, fundability_risk),
        _valuation_net_return_factor(valuation_risk, net_return),
        _missing_data_factor(
            store,
            verified_claims,
            valid_conflicts,
            pmf_level=pmf_level,
            fundability_risk=fundability_risk,
            net_return=net_return,
        ),
    ]


def _evidence_authority_factor(store: EvidenceStore) -> ScoreFactor:
    if not store.evidence:
        return ScoreFactor(
            name="Evidence authority and freshness",
            score=0,
            max_score=15,
            explanation="Hail Mary found no usable source-linked evidence.",
            support_status=ScoreSupportStatus.NEEDS_DILIGENCE,
            missing_inputs=["source-linked evidence"],
        )
    stale_count = sum(
        1 for evidence in store.evidence if evidence.source_freshness == SourceFreshness.STALE
    )
    unknown_count = sum(
        1 for evidence in store.evidence if evidence.source_freshness == SourceFreshness.UNKNOWN
    )
    score = min(12, len(store.evidence) * 4) + 3
    score -= min(6, stale_count * 3)
    score -= min(3, unknown_count)
    score = max(0, min(15, score))
    explanation = (
        f"Hail Mary found {len(store.evidence)} source-linked evidence records; "
        f"{stale_count} are stale and {unknown_count} have unknown freshness."
    )
    return ScoreFactor(
        name="Evidence authority and freshness",
        score=score,
        max_score=15,
        explanation=explanation,
        evidence_ids=[evidence.id for evidence in store.evidence[:5]],
        support_status=(
            ScoreSupportStatus.NEEDS_DILIGENCE
            if unknown_count
            else ScoreSupportStatus.VERIFIED
        ),
        missing_inputs=(
            ["current source dates"] if unknown_count else []
        ),
    )


def _deal_terms_factor(
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
) -> ScoreFactor:
    unique_labels = sorted({claim.label for claim in verified_claims})
    score = min(12, len(unique_labels) * 4)
    has_readable_pricing = _has_readable_pricing_term(verified_claims)
    if has_readable_pricing:
        score += 3
    if any(claim.label == "round size" for claim in verified_claims):
        score += 2
    if any(claim.label == "discount" for claim in verified_claims):
        score += 1
    if any(claim.label == "minimum investment" for claim in verified_claims):
        score += 1
    score = min(20, score)
    if valid_conflicts:
        score = min(score, 6)
    missing_inputs = []
    if not verified_claims:
        missing_inputs.append("verified deal terms")
    if verified_claims and not has_readable_pricing:
        missing_inputs.append("verified valuation or valuation cap")
    return ScoreFactor(
        name="Deal terms and platform access",
        score=score,
        max_score=20,
        explanation=(
            f"Verified deal-term labels: {', '.join(unique_labels)}."
            if unique_labels
            else "No verified deal-term labels were found."
        ),
        evidence_ids=_claim_evidence_ids(verified_claims),
        support_status=(
            ScoreSupportStatus.VERIFIED
            if verified_claims and not valid_conflicts and not missing_inputs
            else ScoreSupportStatus.NEEDS_DILIGENCE
        ),
        missing_inputs=missing_inputs,
    )


def _stage_pmf_factor(
    store: EvidenceStore,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
) -> ScoreFactor:
    score_by_stage_and_pmf = {
        CompanyStage.UNKNOWN: {
            PMFLevel.UNKNOWN: 3,
            PMFLevel.EARLY: 8,
            PMFLevel.DEVELOPING: 13,
        },
        CompanyStage.PRE_SEED: {
            PMFLevel.UNKNOWN: 8,
            PMFLevel.EARLY: 14,
            PMFLevel.DEVELOPING: 17,
        },
        CompanyStage.SEED: {
            PMFLevel.UNKNOWN: 5,
            PMFLevel.EARLY: 12,
            PMFLevel.DEVELOPING: 17,
        },
        CompanyStage.SERIES_A: {
            PMFLevel.UNKNOWN: 3,
            PMFLevel.EARLY: 8,
            PMFLevel.DEVELOPING: 18,
        },
        CompanyStage.SERIES_B_PLUS: {
            PMFLevel.UNKNOWN: 2,
            PMFLevel.EARLY: 6,
            PMFLevel.DEVELOPING: 16,
        },
        CompanyStage.HARD_TECH_DEFENSE: {
            PMFLevel.UNKNOWN: 6,
            PMFLevel.EARLY: 13,
            PMFLevel.DEVELOPING: 18,
        },
    }
    pmf_evidence = _pmf_evidence(store.evidence, pmf_level)
    stage_evidence = _stage_evidence(store.evidence, company_stage)
    evidence_ids = _stage_pmf_evidence_ids(stage_evidence, pmf_evidence)
    missing_inputs = []
    if company_stage == CompanyStage.UNKNOWN:
        missing_inputs.append("explicit company stage")
    if pmf_level == PMFLevel.UNKNOWN:
        missing_inputs.append("customer, revenue, retention, usage, pilot, or design-partner proof")
    return ScoreFactor(
        name="Stage and product-market fit",
        score=score_by_stage_and_pmf[company_stage][pmf_level],
        max_score=20,
        explanation=(
            f"Company stage is {company_stage}; product-market fit level is {pmf_level}."
        ),
        evidence_ids=evidence_ids,
        support_status=(
            ScoreSupportStatus.VERIFIED
            if evidence_ids and not missing_inputs
            else ScoreSupportStatus.NEEDS_DILIGENCE
        ),
        missing_inputs=missing_inputs,
    )


def _fundability_factor(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
) -> ScoreFactor:
    score_by_risk = {
        FundabilityRisk.UNKNOWN: 3,
        FundabilityRisk.HIGH: 5,
        FundabilityRisk.MEDIUM: 10,
        FundabilityRisk.LOW: 14,
    }
    matched_evidence = _positive_funding_evidence(store.evidence)
    missing_inputs: list[str] = []
    if not matched_evidence:
        missing_inputs.append(
            "lead investor, institutional investor, or follow-on financing evidence"
        )
    if not verified_claims:
        missing_inputs.append("verified deal terms")
    if pmf_level != PMFLevel.DEVELOPING:
        missing_inputs.append("customer, revenue, retention, or usage evidence")
    return ScoreFactor(
        name="Fundability and next-round risk",
        score=score_by_risk[fundability_risk],
        max_score=15,
        explanation=f"Next-round fundability risk is {fundability_risk}.",
        evidence_ids=[evidence.id for evidence in matched_evidence[:5]],
        support_status=(
            ScoreSupportStatus.VERIFIED
            if matched_evidence and not missing_inputs
            else ScoreSupportStatus.NEEDS_DILIGENCE
        ),
        missing_inputs=missing_inputs,
    )


def _valuation_net_return_factor(
    valuation_risk: ValuationRisk,
    net_return: NetReturnEstimate,
) -> ScoreFactor:
    score_by_risk = {
        ValuationRisk.UNKNOWN: 0,
        ValuationRisk.HIGH: 4,
        ValuationRisk.MEDIUM: 9,
        ValuationRisk.LOW: 13,
    }
    score = score_by_risk[valuation_risk]
    if net_return.net_return_multiple is not None:
        if net_return.net_return_multiple >= 10:
            score += 7
        elif net_return.net_return_multiple >= 5:
            score += 4
        elif net_return.net_return_multiple >= 2:
            score += 2
    score = min(20, score)
    return ScoreFactor(
        name="Valuation and net return",
        score=score,
        max_score=20,
        explanation=f"Valuation risk is {valuation_risk}. {net_return.explanation}",
        evidence_ids=net_return.evidence_ids,
        support_status=net_return.support_status,
        missing_inputs=net_return.missing_inputs,
    )


def _missing_data_factor(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
    *,
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
    net_return: NetReturnEstimate,
) -> ScoreFactor:
    stale_count = sum(
        1 for evidence in store.evidence if evidence.source_freshness == SourceFreshness.STALE
    )
    unknown_count = sum(
        1 for evidence in store.evidence if evidence.source_freshness == SourceFreshness.UNKNOWN
    )
    missing_inputs: list[str] = []
    score = 10
    if not store.evidence:
        score -= 10
        missing_inputs.append("source-linked evidence")
    if store.evidence and not verified_claims:
        score -= 5
        missing_inputs.append("verified deal terms")
    if valid_conflicts:
        score -= 7
        missing_inputs.append("resolved conflicting deal terms")
    if verified_claims and not _has_readable_pricing_term(verified_claims):
        score -= 3
        missing_inputs.append("verified valuation or valuation cap")
    if pmf_level == PMFLevel.UNKNOWN:
        score -= 2
        missing_inputs.append("product-market fit evidence")
    if fundability_risk in {FundabilityRisk.HIGH, FundabilityRisk.UNKNOWN}:
        score -= 2
        missing_inputs.append("next-round financing evidence")
    if net_return.missing_inputs:
        score -= min(2, len(net_return.missing_inputs))
        missing_inputs.extend(net_return.missing_inputs)
    score -= min(4, stale_count * 2)
    score -= min(2, unknown_count)
    score = max(0, score)
    explanation = (
        "Missing-data review found no blocking gap."
        if not missing_inputs
        else f"Missing or uncertain inputs: {', '.join(dict.fromkeys(missing_inputs))}."
    )
    return ScoreFactor(
        name="Missing data, conflicts, and staleness",
        score=score,
        max_score=10,
        explanation=explanation,
        evidence_ids=[
            evidence.id
            for evidence in store.evidence
            if evidence.source_freshness in {SourceFreshness.STALE, SourceFreshness.UNKNOWN}
        ][:5],
        support_status=(
            ScoreSupportStatus.INFERRED
            if not missing_inputs and not valid_conflicts
            else ScoreSupportStatus.NEEDS_DILIGENCE
        ),
        missing_inputs=list(dict.fromkeys(missing_inputs)),
    )


def _company_stage(evidence: list[EvidenceRecord]) -> CompanyStage:
    for stage, keywords in STAGE_KEYWORDS:
        if any(
            _contains_positive_keyword(
                record.text,
                keyword,
                negated_patterns=STAGE_NEGATED_PATTERNS,
            )
            for record in evidence
            for keyword in keywords
        ):
            return stage
    return CompanyStage.UNKNOWN


def _stage_evidence(
    evidence: list[EvidenceRecord],
    company_stage: CompanyStage,
) -> list[EvidenceRecord]:
    if company_stage == CompanyStage.UNKNOWN:
        return []
    keywords = next(
        keywords
        for stage, keywords in STAGE_KEYWORDS
        if stage == company_stage
    )
    return [
        record
        for record in evidence
        if any(
            _contains_positive_keyword(
                record.text,
                keyword,
                negated_patterns=STAGE_NEGATED_PATTERNS,
            )
            for keyword in keywords
        )
    ]


def _valuation_risk(
    entry_valuation: int | None,
    *,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
) -> ValuationRisk:
    if entry_valuation is None:
        return ValuationRisk.UNKNOWN
    if _valuation_is_far_ahead(
        entry_valuation,
        company_stage=company_stage,
        pmf_level=pmf_level,
    ):
        return ValuationRisk.HIGH
    medium_thresholds = {
        CompanyStage.PRE_SEED: 15_000_000,
        CompanyStage.SEED: 35_000_000,
        CompanyStage.SERIES_A: 100_000_000,
        CompanyStage.SERIES_B_PLUS: 500_000_000,
        CompanyStage.HARD_TECH_DEFENSE: 50_000_000,
        CompanyStage.UNKNOWN: 25_000_000,
    }
    if (
        entry_valuation >= medium_thresholds[company_stage]
        or pmf_level == PMFLevel.UNKNOWN
    ):
        return ValuationRisk.MEDIUM
    return ValuationRisk.LOW


def _valuation_is_far_ahead(
    entry_valuation: int,
    *,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
) -> bool:
    if company_stage == CompanyStage.PRE_SEED:
        return entry_valuation >= 45_000_000 or (
            entry_valuation >= 25_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SEED:
        return entry_valuation >= 120_000_000 or (
            entry_valuation >= 75_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SERIES_A:
        return entry_valuation >= 300_000_000 or (
            entry_valuation >= 175_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SERIES_B_PLUS:
        return entry_valuation >= 1_000_000_000 or (
            entry_valuation >= 750_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.HARD_TECH_DEFENSE:
        return entry_valuation >= 500_000_000 or (
            entry_valuation >= 150_000_000 and pmf_level == PMFLevel.UNKNOWN
        )
    if company_stage == CompanyStage.UNKNOWN:
        return entry_valuation >= 50_000_000
    return entry_valuation >= 50_000_000 and pmf_level != PMFLevel.DEVELOPING


def _net_return_estimate(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
) -> NetReturnEstimate:
    valuation_claim = _valuation_claim(verified_claims)
    if valuation_claim is None:
        return NetReturnEstimate(
            missing_inputs=["verified entry valuation or valuation cap"],
            explanation=(
                "Net return math needs a verified valuation or valuation cap before "
                "Hail Mary can model returns."
            ),
            support_status=ScoreSupportStatus.NEEDS_DILIGENCE,
        )

    valuation_value = _claim_money_value(valuation_claim)
    evidence_ids = _claim_evidence_ids([valuation_claim])
    if valuation_value is None:
        return NetReturnEstimate(
            missing_inputs=["machine-readable entry valuation"],
            explanation=(
                "A valuation claim is cited, but Hail Mary could not read it as a "
                "whole-dollar value."
            ),
            evidence_ids=evidence_ids,
            support_status=ScoreSupportStatus.NEEDS_DILIGENCE,
        )
    round_size_claim = _round_size_claim(verified_claims)
    if valuation_claim.label == "pre-money valuation":
        if round_size_claim is None or _claim_money_value(round_size_claim) is None:
            return NetReturnEstimate(
                missing_inputs=["verified round size for pre-money valuation"],
                explanation=(
                    "A pre-money valuation needs a verified round size before Hail Mary "
                    "can model post-money entry value."
                ),
                evidence_ids=evidence_ids,
                support_status=ScoreSupportStatus.NEEDS_DILIGENCE,
            )
        evidence_ids = list(
            dict.fromkeys([*evidence_ids, *_claim_evidence_ids([round_size_claim])])
        )
    entry_valuation = _entry_valuation(verified_claims, valuation_claim)
    if entry_valuation is None:
        return NetReturnEstimate(
            missing_inputs=["verified entry valuation or valuation cap"],
            explanation=(
                "Net return math needs a verified valuation or valuation cap before "
                "Hail Mary can model returns."
            ),
            evidence_ids=evidence_ids,
            support_status=ScoreSupportStatus.NEEDS_DILIGENCE,
        )

    return_inputs = _return_inputs(store.evidence)
    evidence_ids = list(dict.fromkeys([*evidence_ids, *return_inputs.evidence_ids]))
    missing_inputs: list[str] = []
    if return_inputs.dilution_percent is None:
        missing_inputs.append("dilution")
    if return_inputs.fees_and_carry_percent is None:
        missing_inputs.append("fees or carry")
    if return_inputs.gross_exit_value is None:
        missing_inputs.append("gross exit scenario")

    net_multiple: float | None = None
    if (
        return_inputs.dilution_percent is not None
        and return_inputs.fees_and_carry_percent is not None
        and return_inputs.gross_exit_value is not None
        and entry_valuation > 0
    ):
        ownership_after_dilution = max(0.0, 1 - (return_inputs.dilution_percent / 100))
        proceeds_after_fees = max(0.0, 1 - (return_inputs.fees_and_carry_percent / 100))
        net_multiple = round(
            (return_inputs.gross_exit_value / entry_valuation)
            * ownership_after_dilution
            * proceeds_after_fees,
            2,
        )

    if net_multiple is None:
        explanation = (
            f"Verified entry valuation is {_format_dollars(entry_valuation)}. "
            f"Missing {', '.join(missing_inputs)}, so Hail Mary did not invent a net return."
        )
        support_status = ScoreSupportStatus.NEEDS_DILIGENCE
    else:
        explanation = (
            f"Verified entry valuation is {_format_dollars(entry_valuation)}. "
            f"Using cited dilution, fees or carry, and exit value, estimated net return "
            f"is {net_multiple:g}x."
        )
        support_status = ScoreSupportStatus.VERIFIED

    return NetReturnEstimate(
        entry_valuation=entry_valuation,
        estimated_dilution_percent=return_inputs.dilution_percent,
        estimated_fees_and_carry_percent=return_inputs.fees_and_carry_percent,
        gross_exit_value=return_inputs.gross_exit_value,
        net_return_multiple=net_multiple,
        missing_inputs=missing_inputs,
        explanation=explanation,
        evidence_ids=evidence_ids,
        support_status=support_status,
    )


class _ReturnInputs:
    def __init__(
        self,
        *,
        dilution_percent: float | None,
        fees_and_carry_percent: float | None,
        gross_exit_value: int | None,
        evidence_ids: list[str],
    ) -> None:
        self.dilution_percent = dilution_percent
        self.fees_and_carry_percent = fees_and_carry_percent
        self.gross_exit_value = gross_exit_value
        self.evidence_ids = evidence_ids


def _return_inputs(evidence: list[EvidenceRecord]) -> _ReturnInputs:
    dilution_percent: float | None = None
    fees_percent: float | None = None
    carry_percent: float | None = None
    combined_fees_and_carry_percent: float | None = None
    gross_exit_value: int | None = None
    evidence_ids: list[str] = []
    for record in evidence:
        dilution_match = RETURN_INPUT_PATTERNS["dilution"].search(record.text)
        if dilution_percent is None and dilution_match:
            dilution_percent = _float_text(dilution_match.group("value"))
            evidence_ids.append(record.id)
        fees_match = RETURN_INPUT_PATTERNS["fees"].search(record.text)
        if fees_percent is None and fees_match:
            fees_percent = _float_text(fees_match.group("value"))
            evidence_ids.append(record.id)
        carry_match = RETURN_INPUT_PATTERNS["carry"].search(record.text)
        if carry_percent is None and carry_match:
            carry_percent = _float_text(carry_match.group("value"))
            evidence_ids.append(record.id)
        combined_match = RETURN_INPUT_PATTERNS["combined_fees_and_carry"].search(record.text)
        if combined_fees_and_carry_percent is None and combined_match:
            combined_fees_and_carry_percent = _float_text(combined_match.group("value"))
            evidence_ids.append(record.id)
        exit_match = RETURN_INPUT_PATTERNS["gross_exit_value"].search(record.text)
        if gross_exit_value is None and exit_match:
            gross_exit_value = _money_text_to_dollars(exit_match.group("value"))
            evidence_ids.append(record.id)
    fees_and_carry: float | None
    if combined_fees_and_carry_percent is not None:
        fees_and_carry = combined_fees_and_carry_percent
    elif fees_percent is not None and carry_percent is not None:
        fees_and_carry = fees_percent + carry_percent
    else:
        fees_and_carry = None
    return _ReturnInputs(
        dilution_percent=dilution_percent,
        fees_and_carry_percent=fees_and_carry,
        gross_exit_value=gross_exit_value,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )


def _float_text(raw_value: str) -> float | None:
    try:
        return float(raw_value.strip())
    except ValueError:
        return None


def _valuation_claim(verified_claims: list[ClaimRecord]) -> ClaimRecord | None:
    for label in ("post-money valuation", "valuation cap", "pre-money valuation"):
        for claim in verified_claims:
            if claim.label == label:
                return claim
    return None


def _round_size_claim(verified_claims: list[ClaimRecord]) -> ClaimRecord | None:
    for claim in verified_claims:
        if claim.label == "round size":
            return claim
    return None


def _entry_valuation(
    verified_claims: list[ClaimRecord],
    valuation_claim: ClaimRecord,
) -> int | None:
    valuation = _claim_money_value(valuation_claim)
    if valuation is None:
        return None
    if valuation_claim.label != "pre-money valuation":
        return valuation
    round_size_claim = _round_size_claim(verified_claims)
    if round_size_claim is None:
        return None
    round_size = _claim_money_value(round_size_claim)
    if round_size is None:
        return None
    return valuation + round_size


def _pmf_level(evidence: list[EvidenceRecord]) -> PMFLevel:
    if _positive_traction_evidence(evidence):
        return PMFLevel.DEVELOPING
    if _positive_early_pmf_evidence(evidence):
        return PMFLevel.EARLY
    return PMFLevel.UNKNOWN


def _pmf_evidence(
    evidence: list[EvidenceRecord],
    pmf_level: PMFLevel,
) -> list[EvidenceRecord]:
    if pmf_level == PMFLevel.DEVELOPING:
        return _positive_traction_evidence(evidence)
    if pmf_level == PMFLevel.EARLY:
        return _positive_early_pmf_evidence(evidence)
    return []


def _fundability_risk(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
) -> FundabilityRisk:
    if not store.evidence:
        return FundabilityRisk.UNKNOWN
    has_terms = bool(verified_claims)
    has_traction = bool(_positive_traction_evidence(store.evidence))
    has_funding_signal = bool(_positive_funding_evidence(store.evidence))
    if has_terms and has_traction and has_funding_signal:
        return FundabilityRisk.LOW
    if has_terms and has_traction:
        return FundabilityRisk.MEDIUM
    return FundabilityRisk.HIGH


def _diligence_questions(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    *,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
    valuation_risk: ValuationRisk,
    net_return: NetReturnEstimate,
    valid_conflicts: list[ClaimConflict],
) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    if valid_conflicts:
        questions.append(
            DiligenceQuestion(
                priority=1,
                question="Resolve the conflicting deal terms in the original documents.",
                reason="The evidence store has conflicting extracted values.",
                evidence_ids=_conflict_evidence_ids(store, valid_conflicts),
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
    elif not _has_readable_pricing_term(verified_claims):
        questions.append(
            DiligenceQuestion(
                priority=2,
                question="Confirm the valuation, valuation cap, or priced-round valuation.",
                reason="The evidence did not include a verified pricing term.",
                evidence_ids=_claim_evidence_ids(verified_claims),
            )
        )
    if company_stage == CompanyStage.UNKNOWN:
        questions.append(
            DiligenceQuestion(
                priority=3,
                question="Confirm the company stage before applying the underwriting bar.",
                reason="The extracted evidence did not include an explicit stage signal.",
            )
        )
    if pmf_level == PMFLevel.UNKNOWN:
        questions.append(
            DiligenceQuestion(
                priority=4,
                question="Find concrete customer, revenue, retention, or usage evidence.",
                reason="The extracted evidence did not show product-market fit signals.",
            )
        )
    if fundability_risk in {FundabilityRisk.HIGH, FundabilityRisk.UNKNOWN}:
        questions.append(
            DiligenceQuestion(
                priority=5,
                question="Check whether the company can raise the next round.",
                reason="The evidence has limited investor or growth signals.",
            )
        )
    if valuation_risk == ValuationRisk.HIGH:
        valuation_evidence_ids = _valuation_gate_evidence_ids(
            store,
            verified_claims,
            company_stage,
            pmf_level,
        )
        questions.append(
            DiligenceQuestion(
                priority=1,
                question="Confirm why the valuation is justified by current evidence.",
                reason="The verified valuation appears far ahead of stage and traction.",
                evidence_ids=valuation_evidence_ids,
            )
        )
    if net_return.missing_inputs:
        questions.append(
            DiligenceQuestion(
                priority=6,
                question="Collect the missing return-math inputs before sizing the check.",
                reason=(
                    "Hail Mary needs verified "
                    f"{', '.join(net_return.missing_inputs)} to model net return."
                ),
                evidence_ids=net_return.evidence_ids,
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


def _check_size_for_score(
    total_score: int,
    *,
    config: AppConfig,
    confidence: ConfidenceLevel,
    platform_minimum_check: int | None,
    capital_remaining: int,
) -> int:
    target = _target_check_size(total_score, confidence=confidence)
    if platform_minimum_check is not None:
        target = max(target, platform_minimum_check)
    allowed_tiers = _available_nonzero_tiers(
        config,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=capital_remaining,
    )
    if not allowed_tiers:
        return 0
    tiers_at_or_above_target = [tier for tier in allowed_tiers if tier >= target]
    if tiers_at_or_above_target:
        return min(tiers_at_or_above_target)
    return max(allowed_tiers)


def _target_check_size(total_score: int, *, confidence: ConfidenceLevel) -> int:
    if total_score >= 90 and confidence == ConfidenceLevel.HIGH:
        target = 10_000
    elif total_score >= 86 and confidence != ConfidenceLevel.LOW:
        target = 7_500
    elif total_score >= 82 and confidence != ConfidenceLevel.LOW:
        target = 5_000
    elif total_score >= INVEST_MINIMUM_SCORE:
        target = 1_000 if confidence == ConfidenceLevel.LOW else 2_500
    else:
        target = 0
    return target


def _available_nonzero_tiers(
    config: AppConfig,
    *,
    platform_minimum_check: int | None,
    capital_remaining: int,
) -> list[int]:
    minimum_check = config.min_check
    if platform_minimum_check is not None:
        minimum_check = max(minimum_check, platform_minimum_check)
    maximum_check = min(config.max_check, capital_remaining, HARD_MAX_CHECK)
    return [
        tier
        for tier in CHECK_SIZE_TIERS
        if tier > 0 and minimum_check <= tier <= maximum_check
    ]


def _platform_minimum_check(verified_claims: list[ClaimRecord]) -> int | None:
    minimum_checks = [
        parsed_value
        for claim in verified_claims
        if claim.label == "minimum investment"
        for parsed_value in [_claim_money_value(claim)]
        if parsed_value is not None
    ]
    if not minimum_checks:
        return None
    return max(minimum_checks)


def _has_pricing_term(verified_claims: list[ClaimRecord]) -> bool:
    return any(claim.label in PRICING_TERM_LABELS for claim in verified_claims)


def _has_readable_pricing_term(verified_claims: list[ClaimRecord]) -> bool:
    valuation_claim = _valuation_claim(verified_claims)
    if valuation_claim is None:
        return False
    return _entry_valuation(verified_claims, valuation_claim) is not None


def _claim_money_value(claim: ClaimRecord) -> int | None:
    normalized_prefix = "usd_cents:"
    if claim.normalized_value.startswith(normalized_prefix):
        try:
            return int(claim.normalized_value.removeprefix(normalized_prefix)) // 100
        except ValueError:
            return None
    return _money_text_to_dollars(claim.value)


def _money_text_to_dollars(raw_value: str) -> int | None:
    normalized = raw_value.lower().replace("$", "").replace(",", "").strip()
    multiplier = Decimal("1")
    suffixes = {
        "thousand": Decimal("1000"),
        "million": Decimal("1000000"),
        "billion": Decimal("1000000000"),
        "k": Decimal("1000"),
        "m": Decimal("1000000"),
        "b": Decimal("1000000000"),
    }
    for suffix, suffix_multiplier in suffixes.items():
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            multiplier = suffix_multiplier
            break
    try:
        return int((Decimal(normalized) * multiplier).to_integral_value())
    except InvalidOperation:
        return None


def _confidence_level(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
) -> ConfidenceLevel:
    if valid_conflicts or not store.evidence or not verified_claims:
        return ConfidenceLevel.LOW
    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    cited_evidence = [
        evidence_by_id[citation.evidence_id]
        for claim in verified_claims
        for citation in claim.citations
        if citation.evidence_id in evidence_by_id
    ]
    source_document_ids = {evidence.document_id for evidence in cited_evidence}
    source_kinds = {evidence.source_kind for evidence in cited_evidence}
    has_independent_sources = len(source_document_ids) >= 2 or len(source_kinds) >= 2
    if len(store.evidence) >= 3 and len(verified_claims) >= 3 and has_independent_sources:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.MEDIUM


def _one_line_reason(
    recommendation: Recommendation,
    *,
    total_score: int,
    confidence: ConfidenceLevel,
    kill_gates: list[KillGate],
) -> str:
    triggered_gates = [gate for gate in kill_gates if gate.triggered]
    if triggered_gates:
        return f"Passed because {triggered_gates[0].reason}"
    if recommendation == Recommendation.PASS:
        return f"Passed because the score was {total_score}/100, below the investment bar."
    return (
        f"Recommended because the score was {total_score}/100, confidence was "
        f"{confidence}, and no kill gate triggered."
    )


def _positive_traction_evidence(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if _text_contains_positive_keyword(
            record.text,
            TRACTION_KEYWORDS,
            negated_patterns=NEGATED_TRACTION_PATTERNS,
        )
    ]


def _positive_funding_evidence(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if _text_contains_positive_keyword(
            record.text,
            FUNDABILITY_KEYWORDS,
            negated_patterns=NEGATED_FUNDING_PATTERNS,
        )
    ]


def _positive_early_pmf_evidence(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if _text_contains_positive_keyword(
            record.text,
            EARLY_PMF_KEYWORDS,
            negated_patterns=NEGATED_TRACTION_PATTERNS,
        )
    ]


def _text_contains_positive_keyword(
    text: str,
    keywords: tuple[str, ...],
    *,
    negated_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    return any(
        _contains_positive_keyword(
            text,
            keyword,
            negated_patterns=negated_patterns,
        )
        for keyword in keywords
    )


def _contains_positive_keyword(
    text: str,
    keyword: str,
    *,
    negated_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    pattern = _keyword_pattern(keyword)
    negated_spans = _negated_spans(text, negated_patterns)
    return any(
        not _span_overlaps(match.span(), negated_spans)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    )


def _negated_spans(
    text: str,
    negated_patterns: tuple[re.Pattern[str], ...],
) -> list[tuple[int, int]]:
    return [
        match.span()
        for pattern in negated_patterns
        for match in pattern.finditer(text)
    ]


def _span_overlaps(
    span: tuple[int, int],
    spans: list[tuple[int, int]],
) -> bool:
    start, end = span
    return any(start < negated_end and negated_start < end for negated_start, negated_end in spans)


def _contains_keyword(text: str, keyword: str) -> bool:
    pattern = _keyword_pattern(keyword)
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _keyword_pattern(keyword: str) -> str:
    escaped_words = [re.escape(part) for part in keyword.split()]
    escaped_phrase = r"\s+".join(escaped_words)
    return rf"(?<![A-Za-z0-9]){escaped_phrase}(?![A-Za-z0-9])"


def _claim_evidence_ids(claims: list[ClaimRecord]) -> list[str]:
    evidence_ids: list[str] = []
    for claim in claims:
        for citation in claim.citations:
            if citation.evidence_id not in evidence_ids:
                evidence_ids.append(citation.evidence_id)
    return evidence_ids[:5]


def _valuation_gate_evidence_ids(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
) -> list[str]:
    valuation_claim = _valuation_claim(verified_claims)
    evidence_ids = (
        _claim_evidence_ids([valuation_claim])
        if valuation_claim is not None
        else []
    )
    if valuation_claim is not None and valuation_claim.label == "pre-money valuation":
        round_size_claim = _round_size_claim(verified_claims)
        if round_size_claim is not None:
            evidence_ids.extend(_claim_evidence_ids([round_size_claim]))
    evidence_ids.extend(_dedupe_evidence_ids(_stage_evidence(store.evidence, company_stage)))
    evidence_ids.extend(_dedupe_evidence_ids(_pmf_evidence(store.evidence, pmf_level)))
    return list(dict.fromkeys(evidence_ids))[:5]


def _conflict_evidence_ids(
    store: EvidenceStore,
    valid_conflicts: list[ClaimConflict],
) -> list[str]:
    claim_by_id = {claim.id: claim for claim in store.claims}
    evidence_ids: list[str] = []
    for conflict in valid_conflicts:
        for claim_id in conflict.claim_ids:
            claim = claim_by_id.get(claim_id)
            if claim is None:
                continue
            for citation in claim.citations:
                if citation.evidence_id not in evidence_ids:
                    evidence_ids.append(citation.evidence_id)
    return evidence_ids[:5]


def _dedupe_evidence_ids(evidence: list[EvidenceRecord]) -> list[str]:
    evidence_ids: list[str] = []
    for record in evidence:
        if record.id not in evidence_ids:
            evidence_ids.append(record.id)
    return evidence_ids[:5]


def _stage_pmf_evidence_ids(
    stage_evidence: list[EvidenceRecord],
    pmf_evidence: list[EvidenceRecord],
) -> list[str]:
    ordered: list[EvidenceRecord] = []
    if stage_evidence:
        ordered.append(stage_evidence[0])
    if pmf_evidence:
        ordered.append(pmf_evidence[0])
    ordered.extend(stage_evidence[1:])
    ordered.extend(pmf_evidence[1:])
    return _dedupe_evidence_ids(ordered)


def _format_dollars(value: int) -> str:
    if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
        return f"${value // 1_000_000_000}B"
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"${value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"${value // 1_000}K"
    return f"${value:,}"
