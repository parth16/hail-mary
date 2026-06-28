from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from hailmary.config import CHECK_SIZE_TIERS, AppConfig
from hailmary.portfolio.scenario import portfolio_scenario
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
    CheckSizingDecision,
    CompanyStage,
    ConfidenceLevel,
    DiligenceQuestion,
    DiligenceQuestionCategory,
    DiligenceResearchContext,
    FundabilityRisk,
    KillGate,
    NetReturnEstimate,
    PMFLevel,
    PortfolioAllocationScenario,
    Recommendation,
    ScoredDeal,
    ScoreFactor,
    ScoreSupportStatus,
    ValuationRisk,
)
from hailmary.scoring.portfolio import (
    PortfolioExposureState,
    empty_portfolio_exposure_state,
    portfolio_exposure_cap,
    portfolio_exposure_checks,
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
    re.compile(
        rf"\b(?:no|without)\s+(?:current\s+|active\s+|committed\s+|"
        rf"verified\s+|confirmed\s+)?{STAGE_NEGATED_SIGNAL}\s+"
        rf"(?:funding|financing|round|raise)\b",
        re.IGNORECASE,
    ),
)
RETURN_INPUT_PATTERNS = {
    "ownership": re.compile(
        r"\b(?:(?:investor|investment|spv|fund|platform|check|our|target|"
        r"post[-\s]?money|pro[-\s]?forma)\s+ownership|"
        r"ownership\s+(?:target|for\s+(?:our\s+)?(?:check|investment|"
        r"investor|spv|fund|platform)))\b"
        r"\s*(?:is|of|at|:)?\s*(?P<value>\d+(?:\.\d+)?)\s?%",
        re.IGNORECASE,
    ),
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
NEGATED_GROSS_EXIT_PATTERNS = (
    re.compile(
        r"\b(?:no|without)\s+(?:gross\s+)?(?:exit value|exit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gross\s+)?(?:exit value|exit)\b(?:\s+\S+){0,4}\s+"
        r"(?:is|are|was|were|has been|have been)\s+not\s+"
        r"(?:provided|available|disclosed|included)\b",
        re.IGNORECASE,
    ),
)
TRACTION_NEGATED_SIGNAL = (
    r"(?:customers?|revenue|arr|usage|retention|growth|pilots?|beta|lois?|waitlist)"
)
TRACTION_NEGATED_QUALIFIERS = (
    r"(?:(?:any|actual|customer|meaningful|material|measurable|real|recurring|commercial|signed|"
    r"active|current|clear|validated|paying|paid|confirmed|contracted|annual|"
    r"production|live)\s+){0,3}"
)
BENIGN_NEGATED_TRACTION_NOUNS = r"(?:issues?|concerns?|problems?|churn|complaints?)"
TRACTION_NEGATED_TERM_PATTERN = re.compile(
    rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b",
    re.IGNORECASE,
)
WITHOUT_TRACTION_PATTERN = re.compile(r"\bwithout\s+", re.IGNORECASE)
WITHOUT_TRACTION_SEPARATOR_PATTERN = re.compile(
    r"(?:\s*,\s*(?:(?:or|and)\s+)?|\s+(?:or|and)\s+)",
    re.IGNORECASE,
)
TRACTION_POSITIVE_PREDICATE = (
    r"(?:(?:is|are|was|were|has|have|had|reports?|reported|shows?|showed)\s+"
    r"(?!(?:not|no|without|missing|absent|none)\b)"
    r"(?:\S+\s+){0,5}?"
    r"(?:\$|\d|strong|positive|validated|active|paying|paid|retained|"
    r"growing|growth|grew|reached|meaningful|commercial)"
    r"|(?:reaches|reached|grew|grows)\b)"
)
TRACTION_POSITIVE_CLAUSE_PATTERN = re.compile(
    rf"\A{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
    rf"(?:\s+(?:or|and)\s+{TRACTION_NEGATED_QUALIFIERS}"
    rf"{TRACTION_NEGATED_SIGNAL}\b)*"
    rf"(?:\s+(?:metrics?|evidence|proof|data|numbers?|claims?))*"
    rf"\s+{TRACTION_POSITIVE_PREDICATE}",
    re.IGNORECASE,
)
NEGATED_TRACTION_PATTERNS = (
    re.compile(r"\bpre[-\s]?revenue\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:planned|projected|expected|future|target|targeted)\s+"
        rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?:\s+{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b)*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bnot[-\s]?yet\s+{TRACTION_NEGATED_QUALIFIERS}"
        rf"{TRACTION_NEGATED_SIGNAL}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bno\s+{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b)"
        rf"(?:(?:(?:\s*,\s*(?:(?:or|and)\s+)?)|\s+(?:or|and)\s+)"
        rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b))*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\black(?:s|ing)?\s+{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b)"
        rf"(?:(?:(?:\s*,\s*(?:(?:or|and)\s+)?)|\s+(?:or|and)\s+)"
        rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b))*",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot\s+(?:yet\s+)?(?:generating\s+)?revenue\b", re.IGNORECASE),
    re.compile(
        r"\bnot\s+(?:yet\s+)?(?:showing\s+)?"
        r"(?:usage|retention|growth|pilots?|beta|lois?|waitlist)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:has|have|does|do|did)\s+not\s+have\s+"
        rf"{TRACTION_NEGATED_QUALIFIERS}{TRACTION_NEGATED_SIGNAL}\b"
        rf"(?!\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b)",
        re.IGNORECASE,
    ),
)
ABSENCE_TRACTION_PATTERNS = (
    NEGATED_TRACTION_PATTERNS[0],
    *NEGATED_TRACTION_PATTERNS[2:],
)
BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS = r"(?:concerns?|issues?|problems?|complaints?)"
BENIGN_INSTITUTIONAL_FOLLOWING_NOUNS = r"(?:concerns?|issues?|problems?|complaints?)"
BENIGN_FUNDING_CONCERN_NOUNS = r"(?:concerns?|issues?|problems?|complaints?)"
NEGATED_FUNDING_PATTERNS = (
    re.compile(
        r"\b(?:planned|projected|expected|future|upcoming|target|targeted)\s+"
        r"(?:lead\s+investor|institutional(?:\s+investors?)?|"
        r"follow[-\s]?on)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot[-\s]?yet\s+"
        r"(?:secured|identified|confirmed|named|signed)?\s*"
        r"(?:a\s+)?(?:lead\s+investor|institutional(?:\s+investors?)?|"
        r"follow[-\s]?on)\b",
        re.IGNORECASE,
    ),
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
    re.compile(
        r"\black(?:s|ing)?\s+(?:a\s+|any\s+)?(?:(?:committed|identified|"
        r"confirmed|named|signed|secured|current|active)\s+)?lead\s+investor\b"
        rf"(?!\s+{BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\black(?:s|ing)?\s+(?:any\s+)?institutional(?:\s+investors?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhas\s+not\s+(?:secured|identified|confirmed|named|signed)\s+"
        r"(?:a\s+)?lead\s+investor\b"
        rf"(?!\s+{BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does|do|did)\s+not\s+have\s+(?:a\s+)?lead\s+investor\b"
        rf"(?!\s+{BENIGN_LEAD_INVESTOR_FOLLOWING_NOUNS}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+institutional"
        rf"(?!(?:\s+investors?)?\s+{BENIGN_INSTITUTIONAL_FOLLOWING_NOUNS}\b)"
        r"(?:\s+(?:investors?|follow[-\s]?on))?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout\s+institutional"
        rf"(?!(?:\s+investors?)?\s+{BENIGN_INSTITUTIONAL_FOLLOWING_NOUNS}\b)"
        r"(?:\s+investors?)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+"
        rf"(?!(?:[\w-]+\s+){{0,4}}{BENIGN_FUNDING_CONCERN_NOUNS}\b)"
        r"(?:seed|follow[-\s]?on)(?:\s+\w+){0,3}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+"
        rf"(?!(?:[\w-]+\s+){{0,4}}{BENIGN_FUNDING_CONCERN_NOUNS}\b)"
        r"(?:\w+\s+){0,3}follow[-\s]?on\b",
        re.IGNORECASE,
    ),
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
    research_context: DiligenceResearchContext | None = None,
    exposure_state: PortfolioExposureState | None = None,
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
        net_return=net_return,
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
    check_sizing = _check_sizing_decision(
        store,
        total_score,
        config=config,
        confidence=confidence,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=available_capital,
        valuation_risk=valuation_risk,
        fundability_risk=fundability_risk,
        net_return=net_return,
        company_stage=company_stage,
        exposure_state=exposure_state,
    )
    check_size = (
        0
        if recommendation == Recommendation.PASS
        else check_sizing.selected_tier
    )
    if recommendation == Recommendation.PASS:
        check_sizing = check_sizing.model_copy(
            update={
                "selected_tier": 0,
                "reason_codes": [
                    *check_sizing.reason_codes,
                    "deterministic_pass",
                ],
            }
        )
    if recommendation == Recommendation.INVEST and check_size == 0:
        recommendation = Recommendation.PASS
        kill_gates.append(
            KillGate(
                name="No available check size",
                triggered=True,
                reason=_no_available_check_size_reason(
                    total_score,
                    config=config,
                    confidence=confidence,
                    platform_minimum_check=platform_minimum_check,
                    capital_remaining=available_capital,
                    valuation_risk=valuation_risk,
                    fundability_risk=fundability_risk,
                    net_return=net_return,
                    check_sizing=check_sizing,
                ),
            )
        )

    allocation_scenario = _allocation_scenario(
        config=config,
        capital_remaining=available_capital,
        check_size=check_size,
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
            kill_gates=kill_gates,
            research_context=research_context,
        ),
        capital_remaining_before=available_capital,
        capital_remaining_after=max(0, available_capital - check_size),
        allocation_scenario=allocation_scenario,
        check_sizing=check_sizing,
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
    net_return: NetReturnEstimate,
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
    missing_key_terms_reason = _missing_key_terms_reason(verified_claims)
    missing_key_terms_evidence_ids = _missing_key_terms_evidence_ids(verified_claims)
    missing_key_terms = has_scorable_deal and not has_pricing_term
    valuation_too_high = has_scorable_deal and valuation_risk == ValuationRisk.HIGH
    low_verified_return = (
        has_scorable_deal
        and net_return.support_status == ScoreSupportStatus.VERIFIED
        and net_return.net_return_multiple is not None
        and net_return.net_return_multiple < 1
    )
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
                missing_key_terms_reason
                if not has_pricing_term
                else "A verified valuation or valuation-cap term is available."
            ),
            evidence_ids=(
                _pricing_evidence_ids(verified_claims)
                if has_pricing_term
                else missing_key_terms_evidence_ids
            ),
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
            name="Verified return below capital back",
            triggered=low_verified_return,
            reason=(
                "Verified return math is below 1x capital back."
                if low_verified_return
                else "Verified return math is not below 1x capital back."
            ),
            evidence_ids=net_return.evidence_ids,
            support_status=(
                ScoreSupportStatus.VERIFIED
                if low_verified_return
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
        _valuation_net_return_factor(
            store,
            company_stage,
            pmf_level,
            valuation_risk,
            net_return,
        ),
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
        evidence_ids=_evidence_authority_evidence_ids(store.evidence),
        support_status=(
            ScoreSupportStatus.NEEDS_DILIGENCE
            if stale_count or unknown_count
            else ScoreSupportStatus.VERIFIED
        ),
        missing_inputs=(
            ["current source dates"] if stale_count or unknown_count else []
        ),
    )


def _evidence_authority_evidence_ids(evidence: list[EvidenceRecord]) -> list[str]:
    freshness_impacted = [
        record
        for record in evidence
        if record.source_freshness in {SourceFreshness.STALE, SourceFreshness.UNKNOWN}
    ]
    ordered = [*freshness_impacted, *evidence]
    return _dedupe_evidence_ids(ordered)


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
    positive_funding_evidence = _positive_funding_evidence(store.evidence)
    negative_funding_evidence = _negative_funding_evidence(store.evidence)
    positive_traction_evidence = _positive_traction_evidence(store.evidence)
    positive_funding_is_not_current = _all_records_not_current(positive_funding_evidence)
    positive_traction_is_not_current = _all_records_not_current(positive_traction_evidence)
    matched_evidence = _dedupe_evidence_records(
        [
            *positive_funding_evidence,
            *negative_funding_evidence,
            *(positive_traction_evidence if positive_traction_is_not_current else []),
        ]
    )
    missing_inputs: list[str] = []
    if not positive_funding_evidence:
        missing_inputs.append(
            "lead investor, institutional investor, or follow-on financing evidence"
        )
    elif positive_funding_is_not_current:
        missing_inputs.append(
            "current lead investor, institutional investor, or follow-on evidence"
        )
    if negative_funding_evidence:
        missing_inputs.append(
            "resolved lead investor, institutional investor, or follow-on financing conflict"
        )
    if not verified_claims:
        missing_inputs.append("verified deal terms")
    if pmf_level != PMFLevel.DEVELOPING:
        missing_inputs.append("customer, revenue, retention, or usage evidence")
    elif positive_traction_is_not_current:
        missing_inputs.append("current customer, revenue, retention, or usage evidence")
    explanation = f"Next-round fundability risk is {fundability_risk}."
    if negative_funding_evidence and positive_funding_evidence:
        explanation = (
            "Next-round fundability risk is high because funding support is "
            "contradicted by source-linked missing lead, institutional, or "
            "follow-on financing evidence."
        )
    elif negative_funding_evidence:
        explanation = (
            "Next-round fundability risk is high because source-linked evidence says "
            "lead, institutional, or follow-on financing is missing."
        )
    elif positive_funding_is_not_current:
        explanation = (
            "Next-round fundability risk is high because funding support appears only "
            "in stale or undated evidence."
        )
    elif positive_traction_is_not_current:
        explanation = (
            "Next-round fundability risk is high because traction support appears only "
            "in stale or undated evidence."
        )
    return ScoreFactor(
        name="Fundability and next-round risk",
        score=score_by_risk[fundability_risk],
        max_score=15,
        explanation=explanation,
        evidence_ids=[evidence.id for evidence in matched_evidence[:5]],
        support_status=(
            ScoreSupportStatus.VERIFIED
            if positive_funding_evidence and not missing_inputs
            else ScoreSupportStatus.NEEDS_DILIGENCE
        ),
        missing_inputs=missing_inputs,
    )


def _valuation_net_return_factor(
    store: EvidenceStore,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
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
    if (
        net_return.net_return_multiple is not None
        and net_return.support_status == ScoreSupportStatus.VERIFIED
    ):
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
        evidence_ids=_valuation_factor_evidence_ids(
            net_return,
            store,
            company_stage,
            pmf_level,
        ),
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
        CompanyStage.SEED: 30_000_000,
        CompanyStage.SERIES_A: 80_000_000,
        CompanyStage.SERIES_B_PLUS: 450_000_000,
        CompanyStage.HARD_TECH_DEFENSE: 45_000_000,
        CompanyStage.UNKNOWN: 20_000_000,
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
        return entry_valuation >= 40_000_000 or (
            entry_valuation >= 25_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SEED:
        return entry_valuation >= 100_000_000 or (
            entry_valuation >= 60_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SERIES_A:
        return entry_valuation >= 250_000_000 or (
            entry_valuation >= 150_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.SERIES_B_PLUS:
        return entry_valuation >= 900_000_000 or (
            entry_valuation >= 650_000_000 and pmf_level != PMFLevel.DEVELOPING
        )
    if company_stage == CompanyStage.HARD_TECH_DEFENSE:
        return entry_valuation >= 450_000_000 or (
            entry_valuation >= 125_000_000 and pmf_level == PMFLevel.UNKNOWN
        )
    if company_stage == CompanyStage.UNKNOWN:
        return entry_valuation >= 40_000_000
    return entry_valuation >= 40_000_000 and pmf_level != PMFLevel.DEVELOPING


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
    if return_inputs.ownership_percent is None:
        missing_inputs.append("ownership")
    if return_inputs.dilution_percent is None:
        missing_inputs.append("dilution")
    if return_inputs.fees_and_carry_percent is None:
        missing_inputs.append("fees or carry")
    if return_inputs.gross_exit_value is None:
        missing_inputs.append("gross exit scenario")

    net_multiple: float | None = None
    if (
        return_inputs.ownership_percent is not None
        and return_inputs.dilution_percent is not None
        and return_inputs.fees_and_carry_percent is not None
        and return_inputs.gross_exit_value is not None
        and entry_valuation > 0
    ):
        net_multiple = _net_return_multiple(
            entry_valuation=entry_valuation,
            return_inputs=return_inputs,
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
            f"Using cited ownership, dilution, fees or carry, and exit value "
            f"against implied invested capital, "
            f"estimated net return is {net_multiple:g}x."
        )
        support_status = ScoreSupportStatus.VERIFIED

    return NetReturnEstimate(
        entry_valuation=entry_valuation,
        estimated_ownership_percent=return_inputs.ownership_percent,
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
        ownership_percent: float | None,
        dilution_percent: float | None,
        platform_fee_percent: float | None,
        carry_percent: float | None,
        combined_fees_and_carry_percent: float | None,
        gross_exit_value: int | None,
        evidence_ids: list[str],
    ) -> None:
        self.ownership_percent = ownership_percent
        self.dilution_percent = dilution_percent
        self.platform_fee_percent = platform_fee_percent
        self.carry_percent = carry_percent
        self.combined_fees_and_carry_percent = combined_fees_and_carry_percent
        self.gross_exit_value = gross_exit_value
        self.evidence_ids = evidence_ids

    @property
    def fees_and_carry_percent(self) -> float | None:
        if self.combined_fees_and_carry_percent is not None:
            return self.combined_fees_and_carry_percent
        if self.platform_fee_percent is None or self.carry_percent is None:
            return None
        return self.platform_fee_percent + self.carry_percent


def _return_inputs(evidence: list[EvidenceRecord]) -> _ReturnInputs:
    ownership_percent: float | None = None
    dilution_percent: float | None = None
    fees_percent: float | None = None
    carry_percent: float | None = None
    combined_fees_and_carry_percent: float | None = None
    gross_exit_value: int | None = None
    evidence_ids: list[str] = []
    for record in evidence:
        ownership_match = RETURN_INPUT_PATTERNS["ownership"].search(record.text)
        if ownership_percent is None and ownership_match:
            parsed_ownership = _float_text(ownership_match.group("value"))
            if parsed_ownership is not None and 0 < parsed_ownership <= 100:
                ownership_percent = parsed_ownership
                evidence_ids.append(record.id)
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
        if (
            gross_exit_value is None
            and exit_match
            and not _match_overlaps_pattern(
                record.text,
                exit_match,
                NEGATED_GROSS_EXIT_PATTERNS,
            )
        ):
            gross_exit_value = _money_text_to_dollars(exit_match.group("value"))
            evidence_ids.append(record.id)
    return _ReturnInputs(
        ownership_percent=ownership_percent,
        dilution_percent=dilution_percent,
        platform_fee_percent=fees_percent,
        carry_percent=carry_percent,
        combined_fees_and_carry_percent=combined_fees_and_carry_percent,
        gross_exit_value=gross_exit_value,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )


def _net_return_multiple(
    *,
    entry_valuation: int,
    return_inputs: _ReturnInputs,
) -> float | None:
    ownership_percent = return_inputs.ownership_percent
    dilution_percent = return_inputs.dilution_percent
    gross_exit_value = return_inputs.gross_exit_value
    if (
        ownership_percent is None
        or ownership_percent <= 0
        or dilution_percent is None
        or gross_exit_value is None
        or entry_valuation <= 0
    ):
        return None

    ownership_fraction = Decimal(str(ownership_percent)) / Decimal("100")
    invested_capital = Decimal(entry_valuation) * ownership_fraction
    if invested_capital <= 0:
        return None

    gross_proceeds = Decimal(gross_exit_value) * ownership_fraction
    dilution_factor = max(
        Decimal("0"),
        Decimal("1") - (Decimal(str(dilution_percent)) / Decimal("100")),
    )
    value_after_dilution = gross_proceeds * dilution_factor

    if return_inputs.combined_fees_and_carry_percent is not None:
        proceeds_after_fees = max(
            Decimal("0"),
            Decimal("1")
            - (
                Decimal(str(return_inputs.combined_fees_and_carry_percent))
                / Decimal("100")
            ),
        )
        net_cash_returned = value_after_dilution * proceeds_after_fees
        cash_in = invested_capital
    elif (
        return_inputs.platform_fee_percent is not None
        and return_inputs.carry_percent is not None
    ):
        platform_fee = (
            invested_capital
            * Decimal(str(return_inputs.platform_fee_percent))
            / Decimal("100")
        )
        profit_after_dilution = max(
            Decimal("0"),
            value_after_dilution - invested_capital,
        )
        carry = (
            profit_after_dilution
            * Decimal(str(return_inputs.carry_percent))
            / Decimal("100")
        )
        net_cash_returned = value_after_dilution - carry
        cash_in = invested_capital + platform_fee
    else:
        return None

    if cash_in <= 0:
        return None
    return round(float(net_cash_returned / cash_in), 2)


def _float_text(raw_value: str) -> float | None:
    try:
        return float(raw_value.strip())
    except ValueError:
        return None


def _valuation_claim(verified_claims: list[ClaimRecord]) -> ClaimRecord | None:
    fallback_claim: ClaimRecord | None = None
    for label in ("post-money valuation", "valuation cap", "pre-money valuation"):
        for claim in verified_claims:
            if claim.label != label:
                continue
            if fallback_claim is None:
                fallback_claim = claim
            if _entry_valuation(verified_claims, claim) is not None:
                return claim
    return fallback_claim


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
    if valuation is None or valuation <= 0:
        return None
    if valuation_claim.label != "pre-money valuation":
        return valuation
    round_size_claim = _round_size_claim(verified_claims)
    if round_size_claim is None:
        return None
    round_size = _claim_money_value(round_size_claim)
    if round_size is None:
        return None
    entry_valuation = valuation + round_size
    return entry_valuation if entry_valuation > 0 else None


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
    traction_evidence = _positive_traction_evidence(store.evidence)
    funding_evidence = _positive_funding_evidence(store.evidence)
    has_traction = bool(traction_evidence)
    has_funding_signal = bool(funding_evidence)
    has_missing_funding_signal = bool(_negative_funding_evidence(store.evidence))
    has_only_non_current_support = (
        (has_traction and _all_records_not_current(traction_evidence))
        or (has_funding_signal and _all_records_not_current(funding_evidence))
    )
    if has_missing_funding_signal:
        return FundabilityRisk.HIGH
    if has_terms and has_traction and has_only_non_current_support:
        return FundabilityRisk.HIGH
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
    kill_gates: list[KillGate],
    research_context: DiligenceResearchContext | None,
) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    if not store.evidence:
        questions.extend(
            [
                _question(
                    category=DiligenceQuestionCategory.COMPANY,
                    question=(
                        "Provide source-linked company materials before relying on this "
                        "deal."
                    ),
                    reason=(
                        "No usable source evidence was available, so material claims "
                        "cannot be checked."
                    ),
                    materiality_score=5,
                    decision_impact_score=5,
                    missing_evidence_score=5,
                    confidence_gap_score=5,
                    missing_evidence=["source-linked evidence"],
                ),
                _question(
                    category=DiligenceQuestionCategory.FINANCING_TERMS,
                    question="Confirm valuation, round size, discount, and minimum check.",
                    reason=(
                        "The check size and recommendation cannot be trusted without "
                        "verified investment terms."
                    ),
                    materiality_score=5,
                    decision_impact_score=5,
                    missing_evidence_score=5,
                    confidence_gap_score=5,
                    missing_evidence=[
                        "verified valuation or valuation cap",
                        "verified round size",
                        "verified discount",
                        "verified minimum check",
                    ],
                ),
                _question(
                    category=DiligenceQuestionCategory.CUSTOMERS,
                    question="Find concrete customer, revenue, retention, or usage evidence.",
                    reason=(
                        "Customer proof is needed to judge whether product demand is real."
                    ),
                    materiality_score=4,
                    decision_impact_score=4,
                    missing_evidence_score=5,
                    confidence_gap_score=4,
                    missing_evidence=[
                        "customer, revenue, retention, usage, pilot, or design-partner proof"
                    ],
                ),
                _question(
                    category=DiligenceQuestionCategory.LEGAL_COMPLIANCE,
                    question=(
                        "Check for legal, regulatory, or platform restrictions before "
                        "advancing diligence."
                    ),
                    reason=(
                        "Without source materials, Hail Mary cannot tell whether any "
                        "legal or platform constraint blocks an investment."
                    ),
                    materiality_score=4,
                    decision_impact_score=4,
                    missing_evidence_score=5,
                    confidence_gap_score=4,
                    missing_evidence=["legal, regulatory, and platform constraint evidence"],
                ),
            ]
        )
        research_question = _research_status_question(research_context)
        if research_question is not None:
            questions.append(research_question)
        return _rank_questions(questions)

    if valid_conflicts:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Resolve the conflicting deal terms in the original documents.",
                reason=(
                    "Conflicting deal terms can change price, ownership, and whether "
                    "the check fits the portfolio."
                ),
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=3,
                confidence_gap_score=5,
                conflicting_evidence_ids=_conflict_evidence_ids(store, valid_conflicts),
                missing_evidence=["resolved source for conflicting deal terms"],
            )
        )
    positive_traction = _positive_traction_evidence(store.evidence)
    negative_traction = _negative_traction_evidence(store.evidence)
    if positive_traction and negative_traction:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.CUSTOMERS,
                question="Resolve the conflicting customer traction signals.",
                reason=(
                    "The evidence contains both traction support and statements that "
                    "traction is missing, which can change product and customer risk."
                ),
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=3,
                confidence_gap_score=5,
                supporting_evidence_ids=_dedupe_evidence_ids(positive_traction),
                conflicting_evidence_ids=_dedupe_evidence_ids(negative_traction),
                missing_evidence=["resolved customer traction evidence"],
            )
        )
    if not verified_claims:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Confirm valuation, round size, discount, and minimum check.",
                reason=(
                    "No verified deal-term claims were available, so pricing and "
                    "basic access terms are still unconfirmed."
                ),
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=5,
                confidence_gap_score=5,
                missing_evidence=[
                    "verified valuation or valuation cap",
                    "verified round size",
                    "verified discount",
                    "verified minimum check",
                ],
            )
        )
    elif "verified round size for pre-money valuation" in net_return.missing_inputs:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Confirm the round size for the verified pre-money valuation.",
                reason=(
                    "A pre-money valuation needs the round size before Hail Mary can "
                    "calculate the post-money entry valuation."
                ),
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=5,
                confidence_gap_score=4,
                supporting_evidence_ids=net_return.evidence_ids,
                missing_evidence=["verified round size for pre-money valuation"],
            )
        )
    elif not _has_readable_pricing_term(verified_claims):
        questions.append(
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Confirm the valuation, valuation cap, or priced-round valuation.",
                reason=(
                    "The evidence did not include a verified pricing term, so Hail "
                    "Mary cannot model entry price or return."
                ),
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=5,
                confidence_gap_score=4,
                supporting_evidence_ids=_claim_evidence_ids(verified_claims),
                missing_evidence=["verified valuation or valuation cap"],
            )
        )
    if company_stage == CompanyStage.UNKNOWN:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.COMPANY,
                question="Confirm the company stage before applying the underwriting bar.",
                reason="The extracted evidence did not include an explicit stage signal.",
                materiality_score=4,
                decision_impact_score=4,
                missing_evidence_score=4,
                confidence_gap_score=3,
                missing_evidence=["explicit company stage"],
            )
        )
    if pmf_level == PMFLevel.UNKNOWN:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.CUSTOMERS,
                question="Find concrete customer, revenue, retention, or usage evidence.",
                reason="The extracted evidence did not show product-market fit signals.",
                materiality_score=4,
                decision_impact_score=4,
                missing_evidence_score=5,
                confidence_gap_score=4,
                missing_evidence=[
                    "customer, revenue, retention, usage, pilot, or design-partner proof"
                ],
            )
        )
    if fundability_risk in {FundabilityRisk.HIGH, FundabilityRisk.UNKNOWN}:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Check whether the company can raise the next round.",
                reason="The evidence has limited investor or growth signals.",
                materiality_score=4,
                decision_impact_score=4,
                missing_evidence_score=4,
                confidence_gap_score=4,
                conflicting_evidence_ids=[
                    evidence.id for evidence in _negative_funding_evidence(store.evidence)
                ],
                missing_evidence=[
                    "lead investor, institutional investor, or follow-on financing evidence"
                ],
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
            _question(
                category=DiligenceQuestionCategory.FINANCING_TERMS,
                question="Confirm why the valuation is justified by current evidence.",
                reason="The verified valuation appears far ahead of stage and traction.",
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=3,
                confidence_gap_score=4,
                supporting_evidence_ids=valuation_evidence_ids,
                missing_evidence=["current evidence justifying the valuation"],
            )
        )
    if net_return.missing_inputs:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.PORTFOLIO_FIT,
                question="Collect the missing return-math inputs before sizing the check.",
                reason=(
                    "Hail Mary needs verified "
                    f"{', '.join(net_return.missing_inputs)} to model net return."
                ),
                materiality_score=4,
                decision_impact_score=5,
                missing_evidence_score=5,
                confidence_gap_score=4,
                supporting_evidence_ids=net_return.evidence_ids,
                missing_evidence=net_return.missing_inputs,
            )
        )
    questions.extend(
        _evidence_verification_questions(
            store,
            verified_claims,
        )
    )
    research_question = _research_status_question(research_context)
    if research_question is not None:
        questions.append(research_question)
    questions.extend(_triggered_gate_questions(kill_gates))
    if not questions:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.COMPANY,
                question="Verify that the strongest claims remain true in current materials.",
                reason="The deterministic checks did not find a blocking gap.",
                materiality_score=3,
                decision_impact_score=3,
                missing_evidence_score=2,
                confidence_gap_score=2,
                supporting_evidence_ids=_dedupe_evidence_ids(store.evidence),
                missing_evidence=["current confirmation of strongest claims"],
            )
        )
    return _rank_questions(questions)


def _question(
    *,
    category: DiligenceQuestionCategory,
    question: str,
    reason: str,
    materiality_score: int,
    decision_impact_score: int,
    missing_evidence_score: int,
    confidence_gap_score: int,
    supporting_evidence_ids: Iterable[str] = (),
    conflicting_evidence_ids: Iterable[str] = (),
    missing_evidence: Iterable[str] = (),
    support_status: ScoreSupportStatus = ScoreSupportStatus.NEEDS_DILIGENCE,
) -> DiligenceQuestion:
    supporting_ids = _dedupe_strings(supporting_evidence_ids)
    conflicting_ids = _dedupe_strings(conflicting_evidence_ids)
    missing_labels = _dedupe_strings(missing_evidence, limit=None)
    evidence_ids = _question_evidence_ids(supporting_ids, conflicting_ids)
    return DiligenceQuestion(
        priority=0,
        question=question,
        reason=reason,
        evidence_ids=evidence_ids,
        support_status=support_status,
        category=category,
        rank_score=(
            materiality_score
            + decision_impact_score
            + missing_evidence_score
            + confidence_gap_score
        ),
        materiality_score=materiality_score,
        decision_impact_score=decision_impact_score,
        missing_evidence_score=missing_evidence_score,
        confidence_gap_score=confidence_gap_score,
        supporting_evidence_ids=supporting_ids,
        conflicting_evidence_ids=conflicting_ids,
        missing_evidence=missing_labels,
    )


def _question_evidence_ids(
    supporting_evidence_ids: list[str],
    conflicting_evidence_ids: list[str],
) -> list[str]:
    if not conflicting_evidence_ids:
        return _dedupe_strings(supporting_evidence_ids)
    if not supporting_evidence_ids:
        return _dedupe_strings(conflicting_evidence_ids)
    evidence_ids = _dedupe_strings(
        [*supporting_evidence_ids[:4], *conflicting_evidence_ids]
    )
    if any(evidence_id in evidence_ids for evidence_id in conflicting_evidence_ids):
        return evidence_ids
    return _dedupe_strings([*supporting_evidence_ids[:4], conflicting_evidence_ids[0]])


def _rank_questions(questions: list[DiligenceQuestion]) -> list[DiligenceQuestion]:
    deduped: dict[str, DiligenceQuestion] = {}
    for question in questions:
        existing = deduped.get(question.question)
        if existing is None or question.rank_score > existing.rank_score:
            deduped[question.question] = question
    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -item.rank_score,
            item.category.value,
            item.question.casefold(),
        ),
    )
    return [
        question.model_copy(update={"priority": priority})
        for priority, question in enumerate(ranked, start=1)
    ]


def _evidence_verification_questions(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    stale_or_unknown_ids = [
        evidence.id
        for evidence in store.evidence
        if evidence.source_freshness in {SourceFreshness.STALE, SourceFreshness.UNKNOWN}
    ]
    if stale_or_unknown_ids:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.COMPANY,
                question="Verify stale or undated evidence before relying on it.",
                reason=(
                    "Old or undated evidence may no longer reflect the company's "
                    "current traction, terms, or risks."
                ),
                materiality_score=4,
                decision_impact_score=3,
                missing_evidence_score=3,
                confidence_gap_score=4,
                supporting_evidence_ids=stale_or_unknown_ids,
                missing_evidence=["current source dates"],
            )
        )
    low_confidence_claim_ids = _low_confidence_claim_evidence_ids(verified_claims)
    if low_confidence_claim_ids:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.COMPANY,
                question="Verify low-confidence extracted claims against the source.",
                reason=(
                    "Low-confidence claims can distort score factors if extraction or "
                    "source quality was weak."
                ),
                materiality_score=4,
                decision_impact_score=3,
                missing_evidence_score=3,
                confidence_gap_score=5,
                supporting_evidence_ids=low_confidence_claim_ids,
                missing_evidence=["higher-confidence support for extracted claims"],
            )
        )
    image_text_ids = [
        evidence.id
        for evidence in store.evidence
        if evidence.ocr_applied
        and (
            evidence.ocr_confidence is None
            or evidence.ocr_confidence < 0.75
        )
    ]
    if image_text_ids:
        questions.append(
            _question(
                category=DiligenceQuestionCategory.COMPANY,
                question="Review low-confidence image-read evidence against the source.",
                reason=(
                    "Image-read text can misread numbers or names, so it needs source "
                    "review before it drives diligence."
                ),
                materiality_score=4,
                decision_impact_score=3,
                missing_evidence_score=3,
                confidence_gap_score=5,
                supporting_evidence_ids=image_text_ids,
                missing_evidence=["source review for low-confidence image-read evidence"],
            )
        )
    return questions


def _research_status_question(
    research_context: DiligenceResearchContext | None,
) -> DiligenceQuestion | None:
    if research_context is None or research_context.planned_task_count == 0:
        return None
    missing_evidence: list[str] = []
    if research_context.failed_provider_count:
        missing_evidence.append("resolved failed external research providers")
    if research_context.incomplete_search_count:
        missing_evidence.append("completed external research searches")
    if research_context.no_exact_result_provider_count:
        missing_evidence.append("exact-match external research results")
    if research_context.manual_needed_provider_count:
        missing_evidence.append("manual external research results")
    if research_context.not_run_provider_count:
        missing_evidence.append("run or explicitly skipped external research providers")
    if research_context.stale_record_count:
        missing_evidence.append("current external research records")
    if research_context.stale_only_research:
        missing_evidence.append("fresh external research support")
    if research_context.unknown_reliability_record_count:
        missing_evidence.append("verified source reliability for external research")
    if research_context.ambiguous_or_related_match_count:
        missing_evidence.append("resolved ambiguous external research identity matches")
    if research_context.identity_mismatch_count:
        missing_evidence.append("resolved external research identity mismatches")
    if research_context.warning_count:
        missing_evidence.append("resolved external research warnings")
    if research_context.no_prepared_result_companies:
        missing_evidence.append("prepared external research for requested companies")
    if not missing_evidence:
        return None
    return _question(
        category=DiligenceQuestionCategory.MARKET,
        question="Finish unresolved external research before relying on public-source gaps.",
        reason=(
            "Incomplete public-source research can leave market, customer, financing, "
            "or legal facts unverified."
        ),
        materiality_score=4,
        decision_impact_score=3,
        missing_evidence_score=4,
        confidence_gap_score=4,
        missing_evidence=missing_evidence,
    )


def _triggered_gate_questions(kill_gates: list[KillGate]) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    for gate in kill_gates:
        if not gate.triggered or gate.name in {
            "No usable source-linked evidence",
            "No verified deal terms",
            "Missing key investment terms",
            "Conflicting material deal terms",
            "Valuation far ahead of evidence",
        }:
            continue
        questions.append(
            _question(
                category=DiligenceQuestionCategory.PORTFOLIO_FIT,
                question=f"Resolve the triggered gate: {gate.name}.",
                reason=gate.reason,
                materiality_score=5,
                decision_impact_score=5,
                missing_evidence_score=3,
                confidence_gap_score=4,
                supporting_evidence_ids=gate.evidence_ids,
                missing_evidence=[gate.name],
            )
        )
    return questions


def _check_sizing_decision(
    store: EvidenceStore,
    total_score: int,
    *,
    config: AppConfig,
    confidence: ConfidenceLevel,
    platform_minimum_check: int | None,
    capital_remaining: int,
    valuation_risk: ValuationRisk,
    fundability_risk: FundabilityRisk,
    net_return: NetReturnEstimate,
    company_stage: CompanyStage,
    exposure_state: PortfolioExposureState | None,
) -> CheckSizingDecision:
    target = _target_check_size(total_score, confidence=confidence)
    risk_cap = _check_size_cap(
        target,
        confidence=confidence,
        valuation_risk=valuation_risk,
        fundability_risk=fundability_risk,
        net_return=net_return,
    )
    reason_codes = ["score_target"]
    effective_cap = risk_cap
    if risk_cap is not None:
        target = min(target, risk_cap)
        reason_codes.append("risk_cap_applied")
    if platform_minimum_check is not None:
        if platform_minimum_check > target:
            reason_codes.append("platform_minimum_applied")
        target = max(target, platform_minimum_check)
    exposure_checks = portfolio_exposure_checks(
        config=config,
        store=store,
        company_stage=company_stage,
        confidence=confidence,
        exposure_state=exposure_state or empty_portfolio_exposure_state(),
    )
    exposure_cap = portfolio_exposure_cap(exposure_checks)
    if exposure_cap is not None:
        effective_cap = (
            exposure_cap if effective_cap is None else min(effective_cap, exposure_cap)
        )
        reason_codes.append("exposure_limit_applied")
    if capital_remaining < config.max_check:
        reason_codes.append("capital_limited")
    allowed_tiers = _available_nonzero_tiers(
        config,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=capital_remaining,
        check_size_cap=effective_cap,
    )
    if not allowed_tiers:
        reason_codes.extend(
            _no_tier_reason_codes(
                config=config,
                platform_minimum_check=platform_minimum_check,
                capital_remaining=capital_remaining,
                risk_cap=risk_cap,
                exposure_cap=exposure_cap,
            )
        )
        return CheckSizingDecision(
            score_target=_target_check_size(total_score, confidence=confidence),
            risk_cap=risk_cap,
            platform_minimum_check=platform_minimum_check,
            exposure_cap=exposure_cap,
            allowed_tiers=[],
            selected_tier=0,
            reason_codes=list(dict.fromkeys(reason_codes)),
            exposure_checks=exposure_checks,
        )
    tiers_at_or_above_target = [tier for tier in allowed_tiers if tier >= target]
    if tiers_at_or_above_target:
        selected_tier = min(tiers_at_or_above_target)
    else:
        selected_tier = max(allowed_tiers)
        reason_codes.append("selected_lower_allowed_tier")
    return CheckSizingDecision(
        score_target=_target_check_size(total_score, confidence=confidence),
        risk_cap=risk_cap,
        platform_minimum_check=platform_minimum_check,
        exposure_cap=exposure_cap,
        allowed_tiers=allowed_tiers,
        selected_tier=selected_tier,
        reason_codes=list(dict.fromkeys(reason_codes)),
        exposure_checks=exposure_checks,
    )


def _check_size_cap(
    target: int,
    *,
    confidence: ConfidenceLevel,
    valuation_risk: ValuationRisk,
    fundability_risk: FundabilityRisk,
    net_return: NetReturnEstimate,
) -> int | None:
    cap: int | None = None
    if confidence == ConfidenceLevel.LOW:
        cap = 1_000
    if (
        valuation_risk == ValuationRisk.MEDIUM
        or fundability_risk in {FundabilityRisk.MEDIUM, FundabilityRisk.HIGH}
        or net_return.missing_inputs
    ):
        cap = _min_optional_cap(cap, _one_tier_lower(target))
    return cap


def _no_available_check_size_reason(
    total_score: int,
    *,
    config: AppConfig,
    confidence: ConfidenceLevel,
    platform_minimum_check: int | None,
    capital_remaining: int,
    valuation_risk: ValuationRisk,
    fundability_risk: FundabilityRisk,
    net_return: NetReturnEstimate,
    check_sizing: CheckSizingDecision,
) -> str:
    reason_codes = set(check_sizing.reason_codes)
    if "risk_cap_below_minimum" in reason_codes:
        return (
            "Risk caps lowered the maximum check below the configured or "
            "platform minimum check."
        )
    if "risk_cap_no_tier" in reason_codes:
        return (
            "No configured check size fits the risk cap, platform minimum, "
            "and remaining capital."
        )
    if "exposure_limit_no_tier" in reason_codes or "exposure_cap_below_minimum" in reason_codes:
        return (
            "No configured check size fits the exposure limits, platform minimum, "
            "and remaining capital."
        )
    target = _target_check_size(total_score, confidence=confidence)
    cap = _check_size_cap(
        target,
        confidence=confidence,
        valuation_risk=valuation_risk,
        fundability_risk=fundability_risk,
        net_return=net_return,
    )
    if cap is not None:
        base_tiers = _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
        )
        capped_tiers = _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
            check_size_cap=cap,
        )
        if base_tiers and not capped_tiers:
            minimum_check = config.min_check
            if platform_minimum_check is not None:
                minimum_check = max(minimum_check, platform_minimum_check)
            if cap < minimum_check:
                return (
                    "Risk caps lowered the maximum check below the configured or "
                    "platform minimum check."
                )
            return (
                "No configured check size fits the risk cap, platform minimum, "
                "and remaining capital."
            )
    return "No configured check size fits the platform minimum and remaining capital."


def _no_tier_reason_codes(
    *,
    config: AppConfig,
    platform_minimum_check: int | None,
    capital_remaining: int,
    risk_cap: int | None,
    exposure_cap: int | None,
) -> list[str]:
    reason_codes = ["no_allowed_tier"]
    minimum_check = config.min_check
    if platform_minimum_check is not None:
        minimum_check = max(minimum_check, platform_minimum_check)
    if capital_remaining < minimum_check:
        reason_codes.append("capital_below_minimum")
    base_tiers = _available_nonzero_tiers(
        config,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=capital_remaining,
    )
    if risk_cap is not None:
        risk_tiers = _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
            check_size_cap=risk_cap,
        )
        if base_tiers and not risk_tiers:
            reason_codes.append(
                "risk_cap_below_minimum"
                if risk_cap < minimum_check
                else "risk_cap_no_tier"
            )
    if exposure_cap is not None:
        cap_before_exposure = risk_cap
        before_exposure_tiers = _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
            check_size_cap=cap_before_exposure,
        )
        effective_cap = (
            exposure_cap
            if cap_before_exposure is None
            else min(cap_before_exposure, exposure_cap)
        )
        after_exposure_tiers = _available_nonzero_tiers(
            config,
            platform_minimum_check=platform_minimum_check,
            capital_remaining=capital_remaining,
            check_size_cap=effective_cap,
        )
        if before_exposure_tiers and not after_exposure_tiers:
            reason_codes.append(
                "exposure_cap_below_minimum"
                if exposure_cap < minimum_check
                else "exposure_limit_no_tier"
            )
    return reason_codes


def _allocation_scenario(
    *,
    config: AppConfig,
    capital_remaining: int,
    check_size: int,
) -> PortfolioAllocationScenario:
    scenario = portfolio_scenario(config)
    return PortfolioAllocationScenario(
        starting_capital=scenario.starting_capital,
        follow_on_reserve=scenario.reserve_amount,
        allocatable_capital=scenario.allocatable_capital,
        capital_before_check=capital_remaining,
        current_check=check_size,
        capital_after_check=max(0, capital_remaining - check_size),
    )


def _min_optional_cap(current_cap: int | None, candidate_cap: int) -> int:
    if current_cap is None:
        return candidate_cap
    return min(current_cap, candidate_cap)


def _one_tier_lower(target: int) -> int:
    nonzero_tiers = [tier for tier in CHECK_SIZE_TIERS if tier > 0]
    lower_or_equal = [tier for tier in nonzero_tiers if tier <= target]
    if not lower_or_equal:
        return nonzero_tiers[0]
    current_index = nonzero_tiers.index(max(lower_or_equal))
    return nonzero_tiers[max(0, current_index - 1)]


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
    check_size_cap: int | None = None,
) -> list[int]:
    minimum_check = config.min_check
    if platform_minimum_check is not None:
        minimum_check = max(minimum_check, platform_minimum_check)
    maximum_check = min(config.max_check, capital_remaining, HARD_MAX_CHECK)
    if check_size_cap is not None:
        maximum_check = min(maximum_check, check_size_cap)
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


def _missing_key_terms_reason(verified_claims: list[ClaimRecord]) -> str:
    if _pre_money_claim_needs_round_size(verified_claims) is not None:
        return (
            "A verified pre-money valuation needs a verified round size before "
            "Hail Mary can calculate the entry valuation."
        )
    return "No verified valuation or valuation-cap term was found."


def _missing_key_terms_evidence_ids(verified_claims: list[ClaimRecord]) -> list[str]:
    pre_money_claim = _pre_money_claim_needs_round_size(verified_claims)
    if pre_money_claim is None:
        return _claim_evidence_ids(verified_claims)
    evidence_ids = _claim_evidence_ids([pre_money_claim])
    round_size_claim = _round_size_claim(verified_claims)
    if round_size_claim is not None:
        evidence_ids.extend(_claim_evidence_ids([round_size_claim]))
    return list(dict.fromkeys(evidence_ids))


def _pre_money_claim_needs_round_size(
    verified_claims: list[ClaimRecord],
) -> ClaimRecord | None:
    round_size_claim = _round_size_claim(verified_claims)
    round_size = _claim_money_value(round_size_claim) if round_size_claim is not None else None
    if round_size is not None:
        return None
    for claim in verified_claims:
        if claim.label == "pre-money valuation":
            valuation = _claim_money_value(claim)
            if valuation is not None and valuation > 0:
                return claim
    return None


def _claim_money_value(claim: ClaimRecord) -> int | None:
    normalized_prefix = "usd_cents:"
    if claim.normalized_value.startswith(normalized_prefix):
        try:
            cents = int(claim.normalized_value.removeprefix(normalized_prefix))
        except ValueError:
            return None
        return _ceil_cents_to_dollars(cents)
    return _money_text_to_dollars(claim.value)


def _ceil_cents_to_dollars(cents: int) -> int:
    if cents > 0:
        return (cents + 99) // 100
    return cents // 100


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
        return int(
            (Decimal(normalized) * multiplier).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
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


def _negative_traction_evidence(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if _negated_spans(record.text, ABSENCE_TRACTION_PATTERNS)
    ]


def _dedupe_evidence_records(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return list({record.id: record for record in evidence}.values())


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


def _negative_funding_evidence(evidence: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        record
        for record in evidence
        if _negated_spans(record.text, NEGATED_FUNDING_PATTERNS)
    ]


def _all_records_not_current(evidence: list[EvidenceRecord]) -> bool:
    return bool(evidence) and all(
        record.source_freshness in {SourceFreshness.STALE, SourceFreshness.UNKNOWN}
        for record in evidence
    )


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
    spans = [
        match.span()
        for pattern in negated_patterns
        for match in pattern.finditer(text)
    ]
    if (
        negated_patterns is NEGATED_TRACTION_PATTERNS
        or negated_patterns is ABSENCE_TRACTION_PATTERNS
    ):
        spans.extend(_without_traction_negated_spans(text))
    return spans


def _without_traction_negated_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for without_match in WITHOUT_TRACTION_PATTERN.finditer(text):
        first_term = TRACTION_NEGATED_TERM_PATTERN.match(text, without_match.end())
        if first_term is None or _traction_term_has_benign_noun(text, first_term.end()):
            continue
        spans.append(first_term.span())
        cursor = first_term.end()

        while separator := WITHOUT_TRACTION_SEPARATOR_PATTERN.match(text, cursor):
            term = TRACTION_NEGATED_TERM_PATTERN.match(text, separator.end())
            if term is None or _traction_term_has_benign_noun(text, term.end()):
                break
            if TRACTION_POSITIVE_CLAUSE_PATTERN.match(text[term.start() :]):
                break
            spans.append(term.span())
            cursor = term.end()

    return spans


def _traction_term_has_benign_noun(text: str, term_end: int) -> bool:
    return bool(
        re.match(
            rf"\s+{BENIGN_NEGATED_TRACTION_NOUNS}\b",
            text[term_end:],
            flags=re.IGNORECASE,
        )
    )


def _span_overlaps(
    span: tuple[int, int],
    spans: list[tuple[int, int]],
) -> bool:
    start, end = span
    return any(start < negated_end and negated_start < end for negated_start, negated_end in spans)


def _match_overlaps_pattern(
    text: str,
    match: re.Match[str],
    patterns: tuple[re.Pattern[str], ...],
) -> bool:
    return _span_overlaps(match.span(), _negated_spans(text, patterns))


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


def _low_confidence_claim_evidence_ids(claims: list[ClaimRecord]) -> list[str]:
    return _claim_evidence_ids(
        [claim for claim in claims if claim.quality.confidence < 0.75]
    )


def _pricing_evidence_ids(verified_claims: list[ClaimRecord]) -> list[str]:
    valuation_claim = _valuation_claim(verified_claims)
    if valuation_claim is None or _entry_valuation(verified_claims, valuation_claim) is None:
        return []
    evidence_ids = _claim_evidence_ids([valuation_claim])
    if valuation_claim.label == "pre-money valuation":
        round_size_claim = _round_size_claim(verified_claims)
        if round_size_claim is not None and _claim_money_value(round_size_claim) is not None:
            evidence_ids.extend(_claim_evidence_ids([round_size_claim]))
    return list(dict.fromkeys(evidence_ids))


def _valuation_factor_evidence_ids(
    net_return: NetReturnEstimate,
    store: EvidenceStore,
    company_stage: CompanyStage,
    pmf_level: PMFLevel,
) -> list[str]:
    evidence_ids = list(net_return.evidence_ids)
    evidence_ids.extend(_dedupe_evidence_ids(_stage_evidence(store.evidence, company_stage)))
    evidence_ids.extend(_dedupe_evidence_ids(_pmf_evidence(store.evidence, pmf_level)))
    return list(dict.fromkeys(evidence_ids))


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


def _dedupe_strings(values: Iterable[str], *, limit: int | None = 5) -> list[str]:
    deduped: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in deduped:
            continue
        deduped.append(stripped)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


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
