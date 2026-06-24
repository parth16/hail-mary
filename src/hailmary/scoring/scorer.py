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
    ConfidenceLevel,
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
INVEST_MINIMUM_SCORE = 75
HARD_MAX_CHECK = max(CHECK_SIZE_TIERS)
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
    fundability_risk = _fundability_risk(store, verified_claims)
    confidence = _confidence_level(store, verified_claims, valid_conflicts)
    kill_gates = _kill_gates(
        store,
        verified_claims,
        valid_conflicts,
        config=config,
        platform_minimum_check=platform_minimum_check,
        capital_remaining=available_capital,
    )
    score_factors = _score_factors(
        store,
        verified_claims,
        valid_conflicts,
        pmf_level,
        fundability_risk,
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
        kill_gates=kill_gates,
        score_factors=score_factors,
        diligence_questions=_diligence_questions(
            store,
            verified_claims,
            pmf_level=pmf_level,
            fundability_risk=fundability_risk,
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
    missing_key_terms = has_scorable_deal and not _has_pricing_term(verified_claims)
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
            triggered=bool(valid_conflicts),
            reason=(
                "One or more extracted deal terms conflict and still have valid citations."
                if valid_conflicts
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
        KillGate(
            name="Missing key investment terms",
            triggered=missing_key_terms,
            reason=(
                "No verified valuation or valuation-cap term was found."
                if missing_key_terms
                else "A verified valuation or valuation-cap term is available."
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
        ),
        KillGate(
            name="No available check size",
            triggered=no_check_available,
            reason=(
                "No configured check size fits the platform minimum and remaining capital."
                if no_check_available
                else "At least one configured check size fits the remaining capital."
            ),
        ),
    ]


def _score_factors(
    store: EvidenceStore,
    verified_claims: list[ClaimRecord],
    valid_conflicts: list[ClaimConflict],
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
) -> list[ScoreFactor]:
    return [
        _evidence_coverage_factor(store),
        _deal_terms_factor(verified_claims, valid_conflicts),
        _pmf_factor(store, pmf_level),
        _fundability_factor(store, fundability_risk),
        _evidence_quality_factor(store, valid_conflicts),
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
    valid_conflicts: list[ClaimConflict],
) -> ScoreFactor:
    unique_labels = sorted({claim.label for claim in verified_claims})
    score = min(25, len(unique_labels) * 7)
    if valid_conflicts:
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
    if pmf_level == PMFLevel.DEVELOPING:
        matched_evidence = _positive_traction_evidence(store.evidence)
    elif pmf_level == PMFLevel.EARLY:
        matched_evidence = _positive_early_pmf_evidence(store.evidence)
    else:
        matched_evidence = []
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
    matched_evidence = _positive_funding_evidence(store.evidence)
    return ScoreFactor(
        name="Next-round fundability",
        score=score_by_risk[fundability_risk],
        max_score=20,
        explanation=f"Next-round fundability risk is {fundability_risk}.",
        evidence_ids=[evidence.id for evidence in matched_evidence[:5]],
    )


def _evidence_quality_factor(
    store: EvidenceStore,
    valid_conflicts: list[ClaimConflict],
) -> ScoreFactor:
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
        if valid_conflicts:
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


def _pmf_level(evidence: list[EvidenceRecord]) -> PMFLevel:
    if _positive_traction_evidence(evidence):
        return PMFLevel.DEVELOPING
    if _positive_early_pmf_evidence(evidence):
        return PMFLevel.EARLY
    return PMFLevel.UNKNOWN


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
    pmf_level: PMFLevel,
    fundability_risk: FundabilityRisk,
    valid_conflicts: list[ClaimConflict],
) -> list[DiligenceQuestion]:
    questions: list[DiligenceQuestion] = []
    if valid_conflicts:
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
    elif not _has_pricing_term(verified_claims):
        questions.append(
            DiligenceQuestion(
                priority=2,
                question="Confirm the valuation, valuation cap, or priced-round valuation.",
                reason="The evidence did not include a verified pricing term.",
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
