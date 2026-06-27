from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from hailmary.evidence.store import verify_citation
from hailmary.schemas.evidence import (
    ClaimRecord,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)
from hailmary.schemas.scoring import ScoredDeal, ScoreSupportStatus
from hailmary.scoring.scorer import validated_conflicts, validated_verified_claims
from hailmary.utils.source_instructions import looks_like_embedded_source_instruction

LOW_CLAIM_CONFIDENCE_THRESHOLD = 0.5


class EvidenceAuditReadiness(StrEnum):
    SUFFICIENT = "sufficient"
    NEEDS_DILIGENCE = "needs_diligence"
    INSUFFICIENT = "insufficient"


class EvidenceAuditSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"


class EvidenceAuditFindingKind(StrEnum):
    EMPTY_STORE = "empty_store"
    MISSING_TERM = "missing_term"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    WEAK_SUPPORT = "weak_support"
    STALE_EVIDENCE = "stale_evidence"
    CONFLICT = "conflict"
    UNSAFE_SOURCE = "unsafe_source"


class EvidenceAuditTerm(StrEnum):
    PRICE_VALUATION = "price_valuation"
    ROUND_SIZE = "round_size"
    SECURITY_TYPE = "security_type"
    MINIMUM_CHECK = "minimum_check"
    PLATFORM_FEES_CARRY = "platform_fees_carry"
    DILUTION = "dilution"
    LEAD_INVESTOR = "lead_investor"
    TRACTION = "traction"
    REVENUE = "revenue"
    CUSTOMER_PROOF = "customer_proof"
    MARKET = "market"
    TEAM = "team"
    USE_OF_FUNDS = "use_of_funds"


class EvidenceAuditCoverageStatus(StrEnum):
    SUPPORTED = "supported"
    MISSING = "missing"
    WEAK = "weak"
    STALE = "stale"
    UNSAFE = "unsafe"


class EvidenceAuditFinding(BaseModel):
    id: str
    kind: EvidenceAuditFindingKind
    severity: EvidenceAuditSeverity
    title: str
    explanation: str
    term: EvidenceAuditTerm | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False
    claim_ids: list[str] = Field(default_factory=list)
    source_kinds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def must_have_lineage_or_missing_marker(self) -> Self:
        if not self.evidence_ids and not self.missing_evidence:
            raise ValueError(
                "Audit findings must include evidence_ids or set missing_evidence."
            )
        return self


class EvidenceAuditTermStatus(BaseModel):
    term: EvidenceAuditTerm
    label: str
    status: EvidenceAuditCoverageStatus
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False


class EvidenceAuditQuestion(BaseModel):
    priority: int
    question: str
    reason: str
    term: EvidenceAuditTerm | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False


class EvidenceCompletenessAudit(BaseModel):
    deal_id: str
    company_name: str
    readiness: EvidenceAuditReadiness
    term_statuses: list[EvidenceAuditTermStatus] = Field(default_factory=list)
    findings: list[EvidenceAuditFinding] = Field(default_factory=list)
    questions: list[EvidenceAuditQuestion] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[EvidenceAuditFinding]:
        return [
            finding
            for finding in self.findings
            if finding.severity == EvidenceAuditSeverity.BLOCKING
        ]


_TERM_LABELS = {
    EvidenceAuditTerm.PRICE_VALUATION: "price or valuation",
    EvidenceAuditTerm.ROUND_SIZE: "round size",
    EvidenceAuditTerm.SECURITY_TYPE: "security type",
    EvidenceAuditTerm.MINIMUM_CHECK: "minimum check",
    EvidenceAuditTerm.PLATFORM_FEES_CARRY: "platform fees or carry",
    EvidenceAuditTerm.DILUTION: "dilution",
    EvidenceAuditTerm.LEAD_INVESTOR: "lead investor",
    EvidenceAuditTerm.TRACTION: "traction",
    EvidenceAuditTerm.REVENUE: "revenue",
    EvidenceAuditTerm.CUSTOMER_PROOF: "customer proof",
    EvidenceAuditTerm.MARKET: "market",
    EvidenceAuditTerm.TEAM: "team",
    EvidenceAuditTerm.USE_OF_FUNDS: "use of funds",
}

_CLAIM_LABEL_TERMS = {
    "valuation cap": EvidenceAuditTerm.PRICE_VALUATION,
    "pre-money valuation": EvidenceAuditTerm.PRICE_VALUATION,
    "post-money valuation": EvidenceAuditTerm.PRICE_VALUATION,
    "round size": EvidenceAuditTerm.ROUND_SIZE,
    "minimum investment": EvidenceAuditTerm.MINIMUM_CHECK,
}

_KEYWORD_TERMS: dict[EvidenceAuditTerm, tuple[str, ...]] = {
    EvidenceAuditTerm.SECURITY_TYPE: (
        "simple agreement for future equity",
        "convertible note",
        "priced round",
        "preferred stock",
        "common stock",
        "security type",
        "SAFE",
    ),
    EvidenceAuditTerm.PLATFORM_FEES_CARRY: (
        "platform fee",
        "management fee",
        "fees and carry",
        "fees & carry",
        "carry",
        "SPV fee",
    ),
    EvidenceAuditTerm.DILUTION: ("dilution", "ownership after dilution"),
    EvidenceAuditTerm.LEAD_INVESTOR: (
        "lead investor",
        "led by",
        "co-led by",
        "anchor investor",
        "institutional investor",
    ),
    EvidenceAuditTerm.TRACTION: (
        "traction",
        "usage",
        "retention",
        "growth",
        "paid users",
        "ARR",
        "MRR",
    ),
    EvidenceAuditTerm.REVENUE: ("revenue", "ARR", "MRR", "bookings"),
    EvidenceAuditTerm.CUSTOMER_PROOF: (
        "customer",
        "customers",
        "pilot",
        "design partner",
        "LOI",
        "case study",
    ),
    EvidenceAuditTerm.MARKET: (
        "market size",
        "market opportunity",
        "total addressable market",
        "TAM",
        "competition",
        "competitor",
    ),
    EvidenceAuditTerm.TEAM: (
        "founder",
        "co-founder",
        "CEO",
        "CTO",
        "team",
        "operator",
        "experience",
    ),
    EvidenceAuditTerm.USE_OF_FUNDS: (
        "use of funds",
        "use of proceeds",
        "proceeds will",
        "runway",
        "hire",
        "go-to-market",
    ),
}

_BLOCKING_MISSING_TERMS = frozenset({EvidenceAuditTerm.PRICE_VALUATION})
_QUESTION_PRIORITY = {
    EvidenceAuditTerm.PRICE_VALUATION: 1,
    EvidenceAuditTerm.ROUND_SIZE: 2,
    EvidenceAuditTerm.SECURITY_TYPE: 3,
    EvidenceAuditTerm.MINIMUM_CHECK: 4,
    EvidenceAuditTerm.PLATFORM_FEES_CARRY: 5,
    EvidenceAuditTerm.DILUTION: 6,
    EvidenceAuditTerm.LEAD_INVESTOR: 7,
    EvidenceAuditTerm.TRACTION: 8,
    EvidenceAuditTerm.REVENUE: 9,
    EvidenceAuditTerm.CUSTOMER_PROOF: 10,
    EvidenceAuditTerm.MARKET: 11,
    EvidenceAuditTerm.TEAM: 12,
    EvidenceAuditTerm.USE_OF_FUNDS: 13,
}

_NEGATED_KEYWORD_PREFIX = re.compile(
    r"\b(?:no|without|lacks?|lacking|does\s+not\s+have|do\s+not\s+have|"
    r"has\s+not|not\s+yet|pre[-\s]?revenue)\b(?:[\W_]+\w+){0,5}[\W_]*$",
    re.IGNORECASE,
)


def build_evidence_completeness_audit(
    store: EvidenceStore,
    *,
    scored_deal: ScoredDeal | None = None,
) -> EvidenceCompletenessAudit:
    """Build a structured, read-only audit of decision-support evidence."""

    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    unsafe_evidence_ids = {
        evidence.id
        for evidence in store.evidence
        if looks_like_embedded_source_instruction(evidence.text)
    }
    safe_evidence = [
        evidence for evidence in store.evidence if evidence.id not in unsafe_evidence_ids
    ]
    safe_evidence_by_id = {evidence.id: evidence for evidence in safe_evidence}
    verified_claims = [
        claim
        for claim in validated_verified_claims(store)
        if _claim_citations_are_safe(claim, unsafe_evidence_ids)
    ]
    term_statuses = _term_statuses(
        store,
        safe_evidence=safe_evidence,
        verified_claims=verified_claims,
        unsafe_evidence_ids=unsafe_evidence_ids,
    )
    findings: list[EvidenceAuditFinding] = []
    findings.extend(_empty_store_findings(store))
    findings.extend(_unsafe_source_findings(store, unsafe_evidence_ids))
    findings.extend(_missing_term_findings(term_statuses))
    findings.extend(_unsupported_claim_findings(store, evidence_by_id, unsafe_evidence_ids))
    findings.extend(_freshness_findings(store))
    findings.extend(_conflict_findings(store))
    if scored_deal is not None:
        findings.extend(_scoring_support_findings(scored_deal, safe_evidence_by_id))

    findings = _dedupe_findings(findings)
    questions = _questions_from_findings(term_statuses, findings, scored_deal=scored_deal)
    readiness = _readiness(term_statuses, findings)
    return EvidenceCompletenessAudit(
        deal_id=store.deal_id,
        company_name=store.company_name,
        readiness=readiness,
        term_statuses=term_statuses,
        findings=findings,
        questions=questions,
    )


def _term_statuses(
    store: EvidenceStore,
    *,
    safe_evidence: list[EvidenceRecord],
    verified_claims: list[ClaimRecord],
    unsafe_evidence_ids: set[str],
) -> list[EvidenceAuditTermStatus]:
    evidence_by_term: dict[EvidenceAuditTerm, list[str]] = {
        term: [] for term in EvidenceAuditTerm
    }
    unsafe_by_term: dict[EvidenceAuditTerm, list[str]] = {
        term: [] for term in EvidenceAuditTerm
    }
    for claim in verified_claims:
        term = _CLAIM_LABEL_TERMS.get(claim.label)
        if term is None:
            continue
        evidence_by_term[term].extend(_claim_evidence_ids([claim]))

    for evidence in safe_evidence:
        for term, keywords in _KEYWORD_TERMS.items():
            if _contains_any_positive_keyword(evidence.text, keywords):
                evidence_by_term[term].append(evidence.id)

    for evidence in store.evidence:
        if evidence.id not in unsafe_evidence_ids:
            continue
        for term, keywords in _KEYWORD_TERMS.items():
            if _contains_any_positive_keyword(evidence.text, keywords):
                unsafe_by_term[term].append(evidence.id)
        for claim in store.claims:
            term = _CLAIM_LABEL_TERMS.get(claim.label)
            if term is None:
                continue
            if evidence.id in _claim_evidence_ids([claim]):
                unsafe_by_term[term].append(evidence.id)

    evidence_by_id = {evidence.id: evidence for evidence in store.evidence}
    statuses = []
    for term in EvidenceAuditTerm:
        safe_ids = _dedupe(evidence_by_term[term])
        unsafe_ids = _dedupe(unsafe_by_term[term])
        if safe_ids:
            status = _coverage_status(safe_ids, evidence_by_id)
            explanation = _coverage_explanation(term, status, safe_ids, evidence_by_id)
            statuses.append(
                EvidenceAuditTermStatus(
                    term=term,
                    label=_TERM_LABELS[term],
                    status=status,
                    explanation=explanation,
                    evidence_ids=safe_ids,
                    missing_evidence=False,
                )
            )
            continue
        if unsafe_ids:
            statuses.append(
                EvidenceAuditTermStatus(
                    term=term,
                    label=_TERM_LABELS[term],
                    status=EvidenceAuditCoverageStatus.UNSAFE,
                    explanation=(
                        f"The only detected {_TERM_LABELS[term]} support appears in "
                        "source text that looks like instructions, so it cannot be used."
                    ),
                    evidence_ids=unsafe_ids,
                    missing_evidence=False,
                )
            )
            continue
        statuses.append(
            EvidenceAuditTermStatus(
                term=term,
                label=_TERM_LABELS[term],
                status=EvidenceAuditCoverageStatus.MISSING,
                explanation=f"No usable evidence was found for {_TERM_LABELS[term]}.",
                missing_evidence=True,
            )
        )
    return statuses


def _coverage_status(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceRecord],
) -> EvidenceAuditCoverageStatus:
    freshnesses = {
        evidence_by_id[evidence_id].source_freshness
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
    }
    if SourceFreshness.CURRENT in freshnesses:
        return EvidenceAuditCoverageStatus.SUPPORTED
    if freshnesses == {SourceFreshness.STALE}:
        return EvidenceAuditCoverageStatus.STALE
    return EvidenceAuditCoverageStatus.WEAK


def _coverage_explanation(
    term: EvidenceAuditTerm,
    status: EvidenceAuditCoverageStatus,
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceRecord],
) -> str:
    label = _TERM_LABELS[term]
    if status == EvidenceAuditCoverageStatus.SUPPORTED:
        return f"Usable current evidence supports {label}."
    if status == EvidenceAuditCoverageStatus.STALE:
        source_kinds = _source_kinds(evidence_ids, evidence_by_id)
        return (
            f"{label.capitalize()} support appears only in stale "
            f"{_join_plain(source_kinds)} evidence."
        )
    return f"{label.capitalize()} support has unknown or weak freshness."


def _empty_store_findings(store: EvidenceStore) -> list[EvidenceAuditFinding]:
    if store.evidence:
        return []
    return [
        EvidenceAuditFinding(
            id="finding_empty_store",
            kind=EvidenceAuditFindingKind.EMPTY_STORE,
            severity=EvidenceAuditSeverity.BLOCKING,
            title="No usable evidence",
            explanation=(
                "The evidence store has no records, so no investment decision can be "
                "supported by source evidence."
            ),
            missing_evidence=True,
        )
    ]


def _unsafe_source_findings(
    store: EvidenceStore,
    unsafe_evidence_ids: set[str],
) -> list[EvidenceAuditFinding]:
    if not unsafe_evidence_ids:
        return []
    cited_ids = {
        citation.evidence_id
        for claim in store.claims
        for citation in claim.citations
    }
    cited_unsafe_ids = sorted(unsafe_evidence_ids & cited_ids)
    uncited_unsafe_ids = sorted(unsafe_evidence_ids - cited_ids)
    findings = []
    if cited_unsafe_ids:
        findings.append(
            EvidenceAuditFinding(
                id="finding_unsafe_cited_source_text",
                kind=EvidenceAuditFindingKind.UNSAFE_SOURCE,
                severity=EvidenceAuditSeverity.BLOCKING,
                title="Cited source text contains instructions",
                explanation=(
                    "One or more cited evidence records contain text that looks like "
                    "instructions from a source document, not diligence evidence."
                ),
                evidence_ids=cited_unsafe_ids,
                source_kinds=_source_kinds(cited_unsafe_ids, _evidence_by_id(store.evidence)),
            )
        )
    if uncited_unsafe_ids:
        findings.append(
            EvidenceAuditFinding(
                id="finding_unsafe_uncited_source_text",
                kind=EvidenceAuditFindingKind.UNSAFE_SOURCE,
                severity=EvidenceAuditSeverity.WARNING,
                title="Source text contains instructions",
                explanation=(
                    "One or more evidence records contain text that looks like source "
                    "document instructions. The audit does not use those records as support."
                ),
                evidence_ids=uncited_unsafe_ids,
                source_kinds=_source_kinds(uncited_unsafe_ids, _evidence_by_id(store.evidence)),
            )
        )
    return findings


def _missing_term_findings(
    term_statuses: list[EvidenceAuditTermStatus],
) -> list[EvidenceAuditFinding]:
    findings = []
    for status in term_statuses:
        if status.status == EvidenceAuditCoverageStatus.SUPPORTED:
            continue
        severity = (
            EvidenceAuditSeverity.BLOCKING
            if status.term in _BLOCKING_MISSING_TERMS
            else EvidenceAuditSeverity.WARNING
        )
        title_prefix = {
            EvidenceAuditCoverageStatus.MISSING: "Missing",
            EvidenceAuditCoverageStatus.STALE: "Stale",
            EvidenceAuditCoverageStatus.WEAK: "Weak",
            EvidenceAuditCoverageStatus.UNSAFE: "Unsafe",
            EvidenceAuditCoverageStatus.SUPPORTED: "Supported",
        }[status.status]
        findings.append(
            EvidenceAuditFinding(
                id=f"finding_term_{status.term.value}",
                kind=EvidenceAuditFindingKind.MISSING_TERM,
                severity=severity,
                title=f"{title_prefix} {status.label}",
                explanation=status.explanation,
                term=status.term,
                evidence_ids=status.evidence_ids,
                missing_evidence=status.missing_evidence,
            )
        )
    return findings


def _unsupported_claim_findings(
    store: EvidenceStore,
    evidence_by_id: dict[str, EvidenceRecord],
    unsafe_evidence_ids: set[str],
) -> list[EvidenceAuditFinding]:
    findings = []
    for claim in store.claims:
        if not _is_material_claim(claim):
            continue
        evidence_ids = _claim_evidence_ids([claim])
        if not claim.citations:
            findings.append(
                _claim_finding(
                    claim,
                    kind=EvidenceAuditFindingKind.UNSUPPORTED_CLAIM,
                    severity=EvidenceAuditSeverity.WARNING,
                    title=f"Claim has no citation: {claim.label}",
                    explanation=(
                        f"The material claim for {claim.label} has no citation, so it "
                        "cannot support an investment decision."
                    ),
                    missing_evidence=True,
                )
            )
            continue
        invalid_statuses = [
            verify_citation(citation, evidence_by_id)
            for citation in claim.citations
            if verify_citation(citation, evidence_by_id) != VerificationStatus.VERIFIED
        ]
        if invalid_statuses:
            findings.append(
                _claim_finding(
                    claim,
                    kind=EvidenceAuditFindingKind.UNSUPPORTED_CLAIM,
                    severity=EvidenceAuditSeverity.WARNING,
                    title=f"Claim citation is invalid: {claim.label}",
                    explanation=(
                        f"The material claim for {claim.label} has a citation that no "
                        "longer matches the stored evidence text."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
            continue
        if any(evidence_id in unsafe_evidence_ids for evidence_id in evidence_ids):
            findings.append(
                _claim_finding(
                    claim,
                    kind=EvidenceAuditFindingKind.UNSUPPORTED_CLAIM,
                    severity=EvidenceAuditSeverity.BLOCKING,
                    title=f"Claim cites unsafe source text: {claim.label}",
                    explanation=(
                        f"The material claim for {claim.label} cites source text that "
                        "looks like instructions, so the claim cannot be used as support."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
            continue
        if claim.quality.confidence < LOW_CLAIM_CONFIDENCE_THRESHOLD:
            findings.append(
                _claim_finding(
                    claim,
                    kind=EvidenceAuditFindingKind.WEAK_SUPPORT,
                    severity=EvidenceAuditSeverity.WARNING,
                    title=f"Claim support is weak: {claim.label}",
                    explanation=(
                        f"The material claim for {claim.label} has low extraction "
                        "confidence and needs review."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        cited_records = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        ]
        if cited_records and all(
            record.source_freshness in {SourceFreshness.STALE, SourceFreshness.UNKNOWN}
            for record in cited_records
        ):
            findings.append(
                _claim_finding(
                    claim,
                    kind=EvidenceAuditFindingKind.WEAK_SUPPORT,
                    severity=EvidenceAuditSeverity.WARNING,
                    title=f"Claim support is not current: {claim.label}",
                    explanation=(
                        f"The material claim for {claim.label} is supported only by "
                        "stale or undated evidence."
                    ),
                    evidence_ids=evidence_ids,
                )
            )
    return findings


def _freshness_findings(store: EvidenceStore) -> list[EvidenceAuditFinding]:
    evidence_by_id = _evidence_by_id(store.evidence)
    findings = []
    for freshness, label in (
        (SourceFreshness.STALE, "stale"),
        (SourceFreshness.UNKNOWN, "undated"),
    ):
        grouped: dict[str, list[str]] = {}
        for evidence in store.evidence:
            if evidence.source_freshness != freshness:
                continue
            grouped.setdefault(str(evidence.source_kind), []).append(evidence.id)
        for source_kind, evidence_ids in sorted(grouped.items()):
            title = f"{label.capitalize()} {source_kind} evidence"
            explanation = (
                f"{len(evidence_ids)} {source_kind} evidence record"
                f"{'' if len(evidence_ids) == 1 else 's'} are {label}. "
                "Find a newer source or confirm that the source is still accurate."
            )
            findings.append(
                EvidenceAuditFinding(
                    id=f"finding_{freshness.value}_{_slug(source_kind)}",
                    kind=EvidenceAuditFindingKind.STALE_EVIDENCE,
                    severity=EvidenceAuditSeverity.WARNING,
                    title=title,
                    explanation=explanation,
                    evidence_ids=evidence_ids,
                    source_kinds=_source_kinds(evidence_ids, evidence_by_id),
                )
            )
    return findings


def _conflict_findings(store: EvidenceStore) -> list[EvidenceAuditFinding]:
    claim_by_id = {claim.id: claim for claim in store.claims}
    evidence_by_id = _evidence_by_id(store.evidence)
    findings = []
    for conflict in validated_conflicts(store):
        evidence_ids = _dedupe(
            evidence_id
            for claim_id in conflict.claim_ids
            if claim_id in claim_by_id
            for evidence_id in _claim_evidence_ids([claim_by_id[claim_id]])
        )
        values = [
            claim_by_id[claim_id].value
            for claim_id in conflict.claim_ids
            if claim_id in claim_by_id
        ]
        findings.append(
            EvidenceAuditFinding(
                id=f"finding_conflict_{_slug(conflict.label)}",
                kind=EvidenceAuditFindingKind.CONFLICT,
                severity=EvidenceAuditSeverity.BLOCKING,
                title=f"Conflicting {conflict.label}",
                explanation=(
                    f"Multiple still-valid values were found for {conflict.label}: "
                    f"{_join_plain(_dedupe(values))}. Resolve the original source "
                    "before relying on this term."
                ),
                term=_CLAIM_LABEL_TERMS.get(conflict.label),
                evidence_ids=evidence_ids,
                claim_ids=conflict.claim_ids,
                source_kinds=_source_kinds(evidence_ids, evidence_by_id),
            )
        )
    return findings


def _scoring_support_findings(
    scored_deal: ScoredDeal,
    evidence_by_id: dict[str, EvidenceRecord],
) -> list[EvidenceAuditFinding]:
    findings = []
    for factor in scored_deal.score_factors:
        if factor.support_status not in {
            ScoreSupportStatus.NEEDS_DILIGENCE,
            ScoreSupportStatus.UNVERIFIED,
        }:
            continue
        evidence_ids = [
            evidence_id
            for evidence_id in factor.evidence_ids
            if evidence_id in evidence_by_id
        ]
        findings.append(
            EvidenceAuditFinding(
                id=f"finding_scoring_{_slug(factor.name)}",
                kind=EvidenceAuditFindingKind.WEAK_SUPPORT,
                severity=EvidenceAuditSeverity.WARNING,
                title=f"Scoring support needs diligence: {factor.name}",
                explanation=(
                    factor.explanation
                    if factor.missing_inputs
                    else f"{factor.name} is not fully supported by verified evidence."
                ),
                evidence_ids=evidence_ids,
                missing_evidence=not evidence_ids,
            )
        )
    if scored_deal.net_return.support_status in {
        ScoreSupportStatus.NEEDS_DILIGENCE,
        ScoreSupportStatus.UNVERIFIED,
    } and scored_deal.net_return.missing_inputs:
        evidence_ids = [
            evidence_id
            for evidence_id in scored_deal.net_return.evidence_ids
            if evidence_id in evidence_by_id
        ]
        findings.append(
            EvidenceAuditFinding(
                id="finding_scoring_net_return",
                kind=EvidenceAuditFindingKind.WEAK_SUPPORT,
                severity=EvidenceAuditSeverity.WARNING,
                title="Return math needs diligence",
                explanation=scored_deal.net_return.explanation,
                evidence_ids=evidence_ids,
                missing_evidence=not evidence_ids,
            )
        )
    return findings


def _questions_from_findings(
    term_statuses: list[EvidenceAuditTermStatus],
    findings: list[EvidenceAuditFinding],
    *,
    scored_deal: ScoredDeal | None,
) -> list[EvidenceAuditQuestion]:
    questions: list[EvidenceAuditQuestion] = []
    for finding in findings:
        if finding.kind == EvidenceAuditFindingKind.CONFLICT:
            questions.append(
                EvidenceAuditQuestion(
                    priority=1,
                    question=(
                        "Resolve the conflicting "
                        f"{finding.title.removeprefix('Conflicting ').lower()} "
                        "in the source documents."
                    ),
                    reason=finding.explanation,
                    term=finding.term,
                    evidence_ids=finding.evidence_ids,
                )
            )
        elif finding.kind == EvidenceAuditFindingKind.UNSAFE_SOURCE:
            questions.append(
                EvidenceAuditQuestion(
                    priority=1,
                    question="Replace unsafe source-text support with clean evidence.",
                    reason=finding.explanation,
                    evidence_ids=finding.evidence_ids,
                )
            )
    for status in term_statuses:
        if status.status == EvidenceAuditCoverageStatus.SUPPORTED:
            continue
        questions.append(
            EvidenceAuditQuestion(
                priority=_QUESTION_PRIORITY[status.term],
                question=_term_question(status.term),
                reason=status.explanation,
                term=status.term,
                evidence_ids=status.evidence_ids,
                missing_evidence=status.missing_evidence,
            )
        )
    if scored_deal is not None:
        for question in scored_deal.diligence_questions:
            questions.append(
                EvidenceAuditQuestion(
                    priority=question.priority + 20,
                    question=question.question,
                    reason=question.reason,
                    evidence_ids=question.evidence_ids,
                    missing_evidence=not question.evidence_ids,
                )
            )
    return sorted(_dedupe_questions(questions), key=lambda question: question.priority)


def _readiness(
    term_statuses: list[EvidenceAuditTermStatus],
    findings: list[EvidenceAuditFinding],
) -> EvidenceAuditReadiness:
    if any(finding.severity == EvidenceAuditSeverity.BLOCKING for finding in findings):
        return EvidenceAuditReadiness.INSUFFICIENT
    if any(
        status.term == EvidenceAuditTerm.PRICE_VALUATION
        and status.status != EvidenceAuditCoverageStatus.SUPPORTED
        for status in term_statuses
    ):
        return EvidenceAuditReadiness.INSUFFICIENT
    if any(
        status.status != EvidenceAuditCoverageStatus.SUPPORTED
        for status in term_statuses
    ) or findings:
        return EvidenceAuditReadiness.NEEDS_DILIGENCE
    return EvidenceAuditReadiness.SUFFICIENT


def _claim_finding(
    claim: ClaimRecord,
    *,
    kind: EvidenceAuditFindingKind,
    severity: EvidenceAuditSeverity,
    title: str,
    explanation: str,
    evidence_ids: list[str] | None = None,
    missing_evidence: bool = False,
) -> EvidenceAuditFinding:
    return EvidenceAuditFinding(
        id=f"finding_claim_{_slug(claim.id)}_{kind.value}",
        kind=kind,
        severity=severity,
        title=title,
        explanation=explanation,
        term=_CLAIM_LABEL_TERMS.get(claim.label),
        evidence_ids=evidence_ids or [],
        missing_evidence=missing_evidence,
        claim_ids=[claim.id],
    )


def _is_material_claim(claim: ClaimRecord) -> bool:
    materiality = claim.quality.materiality.casefold().strip()
    return materiality in {"high", "material", "critical"} or claim.label in _CLAIM_LABEL_TERMS


def _claim_citations_are_safe(
    claim: ClaimRecord,
    unsafe_evidence_ids: set[str],
) -> bool:
    return all(citation.evidence_id not in unsafe_evidence_ids for citation in claim.citations)


def _contains_any_positive_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_contains_positive_keyword(text, keyword) for keyword in keywords)


def _contains_positive_keyword(text: str, keyword: str) -> bool:
    flags = 0 if keyword.isupper() else re.IGNORECASE
    for match in re.finditer(_keyword_pattern(keyword), text, flags=flags):
        prefix = text[max(0, match.start() - 90) : match.start()]
        if _NEGATED_KEYWORD_PREFIX.search(prefix):
            continue
        return True
    return False


def _keyword_pattern(keyword: str) -> str:
    escaped_words = [re.escape(part) for part in keyword.split()]
    escaped_phrase = r"\s+".join(escaped_words)
    return rf"(?<![A-Za-z0-9]){escaped_phrase}(?![A-Za-z0-9])"


def _claim_evidence_ids(claims: list[ClaimRecord]) -> list[str]:
    return _dedupe(
        citation.evidence_id
        for claim in claims
        for citation in claim.citations
    )


def _evidence_by_id(evidence: list[EvidenceRecord]) -> dict[str, EvidenceRecord]:
    return {record.id: record for record in evidence}


def _source_kinds(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceRecord],
) -> list[str]:
    return sorted(
        {
            str(evidence_by_id[evidence_id].source_kind)
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        }
    )


def _term_question(term: EvidenceAuditTerm) -> str:
    if term == EvidenceAuditTerm.PRICE_VALUATION:
        return "Confirm the current valuation, valuation cap, or priced-round price."
    if term == EvidenceAuditTerm.ROUND_SIZE:
        return "Confirm the round size so entry price and ownership can be checked."
    if term == EvidenceAuditTerm.SECURITY_TYPE:
        return (
            "Confirm the security type, meaning what legal investment instrument is "
            "being bought."
        )
    if term == EvidenceAuditTerm.MINIMUM_CHECK:
        return (
            "Confirm the minimum check, meaning the smallest investment the platform or "
            "company will accept."
        )
    if term == EvidenceAuditTerm.PLATFORM_FEES_CARRY:
        return (
            "Confirm platform fees and carry, meaning costs and profit share paid to "
            "the investment wrapper."
        )
    if term == EvidenceAuditTerm.DILUTION:
        return (
            "Confirm expected dilution, meaning how much ownership may shrink after "
            "later financings."
        )
    if term == EvidenceAuditTerm.LEAD_INVESTOR:
        return "Confirm whether there is a lead investor and who is setting the round terms."
    if term == EvidenceAuditTerm.TRACTION:
        return "Collect current traction evidence such as usage, retention, or growth."
    if term == EvidenceAuditTerm.REVENUE:
        return "Collect current revenue evidence."
    if term == EvidenceAuditTerm.CUSTOMER_PROOF:
        return "Collect customer proof such as signed customers, pilots, or design partners."
    if term == EvidenceAuditTerm.MARKET:
        return (
            "Collect evidence for market size, competition, and why this market can "
            "support the outcome."
        )
    if term == EvidenceAuditTerm.TEAM:
        return "Collect evidence about the founders and team."
    return "Collect evidence showing how the company will use the investment proceeds."


def _dedupe_findings(
    findings: list[EvidenceAuditFinding],
) -> list[EvidenceAuditFinding]:
    seen: set[str] = set()
    deduped = []
    for finding in findings:
        if finding.id in seen:
            continue
        seen.add(finding.id)
        deduped.append(finding)
    return deduped


def _dedupe_questions(
    questions: list[EvidenceAuditQuestion],
) -> list[EvidenceAuditQuestion]:
    seen: set[tuple[str, EvidenceAuditTerm | None]] = set()
    deduped = []
    for question in questions:
        key = (question.question, question.term)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    return deduped


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _join_plain(values: list[str]) -> str:
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "item"
