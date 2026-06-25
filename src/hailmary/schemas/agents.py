from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hailmary.config import CHECK_SIZE_TIERS
from hailmary.schemas.documents import DocumentType, SourceKind
from hailmary.schemas.evidence import ClaimType, EvidenceKind, SourceFreshness
from hailmary.schemas.scoring import (
    ConfidenceLevel,
    FundabilityRisk,
    PMFLevel,
    Recommendation,
)


class AgentRole(StrEnum):
    PRODUCT_CUSTOMER_TRACTION = "product_customer_traction"
    MARKET_COMPETITION = "market_competition"
    TEAM_EXECUTION = "team_execution"
    FINANCING_NEXT_ROUND_RISK = "financing_next_round_risk"
    EXTRACTION = "extraction"
    STAGE_NORMALIZER = "stage_normalizer"
    DEAL_TERMS = "deal_terms"
    TEAM = "team"
    MARKET = "market"
    PRODUCT_TECHNICAL = "product_technical"
    PRODUCT_MARKET_FIT = "product_market_fit"
    CUSTOMER_SALES = "customer_sales"
    COMPETITION = "competition"
    BUSINESS_MODEL = "business_model"
    RETURN_MATH = "return_math"
    LEGAL_FUND_WRAPPER = "legal_fund_wrapper"
    REGULATORY_ETHICS = "regulatory_ethics"
    FUNDABILITY = "fundability"
    BULL = "bull"
    BEAR = "bear"
    RISKS = "risks"
    PORTFOLIO = "portfolio"
    GROUNDING_AUDITOR = "grounding_auditor"
    FINAL_DECISION = "final_decision"
    OVERALL = "overall"


class AgentEvidenceItem(BaseModel):
    id: str
    text: str
    source_kind: SourceKind
    document_type: DocumentType
    evidence_kind: EvidenceKind
    source_freshness: SourceFreshness
    page_number: int | None = None
    table_index: int | None = None
    ocr_applied: bool = False
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    truncated: bool = False


class AgentClaimItem(BaseModel):
    id: str
    claim_type: ClaimType
    label: str
    value: str
    normalized_value: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentScoreSnapshot(BaseModel):
    recommendation: Recommendation
    check_size: int
    total_score: int
    max_score: int
    confidence: ConfidenceLevel
    one_line_reason: str
    pmf_level: PMFLevel
    fundability_risk: FundabilityRisk


class AgentScoreFactorItem(BaseModel):
    name: str
    score: int
    max_score: int
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentKillGateItem(BaseModel):
    name: str
    triggered: bool
    reason: str


class AgentConflictItem(BaseModel):
    id: str
    claim_type: ClaimType
    label: str
    normalized_values: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentInputPacket(BaseModel):
    created_at: datetime
    deal_id: str
    company_name: str
    agent_role: AgentRole
    allowed_evidence_ids: list[str] = Field(default_factory=list)
    allowed_claim_ids: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    output_schema_name: str = "AgentReviewOutput"
    output_schema: dict[str, Any] = Field(default_factory=dict)
    score: AgentScoreSnapshot
    score_factors: list[AgentScoreFactorItem] = Field(default_factory=list)
    triggered_kill_gates: list[AgentKillGateItem] = Field(default_factory=list)
    conflicts: list[AgentConflictItem] = Field(default_factory=list)
    packet_limitations: list[str] = Field(default_factory=list)
    evidence: list[AgentEvidenceItem] = Field(default_factory=list)
    verified_claims: list[AgentClaimItem] = Field(default_factory=list)
    diligence_questions: list[str] = Field(default_factory=list)


class AgentPacketFile(BaseModel):
    deal_id: str
    company_name: str
    agent_role: AgentRole
    path: Path


class AgentPacketRunSummary(BaseModel):
    output_dir: Path
    packets: list[AgentPacketFile] = Field(default_factory=list)

    @property
    def packet_count(self) -> int:
        return len(self.packets)


class StrictAgentOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentEvidenceReference(StrictAgentOutputModel):
    evidence_id: str
    quote: str | None = None


class AgentFinding(StrictAgentOutputModel):
    title: str
    finding: str
    confidence: ConfidenceLevel
    materiality: str
    score_delta: int = Field(ge=-20, le=20, default=0)
    evidence: list[AgentEvidenceReference] = Field(default_factory=list)
    unsupported: bool = False


class AgentDiligenceQuestion(StrictAgentOutputModel):
    question: str
    reason: str
    evidence: list[AgentEvidenceReference] = Field(default_factory=list)


class AgentSummaryPoint(StrictAgentOutputModel):
    summary: str
    evidence: list[AgentEvidenceReference] = Field(default_factory=list)
    unsupported: bool = False


class AgentRecommendationRationale(StrictAgentOutputModel):
    recommendation: Recommendation
    check_size: int = Field(default=0)
    reason: str
    evidence: list[AgentEvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_size_matches_recommendation(self) -> Self:
        if self.check_size not in CHECK_SIZE_TIERS:
            raise ValueError(
                "check_size must be one of $0, $1K, $2.5K, $5K, $7.5K, or $10K."
            )
        if self.recommendation == Recommendation.PASS and self.check_size != 0:
            raise ValueError("PASS recommendations must use a $0 check size.")
        if self.recommendation == Recommendation.INVEST and self.check_size == 0:
            raise ValueError("INVEST recommendations must use a nonzero check size.")
        return self


class AgentReviewOutput(StrictAgentOutputModel):
    deal_id: str
    company_name: str
    agent_role: AgentRole
    summary: list[AgentSummaryPoint] = Field(default_factory=list)
    findings: list[AgentFinding] = Field(default_factory=list)
    diligence_questions: list[AgentDiligenceQuestion] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: AgentRecommendationRationale | None = None


class AgentValidationIssue(BaseModel):
    location: str
    message: str


class AgentValidationResult(BaseModel):
    issues: list[AgentValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


def is_allowed_check_size(check_size: int) -> bool:
    return check_size in CHECK_SIZE_TIERS
