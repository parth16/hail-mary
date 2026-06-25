from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Recommendation(StrEnum):
    INVEST = "INVEST"
    PASS = "PASS"


class PMFLevel(StrEnum):
    UNKNOWN = "unknown"
    EARLY = "early"
    DEVELOPING = "developing"


class FundabilityRisk(StrEnum):
    UNKNOWN = "unknown"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class KillGate(BaseModel):
    name: str
    triggered: bool
    reason: str


class ScoreFactor(BaseModel):
    name: str
    score: int
    max_score: int
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list)


class DiligenceQuestion(BaseModel):
    priority: int
    question: str
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class ScoredDeal(BaseModel):
    deal_id: str
    company_name: str
    recommendation: Recommendation
    check_size: int
    total_score: int
    max_score: int = 100
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    one_line_reason: str
    pmf_level: PMFLevel = PMFLevel.UNKNOWN
    fundability_risk: FundabilityRisk = FundabilityRisk.UNKNOWN
    kill_gates: list[KillGate] = Field(default_factory=list)
    score_factors: list[ScoreFactor] = Field(default_factory=list)
    diligence_questions: list[DiligenceQuestion] = Field(default_factory=list)
    capital_remaining_before: int | None = None
    capital_remaining_after: int | None = None
    memo_path: Path | None = None
    portfolio_rank: int | None = None

    @property
    def triggered_kill_gates(self) -> list[KillGate]:
        return [gate for gate in self.kill_gates if gate.triggered]


class MemoRunSummary(BaseModel):
    report_dir: Path
    scored_deals: list[ScoredDeal] = Field(default_factory=list)
    portfolio_report_path: Path | None = None

    @property
    def deal_count(self) -> int:
        return len(self.scored_deals)
