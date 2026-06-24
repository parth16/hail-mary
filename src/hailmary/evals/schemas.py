from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EvalCategory(StrEnum):
    EXTRACTION = "extraction"
    CITATION = "citation"
    CONTRADICTION = "contradiction"
    PROMPT_INJECTION = "prompt_injection"
    SCORE_CALIBRATION = "score_calibration"


class EvalCaseResult(BaseModel):
    id: str
    category: EvalCategory
    name: str
    passed: bool
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class EvalCaseMetadata(BaseModel):
    id: str
    category: EvalCategory
    name: str
    description: str


class EvalRunSummary(BaseModel):
    results: list[EvalCaseResult] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return self.total_count - self.passed_count

    @property
    def passed(self) -> bool:
        return self.failed_count == 0

    @property
    def failed_results(self) -> list[EvalCaseResult]:
        return [result for result in self.results if not result.passed]
