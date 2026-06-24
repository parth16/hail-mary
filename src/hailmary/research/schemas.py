from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from hailmary.schemas.documents import SourceKind


class ResearchProviderCategory(StrEnum):
    FREE_PUBLIC = "free_public"
    PAID_OPTIONAL = "paid_optional"
    AUTHENTICATED_PORTAL = "authenticated_portal"


class ResearchAccessMode(StrEnum):
    WEB_PAGE = "web_page"
    API = "api"
    MANUAL_PORTAL = "manual_portal"


class ResearchTaskStatus(StrEnum):
    PLANNED = "planned"
    NEEDS_OPERATOR = "needs_operator"


class ResearchProvider(BaseModel):
    id: str
    name: str
    category: ResearchProviderCategory
    access_mode: ResearchAccessMode
    source_kind: SourceKind
    description: str
    default_enabled: bool = True
    requires_api_key: bool = False
    requires_authenticated_session: bool = False
    licensing_notes: str
    operator_note: str


class ResearchDealInput(BaseModel):
    deal_id: str
    company_name: str
    website_url: str | None = None


class ResearchTask(BaseModel):
    id: str
    deal_id: str
    company_name: str
    provider_id: str
    provider_name: str
    provider_category: ResearchProviderCategory
    source_kind: SourceKind
    status: ResearchTaskStatus
    query: str
    url: str | None = None
    created_at: datetime
    confidence: str = "not_collected"
    licensing_notes: str
    evidence_policy: str
    operator_note: str


class ResearchPlan(BaseModel):
    version: str = "1"
    created_at: datetime
    local_only: bool
    web_research_enabled: bool
    include_paid: bool = False
    deals: list[ResearchDealInput] = Field(default_factory=list)
    providers: list[ResearchProvider] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


class ResearchPlanRunSummary(BaseModel):
    output_path: Path
    plan: ResearchPlan

    @property
    def deal_count(self) -> int:
        return len(self.plan.deals)

    @property
    def task_count(self) -> int:
        return self.plan.task_count
