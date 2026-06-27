from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from hailmary.schemas.documents import DocumentType, SourceKind

BUILTIN_RESEARCH_PROVIDER_SOURCE_KINDS: dict[str, SourceKind] = {
    "company_website": SourceKind.WEB,
    "sec_form_d": SourceKind.WEB,
    "sam_gov": SourceKind.WEB,
    "usaspending": SourceKind.WEB,
    "sbir": SourceKind.WEB,
    "uspto": SourceKind.WEB,
    "github": SourceKind.WEB,
    "public_web": SourceKind.WEB,
    "meridian": SourceKind.MERIDIAN,
    "crunchbase": SourceKind.WEB,
    "people_data_labs": SourceKind.WEB,
    "newsapi": SourceKind.WEB,
    "similarweb": SourceKind.WEB,
    "sensor_tower": SourceKind.WEB,
    "pitchbook": SourceKind.WEB,
    "cb_insights": SourceKind.WEB,
}


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
    from_ingestion: bool = False


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
    what_to_look_for: list[str] = Field(default_factory=list)
    do_not_copy: list[str] = Field(default_factory=list)
    required_metadata: list[str] = Field(default_factory=list)


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


class ResearchResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _original_row_number: int | None = PrivateAttr(default=None)

    deal_id: str | None = None
    company_name: str | None = None
    provider_id: str
    provider_name: str | None = None
    title: str
    text: str
    retrieved_at: datetime
    source_url: str | None = None
    source_api: str | None = None
    confidence: str
    licensing_notes: str
    source_kind: SourceKind = SourceKind.WEB
    document_type: DocumentType = DocumentType.WEB_PAGE

    @model_validator(mode="before")
    @classmethod
    def default_known_provider_source_kind(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        provider_id = data.get("provider_id")
        if not isinstance(provider_id, str):
            return data
        provider_id = provider_id.strip()
        known_source_kind = BUILTIN_RESEARCH_PROVIDER_SOURCE_KINDS.get(provider_id)
        if known_source_kind is None:
            return data

        raw_source_kind = data.get("source_kind")
        if raw_source_kind is None or (
            isinstance(raw_source_kind, str) and not raw_source_kind.strip()
        ):
            updated_data = dict(data)
            updated_data["source_kind"] = known_source_kind
            return updated_data

        try:
            source_kind = (
                raw_source_kind
                if isinstance(raw_source_kind, SourceKind)
                else SourceKind(str(raw_source_kind).strip())
            )
        except ValueError:
            return data
        if source_kind != known_source_kind:
            raise ValueError(
                f"provider_id {provider_id} must use source_kind "
                f"{known_source_kind.value}."
            )

        updated_data = dict(data)
        updated_data["source_kind"] = source_kind
        return updated_data

    @field_validator(
        "deal_id",
        "company_name",
        "provider_name",
        "source_url",
        "source_api",
        mode="before",
    )
    @classmethod
    def blank_optional_text_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "provider_id",
        "provider_name",
        "title",
        "text",
        "source_url",
        "source_api",
        "confidence",
        "licensing_notes",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("provider_id", "title", "text", "confidence", "licensing_notes")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def require_deal_and_source_reference(self) -> Self:
        if self.deal_id is None and self.company_name is None:
            raise ValueError("Each research result needs a deal_id or company_name.")
        if self.source_url is None and self.source_api is None:
            raise ValueError("Each research result needs a source_url or source_api.")
        return self


class ResearchResultsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _skipped_blank_template_row_count: int = PrivateAttr(default=0)

    results: list[ResearchResultInput]

    @field_validator("results")
    @classmethod
    def require_results(cls, value: list[ResearchResultInput]) -> list[ResearchResultInput]:
        if not value:
            raise ValueError("Research results file must contain at least one result.")
        return value


class ResearchImportDealSummary(BaseModel):
    deal_id: str
    company_name: str
    evidence_store_path: Path
    imported_count: int = 0
    skipped_duplicate_count: int = 0
    stale_count: int = 0


class ResearchImportRunSummary(BaseModel):
    input_path: Path
    imported_at: datetime
    dry_run: bool = False
    skipped_blank_template_row_count: int = 0
    provider_imported_counts: dict[str, int] = Field(default_factory=dict)
    provider_stale_counts: dict[str, int] = Field(default_factory=dict)
    provider_names: dict[str, str] = Field(default_factory=dict)
    deals: list[ResearchImportDealSummary] = Field(default_factory=list)

    @property
    def deal_count(self) -> int:
        return len(self.deals)

    @property
    def imported_count(self) -> int:
        return sum(deal.imported_count for deal in self.deals)

    @property
    def skipped_duplicate_count(self) -> int:
        return sum(deal.skipped_duplicate_count for deal in self.deals)

    @property
    def stale_count(self) -> int:
        return sum(deal.stale_count for deal in self.deals)

    @property
    def updated_store_paths(self) -> list[Path]:
        if self.dry_run:
            return []
        return [
            deal.evidence_store_path
            for deal in self.deals
            if deal.imported_count
        ]


class ResearchResultsTemplateRunSummary(BaseModel):
    output_path: Path
    plan_path: Path
    created_at: datetime
    result_count: int
