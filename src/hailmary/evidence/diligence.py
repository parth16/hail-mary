from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.evidence.actions import select_action_context
from hailmary.evidence.audit import EvidenceCompletenessAudit
from hailmary.schemas.agents import AgentReviewOutput
from hailmary.schemas.scoring import ScoredDeal
from hailmary.utils.slug import slugify

QUESTION_QUEUE_VERSION = "1"
ANSWER_LOG_VERSION = "1"
MAX_OPERATOR_ANSWER_CHARS = 2_000
MAX_TRIAGE_QUESTION_IDS = 8
MAX_EMAIL_QUESTION_CHARS = 280


class DiligenceLoopError(RuntimeError):
    """Local diligence-question state could not be read or written safely."""


class DiligenceQuestionSource(StrEnum):
    EVIDENCE_AUDIT = "evidence_completeness_audit"
    RULE_BASED_SCORING = "rule_based_scoring"
    MERIDIAN_WORKFLOW = "meridian_manual_workflow"
    FINAL_REVIEW = "final_review"
    SPECIALIST_REVIEW = "specialist_review"


class DiligenceAnswerStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class DiligenceResolutionPath(StrEnum):
    WEB_RESEARCH = "web_research"
    PAID_DATA_SOURCE = "paid_data_source"
    MERIDIAN_EMAIL = "meridian_email"
    SOURCE_REVIEW = "source_review"


class DiligenceTriageStatus(StrEnum):
    DECISION_BLOCKER = "decision_blocker"
    FOLLOW_UP = "follow_up"


class DiligenceAnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_id: str
    deal_id: str
    question_id: str
    status: DiligenceAnswerStatus
    answer: str = Field(max_length=MAX_OPERATOR_ANSWER_CHARS)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime

    @field_validator("answer_id", "deal_id", "question_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("answer")
    @classmethod
    def _clean_answer(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        if any(character in stripped for character in ("\x00", "\r")):
            raise ValueError("must be plain text")
        return stripped

    @field_validator("evidence_ids")
    @classmethod
    def _clean_evidence_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for evidence_id in value:
            stripped = evidence_id.strip()
            if not stripped:
                raise ValueError("evidence IDs must not be blank")
            if stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class DiligenceAnswerLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = ANSWER_LOG_VERSION
    deal_id: str
    answers: list[DiligenceAnswerRecord] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value != ANSWER_LOG_VERSION:
            raise ValueError("unsupported diligence answer file version")
        return value

    @field_validator("deal_id")
    @classmethod
    def _deal_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @model_validator(mode="after")
    def _validate_answers(self) -> Self:
        seen_answer_ids: set[str] = set()
        for answer in self.answers:
            if answer.deal_id != self.deal_id:
                raise ValueError("answer deal ID does not match the answer file deal ID")
            if answer.answer_id in seen_answer_ids:
                raise ValueError("duplicate answer IDs are not allowed")
            seen_answer_ids.add(answer.answer_id)
        return self


class DiligenceQuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    source: DiligenceQuestionSource
    priority: int = Field(ge=1)
    question: str
    reason: str
    category: str | None = None
    source_role: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False
    answer_status: DiligenceAnswerStatus = DiligenceAnswerStatus.UNRESOLVED
    latest_answer_id: str | None = None
    answered_at: datetime | None = None
    answer_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("question_id", "question", "reason")
    @classmethod
    def _question_fields_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("category", "source_role")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("evidence_ids", "answer_evidence_ids")
    @classmethod
    def _dedupe_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class DiligenceTriageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triage_id: str
    title: str
    status: DiligenceTriageStatus
    resolution_path: DiligenceResolutionPath
    priority: int = Field(ge=1)
    unresolved_question_count: int = Field(ge=0)
    representative_question: str
    question_texts: list[str] = Field(default_factory=list)
    why_it_matters: str
    next_step: str
    question_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "triage_id",
        "title",
        "representative_question",
        "why_it_matters",
        "next_step",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("question_texts", "question_ids", "evidence_ids")
    @classmethod
    def _dedupe_text_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned


class DiligenceEmailDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    body: str
    question_ids: list[str] = Field(default_factory=list)

    @field_validator("subject", "body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class DiligenceTriageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    total_question_count: int = Field(ge=0)
    unresolved_question_count: int = Field(ge=0)
    resolved_question_count: int = Field(ge=0)
    decision_blocker_count: int = Field(ge=0)
    follow_up_count: int = Field(ge=0)
    resolution_counts: dict[DiligenceResolutionPath, int] = Field(default_factory=dict)
    items: list[DiligenceTriageItem] = Field(default_factory=list)
    meridian_email_draft: DiligenceEmailDraft | None = None

    @property
    def top_decision_blockers(self) -> list[DiligenceTriageItem]:
        return [
            item
            for item in self.items
            if item.status == DiligenceTriageStatus.DECISION_BLOCKER
        ]


class DiligenceQuestionQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = QUESTION_QUEUE_VERSION
    deal_id: str
    company_name: str
    created_at: datetime
    questions: list[DiligenceQuestionItem] = Field(default_factory=list)
    triage: DiligenceTriageSummary | None = None

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value != QUESTION_QUEUE_VERSION:
            raise ValueError("unsupported diligence question file version")
        return value

    @field_validator("deal_id", "company_name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @property
    def resolved_count(self) -> int:
        return sum(
            1
            for question in self.questions
            if question.answer_status == DiligenceAnswerStatus.RESOLVED
        )

    @property
    def unresolved_count(self) -> int:
        return sum(
            1
            for question in self.questions
            if question.answer_status == DiligenceAnswerStatus.UNRESOLVED
        )


@dataclass(frozen=True)
class DiligenceQuestionQueueContext:
    queue: DiligenceQuestionQueue
    queue_path: Path
    answer_log: DiligenceAnswerLog
    answer_log_path: Path


@dataclass(frozen=True)
class DiligenceQuestionCandidate:
    source: DiligenceQuestionSource
    priority: int
    question: str
    reason: str
    category: str | None = None
    evidence_ids: list[str] | None = None
    missing_evidence: bool = False


@dataclass(frozen=True)
class DiligenceAnswerWriteResult:
    deal_id: str
    company_name: str
    question_id: str
    status: DiligenceAnswerStatus
    answer_log_path: Path
    question_queue_path: Path


def diligence_question_queue_path(*, config: AppConfig, deal_id: str) -> Path:
    return _state_path(config=config, deal_id=deal_id, folder="diligence-questions")


def diligence_answer_log_path(*, config: AppConfig, deal_id: str) -> Path:
    return _state_path(config=config, deal_id=deal_id, folder="diligence-answers")


def load_diligence_answer_log(
    *,
    config: AppConfig,
    deal_id: str,
) -> DiligenceAnswerLog:
    path = diligence_answer_log_path(config=config, deal_id=deal_id)
    _ensure_private_parent(path.parent, data_dir=config.data_dir, create=False)
    if not path.exists():
        return DiligenceAnswerLog(deal_id=deal_id, answers=[])
    if path.is_symlink():
        raise DiligenceLoopError(
            f"The diligence answer file for deal {deal_id} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    try:
        raw_log = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DiligenceLoopError(
            f"The diligence answer file for deal {deal_id} is not plain text."
        ) from exc
    except OSError as exc:
        raise DiligenceLoopError(
            f"Could not read the diligence answer file for deal {deal_id} at {path}: {exc}"
        ) from exc
    try:
        log = DiligenceAnswerLog.model_validate_json(raw_log)
    except ValidationError as exc:
        raise DiligenceLoopError(
            f"The diligence answer file for deal {deal_id} could not be read. "
            f"First problem: {_validation_error_detail(exc)}"
        ) from exc
    if log.deal_id != deal_id:
        raise DiligenceLoopError(
            f"The diligence answer file at {path} belongs to deal {log.deal_id}, "
            f"not {deal_id}."
        )
    return log


def write_diligence_answer_log(
    *,
    config: AppConfig,
    log: DiligenceAnswerLog,
) -> Path:
    path = diligence_answer_log_path(config=config, deal_id=log.deal_id)
    _write_private_model(path, log, data_dir=config.data_dir, description="diligence answer")
    return path


def load_diligence_question_queue(
    *,
    config: AppConfig,
    deal_id: str,
) -> DiligenceQuestionQueue:
    path = diligence_question_queue_path(config=config, deal_id=deal_id)
    _ensure_private_parent(path.parent, data_dir=config.data_dir, create=False)
    if not path.exists():
        raise DiligenceLoopError(
            "No diligence question queue was found. Run `hailmary evaluate-deal "
            "<company-folder>` before answering questions."
        )
    if path.is_symlink():
        raise DiligenceLoopError(
            f"The diligence question file for deal {deal_id} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    try:
        raw_queue = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DiligenceLoopError(
            f"The diligence question file for deal {deal_id} is not plain text."
        ) from exc
    except OSError as exc:
        raise DiligenceLoopError(
            f"Could not read the diligence question file for deal {deal_id} at {path}: {exc}"
        ) from exc
    try:
        queue = DiligenceQuestionQueue.model_validate_json(raw_queue)
    except ValidationError as exc:
        raise DiligenceLoopError(
            f"The diligence question file for deal {deal_id} could not be read. "
            f"First problem: {_validation_error_detail(exc)}"
        ) from exc
    if queue.deal_id != deal_id:
        raise DiligenceLoopError(
            f"The diligence question file at {path} belongs to deal {queue.deal_id}, "
            f"not {deal_id}."
        )
    return queue


def write_diligence_question_queue(
    *,
    config: AppConfig,
    queue: DiligenceQuestionQueue,
) -> Path:
    path = diligence_question_queue_path(config=config, deal_id=queue.deal_id)
    _write_private_model(path, queue, data_dir=config.data_dir, description="diligence question")
    return path


def build_diligence_question_queue(
    scored_deal: ScoredDeal,
    *,
    evidence_audit: EvidenceCompletenessAudit | None = None,
    extra_questions: list[DiligenceQuestionCandidate] | None = None,
    final_output: AgentReviewOutput | None = None,
    specialist_outputs: list[AgentReviewOutput] | None = None,
    answer_log: DiligenceAnswerLog | None = None,
    created_at: datetime | None = None,
) -> DiligenceQuestionQueue:
    candidates: list[DiligenceQuestionItem] = []
    if evidence_audit is not None:
        for audit_question in evidence_audit.questions:
            candidates.append(
                _question_item(
                    deal_id=scored_deal.deal_id,
                    source=DiligenceQuestionSource.EVIDENCE_AUDIT,
                    priority=audit_question.priority,
                    question=audit_question.question,
                    reason=audit_question.reason,
                    category=(
                        audit_question.term.value
                        if audit_question.term is not None
                        else None
                    ),
                    evidence_ids=audit_question.evidence_ids,
                    missing_evidence=audit_question.missing_evidence,
                )
            )
    for scoring_question in scored_deal.diligence_questions:
        candidates.append(
            _question_item(
                deal_id=scored_deal.deal_id,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                priority=scoring_question.priority,
                question=scoring_question.question,
                reason=scoring_question.reason,
                category=scoring_question.category.value,
                evidence_ids=scoring_question.evidence_ids,
                missing_evidence=bool(scoring_question.missing_evidence),
            )
        )
    for extra_question in extra_questions or []:
        candidates.append(
            _question_item(
                deal_id=scored_deal.deal_id,
                source=extra_question.source,
                priority=extra_question.priority,
                question=extra_question.question,
                reason=extra_question.reason,
                category=extra_question.category,
                evidence_ids=extra_question.evidence_ids,
                missing_evidence=extra_question.missing_evidence,
            )
        )
    if final_output is not None:
        candidates.extend(
            _agent_question_items(
                scored_deal=scored_deal,
                output=final_output,
                source=DiligenceQuestionSource.FINAL_REVIEW,
                priority_offset=100,
            )
        )
    for output in specialist_outputs or []:
        candidates.extend(
            _agent_question_items(
                scored_deal=scored_deal,
                output=output,
                source=DiligenceQuestionSource.SPECIALIST_REVIEW,
                priority_offset=200,
            )
        )

    questions = _dedupe_questions(candidates)
    queue = DiligenceQuestionQueue(
        deal_id=scored_deal.deal_id,
        company_name=scored_deal.company_name,
        created_at=created_at or datetime.now(UTC),
        questions=questions,
    )
    if answer_log is None:
        return queue.model_copy(update={"triage": build_diligence_triage(queue)})
    return apply_diligence_answers(queue, answer_log)


def apply_diligence_answers(
    queue: DiligenceQuestionQueue,
    answer_log: DiligenceAnswerLog,
) -> DiligenceQuestionQueue:
    latest_answers = effective_diligence_answers(answer_log)
    updated_questions = []
    for question in queue.questions:
        answer = latest_answers.get(question.question_id)
        if answer is None:
            updated_questions.append(question)
            continue
        updated_questions.append(
            question.model_copy(
                update={
                    "answer_status": answer.status,
                    "latest_answer_id": answer.answer_id,
                    "answered_at": answer.created_at,
                    "answer_evidence_ids": answer.evidence_ids,
                }
            )
        )
    updated_queue = queue.model_copy(update={"questions": updated_questions})
    return updated_queue.model_copy(update={"triage": build_diligence_triage(updated_queue)})


def build_diligence_triage(queue: DiligenceQuestionQueue) -> DiligenceTriageSummary:
    unresolved_questions = [
        question
        for question in queue.questions
        if question.answer_status == DiligenceAnswerStatus.UNRESOLVED
    ]
    grouped_questions: dict[str, list[DiligenceQuestionItem]] = {}
    for question in unresolved_questions:
        grouped_questions.setdefault(_triage_theme_key(question), []).append(question)

    items = [
        _triage_item(
            deal_id=queue.deal_id,
            theme_key=theme_key,
            questions=questions,
        )
        for theme_key, questions in grouped_questions.items()
    ]
    items = sorted(
        items,
        key=lambda item: (
            0 if item.status == DiligenceTriageStatus.DECISION_BLOCKER else 1,
            item.priority,
            item.title.casefold(),
        ),
    )
    decision_blocker_count = sum(
        1 for item in items if item.status == DiligenceTriageStatus.DECISION_BLOCKER
    )
    resolution_counts = Counter(item.resolution_path for item in items)
    triage = DiligenceTriageSummary(
        generated_at=queue.created_at,
        total_question_count=len(queue.questions),
        unresolved_question_count=len(unresolved_questions),
        resolved_question_count=queue.resolved_count,
        decision_blocker_count=decision_blocker_count,
        follow_up_count=len(items) - decision_blocker_count,
        resolution_counts={
            path: resolution_counts[path]
            for path in DiligenceResolutionPath
            if resolution_counts[path]
        },
        items=items,
        meridian_email_draft=_meridian_email_draft(
            company_name=queue.company_name,
            items=items,
        ),
    )
    return triage


def effective_diligence_answers(
    answer_log: DiligenceAnswerLog,
) -> dict[str, DiligenceAnswerRecord]:
    answers: dict[str, DiligenceAnswerRecord] = {}
    for answer in answer_log.answers:
        answers[answer.question_id] = answer
    return answers


def select_diligence_question_queue(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
) -> DiligenceQuestionQueueContext:
    try:
        config = validate_local_state(config, update_git_exclude=False)
    except ConfigError as exc:
        raise DiligenceLoopError(
            f"Local generated-data setup failed: {exc}"
        ) from exc
    context = select_action_context(
        config=config,
        deal_id=deal_id,
        company_name=company_name,
    )
    queue = load_diligence_question_queue(config=config, deal_id=context.deal.id)
    answer_log = load_diligence_answer_log(config=config, deal_id=context.deal.id)
    queue = apply_diligence_answers(queue, answer_log)
    return DiligenceQuestionQueueContext(
        queue=queue,
        queue_path=diligence_question_queue_path(config=config, deal_id=context.deal.id),
        answer_log=answer_log,
        answer_log_path=diligence_answer_log_path(config=config, deal_id=context.deal.id),
    )


def record_diligence_answer(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
    question_id: str,
    status: DiligenceAnswerStatus,
    answer: str,
    evidence_ids: list[str] | None = None,
    created_at: datetime | None = None,
) -> DiligenceAnswerWriteResult:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise DiligenceLoopError(
            f"Local generated-data setup failed: {exc}"
        ) from exc
    context = select_action_context(
        config=config,
        deal_id=deal_id,
        company_name=company_name,
    )
    queue = load_diligence_question_queue(config=config, deal_id=context.deal.id)
    known_question_ids = {question.question_id for question in queue.questions}
    clean_question_id = question_id.strip()
    if clean_question_id not in known_question_ids:
        raise DiligenceLoopError(
            f"No diligence question {clean_question_id} was found for {context.deal.company_name}. "
            "Run `hailmary diligence list` to see current question IDs."
        )
    clean_evidence_ids = _validate_answer_evidence_ids(
        evidence_ids or [],
        current_evidence_ids={evidence.id for evidence in context.store.evidence},
        company_name=context.deal.company_name,
    )
    answer_created_at = created_at or datetime.now(UTC)
    answer_record = DiligenceAnswerRecord(
        answer_id=_answer_id(
            deal_id=context.deal.id,
            question_id=clean_question_id,
            status=status,
            answer=answer,
            evidence_ids=clean_evidence_ids,
            created_at=answer_created_at,
        ),
        deal_id=context.deal.id,
        question_id=clean_question_id,
        status=status,
        answer=answer,
        evidence_ids=clean_evidence_ids,
        created_at=answer_created_at,
    )
    log = load_diligence_answer_log(config=config, deal_id=context.deal.id)
    updated_log = DiligenceAnswerLog(
        deal_id=log.deal_id,
        answers=[*log.answers, answer_record],
    )
    answer_log_path = write_diligence_answer_log(config=config, log=updated_log)
    updated_queue = apply_diligence_answers(queue, updated_log)
    question_queue_path = write_diligence_question_queue(
        config=config,
        queue=updated_queue,
    )
    return DiligenceAnswerWriteResult(
        deal_id=context.deal.id,
        company_name=context.deal.company_name,
        question_id=clean_question_id,
        status=status,
        answer_log_path=answer_log_path,
        question_queue_path=question_queue_path,
    )


def diligence_answer_status_counts(
    queue: DiligenceQuestionQueue,
) -> list[tuple[DiligenceAnswerStatus, int]]:
    counts = Counter(question.answer_status for question in queue.questions)
    return [
        (status, counts[status])
        for status in (DiligenceAnswerStatus.RESOLVED, DiligenceAnswerStatus.UNRESOLVED)
        if counts[status]
    ]


def _question_item(
    *,
    deal_id: str,
    source: DiligenceQuestionSource,
    priority: int,
    question: str,
    reason: str,
    category: str | None = None,
    source_role: str | None = None,
    evidence_ids: list[str] | None = None,
    missing_evidence: bool = False,
) -> DiligenceQuestionItem:
    return DiligenceQuestionItem(
        question_id=_question_id(deal_id=deal_id, source=source, question=question),
        source=source,
        priority=max(priority, 1),
        question=question,
        reason=reason,
        category=category,
        source_role=source_role,
        evidence_ids=evidence_ids or [],
        missing_evidence=missing_evidence,
    )


def _agent_question_items(
    *,
    scored_deal: ScoredDeal,
    output: AgentReviewOutput,
    source: DiligenceQuestionSource,
    priority_offset: int,
) -> list[DiligenceQuestionItem]:
    items: list[DiligenceQuestionItem] = []
    for index, question in enumerate(output.diligence_questions, start=1):
        items.append(
            _question_item(
                deal_id=scored_deal.deal_id,
                source=source,
                priority=priority_offset + index,
                question=question.question,
                reason=question.reason,
                source_role=output.agent_role.value,
                evidence_ids=[
                    reference.evidence_id for reference in question.evidence
                ],
                missing_evidence=not question.evidence,
            )
        )
    return items


def _dedupe_questions(
    questions: list[DiligenceQuestionItem],
) -> list[DiligenceQuestionItem]:
    by_id: dict[str, DiligenceQuestionItem] = {}
    for question in questions:
        existing = by_id.get(question.question_id)
        if existing is None:
            by_id[question.question_id] = question
            continue
        merged_ids = [*existing.evidence_ids]
        for evidence_id in question.evidence_ids:
            if evidence_id not in merged_ids:
                merged_ids.append(evidence_id)
        by_id[question.question_id] = existing.model_copy(
            update={
                "priority": min(existing.priority, question.priority),
                "evidence_ids": merged_ids,
                "missing_evidence": existing.missing_evidence
                or question.missing_evidence,
            }
        )
    return sorted(
        by_id.values(),
        key=lambda question: (question.priority, question.question.casefold()),
    )


def _triage_theme_key(question: DiligenceQuestionItem) -> str:
    category_text = (question.category or "").casefold()
    if _contains_any(category_text, ("financing terms",)):
        text_without_category = f"{question.question} {question.reason}".casefold()
        if _contains_any(
            text_without_category,
            (
                "valuation is justified",
                "valuation justified",
                "entry valuation is high",
                "high valuation",
                "valuation support",
                "valuation benchmark",
            ),
        ):
            return "valuation"
        return "deal_terms"
    if category_text in {"team", "use_of_funds"}:
        return "team_and_runway"
    text = f"{category_text} {question.question} {question.reason}".casefold()
    if _contains_any(
        text,
        (
            "ownership",
            "dilution",
            "fee",
            "fees",
            "carry",
            "net return",
            "return-math",
            "return math",
            "exit scenario",
            "price/share",
            "number of shares",
            "fully diluted",
            "pro rata",
            "information rights",
            "investment economics",
            "side-letter",
        ),
    ):
        return "return_math"
    if _contains_any(text, ("valuation", "post-money", "pre-money", "benchmark")):
        return "valuation"
    if _contains_any(
        text,
        (
            "financing terms",
            "security type",
            "investment instrument",
            "minimum check",
            "minimum investment",
            "round size",
            "lead investor",
            "investor signal",
            "growth signal",
            "fundability",
            "next-round",
            "next round",
            "raise the next round",
            "closing",
            "subscribed",
            "deal terms",
        ),
    ):
        return "deal_terms"
    if _contains_any(
        text,
        (
            "external research",
            "public-source",
            "third-party",
            "audited",
            "independent verification",
            "attestation",
            "competitor",
            "polymarket",
            "market share",
        ),
    ):
        return "external_validation"
    if _contains_any(
        text,
        (
            "revenue",
            "customer",
            "active user",
            "mau",
            "wau",
            "dau",
            "paying user",
            "retention",
            "cohort",
            "trading volume",
            "take rate",
            "partnership",
            "liquidity",
            "market-maker",
            "market maker",
            "bid-ask",
            "conversion funnel",
            "gross margin",
        ),
    ):
        return "traction_metrics"
    if _contains_any(
        text,
        (
            "regulatory",
            "cftc",
            "state actions",
            "50-state",
            "50 state",
            "140+ countries",
            "coverage map",
            "compliance",
            "cross-border",
        ),
    ):
        return "regulatory_scope"
    if _contains_any(
        text,
        (
            "org chart",
            "team",
            "founder",
            "founders",
            "founders and team",
            "headcount",
            "hiring",
            "critical hires",
            "burn",
            "runway",
            "use_of_funds",
            "use of funds",
            "use of proceeds",
            "use the investment proceeds",
            "investment proceeds",
        ),
    ):
        return "team_and_runway"
    if _contains_any(
        text,
        (
            "stale",
            "undated",
            "source dates",
            "low-confidence",
            "low confidence",
            "verify",
            "extracted claims",
        ),
    ):
        return "source_review"
    return "other"


def _triage_item(
    *,
    deal_id: str,
    theme_key: str,
    questions: list[DiligenceQuestionItem],
) -> DiligenceTriageItem:
    sorted_questions = sorted(
        questions,
        key=lambda question: (question.priority, question.question.casefold()),
    )
    theme = _theme_metadata(theme_key)
    question_ids = [question.question_id for question in sorted_questions]
    evidence_ids = _dedupe_strings(
        evidence_id
        for question in sorted_questions
        for evidence_id in question.evidence_ids
    )
    priority = min(question.priority for question in sorted_questions)
    status = (
        DiligenceTriageStatus.DECISION_BLOCKER
        if _is_decision_blocker(theme_key, sorted_questions)
        else DiligenceTriageStatus.FOLLOW_UP
    )
    resolution_path = _resolution_path(theme_key, sorted_questions)
    return DiligenceTriageItem(
        triage_id=_triage_id(deal_id=deal_id, theme_key=theme_key),
        title=theme["title"],
        status=status,
        resolution_path=resolution_path,
        priority=priority,
        unresolved_question_count=len(sorted_questions),
        representative_question=sorted_questions[0].question,
        question_texts=_dedupe_strings(question.question for question in sorted_questions),
        why_it_matters=theme["why_it_matters"],
        next_step=_next_step(resolution_path, theme_key),
        question_ids=question_ids[:MAX_TRIAGE_QUESTION_IDS],
        evidence_ids=evidence_ids,
    )


def _theme_metadata(theme_key: str) -> dict[str, str]:
    themes = {
        "return_math": {
            "title": "Return math and ownership",
            "why_it_matters": (
                "Hail Mary cannot size a check without ownership, dilution, fees, "
                "carry, and a believable exit case."
            ),
        },
        "valuation": {
            "title": "Valuation support",
            "why_it_matters": (
                "A high valuation can force PASS unless current traction and market "
                "evidence justify the entry price."
            ),
        },
        "deal_terms": {
            "title": "Investment terms",
            "why_it_matters": (
                "Security type, minimum check, round size, lead investor, and closing "
                "status control whether the deal is investable."
            ),
        },
        "external_validation": {
            "title": "External validation",
            "why_it_matters": (
                "Public and third-party sources reduce reliance on company or platform "
                "claims."
            ),
        },
        "traction_metrics": {
            "title": "Traction and customer metrics",
            "why_it_matters": (
                "Revenue, user activity, retention, trading volume, and liquidity show "
                "whether demand is durable."
            ),
        },
        "regulatory_scope": {
            "title": "Regulatory scope",
            "why_it_matters": (
                "Regulatory limits can block growth even when customer demand looks strong."
            ),
        },
        "team_and_runway": {
            "title": "Team, runway, and use of funds",
            "why_it_matters": (
                "Hiring plan, cash runway, and use of proceeds show whether the team can "
                "execute after the round."
            ),
        },
        "source_review": {
            "title": "Source quality review",
            "why_it_matters": (
                "Old, undated, or low-confidence evidence should not drive an investment "
                "decision until rechecked."
            ),
        },
        "other": {
            "title": "Other diligence gaps",
            "why_it_matters": (
                "These questions may still matter, but they are less clearly tied to the "
                "first decision blocker."
            ),
        },
    }
    return themes.get(theme_key, themes["other"])


def _is_decision_blocker(
    theme_key: str,
    questions: list[DiligenceQuestionItem],
) -> bool:
    if any(question.priority <= 25 for question in questions):
        return True
    return theme_key in {
        "return_math",
        "valuation",
        "deal_terms",
        "external_validation",
    }


def _resolution_path(
    theme_key: str,
    questions: list[DiligenceQuestionItem],
) -> DiligenceResolutionPath:
    if any(
        question.source == DiligenceQuestionSource.MERIDIAN_WORKFLOW
        for question in questions
    ):
        return DiligenceResolutionPath.MERIDIAN_EMAIL
    if theme_key in {"return_math", "deal_terms", "team_and_runway"}:
        return DiligenceResolutionPath.MERIDIAN_EMAIL
    if theme_key in {"valuation", "traction_metrics"}:
        return DiligenceResolutionPath.PAID_DATA_SOURCE
    if theme_key in {"external_validation", "regulatory_scope"}:
        return DiligenceResolutionPath.WEB_RESEARCH
    if theme_key == "source_review":
        return DiligenceResolutionPath.SOURCE_REVIEW
    return DiligenceResolutionPath.WEB_RESEARCH


def _next_step(
    resolution_path: DiligenceResolutionPath,
    theme_key: str,
) -> str:
    if resolution_path == DiligenceResolutionPath.WEB_RESEARCH:
        return (
            "Hail Mary should run public web and public API research, save exact "
            "source-linked results, and rerun evaluate-deal."
        )
    if resolution_path == DiligenceResolutionPath.PAID_DATA_SOURCE:
        return (
            "Hail Mary should use public sources first, then query configured paid "
            "data providers if public sources cannot verify the claim."
        )
    if resolution_path == DiligenceResolutionPath.SOURCE_REVIEW:
        return (
            "Hail Mary should re-check the existing local source records and extraction "
            "quality before treating this as a missing-data blocker."
        )
    if theme_key == "return_math":
        return (
            "Hail Mary should include this in the Meridian email draft because the "
            "answer depends on private vehicle economics."
        )
    return (
        "Hail Mary should include this in the Meridian email draft because the answer "
        "is likely inside the authenticated deal page or held by the platform."
    )


def _meridian_email_draft(
    *,
    company_name: str,
    items: list[DiligenceTriageItem],
) -> DiligenceEmailDraft | None:
    email_items = [
        item
        for item in items
        if item.resolution_path == DiligenceResolutionPath.MERIDIAN_EMAIL
    ]
    if not email_items:
        return None
    clean_company_name = _email_line(company_name, max_chars=120)
    subject = f"Follow-up diligence questions for {clean_company_name}"
    bullets = [
        f"- {item.title}: {_email_line(question_text)}"
        for item in email_items
        for question_text in (item.question_texts or [item.representative_question])
    ]
    body_lines = [
        "Hi AngelList Meridian team,",
        "",
        f"I am reviewing {clean_company_name} and need source-backed answers to a few "
        "diligence questions before I can decide whether to invest or size a check.",
        "",
        *bullets,
        "",
        "If available, please include the relevant source document, page reference, "
        "or platform field for each answer. Short factual answers are enough; I am "
        "not asking for screenshots, raw page exports, cookies, signed links, or "
        "anything that bypasses normal authenticated access.",
        "",
        "Thanks,",
        "Parth",
    ]
    return DiligenceEmailDraft(
        subject=subject,
        body="\n".join(body_lines),
        question_ids=[
            question_id
            for item in email_items
            for question_id in item.question_ids
        ],
    )


def _triage_id(*, deal_id: str, theme_key: str) -> str:
    digest = hashlib.sha256(f"{deal_id}:{theme_key}".encode()).hexdigest()
    return f"dt_{digest[:12]}"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, needle) for needle in needles)


def _contains_phrase(text: str, needle: str) -> bool:
    normalized_needle = needle.casefold().strip()
    if not normalized_needle:
        return False
    escaped = re.escape(normalized_needle).replace(r"\ ", r"\s+")
    if any(character.isalnum() for character in normalized_needle):
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return normalized_needle in text


def _email_line(value: str, *, max_chars: int = MAX_EMAIL_QUESTION_CHARS) -> str:
    line = re.sub(r"(?m)^\s*[-*#>]+\s*", "", value)
    line = re.sub(r"\s+", " ", line).strip()
    line = line.lstrip("-*#> ")
    if len(line) <= max_chars:
        return line
    return f"{line[: max_chars - 1].rstrip()}..."


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        stripped = value.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return cleaned


def _validate_answer_evidence_ids(
    evidence_ids: list[str],
    *,
    current_evidence_ids: set[str],
    company_name: str,
) -> list[str]:
    cleaned: list[str] = []
    for evidence_id in evidence_ids:
        stripped = evidence_id.strip()
        if not stripped:
            raise DiligenceLoopError("Attached evidence IDs cannot be blank.")
        if stripped not in current_evidence_ids:
            raise DiligenceLoopError(
                f"No evidence record {stripped} was found for {company_name}. "
                "Run `hailmary review-evidence` to see current evidence IDs."
            )
        if stripped not in cleaned:
            cleaned.append(stripped)
    return cleaned


def _question_id(
    *,
    deal_id: str,
    source: DiligenceQuestionSource,
    question: str,
) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().casefold())
    digest = hashlib.sha256(f"{deal_id}:{source.value}:{normalized}".encode()).hexdigest()
    return f"dq_{digest[:16]}"


def _answer_id(
    *,
    deal_id: str,
    question_id: str,
    status: DiligenceAnswerStatus,
    answer: str,
    evidence_ids: list[str],
    created_at: datetime,
) -> str:
    payload = (
        f"{deal_id}:{question_id}:{status.value}:{answer.strip()}:"
        f"{','.join(evidence_ids)}:{created_at.isoformat()}"
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"ans_{digest[:16]}"


def _state_path(*, config: AppConfig, deal_id: str, folder: str) -> Path:
    safe_deal_id = slugify(deal_id) or "deal"
    digest = hashlib.sha256(deal_id.encode("utf-8")).hexdigest()[:12]
    return _absolute_path(config.data_dir) / folder / f"{safe_deal_id}-{digest}.json"


def _write_private_model(
    path: Path,
    model: BaseModel,
    *,
    data_dir: Path,
    description: str,
) -> None:
    _ensure_private_parent(path.parent, data_dir=data_dir, create=True)
    if path.is_symlink():
        raise DiligenceLoopError(
            f"Could not write {description} state at {path}: output file is a symlink."
        )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise DiligenceLoopError(
            f"Could not write {description} state at {path}: the text cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise DiligenceLoopError(
            f"Could not write {description} state at {path}: {exc}"
        ) from exc


def _ensure_private_parent(path: Path, *, data_dir: Path, create: bool) -> None:
    data_root = _absolute_path(data_dir)
    if data_root.is_symlink():
        raise DiligenceLoopError(
            f"The local data directory at {data_root} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    if not data_root.exists():
        if create:
            try:
                data_root.mkdir(parents=True, exist_ok=True)
                data_root.chmod(0o700)
            except OSError as exc:
                raise DiligenceLoopError(
                    f"Could not create local data folder at {data_root}: {exc}"
                ) from exc
        else:
            raise DiligenceLoopError(
                "No local generated-data folder was found. Run `hailmary evaluate-deal "
                "<company-folder>` before using diligence answers."
            )
    if not data_root.is_dir():
        raise DiligenceLoopError(
            f"Hail Mary local data at {data_root} is not a folder."
        )
    resolved_root = data_root.resolve(strict=False)
    resolved_path = _absolute_path(path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise DiligenceLoopError(
            f"The diligence state folder {path} is outside the private data directory."
        ) from None
    relative_parts = resolved_path.relative_to(resolved_root).parts
    current = resolved_root
    if current.is_symlink():
        raise DiligenceLoopError(
            f"The local data directory at {data_root} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    for part in relative_parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DiligenceLoopError(
                f"Hail Mary cannot use diligence state at {path} because {current} "
                "is a symlinked parent folder."
            )
    if path.is_symlink():
        raise DiligenceLoopError(
            f"The diligence state folder at {path} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    if path.exists() and not path.is_dir():
        raise DiligenceLoopError(
            f"The diligence state path at {path} is not a folder."
        )
    if not create:
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise DiligenceLoopError(
            f"Could not create diligence state folder at {path}: {exc}"
        ) from exc


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _validation_error_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first_error = errors[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", exc))
    return f"{location}: {message}" if location else message
