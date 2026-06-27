from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.evidence.store import refresh_existing_claim_conflicts
from hailmary.schemas.documents import IngestedDeal, IngestionSummary
from hailmary.schemas.evidence import EvidenceStore
from hailmary.utils.slug import slugify

MAX_OPERATOR_NOTE_CHARS = 1_000
ACTION_LOG_VERSION = "1"


class EvidenceActionError(RuntimeError):
    """Evidence action state could not be read or written safely."""


class EvidenceActionTarget(StrEnum):
    EVIDENCE = "evidence"
    CLAIM = "claim"


class EvidenceActionStatus(StrEnum):
    USABLE = "usable"
    APPROVED = "approved"
    EXCLUDED = "excluded"
    NEEDS_REVIEW = "needs_review"


class EvidenceActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    deal_id: str
    target_type: EvidenceActionTarget
    target_id: str
    status: EvidenceActionStatus
    created_at: datetime
    operator_note: str | None = Field(default=None, max_length=MAX_OPERATOR_NOTE_CHARS)

    @field_validator("action_id", "deal_id", "target_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("operator_note")
    @classmethod
    def _clean_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if any(character in stripped for character in ("\x00", "\r")):
            raise ValueError("must be plain text")
        return stripped


class EvidenceActionLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = ACTION_LOG_VERSION
    deal_id: str
    actions: list[EvidenceActionRecord] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value != ACTION_LOG_VERSION:
            raise ValueError("unsupported evidence action file version")
        return value

    @field_validator("deal_id")
    @classmethod
    def _deal_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @model_validator(mode="after")
    def _validate_actions(self) -> Self:
        seen_action_ids: set[str] = set()
        for action in self.actions:
            if action.deal_id != self.deal_id:
                raise ValueError("action deal ID does not match the action file deal ID")
            if action.action_id in seen_action_ids:
                raise ValueError("duplicate action IDs are not allowed")
            seen_action_ids.add(action.action_id)
        return self


@dataclass(frozen=True)
class EvidenceActionState:
    action_id: str
    target_type: EvidenceActionTarget
    target_id: str
    status: EvidenceActionStatus
    created_at: datetime
    operator_note: str | None = None


@dataclass(frozen=True)
class EvidenceActionStatusCount:
    status: EvidenceActionStatus
    count: int


@dataclass(frozen=True)
class EvidenceActionSummary:
    action_file_path: Path
    action_count: int
    valid_action_count: int
    stale_action_count: int
    status_counts: list[EvidenceActionStatusCount]
    evidence_states: dict[str, EvidenceActionState]
    claim_states: dict[str, EvidenceActionState]
    stale_states: list[EvidenceActionState]

    @property
    def excluded_evidence_count(self) -> int:
        return sum(
            1
            for state in self.evidence_states.values()
            if state.status == EvidenceActionStatus.EXCLUDED
        )

    @property
    def excluded_claim_count(self) -> int:
        return sum(
            1
            for state in self.claim_states.values()
            if state.status == EvidenceActionStatus.EXCLUDED
        )

    @property
    def needs_review_count(self) -> int:
        return sum(
            1
            for state in [*self.evidence_states.values(), *self.claim_states.values()]
            if state.status == EvidenceActionStatus.NEEDS_REVIEW
        )


@dataclass(frozen=True)
class EvidenceActionApplication:
    store: EvidenceStore
    summary: EvidenceActionSummary
    excluded_evidence_ids: set[str]
    excluded_claim_ids: set[str]
    packet_quote_only_evidence_ids: set[str]


@dataclass(frozen=True)
class EvidenceActionContext:
    data_dir: Path
    summary_path: Path
    deal: IngestedDeal
    store: EvidenceStore
    evidence_store_path: Path


@dataclass(frozen=True)
class EvidenceActionWriteResult:
    deal_id: str
    company_name: str
    target_type: EvidenceActionTarget
    target_id: str
    status: EvidenceActionStatus
    action_file_path: Path


@dataclass(frozen=True)
class EvidenceActionPruneResult:
    deal_id: str
    company_name: str
    removed_action_count: int
    removed_target_count: int
    action_file_path: Path


def action_log_path(*, config: AppConfig, deal_id: str) -> Path:
    data_dir = _absolute_path(config.data_dir)
    action_root = data_dir / "evidence-actions"
    safe_deal_id = slugify(deal_id) or "deal"
    digest = hashlib.sha256(deal_id.encode("utf-8")).hexdigest()[:12]
    return action_root / f"{safe_deal_id}-{digest}.json"


def load_action_log(*, config: AppConfig, deal_id: str) -> EvidenceActionLog:
    path = action_log_path(config=config, deal_id=deal_id)
    _ensure_private_action_parent(path.parent, data_dir=config.data_dir, create=False)
    if not path.exists():
        return EvidenceActionLog(deal_id=deal_id, actions=[])
    if path.is_symlink():
        raise EvidenceActionError(
            f"The evidence action file for deal {deal_id} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    try:
        raw_log = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceActionError(
            f"The evidence action file for deal {deal_id} is not plain text."
        ) from exc
    except OSError as exc:
        raise EvidenceActionError(
            f"Could not read the evidence action file for deal {deal_id} at {path}: {exc}"
        ) from exc
    try:
        log = EvidenceActionLog.model_validate_json(raw_log)
    except ValidationError as exc:
        detail = _validation_error_detail(exc)
        raise EvidenceActionError(
            f"The evidence action file for deal {deal_id} could not be read. "
            f"First problem: {detail}"
        ) from exc
    if log.deal_id != deal_id:
        raise EvidenceActionError(
            f"The evidence action file at {path} belongs to deal {log.deal_id}, not "
            f"{deal_id}."
        )
    return log


def write_action_log(*, config: AppConfig, log: EvidenceActionLog) -> Path:
    path = action_log_path(config=config, deal_id=log.deal_id)
    _ensure_private_action_parent(path.parent, data_dir=config.data_dir, create=True)
    if path.is_symlink():
        raise EvidenceActionError(
            f"Could not write evidence actions at {path}: output file is a symlink."
        )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(log.model_dump_json(indent=2))
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise EvidenceActionError(
            f"Could not write evidence actions at {path}: the text cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise EvidenceActionError(f"Could not write evidence actions at {path}: {exc}") from exc
    return path


def select_action_context(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
) -> EvidenceActionContext:
    _validate_selector(deal_id=deal_id, company_name=company_name)
    configured_data_dir = _absolute_path(config.data_dir)
    _ensure_local_data(configured_data_dir)
    data_dir = configured_data_dir.resolve(strict=False)
    summary_path = data_dir / "processed" / "ingestion_summary.json"
    if not summary_path.exists():
        raise EvidenceActionError(
            "No generated evidence was found. Run `hailmary evaluate-deal <company-folder>` "
            "before recording evidence actions."
        )
    summary = _load_ingestion_summary(summary_path)
    deal = _select_one_deal(summary.deals, deal_id=deal_id, company_name=company_name)
    evidence_store_path = _evidence_store_path_for_deal(
        deal,
        data_dir=data_dir,
        summary_path=summary_path,
    )
    store = _load_evidence_store(evidence_store_path, company_name=deal.company_name)
    return EvidenceActionContext(
        data_dir=data_dir,
        summary_path=summary_path,
        deal=deal,
        store=store,
        evidence_store_path=evidence_store_path,
    )


def record_evidence_action(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
    evidence_id: str | None = None,
    claim_id: str | None = None,
    status: EvidenceActionStatus | None,
    note: str | None = None,
    created_at: datetime | None = None,
) -> EvidenceActionWriteResult:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise EvidenceActionError(
            f"Local generated-data setup failed: {exc}"
        ) from exc
    context = select_action_context(
        config=config,
        deal_id=deal_id,
        company_name=company_name,
    )
    target_type, target_id = _validated_target(
        context.store,
        company_name=context.deal.company_name,
        evidence_id=evidence_id,
        claim_id=claim_id,
    )
    clean_note = _normalize_note(note, require_note=status is None)
    log = load_action_log(config=config, deal_id=context.deal.id)
    if status is None:
        current_state = _effective_actions(log).get((target_type, target_id))
        status = (
            current_state.status
            if current_state is not None
            else EvidenceActionStatus.USABLE
        )
    action_created_at = created_at or datetime.now(UTC)
    action = EvidenceActionRecord(
        action_id=_action_id(
            deal_id=context.deal.id,
            target_type=target_type,
            target_id=target_id,
            status=status,
            note=clean_note,
            created_at=action_created_at,
        ),
        deal_id=context.deal.id,
        target_type=target_type,
        target_id=target_id,
        status=status,
        created_at=action_created_at,
        operator_note=clean_note,
    )
    updated_log = EvidenceActionLog(deal_id=log.deal_id, actions=[*log.actions, action])
    path = write_action_log(config=config, log=updated_log)
    return EvidenceActionWriteResult(
        deal_id=context.deal.id,
        company_name=context.deal.company_name,
        target_type=target_type,
        target_id=target_id,
        status=status,
        action_file_path=path,
    )


def prune_stale_evidence_actions(
    *,
    config: AppConfig,
    deal_id: str | None = None,
    company_name: str | None = None,
) -> EvidenceActionPruneResult:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise EvidenceActionError(
            f"Local generated-data setup failed: {exc}"
        ) from exc
    context = select_action_context(
        config=config,
        deal_id=deal_id,
        company_name=company_name,
    )
    log = load_action_log(config=config, deal_id=context.deal.id)
    summary = summarize_evidence_actions(config=config, store=context.store)
    stale_targets = {
        (state.target_type, state.target_id) for state in summary.stale_states
    }
    if not stale_targets:
        return EvidenceActionPruneResult(
            deal_id=context.deal.id,
            company_name=context.deal.company_name,
            removed_action_count=0,
            removed_target_count=0,
            action_file_path=summary.action_file_path,
        )

    kept_actions = [
        action
        for action in log.actions
        if (action.target_type, action.target_id) not in stale_targets
    ]
    path = write_action_log(
        config=config,
        log=EvidenceActionLog(deal_id=log.deal_id, actions=kept_actions),
    )
    return EvidenceActionPruneResult(
        deal_id=context.deal.id,
        company_name=context.deal.company_name,
        removed_action_count=len(log.actions) - len(kept_actions),
        removed_target_count=len(stale_targets),
        action_file_path=path,
    )


def summarize_evidence_actions(
    *,
    config: AppConfig,
    store: EvidenceStore,
) -> EvidenceActionSummary:
    path = action_log_path(config=config, deal_id=store.deal_id)
    log = load_action_log(config=config, deal_id=store.deal_id)
    effective = _effective_actions(log)
    evidence_ids = {evidence.id for evidence in store.evidence}
    claim_ids = {claim.id for claim in store.claims}
    evidence_states: dict[str, EvidenceActionState] = {}
    claim_states: dict[str, EvidenceActionState] = {}
    stale_states: list[EvidenceActionState] = []
    for (target_type, target_id), state in effective.items():
        if target_type == EvidenceActionTarget.EVIDENCE and target_id in evidence_ids:
            evidence_states[target_id] = state
        elif target_type == EvidenceActionTarget.CLAIM and target_id in claim_ids:
            claim_states[target_id] = state
        else:
            stale_states.append(state)
    counts = Counter(
        state.status for state in [*evidence_states.values(), *claim_states.values()]
    )
    status_counts = [
        EvidenceActionStatusCount(status=status, count=counts[status])
        for status in sorted(counts, key=lambda value: value.value)
    ]
    return EvidenceActionSummary(
        action_file_path=path,
        action_count=len(log.actions),
        valid_action_count=len(evidence_states) + len(claim_states),
        stale_action_count=len(stale_states),
        status_counts=status_counts,
        evidence_states=evidence_states,
        claim_states=claim_states,
        stale_states=sorted(
            stale_states,
            key=lambda state: (state.target_type.value, state.target_id),
        ),
    )


def apply_evidence_actions(
    *,
    config: AppConfig,
    store: EvidenceStore,
) -> EvidenceActionApplication:
    summary = summarize_evidence_actions(config=config, store=store)
    excluded_evidence_ids = {
        evidence_id
        for evidence_id, state in summary.evidence_states.items()
        if state.status == EvidenceActionStatus.EXCLUDED
    }
    excluded_claim_ids = {
        claim_id
        for claim_id, state in summary.claim_states.items()
        if state.status == EvidenceActionStatus.EXCLUDED
    }
    store_evidence_ids = {evidence.id for evidence in store.evidence}
    packet_quote_only_evidence_ids = {
        citation.evidence_id
        for claim in store.claims
        if claim.id in excluded_claim_ids
        for citation in claim.citations
        if citation.evidence_id in store_evidence_ids
    }
    if not excluded_evidence_ids and not excluded_claim_ids:
        return EvidenceActionApplication(
            store=store,
            summary=summary,
            excluded_evidence_ids=set(),
            excluded_claim_ids=set(),
            packet_quote_only_evidence_ids=set(),
        )

    remaining_evidence = [
        evidence for evidence in store.evidence if evidence.id not in excluded_evidence_ids
    ]
    remaining_claims = [
        claim
        for claim in store.claims
        if claim.id not in excluded_claim_ids
        and not any(
            citation.evidence_id in excluded_evidence_ids for citation in claim.citations
        )
    ]
    filtered_store = refresh_existing_claim_conflicts(
        store.model_copy(
            update={
                "evidence": remaining_evidence,
                "claims": remaining_claims,
            }
        )
    )
    return EvidenceActionApplication(
        store=filtered_store,
        summary=summary,
        excluded_evidence_ids=excluded_evidence_ids,
        excluded_claim_ids=excluded_claim_ids,
        packet_quote_only_evidence_ids=packet_quote_only_evidence_ids
        - excluded_evidence_ids,
    )


def _effective_actions(
    log: EvidenceActionLog,
) -> dict[tuple[EvidenceActionTarget, str], EvidenceActionState]:
    states: dict[tuple[EvidenceActionTarget, str], EvidenceActionState] = {}
    for action in log.actions:
        states[(action.target_type, action.target_id)] = EvidenceActionState(
            action_id=action.action_id,
            target_type=action.target_type,
            target_id=action.target_id,
            status=action.status,
            created_at=action.created_at,
            operator_note=action.operator_note,
        )
    return states


def _validated_target(
    store: EvidenceStore,
    *,
    company_name: str,
    evidence_id: str | None,
    claim_id: str | None,
) -> tuple[EvidenceActionTarget, str]:
    evidence_id = evidence_id.strip() if evidence_id is not None else None
    claim_id = claim_id.strip() if claim_id is not None else None
    if bool(evidence_id) == bool(claim_id):
        raise EvidenceActionError("Choose exactly one target: --evidence-id or --claim-id.")
    if evidence_id is not None:
        if not any(evidence.id == evidence_id for evidence in store.evidence):
            raise EvidenceActionError(
                f"No evidence record {evidence_id} was found for {company_name}. "
                "Check the current evidence ID with `hailmary review-evidence`."
            )
        return EvidenceActionTarget.EVIDENCE, evidence_id
    if claim_id is not None:
        if not any(claim.id == claim_id for claim in store.claims):
            raise EvidenceActionError(
                f"No claim {claim_id} was found for {company_name}. "
                "Check the current claim ID with `hailmary review-evidence`."
            )
        return EvidenceActionTarget.CLAIM, claim_id
    raise EvidenceActionError("Choose exactly one target: --evidence-id or --claim-id.")


def _normalize_note(note: str | None, *, require_note: bool = False) -> str | None:
    if note is None:
        if require_note:
            raise EvidenceActionError("The --note value is required for this command.")
        return None
    stripped = note.strip()
    if not stripped:
        if require_note:
            raise EvidenceActionError("The --note value cannot be blank.")
        return None
    if len(stripped) > MAX_OPERATOR_NOTE_CHARS:
        raise EvidenceActionError(
            f"The operator note is too long. Keep it under {MAX_OPERATOR_NOTE_CHARS} "
            "characters."
        )
    if any(character in stripped for character in ("\x00", "\r")):
        raise EvidenceActionError("The operator note must be plain text.")
    return stripped


def _action_id(
    *,
    deal_id: str,
    target_type: EvidenceActionTarget,
    target_id: str,
    status: EvidenceActionStatus,
    note: str | None,
    created_at: datetime,
) -> str:
    timestamp = created_at.isoformat()
    digest = hashlib.sha256(
        f"{deal_id}:{target_type}:{target_id}:{status}:{note or ''}:{timestamp}".encode()
    ).hexdigest()
    return f"act_{digest[:16]}"


def _validate_selector(*, deal_id: str | None, company_name: str | None) -> None:
    selector_count = sum(
        [
            deal_id is not None and deal_id.strip() != "",
            company_name is not None and company_name.strip() != "",
        ]
    )
    if selector_count > 1:
        raise EvidenceActionError("Choose only one deal selector: --deal-id or --company.")
    if deal_id is not None and not deal_id.strip():
        raise EvidenceActionError("The --deal-id value cannot be blank.")
    if company_name is not None and not company_name.strip():
        raise EvidenceActionError("The --company value cannot be blank.")


def _ensure_local_data(data_dir: Path) -> None:
    if data_dir.is_symlink():
        raise EvidenceActionError(
            f"The local data directory at {data_dir} is a symlink. Choose the real "
            "Hail Mary data folder."
        )
    for parent in data_dir.parents:
        if parent.is_symlink():
            raise EvidenceActionError(
                f"Hail Mary cannot use evidence actions at {data_dir} because {parent} "
                "is a symlinked parent folder."
            )
    if not data_dir.exists():
        raise EvidenceActionError(
            "No local generated-data folder was found. Run "
            "`hailmary evaluate-deal <company-folder>` before recording evidence actions."
        )
    if not data_dir.is_dir():
        raise EvidenceActionError(
            f"Hail Mary local data at {data_dir} is not a folder. Choose the data "
            "folder created by `hailmary evaluate-deal`."
        )


def _ensure_private_action_parent(path: Path, *, data_dir: Path, create: bool) -> None:
    configured_data_root = _absolute_path(data_dir)
    _ensure_local_data(configured_data_root)
    if path.is_symlink():
        raise EvidenceActionError(
            f"The evidence action folder at {path} is a symlink. Choose a real "
            "Hail Mary data folder."
        )
    data_root = configured_data_root.resolve(strict=False)
    resolved_path = _absolute_path(path).resolve(strict=False)
    if not _is_relative_to(resolved_path, data_root):
        raise EvidenceActionError(
            f"The evidence action folder {path} is outside the private data directory."
        )
    if path.exists() and not path.is_dir():
        raise EvidenceActionError(
            f"The evidence action path at {path} is not a folder. Choose a real "
            "Hail Mary data folder."
        )
    if not create:
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise EvidenceActionError(
            f"Could not create the evidence action folder at {path}: {exc}"
        ) from exc


def _load_ingestion_summary(summary_path: Path) -> IngestionSummary:
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceActionError(
            "The ingestion summary is not plain text. Run `hailmary evaluate-deal` again "
            "before recording evidence actions."
        ) from exc
    except OSError as exc:
        raise EvidenceActionError(
            f"Could not read the ingestion summary at {summary_path}: {exc}"
        ) from exc
    try:
        return IngestionSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise EvidenceActionError(
            "The ingestion summary could not be read. Run `hailmary evaluate-deal` again "
            "before recording evidence actions."
        ) from exc


def _select_one_deal(
    deals: list[IngestedDeal],
    *,
    deal_id: str | None,
    company_name: str | None,
) -> IngestedDeal:
    if not deals:
        raise EvidenceActionError(
            "The latest ingestion summary has no deals. Run `hailmary evaluate-deal` "
            "with deal documents before recording evidence actions."
        )
    if deal_id is not None:
        normalized_deal_id = deal_id.strip()
        matches = [deal for deal in deals if deal.id == normalized_deal_id]
        if not matches:
            raise EvidenceActionError(
                f"No ingested deal has deal ID {normalized_deal_id}."
            )
        return matches[0]
    if company_name is not None:
        normalized_name = company_name.strip().casefold()
        matches = [deal for deal in deals if deal.company_name.casefold() == normalized_name]
        if not matches:
            raise EvidenceActionError(
                f"No ingested deal has exact company name {company_name.strip()}."
            )
        if len(matches) > 1:
            raise EvidenceActionError(
                f"Company name {company_name.strip()} matches more than one ingested deal. "
                "Use --deal-id instead."
            )
        return matches[0]
    if len(deals) > 1:
        raise EvidenceActionError(
            f"The latest ingestion summary has {len(deals)} deals. Use --deal-id or "
            "--company to choose which evidence actions to review."
        )
    return deals[0]


def _evidence_store_path_for_deal(
    deal: IngestedDeal,
    *,
    data_dir: Path,
    summary_path: Path,
) -> Path:
    if deal.evidence_store_path is None:
        raise EvidenceActionError(
            f"No evidence store was found for {deal.company_name}. Run "
            "`hailmary evaluate-deal` again before recording evidence actions."
        )
    store_path = _resolve_saved_path(
        deal.evidence_store_path,
        data_dir=data_dir,
        summary_path=summary_path,
    )
    if not store_path.exists():
        raise EvidenceActionError(
            f"The evidence store for {deal.company_name} is missing at {store_path}. "
            "Run `hailmary evaluate-deal` again before recording evidence actions."
        )
    return store_path


def _load_evidence_store(path: Path, *, company_name: str) -> EvidenceStore:
    try:
        raw_store = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceActionError(
            f"The evidence store for {company_name} is not plain text. Run "
            "`hailmary evaluate-deal` again before recording evidence actions."
        ) from exc
    except OSError as exc:
        raise EvidenceActionError(
            f"Could not read the evidence store for {company_name} at {path}: {exc}"
        ) from exc
    try:
        return EvidenceStore.model_validate_json(raw_store)
    except ValidationError as exc:
        raise EvidenceActionError(
            f"The evidence store for {company_name} could not be read. Run "
            "`hailmary evaluate-deal` again before recording evidence actions."
        ) from exc


def _resolve_saved_path(path: Path, *, data_dir: Path, summary_path: Path) -> Path:
    absolute_data_dir = _absolute_path(data_dir).resolve(strict=False)
    if path.is_absolute():
        resolved_path = path.resolve(strict=False)
        if not _is_relative_to(resolved_path, absolute_data_dir):
            raise EvidenceActionError(
                f"The evidence store path {path} is outside the private data directory."
            )
        return resolved_path

    absolute_summary_path = _absolute_path(summary_path).resolve(strict=False)
    candidate_roots = [
        Path.cwd().resolve(strict=False),
        *absolute_data_dir.parents,
        absolute_data_dir,
        absolute_summary_path.parent,
    ]
    candidates = [root / path for root in candidate_roots]
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir) and candidate.exists():
            return resolved_candidate
    for candidate in candidates:
        resolved_candidate = candidate.resolve(strict=False)
        if _is_relative_to(resolved_candidate, absolute_data_dir):
            return resolved_candidate
    return (absolute_data_dir / path.name).resolve(strict=False)


def _validation_error_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "document: Invalid structured JSON."
    first_error = errors[0]
    location = first_error.get("loc", ())
    location_text = ".".join(str(part) for part in location) or "document"
    message = str(first_error.get("msg", "Invalid structured JSON."))
    return f"{location_text}: {message}."


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
