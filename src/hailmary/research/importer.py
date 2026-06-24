from __future__ import annotations

import hashlib
import json
import os
import secrets
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.evidence import refresh_deal_term_claims
from hailmary.schemas.documents import (
    DocumentType,
    FileType,
    IngestedDeal,
    IngestionSummary,
    SourceKind,
)
from hailmary.schemas.evidence import (
    EvidenceKind,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
)
from hailmary.utils.slug import slugify

from .providers import builtin_research_providers
from .schemas import (
    ResearchImportDealSummary,
    ResearchImportRunSummary,
    ResearchProvider,
    ResearchResultInput,
    ResearchResultsFile,
)


class ResearchImportError(RuntimeError):
    """External research results could not be imported safely."""


STALE_SOURCE_DAYS = 365
SUPPORTED_SOURCE_KINDS = {SourceKind.WEB, SourceKind.MERIDIAN, SourceKind.MANUAL_NOTE}
SUPPORTED_DOCUMENT_TYPES = {
    DocumentType.WEB_PAGE,
    DocumentType.PLATFORM_DEAL_PAGE,
    DocumentType.MEMO,
}


@dataclass(frozen=True)
class _MatchedResearchResult:
    result: ResearchResultInput
    deal: IngestedDeal
    evidence: EvidenceRecord


def import_research_results(
    *,
    config: AppConfig,
    results_path: Path,
    imported_at: datetime | None = None,
) -> ResearchImportRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchImportError(str(exc)) from exc

    imported_at = _as_utc(imported_at or datetime.now(UTC))
    input_path = _resolve_input_file(results_path)
    results_file = _load_results_file(input_path)
    _validate_results(results_file.results, imported_at=imported_at)

    summary_path = config.data_dir / "processed" / "ingestion_summary.json"
    if not summary_path.exists():
        raise ResearchImportError(
            "No ingested deals were found. Run `hailmary ingest-folder` before importing "
            "research results."
        )
    summary = _load_ingestion_summary(summary_path)
    matches = _match_results_to_deals(
        results_file.results,
        summary.deals,
        imported_at=imported_at,
    )
    stores, store_paths = _load_required_evidence_stores(
        matches,
        data_dir=config.data_dir,
        summary_path=summary_path,
    )

    evidence_by_deal_id = {
        deal_id: list(store.evidence) for deal_id, store in stores.items()
    }
    existing_ids_by_deal_id = {
        deal_id: {evidence.id for evidence in store.evidence}
        for deal_id, store in stores.items()
    }
    imported_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}

    for match in matches:
        deal_id = match.deal.id
        if match.evidence.id in existing_ids_by_deal_id[deal_id]:
            duplicate_counts[deal_id] = duplicate_counts.get(deal_id, 0) + 1
            continue
        evidence_by_deal_id[deal_id].append(match.evidence)
        existing_ids_by_deal_id[deal_id].add(match.evidence.id)
        imported_counts[deal_id] = imported_counts.get(deal_id, 0) + 1

    updated_stores: dict[str, EvidenceStore] = {}
    for deal_id, imported_count in imported_counts.items():
        updated_store = stores[deal_id].model_copy(
            update={
                "evidence": evidence_by_deal_id[deal_id],
                "notes": _add_import_note(
                    stores[deal_id].notes,
                    input_path=input_path,
                    imported_at=imported_at,
                    imported_count=imported_count,
                ),
            }
        )
        updated_stores[deal_id] = refresh_deal_term_claims(updated_store)

    for path in [*store_paths.values(), summary_path]:
        _preflight_private_output_path(path, private_root=config.data_dir)

    for deal_id, store in updated_stores.items():
        _write_private_json(
            store_paths[deal_id],
            store.model_dump_json(indent=2),
            description=f"evidence store for {store.company_name}",
        )

    if updated_stores:
        summary = _summary_with_updated_counts(summary, updated_stores)
        _write_private_json(
            summary_path,
            summary.model_dump_json(indent=2),
            description="ingestion summary",
        )

    return ResearchImportRunSummary(
        input_path=input_path,
        imported_at=imported_at,
        deals=_import_deal_summaries(
            matches,
            store_paths=store_paths,
            imported_counts=imported_counts,
            duplicate_counts=duplicate_counts,
        ),
    )


def _resolve_input_file(path: Path) -> Path:
    expanded_path = path.expanduser()
    absolute_path = expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path
    if absolute_path.is_symlink():
        raise ResearchImportError("The research results file cannot be a symlink.")
    for parent in absolute_path.parents:
        if parent.is_symlink():
            raise ResearchImportError(
                f"Hail Mary cannot read {path} because {parent} is a symlinked parent folder."
            )
    resolved_path = absolute_path.resolve(strict=False)
    if not resolved_path.exists():
        raise ResearchImportError(f"The research results file does not exist: {path}")
    if not resolved_path.is_file():
        raise ResearchImportError(f"The research results path is not a file: {path}")
    return resolved_path


def _load_results_file(path: Path) -> ResearchResultsFile:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchImportError(
            "The research results file is not plain UTF-8 text. Save it as JSON text "
            "and try again."
        ) from exc
    except OSError as exc:
        raise ResearchImportError(
            f"Could not read the research results file at {path}: {exc}"
        ) from exc

    try:
        raw_payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResearchImportError(
            f"The research results file is not valid JSON: {exc.msg}."
        ) from exc
    if isinstance(raw_payload, list):
        raw_payload = {"results": raw_payload}
    elif not isinstance(raw_payload, dict):
        raise ResearchImportError(
            "The research results file must be a JSON object with a `results` list."
        )
    try:
        return ResearchResultsFile.model_validate(raw_payload)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchImportError(f"The research results file is incomplete: {detail}") from exc


def _validate_results(results: list[ResearchResultInput], *, imported_at: datetime) -> None:
    for index, result in enumerate(results, start=1):
        if result.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise ResearchImportError(
                f"Research result {index} uses source kind {result.source_kind}. "
                "Use web, meridian, or manual_note."
            )
        if result.document_type not in SUPPORTED_DOCUMENT_TYPES:
            raise ResearchImportError(
                f"Research result {index} uses document type {result.document_type}. "
                "Use web_page, platform_deal_page, or memo."
            )
        if result.source_url is not None:
            _validate_source_url(result.source_url, index=index)
        retrieved_at = _as_utc(result.retrieved_at)
        if retrieved_at > imported_at:
            raise ResearchImportError(
                f"Research result {index} has a retrieved_at timestamp in the future."
            )
        _provider_name(result)


def _validate_source_url(source_url: str, *, index: int) -> None:
    try:
        parsed = urlparse(source_url)
    except ValueError as exc:
        raise ResearchImportError(
            f"Research result {index} has a source_url that is not a valid URL."
        ) from exc
    if parsed.scheme not in {"http", "https"}:
        raise ResearchImportError(
            f"Research result {index} source_url must start with http:// or https://."
        )
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise ResearchImportError(
            f"Research result {index} has a source_url that is not a valid URL."
        ) from exc
    if not parsed.netloc or host is None:
        raise ResearchImportError(
            f"Research result {index} source_url must include a website host."
        )
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ResearchImportError(
            f"Research result {index} source_url has an invalid port."
        ) from exc
    if parsed.username is not None or parsed.password is not None:
        raise ResearchImportError(
            f"Research result {index} source_url cannot include a username or password."
        )
    if any(character.isspace() for character in source_url):
        raise ResearchImportError(f"Research result {index} source_url cannot contain spaces.")


def _match_results_to_deals(
    results: list[ResearchResultInput],
    deals: list[IngestedDeal],
    *,
    imported_at: datetime,
) -> list[_MatchedResearchResult]:
    if not deals:
        raise ResearchImportError(
            "The latest ingestion summary does not contain any deals. Run "
            "`hailmary ingest-folder` again before importing research results."
        )
    deal_by_id = {deal.id: deal for deal in deals}
    deals_by_name: dict[str, list[IngestedDeal]] = {}
    for deal in deals:
        deals_by_name.setdefault(deal.company_name.casefold(), []).append(deal)

    matches: list[_MatchedResearchResult] = []
    for index, result in enumerate(results, start=1):
        deal = _match_result_to_deal(
            result,
            index=index,
            deal_by_id=deal_by_id,
            deals_by_name=deals_by_name,
        )
        matches.append(
            _MatchedResearchResult(
                result=result,
                deal=deal,
                evidence=_evidence_record_for_result(
                    result,
                    deal=deal,
                    imported_at=imported_at,
                ),
            )
        )
    return matches


def _match_result_to_deal(
    result: ResearchResultInput,
    *,
    index: int,
    deal_by_id: dict[str, IngestedDeal],
    deals_by_name: dict[str, list[IngestedDeal]],
) -> IngestedDeal:
    matched_by_id = deal_by_id.get(result.deal_id) if result.deal_id is not None else None
    matched_by_name: IngestedDeal | None = None
    if result.company_name is not None:
        name_matches = deals_by_name.get(result.company_name.casefold(), [])
        if len(name_matches) > 1:
            raise ResearchImportError(
                f"Research result {index} matches more than one deal named "
                f"{result.company_name}. Use deal_id instead."
            )
        if name_matches:
            matched_by_name = name_matches[0]

    if result.deal_id is not None and matched_by_id is None:
        raise ResearchImportError(
            f"Research result {index} uses deal_id {result.deal_id}, but no ingested "
            "deal has that ID."
        )
    if result.company_name is not None and matched_by_name is None:
        raise ResearchImportError(
            f"Research result {index} uses company_name {result.company_name}, but no "
            "ingested deal has that exact name."
        )
    if (
        matched_by_id is not None
        and matched_by_name is not None
        and matched_by_id.id != matched_by_name.id
    ):
        raise ResearchImportError(
            f"Research result {index} has a deal_id and company_name that refer to "
            "different deals."
        )
    return matched_by_id or matched_by_name or _raise_missing_deal_reference(index)


def _raise_missing_deal_reference(index: int) -> IngestedDeal:
    raise ResearchImportError(f"Research result {index} needs a deal_id or company_name.")


def _load_required_evidence_stores(
    matches: list[_MatchedResearchResult],
    *,
    data_dir: Path,
    summary_path: Path,
) -> tuple[dict[str, EvidenceStore], dict[str, Path]]:
    stores: dict[str, EvidenceStore] = {}
    store_paths: dict[str, Path] = {}
    for match in matches:
        deal = match.deal
        if deal.id in stores:
            continue
        if deal.evidence_store_path is None:
            raise ResearchImportError(
                f"No evidence store was found for {deal.company_name}. "
                "Run `hailmary ingest-folder` again before importing research results."
            )
        evidence_store_path = _resolve_saved_path(
            deal.evidence_store_path,
            data_dir=data_dir,
            summary_path=summary_path,
        )
        if not evidence_store_path.exists():
            raise ResearchImportError(
                f"The evidence store for {deal.company_name} is missing at "
                f"{evidence_store_path}. Run `hailmary ingest-folder` again."
            )
        stores[deal.id] = _load_evidence_store(evidence_store_path, company_name=deal.company_name)
        store_paths[deal.id] = evidence_store_path
    return stores, store_paths


def _load_ingestion_summary(summary_path: Path) -> IngestionSummary:
    try:
        raw_summary = summary_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchImportError(
            "The ingestion summary is not plain text. Run `hailmary ingest-folder` again."
        ) from exc
    except OSError as exc:
        raise ResearchImportError(
            f"Could not read the ingestion summary at {summary_path}: {exc}"
        ) from exc
    try:
        return IngestionSummary.model_validate_json(raw_summary)
    except ValidationError as exc:
        raise ResearchImportError(
            "The ingestion summary could not be read. Run `hailmary ingest-folder` again."
        ) from exc


def _load_evidence_store(path: Path, *, company_name: str) -> EvidenceStore:
    try:
        raw_store = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchImportError(
            f"The evidence store for {company_name} is not plain text. "
            "Run `hailmary ingest-folder` again."
        ) from exc
    except OSError as exc:
        raise ResearchImportError(
            f"Could not read the evidence store for {company_name} at {path}: {exc}"
        ) from exc
    try:
        return EvidenceStore.model_validate_json(raw_store)
    except ValidationError as exc:
        raise ResearchImportError(
            f"The evidence store for {company_name} could not be read. "
            "Run `hailmary ingest-folder` again."
        ) from exc


def _evidence_record_for_result(
    result: ResearchResultInput,
    *,
    deal: IngestedDeal,
    imported_at: datetime,
) -> EvidenceRecord:
    text = result.text.strip()
    digest = _result_digest(result, deal_id=deal.id)
    provider_name = _provider_name(result)
    document_type = result.document_type
    if result.source_kind == SourceKind.MERIDIAN and document_type == DocumentType.WEB_PAGE:
        document_type = DocumentType.PLATFORM_DEAL_PAGE

    return EvidenceRecord(
        id=f"ev_external_{slugify(result.provider_id)}_{digest[:16]}",
        deal_id=deal.id,
        document_id=_external_document_id(result, deal_id=deal.id),
        document_path=Path("external-research")
        / slugify(result.provider_id)
        / f"{slugify(result.title)}-{digest[:8]}.txt",
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=result.source_kind,
        document_type=document_type,
        file_type=FileType.TXT,
        text=text,
        source_span_start=0,
        source_span_end=len(text),
        source_freshness=_source_freshness(result.retrieved_at, now=imported_at),
        provider_id=result.provider_id,
        provider_name=provider_name,
        source_url=result.source_url,
        source_api=result.source_api,
        retrieved_at=_as_utc(result.retrieved_at),
        external_confidence=result.confidence,
        licensing_notes=result.licensing_notes,
    )


def _result_digest(result: ResearchResultInput, *, deal_id: str) -> str:
    source_reference = result.source_url or result.source_api or ""
    payload = "\0".join(
        [
            deal_id,
            result.provider_id,
            result.source_kind.value,
            source_reference,
            result.text,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _external_document_id(result: ResearchResultInput, *, deal_id: str) -> str:
    source_reference = result.source_url or result.source_api or ""
    payload = "\0".join(
        [
            deal_id,
            result.provider_id,
            result.source_kind.value,
            source_reference,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"doc_external_{slugify(result.provider_id)}_{digest[:12]}"


def _provider_name(result: ResearchResultInput) -> str:
    if result.provider_name:
        return result.provider_name
    provider = _known_providers().get(result.provider_id)
    if provider is not None:
        return provider.name
    return result.provider_id


def _known_providers() -> dict[str, ResearchProvider]:
    return {
        provider.id: provider
        for provider in builtin_research_providers(include_paid=True, include_meridian=True)
    }


def _source_freshness(retrieved_at: datetime, *, now: datetime) -> SourceFreshness:
    retrieved_at = _as_utc(retrieved_at)
    age_days = (_as_utc(now) - retrieved_at).days
    if age_days > STALE_SOURCE_DAYS:
        return SourceFreshness.STALE
    return SourceFreshness.CURRENT


def _summary_with_updated_counts(
    summary: IngestionSummary,
    updated_stores: dict[str, EvidenceStore],
) -> IngestionSummary:
    updated_deals = []
    for deal in summary.deals:
        store = updated_stores.get(deal.id)
        if store is None:
            updated_deals.append(deal)
            continue
        updated_deals.append(
            deal.model_copy(
                update={
                    "evidence_count": store.evidence_count,
                    "claim_count": store.claim_count,
                    "conflict_count": store.conflict_count,
                }
            )
        )
    return summary.model_copy(update={"deals": updated_deals})


def _import_deal_summaries(
    matches: list[_MatchedResearchResult],
    *,
    store_paths: dict[str, Path],
    imported_counts: dict[str, int],
    duplicate_counts: dict[str, int],
) -> list[ResearchImportDealSummary]:
    deal_by_id = {match.deal.id: match.deal for match in matches}
    touched_deal_ids = sorted(set(imported_counts) | set(duplicate_counts))
    return [
        ResearchImportDealSummary(
            deal_id=deal_id,
            company_name=deal_by_id[deal_id].company_name,
            evidence_store_path=store_paths[deal_id],
            imported_count=imported_counts.get(deal_id, 0),
            skipped_duplicate_count=duplicate_counts.get(deal_id, 0),
        )
        for deal_id in touched_deal_ids
    ]


def _add_import_note(
    notes: list[str],
    *,
    input_path: Path,
    imported_at: datetime,
    imported_count: int,
) -> list[str]:
    record_word = "record" if imported_count == 1 else "records"
    note = (
        f"Imported {imported_count} external research evidence {record_word} from "
        f"{input_path.name} at {_as_utc(imported_at).isoformat()}. "
        "No websites or APIs were contacted by this command."
    )
    return [*notes, note]


def _resolve_saved_path(path: Path, *, data_dir: Path, summary_path: Path) -> Path:
    absolute_data_dir = _absolute_path(data_dir).resolve(strict=False)
    if path.is_absolute():
        resolved_path = path.resolve(strict=False)
        if not _is_relative_to(resolved_path, absolute_data_dir):
            raise ResearchImportError(
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


def _preflight_private_output_path(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ResearchImportError(
            f"Generated output path {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ResearchImportError(f"Generated output path {path} is a symlink.")
    if path.exists() and not path.is_file():
        raise ResearchImportError(f"Generated output path {path} is not a file.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
    except OSError as exc:
        raise ResearchImportError(f"Could not prepare generated output at {path}: {exc}") from exc


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ResearchImportError(
            f"Could not write {description} at {path}: output file is a symlink."
        )
    token = secrets.token_hex(8)
    temp_path = path.with_name(f".{path.name}.{token}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(temp_path, flags, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        temp_path.chmod(0o600)
        os.replace(temp_path, path)
        path.chmod(0o600)
    except UnicodeEncodeError as exc:
        raise ResearchImportError(
            f"Could not write {description} at {path}: the imported evidence contains "
            "text that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ResearchImportError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        if temp_path.exists():
            with suppress(OSError):
                temp_path.unlink()


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _first_validation_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "The JSON did not match the expected shape."
    first_error = errors[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "Invalid value."))
    return f"{location}: {message}" if location else message
