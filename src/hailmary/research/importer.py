from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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

from .meridian import (
    MERIDIAN_LEGACY_PLACEHOLDER_CONFIDENCE,
    MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER,
    MERIDIAN_PLACEHOLDER_CONFIDENCE,
    MERIDIAN_RECOMMENDED_FACTS,
    MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER,
    MERIDIAN_WORKFLOW_SOURCE_URL_MARKER_PREFIX,
    MERIDIAN_WORKFLOW_TEMPLATE_MARKER,
    MeridianWorkflowError,
    clean_meridian_url,
)
from .providers import builtin_research_providers
from .schemas import (
    ResearchImportDealSummary,
    ResearchImportRunSummary,
    ResearchProvider,
    ResearchResultInput,
    ResearchResultsFile,
)
from .source_urls import (
    source_reference_looks_like_url,
    validate_http_url,
    validate_provider_source_url,
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
RESEARCH_RESULT_FIELDS = {
    "deal_id",
    "company_name",
    "provider_id",
    "provider_name",
    "title",
    "text",
    "retrieved_at",
    "source_url",
    "source_api",
    "confidence",
    "licensing_notes",
    "source_kind",
    "document_type",
}
TEMPLATE_REQUIRED_FACT_FIELDS = {
    "title",
    "text",
    "retrieved_at",
    "confidence",
}
MERIDIAN_PLACEHOLDER_TITLES = {
    f"Meridian: {fact}" for fact in MERIDIAN_RECOMMENDED_FACTS
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
    dry_run: bool = False,
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

    if not dry_run:
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
        dry_run=dry_run,
        skipped_blank_template_row_count=results_file._skipped_blank_template_row_count,
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
    if not isinstance(raw_payload, dict):
        raise ResearchImportError(
            "The research results file must be a JSON object with a `results` list."
        )
    try:
        return _validate_results_file_payload(raw_payload)
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise ResearchImportError(f"The research results file is incomplete: {detail}") from exc


def _validate_results_file_payload(raw_payload: dict[str, object]) -> ResearchResultsFile:
    raw_results = raw_payload.get("results")
    if not isinstance(raw_results, list):
        return ResearchResultsFile.model_validate(raw_payload)

    extra_top_level_fields = set(raw_payload) - {"results"}
    if extra_top_level_fields:
        return ResearchResultsFile.model_validate(raw_payload)

    results: list[ResearchResultInput] = []
    skipped_blank_template_row_count = 0
    for index, raw_result in enumerate(raw_results, start=1):
        if _is_blank_template_result(raw_result):
            skipped_blank_template_row_count += 1
            continue
        try:
            result = ResearchResultInput.model_validate(raw_result)
        except ValidationError as exc:
            detail = _first_validation_detail(exc)
            raise ResearchImportError(
                f"The research results file is incomplete: row {index}: {detail}"
            ) from exc
        result._original_row_number = index
        results.append(result)
    results_file = ResearchResultsFile.model_validate({"results": results})
    results_file._skipped_blank_template_row_count = skipped_blank_template_row_count
    return results_file


def _is_blank_template_result(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    if set(result) != RESEARCH_RESULT_FIELDS:
        return False
    if _is_untouched_meridian_placeholder_result(result):
        return True
    if not all(
        _is_blank_template_value(result.get(field))
        for field in TEMPLATE_REQUIRED_FACT_FIELDS
    ):
        return False
    if _is_blank_template_value(result.get("source_url")) and _is_blank_template_value(
        result.get("source_api")
    ):
        return True
    return _is_blank_prefilled_meridian_template_result(result)


def _is_blank_template_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_untouched_meridian_placeholder_result(result: dict[str, object]) -> bool:
    if result.get("provider_id") != "meridian":
        return False
    if result.get("source_kind") != SourceKind.MERIDIAN.value:
        return False
    if result.get("document_type") != DocumentType.PLATFORM_DEAL_PAGE.value:
        return False
    if not _is_meridian_placeholder_title(result.get("title")):
        return False
    if not _is_blank_template_value(result.get("text")):
        return False
    if not _is_blank_template_value(result.get("retrieved_at")):
        return False
    if not _is_blank_template_value(result.get("source_api")):
        return False
    if not _is_meridian_placeholder_confidence(result.get("confidence")):
        return False

    licensing_notes = result.get("licensing_notes")
    if not isinstance(licensing_notes, str):
        return False
    if not _has_meridian_placeholder_marker(licensing_notes):
        return False
    if MERIDIAN_WORKFLOW_TEMPLATE_MARKER not in licensing_notes:
        return False
    generated_source_url = _generated_meridian_source_url(licensing_notes)
    if generated_source_url is None:
        return False

    source_url = result.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        return False
    if source_url.strip() != generated_source_url:
        return False
    try:
        clean_meridian_url(source_url)
    except MeridianWorkflowError:
        return False
    return True


def _is_meridian_placeholder_confidence(value: object) -> bool:
    return isinstance(value, str) and value.strip() in {
        MERIDIAN_PLACEHOLDER_CONFIDENCE,
        MERIDIAN_LEGACY_PLACEHOLDER_CONFIDENCE,
    }


def _has_meridian_placeholder_marker(value: str) -> bool:
    return (
        MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER in value
        or MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER in value
    )


def _is_meridian_placeholder_title(value: object) -> bool:
    return isinstance(value, str) and value.strip() in MERIDIAN_PLACEHOLDER_TITLES


def _generated_meridian_source_url(licensing_notes: str) -> str | None:
    match = re.search(
        rf"{re.escape(MERIDIAN_WORKFLOW_SOURCE_URL_MARKER_PREFIX)}(\S+)",
        licensing_notes,
    )
    if match is None:
        return None
    return match.group(1)


def _is_blank_prefilled_meridian_template_result(result: dict[str, object]) -> bool:
    if result.get("provider_id") != "meridian":
        return False
    if result.get("source_kind") != SourceKind.MERIDIAN.value:
        return False
    if result.get("document_type") != DocumentType.PLATFORM_DEAL_PAGE.value:
        return False
    licensing_notes = result.get("licensing_notes")
    if (
        not isinstance(licensing_notes, str)
        or MERIDIAN_WORKFLOW_TEMPLATE_MARKER not in licensing_notes
    ):
        return False
    if not _is_blank_template_value(result.get("source_api")):
        return False
    source_url = result.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        return False
    try:
        clean_meridian_url(source_url)
    except MeridianWorkflowError:
        return False
    return True


def _validate_results(results: list[ResearchResultInput], *, imported_at: datetime) -> None:
    for index, result in enumerate(results, start=1):
        display_index = _result_index(result, fallback=index)
        if result.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise ResearchImportError(
                f"Research result {display_index} uses source kind {result.source_kind}. "
                "Use web, meridian, or manual_note."
            )
        if result.document_type not in SUPPORTED_DOCUMENT_TYPES:
            raise ResearchImportError(
                f"Research result {display_index} uses document type {result.document_type}. "
                "Use web_page, platform_deal_page, or memo."
            )
        if result.source_kind == SourceKind.MERIDIAN:
            _validate_meridian_result_source(result, index=display_index)
            _validate_completed_meridian_placeholder(result, index=display_index)
            _validate_saved_meridian_licensing_notes(result, index=display_index)
        elif result.source_url is not None:
            _validate_url_reference(
                result.source_url,
                index=display_index,
                field_name="source_url",
            )
        if result.source_api is not None:
            _validate_source_api(result.source_api, index=display_index)
        _validate_known_provider_source_kind(result, index=display_index)
        _validate_known_provider_source_locations(result, index=display_index)
        retrieved_at = _as_utc(result.retrieved_at)
        if retrieved_at > imported_at:
            raise ResearchImportError(
                f"Research result {display_index} has a retrieved_at timestamp in the future."
            )
        _provider_name(result)


def _validate_url_reference(source_url: str, *, index: int, field_name: str) -> None:
    try:
        validate_http_url(source_url, field_name=field_name)
    except ValueError as exc:
        raise ResearchImportError(
            f"Research result {index} {exc}."
        ) from exc


def _validate_source_api(source_api: str, *, index: int) -> None:
    if source_reference_looks_like_url(source_api):
        _validate_url_reference(source_api, index=index, field_name="source_api")


def _validate_meridian_result_source(
    result: ResearchResultInput,
    *,
    index: int,
) -> None:
    if result.source_url is None:
        raise ResearchImportError(
            f"Research result {index} uses Meridian evidence, so source_url must be "
            "a Meridian deal page URL."
        )
    if result.source_api is not None:
        raise ResearchImportError(
            f"Research result {index} uses Meridian evidence, so source_api must be "
            "blank. Keep the safe Meridian deal page URL in source_url."
        )
    try:
        clean_meridian_url(result.source_url)
    except MeridianWorkflowError as exc:
        raise ResearchImportError(
            f"Research result {index} source_url is not a safe Meridian deal page URL: "
            f"{exc}"
        ) from exc


def _validate_saved_meridian_licensing_notes(
    result: ResearchResultInput,
    *,
    index: int,
) -> None:
    if _saved_licensing_notes(result.licensing_notes):
        return
    raise ResearchImportError(
        f"Research result {index} uses Meridian evidence, so licensing_notes must "
        "explain the source permissions after the generated template marker is removed."
    )


def _validate_completed_meridian_placeholder(
    result: ResearchResultInput,
    *,
    index: int,
) -> None:
    generated_source_url = _generated_meridian_source_url(result.licensing_notes)
    if _is_meridian_placeholder_title(result.title) and generated_source_url is None:
        raise ResearchImportError(
            f"Research result {index} uses a generated Meridian placeholder title, so "
            "licensing_notes must keep the generated placeholder marker until import."
        )
    if generated_source_url is not None and result.source_url != generated_source_url:
        raise ResearchImportError(
            f"Research result {index} uses a generated Meridian placeholder, so "
            "source_url must stay as the generated Meridian deal page URL."
        )
    if not _is_meridian_placeholder_confidence(result.confidence):
        return
    raise ResearchImportError(
        f"Research result {index} uses Meridian placeholder confidence guidance. "
        "Replace confidence with your own confidence note before importing."
    )


def _validate_known_provider_source_kind(
    result: ResearchResultInput,
    *,
    index: int,
) -> None:
    provider = _known_providers().get(result.provider_id)
    if provider is None or result.source_kind == provider.source_kind:
        return
    raise ResearchImportError(
        f"Research result {index} uses provider_id {result.provider_id}, so "
        f"source_kind must be {provider.source_kind.value}."
    )


def _validate_known_provider_source_locations(
    result: ResearchResultInput,
    *,
    index: int,
) -> None:
    if result.source_url is not None:
        _validate_provider_reference_host(
            result.provider_id,
            result.source_url,
            index=index,
            field_name="source_url",
        )
    if result.source_api is not None and source_reference_looks_like_url(result.source_api):
        _validate_provider_reference_host(
            result.provider_id,
            result.source_api,
            index=index,
            field_name="source_api",
        )


def _validate_provider_reference_host(
    provider_id: str,
    value: str,
    *,
    index: int,
    field_name: str,
) -> None:
    try:
        validate_provider_source_url(provider_id, value, field_name=field_name)
    except ValueError as exc:
        raise ResearchImportError(
            f"Research result {index} uses provider_id {provider_id}, so {exc}."
        ) from exc


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
        display_index = _result_index(result, fallback=index)
        deal = _match_result_to_deal(
            result,
            index=display_index,
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


def _result_index(result: ResearchResultInput, *, fallback: int) -> int:
    return result._original_row_number or fallback


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
        licensing_notes=_saved_licensing_notes(result.licensing_notes),
    )


def _saved_licensing_notes(licensing_notes: str) -> str:
    if (
        MERIDIAN_WORKFLOW_TEMPLATE_MARKER not in licensing_notes
        and not _has_meridian_placeholder_marker(licensing_notes)
        and MERIDIAN_WORKFLOW_SOURCE_URL_MARKER_PREFIX not in licensing_notes
    ):
        return licensing_notes
    cleaned = licensing_notes
    cleaned = cleaned.replace(MERIDIAN_WORKFLOW_TEMPLATE_MARKER, "")
    cleaned = cleaned.replace(MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER, "")
    cleaned = cleaned.replace(MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER, "")
    cleaned = re.sub(
        rf"{re.escape(MERIDIAN_WORKFLOW_SOURCE_URL_MARKER_PREFIX)}\S+",
        "",
        cleaned,
    )
    return " ".join(cleaned.split())


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
    location_parts = tuple(first_error.get("loc", ()))
    location = ".".join(str(part) for part in location_parts)
    message = str(first_error.get("msg", "Invalid value."))
    error_type = str(first_error.get("type", ""))
    field_name = str(location_parts[-1]) if location_parts else ""
    plain_message = _plain_research_result_validation_message(
        field_name=field_name,
        message=message,
        error_type=error_type,
    )
    return f"{location}: {plain_message}" if location else plain_message


def _plain_research_result_validation_message(
    *,
    field_name: str,
    message: str,
    error_type: str,
) -> str:
    if field_name == "retrieved_at":
        if error_type != "missing":
            return (
                "retrieved_at is invalid. Use an ISO 8601 timestamp for when the "
                "source was retrieved or viewed, such as 2026-01-01T12:00:00Z."
            )
        return (
            "retrieved_at is required. Enter the time the source was retrieved or "
            "viewed, such as 2026-01-01T12:00:00Z."
        )
    if field_name == "licensing_notes":
        return (
            "licensing_notes is required. Explain why this source or short excerpt "
            "can be saved and used for diligence."
        )
    if field_name == "confidence":
        return (
            "confidence is required. Add your confidence note, such as high: exact "
            "source match, or leave an untouched generated placeholder row unchanged."
        )
    if field_name == "text":
        return (
            "text is required. Add only the short source-backed evidence text, or "
            "leave an untouched generated placeholder row unchanged."
        )
    if field_name == "title":
        return (
            "title is required. Complete the row with a short source-backed title, "
            "or leave an untouched generated placeholder row unchanged."
        )
    if field_name == "source_url":
        return (
            "source_url is incomplete. Use the exact source URL, or use source_api "
            "for an API source reference."
        )
    if field_name == "source_api":
        return (
            "source_api is incomplete. Use the exact API source reference, or use "
            "source_url for a web page."
        )
    return message
