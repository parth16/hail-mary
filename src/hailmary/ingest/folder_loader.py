from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from hailmary.config import AppConfig
from hailmary.ingest.document_classifier import (
    SUPPORTED_SUFFIXES,
    classify_document,
    classify_file_type,
    is_ignored_path,
)
from hailmary.ingest.extractors import ExtractionResult, extract_document
from hailmary.schemas.documents import (
    IngestedDeal,
    IngestedDocument,
    IngestionSummary,
    SourceDocument,
    SourceKind,
)
from hailmary.utils.slug import slugify


def ingest_folder(root_path: Path, *, config: AppConfig) -> IngestionSummary:
    """Scan a local folder and store extracted document metadata."""

    root_path = root_path.expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"The folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"This is not a folder: {root_path}")

    run_started_at = datetime.now(UTC)
    deals_by_id: dict[str, IngestedDeal] = {}
    deal_ids_by_name: dict[str, str] = {}
    skipped_files: list[str] = []

    for path in sorted(root_path.rglob("*")):
        relative_path = path.relative_to(root_path)
        if path.is_symlink():
            skipped_files.append(str(relative_path))
            continue
        if not path.is_file():
            continue
        if is_ignored_path(relative_path) or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped_files.append(str(relative_path))
            continue

        deal_name = _deal_name_for_path(relative_path)
        deal_id = _deal_id_for_name(
            deal_name,
            deal_ids_by_name=deal_ids_by_name,
            used_deal_ids=set(deals_by_id),
        )
        extraction = extract_document(path)
        document = _build_document(path, root_path, deal_id, extraction, run_started_at)

        processed_document = IngestedDocument(
            source=document,
            pages=extraction.pages,
            output_path=_write_document(config, deal_id, document, extraction),
        )

        if deal_id not in deals_by_id:
            deals_by_id[deal_id] = IngestedDeal(
                id=deal_id,
                company_name=deal_name,
                documents=[],
            )
        deals_by_id[deal_id].documents.append(processed_document)

    summary = IngestionSummary(
        root_path=root_path,
        scanned_at=run_started_at,
        deals=list(deals_by_id.values()),
        skipped_files=skipped_files,
        summary_path=config.data_dir / "processed" / "ingestion_summary.json",
    )
    _write_summary(summary)
    return summary


def _deal_name_for_path(relative_path: Path) -> str:
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return relative_path.stem


def _deal_id_for_name(
    deal_name: str,
    *,
    deal_ids_by_name: dict[str, str],
    used_deal_ids: set[str],
) -> str:
    existing_deal_id = deal_ids_by_name.get(deal_name)
    if existing_deal_id is not None:
        return existing_deal_id

    base_deal_id = slugify(deal_name)
    deal_id = base_deal_id
    if deal_id in used_deal_ids:
        digest = hashlib.sha256(deal_name.encode("utf-8")).hexdigest()[:8]
        deal_id = f"{base_deal_id}-{digest}"

    suffix = 2
    while deal_id in used_deal_ids:
        deal_id = f"{base_deal_id}-{suffix}"
        suffix += 1

    deal_ids_by_name[deal_name] = deal_id
    return deal_id


def _build_document(
    path: Path,
    root_path: Path,
    deal_id: str,
    extraction: ExtractionResult,
    ingested_at: datetime,
) -> SourceDocument:
    text_sample = extraction.combined_text[:4_000]
    raw_text_sample = extraction.combined_raw_text[:20_000]
    relative_path = path.relative_to(root_path)
    title = path.stem.strip()
    path_digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    sha256, hash_note = _sha256_or_note(path)
    document_id = (
        f"doc_{sha256[:12]}_{path_digest}" if sha256 else f"doc_unreadable_{path_digest}"
    )

    return SourceDocument(
        id=document_id,
        deal_id=deal_id,
        path=relative_path,
        source_url=None,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=classify_document(relative_path, text_sample=text_sample),
        file_type=classify_file_type(path),
        title=title,
        company_name=None,
        created_at=None,
        ingested_at=ingested_at,
        retrieved_at=None,
        page_count=extraction.page_count,
        sha256=sha256,
        confidentiality_detected=_has_confidentiality_marker(raw_text_sample),
        extraction_quality=extraction.extraction_quality,
        notes=_join_notes(extraction.notes, hash_note),
    )


def _write_document(
    config: AppConfig,
    deal_id: str,
    document: SourceDocument,
    extraction: ExtractionResult,
) -> Path:
    output_dir = config.data_dir / "processed" / "deals" / deal_id / "documents"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document.id}.json"

    payload = IngestedDocument(
        source=document,
        pages=extraction.pages,
        output_path=output_path,
    )
    output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def _write_summary(summary: IngestionSummary) -> None:
    summary.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_or_note(path: Path) -> tuple[str | None, str | None]:
    try:
        return _sha256(path), None
    except OSError as exc:
        return None, f"Could not calculate the file fingerprint: {exc}"


def _join_notes(*notes: str | None) -> str | None:
    present_notes = [note for note in notes if note]
    if not present_notes:
        return None
    return " ".join(present_notes)


def _has_confidentiality_marker(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ["confidential", "private placement", "not for distribution"]
    )
