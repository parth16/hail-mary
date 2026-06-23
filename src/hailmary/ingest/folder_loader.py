from __future__ import annotations

import hashlib
import os
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


class IngestionError(RuntimeError):
    """The scan could not safely write generated ingestion output."""


def ingest_folder(root_path: Path, *, config: AppConfig) -> IngestionSummary:
    """Scan a local folder and store extracted document metadata."""

    root_path = root_path.expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"The folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"This is not a folder: {root_path}")

    run_started_at = datetime.now(UTC)
    deals_by_id: dict[str, IngestedDeal] = {}
    skipped_files: list[str] = []
    candidate_files: list[tuple[Path, str]] = []

    for path in sorted(root_path.rglob("*")):
        relative_path = path.relative_to(root_path)
        if path.is_symlink():
            skipped_files.append(str(relative_path))
            continue
        if not path.is_file():
            continue
        if (
            is_ignored_path(relative_path)
            or _is_generated_output_path(path, config)
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            skipped_files.append(str(relative_path))
            continue

        deal_name = _deal_name_for_path(relative_path)
        candidate_files.append((path, deal_name))

    deal_ids_by_name = {
        deal_name: _deal_id_for_name(deal_name)
        for deal_name in {deal_name for _, deal_name in candidate_files}
    }

    for path, deal_name in candidate_files:
        deal_id = deal_ids_by_name[deal_name]
        extraction = extract_document(path)
        document = _build_document(
            path,
            root_path,
            deal_id,
            deal_name,
            extraction,
            run_started_at,
        )

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


def _deal_id_for_name(deal_name: str) -> str:
    base_deal_id = slugify(deal_name)
    digest = hashlib.sha256(deal_name.encode("utf-8")).hexdigest()[:8]
    return f"{base_deal_id}-{digest}"


def _build_document(
    path: Path,
    root_path: Path,
    deal_id: str,
    deal_name: str,
    extraction: ExtractionResult,
    ingested_at: datetime,
) -> SourceDocument:
    text_sample = extraction.combined_text[:4_000]
    relative_path = path.relative_to(root_path)
    title = path.stem.strip()
    path_digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    sha256, hash_note = _sha256_or_note(path)
    document_id = (
        f"doc_{sha256[:12]}_{path_digest}" if sha256 else f"doc_unreadable_{path_digest}"
    )
    confidentiality_text = f"{relative_path.as_posix()} {extraction.combined_raw_text}"

    return SourceDocument(
        id=document_id,
        deal_id=deal_id,
        path=relative_path,
        source_url=None,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=classify_document(relative_path, text_sample=text_sample),
        file_type=classify_file_type(path),
        title=title,
        company_name=deal_name,
        created_at=None,
        ingested_at=ingested_at,
        retrieved_at=None,
        page_count=extraction.page_count,
        sha256=sha256,
        confidentiality_detected=_has_confidentiality_marker(confidentiality_text),
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
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = output_dir / f"{document.id}.json"

    payload = IngestedDocument(
        source=document,
        pages=extraction.pages,
        output_path=output_path,
    )
    _write_private_text(
        output_path,
        payload.model_dump_json(indent=2),
        description="document output",
    )
    return output_path


def _write_summary(summary: IngestionSummary) -> None:
    _ensure_private_directory(
        summary.summary_path.parent,
        private_root=summary.summary_path.parents[1],
    )
    _write_private_text(
        summary.summary_path,
        summary.model_dump_json(indent=2),
        description="scan summary",
    )


def _is_generated_output_path(path: Path, config: AppConfig) -> bool:
    resolved_path = path.resolve(strict=False)
    generated_roots = [
        config.data_dir,
        config.meridian_profile_dir,
    ]
    for generated_root in generated_roots:
        resolved_root = (
            generated_root if generated_root.is_absolute() else Path.cwd() / generated_root
        ).resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
            return True
        except ValueError:
            continue
    return False


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    resolved_root = (
        private_root if private_root.is_absolute() else Path.cwd() / private_root
    ).resolve(strict=False)
    _reject_output_symlink_escape(path, resolved_root=resolved_root)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IngestionError(f"Could not create private output folder at {path}: {exc}") from exc

    resolved_path = path.resolve(strict=False)
    try:
        relative_parts = resolved_path.relative_to(resolved_root).parts
    except ValueError:
        raise IngestionError(
            f"Output folder {path} resolves outside the private data directory."
        ) from None

    directories = [resolved_root]
    current = resolved_root
    for part in relative_parts:
        current = current / part
        directories.append(current)

    for directory in directories:
        try:
            directory.chmod(0o700)
        except OSError as exc:
            raise IngestionError(
                f"Could not make private output folder at {directory}: {exc}"
            ) from exc


def _reject_output_symlink_escape(path: Path, *, resolved_root: Path) -> None:
    absolute_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        relative_parts = absolute_path.relative_to(resolved_root).parts
    except ValueError:
        raise IngestionError(
            f"Output folder {path} is outside the private data directory."
        ) from None

    current = resolved_root
    if current.is_symlink():
        raise IngestionError(
            f"Output folder {path} uses a symlinked private data directory."
        )

    for part in relative_parts:
        current = current / part
        if not current.is_symlink():
            continue
        try:
            current.resolve(strict=True).relative_to(resolved_root)
        except ValueError:
            raise IngestionError(
                f"Output folder {path} resolves outside the private data directory."
            ) from None


def _write_private_text(path: Path, text: str, *, description: str) -> None:
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(
            path,
            flags,
            0o600,
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        path.chmod(0o600)
    except OSError as exc:
        raise IngestionError(f"Could not write {description} at {path}: {exc}") from exc


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
