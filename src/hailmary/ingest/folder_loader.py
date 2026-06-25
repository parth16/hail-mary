from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hailmary.config import AppConfig
from hailmary.evidence import build_evidence_store
from hailmary.ingest.document_classifier import (
    SUPPORTED_SUFFIXES,
    classify_document,
    classify_file_type,
    is_ignored_path,
)
from hailmary.ingest.extractors import ExtractionResult, extract_document
from hailmary.ingest.ocr import LocalOcrEngine, SubprocessLocalOcrEngine
from hailmary.schemas.documents import (
    IngestedDeal,
    IngestedDocument,
    IngestionSummary,
    SourceDocument,
    SourceKind,
)
from hailmary.schemas.evidence import EvidenceStore
from hailmary.utils.slug import slugify


class IngestionError(RuntimeError):
    """The scan could not safely write generated ingestion output."""


@dataclass(frozen=True)
class DealFolderInspection:
    root_path: Path
    deal_names: tuple[str, ...]
    readable_document_count: int
    skipped_files: tuple[str, ...]
    unreadable_paths: tuple[str, ...]


def inspect_deal_folder(root_path: Path, *, config: AppConfig) -> DealFolderInspection:
    """Inspect local input using the same document-selection rules as ingestion."""

    resolved_root = _resolve_scan_root(root_path)
    candidate_files, skipped_files, unreadable_paths = _candidate_documents(
        resolved_root,
        config=config,
    )
    is_private_raw_collection_root = _is_private_raw_collection_root(
        resolved_root,
        config,
    )
    use_collection_subfolders = _uses_collection_subfolders(
        resolved_root,
        is_private_raw_collection_root=is_private_raw_collection_root,
    )
    deal_names_set = {deal_name for _, deal_name in candidate_files}
    deal_names_set.update(
        _unreadable_deal_names(
            unreadable_paths,
            root_path=resolved_root,
            use_collection_subfolders=use_collection_subfolders,
        )
    )
    deal_names = tuple(sorted(deal_names_set))
    return DealFolderInspection(
        root_path=resolved_root,
        deal_names=deal_names,
        readable_document_count=len(candidate_files),
        skipped_files=tuple(skipped_files),
        unreadable_paths=tuple(unreadable_paths),
    )


def ingest_folder(
    root_path: Path,
    *,
    config: AppConfig,
    ocr_engine: LocalOcrEngine | None = None,
) -> IngestionSummary:
    """Scan a local folder and store extracted document metadata."""

    root_path = _resolve_scan_root(root_path)
    active_ocr_engine = ocr_engine
    if active_ocr_engine is None and config.enable_ocr:
        active_ocr_engine = SubprocessLocalOcrEngine()
    run_started_at = datetime.now(UTC)
    deals_by_id: dict[str, IngestedDeal] = {}
    candidate_files, skipped_files, unreadable_paths = _candidate_documents(
        root_path,
        config=config,
    )

    deal_ids_by_name = {
        deal_name: _deal_id_for_name(deal_name)
        for deal_name in {deal_name for _, deal_name in candidate_files}
    }

    for path, deal_name in candidate_files:
        deal_id = deal_ids_by_name[deal_name]
        extraction = extract_document(path, ocr_engine=active_ocr_engine)
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
            tables=extraction.tables,
            output_path=_write_document(config, deal_id, document, extraction),
        )

        if deal_id not in deals_by_id:
            deals_by_id[deal_id] = IngestedDeal(
                id=deal_id,
                company_name=deal_name,
                documents=[],
            )
        deals_by_id[deal_id].documents.append(processed_document)

    for deal in deals_by_id.values():
        evidence_store = build_evidence_store(deal, created_at=run_started_at)
        deal.evidence_store_path = _write_evidence_store(config, deal.id, evidence_store)
        deal.evidence_count = evidence_store.evidence_count
        deal.claim_count = evidence_store.claim_count
        deal.conflict_count = evidence_store.conflict_count

    summary = IngestionSummary(
        root_path=root_path,
        scanned_at=run_started_at,
        deals=list(deals_by_id.values()),
        skipped_files=skipped_files,
        unreadable_paths=unreadable_paths,
        summary_path=config.data_dir / "processed" / "ingestion_summary.json",
    )
    _write_summary(summary)
    return summary


def _candidate_documents(
    root_path: Path,
    *,
    config: AppConfig,
) -> tuple[list[tuple[Path, str]], list[str], list[str]]:
    allow_private_raw_root = _is_private_raw_root(root_path, config)
    is_private_raw_collection_root = _is_private_raw_collection_root(root_path, config)
    use_collection_subfolders = _uses_collection_subfolders(
        root_path,
        is_private_raw_collection_root=is_private_raw_collection_root,
    )
    skipped_files: list[str] = []
    unreadable_paths: list[str] = []
    candidate_files: list[tuple[Path, str]] = []

    scan_paths, unreadable_dirs = _scan_input_paths(root_path)
    for path in unreadable_dirs:
        try:
            relative_path = path.relative_to(root_path)
        except ValueError:
            unreadable_paths.append(_relative_display_path(path, root_path))
            continue
        if is_ignored_path(relative_path) or _is_generated_output_path(
            path,
            config,
            allow_private_raw=allow_private_raw_root,
        ):
            continue
        unreadable_paths.append(str(relative_path))

    for path in scan_paths:
        relative_path = path.relative_to(root_path)
        if path.is_symlink():
            skipped_files.append(str(relative_path))
            continue
        if not path.is_file():
            continue
        if (
            is_ignored_path(relative_path)
            or _is_generated_output_path(
                path,
                config,
                allow_private_raw=allow_private_raw_root,
            )
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            skipped_files.append(str(relative_path))
            continue
        if not os.access(path, os.R_OK):
            unreadable_paths.append(str(relative_path))
            continue

        deal_name = _deal_name_for_path(
            relative_path,
            root_path=root_path,
            use_collection_subfolders=use_collection_subfolders,
        )
        candidate_files.append((path, deal_name))

    return candidate_files, skipped_files, unreadable_paths


def _unreadable_deal_names(
    unreadable_paths: list[str],
    *,
    root_path: Path,
    use_collection_subfolders: bool,
) -> list[str]:
    deal_names: list[str] = []
    for display_path in unreadable_paths:
        relative_path = Path(display_path)
        if relative_path.is_absolute() or not relative_path.parts:
            continue
        deal_name = _deal_name_for_path(
            relative_path,
            root_path=root_path,
            use_collection_subfolders=use_collection_subfolders,
        )
        if deal_name not in deal_names:
            deal_names.append(deal_name)
    return deal_names


def _scan_input_paths(root_path: Path) -> tuple[list[Path], list[Path]]:
    paths: list[Path] = []
    unreadable_dirs: list[Path] = []

    def record_unreadable(error: OSError) -> None:
        if error.filename:
            unreadable_dirs.append(Path(error.filename))

    for current_dir, dir_names, file_names in os.walk(
        root_path,
        topdown=True,
        onerror=record_unreadable,
        followlinks=False,
    ):
        current_path = Path(current_dir)
        dir_names.sort()
        file_names.sort()

        traversable_dir_names: list[str] = []
        for dir_name in dir_names:
            dir_path = current_path / dir_name
            if dir_path.is_symlink():
                paths.append(dir_path)
                continue
            traversable_dir_names.append(dir_name)
        dir_names[:] = traversable_dir_names

        paths.extend(current_path / file_name for file_name in file_names)

    return paths, unreadable_dirs


def _relative_display_path(path: Path, root_path: Path) -> str:
    try:
        return str(path.relative_to(root_path))
    except ValueError:
        return str(path)


def _resolve_scan_root(root_path: Path) -> Path:
    expanded_path = root_path.expanduser()
    absolute_path = expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path

    if absolute_path.is_symlink():
        raise IngestionError("The scan folder cannot be a symlink. Choose a real folder.")
    for parent in absolute_path.parents:
        if parent.is_symlink():
            raise IngestionError(
                f"Hail Mary cannot scan {expanded_path} because {parent} is a symlinked "
                "parent folder."
            )

    resolved_path = absolute_path.resolve(strict=False)
    if not resolved_path.exists():
        raise FileNotFoundError(f"The folder does not exist: {expanded_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"This is not a folder: {expanded_path}")
    return resolved_path


def _uses_collection_subfolders(
    root_path: Path,
    *,
    is_private_raw_collection_root: bool,
) -> bool:
    collection_folder_names = {
        "companies",
        "deal-documents",
        "deal-docs",
        "deals",
        "decks",
        "diligence",
        "diligence-documents",
        "diligence-docs",
        "documents",
        "investment-documents",
        "investment-docs",
        "pitch-decks",
        "startup-documents",
        "startup-docs",
        "startups",
    }
    normalized_root_name = slugify(root_path.name)
    return is_private_raw_collection_root or normalized_root_name in collection_folder_names


def _deal_name_for_path(
    relative_path: Path,
    *,
    root_path: Path,
    use_collection_subfolders: bool,
) -> str:
    if use_collection_subfolders and len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return root_path.name


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
    source_created_at, timestamp_note = _file_modified_at_or_note(path)

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
        created_at=source_created_at,
        ingested_at=ingested_at,
        retrieved_at=None,
        page_count=extraction.page_count,
        table_count=extraction.table_count,
        sha256=sha256,
        confidentiality_detected=_has_confidentiality_marker(confidentiality_text),
        extraction_quality=extraction.extraction_quality,
        ocr_recommended=extraction.ocr_recommended,
        ocr_applied=extraction.ocr_applied,
        ocr_confidence=extraction.ocr_confidence,
        vision_recommended=extraction.vision_recommended,
        notes=_join_notes(extraction.notes, hash_note, timestamp_note),
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
        tables=extraction.tables,
        output_path=output_path,
    )
    _write_private_text(
        output_path,
        payload.model_dump_json(indent=2),
        description="document output",
    )
    return output_path


def _write_evidence_store(
    config: AppConfig,
    deal_id: str,
    evidence_store: EvidenceStore,
) -> Path:
    output_dir = config.data_dir / "processed" / "deals" / deal_id
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = output_dir / "evidence_store.json"
    _write_private_text(
        output_path,
        evidence_store.model_dump_json(indent=2),
        description="evidence store",
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


def _is_generated_output_path(
    path: Path,
    config: AppConfig,
    *,
    allow_private_raw: bool = False,
) -> bool:
    resolved_path = path.resolve(strict=False)
    generated_roots = [
        config.data_dir,
        config.data_dir / "processed",
        config.data_dir / "reports",
        config.data_dir / "agent-packets",
        config.data_dir / "agent-outputs",
        config.data_dir / "research-plans",
        config.data_dir / "research-results",
        config.data_dir / "research-results-templates",
        config.data_dir / "browser-profiles",
        config.data_dir / "meridian-workflows",
        config.meridian_profile_dir,
    ]
    for generated_root in generated_roots:
        if allow_private_raw and generated_root == config.data_dir:
            continue
        resolved_root = (
            generated_root if generated_root.is_absolute() else Path.cwd() / generated_root
        ).resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_root)
            return True
        except ValueError:
            continue
    return False


def _is_private_raw_root(root_path: Path, config: AppConfig) -> bool:
    raw_root = _private_raw_root(config)
    try:
        root_path.relative_to(raw_root)
    except ValueError:
        return False
    return True


def _is_private_raw_collection_root(root_path: Path, config: AppConfig) -> bool:
    return root_path == _private_raw_root(config)


def _private_raw_root(config: AppConfig) -> Path:
    raw_root = (
        config.data_dir / "raw"
        if config.data_dir.is_absolute()
        else Path.cwd() / config.data_dir / "raw"
    )
    return raw_root.resolve(strict=False)


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
    if path.is_symlink():
        raise IngestionError(f"Could not write {description} at {path}: output file is a symlink.")
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


def _file_modified_at_or_note(path: Path) -> tuple[datetime | None, str | None]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC), None
    except OSError as exc:
        return None, f"Could not read the file modified time: {exc}"


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
