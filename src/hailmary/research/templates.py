from __future__ import annotations

import json
import os
import re
import secrets
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import DocumentType, SourceKind

from .schemas import ResearchPlan, ResearchResultsTemplateRunSummary, ResearchTask


class ResearchTemplateError(RuntimeError):
    """A research results template could not be prepared safely."""


def prepare_research_results_template(
    *,
    config: AppConfig,
    plan_path: Path | None = None,
    created_at: datetime | None = None,
) -> ResearchResultsTemplateRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise ResearchTemplateError(str(exc)) from exc

    created_at = _as_utc(created_at or datetime.now(UTC))
    resolved_plan_path = _resolve_plan_path(plan_path, data_dir=config.data_dir)
    plan = _load_research_plan(resolved_plan_path)
    results = [_template_result_for_task(task) for task in plan.tasks]

    output_dir = config.data_dir / "research-results-templates"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_template_path(output_dir, created_at)
    _write_private_json(
        output_path,
        json.dumps({"results": results}, indent=2),
        description="research results template",
    )
    return ResearchResultsTemplateRunSummary(
        output_path=output_path,
        plan_path=resolved_plan_path,
        created_at=created_at,
        result_count=len(results),
    )


def _resolve_plan_path(plan_path: Path | None, *, data_dir: Path) -> Path:
    if plan_path is None:
        return _latest_research_plan_path(data_dir)
    return _resolve_input_file(plan_path, description="research plan")


def _latest_research_plan_path(data_dir: Path) -> Path:
    plans_dir = data_dir / "research-plans"
    if plans_dir.is_symlink():
        raise ResearchTemplateError("The research plans folder cannot be a symlink.")
    if not plans_dir.exists():
        raise ResearchTemplateError(
            "No research plans were found. Run `hailmary prepare-research-plan` first."
        )
    if not plans_dir.is_dir():
        raise ResearchTemplateError("The research plans path is not a folder.")
    candidates = [
        path
        for path in plans_dir.glob("*.json")
        if path.is_file() and not path.is_symlink()
    ]
    if not candidates:
        raise ResearchTemplateError(
            "No research plan JSON files were found. Run `hailmary prepare-research-plan` first."
        )
    return max(candidates, key=_research_plan_sort_key).resolve(strict=False)


def _research_plan_sort_key(path: Path) -> tuple[str, int]:
    match = re.fullmatch(
        r"research-plan-(\d{8}-\d{6})(?:-(\d+))?\.json",
        path.name,
    )
    if match is None:
        return (path.name, 0)
    suffix = int(match.group(2) or "1")
    return (match.group(1), suffix)


def _resolve_input_file(path: Path, *, description: str) -> Path:
    expanded_path = path.expanduser()
    absolute_path = expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path
    if absolute_path.is_symlink():
        raise ResearchTemplateError(f"The {description} file cannot be a symlink.")
    for parent in absolute_path.parents:
        if parent.is_symlink():
            raise ResearchTemplateError(
                f"Hail Mary cannot read {path} because {parent} is a symlinked parent folder."
            )
    resolved_path = absolute_path.resolve(strict=False)
    if not resolved_path.exists():
        raise ResearchTemplateError(f"The {description} file does not exist: {path}")
    if not resolved_path.is_file():
        raise ResearchTemplateError(f"The {description} path is not a file: {path}")
    return resolved_path


def _load_research_plan(path: Path) -> ResearchPlan:
    try:
        raw_plan = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchTemplateError(
            "The research plan is not plain UTF-8 text. Run `hailmary prepare-research-plan` again."
        ) from exc
    except OSError as exc:
        raise ResearchTemplateError(f"Could not read the research plan at {path}: {exc}") from exc
    try:
        plan = ResearchPlan.model_validate_json(raw_plan)
    except ValidationError as exc:
        raise ResearchTemplateError(
            "The research plan could not be read. Run `hailmary prepare-research-plan` again."
        ) from exc
    if not plan.tasks:
        raise ResearchTemplateError(
            "The research plan does not contain any tasks. Run "
            "`hailmary prepare-research-plan` again."
        )
    return plan


def _template_result_for_task(task: ResearchTask) -> dict[str, str]:
    return {
        "deal_id": task.deal_id,
        "company_name": task.company_name,
        "provider_id": task.provider_id,
        "provider_name": task.provider_name,
        "title": "",
        "text": "",
        "retrieved_at": "",
        "source_url": "",
        "source_api": "",
        "confidence": "",
        "licensing_notes": task.licensing_notes,
        "source_kind": task.source_kind.value,
        "document_type": _document_type_for_task(task).value,
    }


def _document_type_for_task(task: ResearchTask) -> DocumentType:
    if task.source_kind == SourceKind.MERIDIAN:
        return DocumentType.PLATFORM_DEAL_PAGE
    return DocumentType.WEB_PAGE


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise ResearchTemplateError(
            f"Research results template folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise ResearchTemplateError(f"Research results template folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise ResearchTemplateError(
            f"Could not create research results template folder at {path}: {exc}"
        ) from exc


def _unique_template_path(output_dir: Path, created_at: datetime) -> Path:
    base_name = f"research-results-template-{created_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise ResearchTemplateError(
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
        raise ResearchTemplateError(
            f"Could not write {description} at {path}: the template contains text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise ResearchTemplateError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        with suppress(OSError):
            temp_path.unlink()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
