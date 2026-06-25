from __future__ import annotations

import json
import os
import secrets
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import DocumentType, SourceKind

from .providers import builtin_research_providers
from .schemas import ResearchProvider


class MeridianWorkflowError(RuntimeError):
    """A Meridian manual workflow could not be prepared safely."""


class MeridianWorkflow(BaseModel):
    version: str = "1"
    created_at: datetime
    company_name: str
    meridian_url: str
    local_only: bool
    result_template_path: Path
    safety_rules: list[str] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    import_command: str


class MeridianWorkflowRunSummary(BaseModel):
    output_path: Path
    result_template_path: Path
    workflow: MeridianWorkflow


def prepare_meridian_workflow(
    *,
    config: AppConfig,
    company_name: str,
    meridian_url: str,
    created_at: datetime | None = None,
) -> MeridianWorkflowRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise MeridianWorkflowError(str(exc)) from exc

    created_at = _as_utc(created_at or datetime.now(UTC))
    cleaned_company_name = _clean_company_name(company_name)
    cleaned_meridian_url = clean_meridian_url(meridian_url)
    provider = _meridian_provider()

    template_dir = config.data_dir / "research-results-templates"
    workflow_dir = config.data_dir / "meridian-workflows"
    _ensure_private_directory(template_dir, private_root=config.data_dir)
    _ensure_private_directory(workflow_dir, private_root=config.data_dir)

    template_path = _unique_template_path(template_dir, created_at)
    workflow_path = _unique_workflow_path(workflow_dir, created_at)
    template_payload = {
        "results": [
            {
                "deal_id": "",
                "company_name": cleaned_company_name,
                "provider_id": provider.id,
                "provider_name": provider.name,
                "title": "",
                "text": "",
                "retrieved_at": "",
                "source_url": cleaned_meridian_url,
                "source_api": "",
                "confidence": "",
                "licensing_notes": provider.licensing_notes,
                "source_kind": SourceKind.MERIDIAN.value,
                "document_type": DocumentType.PLATFORM_DEAL_PAGE.value,
            }
        ]
    }
    workflow = MeridianWorkflow(
        created_at=created_at,
        company_name=cleaned_company_name,
        meridian_url=cleaned_meridian_url,
        local_only=config.local_only,
        result_template_path=template_path,
        safety_rules=_safety_rules(),
        manual_steps=_manual_steps(),
        import_command=(
            f"hailmary import-research-results {template_path} "
            f"--data-dir {config.data_dir} --dry-run"
        ),
    )

    _write_private_json(
        template_path,
        json.dumps(template_payload, indent=2),
        description="Meridian results template",
    )
    _write_private_json(
        workflow_path,
        workflow.model_dump_json(indent=2),
        description="Meridian manual workflow",
    )
    return MeridianWorkflowRunSummary(
        output_path=workflow_path,
        result_template_path=template_path,
        workflow=workflow,
    )


def _clean_company_name(company_name: str) -> str:
    cleaned = company_name.strip()
    if not cleaned:
        raise MeridianWorkflowError("Company name cannot be blank.")
    return cleaned


def clean_meridian_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise MeridianWorkflowError("Meridian URL cannot be blank.")
    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        raise MeridianWorkflowError("The Meridian URL is not a valid URL.") from exc
    if parsed.scheme != "https":
        raise MeridianWorkflowError("The Meridian URL must start with https://.")
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise MeridianWorkflowError("The Meridian URL is not a valid URL.") from exc
    if not parsed.netloc or host is None:
        raise MeridianWorkflowError("The Meridian URL must include a website host.")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise MeridianWorkflowError("The Meridian URL has an invalid port.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise MeridianWorkflowError(
            "The Meridian URL cannot include a username or password."
        )
    if any(character.isspace() for character in cleaned):
        raise MeridianWorkflowError("The Meridian URL cannot contain spaces.")
    if ";" in parsed.path or parsed.params or parsed.query or parsed.fragment:
        raise MeridianWorkflowError(
            "The Meridian URL cannot include extra text after ;, ?, or #. "
            "Use the base deal page URL."
        )
    if host.casefold() != "portal.angellist.com":
        raise MeridianWorkflowError(
            "The Meridian URL must use the portal.angellist.com website host."
        )

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 3 or path_parts[0] != "m" or path_parts[-1] != "invest":
        raise MeridianWorkflowError(
            "The Meridian URL must look like a Meridian deal page, such as "
            "https://portal.angellist.com/m/example/invest."
        )
    return cleaned


def _meridian_provider() -> ResearchProvider:
    providers = {
        provider.id: provider
        for provider in builtin_research_providers(include_meridian=True)
    }
    provider = providers.get("meridian")
    if provider is None:
        raise MeridianWorkflowError("The Meridian research provider is not configured.")
    return provider


def _safety_rules() -> list[str]:
    return [
        "Use normal authenticated access only.",
        (
            "Do not bypass login, CAPTCHA or other human checks, two-factor sign-in, "
            "paywalls, or platform restrictions."
        ),
        (
            "Do not save browser cookies, tokens, signed links, screenshots, "
            "or raw portal pages in the repository."
        ),
        "Copy only facts and short excerpts you are allowed to save locally.",
        (
            "Every completed result row must keep the generated Meridian page URL "
            "and include the time viewed, confidence, and licensing notes."
        ),
    ]


def _manual_steps() -> list[str]:
    return [
        "Open the Meridian URL in your own authenticated browser session.",
        "Complete normal sign-in and access checks yourself.",
        "Review the page and collect only permitted facts tied to page text.",
        (
            "Paste each fact into the results template. Keep source_url as the "
            "generated Meridian page URL for completed rows."
        ),
        "Run the dry-run import command before importing evidence.",
    ]


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise MeridianWorkflowError(
            f"Meridian workflow folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise MeridianWorkflowError(f"Meridian workflow folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise MeridianWorkflowError(
            f"Could not create Meridian workflow folder at {path}: {exc}"
        ) from exc


def _unique_template_path(output_dir: Path, created_at: datetime) -> Path:
    return _unique_output_path(
        output_dir,
        prefix="meridian-results-template",
        created_at=created_at,
    )


def _unique_workflow_path(output_dir: Path, created_at: datetime) -> Path:
    return _unique_output_path(
        output_dir,
        prefix="meridian-workflow",
        created_at=created_at,
    )


def _unique_output_path(output_dir: Path, *, prefix: str, created_at: datetime) -> Path:
    base_name = f"{prefix}-{created_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise MeridianWorkflowError(
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
        raise MeridianWorkflowError(
            f"Could not write {description} at {path}: the file contains text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise MeridianWorkflowError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        with suppress(OSError):
            temp_path.unlink()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
