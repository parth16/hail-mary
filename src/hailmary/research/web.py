from __future__ import annotations

import errno
import http.client
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import DocumentType, SourceKind

from .matching import CompanyMatchKind
from .quality import source_reliability_for_provider
from .schemas import (
    ResearchPlan,
    ResearchProviderCategory,
    ResearchResultInput,
    ResearchResultsFile,
    ResearchTask,
    ResearchTaskStatus,
)
from .source_urls import validate_provider_source_url


class WebResearchError(RuntimeError):
    """Public web research could not be collected safely."""


class WebResearchFetchError(RuntimeError):
    """One public web page could not be fetched safely."""


MAX_FETCH_BYTES = 2_000_000
MAX_TEXT_CHARS = 40_000
DEFAULT_TIMEOUT_SECONDS = 10.0
GENERATED_SEARCH_PAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "sec_form_d": ("https://www.sec.gov/edgar/search/",),
    "sam_gov": ("https://sam.gov/search/", "https://www.sam.gov/search/"),
    "usaspending": (
        "https://www.usaspending.gov/search/",
        "https://usaspending.gov/search/",
    ),
    "sbir": ("https://www.sbir.gov/award?", "https://sbir.gov/award?"),
    "uspto": (
        "https://tmsearch.uspto.gov/search/search-results?",
        "https://www.tmsearch.uspto.gov/search/search-results?",
    ),
    "github": ("https://github.com/search?",),
}


class WebFetchResponse(BaseModel):
    final_url: str
    content_type: str
    text: str
    status_code: int = 200


class WebResearchTaskSummary(BaseModel):
    company_name: str
    provider_id: str
    provider_name: str
    url: str | None = None
    status: str
    reason: str


class WebResearchRunSummary(BaseModel):
    output_path: Path | None = None
    plan_path: Path
    collected_at: datetime
    dry_run: bool = False
    tasks: list[WebResearchTaskSummary] = Field(default_factory=list)

    @property
    def fetched_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "fetched")

    @property
    def planned_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "planned")

    @property
    def failed_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "failed")

    @property
    def skipped_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "skipped")


class WebResearchClient(Protocol):
    def fetch(
        self,
        url: str,
        *,
        provider_id: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        """Fetch one public URL and return decoded page text."""


@dataclass(frozen=True)
class UrlLibWebResearchClient:
    user_agent: str = "HailMary/0.1 public web research"

    def fetch(
        self,
        url: str,
        *,
        provider_id: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        _validate_public_fetch_url(url, provider_id=provider_id)
        _ensure_resolved_public_host(url)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,text/plain,application/xhtml+xml",
            },
        )
        opener = _build_guarded_opener(provider_id)
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_public_fetch_url(final_url, provider_id=provider_id)
                _ensure_resolved_public_host(final_url)
                content_type = response.headers.get("Content-Type", "")
                raw_content = response.read(max_bytes + 1)
                if len(raw_content) > max_bytes:
                    raise WebResearchFetchError(
                        "The page is larger than Hail Mary's public web research limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            raise WebResearchFetchError(
                f"The website returned HTTP {exc.code}."
            ) from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise WebResearchFetchError(f"Could not fetch the page: {exc}") from exc
        try:
            decoded_text = raw_content.decode(charset, errors="replace")
        except LookupError:
            decoded_text = raw_content.decode("utf-8", errors="replace")
        return WebFetchResponse(
            final_url=final_url,
            content_type=content_type,
            text=decoded_text,
            status_code=status_code,
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, provider_id: str) -> None:
        super().__init__()
        self.provider_id = provider_id

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_public_fetch_url(newurl, provider_id=self.provider_id)
        _ensure_resolved_public_host(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not isinstance(redirected, urllib.request.Request):
            raise WebResearchFetchError("The website returned an unsupported redirect.")
        return redirected


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, provider_id: str) -> None:
        super().__init__()
        self.provider_id = provider_id

    def http_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_connection_factory(self.provider_id, secure=False), req)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, provider_id: str) -> None:
        super().__init__()
        self.provider_id = provider_id

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(_connection_factory(self.provider_id, secure=True), req)


class _ConnectionWithSocketFactory(Protocol):
    host: str
    port: int
    timeout: float | None
    source_address: object
    _create_connection: Callable[[tuple[str, int], float | None, object], socket.socket]


class _BoundHTTPConnection(http.client.HTTPConnection):
    _create_connection: Callable[[tuple[str, int], float | None, object], socket.socket]
    source_address: object

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        provider_id: str,
        **kwargs: Any,
    ) -> None:
        self._provider_id = provider_id
        super().__init__(host, port=port, **kwargs)

    def connect(self) -> None:
        self.sock = _open_vetted_socket(self)


class _BoundHTTPSConnection(http.client.HTTPSConnection):
    _context: Any
    _create_connection: Callable[[tuple[str, int], float | None, object], socket.socket]
    source_address: object

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        provider_id: str,
        **kwargs: Any,
    ) -> None:
        self._provider_id = provider_id
        super().__init__(host, port=port, **kwargs)

    def connect(self) -> None:
        self.sock = _open_vetted_socket(self)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _build_guarded_opener(provider_id: str) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPHandler(provider_id),
        _GuardedHTTPSHandler(provider_id),
        _SafeRedirectHandler(provider_id),
    )


def _connection_factory(
    provider_id: str,
    *,
    secure: bool,
) -> Any:
    connection_class = _BoundHTTPSConnection if secure else _BoundHTTPConnection

    def factory(host: str, **kwargs: Any) -> http.client.HTTPConnection:
        return connection_class(host, provider_id=provider_id, **kwargs)

    return factory


def _open_vetted_socket(
    connection: _ConnectionWithSocketFactory,
) -> socket.socket:
    vetted_address = _resolve_public_host_address(connection.host, connection.port)
    sys.audit("http.client.connect", connection, connection.host, connection.port)
    opened_socket = connection._create_connection(
        (vetted_address, connection.port),
        connection.timeout,
        connection.source_address,
    )
    try:
        opened_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as exc:
        if exc.errno != errno.ENOPROTOOPT:
            raise
    return opened_socket


def collect_web_research(
    *,
    config: AppConfig,
    plan_path: Path | None = None,
    provider_ids: list[str] | None = None,
    dry_run: bool = False,
    client: WebResearchClient | None = None,
    collected_at: datetime | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FETCH_BYTES,
) -> WebResearchRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise WebResearchError(str(exc)) from exc
    _ensure_web_research_enabled(config)
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    resolved_plan_path = _resolve_plan_path(plan_path, data_dir=config.data_dir)
    plan = _load_research_plan(resolved_plan_path)
    provider_filter = {provider_id.strip() for provider_id in provider_ids or []}
    if any(not provider_id for provider_id in provider_filter):
        raise WebResearchError("Provider IDs cannot be blank.")
    _ensure_provider_filter_matches_plan(provider_filter, plan)
    ingested_deal_ids = {
        deal.deal_id for deal in plan.deals if deal.from_ingestion
    }
    web_client = client or UrlLibWebResearchClient()

    summaries: list[WebResearchTaskSummary] = []
    results: list[ResearchResultInput] = []
    for task in plan.tasks:
        if provider_filter and task.provider_id not in provider_filter:
            continue
        eligibility = _task_fetch_eligibility(task)
        if eligibility is not None:
            summaries.append(_task_summary(task, status="skipped", reason=eligibility))
            continue
        assert task.url is not None
        try:
            _validate_public_fetch_url(task.url, provider_id=task.provider_id)
        except ValueError as exc:
            summaries.append(_task_summary(task, status="failed", reason=str(exc)))
            continue
        if dry_run:
            summaries.append(
                _task_summary(
                    task,
                    status="planned",
                    reason="Dry run: the page was not fetched.",
                )
            )
            continue
        try:
            fetch_response = web_client.fetch(
                task.url,
                provider_id=task.provider_id,
                timeout_seconds=timeout_seconds,
                max_bytes=max_bytes,
            )
            _validate_public_fetch_url(fetch_response.final_url, provider_id=task.provider_id)
            page = _page_result_from_response(fetch_response, fallback_url=task.url)
            result = _research_result_for_task(
                task,
                page=page,
                collected_at=collected_at,
                ingested_deal_ids=ingested_deal_ids,
            )
        except (ValidationError, ValueError, WebResearchFetchError) as exc:
            summaries.append(_task_summary(task, status="failed", reason=str(exc)))
            continue
        results.append(result)
        summaries.append(
            _task_summary(
                task,
                status="fetched",
                reason=f"Fetched {page.source_url}.",
            )
        )

    if not results:
        return WebResearchRunSummary(
            output_path=None,
            plan_path=resolved_plan_path,
            collected_at=collected_at,
            dry_run=dry_run,
            tasks=summaries,
        )

    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise WebResearchError(
            f"Collected web research did not pass validation: {detail}"
        ) from exc

    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="web research results",
    )
    return WebResearchRunSummary(
        output_path=output_path,
        plan_path=resolved_plan_path,
        collected_at=collected_at,
        dry_run=dry_run,
        tasks=summaries,
    )


@dataclass(frozen=True)
class _FetchedPage:
    source_url: str
    title: str
    text: str


def _ensure_web_research_enabled(config: AppConfig) -> None:
    if config.local_only:
        raise WebResearchError(
            "Local-only mode is on. Set HAILMARY_LOCAL_ONLY=false before collecting web research."
        )
    if not config.enable_web_research:
        raise WebResearchError(
            "Web research is disabled. Set HAILMARY_ENABLE_WEB_RESEARCH=true before "
            "collecting web research."
        )


def _task_fetch_eligibility(task: ResearchTask) -> str | None:
    if task.provider_category != ResearchProviderCategory.FREE_PUBLIC:
        return "Skipped because this source needs manual action or a paid account."
    if task.source_kind != SourceKind.WEB:
        return "Skipped because this source is not public web evidence."
    if task.status != ResearchTaskStatus.PLANNED:
        return "Skipped because this task is marked for manual action."
    if task.url is None:
        return "Skipped because the research plan does not include a direct URL."
    if _is_generated_search_page(task):
        return (
            "Skipped because this generated search page is only a starting point. "
            "Import an exact source page instead."
        )
    return None


def _page_result_from_response(
    response: WebFetchResponse,
    *,
    fallback_url: str,
) -> _FetchedPage:
    content_type = response.content_type.split(";", 1)[0].strip().casefold()
    if content_type and content_type not in {
        "text/html",
        "text/plain",
        "application/xhtml+xml",
    }:
        raise WebResearchFetchError(
            f"The page returned {response.content_type or 'an unsupported content type'}."
        )
    source_url = response.final_url or fallback_url
    if content_type == "text/plain":
        title = _title_from_url(source_url)
        text = _collapse_text(response.text)
    else:
        title, text = _extract_html_title_and_text(response.text, fallback_url=source_url)
    if not text:
        raise WebResearchFetchError("The page did not contain readable text.")
    return _FetchedPage(
        source_url=source_url,
        title=title or _title_from_url(source_url),
        text=_truncate_text(text),
    )


def _extract_html_title_and_text(html: str, *, fallback_url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    title = ""
    if soup.title is not None:
        title = _collapse_text(soup.title.get_text(" ", strip=True))
    text = _collapse_text(soup.get_text(" ", strip=True))
    return title or _title_from_url(fallback_url), text


def _research_result_for_task(
    task: ResearchTask,
    *,
    page: _FetchedPage,
    collected_at: datetime,
    ingested_deal_ids: set[str],
) -> ResearchResultInput:
    return ResearchResultInput(
        deal_id=task.deal_id if task.deal_id in ingested_deal_ids else None,
        company_name=task.company_name,
        provider_id=task.provider_id,
        provider_name=task.provider_name,
        title=page.title,
        text=page.text,
        retrieved_at=collected_at,
        source_url=page.source_url,
        confidence=(
            "medium: fetched from a planned public web URL; Hail Mary did not "
            "verify individual claims"
        ),
        licensing_notes=(
            f"{task.licensing_notes} Automatically fetched from a public URL in the "
            "research plan. Respect the source website's terms before relying on it."
        ),
        source_kind=SourceKind.WEB,
        document_type=DocumentType.WEB_PAGE,
        source_reliability=source_reliability_for_provider(
            task.provider_id,
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        ),
        identity_match_kind=CompanyMatchKind.EXACT,
        identity_match_reason="The page was fetched for the exact requested company in the plan.",
    )


def _validate_public_fetch_url(url: str, *, provider_id: str) -> None:
    validate_provider_source_url(provider_id, url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("source_url cannot use localhost or a local-only website host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("source_url cannot use a private, local, or reserved network address")


def _ensure_resolved_public_host(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        raise WebResearchFetchError("The page URL does not include a website host.")
    _resolve_public_host_address(host, _port_for_url(url))


def _resolve_public_host_address(host: str, port: int) -> str:
    try:
        address_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebResearchFetchError(f"Could not resolve website host {host}.") from exc
    vetted_addresses: list[str] = []
    for address_info in address_infos:
        sockaddr = address_info[4]
        raw_address = str(sockaddr[0])
        address = ipaddress.ip_address(raw_address)
        if not address.is_global:
            raise WebResearchFetchError(
                f"Website host {host} resolves to a private, local, or reserved network address."
            )
        if raw_address not in vetted_addresses:
            vetted_addresses.append(raw_address)
    if not vetted_addresses:
        raise WebResearchFetchError(f"Could not resolve website host {host}.")
    return vetted_addresses[0]


def _port_for_url(url: str) -> int:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebResearchFetchError("The page URL has an invalid port.") from exc
    if port is not None:
        return port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    raise WebResearchFetchError("The page URL must start with http:// or https://.")


def _ensure_provider_filter_matches_plan(
    provider_filter: set[str],
    plan: ResearchPlan,
) -> None:
    if not provider_filter:
        return
    plan_provider_ids = {task.provider_id for task in plan.tasks}
    unmatched_ids = sorted(provider_filter - plan_provider_ids)
    if not unmatched_ids:
        return
    available = ", ".join(sorted(plan_provider_ids)) or "none"
    requested = ", ".join(unmatched_ids)
    raise WebResearchError(
        f"Provider filter {requested} does not match this research plan. "
        f"Available providers: {available}."
    )


def _is_generated_search_page(task: ResearchTask) -> bool:
    prefixes = GENERATED_SEARCH_PAGE_PREFIXES.get(task.provider_id)
    if prefixes is None or task.url is None:
        return False
    normalized_url = task.url.casefold()
    return any(normalized_url.startswith(prefix.casefold()) for prefix in prefixes)


def _task_summary(task: ResearchTask, *, status: str, reason: str) -> WebResearchTaskSummary:
    return WebResearchTaskSummary(
        company_name=task.company_name,
        provider_id=task.provider_id,
        provider_name=task.provider_name,
        url=task.url,
        status=status,
        reason=reason,
    )


def _resolve_plan_path(plan_path: Path | None, *, data_dir: Path) -> Path:
    if plan_path is None:
        return _latest_research_plan_path(data_dir)
    return _resolve_input_file(plan_path, description="research plan")


def _latest_research_plan_path(data_dir: Path) -> Path:
    plans_dir = data_dir / "research-plans"
    if plans_dir.is_symlink():
        raise WebResearchError("The research plans folder cannot be a symlink.")
    if not plans_dir.exists():
        raise WebResearchError(
            "No research plans were found. Run `hailmary prepare-research-plan` first."
        )
    if not plans_dir.is_dir():
        raise WebResearchError("The research plans path is not a folder.")
    candidates = [
        path
        for path in plans_dir.glob("*.json")
        if path.is_file() and not path.is_symlink()
    ]
    if not candidates:
        raise WebResearchError(
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
        raise WebResearchError(f"The {description} file cannot be a symlink.")
    for parent in absolute_path.parents:
        if parent.is_symlink():
            raise WebResearchError(
                f"Hail Mary cannot read {path} because {parent} is a symlinked parent folder."
            )
    resolved_path = absolute_path.resolve(strict=False)
    if not resolved_path.exists():
        raise WebResearchError(f"The {description} file does not exist: {path}")
    if not resolved_path.is_file():
        raise WebResearchError(f"The {description} path is not a file: {path}")
    return resolved_path


def _load_research_plan(path: Path) -> ResearchPlan:
    try:
        raw_plan = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WebResearchError(
            "The research plan is not plain UTF-8 text. Run `hailmary prepare-research-plan` again."
        ) from exc
    except OSError as exc:
        raise WebResearchError(f"Could not read the research plan at {path}: {exc}") from exc
    try:
        plan = ResearchPlan.model_validate_json(raw_plan)
    except ValidationError as exc:
        raise WebResearchError(
            "The research plan could not be read. Run `hailmary prepare-research-plan` again."
        ) from exc
    if not plan.tasks:
        raise WebResearchError(
            "The research plan does not contain any tasks. Run "
            "`hailmary prepare-research-plan` again."
        )
    return plan


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    root_path = private_root if private_root.is_absolute() else Path.cwd() / private_root
    resolved_root = root_path.resolve(strict=False)
    resolved_path = (path if path.is_absolute() else Path.cwd() / path).resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise WebResearchError(
            f"Web research results folder {path} resolves outside the private data directory."
        ) from None
    if path.is_symlink():
        raise WebResearchError(f"Web research results folder {path} is a symlink.")
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        root_path.chmod(0o700)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise WebResearchError(
            f"Could not create web research results folder at {path}: {exc}"
        ) from exc


def _unique_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"web-research-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _write_private_json(path: Path, text: str, *, description: str) -> None:
    if path.is_symlink():
        raise WebResearchError(
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
        raise WebResearchError(
            f"Could not write {description} at {path}: the results contain text "
            "that cannot be saved as UTF-8."
        ) from exc
    except OSError as exc:
        raise WebResearchError(f"Could not write {description} at {path}: {exc}") from exc
    finally:
        with suppress(OSError):
            temp_path.unlink()


def _collapse_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate_text(value: str) -> str:
    if len(value) <= MAX_TEXT_CHARS:
        return value
    return value[: MAX_TEXT_CHARS - 4].rstrip() + " ..."


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "web page"
    path = parsed.path.strip("/")
    return f"Fetched page from {host}{('/' + path) if path else ''}"


def _first_validation_detail(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
