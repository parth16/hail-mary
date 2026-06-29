from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

from hailmary.config import AppConfig, ConfigError, validate_local_state
from hailmary.schemas.documents import DocumentType, SourceKind

from .matching import CompanyMatch, CompanyMatchKind, classify_company_match
from .quality import source_reliability_for_provider
from .schemas import (
    ResearchPlan,
    ResearchProviderCategory,
    ResearchResultInput,
    ResearchResultsFile,
    ResearchTask,
    ResearchTaskStatus,
)
from .source_urls import validate_http_url
from .web import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_FETCH_BYTES,
    UrlLibWebResearchClient,
    WebFetchResponse,
    WebResearchClient,
    WebResearchFetchError,
    _as_utc,
    _collapse_text,
    _ensure_private_directory,
    _ensure_resolved_public_host,
    _first_validation_detail,
    _GuardedHTTPSHandler,
    _page_result_from_response,
    _resolve_plan_path,
    _title_from_url,
    _validate_public_fetch_url,
    _write_private_json,
    source_backed_snippet_text,
)

BRAVE_SEARCH_API_KEY_ENV_VAR = "BRAVE_SEARCH_API_KEY"
BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_SEARCH_RESULTS = 6
MAX_FETCHED_RESULTS_PER_TASK = 2
MAX_SEARCH_RESPONSE_BYTES = 1_000_000
PUBLIC_WEB_PROVIDER_ID = "public_web"
SEARCH_ENGINE_HOSTS = (
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "search.yahoo.com",
    "search.brave.com",
    "brave.com",
)


class PublicWebSearchError(RuntimeError):
    """Autonomous public web discovery could not run safely."""


class PublicWebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class PublicWebSearchTaskSummary(BaseModel):
    company_name: str
    provider_id: str = PUBLIC_WEB_PROVIDER_ID
    provider_name: str = "Public web and press search"
    research_topic: str
    query: str
    status: str
    reason: str
    discovered_count: int = 0
    fetched_count: int = 0
    skipped_count: int = 0


class PublicWebResearchRunSummary(BaseModel):
    output_path: Path | None = None
    plan_path: Path
    collected_at: datetime
    search_provider: str = "brave"
    tasks: list[PublicWebSearchTaskSummary] = Field(default_factory=list)
    match_details: list[CompanyMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def discovered_count(self) -> int:
        return sum(task.discovered_count for task in self.tasks)

    @property
    def fetched_count(self) -> int:
        return sum(task.fetched_count for task in self.tasks)

    @property
    def failed_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "failed")

    @property
    def incomplete_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "incomplete_search")

    @property
    def not_run_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "not_run")

    @property
    def no_exact_result_companies(self) -> list[str]:
        fetched_by_company = {
            task.company_name for task in self.tasks if task.fetched_count > 0
        }
        return sorted(
            {
                task.company_name
                for task in self.tasks
                if task.company_name not in fetched_by_company
                and task.status in {"no_exact_results", "incomplete_search", "fetched"}
            }
        )


class PublicWebSearchClient(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        *,
        count: int,
        timeout_seconds: float,
    ) -> list[PublicWebSearchResult]:
        """Return public web search results for one query."""


@dataclass(frozen=True)
class BravePublicWebSearchClient:
    subscription_token: str
    endpoint: str = BRAVE_WEB_SEARCH_ENDPOINT
    provider_name: str = "brave"

    def search(
        self,
        query: str,
        *,
        count: int,
        timeout_seconds: float,
    ) -> list[PublicWebSearchResult]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise PublicWebSearchError("Public web search query cannot be blank.")
        if not self.subscription_token.strip():
            raise PublicWebSearchError(
                f"Set {BRAVE_SEARCH_API_KEY_ENV_VAR} before autonomous public web search."
            )
        request_url = self._request_url(cleaned_query, count=count)
        _validate_brave_search_url(request_url, self.endpoint)
        request = urllib.request.Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "HailMary/0.1 autonomous public web research",
                "X-Subscription-Token": self.subscription_token,
            },
        )
        opener = _build_brave_search_opener()
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status_code = int(getattr(response, "status", 200))
                final_url = response.geturl()
                _validate_brave_search_url(final_url, self.endpoint)
                raw_content = response.read(MAX_SEARCH_RESPONSE_BYTES + 1)
                if len(raw_content) > MAX_SEARCH_RESPONSE_BYTES:
                    raise PublicWebSearchError(
                        "Brave Search returned more data than Hail Mary's limit."
                    )
                charset = response.headers.get_content_charset() or "utf-8"
        except PublicWebSearchError:
            raise
        except urllib.error.HTTPError as exc:
            raise PublicWebSearchError(f"Brave Search returned HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise PublicWebSearchError(f"Could not reach Brave Search: {exc}") from exc
        if status_code >= 400:
            raise PublicWebSearchError(f"Brave Search returned HTTP {status_code}.")
        try:
            payload = json.loads(raw_content.decode(charset, errors="replace"))
        except (LookupError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicWebSearchError(
                "Brave Search returned a response that was not JSON."
            ) from exc
        return _brave_results_from_payload(payload)

    def _request_url(self, query: str, *, count: int) -> str:
        query_string = urllib.parse.urlencode(
            {
                "q": query,
                "count": max(1, min(count, MAX_SEARCH_RESULTS)),
                "safesearch": "moderate",
                "text_decorations": "false",
            }
        )
        return f"{self.endpoint}?{query_string}"


class _BraveNoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = (req, fp, code, msg, headers, newurl)
        raise PublicWebSearchError(
            "Brave Search returned a redirect. Hail Mary did not follow it because "
            "the search request carries an API token."
        )


def _build_brave_search_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _GuardedHTTPSHandler(PUBLIC_WEB_PROVIDER_ID),
        _BraveNoRedirectHandler(),
    )


def collect_public_web_research(
    *,
    config: AppConfig,
    plan_path: Path | None = None,
    search_client: PublicWebSearchClient | None = None,
    web_client: WebResearchClient | None = None,
    collected_at: datetime | None = None,
    search_result_count: int = MAX_SEARCH_RESULTS,
    max_fetched_results_per_task: int = MAX_FETCHED_RESULTS_PER_TASK,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FETCH_BYTES,
) -> PublicWebResearchRunSummary:
    try:
        config = validate_local_state(config)
    except ConfigError as exc:
        raise PublicWebSearchError(str(exc)) from exc
    collected_at = _as_utc(collected_at or datetime.now(UTC))
    resolved_plan_path = _resolve_plan_path(plan_path, data_dir=config.data_dir)
    plan = _load_research_plan(resolved_plan_path)
    public_web_tasks = [task for task in plan.tasks if task.provider_id == PUBLIC_WEB_PROVIDER_ID]
    provider_name = (
        getattr(search_client, "provider_name", "brave")
        if search_client is not None
        else "brave"
    )
    if not public_web_tasks:
        return PublicWebResearchRunSummary(
            output_path=None,
            plan_path=resolved_plan_path,
            collected_at=collected_at,
            search_provider=provider_name,
        )
    gate_reason = _web_research_gate_reason(config)
    if gate_reason is not None:
        return PublicWebResearchRunSummary(
            output_path=None,
            plan_path=resolved_plan_path,
            collected_at=collected_at,
            search_provider=provider_name,
            tasks=[
                _task_summary(
                    task,
                    status="not_run",
                    reason=gate_reason,
                )
                for task in public_web_tasks
            ],
            warnings=[gate_reason],
        )
    resolved_search_client = search_client or _brave_client_from_env()
    if resolved_search_client is None:
        reason = (
            f"Autonomous public web search did not run because {BRAVE_SEARCH_API_KEY_ENV_VAR} "
            "is not set. Set the key and keep local-only mode off to enable Brave Search."
        )
        return PublicWebResearchRunSummary(
            output_path=None,
            plan_path=resolved_plan_path,
            collected_at=collected_at,
            search_provider="brave",
            tasks=[
                _task_summary(task, status="not_run", reason=reason)
                for task in public_web_tasks
            ],
            warnings=[reason],
        )

    provider_name = getattr(resolved_search_client, "provider_name", "public web search")
    fetch_client = web_client or UrlLibWebResearchClient()
    ingested_deal_ids = {deal.deal_id for deal in plan.deals if deal.from_ingestion}
    summaries: list[PublicWebSearchTaskSummary] = []
    results: list[ResearchResultInput] = []
    warnings: list[str] = []
    match_details: list[CompanyMatch] = []
    seen_result_keys: set[tuple[str, str, str]] = set()

    for task in public_web_tasks:
        task_error = _public_web_task_error(task)
        if task_error is not None:
            summaries.append(_task_summary(task, status="not_run", reason=task_error))
            continue
        try:
            search_results = resolved_search_client.search(
                task.query,
                count=search_result_count,
                timeout_seconds=timeout_seconds,
            )
        except PublicWebSearchError as exc:
            reason = str(exc)
            summaries.append(_task_summary(task, status="failed", reason=reason))
            warnings.append(f"{task.company_name} / {task.research_topic}: {reason}")
            continue

        candidates = _rank_search_results(search_results, task=task)
        fetched_count = 0
        fetch_failure_count = 0
        skipped_count = max(0, len(search_results) - len(candidates))
        discovered_count = len(candidates)
        for search_result in candidates:
            if fetched_count >= max_fetched_results_per_task:
                break
            try:
                result = _fetch_public_web_result(
                    task,
                    search_result=search_result,
                    web_client=fetch_client,
                    collected_at=collected_at,
                    ingested_deal_ids=ingested_deal_ids,
                    timeout_seconds=timeout_seconds,
                    max_bytes=max_bytes,
                )
            except WebResearchFetchError as exc:
                fetch_failure_count += 1
                skipped_count += 1
                warnings.append(
                    f"{task.company_name} / {task.research_topic}: skipped "
                    f"{_safe_source_label(search_result.url)} because {exc}; "
                    "search incomplete."
                )
                continue
            match = classify_company_match(task.company_name, _identity_text_for_result(result))
            match_details.append(match)
            if not match.import_ready:
                skipped_count += 1
                continue
            result = result.model_copy(
                update={
                    "identity_match_kind": match.kind,
                    "identity_match_reason": match.reason,
                }
            )
            result_key = (
                task.company_name.casefold(),
                task.research_topic.casefold(),
                (result.source_url or "").casefold(),
            )
            if result_key in seen_result_keys:
                skipped_count += 1
                continue
            seen_result_keys.add(result_key)
            results.append(result)
            fetched_count += 1

        if fetched_count:
            status = "fetched"
            reason = (
                f"Discovered {discovered_count} safe candidate URL"
                f"{'' if discovered_count == 1 else 's'} and fetched {fetched_count} "
                "exact source-backed result"
                f"{'' if fetched_count == 1 else 's'}."
            )
        elif discovered_count:
            if fetch_failure_count:
                status = "incomplete_search"
                reason = (
                    f"Discovered {discovered_count} safe candidate URL"
                    f"{'' if discovered_count == 1 else 's'}, but "
                    f"{fetch_failure_count} source fetch"
                    f"{'' if fetch_failure_count == 1 else 'es'} failed. "
                    "The public web search is incomplete."
                )
            else:
                status = "no_exact_results"
                reason = (
                    f"Discovered {discovered_count} safe candidate URL"
                    f"{'' if discovered_count == 1 else 's'}, but none produced an "
                    "exact company-match snippet safe to import."
                )
        else:
            status = "no_exact_results"
            reason = (
                "Search returned no safe exact-match source pages that Hail Mary "
                "could fetch and import."
            )
        summaries.append(
            _task_summary(
                task,
                status=status,
                reason=reason,
                discovered_count=discovered_count,
                fetched_count=fetched_count,
                skipped_count=skipped_count,
            )
        )

    output_path = _write_public_web_results(
        config=config,
        results=results,
        collected_at=collected_at,
    )
    return PublicWebResearchRunSummary(
        output_path=output_path,
        plan_path=resolved_plan_path,
        collected_at=collected_at,
        search_provider=provider_name,
        tasks=summaries,
        match_details=match_details,
        warnings=warnings,
    )


def _fetch_public_web_result(
    task: ResearchTask,
    *,
    search_result: PublicWebSearchResult,
    web_client: WebResearchClient,
    collected_at: datetime,
    ingested_deal_ids: set[str],
    timeout_seconds: float,
    max_bytes: int,
) -> ResearchResultInput:
    _validate_public_fetch_url(search_result.url, provider_id=PUBLIC_WEB_PROVIDER_ID)
    if _looks_like_search_result_page(search_result.url):
        raise WebResearchFetchError("the URL is a search-results page, not evidence")
    response = web_client.fetch(
        search_result.url,
        provider_id=PUBLIC_WEB_PROVIDER_ID,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
    )
    _validate_public_fetch_url(response.final_url, provider_id=PUBLIC_WEB_PROVIDER_ID)
    if _looks_like_search_result_page(response.final_url):
        raise WebResearchFetchError("the final URL is a search-results page, not evidence")
    page = _page_result_from_response(
        WebFetchResponse.model_validate(response),
        fallback_url=search_result.url,
    )
    if not _page_mentions_exact_company(page.title, page.text, task.company_name):
        raise WebResearchFetchError(
            "the page did not mention the exact company name clearly enough"
        )
    snippet_text = source_backed_snippet_text(
        page.text,
        company_name=task.company_name,
        topic=task.research_topic,
    )
    if not snippet_text or not _page_mentions_exact_company(
        page.title,
        snippet_text,
        task.company_name,
    ):
        raise WebResearchFetchError(
            "the page did not contain a short exact company-match snippet safe to save"
        )
    title = page.title or search_result.title or _title_from_url(page.source_url)
    return ResearchResultInput(
        deal_id=task.deal_id if task.deal_id in ingested_deal_ids else None,
        company_name=task.company_name,
        provider_id=PUBLIC_WEB_PROVIDER_ID,
        provider_name=task.provider_name,
        research_topic=task.research_topic,
        title=title,
        text=snippet_text,
        retrieved_at=collected_at,
        source_url=page.source_url,
        confidence=(
            "medium: autonomous public web search found an exact company-name "
            "source page; Hail Mary saved short snippets only"
        ),
        licensing_notes=(
            "Brave Search was used only to discover the source URL. Hail Mary fetched "
            "the public source page and saved short snippets, not search results or "
            "the full page. Respect the source website's terms before relying on it."
        ),
        source_kind=SourceKind.WEB,
        document_type=DocumentType.WEB_PAGE,
        source_reliability=source_reliability_for_provider(
            PUBLIC_WEB_PROVIDER_ID,
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        ),
        identity_match_kind=CompanyMatchKind.EXACT,
        identity_match_reason="The fetched page mentioned the exact requested company name.",
    )


def _rank_search_results(
    results: Iterable[PublicWebSearchResult],
    *,
    task: ResearchTask,
) -> list[PublicWebSearchResult]:
    ranked: list[tuple[int, int, PublicWebSearchResult]] = []
    for index, result in enumerate(results):
        score = _search_result_score(result, task=task)
        if score <= 0:
            continue
        ranked.append((score, index, result))
    return [
        result
        for _score, _index, result in sorted(ranked, key=lambda item: (-item[0], item[1]))
    ]


def _search_result_score(result: PublicWebSearchResult, *, task: ResearchTask) -> int:
    url = result.url.strip()
    try:
        _validate_public_fetch_url(url, provider_id=PUBLIC_WEB_PROVIDER_ID)
    except (ValueError, WebResearchFetchError):
        return 0
    if _looks_like_search_result_page(url):
        return 0
    identity_match = classify_company_match(task.company_name, _candidate_identity_text(result))
    if not identity_match.import_ready:
        return 0
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    text = f"{result.title} {result.snippet} {host}".casefold()
    score = 10
    if _slugish(task.company_name) in _slugish(host):
        score += 30
    if any(term in text for term in ("official", "about", "company", "customers")):
        score += 12
    if any(term in text for term in ("press", "news", "release", "case study")):
        score += 8
    if any(term in text for term in ("benchmark", "report", "industry", "market")):
        score += 5
    if any(term in text for term in ("login", "sign in", "paywall", "subscribe")):
        score -= 15
    return max(score, 0)


def _candidate_identity_text(result: PublicWebSearchResult) -> str:
    title = _collapse_text(result.title)
    for separator in (" | ", " - ", " : ", " – ", " — "):
        if separator in title:
            title = title.split(separator, 1)[0].strip()
            break
    return title or _collapse_text(result.snippet) or _safe_source_label(result.url)


def _identity_text_for_result(result: ResearchResultInput) -> str:
    title = _collapse_text(result.title)
    for separator in (" | ", " - ", " : ", " – ", " — "):
        if separator in title:
            return title.split(separator, 1)[0].strip()
    if title and not title.casefold().startswith("fetched page from "):
        return title
    if result.source_url:
        return _safe_source_label(result.source_url)
    return title


def _page_mentions_exact_company(title: str, text: str, company_name: str) -> bool:
    normalized_needles = _company_needles(company_name)
    haystack = f"{title} {text}".casefold()
    return any(needle in haystack for needle in normalized_needles)


def _company_needles(company_name: str) -> set[str]:
    cleaned = _collapse_text(company_name).casefold()
    needles = {cleaned}
    if cleaned.endswith(", inc."):
        needles.add(cleaned[:-6].strip())
    for suffix in (" inc.", " incorporated", " llc", " ltd.", " corp.", " corporation"):
        if cleaned.endswith(suffix):
            needles.add(cleaned[: -len(suffix)].strip())
    return {needle for needle in needles if len(needle) >= 3}


def _public_web_task_error(task: ResearchTask) -> str | None:
    if task.provider_category != ResearchProviderCategory.FREE_PUBLIC:
        return "Skipped because this task is not a free public source."
    if task.source_kind != SourceKind.WEB:
        return "Skipped because this task is not public web evidence."
    if task.status != ResearchTaskStatus.PLANNED:
        return "Skipped because this public web task is not planned for autonomous search."
    if not task.query.strip():
        return "Skipped because this public web task did not include a search query."
    return None


def _task_summary(
    task: ResearchTask,
    *,
    status: str,
    reason: str,
    discovered_count: int = 0,
    fetched_count: int = 0,
    skipped_count: int = 0,
) -> PublicWebSearchTaskSummary:
    return PublicWebSearchTaskSummary(
        company_name=task.company_name,
        provider_name=task.provider_name,
        research_topic=task.research_topic,
        query=task.query,
        status=status,
        reason=reason,
        discovered_count=discovered_count,
        fetched_count=fetched_count,
        skipped_count=skipped_count,
    )


def _write_public_web_results(
    *,
    config: AppConfig,
    results: list[ResearchResultInput],
    collected_at: datetime,
) -> Path | None:
    if not results:
        return None
    try:
        results_file = ResearchResultsFile.model_validate({"results": results})
    except ValidationError as exc:
        detail = _first_validation_detail(exc)
        raise PublicWebSearchError(
            f"Collected public web research did not pass validation: {detail}"
        ) from exc
    output_dir = config.data_dir / "research-results"
    _ensure_private_directory(output_dir, private_root=config.data_dir)
    output_path = _unique_public_web_results_path(output_dir, collected_at)
    payload = {
        "results": [
            result.model_dump(mode="json", exclude_none=True)
            for result in results_file.results
        ]
    }
    _write_private_json(
        output_path,
        json.dumps(payload, indent=2),
        description="public web research results",
    )
    return output_path


def _unique_public_web_results_path(output_dir: Path, collected_at: datetime) -> Path:
    base_name = f"public-web-research-results-{collected_at.strftime('%Y%m%d-%H%M%S')}"
    candidate = output_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def _load_research_plan(path: Path) -> ResearchPlan:
    try:
        raw_plan = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PublicWebSearchError(
            "The research plan is not plain UTF-8 text. Run `hailmary prepare-research-plan` again."
        ) from exc
    except OSError as exc:
        raise PublicWebSearchError(f"Could not read the research plan at {path}: {exc}") from exc
    try:
        return ResearchPlan.model_validate_json(raw_plan)
    except ValidationError as exc:
        raise PublicWebSearchError(
            "The research plan could not be read. Run `hailmary prepare-research-plan` again."
        ) from exc


def _brave_client_from_env() -> BravePublicWebSearchClient | None:
    api_key = os.environ.get(BRAVE_SEARCH_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        return None
    return BravePublicWebSearchClient(subscription_token=api_key)


def _brave_results_from_payload(payload: object) -> list[PublicWebSearchResult]:
    if not isinstance(payload, dict):
        raise PublicWebSearchError("Brave Search returned an unexpected response.")
    web_section = payload.get("web")
    if web_section is None:
        return []
    if not isinstance(web_section, dict) or not isinstance(web_section.get("results"), list):
        raise PublicWebSearchError("Brave Search returned an unexpected web results shape.")
    results: list[PublicWebSearchResult] = []
    for raw_result in web_section["results"]:
        if not isinstance(raw_result, dict):
            continue
        url = raw_result.get("url")
        title = raw_result.get("title")
        snippet = raw_result.get("description") or raw_result.get("snippet") or ""
        if not isinstance(url, str) or not isinstance(title, str):
            continue
        results.append(
            PublicWebSearchResult(
                title=_collapse_text(title),
                url=url.strip(),
                snippet=_collapse_text(str(snippet)),
            )
        )
    return results


def _validate_brave_search_url(url: str, endpoint: str) -> None:
    try:
        validate_http_url(url, field_name="Brave Search URL")
    except ValueError as exc:
        raise PublicWebSearchError(str(exc)) from exc
    parsed_url = urlparse(url)
    parsed_endpoint = urlparse(endpoint)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != parsed_endpoint.hostname
        or parsed_url.path != parsed_endpoint.path
    ):
        raise PublicWebSearchError(
            "Brave Search redirected away from the expected public search API."
        )
    _ensure_resolved_public_host(url)


def _looks_like_search_result_page(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    if any(
        host == search_host or host.endswith(f".{search_host}")
        for search_host in SEARCH_ENGINE_HOSTS
    ):
        return True
    return path.startswith(("/search", "/results"))


def _safe_source_label(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "source page"
    path = parsed.path.strip("/")
    if not path:
        return host
    parts = [part for part in path.split("/") if part][:2]
    return f"{host}/{'/'.join(parts)}"


def _web_research_gate_reason(config: AppConfig) -> str | None:
    if config.local_only:
        return (
            "Autonomous public web search did not run because local-only mode is on. "
            "Set HAILMARY_LOCAL_ONLY=false and HAILMARY_ENABLE_WEB_RESEARCH=true to "
            "allow bounded public web requests."
        )
    if not config.enable_web_research:
        return (
            "Autonomous public web search did not run because web research is disabled. "
            "Set HAILMARY_ENABLE_WEB_RESEARCH=true to allow bounded public web requests."
        )
    return None


def _slugish(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
