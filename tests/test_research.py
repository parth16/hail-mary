from __future__ import annotations

import hashlib
import json
import shlex
import socket
import stat
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import pytest
from typer.testing import CliRunner

import hailmary.cli as cli_module
import hailmary.research.collection as collection_module
from hailmary.agents.packets import build_agent_input_packet
from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.research import (
    CompanyMatch,
    CompanyMatchKind,
    GitHubApiError,
    GitHubRepositoryCollectionRunSummary,
    GitHubRepositorySearchResponse,
    MeridianWorkflowError,
    PaidProviderFact,
    PaidProviderSearchRequest,
    PaidProviderSearchResponse,
    ResearchCollectionDealSummary,
    ResearchCollectionError,
    ResearchImportError,
    ResearchPlanError,
    ResearchProviderCategory,
    ResearchProviderRunStatus,
    ResearchTaskStatus,
    ResearchTemplateError,
    SbirApiError,
    SbirAwardRecord,
    SbirAwardsResponse,
    SbirCollectionRunSummary,
    SecFormDApiError,
    SecFormDCollectionRunSummary,
    SecFormDFilingRecord,
    SecFormDFilingsResponse,
    UrlLibGitHubRepositorySearchClient,
    UrlLibSbirAwardsClient,
    UrlLibSecFormDFilingsClient,
    UrlLibUsaspendingAwardsClient,
    UsaspendingApiError,
    UsaspendingAwardRecord,
    UsaspendingAwardsResponse,
    UsaspendingCollectionRunSummary,
    WebResearchError,
    builtin_research_providers,
    classify_company_match,
    collect_github_repositories,
    collect_paid_research_results,
    collect_sbir_awards,
    collect_sec_form_d_filings,
    collect_usaspending_awards,
    collect_web_research,
    import_research_results,
    prepare_meridian_workflow,
    prepare_public_research_results,
    prepare_research_plan,
    prepare_research_results_template,
    research_quality_status,
    run_research_workflow,
    source_freshness_for_retrieved_at,
    source_reliability_for_provider,
)
from hailmary.research.meridian import (
    MERIDIAN_LEGACY_PLACEHOLDER_CONFIDENCE,
    MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER,
    MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER,
    MERIDIAN_WORKFLOW_TEMPLATE_MARKER,
    clean_meridian_url,
)
from hailmary.research.schemas import ResearchResultInput
from hailmary.research.source_urls import validate_http_url
from hailmary.research.web import (
    WebFetchResponse,
    WebResearchFetchError,
    _BoundHTTPConnection,
    _build_guarded_opener,
)
from hailmary.schemas.agents import AgentRole
from hailmary.schemas.documents import DocumentType, IngestedDeal, SourceKind
from hailmary.schemas.evidence import EvidenceStore, SourceFreshness, SourceReliability
from hailmary.scoring.memo import render_markdown_memo
from hailmary.scoring.scorer import score_evidence_store

runner = CliRunner()
BUILT_AT = datetime(2026, 1, 1, tzinfo=UTC)


class _ProxyHandlerWithProxies(Protocol):
    proxies: dict[str, str]


def test_builtin_research_providers_hide_paid_by_default() -> None:
    providers = builtin_research_providers()

    provider_ids = {provider.id for provider in providers}
    assert "sec_form_d" in provider_ids
    assert "meridian" in provider_ids
    assert "crunchbase" not in provider_ids


def test_builtin_research_providers_include_paid_when_requested() -> None:
    providers = builtin_research_providers(include_paid=True)

    paid_providers = [
        provider
        for provider in providers
        if provider.category == ResearchProviderCategory.PAID_OPTIONAL
    ]
    assert {provider.id for provider in paid_providers} >= {
        "crunchbase",
        "people_data_labs",
        "newsapi",
        "similarweb",
        "sensor_tower",
        "pitchbook",
        "cb_insights",
    }
    assert all(provider.default_enabled is False for provider in paid_providers)
    assert all(provider.credential_env_var for provider in paid_providers)


def test_research_result_source_kind_defaults_match_builtin_registry() -> None:
    providers = builtin_research_providers(include_paid=True, include_meridian=True)
    base_payload = _research_result()

    for provider in providers:
        payload = {**base_payload, "provider_id": provider.id}

        result = ResearchResultInput.model_validate(payload)

        assert result.source_kind == provider.source_kind


def test_company_match_classifies_exact_related_likely_and_rejected() -> None:
    exact = classify_company_match("Acme AI", "Acme AI")
    legal_entity = classify_company_match("Acme AI", "Acme AI, Inc.")
    likely = classify_company_match("Acme AI", "AcmeAI")
    product = classify_company_match("Acme AI", "Acme AI Platform")
    related = classify_company_match("Acme AI", "Acme AI Federal")
    suffix_variant = classify_company_match("Acme LLC", "Acme LP")
    founder = classify_company_match("Acme AI", "Jane Founder, Acme AI")
    investor = classify_company_match("Acme AI", "Acme AI Ventures")
    rejected = classify_company_match("Acme AI", "Unrelated Robotics")

    assert exact.kind == CompanyMatchKind.EXACT
    assert exact.import_ready is True
    assert legal_entity.kind == CompanyMatchKind.LEGAL_ENTITY
    assert legal_entity.import_ready is True
    assert likely.kind == CompanyMatchKind.LIKELY
    assert likely.import_ready is False
    assert product.kind == CompanyMatchKind.PRODUCT_NAME
    assert product.import_ready is False
    assert related.kind == CompanyMatchKind.RELATED
    assert related.import_ready is False
    assert suffix_variant.kind == CompanyMatchKind.AMBIGUOUS
    assert suffix_variant.import_ready is False
    assert founder.kind == CompanyMatchKind.FOUNDER_RELATED
    assert founder.import_ready is False
    assert investor.import_ready is False
    assert rejected.kind == CompanyMatchKind.REJECTED
    assert rejected.import_ready is False


def test_collection_company_cleaning_preserves_suffix_distinct_requests() -> None:
    assert collection_module._clean_company_names(
        ["Acme", " Acme ", "Acme Inc.", "acme inc."]
    ) == ["Acme", "Acme Inc."]


def test_source_reliability_and_freshness_classification() -> None:
    assert (
        source_reliability_for_provider(
            "company_website",
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        )
        == SourceReliability.OFFICIAL_COMPANY
    )
    assert (
        source_reliability_for_provider(
            "sec_form_d",
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        )
        == SourceReliability.GOVERNMENT_FILING
    )
    assert (
        source_reliability_for_provider(
            "github",
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        )
        == SourceReliability.REPOSITORY_METADATA
    )
    assert (
        source_reliability_for_provider(
            "custom_public_source",
            source_kind=SourceKind.WEB,
            document_type=DocumentType.WEB_PAGE,
        )
        == SourceReliability.UNKNOWN
    )
    assert (
        source_freshness_for_retrieved_at(
            datetime(2026, 1, 1, tzinfo=UTC),
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )
        == SourceFreshness.CURRENT
    )
    assert (
        source_freshness_for_retrieved_at(
            datetime(2024, 1, 1, tzinfo=UTC),
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )
        == SourceFreshness.STALE
    )


def test_prepare_research_plan_writes_private_manual_plan(tmp_path: Path) -> None:
    result = prepare_research_plan(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["Acme AI"],
        created_at=BUILT_AT,
    )

    assert result.deal_count == 1
    assert result.task_count == 8
    assert result.output_path.exists()
    assert result.output_path.parent == tmp_path / "data" / "research-plans"
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert any(task.provider_id == "sec_form_d" for task in result.plan.tasks)
    assert all(
        task.provider_category != ResearchProviderCategory.PAID_OPTIONAL
        for task in result.plan.tasks
    )
    assert "No websites, APIs, paid databases" in result.plan.notes[0]

    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["deals"][0]["company_name"] == "Acme AI"
    assert saved["tasks"][0]["confidence"] == "not_collected"
    assert "provider, timestamp, exact URL" in saved["tasks"][0]["evidence_policy"]
    assert any(
        "source_url or source_api" in item
        for item in saved["tasks"][0]["required_metadata"]
    )
    assert any("Do not copy screenshots" in item for item in saved["tasks"][0]["do_not_copy"])
    assert saved["tasks"][0]["what_to_look_for"]


def test_prepare_research_plan_uses_company_website_for_one_deal(tmp_path: Path) -> None:
    result = prepare_research_plan(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )

    website_task = next(task for task in result.plan.tasks if task.provider_id == "company_website")
    assert website_task.url == "https://example.com"
    assert website_task.status == ResearchTaskStatus.PLANNED


def test_prepare_research_plan_rejects_website_for_multiple_companies(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchPlanError, match="exactly one company"):
        prepare_research_plan(
            config=AppConfig(data_dir=tmp_path / "data"),
            company_names=["Acme AI", "Beta Robotics"],
            website_url="https://example.com",
            created_at=BUILT_AT,
        )


@pytest.mark.parametrize(
    "website_url",
    [
        "https://",
        "https://example .com",
        "https://[broken",
        "https://example.com:bad/path",
        "https://example.com:99999/path",
        "https://user:token@example.com/path",
        "mailto:founder@example.com",
    ],
)
def test_prepare_research_plan_rejects_malformed_website_url(
    tmp_path: Path,
    website_url: str,
) -> None:
    with pytest.raises(ResearchPlanError):
        prepare_research_plan(
            config=AppConfig(data_dir=tmp_path / "data"),
            company_names=["Acme AI"],
            website_url=website_url,
            created_at=BUILT_AT,
        )


def test_prepare_research_plan_includes_paid_as_manual_tasks(tmp_path: Path) -> None:
    result = prepare_research_plan(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["Acme AI"],
        include_paid=True,
        created_at=BUILT_AT,
    )

    paid_tasks = [
        task
        for task in result.plan.tasks
        if task.provider_category == ResearchProviderCategory.PAID_OPTIONAL
    ]
    assert {task.provider_id for task in paid_tasks} >= {"crunchbase", "pitchbook"}
    assert all(task.status == ResearchTaskStatus.NEEDS_OPERATOR for task in paid_tasks)
    assert all("Paid optional source" in task.licensing_notes for task in paid_tasks)


def test_paid_provider_collection_is_disabled_by_default(tmp_path: Path) -> None:
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI")],
    )

    result = collect_paid_research_results(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["Acme AI"],
        clients={"crunchbase": client},
        collected_at=BUILT_AT,
    )

    assert result.provider_ids == []
    assert result.output_path is None
    assert result.result_count == 0
    assert client.calls == []


def test_paid_provider_collection_rejects_explicit_disabled_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")

    with pytest.raises(ResearchCollectionError, match="crunchbase is disabled"):
        collect_paid_research_results(
            config=AppConfig(data_dir=tmp_path / "data"),
            company_names=["Acme AI"],
            provider_ids=["crunchbase"],
            clients={"crunchbase": _FakePaidProviderClient(provider_id="crunchbase")},
            collected_at=BUILT_AT,
        )


def test_paid_provider_collection_reports_missing_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRUNCHBASE_API_KEY", raising=False)

    with pytest.raises(ResearchCollectionError, match="CRUNCHBASE_API_KEY is missing"):
        collect_paid_research_results(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=False,
                enable_web_research=True,
                enabled_paid_providers=("crunchbase",),
            ),
            company_names=["Acme AI"],
            clients={"crunchbase": _FakePaidProviderClient(provider_id="crunchbase")},
            collected_at=BUILT_AT,
        )


def test_paid_provider_collection_respects_local_only_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI")],
    )

    with pytest.raises(ResearchCollectionError, match="local-only mode is on"):
        collect_paid_research_results(
            config=AppConfig(
                data_dir=tmp_path / "data",
                enabled_paid_providers=("crunchbase",),
            ),
            company_names=["Acme AI"],
            clients={"crunchbase": client},
            collected_at=BUILT_AT,
        )
    assert client.calls == []


def test_paid_provider_collection_writes_importable_mock_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(
        update={
            "local_only": False,
            "enable_web_research": True,
            "enabled_paid_providers": ("crunchbase",),
        }
    )
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[
            _paid_fact(
                company_name="Acme AI",
                title="Acme AI Crunchbase profile",
                text="Acme AI raised a synthetic seed round from named investors.",
                source_url="https://www.crunchbase.com/organization/acme-ai",
                source_api="https://api.crunchbase.com/api/v4/entities/organizations/acme-ai",
                licensing_notes=(
                    "Licensed Crunchbase account permits saving this short diligence fact."
                ),
            )
        ],
    )

    result = collect_paid_research_results(
        config=config,
        company_names=["Acme AI"],
        clients={"crunchbase": client},
        collected_at=BUILT_AT,
    )

    assert client.calls == ["Acme AI"]
    assert result.output_path is not None
    assert result.result_count == 1
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["provider_id"] == "crunchbase"
    assert saved["results"][0]["provider_name"] == "Crunchbase"
    assert saved["results"][0]["source_api"] == (
        "https://api.crunchbase.com/api/v4/entities/organizations/acme-ai"
    )
    dry_run = import_research_results(
        config=config,
        results_path=result.output_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )
    assert dry_run.imported_count == 1


def test_paid_provider_collection_allows_newsapi_article_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(
        update={
            "local_only": False,
            "enable_web_research": True,
            "enabled_paid_providers": ("newsapi",),
        }
    )
    client = _FakePaidProviderClient(
        provider_id="newsapi",
        facts=[
            _paid_fact(
                company_name="Acme AI",
                title="Acme AI synthetic news article",
                text="Acme AI announced a synthetic customer launch in a news article.",
                source_url="https://techcrunch.com/2025/12/31/acme-ai-launch",
                source_api="https://newsapi.org/v2/everything?q=Acme%20AI",
                licensing_notes=(
                    "Licensed NewsAPI account permits saving this short article summary."
                ),
            )
        ],
    )

    result = collect_paid_research_results(
        config=config,
        company_names=["Acme AI"],
        clients={"newsapi": client},
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["source_url"] == (
        "https://techcrunch.com/2025/12/31/acme-ai-launch"
    )
    dry_run = import_research_results(
        config=config,
        results_path=result.output_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )
    assert dry_run.imported_count == 1


def test_paid_provider_collection_rejects_unsafe_source_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[
            _paid_fact(
                company_name="Acme AI",
                source_url=None,
                source_api=(
                    "https://api.crunchbase.com/api/v4/entities/organizations/acme-ai"
                    "?api_key=secret"
                ),
            )
        ],
    )

    with pytest.raises(ResearchCollectionError, match="token, signature, credential"):
        collect_paid_research_results(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=False,
                enable_web_research=True,
                enabled_paid_providers=("crunchbase",),
            ),
            company_names=["Acme AI"],
            clients={"crunchbase": client},
            collected_at=BUILT_AT,
        )


def test_paid_provider_collection_rejects_unsafe_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[
            _paid_fact(
                company_name="Acme AI",
                source_url="https://www.crunchbase.com/organization/acme-ai?token=secret",
                source_api=None,
            )
        ],
    )

    with pytest.raises(ResearchCollectionError, match="token, signature, credential"):
        collect_paid_research_results(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=False,
                enable_web_research=True,
                enabled_paid_providers=("crunchbase",),
            ),
            company_names=["Acme AI"],
            clients={"crunchbase": client},
            collected_at=BUILT_AT,
        )


def test_paid_provider_collection_rejects_placeholder_licensing_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI", licensing_notes="todo")],
    )

    with pytest.raises(ResearchCollectionError, match="licensing_notes.*placeholder"):
        collect_paid_research_results(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=False,
                enable_web_research=True,
                enabled_paid_providers=("crunchbase",),
            ),
            company_names=["Acme AI"],
            clients={"crunchbase": client},
            collected_at=BUILT_AT,
        )


def test_paid_provider_collection_skips_related_name_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI Holdings")],
    )

    result = collect_paid_research_results(
        config=AppConfig(
            data_dir=tmp_path / "data",
            local_only=False,
            enable_web_research=True,
            enabled_paid_providers=("crunchbase",),
        ),
        company_names=["Acme AI"],
        clients={"crunchbase": client},
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.skipped_non_exact_company_names == ["Acme AI Holdings"]
    assert {match.kind for match in result.match_details} == {CompanyMatchKind.RELATED}


def test_research_workflow_collects_enabled_paid_provider_with_fake_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(
        update={
            "local_only": False,
            "enable_web_research": True,
            "enabled_paid_providers": ("crunchbase",),
        }
    )
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI")],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        include_paid=True,
        paid_clients={"crunchbase": client},
        sec_form_d_client=_FakeSecFormDFilingsClient(
            {("Acme AI", 0): _sec_form_d_response([])}
        ),
        usaspending_client=_FakeUsaspendingAwardsClient(
            {("Acme AI", 1): _usaspending_response([])}
        ),
        sbir_client=_FakeSbirAwardsClient({("Acme AI", 0): _sbir_response([])}),
        github_client=_FakeGitHubRepositorySearchClient(
            {("Acme AI", 1): _github_repository_response([])}
        ),
        created_at=BUILT_AT,
    )

    assert client.calls == ["Acme AI"]
    paid_collection = next(
        collection for collection in result.collections if collection.source_id == "paid_optional"
    )
    assert paid_collection.result_count == 1
    paid_preview = next(
        preview
        for preview in result.import_previews
        if preview.input_path == paid_collection.output_path
    )
    assert paid_preview.imported_count == 1
    crunchbase_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "crunchbase"
    )
    assert crunchbase_status.provider_name == "Crunchbase"
    assert crunchbase_status.status == ResearchProviderRunStatus.PLANNED
    assert crunchbase_status.collected_count == 1
    assert result.privacy_notes[0].startswith("No screenshots, cookies, browser profiles")
    assert "paid-provider facts may be saved" in result.privacy_notes[0]


def test_research_workflow_leaves_paid_sources_manual_without_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(update={"enabled_paid_providers": ("crunchbase",)})

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        include_paid=True,
        created_at=BUILT_AT,
    )

    assert all(collection.source_id != "paid_optional" for collection in result.collections)
    assert not any(issue.source == "optional paid providers" for issue in result.issues)


def test_research_workflow_does_not_call_paid_clients_in_local_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(update={"enabled_paid_providers": ("crunchbase",)})
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI")],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        include_paid=True,
        paid_clients={"crunchbase": client},
        created_at=BUILT_AT,
    )

    assert client.calls == []
    assert all(collection.source_id != "paid_optional" for collection in result.collections)
    assert "paid-source outputs are saved" in result.privacy_notes[0]


def test_research_workflow_requires_explicit_company_names_for_paid_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUNCHBASE_API_KEY", "synthetic-test-key")
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    config = config.model_copy(
        update={
            "local_only": False,
            "enable_web_research": True,
            "enabled_paid_providers": ("crunchbase",),
        }
    )
    client = _FakePaidProviderClient(
        provider_id="crunchbase",
        facts=[_paid_fact(company_name="Acme AI")],
    )

    result = run_research_workflow(
        config=config,
        include_paid=True,
        paid_clients={"crunchbase": client},
        sec_form_d_client=_FakeSecFormDFilingsClient(
            {("Acme AI", 0): _sec_form_d_response([])}
        ),
        usaspending_client=_FakeUsaspendingAwardsClient(
            {("Acme AI", 1): _usaspending_response([])}
        ),
        sbir_client=_FakeSbirAwardsClient({("Acme AI", 0): _sbir_response([])}),
        github_client=_FakeGitHubRepositorySearchClient(
            {("Acme AI", 1): _github_repository_response([])}
        ),
        created_at=BUILT_AT,
    )

    assert client.calls == []
    assert all(collection.source_id != "paid_optional" for collection in result.collections)


def test_research_workflow_command_creates_artifacts_and_reports_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "research-workflow",
            "--company",
            "Acme AI",
            "--website",
            "https://example.com/acme",
            "--meridian-url",
            "https://portal.angellist.com/m/example/invest",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Research workflow" in result.output
    assert "Saved the private research plan" in result.output
    assert "Saved the fillable results template" in result.output
    assert "Saved the manual follow-up queue" in result.output
    assert "Saved the Meridian workflow" in result.output
    assert "Meridian is a manual authenticated workflow" in result.output
    assert "need manual or local-file work" in result.output
    assert "Provider statuses" in result.output
    assert "Live public collection did not run" in result.output
    assert "Ready to import: no completed result files were found yet" in result.output
    assert "No screenshots, cookies, browser profiles" in result.output
    assert len(list((data_dir / "research-plans").glob("research-plan-*.json"))) == 1
    assert len(list((data_dir / "research-manual-tasks").glob("*.json"))) == 1
    assert len(list((data_dir / "research-results-templates").glob("*.json"))) == 2
    assert len(list((data_dir / "meridian-workflows").glob("*.json"))) == 1


def test_research_workflow_command_json_includes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "research-workflow",
            "--company",
            "Acme AI",
            "--data-dir",
            str(data_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["planned_task_count"] == len(payload["plan"]["tasks"])
    assert payload["summary"]["failed_provider_count"] == 0
    assert payload["summary"]["incomplete_search_count"] == 0
    assert payload["summary"]["manual_needed_provider_count"] >= 3
    assert payload["manual_task_queue_path"]
    assert payload["summary"]["provider_statuses"]
    assert any(
        status["provider_id"] == "sec_form_d"
        for status in payload["summary"]["provider_statuses"]
    )
    sec_status = next(
        status
        for status in payload["summary"]["provider_statuses"]
        if status["provider_id"] == "sec_form_d"
    )
    assert sec_status["status"] == ResearchProviderRunStatus.NOT_RUN


def test_research_workflow_rejects_unsafe_meridian_url_before_writing_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "research-workflow",
            "--company",
            "Acme AI",
            "--meridian-url",
            "https://portal.angellist.com/m/example/invest?token=secret",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code != 0
    output = _plain_cli_output(result.output)
    assert "Meridian URL cannot include query strings" in output
    assert not (data_dir / "research-plans").exists()
    assert "token=secret" not in output


@pytest.mark.parametrize(
    "website_url",
    [
        "https://example.com/acme?token=secret#details",
        "https://example.com/acme;token=secret/details",
        "https://example.com/acme%3Ftoken=secret",
        "https://example.com/acme%3Btoken=secret/details",
        "https://example.com/acme%23token=secret",
        "https://example.com/acme%253Ftoken=secret",
        "https://example.com/acme%2525253Ftoken=secret",
    ],
)
def test_research_workflow_rejects_tokenized_website_url_before_writing_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    website_url: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "research-workflow",
            "--company",
            "Acme AI",
            "--website",
            website_url,
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code != 0
    output = _plain_cli_output(result.output)
    assert "website URL cannot include query strings" in output
    assert not (data_dir / "research-plans").exists()
    assert "token=secret" not in output


def test_run_research_workflow_runs_live_collectors_when_web_research_is_enabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    summary = ingest_folder(root, config=config)
    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")

    web_client = _FakeWebResearchClient(
        {
            "https://example.com/acme": WebFetchResponse(
                final_url="https://example.com/acme",
                content_type="text/html",
                text="<html><title>Acme AI</title><body>Acme AI has customers.</body></html>",
            )
        }
    )
    sec_client = _FakeSecFormDFilingsClient(
        {
            ("Acme AI", 0): _sec_form_d_response(
                [_sec_form_d_filing(issuer_name="Acme AI")]
            )
        }
    )
    usaspending_client = _FakeUsaspendingAwardsClient(
        {("Acme AI", 1): _usaspending_response([_usaspending_award(recipient_name="Acme AI")])}
    )
    sbir_client = _FakeSbirAwardsClient(
        {("Acme AI", 0): _sbir_response([_sbir_award(firm="Acme AI", award_title="Acme AI Award")])}
    )
    github_client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="acme-ai",
                        full_name="acme-ai/product",
                        owner_login="acme-ai",
                    )
                ]
            )
        }
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com/acme",
        created_at=BUILT_AT,
        web_client=web_client,
        sec_form_d_client=sec_client,
        usaspending_client=usaspending_client,
        sbir_client=sbir_client,
        github_client=github_client,
    )

    assert result.live_collection_enabled is True
    assert web_client.calls == ["https://example.com/acme"]
    assert sec_client.calls == [("Acme AI", 10, 0)]
    assert usaspending_client.calls == [("Acme AI", 10, 1)]
    assert sbir_client.calls == [("Acme AI", 10, 0)]
    assert github_client.calls == [("Acme AI", 10, 1)]
    collection_ids = {collection.source_id for collection in result.collections}
    assert collection_ids >= {
        "public_web_pages",
        "sec_form_d",
        "usaspending",
        "sbir",
        "github",
    }
    assert result.ready_to_import_count == 5
    assert result.blocking_issue_count == 0
    assert result.no_prepared_result_companies == []
    company_website_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "company_website"
    )
    assert company_website_status.status == ResearchProviderRunStatus.PLANNED
    assert company_website_status.collected_count == 1
    assert company_website_status.imported_count == 0
    sam_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "sam_gov"
    )
    assert sam_status.status == ResearchProviderRunStatus.MANUAL_NEEDED
    assert sam_status.no_exact_result_companies == []
    uspto_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "uspto"
    )
    assert uspto_status.status == ResearchProviderRunStatus.MANUAL_NEEDED
    assert uspto_status.no_exact_result_companies == []
    public_web_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "public_web"
    )
    assert public_web_status.status == ResearchProviderRunStatus.MANUAL_NEEDED
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store


def test_run_research_workflow_treats_corrupt_import_state_as_blocking(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    assert deal.evidence_store_path is not None
    deal.evidence_store_path.unlink()
    sec_results_path = tmp_path / "sec-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        created_at=BUILT_AT,
    )

    assert result.blocking_issue_count == 1
    assert any(
        issue.severity == "error" and "evidence store" in issue.message
        for issue in result.issues
    )


def test_run_research_workflow_treats_live_collection_failures_as_blocking(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    web_client = _FailingWebResearchClient("Could not fetch the public page.")
    sec_client = _FakeSecFormDFilingsClient(
        {},
        error=SecFormDApiError("SEC User-Agent must include a contact email."),
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com/acme",
        created_at=BUILT_AT,
        web_client=web_client,
        sec_form_d_client=sec_client,
        usaspending_client=_FakeUsaspendingAwardsClient(
            {("Acme AI", 1): _usaspending_response([])}
        ),
        sbir_client=_FakeSbirAwardsClient({("Acme AI", 0): _sbir_response([])}),
        github_client=_FakeGitHubRepositorySearchClient(
            {("Acme AI", 1): _github_repository_response([])}
        ),
    )

    assert web_client.calls == ["https://example.com/acme"]
    assert result.blocking_issue_count == 2
    error_messages = [issue.message for issue in result.issues if issue.severity == "error"]
    assert any("Could not fetch the public page." in message for message in error_messages)
    assert any("SEC User-Agent" in message for message in error_messages)
    assert result.summary.failed_provider_count == 2
    assert any(
        status.provider_id == "sec_form_d"
        and status.status == ResearchProviderRunStatus.FAILED
        for status in result.summary.provider_statuses
    )


def test_run_research_workflow_tracks_incomplete_paginated_search(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        created_at=BUILT_AT,
        web_client=_FakeWebResearchClient({}),
        sec_form_d_client=_FakeSecFormDFilingsClient(
            {("Acme AI", 0): _sec_form_d_response([])}
        ),
        usaspending_client=_FakeUsaspendingAwardsClient(
            {
                ("Acme AI", page): _usaspending_response(
                    [
                        _usaspending_award(
                            recipient_name="Acme AI Federal",
                            award_id=f"FAKE-{page}",
                            generated_internal_id=f"CONT_AWD_FAKE_{page}",
                        )
                    ],
                    has_next=True,
                )
                for page in range(1, 21)
            }
        ),
        sbir_client=_FakeSbirAwardsClient({("Acme AI", 0): _sbir_response([])}),
        github_client=_FakeGitHubRepositorySearchClient(
            {("Acme AI", 1): _github_repository_response([])}
        ),
    )

    usaspending = next(
        collection for collection in result.collections if collection.source_id == "usaspending"
    )
    assert usaspending.status == ResearchProviderRunStatus.INCOMPLETE_SEARCH
    assert usaspending.incomplete_search is True
    assert "Acme AI" in usaspending.no_result_companies
    assert result.summary.incomplete_search_count == 1
    assert any(
        status.provider_id == "usaspending"
        and status.status == ResearchProviderRunStatus.INCOMPLETE_SEARCH
        for status in result.summary.provider_statuses
    )


def test_run_research_workflow_keeps_incomplete_warning_with_saved_result(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    usaspending_responses = {
        ("Acme AI", 1): _usaspending_response(
            [
                _usaspending_award(
                    recipient_name="Acme AI",
                    award_id="FAKE-EXACT",
                    generated_internal_id="CONT_AWD_FAKE_EXACT",
                )
            ],
            has_next=True,
        )
    }
    usaspending_responses.update(
        {
            ("Acme AI", page): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI Ventures",
                        award_id=f"FAKE-RELATED-{page}",
                        generated_internal_id=f"CONT_AWD_FAKE_RELATED_{page}",
                    )
                ],
                has_next=True,
            )
            for page in range(2, 21)
        }
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        created_at=BUILT_AT,
        web_client=_FakeWebResearchClient({}),
        sec_form_d_client=_FakeSecFormDFilingsClient(
            {("Acme AI", 0): _sec_form_d_response([])}
        ),
        usaspending_client=_FakeUsaspendingAwardsClient(usaspending_responses),
        sbir_client=_FakeSbirAwardsClient({("Acme AI", 0): _sbir_response([])}),
        github_client=_FakeGitHubRepositorySearchClient(
            {("Acme AI", 1): _github_repository_response([])}
        ),
    )

    usaspending = next(
        collection for collection in result.collections if collection.source_id == "usaspending"
    )
    assert usaspending.result_count == 1
    assert usaspending.status == ResearchProviderRunStatus.INCOMPLETE_SEARCH
    assert usaspending.incomplete_search is True
    assert usaspending.warnings
    assert result.summary.incomplete_search_count == 1


def test_run_research_workflow_reports_local_public_skips_and_no_results(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    assert deal.evidence_store_path is not None
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")
    sec_results_path = tmp_path / "sec-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            },
            {
                "company_name": "Acme AI Holdings",
                "title": "Acme AI Holdings Form D",
                "text": "A related entity that must not be imported.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/holdings/form-d",
            },
        ],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI", "MissingCo"],
        sec_form_d_results_path=sec_results_path,
        created_at=BUILT_AT,
    )

    local_public = next(
        collection for collection in result.collections if collection.source_id == "local_public"
    )
    assert local_public.result_count == 1
    assert local_public.skipped_non_exact_company_names == ["Acme AI Holdings"]
    assert local_public.status == ResearchProviderRunStatus.PLANNED
    assert {match.kind for match in local_public.match_details} == {
        CompanyMatchKind.EXACT,
        CompanyMatchKind.RELATED,
    }
    assert "MissingCo" in local_public.no_result_companies
    assert result.ready_to_import_count == 1
    assert result.no_prepared_result_companies == ["MissingCo"]
    assert result.summary.imported_record_count == 1
    assert result.summary.failed_provider_count == 0
    sec_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "sec_form_d"
    )
    assert sec_status.status == ResearchProviderRunStatus.PLANNED
    assert sec_status.collected_count == 1
    assert sec_status.no_exact_result_companies == ["MissingCo"]
    operator_lines = " ".join(
        line.plain for line in cli_module._research_workflow_collection_lines(local_public)
    )
    assert "skipped related match Acme AI Holdings for Acme AI" in operator_lines
    assert "operator validates the entity" in operator_lines
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store


def test_run_research_workflow_counts_only_unresolved_manual_tasks(
    tmp_path: Path,
) -> None:
    config, _deal, results_path = _ingest_deal_and_write_results(tmp_path)
    _write_results(
        results_path,
        [
            _research_result(
                provider_id="sam_gov",
                provider_name="SAM.gov",
                title="Acme AI SAM.gov record",
                text="Acme AI has a public SAM.gov registration record.",
                retrieved_at="2025-12-31T12:00:00Z",
                source_url="https://sam.gov/entity/acme-ai",
                licensing_notes="Public government source.",
            )
        ],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        results_files=[results_path],
        created_at=BUILT_AT,
    )

    assert result.manual_task_count > 0
    assert result.unresolved_manual_task_count == result.manual_task_count - 1
    sam_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "sam_gov"
    )
    assert sam_status.status == ResearchProviderRunStatus.PLANNED
    assert sam_status.collected_count == 1
    assert not any(
        status.provider_id.startswith("import:")
        for status in result.summary.provider_statuses
    )
    assert result.manual_task_queue_path is not None
    queue_payload = json.loads(result.manual_task_queue_path.read_text(encoding="utf-8"))
    assert all(task["provider_id"] != "sam_gov" for task in queue_payload["tasks"])


def test_run_research_workflow_local_public_manual_provider_is_ready_to_import(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sam_results_path = tmp_path / "sam-gov-results.json"
    _write_public_source_results(
        sam_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI SAM.gov result",
                "text": "Acme AI has a public SAM.gov result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://sam.gov/search/?index=opp&keywords=Acme+AI",
            }
        ],
    )

    result = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        sam_gov_results_path=sam_results_path,
        created_at=BUILT_AT,
    )

    sam_status = next(
        status
        for status in result.summary.provider_statuses
        if status.provider_id == "sam_gov"
    )
    assert sam_status.status == ResearchProviderRunStatus.PLANNED
    assert sam_status.collected_count == 1
    assert result.summary.manual_needed_provider_count == result.manual_task_count - 1


def test_prepare_research_plan_rejects_meridian_url_for_multiple_companies(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchPlanError, match="select one company"):
        prepare_research_plan(
            config=AppConfig(data_dir=tmp_path / "data"),
            company_names=["Acme AI", "Beta Robotics"],
            meridian_url="https://portal.angellist.com/m/example/invest",
            created_at=BUILT_AT,
        )


def test_prepare_research_plan_adds_meridian_manual_task(tmp_path: Path) -> None:
    result = prepare_research_plan(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["Acme AI"],
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )

    meridian_task = next(task for task in result.plan.tasks if task.provider_id == "meridian")
    assert meridian_task.url == "https://portal.angellist.com/m/example/invest"
    assert meridian_task.status == ResearchTaskStatus.NEEDS_OPERATOR
    assert "Do not bypass" in meridian_task.licensing_notes


def test_collect_web_research_requires_enabled_web_research(tmp_path: Path) -> None:
    with pytest.raises(WebResearchError, match="Local-only mode is on"):
        collect_web_research(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=True,
                enable_web_research=True,
            ),
        )


def test_collect_web_research_fetches_public_plan_url(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )
    client = _FakeWebResearchClient(
        {
            "https://example.com": WebFetchResponse(
                final_url="https://example.com",
                content_type="text/html; charset=utf-8",
                text=(
                    "<html><head><title>Acme AI Homepage</title>"
                    "<script>ignoreThis()</script></head>"
                    "<body><h1>Acme AI</h1><p>Customer traction from pilots.</p></body></html>"
                ),
            )
        }
    )

    result = collect_web_research(
        config=config,
        plan_path=plan_result.output_path,
        provider_ids=["company_website"],
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == ["https://example.com"]
    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.fetched_count == 1
    assert result.failed_count == 0
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    fetched = saved["results"][0]
    assert fetched["company_name"] == "Acme AI"
    assert "deal_id" not in fetched
    assert fetched["provider_id"] == "company_website"
    assert fetched["title"] == "Acme AI Homepage"
    assert fetched["source_url"] == "https://example.com"
    assert fetched["retrieved_at"] == "2026-01-01T00:00:00Z"
    assert "Customer traction from pilots" in fetched["text"]
    assert "ignoreThis" not in fetched["text"]
    assert fetched["confidence"].startswith("medium:")


def test_collect_web_research_keeps_ingested_deal_ids(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    summary = ingest_folder(root, config=config)
    plan_result = prepare_research_plan(
        config=config,
        website_url="https://example.com",
        created_at=BUILT_AT,
    )
    client = _FakeWebResearchClient(
        {
            "https://example.com": WebFetchResponse(
                final_url="https://example.com",
                content_type="text/plain",
                text="Acme AI has customer pilots.",
            )
        }
    )

    result = collect_web_research(
        config=config,
        plan_path=plan_result.output_path,
        provider_ids=["company_website"],
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["deal_id"] == summary.deals[0].id


def test_collect_web_research_dry_run_does_not_fetch(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )
    client = _FakeWebResearchClient({})

    result = collect_web_research(
        config=config,
        plan_path=plan_result.output_path,
        provider_ids=["company_website"],
        client=client,
        dry_run=True,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.planned_count == 1
    assert result.fetched_count == 0
    assert result.tasks[0].reason == "Dry run: the page was not fetched."


def test_collect_web_research_rejects_unmatched_provider_filter(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )

    with pytest.raises(WebResearchError, match="does not match this research plan"):
        collect_web_research(
            config=config,
            plan_path=plan_result.output_path,
            provider_ids=["not_a_provider"],
            client=_FakeWebResearchClient({}),
            collected_at=BUILT_AT,
        )


def test_collect_web_research_skips_generated_search_pages(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )
    client = _FakeWebResearchClient({})

    result = collect_web_research(
        config=config,
        plan_path=plan_result.output_path,
        provider_ids=["sec_form_d"],
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.skipped_count == 1
    assert "generated search page" in result.tasks[0].reason


def test_collect_web_research_rejects_private_network_urls(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="http://127.0.0.1:8000",
        created_at=BUILT_AT,
    )
    client = _FakeWebResearchClient({})

    result = collect_web_research(
        config=config,
        plan_path=plan_result.output_path,
        provider_ids=["company_website"],
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.failed_count == 1
    assert "private, local, or reserved network address" in result.tasks[0].reason


def test_guarded_web_opener_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    captured_handlers: list[object] = []

    def fake_build_opener(
        *handlers: object,
    ) -> urllib.request.OpenerDirector:
        captured_handlers.extend(handlers)
        return urllib.request.OpenerDirector()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    _build_guarded_opener("company_website")
    proxy_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    proxy_handler = cast(_ProxyHandlerWithProxies, proxy_handlers[0])
    assert proxy_handler.proxies == {}


def test_bound_http_connection_uses_vetted_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_addresses: list[tuple[str, int]] = []

    class FakeSocket:
        def setsockopt(self, *_args: object) -> None:
            return None

    def fake_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        _ = (family, type, proto, flags)
        assert host == "example.com"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    def fake_create_connection(
        address: tuple[str, int],
        timeout: object | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> FakeSocket:
        _ = (timeout, source_address)
        opened_addresses.append(address)
        return FakeSocket()

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    connection = _BoundHTTPConnection(
        "example.com",
        provider_id="company_website",
        timeout=1.0,
    )
    connection._create_connection = fake_create_connection  # type: ignore[assignment]

    connection.connect()

    assert opened_addresses == [("93.184.216.34", 80)]


def test_collect_web_research_command_dry_run_reports_no_network_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )

    result = runner.invoke(
        app,
        [
            "collect-web-research",
            str(plan_result.output_path),
            "--provider",
            "company_website",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Web research preview" in result.output
    assert "1 public web page would be fetched" in result.output
    assert "No websites were contacted" in result.output
    assert not list((tmp_path / "data" / "research-results").glob("*.json"))


def test_collect_web_research_command_exits_nonzero_when_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="http://127.0.0.1:8000",
        created_at=BUILT_AT,
    )

    result = runner.invoke(
        app,
        [
            "collect-web-research",
            str(plan_result.output_path),
            "--provider",
            "company_website",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "1 web research task failed" in result.output
    assert "private, local, or reserved network address" in _plain_cli_output(
        result.output
    )
    assert "Traceback" not in result.output


def test_collect_usaspending_awards_requires_enabled_web_research(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchCollectionError, match="Local-only mode is on"):
        collect_usaspending_awards(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=True,
                enable_web_research=True,
            ),
            company_names=["Acme AI"],
        )


def test_collect_usaspending_awards_writes_private_exact_matches(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", 1): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI",
                        award_id="FAKE-123",
                        generated_internal_id="CONT_AWD_FAKE_123",
                        award_amount=12345.67,
                        description="Research and development support.",
                    ),
                    _usaspending_award(
                        recipient_name="Acme AI Holdings",
                        award_id="FAKE-999",
                        generated_internal_id="CONT_AWD_FAKE_999",
                        award_amount=999.0,
                    ),
                ],
            ),
            ("Beta Robotics", 1): _usaspending_response([]),
        }
    )

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 1), ("Beta Robotics", 5, 1)]
    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.result_count == 1
    assert [(deal.company_name, deal.result_count) for deal in result.deals] == [
        ("Acme AI", 1),
        ("Beta Robotics", 0),
    ]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    prepared = saved["results"][0]
    assert prepared["company_name"] == "Acme AI"
    assert prepared["provider_id"] == "usaspending"
    assert prepared["provider_name"] == "USAspending search"
    assert prepared["title"] == "USAspending award FAKE-123 for Acme AI"
    assert prepared["retrieved_at"] == "2026-01-01T00:00:00Z"
    assert prepared["source_url"] == (
        "https://www.usaspending.gov/award/CONT_AWD_FAKE_123"
    )
    assert "Award amount: $12,345.67." in prepared["text"]
    assert "Research and development support" in prepared["text"]
    assert "Acme AI Holdings" not in prepared["text"]
    assert prepared["confidence"].startswith("medium:")
    assert "USAspending API" in prepared["licensing_notes"]


def test_collect_usaspending_awards_paginates_until_exact_match(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", 1): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI Federal",
                        award_id="FAKE-111",
                        generated_internal_id="CONT_AWD_FAKE_111",
                    ),
                ],
                has_next=True,
            ),
            ("Acme AI", 2): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI",
                        award_id="FAKE-222",
                        generated_internal_id="CONT_AWD_FAKE_222",
                        award_amount=100.0,
                    ),
                ]
            ),
        }
    )

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 1), ("Acme AI", 5, 2)]
    assert result.result_count == 1
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "USAspending award FAKE-222 for Acme AI"


def test_collect_usaspending_awards_does_not_report_no_match_after_page_cap(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", page): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI Federal",
                        award_id=f"FAKE-{page}",
                        generated_internal_id=f"CONT_AWD_FAKE_{page}",
                    )
                ],
                has_next=True,
            )
            for page in range(1, 21)
        }
    )

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.warnings
    assert "did not find exact recipient-name matches" in result.warnings[0]


def test_collect_usaspending_awards_skips_unrelated_malformed_rows(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", 1): _usaspending_raw_response(
                [
                    {"Award ID": "BROKEN-NO-RECIPIENT"},
                    {"Recipient Name": None, "Award ID": "BROKEN-NULL-RECIPIENT"},
                    {"Recipient Name": "Acme AI Federal"},
                    _usaspending_award(
                        recipient_name="Acme AI",
                        award_id="FAKE-321",
                        generated_internal_id="CONT_AWD_FAKE_321",
                    ).model_dump(by_alias=True),
                ]
            ),
        }
    )

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "USAspending award FAKE-321 for Acme AI"


def test_collect_usaspending_awards_fails_on_malformed_exact_row(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", 1): _usaspending_raw_response(
                [{"Recipient Name": "Acme AI"}],
            ),
        }
    )

    with pytest.raises(
        ResearchCollectionError,
        match="missing Award ID, generated_internal_id",
    ):
        collect_usaspending_awards(
            config=config,
            company_names=["Acme AI"],
            limit=5,
            client=client,
            collected_at=BUILT_AT,
        )


def test_collect_usaspending_awards_keeps_matches_when_page_cap_is_hit(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    responses = {
        ("Acme AI", 1): _usaspending_response(
            [
                _usaspending_award(
                    recipient_name="Acme AI",
                    award_id="FAKE-123",
                    generated_internal_id="CONT_AWD_FAKE_123",
                )
            ],
            has_next=True,
        )
    }
    responses.update(
        {
            ("Acme AI", page): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI Federal",
                        award_id=f"FAKE-{page}",
                        generated_internal_id=f"CONT_AWD_FAKE_{page}",
                    )
                ],
                has_next=True,
            )
            for page in range(2, 21)
        }
    )
    client = _FakeUsaspendingAwardsClient(responses)

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert result.output_path is not None
    assert result.warnings
    assert "more fuzzy result pages" in result.warnings[0]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "USAspending award FAKE-123 for Acme AI"


def test_collect_usaspending_awards_preserves_prior_company_results_at_page_cap(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    responses = {
        ("Acme AI", 1): _usaspending_response(
            [
                _usaspending_award(
                    recipient_name="Acme AI",
                    award_id="FAKE-123",
                    generated_internal_id="CONT_AWD_FAKE_123",
                )
            ]
        )
    }
    responses.update(
        {
            ("Beta Robotics", page): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Beta Robotics Federal",
                        award_id=f"FAKE-BETA-{page}",
                        generated_internal_id=f"CONT_AWD_BETA_{page}",
                    )
                ],
                has_next=True,
            )
            for page in range(1, 21)
        }
    )
    client = _FakeUsaspendingAwardsClient(responses)

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert [(deal.company_name, deal.result_count) for deal in result.deals] == [
        ("Acme AI", 1),
        ("Beta Robotics", 0),
    ]
    assert result.warnings
    assert "Beta Robotics" in result.warnings[0]
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    assert saved["results"][0]["company_name"] == "Acme AI"


def test_collect_usaspending_awards_normalizes_saved_company_name(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {
            ("Acme AI", 1): _usaspending_response(
                [
                    _usaspending_award(
                        recipient_name="Acme AI",
                        award_id="FAKE-123",
                        generated_internal_id="CONT_AWD_FAKE_123",
                    )
                ]
            ),
        }
    )

    result = collect_usaspending_awards(
        config=config,
        company_names=[" Acme   AI "],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 1)]
    assert result.deals[0].company_name == "Acme AI"
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["company_name"] == "Acme AI"


def test_usaspending_response_requires_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_usaspending_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module.USASPENDING_AWARDS_ENDPOINT

        def read(self, _size: int) -> bytes:
            return b'{"messages":["schema changed"]}'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(UsaspendingApiError, match="unexpected response"):
        UrlLibUsaspendingAwardsClient().search_awards(
            "Acme AI",
            limit=1,
            page=1,
            timeout_seconds=1.0,
        )


def test_usaspending_response_requires_pagination_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_usaspending_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module.USASPENDING_AWARDS_ENDPOINT

        def read(self, _size: int) -> bytes:
            return b'{"results":[]}'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(UsaspendingApiError, match="unexpected response"):
        UrlLibUsaspendingAwardsClient().search_awards(
            "Acme AI",
            limit=1,
            page=1,
            timeout_seconds=1.0,
        )


def test_usaspending_redirect_handler_validates_before_following() -> None:
    handler = collection_module._UsaspendingRedirectHandler()
    request = urllib.request.Request(collection_module.USASPENDING_AWARDS_ENDPOINT)

    with pytest.raises(UsaspendingApiError, match="USAspending API URL"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1:8000/private",
        )


def test_usaspending_client_vets_dns_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_urls: list[str] = []

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> object:
            _ = timeout
            raise AssertionError("The API should not be opened after DNS validation fails.")

    def fake_ensure_public_endpoint(url: str) -> None:
        checked_urls.append(url)
        raise UsaspendingApiError(
            "USAspending host api.usaspending.gov resolves to a private, local, "
            "or reserved network address."
        )

    monkeypatch.setattr(
        collection_module,
        "_ensure_usaspending_resolved_public_endpoint",
        fake_ensure_public_endpoint,
    )
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(UsaspendingApiError, match="private, local, or reserved"):
        UrlLibUsaspendingAwardsClient().search_awards(
            "Acme AI",
            limit=1,
            page=1,
            timeout_seconds=1.0,
        )

    assert checked_urls == [collection_module.USASPENDING_AWARDS_ENDPOINT]


def test_usaspending_client_wraps_guarded_dns_failures_during_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_usaspending_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> object:
            _ = timeout
            raise WebResearchFetchError(
                "USAspending host api.usaspending.gov resolves to a private, local, "
                "or reserved network address."
            )

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(UsaspendingApiError, match="Could not reach USAspending"):
        UrlLibUsaspendingAwardsClient().search_awards(
            "Acme AI",
            limit=1,
            page=1,
            timeout_seconds=1.0,
        )


def test_usaspending_awards_payload_uses_requested_page() -> None:
    payload = collection_module._usaspending_awards_payload(
        "Acme AI",
        limit=7,
        page=3,
    )

    assert payload["limit"] == 7
    assert payload["page"] == 3
    filters = payload["filters"]
    assert isinstance(filters, dict)
    award_type_codes = filters["award_type_codes"]
    assert isinstance(award_type_codes, list)
    assert "-1" in award_type_codes


def test_collect_usaspending_awards_dry_run_does_not_call_api(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient({})

    result = collect_usaspending_awards(
        config=config,
        company_names=["Acme AI"],
        limit=3,
        dry_run=True,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.dry_run is True
    assert result.deal_count == 1
    assert result.result_count == 0


def test_collect_usaspending_awards_surfaces_api_failures(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeUsaspendingAwardsClient(
        {},
        error=UsaspendingApiError("Could not reach USAspending: timed out"),
    )

    with pytest.raises(ResearchCollectionError, match="Could not reach USAspending"):
        collect_usaspending_awards(
            config=config,
            company_names=["Acme AI"],
            client=client,
            collected_at=BUILT_AT,
        )


def test_usaspending_client_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setattr(
        collection_module,
        "_ensure_usaspending_resolved_public_endpoint",
        lambda _url: None,
    )
    captured_handlers: list[object] = []

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module.USASPENDING_AWARDS_ENDPOINT

        def read(self, _size: int) -> bytes:
            return b'{"results":[],"page_metadata":{"hasNext":false}}'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    def fake_build_opener(
        *handlers: object,
    ) -> FakeOpener:
        captured_handlers.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    response = UrlLibUsaspendingAwardsClient().search_awards(
        "Acme AI",
        limit=1,
        page=1,
        timeout_seconds=1.0,
    )

    assert response.results == []
    proxy_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    proxy_handler = cast(_ProxyHandlerWithProxies, proxy_handlers[0])
    assert proxy_handler.proxies == {}


def test_collect_usaspending_awards_command_dry_run_reports_no_api_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")

    result = runner.invoke(
        app,
        [
            "collect-usaspending-awards",
            "--company",
            "Acme AI",
            "--limit",
            "3",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "USAspending preview" in result.output
    assert "would send 1 company to the USAspending public API" in _plain_cli_output(
        result.output
    )
    assert "up to 3 award records per page for up to 20 pages" in _plain_cli_output(
        result.output
    )
    assert "No API requests were sent" in result.output
    assert not list((tmp_path / "data" / "research-results").glob("*.json"))


def test_collect_usaspending_awards_command_reports_incomplete_search_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_collect_usaspending_awards(
        *,
        config: AppConfig,
        company_names: list[str] | None,
        limit: int,
        dry_run: bool,
    ) -> UsaspendingCollectionRunSummary:
        _ = config
        assert company_names == ["Acme AI"]
        assert limit == 5
        assert dry_run is False
        return UsaspendingCollectionRunSummary(
            output_path=None,
            collected_at=BUILT_AT,
            deals=[
                ResearchCollectionDealSummary(
                    company_name="Acme AI",
                    result_count=0,
                )
            ],
            warnings=[
                "USAspending still had more fuzzy result pages for Acme AI "
                "after Hail Mary checked 20 pages. Hail Mary did not find "
                "exact recipient-name matches, but more USAspending results may exist."
            ],
        )

    monkeypatch.setattr(
        cli_module,
        "collect_usaspending_awards",
        fake_collect_usaspending_awards,
    )

    result = runner.invoke(
        app,
        [
            "collect-usaspending-awards",
            "--company",
            "Acme AI",
            "--limit",
            "5",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "No exact recipient-name USAspending results were found" in output
    assert "Warning: USAspending still had more fuzzy result pages" in output
    assert "No results file was saved" in output


def test_collect_sbir_awards_requires_enabled_web_research(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchCollectionError, match="SBIR/STTR results"):
        collect_sbir_awards(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=True,
                enable_web_research=True,
            ),
            company_names=["Acme AI"],
        )


def test_collect_sbir_awards_writes_private_exact_matches(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {
            ("Acme AI", 0): _sbir_response(
                [
                    _sbir_award(
                        firm="Acme AI",
                        award_title="Autonomous sensor research",
                        award_amount=250000.0,
                        abstract="Research and development support.",
                    ),
                    _sbir_award(
                        firm="Acme AI Holdings",
                        award_title="Related but not exact",
                    ),
                ],
            ),
            ("Beta Robotics", 0): _sbir_response([]),
        }
    )

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 0), ("Beta Robotics", 5, 0)]
    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.result_count == 1
    assert [(deal.company_name, deal.result_count) for deal in result.deals] == [
        ("Acme AI", 1),
        ("Beta Robotics", 0),
    ]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    prepared = saved["results"][0]
    assert prepared["company_name"] == "Acme AI"
    assert prepared["provider_id"] == "sbir"
    assert prepared["provider_name"] == "SBIR/STTR award search"
    assert prepared["title"] == "SBIR/STTR award Autonomous sensor research for Acme AI"
    assert prepared["retrieved_at"] == "2026-01-01T00:00:00Z"
    assert prepared["source_url"] == "https://www.sbir.gov/awards/123"
    assert prepared["source_api"] == (
        "https://api.www.sbir.gov/public/api/awards?firm=Acme+AI&rows=5&start=0"
    )
    assert "Award amount: $250,000.00." in prepared["text"]
    assert "Research and development support" in prepared["text"]
    assert "Acme AI Holdings" not in prepared["text"]
    assert "founder@example.com" not in prepared["text"]
    assert "555-0100" not in prepared["text"]
    assert prepared["confidence"].startswith("medium:")
    assert "SBIR/STTR API" in prepared["licensing_notes"]


def test_collect_sbir_awards_paginates_until_exact_match(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {
            ("Acme AI", 0): _sbir_response(
                [
                    _sbir_award(
                        firm="Acme AI Federal",
                        award_title="Fuzzy first page",
                    ),
                ],
            ),
            ("Acme AI", 1): _sbir_response(
                [
                    _sbir_award(
                        firm="Acme AI",
                        award_title="Exact second page",
                        award_link="https://www.sbir.gov/awards/222",
                    ),
                ],
            ),
        }
    )

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI"],
        limit=1,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 1, 0), ("Acme AI", 1, 1)]
    assert result.result_count == 1
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "SBIR/STTR award Exact second page for Acme AI"


def test_collect_sbir_awards_does_not_report_no_match_after_page_cap(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {
            ("Acme AI", start): _sbir_response(
                [
                    _sbir_award(
                        firm="Acme AI Federal",
                        award_title=f"Fuzzy page {start}",
                        award_link=f"https://www.sbir.gov/awards/fuzzy-{start}",
                    )
                ],
            )
            for start in range(20)
        }
    )

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI"],
        limit=1,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.warnings
    assert "did not find exact firm-name matches" in result.warnings[0]


def test_collect_sbir_awards_skips_unrelated_malformed_rows(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {
            ("Acme AI", 0): _sbir_raw_response(
                [
                    {"award_title": "Broken no firm"},
                    {"firm": None, "award_title": "Broken null firm"},
                    {"firm": "Acme AI Federal"},
                    _sbir_award(
                        firm="Acme AI",
                        award_title="Exact award",
                        award_link="https://www.sbir.gov/awards/321",
                    ).model_dump(),
                ]
            ),
        }
    )

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "SBIR/STTR award Exact award for Acme AI"


def test_collect_sbir_awards_fails_on_malformed_exact_row(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {
            ("Acme AI", 0): _sbir_raw_response(
                [{"firm": "Acme AI"}],
            ),
        }
    )

    with pytest.raises(
        ResearchCollectionError,
        match="award_title must not be blank",
    ):
        collect_sbir_awards(
            config=config,
            company_names=["Acme AI"],
            limit=5,
            client=client,
            collected_at=BUILT_AT,
        )


def test_collect_sbir_awards_keeps_matches_when_page_cap_is_hit(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    responses = {
        ("Acme AI", 0): _sbir_response(
            [
                _sbir_award(
                    firm="Acme AI",
                    award_title="Exact first page",
                    award_link="https://www.sbir.gov/awards/123",
                ),
                _sbir_award(
                    firm="Acme AI Federal",
                    award_title="Fuzzy first page",
                    award_link="https://www.sbir.gov/awards/fuzzy-first",
                ),
            ],
        )
    }
    responses.update(
        {
            ("Acme AI", start): _sbir_response(
                [
                    _sbir_award(
                        firm="Acme AI Federal",
                        award_title=f"Fuzzy page {start}",
                        award_link=f"https://www.sbir.gov/awards/fuzzy-{start}",
                    ),
                    _sbir_award(
                        firm="Acme AI Federal Two",
                        award_title=f"Fuzzy second row {start}",
                        award_link=f"https://www.sbir.gov/awards/fuzzy-two-{start}",
                    ),
                ],
            )
            for start in range(2, 40, 2)
        }
    )
    client = _FakeSbirAwardsClient(responses)

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI"],
        limit=2,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert result.output_path is not None
    assert result.warnings
    assert "more SBIR/STTR results may exist" in result.warnings[0]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "SBIR/STTR award Exact first page for Acme AI"


def test_collect_sbir_awards_preserves_prior_company_results_at_page_cap(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    responses = {
        ("Acme AI", 0): _sbir_response(
            [
                _sbir_award(
                    firm="Acme AI",
                    award_title="Exact award",
                    award_link="https://www.sbir.gov/awards/123",
                )
            ]
        )
    }
    responses.update(
        {
            ("Beta Robotics", start): _sbir_response(
                [
                    _sbir_award(
                        firm="Beta Robotics Federal",
                        award_title=f"Fuzzy beta page {start}",
                        award_link=f"https://www.sbir.gov/awards/beta-{start}",
                    ),
                    _sbir_award(
                        firm="Beta Robotics Federal Two",
                        award_title=f"Fuzzy beta second row {start}",
                        award_link=f"https://www.sbir.gov/awards/beta-two-{start}",
                    ),
                ],
            )
            for start in range(0, 40, 2)
        }
    )
    client = _FakeSbirAwardsClient(responses)

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=2,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.result_count == 1
    assert [(deal.company_name, deal.result_count) for deal in result.deals] == [
        ("Acme AI", 1),
        ("Beta Robotics", 0),
    ]
    assert result.warnings
    assert "Beta Robotics" in result.warnings[0]
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    assert saved["results"][0]["company_name"] == "Acme AI"


def test_collect_sbir_awards_dry_run_does_not_call_api(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient({})

    result = collect_sbir_awards(
        config=config,
        company_names=["Acme AI"],
        limit=3,
        dry_run=True,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.dry_run is True
    assert result.deal_count == 1
    assert result.result_count == 0


def test_collect_sbir_awards_surfaces_api_failures(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSbirAwardsClient(
        {},
        error=SbirApiError("Could not reach SBIR/STTR: timed out"),
    )

    with pytest.raises(ResearchCollectionError, match="Could not reach SBIR/STTR"):
        collect_sbir_awards(
            config=config,
            company_names=["Acme AI"],
            client=client,
            collected_at=BUILT_AT,
        )


def test_sbir_response_requires_result_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_sbir_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)

        def read(self, _size: int) -> bytes:
            return b'["not an object"]'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(SbirApiError, match="not a JSON object"):
        UrlLibSbirAwardsClient().search_awards(
            "Acme AI",
            rows=1,
            start=0,
            timeout_seconds=1.0,
        )


def test_sbir_response_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_sbir_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)

        def read(self, _size: int) -> bytes:
            return b"{not-json"

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(SbirApiError, match="not valid JSON"):
        UrlLibSbirAwardsClient().search_awards(
            "Acme AI",
            rows=1,
            start=0,
            timeout_seconds=1.0,
        )


def test_sbir_redirect_handler_validates_before_following() -> None:
    handler = collection_module._SbirRedirectHandler()
    request = urllib.request.Request(
        collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)
    )

    with pytest.raises(SbirApiError, match="SBIR/STTR API URL"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1:8000/private",
        )


def test_sbir_redirect_handler_rejects_http_api_redirect() -> None:
    handler = collection_module._SbirRedirectHandler()
    request = urllib.request.Request(
        collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)
    )

    with pytest.raises(SbirApiError, match="expected public API endpoint"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://api.www.sbir.gov/public/api/awards?firm=Acme+AI&rows=1&start=0",
        )


def test_sbir_client_vets_dns_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_urls: list[str] = []

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> object:
            _ = timeout
            raise AssertionError("The API should not be opened after DNS validation fails.")

    def fake_ensure_public_endpoint(url: str) -> None:
        checked_urls.append(url)
        raise SbirApiError(
            "SBIR/STTR host api.www.sbir.gov resolves to a private, local, "
            "or reserved network address."
        )

    monkeypatch.setattr(
        collection_module,
        "_ensure_sbir_resolved_public_endpoint",
        fake_ensure_public_endpoint,
    )
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(SbirApiError, match="private, local, or reserved"):
        UrlLibSbirAwardsClient().search_awards(
            "Acme AI",
            rows=1,
            start=0,
            timeout_seconds=1.0,
        )

    assert checked_urls == [
        collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)
    ]


def test_sbir_client_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setattr(
        collection_module,
        "_ensure_sbir_resolved_public_endpoint",
        lambda _url: None,
    )
    captured_handlers: list[object] = []

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._sbir_awards_api_url("Acme AI", rows=1, start=0)

        def read(self, _size: int) -> bytes:
            return b"[]"

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    def fake_build_opener(
        *handlers: object,
    ) -> FakeOpener:
        captured_handlers.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    response = UrlLibSbirAwardsClient().search_awards(
        "Acme AI",
        rows=1,
        start=0,
        timeout_seconds=1.0,
    )

    assert response.results == []
    proxy_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    proxy_handler = cast(_ProxyHandlerWithProxies, proxy_handlers[0])
    assert proxy_handler.proxies == {}


def test_collect_sbir_awards_command_dry_run_reports_no_api_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")

    result = runner.invoke(
        app,
        [
            "collect-sbir-awards",
            "--company",
            "Acme AI",
            "--limit",
            "3",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "SBIR/STTR preview" in result.output
    assert "would send 1 company to the SBIR/STTR public API" in _plain_cli_output(
        result.output
    )
    assert "up to 3 award records per page for up to 20 pages" in _plain_cli_output(
        result.output
    )
    assert "No API requests were sent" in result.output
    assert not list((tmp_path / "data" / "research-results").glob("*.json"))


def test_collect_sbir_awards_command_reports_incomplete_search_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_collect_sbir_awards(
        *,
        config: AppConfig,
        company_names: list[str] | None,
        limit: int,
        dry_run: bool,
    ) -> SbirCollectionRunSummary:
        _ = config
        assert company_names == ["Acme AI"]
        assert limit == 5
        assert dry_run is False
        return SbirCollectionRunSummary(
            output_path=None,
            collected_at=BUILT_AT,
            deals=[
                ResearchCollectionDealSummary(
                    company_name="Acme AI",
                    result_count=0,
                )
            ],
            warnings=[
                "SBIR/STTR still returned full fuzzy result pages for Acme AI "
                "after Hail Mary checked 20 pages. Hail Mary did not find "
                "exact firm-name matches, but more SBIR/STTR results may exist."
            ],
        )

    monkeypatch.setattr(
        cli_module,
        "collect_sbir_awards",
        fake_collect_sbir_awards,
    )

    result = runner.invoke(
        app,
        [
            "collect-sbir-awards",
            "--company",
            "Acme AI",
            "--limit",
            "5",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "No exact firm-name SBIR/STTR results were found" in output
    assert "Warning: SBIR/STTR still returned full fuzzy result pages" in output
    assert "No results file was saved" in output


def test_collect_sec_form_d_filings_requires_enabled_web_research(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchCollectionError, match="SEC Form D results"):
        collect_sec_form_d_filings(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=True,
                enable_web_research=True,
            ),
            company_names=["Acme AI"],
        )


def test_collect_sec_form_d_filings_writes_private_exact_matches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    ingest_folder(root, config=config)
    client = _FakeSecFormDFilingsClient(
        {
            ("Acme AI", 0): _sec_form_d_response(
                [
                    _sec_form_d_filing(
                        issuer_name="Acme AI",
                        total_offering_amount="$1,000,000",
                    ),
                    _sec_form_d_filing(
                        issuer_name="Acme AI Holdings",
                        accession_number="0001234567-26-000002",
                    ),
                ]
            ),
            ("Beta Robotics", 0): _sec_form_d_response([]),
        }
    )

    result = collect_sec_form_d_filings(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 0), ("Beta Robotics", 5, 0)]
    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.result_count == 1
    assert [(deal.company_name, deal.result_count) for deal in result.deals] == [
        ("Acme AI", 1),
        ("Beta Robotics", 0),
    ]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    prepared = saved["results"][0]
    assert prepared["company_name"] == "Acme AI"
    assert prepared["provider_id"] == "sec_form_d"
    assert prepared["provider_name"] == "SEC EDGAR Form D search"
    assert prepared["title"] == "SEC Form D filing for Acme AI"
    assert prepared["retrieved_at"] == "2026-01-01T00:00:00Z"
    assert prepared["source_url"].startswith("https://www.sec.gov/Archives/")
    assert prepared["source_api"].startswith("https://www.sec.gov/cgi-bin/browse-edgar")
    assert "Total offering amount: $1,000,000." in prepared["text"]
    assert "Acme AI Holdings" not in prepared["text"]
    assert "founder@example.com" not in prepared["text"]
    assert "555-0100" not in prepared["text"]
    assert "Main Street" not in prepared["text"]
    assert "raw filings" in prepared["licensing_notes"]

    import_summary = import_research_results(
        config=config,
        results_path=result.output_path,
        dry_run=True,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert import_summary.imported_count == 1


def test_collect_sec_form_d_filings_no_exact_results_names_companies(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSecFormDFilingsClient(
        {
            ("Acme AI", 0): _sec_form_d_response(
                [_sec_form_d_filing(issuer_name="Acme AI Holdings")]
            ),
            ("Beta Robotics", 0): _sec_form_d_response([]),
        }
    )

    result = collect_sec_form_d_filings(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert [deal.company_name for deal in result.deals if deal.result_count == 0] == [
        "Acme AI",
        "Beta Robotics",
    ]
    assert any(match.kind == CompanyMatchKind.RELATED for match in result.match_details)


def test_collect_github_repositories_skips_likely_repository_name_match(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="acme-ai",
                        full_name="unrelated/acme-ai",
                        owner_login="unrelated",
                    )
                ]
            ),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.match_details[0].kind == CompanyMatchKind.LIKELY
    assert result.match_details[0].import_ready is False


def test_collect_sec_form_d_filings_dry_run_does_not_call_api(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSecFormDFilingsClient({})

    result = collect_sec_form_d_filings(
        config=config,
        company_names=["Acme AI"],
        limit=3,
        dry_run=True,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.dry_run is True
    assert result.deal_count == 1
    assert result.result_count == 0


def test_collect_sec_form_d_filings_requires_sec_user_agent_for_live_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HAILMARY_SEC_USER_AGENT", raising=False)
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )

    with pytest.raises(ResearchCollectionError, match="HAILMARY_SEC_USER_AGENT"):
        collect_sec_form_d_filings(
            config=config,
            company_names=["Acme AI"],
            collected_at=BUILT_AT,
        )


def test_sec_form_d_user_agent_requires_contact_email() -> None:
    with pytest.raises(ResearchCollectionError, match="contact email"):
        collection_module._validate_sec_form_d_user_agent("Hail Mary diligence")


def test_collect_sec_form_d_filings_surfaces_api_failures(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSecFormDFilingsClient(
        {},
        error=SecFormDApiError("Could not reach SEC EDGAR: timed out"),
    )

    with pytest.raises(ResearchCollectionError, match="Could not reach SEC EDGAR"):
        collect_sec_form_d_filings(
            config=config,
            company_names=["Acme AI"],
            client=client,
            collected_at=BUILT_AT,
        )


def test_collect_sec_form_d_filings_rejects_bad_source_url(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeSecFormDFilingsClient(
        {
            ("Acme AI", 0): _sec_form_d_raw_response(
                [
                    {
                        **_sec_form_d_filing(issuer_name="Acme AI").model_dump(),
                        "source_url": "https://example.com/sec-form-d.txt",
                    }
                ]
            )
        }
    )

    with pytest.raises(ResearchCollectionError, match="source_url must use"):
        collect_sec_form_d_filings(
            config=config,
            company_names=["Acme AI"],
            limit=5,
            client=client,
            collected_at=BUILT_AT,
        )


def test_sec_form_d_response_requires_pagination_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_sec_form_d_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._sec_form_d_atom_api_url(
                "Acme AI",
                count=1,
                start=0,
            )

        def read(self, _size: int) -> bytes:
            return b"<feed></feed>"

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(SecFormDApiError, match="missing pagination metadata"):
        UrlLibSecFormDFilingsClient(
            user_agent="Hail Mary tests tests@example.com",
            request_interval_seconds=0,
        ).search_filings(
            "Acme AI",
            count=1,
            start=0,
            timeout_seconds=1.0,
        )


def test_sec_form_d_atom_parser_skips_malformed_related_fuzzy_entry() -> None:
    related_href = (
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000001/0001234567-26-000001-index.htm"
    )
    exact_href = (
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000002/0001234567-26-000002-index.htm"
    )
    xml_text = f"""
    <feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>2</opensearch:totalResults>
      <opensearch:startIndex>0</opensearch:startIndex>
      <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
      <entry>
        <title>D - Acme AI Holdings (0001234567)</title>
        <category term="D" />
        <link href="{related_href}" />
      </entry>
      <entry>
        <title>D - Acme AI (0001234568)</title>
        <category term="D" />
        <link href="{exact_href}" />
      </entry>
    </feed>
    """
    fetched_urls: list[str] = []

    def fake_fetch(source_url: str) -> dict[str, str | list[str] | None]:
        fetched_urls.append(source_url)
        return {"issuer_name": "Acme AI", "federal_exemptions": ["06b"]}

    response = collection_module._parse_sec_form_d_atom_response(
        xml_text,
        requested_company_name="Acme AI",
        source_api=collection_module._sec_form_d_atom_api_url(
            "Acme AI",
            count=2,
            start=0,
        ),
        fetch_submission=fake_fetch,
    )

    assert fetched_urls == [
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000002/0001234567-26-000002.txt"
    ]
    assert len(response.results) == 1
    assert response.results[0]["issuer_name"] == "Acme AI"


def test_sec_form_d_atom_parser_fails_malformed_exact_entry() -> None:
    exact_href = (
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000002/0001234567-26-000002-index.htm"
    )
    xml_text = f"""
    <feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>1</opensearch:totalResults>
      <opensearch:startIndex>0</opensearch:startIndex>
      <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
      <entry>
        <title>D - Acme AI (0001234568)</title>
        <category term="D" />
        <link href="{exact_href}" />
      </entry>
    </feed>
    """

    def fake_fetch(_source_url: str) -> dict[str, str | list[str] | None]:
        raise SecFormDApiError("SEC Form D filing metadata could not be parsed as XML.")

    with pytest.raises(SecFormDApiError, match="could not be parsed"):
        collection_module._parse_sec_form_d_atom_response(
            xml_text,
            requested_company_name="Acme AI",
            source_api=collection_module._sec_form_d_atom_api_url(
                "Acme AI",
                count=1,
                start=0,
            ),
            fetch_submission=fake_fetch,
        )


def test_sec_form_d_complete_submission_url_keeps_dashed_accession_filename() -> None:
    source_url = collection_module._sec_form_d_complete_submission_url(
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000001/0001234567-26-000001-index.htm",
        accession_number="0001234567-26-000001",
    )

    assert source_url.endswith("/0001234567-26-000001.txt")


def test_sec_form_d_complete_submission_url_handles_primary_doc_subdirectory() -> None:
    source_url = collection_module._sec_form_d_complete_submission_url(
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000001/xslFormDX01/primary_doc.xml",
        accession_number="0001234567-26-000001",
    )

    assert source_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000001/0001234567-26-000001.txt"
    )


def test_sec_form_d_redirect_handler_rejects_http_redirect() -> None:
    handler = collection_module._SecFormDRedirectHandler()
    request = urllib.request.Request(
        collection_module._sec_form_d_atom_api_url("Acme AI", count=1, start=0)
    )

    with pytest.raises(SecFormDApiError, match="must stay on HTTPS"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
        )


def test_sec_form_d_redirect_handler_rejects_atom_to_archive_redirect() -> None:
    handler = collection_module._SecFormDRedirectHandler()
    request = urllib.request.Request(
        collection_module._sec_form_d_atom_api_url("Acme AI", count=1, start=0)
    )

    with pytest.raises(SecFormDApiError, match="expected public API endpoint"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://www.sec.gov/Archives/edgar/data/1234567890/"
            "000123456726000001/0001234567-26-000001.txt",
        )


def test_sec_form_d_redirect_handler_rejects_detail_to_search_redirect() -> None:
    handler = collection_module._SecFormDRedirectHandler()
    request = urllib.request.Request(
        "https://www.sec.gov/Archives/edgar/data/1234567890/"
        "000123456726000001/0001234567-26-000001.txt"
    )

    with pytest.raises(SecFormDApiError, match="archive path"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            collection_module._sec_form_d_atom_api_url("Acme AI", count=1, start=0),
        )


def test_sec_form_d_client_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    captured_handlers: list[object] = []

    def fake_build_opener(
        *handlers: object,
    ) -> urllib.request.OpenerDirector:
        captured_handlers.extend(handlers)
        return urllib.request.OpenerDirector()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    collection_module._build_sec_form_d_api_opener()

    proxy_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    proxy_handler = cast(_ProxyHandlerWithProxies, proxy_handlers[0])
    assert proxy_handler.proxies == {}


def test_collect_github_repositories_requires_enabled_web_research(
    tmp_path: Path,
) -> None:
    with pytest.raises(ResearchCollectionError, match="GitHub results"):
        collect_github_repositories(
            config=AppConfig(
                data_dir=tmp_path / "data",
                local_only=True,
                enable_web_research=True,
            ),
            company_names=["Acme AI"],
        )


def test_collect_github_repositories_writes_private_exact_matches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    ingest_folder(root, config=config)
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="product",
                        full_name="acme-ai/product",
                        owner_login="acme-ai",
                    ),
                    _github_repository(
                        name="acme-ai-related",
                        full_name="synthetic/acme-ai-related",
                        owner_login="synthetic",
                    ),
                ]
            ),
            ("Beta Robotics", 1): _github_repository_response([]),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 1), ("Beta Robotics", 5, 1)]
    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.result_count == 1
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
    prepared = saved["results"][0]
    assert prepared["company_name"] == "Acme AI"
    assert prepared["provider_id"] == "github"
    assert prepared["provider_name"] == "GitHub repository search"
    assert prepared["title"] == "GitHub repository acme-ai/product"
    assert prepared["source_url"] == "https://github.com/acme-ai/product"
    assert prepared["source_api"] == "https://api.github.com/repos/acme-ai/product"
    assert "Repository: acme-ai/product." in prepared["text"]
    assert "acme-ai-related" not in prepared["text"]
    assert "did not clone code" in prepared["licensing_notes"]

    import_summary = import_research_results(
        config=config,
        results_path=result.output_path,
        dry_run=True,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert import_summary.imported_count == 1


def test_collect_github_repositories_no_exact_results_names_companies(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="acme-ai-related",
                        full_name="synthetic/acme-ai-related",
                        owner_login="synthetic",
                    )
                ]
            ),
            ("Beta Robotics", 1): _github_repository_response([]),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI", "Beta Robotics"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert [deal.company_name for deal in result.deals if deal.result_count == 0] == [
        "Acme AI",
        "Beta Robotics",
    ]


def test_collect_github_repositories_dry_run_does_not_call_api(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient({})

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI"],
        limit=3,
        dry_run=True,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == []
    assert result.output_path is None
    assert result.dry_run is True
    assert result.deal_count == 1
    assert result.result_count == 0


def test_collect_github_repositories_paginates_when_response_has_next(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    ingest_folder(root, config=config)
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response([], has_next=True),
            ("Acme AI", 2): _github_repository_response(
                [
                    _github_repository(
                        name="product",
                        full_name="acme-ai/product",
                        owner_login="acme-ai",
                    )
                ]
            ),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert client.calls == [("Acme AI", 5, 1), ("Acme AI", 5, 2)]
    assert result.result_count == 1


def test_collect_github_repositories_prioritizes_owner_matches_before_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    ingest_folder(root, config=config)
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="acme-ai",
                        full_name="unrelated/acme-ai",
                        owner_login="unrelated",
                    ),
                    _github_repository(
                        name="owner-tool",
                        full_name="acme-ai/owner-tool",
                        owner_login="acme-ai",
                    ),
                ]
            ),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI"],
        limit=1,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["source_url"] == "https://github.com/acme-ai/owner-tool"


def test_collect_github_repositories_skips_repository_name_only_match(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_repository_response(
                [
                    _github_repository(
                        name="acme-ai",
                        full_name="unrelated/acme-ai",
                        owner_login="unrelated",
                    )
                ]
            ),
        }
    )

    result = collect_github_repositories(
        config=config,
        company_names=["Acme AI"],
        limit=5,
        client=client,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.match_details[0].kind == CompanyMatchKind.LIKELY
    assert result.match_details[0].import_ready is False


def test_collect_github_repositories_surfaces_api_failures(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient(
        {},
        error=GitHubApiError("Could not reach GitHub: timed out"),
    )

    with pytest.raises(ResearchCollectionError, match="Could not reach GitHub"):
        collect_github_repositories(
            config=config,
            company_names=["Acme AI"],
            client=client,
            collected_at=BUILT_AT,
        )


def test_collect_github_repositories_rejects_bad_source_url(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        local_only=False,
        enable_web_research=True,
    )
    client = _FakeGitHubRepositorySearchClient(
        {
            ("Acme AI", 1): _github_raw_repository_response(
                [
                    {
                        **_github_repository(
                            name="product",
                            full_name="acme-ai/product",
                            owner_login="acme-ai",
                        ),
                        "html_url": "https://example.com/acme-ai/product",
                    }
                ]
            )
        }
    )

    with pytest.raises(ResearchCollectionError, match="source_url must use"):
        collect_github_repositories(
            config=config,
            company_names=["Acme AI"],
            limit=5,
            client=client,
            collected_at=BUILT_AT,
        )


def test_github_response_requires_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_github_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._github_repository_search_api_url(
                "Acme AI",
                per_page=1,
                page=1,
            )

        def read(self, _size: int) -> bytes:
            return b'{"total_count":0}'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(GitHubApiError, match="unexpected response"):
        UrlLibGitHubRepositorySearchClient(
            request_interval_seconds=0,
        ).search_repositories(
            "Acme AI",
            per_page=1,
            page=1,
            timeout_seconds=1.0,
        )


def test_github_response_requires_pagination_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_github_resolved_public_endpoint",
        lambda _url: None,
    )

    class FakeResponse:
        headers: object

        def __init__(self) -> None:
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return collection_module._github_repository_search_api_url(
                "Acme AI",
                per_page=1,
                page=1,
            )

        def read(self, _size: int) -> bytes:
            return b'{"items":[]}'

    class FakeOpener:
        def open(
            self,
            _request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            return FakeResponse()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    with pytest.raises(GitHubApiError, match="unexpected response"):
        UrlLibGitHubRepositorySearchClient(
            request_interval_seconds=0,
        ).search_repositories(
            "Acme AI",
            per_page=1,
            page=1,
            timeout_seconds=1.0,
        )


def test_github_repository_search_api_urls_include_owner_scopes() -> None:
    urls = collection_module._github_repository_search_api_urls(
        "Acme AI",
        per_page=5,
        page=1,
    )

    assert "q=user%3Aacme-ai+fork%3Afalse" in urls[0]
    assert "q=org%3Aacme-ai+fork%3Afalse" in urls[1]
    assert "q=Acme+AI+in%3Aname+fork%3Afalse" in urls[2]
    assert any("q=Acme+AI+in%3Aname+fork%3Afalse" in url for url in urls)
    assert any("q=user%3Aacme-ai+fork%3Afalse" in url for url in urls)
    assert any("q=org%3Aacme-ai+fork%3Afalse" in url for url in urls)


def test_github_client_paces_generated_search_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_module,
        "_ensure_github_resolved_public_endpoint",
        lambda _url: None,
    )
    sleeps: list[float] = []
    opened_urls: list[str] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    class FakeResponse:
        headers: object

        def __init__(self, url: str) -> None:
            self._url = url
            self.headers = _FakeHeaders()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return self._url

        def read(self, _size: int) -> bytes:
            return b'{"total_count":0,"incomplete_results":false,"items":[]}'

    class FakeOpener:
        def open(
            self,
            request: urllib.request.Request,
            *,
            timeout: float,
        ) -> FakeResponse:
            _ = timeout
            opened_urls.append(request.full_url)
            return FakeResponse(request.full_url)

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    response = UrlLibGitHubRepositorySearchClient(
        request_interval_seconds=0.25,
    ).search_repositories(
        "Acme AI",
        per_page=1,
        page=1,
        timeout_seconds=1.0,
    )

    assert response.result_count == 0
    assert len(opened_urls) == 3
    assert sleeps == [0.25, 0.25, 0.25]


def test_github_redirect_handler_rejects_outside_host() -> None:
    handler = collection_module._GitHubRedirectHandler()
    request = urllib.request.Request(
        collection_module._github_repository_search_api_url(
            "Acme AI",
            per_page=1,
            page=1,
        )
    )

    with pytest.raises(GitHubApiError, match="GitHub redirect URL"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://example.com/search/repositories",
        )


def test_github_redirect_handler_rejects_non_search_api_redirect() -> None:
    handler = collection_module._GitHubRedirectHandler()
    request = urllib.request.Request(
        collection_module._github_repository_search_api_url(
            "Acme AI",
            per_page=1,
            page=1,
        )
    )

    with pytest.raises(GitHubApiError, match="expected public API endpoint"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://api.github.com/repos/synthetic/acme-ai",
        )


def test_github_client_disables_ambient_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    captured_handlers: list[object] = []

    def fake_build_opener(
        *handlers: object,
    ) -> urllib.request.OpenerDirector:
        captured_handlers.extend(handlers)
        return urllib.request.OpenerDirector()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    collection_module._build_github_api_opener()

    proxy_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers
    proxy_handler = cast(_ProxyHandlerWithProxies, proxy_handlers[0])
    assert proxy_handler.proxies == {}


def test_collect_sec_form_d_filings_command_dry_run_reports_no_sec_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")

    result = runner.invoke(
        app,
        [
            "collect-sec-form-d-filings",
            "--company",
            "Acme AI",
            "--limit",
            "3",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "SEC Form D preview" in output
    assert "would send 1 company to SEC EDGAR public filing search" in output
    assert "up to 3 filing records per page for up to 20 pages" in output
    assert "No SEC requests were sent" in output
    assert not list((tmp_path / "data" / "research-results").glob("*.json"))


def test_collect_github_repositories_command_dry_run_reports_no_api_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_ENABLE_WEB_RESEARCH", "true")

    result = runner.invoke(
        app,
        [
            "collect-github-repositories",
            "--company",
            "Acme AI",
            "--limit",
            "3",
            "--dry-run",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "GitHub repository preview" in output
    assert "would send 1 company to the GitHub public repository search API" in output
    assert "repository-name, user-owner, and organization-owner searches" in output
    assert "up to 3 repository records per page for up to 5 pages" in output
    assert "No GitHub API requests were sent" in output
    assert not list((tmp_path / "data" / "research-results").glob("*.json"))


def test_collect_github_repositories_command_reports_likely_skipped_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_collect_github_repositories(
        *,
        config: AppConfig,
        company_names: list[str] | None,
        limit: int,
        dry_run: bool,
    ) -> GitHubRepositoryCollectionRunSummary:
        _ = (config, limit, dry_run)
        assert company_names == ["Acme AI"]
        return GitHubRepositoryCollectionRunSummary(
            output_path=None,
            collected_at=BUILT_AT,
            deals=[
                ResearchCollectionDealSummary(
                    company_name="Acme AI",
                    result_count=0,
                )
            ],
            match_details=[
                CompanyMatch(
                    requested_name="Acme AI",
                    candidate_name="other/acme-ai",
                    kind=CompanyMatchKind.LIKELY,
                    reason=(
                        "GitHub repository name matches, but the owner does not. "
                        "Operator validation is required before import."
                    ),
                    normalized_requested="acme ai",
                    normalized_candidate="acme ai",
                )
            ],
        )

    monkeypatch.setattr(
        cli_module,
        "collect_github_repositories",
        fake_collect_github_repositories,
    )

    result = runner.invoke(
        app,
        [
            "collect-github-repositories",
            "--company",
            "Acme AI",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "No exact GitHub owner result" in output
    assert "skipped likely match other/acme-ai for Acme AI" in output
    assert "Operator validation is required before import" in output


def test_collect_sec_form_d_filings_command_success_reports_import_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "data" / "research-results" / "sec-form-d-results.json"

    def fake_collect_sec_form_d_filings(
        *,
        config: AppConfig,
        company_names: list[str] | None,
        limit: int,
        dry_run: bool,
    ) -> SecFormDCollectionRunSummary:
        _ = config
        assert company_names == ["Acme AI"]
        assert limit == 5
        assert dry_run is False
        return SecFormDCollectionRunSummary(
            output_path=output_path,
            collected_at=BUILT_AT,
            deals=[
                ResearchCollectionDealSummary(
                    company_name="Acme AI",
                    result_count=1,
                )
            ],
        )

    monkeypatch.setattr(
        cli_module,
        "collect_sec_form_d_filings",
        fake_collect_sec_form_d_filings,
    )

    result = runner.invoke(
        app,
        [
            "collect-sec-form-d-filings",
            "--company",
            "Acme AI",
            "--limit",
            "5",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = _plain_cli_output(result.output)
    assert "SEC Form D results collected" in output
    assert "Collected 1 SEC Form D result for 1 company" in output
    assert "import-research-results" in output
    assert "raw filings or contact details" in output


def test_collect_github_repositories_command_failure_has_plain_english_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_collect_github_repositories(
        *,
        config: AppConfig,
        company_names: list[str] | None,
        limit: int,
        dry_run: bool,
    ) -> GitHubRepositoryCollectionRunSummary:
        _ = (config, company_names, limit, dry_run)
        raise ResearchCollectionError("GitHub returned HTTP 403.")

    monkeypatch.setattr(
        cli_module,
        "collect_github_repositories",
        fake_collect_github_repositories,
    )

    result = runner.invoke(
        app,
        [
            "collect-github-repositories",
            "--company",
            "Acme AI",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "GitHub returned HTTP 403" in result.output
    assert "Traceback" not in result.output


def test_prepare_meridian_workflow_writes_private_workflow_and_template(
    tmp_path: Path,
) -> None:
    result = prepare_meridian_workflow(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/acme-ai/invest",
        created_at=BUILT_AT,
    )

    assert result.output_path.parent == tmp_path / "data" / "meridian-workflows"
    assert result.result_template_path.parent == (
        tmp_path / "data" / "research-results-templates"
    )
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.result_template_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.result_template_path.stat().st_mode) == 0o600

    workflow = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert workflow["version"] == "3"
    assert workflow["company_name"] == "Acme AI"
    assert workflow["meridian_url"] == "https://portal.angellist.com/m/acme-ai/invest"
    assert workflow["result_template_path"] == str(result.result_template_path)
    assert "import-research-results" in workflow["import_command"]
    assert workflow["dry_run_command"] == workflow["import_command"]
    assert shlex.split(workflow["import_command"]) == [
        "hailmary",
        "import-research-results",
        str(result.result_template_path),
        "--data-dir",
        str(tmp_path / "data"),
        "--dry-run",
    ]
    assert any("Do not bypass" in rule for rule in workflow["safety_rules"])
    assert any("normal sign-in" in step for step in workflow["manual_steps"])
    assert any("raw page dump" in step for step in workflow["manual_steps"])
    assert "cookies" in workflow["do_not_collect"]
    assert "auth headers" in workflow["do_not_collect"]
    assert "screenshots" in workflow["do_not_collect"]
    assert "screenshots unless explicitly approved later" not in workflow["do_not_collect"]
    assert "raw full-page HTML" in workflow["do_not_collect"]
    assert "valuation cap" in workflow["term_definitions"]
    assert "pre-money valuation" in workflow["term_definitions"]
    assert "lead investor" in workflow["term_definitions"]
    assert "SAFE" in workflow["term_definitions"]
    assert "convertible note" in workflow["term_definitions"]
    assert "ARR" in workflow["term_definitions"]
    assert "MRR" in workflow["term_definitions"]
    assert "allocation" in workflow["term_definitions"]
    assert "target raise" in workflow["term_definitions"]
    assert "closing date" in workflow["term_definitions"]
    assert workflow["required_result_fields"]["source_url"].startswith("Keep the generated")
    assert "text" in workflow["required_result_fields"]
    assert "retrieved_at" in workflow["required_result_fields"]
    assert "confidence" in workflow["required_result_fields"]
    assert "licensing_notes" in workflow["required_result_fields"]
    assert "Do not enter INVEST, PASS" in workflow["recommendation_policy"]
    assert any("Do not write INVEST, PASS" in step for step in workflow["manual_steps"])
    assert set(workflow["required_when_visible_sections"]) == {
        "deal_terms",
        "traction_customer_evidence",
        "revenue_evidence",
        "founder_team_facts",
        "risks_disclaimers",
        "deadline_allocation",
    }
    assert set(workflow["optional_when_visible_sections"]) == {
        "product",
        "market",
        "use_of_funds",
    }
    assert any("short allowed fact" in item for item in workflow["before_import_checklist"])
    assert any("No screenshots" in item for item in workflow["before_import_checklist"])
    assert any("source_url" in item for item in workflow["before_import_checklist"])
    assert any("dry-run command" in item for item in workflow["before_import_checklist"])
    assert "Minimum investment" in workflow["recommended_facts"]
    assert "Revenue claims" in workflow["recommended_facts"]
    assert "Product facts, if visible" in workflow["recommended_facts"]
    assert "Use of funds, if visible" in workflow["recommended_facts"]

    template = json.loads(result.result_template_path.read_text(encoding="utf-8"))
    assert list(template) == ["results"]
    assert len(template["results"]) == len(workflow["recommended_facts"])
    titles = [row["title"] for row in template["results"]]
    assert "Meridian: Company name" in titles
    assert "Meridian: Valuation cap or pre-money valuation" in titles
    assert "Meridian: Closing date or allocation deadline, if visible" in titles
    row = template["results"][0]
    assert set(row) == {
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
        "source_reliability",
        "identity_match_kind",
        "identity_match_reason",
    }
    assert row["company_name"] == "Acme AI"
    assert row["provider_id"] == "meridian"
    assert row["provider_name"] == "Meridian deal page"
    assert row["title"].startswith("Meridian: ")
    assert row["text"] == ""
    assert row["source_url"] == "https://portal.angellist.com/m/acme-ai/invest"
    assert row["retrieved_at"] == ""
    assert row["confidence"].startswith("Replace with confidence")
    assert row["source_kind"] == "meridian"
    assert row["document_type"] == "platform_deal_page"
    assert row["source_reliability"] == ""
    assert row["identity_match_kind"] == ""
    assert row["identity_match_reason"] == ""
    assert "Do not bypass" in row["licensing_notes"]
    assert "Generated by Hail Mary prepare-meridian-workflow" in row["licensing_notes"]
    assert "Generated Meridian placeholder" in row["licensing_notes"]
    assert "Keep source_url unchanged" in row["licensing_notes"]


def test_clean_meridian_url_returns_canonical_safe_url() -> None:
    assert (
        clean_meridian_url("https://PORTAL.ANGELLIST.com/m/acme-ai/invest")
        == "https://portal.angellist.com/m/acme-ai/invest"
    )


@pytest.mark.parametrize(
    "meridian_url",
    [
        "",
        " https://portal.angellist.com/m/acme-ai/invest",
        "https://portal.angellist.com/m/acme-ai/invest ",
        "mailto:founder@example.com",
        "http://portal.angellist.com/m/acme-ai/invest",
        "https://example.com/m/acme-ai/invest",
        "https://portal.angellist.com/not-m/acme-ai/invest",
        "https://portal.angellist.com/m/acme-ai/profile",
        "https://portal.angellist.com//m/acme-ai/invest",
        "https://portal.angellist.com/m//acme-ai/invest",
        "https://portal.angellist.com/m/acme-ai//invest",
        "https://portal.angellist.com/m/acme-ai/invest/",
        "https://portal.angellist.com/m/acme-ai/session-token/invest",
        "https://user:token@portal.angellist.com/m/acme-ai/invest",
        "https://portal.angellist.com%5B/m/acme-ai/invest",
        "https://portal.angellist.com:bad/m/acme-ai/invest",
        "https://portal.angellist.com:444/m/acme-ai/invest",
        "https://portal.angellist.com:/m/acme-ai/invest",
        "https://portal.angellist.com/m/acme ai/invest",
        "https://portal.angellist.com/m/acme-ai;jsessionid=secret/invest",
        "https://portal.angellist.com/m/acme-ai/invest;jsessionid=secret",
        "https://portal.angellist.com/m/acme%3Bjsessionid=secret/invest",
        "https://portal.angellist.com/m/acme%3Ftoken=secret/invest",
        "https://portal.angellist.com/m/acme%253Ftoken=secret/invest",
        "https://portal.angellist.com/m/acme%2525253Ftoken=secret/invest",
        "https://portal.angellist.com/m/acme-ai/invest?token=secret",
        "https://portal.angellist.com/m/acme-ai/invest?signature=secret",
        "https://portal.angellist.com/m/acme-ai/invest#details",
    ],
)
def test_prepare_meridian_workflow_rejects_malformed_meridian_url(
    tmp_path: Path,
    meridian_url: str,
) -> None:
    with pytest.raises(MeridianWorkflowError):
        prepare_meridian_workflow(
            config=AppConfig(data_dir=tmp_path / "data"),
            company_name="Acme AI",
            meridian_url=meridian_url,
            created_at=BUILT_AT,
        )


def test_prepare_meridian_workflow_command_writes_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-meridian-workflow",
            "--company",
            "Acme AI",
            "--meridian-url",
            "https://portal.angellist.com/m/acme-ai/invest",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Meridian workflow prepared" in result.output
    assert "Prepared a Meridian manual workflow for Acme AI" in result.output
    assert "did not open Meridian, sign in, bypass access controls" in result.output
    assert "scrape pages" in result.output
    assert "short allowed evidence snippets" in result.output
    assert "not screenshots, raw page dumps, hidden page data" in result.output
    assert "browser profiles, cookies, tokens, signed URLs" in result.output
    assert "before-import checklist" in result.output
    assert "import-research-results" in result.output
    assert (tmp_path / "data" / "meridian-workflows").is_dir()
    workflows = list((tmp_path / "data" / "meridian-workflows").glob("*.json"))
    templates = list((tmp_path / "data" / "research-results-templates").glob("*.json"))
    assert len(workflows) == 1
    assert len(templates) == 1
    assert workflows[0].name in result.output.replace("\n", "")
    assert templates[0].name in result.output.replace("\n", "")


def test_prepare_meridian_workflow_command_quotes_printed_import_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "Hail Mary Data"

    result = runner.invoke(
        app,
        [
            "prepare-meridian-workflow",
            "--company",
            "Acme AI",
            "--meridian-url",
            "https://portal.angellist.com/m/acme-ai/invest",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    output = result.output.replace("\n", "")
    assert f"--data-dir {shlex.quote(str(data_dir))} --dry-run" in output
    assert "Hail Mary Data --dry-run" not in output


def test_prepare_meridian_workflow_quotes_space_containing_paths(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "Hail Mary Data"

    result = prepare_meridian_workflow(
        config=AppConfig(data_dir=data_dir),
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/acme-ai/invest",
        created_at=BUILT_AT,
    )

    assert "Hail Mary Data" in result.workflow.import_command
    assert result.workflow.dry_run_command == result.workflow.import_command
    assert shlex.split(result.workflow.import_command) == [
        "hailmary",
        "import-research-results",
        str(result.result_template_path),
        "--data-dir",
        str(data_dir),
        "--dry-run",
    ]


def test_prepare_meridian_workflow_command_has_plain_english_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-meridian-workflow",
            "--company",
            "Acme AI",
            "--meridian-url",
            "https://example.com/m/acme-ai/invest",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "portal.angellist.com" in result.output
    assert "Traceback" not in result.output


def test_prepare_research_plan_uses_latest_ingestion_summary(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "IngestedCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / "data")
    ingest_folder(root, config=config)

    result = prepare_research_plan(config=config, created_at=BUILT_AT)

    assert result.plan.deals[0].company_name == "IngestedCo"
    assert all(task.company_name == "IngestedCo" for task in result.plan.tasks)


def test_prepare_research_plan_requires_company_or_ingestion(tmp_path: Path) -> None:
    with pytest.raises(ResearchPlanError, match="No ingested deals"):
        prepare_research_plan(
            config=AppConfig(data_dir=tmp_path / "data"),
            created_at=BUILT_AT,
        )


def test_list_research_providers_command_hides_paid_by_default() -> None:
    result = runner.invoke(app, ["list-research-providers"])

    assert result.exit_code == 0, result.output
    assert "Research providers" in result.output
    assert "Free and public sources" in result.output
    assert "Meridian deal page" in result.output
    assert "Crunchbase" not in result.output


def test_list_research_providers_command_prints_json() -> None:
    result = runner.invoke(app, ["list-research-providers", "--include-paid", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert any(provider["id"] == "crunchbase" for provider in payload)


def test_prepare_research_plan_command_writes_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-research-plan",
            "--company",
            "Acme AI",
            "--website",
            "https://example.com",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Research plan prepared" in result.output
    assert "Prepared an external research plan for 1 deal" in result.output
    assert "No websites, APIs, paid databases, or Meridian pages were contacted" in result.output
    assert (tmp_path / "data" / "research-plans").is_dir()


def test_prepare_research_plan_command_has_plain_english_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-research-plan",
            "--company",
            "Acme AI",
            "--company",
            "Beta Robotics",
            "--website",
            "https://example.com",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "Use --website only when the plan has exactly one company" in result.output
    assert "Traceback" not in result.output


def test_prepare_research_results_template_writes_private_fillable_file(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        website_url="https://example.com",
        created_at=BUILT_AT,
    )

    result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert result.plan_path == plan_result.output_path.resolve(strict=False)
    assert result.result_count == plan_result.task_count
    assert result.output_path.parent == tmp_path / "data" / "research-results-templates"
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert list(saved) == ["results"]
    assert len(saved["results"]) == plan_result.task_count

    website_result = next(
        item for item in saved["results"] if item["provider_id"] == "company_website"
    )
    assert website_result == {
        "deal_id": "",
        "company_name": "Acme AI",
        "provider_id": "company_website",
        "provider_name": "Company website",
        "title": "",
        "text": "",
        "retrieved_at": "",
        "source_url": "",
        "source_api": "",
        "confidence": "",
        "licensing_notes": (
            "Use public pages for diligence notes. Save the exact URL and access time "
            "before turning anything into evidence."
        ),
        "source_kind": "web",
        "document_type": "web_page",
        "source_reliability": "",
        "identity_match_kind": "",
        "identity_match_reason": "",
    }


def test_prepare_research_results_template_keeps_ingested_deal_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / "data")
    summary = ingest_folder(root, config=config)
    plan_result = prepare_research_plan(config=config, created_at=BUILT_AT)

    result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert {item["deal_id"] for item in saved["results"]} == {summary.deals[0].id}


def test_prepare_research_results_template_marks_meridian_as_platform_source(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )

    result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    meridian_result = next(
        item for item in saved["results"] if item["provider_id"] == "meridian"
    )
    assert meridian_result["source_kind"] == "meridian"
    assert meridian_result["document_type"] == "platform_deal_page"
    assert meridian_result["source_url"] == ""


def test_prepare_research_results_template_uses_latest_plan_when_omitted(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    prepare_research_plan(
        config=config,
        company_names=["OlderCo"],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    latest_plan = prepare_research_plan(
        config=config,
        company_names=["NewerCo"],
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = prepare_research_results_template(
        config=config,
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert result.plan_path == latest_plan.output_path.resolve(strict=False)
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert {item["company_name"] for item in saved["results"]} == {"NewerCo"}


def test_prepare_research_results_template_uses_suffixed_plan_for_same_second(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    prepare_research_plan(
        config=config,
        company_names=["OlderCo"],
        created_at=created_at,
    )
    latest_plan = prepare_research_plan(
        config=config,
        company_names=["NewerCo"],
        created_at=created_at,
    )

    result = prepare_research_results_template(
        config=config,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert latest_plan.output_path.name.endswith("-2.json")
    assert result.plan_path == latest_plan.output_path.resolve(strict=False)
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert {item["company_name"] for item in saved["results"]} == {"NewerCo"}


def test_prepare_research_results_template_rejects_symlink_plan(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        created_at=BUILT_AT,
    )
    symlink_path = tmp_path / "research-plan-link.json"
    symlink_path.symlink_to(plan_result.output_path)

    with pytest.raises(ResearchTemplateError, match="cannot be a symlink"):
        prepare_research_results_template(
            config=config,
            plan_path=symlink_path,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_prepare_research_results_template_command_writes_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = AppConfig(data_dir=tmp_path / "data")
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        created_at=BUILT_AT,
    )

    result = runner.invoke(
        app,
        [
            "prepare-research-results-template",
            str(plan_result.output_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Research template prepared" in result.output
    assert "Prepared a fillable external research results template" in result.output
    assert "No websites, APIs, paid databases, or Meridian pages were contacted" in result.output
    assert "import-research-results" in result.output
    assert "--dry-run" in result.output
    assert f"--data-dir {tmp_path / 'data'}" in result.output.replace("\n", "")
    templates = list((tmp_path / "data" / "research-results-templates").glob("*.json"))
    assert len(templates) == 1
    assert templates[0].name in result.output.replace("\n", "")


def test_prepare_research_results_template_command_quotes_printed_import_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "Hail Mary Data"
    config = AppConfig(data_dir=data_dir)
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        created_at=BUILT_AT,
    )

    result = runner.invoke(
        app,
        [
            "prepare-research-results-template",
            str(plan_result.output_path),
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    output = result.output.replace("\n", "")
    assert f"--data-dir {shlex.quote(str(data_dir))} --dry-run" in output
    assert "Hail Mary Data --dry-run" not in output


def test_prepare_research_results_template_command_has_plain_english_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-research-results-template",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "No research plans were found" in result.output
    assert "Traceback" not in result.output


def test_prepare_public_research_results_writes_private_importable_file(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D for a $1,000,000 offering.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            },
            {
                "company_name": "Acme AI Holdings",
                "title": "Acme AI Holdings Form D",
                "text": "Acme AI Holdings filed a Form D for a $2,000,000 offering.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme-holdings/form-d",
            },
            {
                "company_name": "Unrelated Robotics",
                "title": "Unrelated Robotics Form D",
                "text": "Unrelated Robotics filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/unrelated/form-d",
            },
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI", " acme ai "],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    assert result.output_path.parent == tmp_path / "data" / "research-results"
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert result.deal_count == 1
    assert result.result_count == 1
    assert result.skipped_non_exact_company_names == [
        "Acme AI Holdings",
        "Unrelated Robotics",
    ]
    assert {match.kind for match in result.match_details} == {
        CompanyMatchKind.EXACT,
        CompanyMatchKind.RELATED,
        CompanyMatchKind.REJECTED,
    }
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert {item["title"] for item in saved["results"]} == {"Acme AI Form D"}
    assert saved["results"][0]["provider_id"] == "sec_form_d"
    assert saved["results"][0]["provider_name"] == "SEC EDGAR Form D search"
    assert saved["results"][0]["retrieved_at"] == "2025-12-31T12:00:00Z"
    assert saved["results"][0]["confidence"].startswith("high:")
    assert "licensing_notes" in saved["results"][0]

    dry_run = import_research_results(
        config=config,
        results_path=result.output_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )
    assert dry_run.imported_count == 1
    assert dry_run.updated_store_paths == []


def test_prepare_public_research_results_prefers_exact_over_related_match(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI Ventures",
                "title": "Related investor Form D",
                "text": "A related investor entity that must not become company evidence.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme-ventures/form-d",
            },
            {
                "company_name": "Acme AI",
                "title": "Exact Acme AI Form D",
                "text": "Acme AI filed a Form D for a synthetic offering.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            },
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    assert result.result_count == 1
    assert result.skipped_non_exact_company_names == ["Acme AI Ventures"]
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert [item["title"] for item in saved["results"]] == ["Exact Acme AI Form D"]


def test_prepare_public_research_results_collapses_duplicate_public_results(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Older duplicate Form D",
                "text": "Acme AI filed a Form D for a synthetic offering.",
                "retrieved_at": "2024-01-01T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            },
            {
                "company_name": "Acme AI",
                "title": "Fresh duplicate Form D",
                "text": "Acme AI filed  a Form D for a synthetic offering.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d/",
            },
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    assert result.result_count == 1
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["title"] == "Fresh duplicate Form D"
    assert saved["results"][0]["retrieved_at"] == "2025-12-31T12:00:00Z"


def test_prepare_public_research_results_ranks_fresh_reliable_source_first(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Older reliable Form D",
                "text": "Acme AI filed an older synthetic Form D.",
                "retrieved_at": "2024-01-01T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/old-form-d",
            },
            {
                "company_name": "Acme AI",
                "title": "Fresh reliable Form D",
                "text": "Acme AI filed a fresher synthetic Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/fresh-form-d",
            },
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert [item["title"] for item in saved["results"]] == [
        "Fresh reliable Form D",
        "Older reliable Form D",
    ]


def test_prepare_public_research_results_preserves_suffix_distinct_requested_companies(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme LLC",
                "title": "Acme LLC Form D",
                "text": "Acme LLC filed a Form D for a synthetic offering.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme-llc/form-d",
            }
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme LLC", "Acme LP"],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.deal_count == 2
    assert result.result_count == 1
    assert {
        (summary.company_name, summary.result_count) for summary in result.deals
    } == {
        ("Acme LLC", 1),
        ("Acme LP", 0),
    }
    assert result.skipped_non_exact_company_names == []
    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert [item["company_name"] for item in saved["results"]] == ["Acme LLC"]


def test_research_workflow_warns_with_reasons_for_skipped_identity_matches(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a synthetic Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/exact",
            },
            {
                "company_name": "Acme AI Platform",
                "title": "Acme AI Platform result",
                "text": "A similarly named product result exists.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/platform",
            },
            {
                "company_name": "Unrelated Robotics",
                "title": "Unrelated result",
                "text": "An unrelated synthetic result exists.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/unrelated/result",
            },
        ],
    )

    workflow = run_research_workflow(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        created_at=BUILT_AT,
    )

    assert workflow.collections
    warnings = workflow.collections[0].warnings
    assert any("Acme AI Platform" in warning and "product name" in warning for warning in warnings)
    assert any("Unrelated Robotics" in warning and "rejected" in warning for warning in warnings)
    assert workflow.summary.warning_count >= 2


def test_prepare_public_research_results_combines_free_public_source_files(
    tmp_path: Path,
) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    sec_results_path = tmp_path / "sec-form-d-results.json"
    sam_results_path = tmp_path / "sam-gov-results.json"
    usaspending_results_path = tmp_path / "usaspending-results.json"
    sbir_results_path = tmp_path / "sbir-results.json"
    uspto_results_path = tmp_path / "uspto-results.json"
    github_results_path = tmp_path / "github-results.json"
    retrieved_at = "2025-12-31T12:00:00Z"

    _write_public_source_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": retrieved_at,
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )
    _write_public_source_results(
        sam_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI SAM.gov result",
                "text": "Acme AI has a public SAM.gov result.",
                "retrieved_at": retrieved_at,
                "source_url": "https://sam.gov/search/?index=opp&keywords=Acme+AI",
            },
            {
                "company_name": "Acme AI Holdings",
                "title": "Acme AI Holdings SAM.gov result",
                "text": "Related legal entity result that is not an exact match.",
                "retrieved_at": retrieved_at,
                "source_url": "https://sam.gov/search/?index=opp&keywords=Acme+AI+Holdings",
            },
        ],
    )
    _write_public_source_results(
        usaspending_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI USAspending result",
                "text": "Acme AI has a public USAspending result.",
                "retrieved_at": retrieved_at,
                "source_url": "https://www.usaspending.gov/search/?keywords=Acme+AI",
            }
        ],
    )
    _write_public_source_results(
        sbir_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI SBIR result",
                "text": "Acme AI has a public SBIR/STTR result.",
                "retrieved_at": retrieved_at,
                "source_url": "https://www.sbir.gov/award/acme-ai",
            }
        ],
    )
    _write_public_source_results(
        uspto_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI USPTO result",
                "text": "Acme AI has a public USPTO result.",
                "retrieved_at": retrieved_at,
                "source_url": "https://tmsearch.uspto.gov/search/search-results?query=Acme+AI",
            }
        ],
    )
    _write_public_source_results(
        github_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI GitHub result",
                "text": "Acme AI has a public GitHub repository result.",
                "retrieved_at": retrieved_at,
                "source_url": "https://github.com/acme-ai/example",
            }
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI"],
        sec_form_d_results_path=sec_results_path,
        sam_gov_results_path=sam_results_path,
        usaspending_results_path=usaspending_results_path,
        sbir_results_path=sbir_results_path,
        uspto_results_path=uspto_results_path,
        github_results_path=github_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    assert result.result_count == 6
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    provider_ids = {item["provider_id"] for item in saved["results"]}
    assert provider_ids == {
        "sec_form_d",
        "sam_gov",
        "usaspending",
        "sbir",
        "uspto",
        "github",
    }
    titles = {item["title"] for item in saved["results"]}
    assert "Acme AI Holdings SAM.gov result" not in titles
    assert all(item["company_name"] == "Acme AI" for item in saved["results"])


@pytest.mark.parametrize(
    ("provider_id", "source_api"),
    [
        ("sec_form_d", "https://data.sec.gov/submissions/CIK0000000000.json"),
        ("sam_gov", "https://api.sam.gov/opportunities/v2/search?title=Acme+AI"),
        ("usaspending", "https://api.usaspending.gov/api/v2/search/spending_by_award/"),
        ("sbir", "https://api.www.sbir.gov/public/api/awards?firm=Acme+AI"),
        ("uspto", "https://data.uspto.gov/apis/bulk-data/search"),
        ("github", "https://api.github.com/search/repositories?q=Acme+AI"),
    ],
)
def test_public_source_file_loader_accepts_source_api_for_each_provider(
    tmp_path: Path,
    provider_id: str,
    source_api: str,
) -> None:
    results_path = tmp_path / f"{provider_id}-results.json"
    _write_public_source_results(
        results_path,
        [
            {
                "company_name": "Acme AI",
                "title": f"Acme AI {provider_id} result",
                "text": "Acme AI has a public source result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_api": source_api,
            }
        ],
    )

    results_file = collection_module._load_public_source_search_results(
        results_path,
        provider_id=provider_id,
        description=f"{provider_id} results",
    )

    assert results_file.results[0].source_url is None
    assert results_file.results[0].source_api == source_api


def test_prepare_public_research_results_writes_source_api_only_results(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    github_results_path = tmp_path / "github-results.json"
    _write_public_source_results(
        github_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI GitHub API result",
                "text": "Acme AI has a public GitHub repository result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_api": "https://api.github.com/search/repositories?q=Acme+AI",
            }
        ],
    )

    result = prepare_public_research_results(
        config=config,
        company_names=["Acme AI"],
        github_results_path=github_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is not None
    saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    prepared = saved["results"][0]
    assert prepared["provider_id"] == "github"
    assert "source_url" not in prepared
    assert prepared["source_api"] == "https://api.github.com/search/repositories?q=Acme+AI"

    dry_run = import_research_results(
        config=config,
        results_path=result.output_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )
    assert dry_run.imported_count == 1


@pytest.mark.parametrize(
    ("provider_id", "source_api", "message"),
    [
        (
            "sec_form_d",
            "https://example.com/sec-result",
            "source_api must use an SEC website host",
        ),
        (
            "sam_gov",
            "https://example.com/sam-result",
            "source_api must use a SAM.gov website host",
        ),
        (
            "sbir",
            "https://example.com/sbir-result",
            "source_api must use an SBIR website host",
        ),
        (
            "uspto",
            "https://example.com/uspto-result",
            "source_api must use a USPTO website host",
        ),
        (
            "github",
            "https://example.com/github-result",
            "source_api must use the GitHub website host",
        ),
    ],
)
def test_public_source_file_loader_rejects_wrong_provider_source_api_host(
    tmp_path: Path,
    provider_id: str,
    source_api: str,
    message: str,
) -> None:
    results_path = tmp_path / f"bad-{provider_id}-results.json"
    _write_public_source_results(
        results_path,
        [
            {
                "company_name": "Acme AI",
                "title": f"Acme AI {provider_id} result",
                "text": "Acme AI has a public source result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_api": source_api,
            }
        ],
    )

    with pytest.raises(ResearchCollectionError, match=message):
        collection_module._load_public_source_search_results(
            results_path,
            provider_id=provider_id,
            description=f"{provider_id} results",
        )


def test_prepare_public_research_results_returns_no_file_when_no_results(
    tmp_path: Path,
) -> None:
    sec_results_path = tmp_path / "sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )

    result = prepare_public_research_results(
        config=AppConfig(data_dir=tmp_path / "data"),
        company_names=["MissingCo"],
        sec_form_d_results_path=sec_results_path,
        collected_at=BUILT_AT,
    )

    assert result.output_path is None
    assert result.result_count == 0
    assert result.deals[0].company_name == "MissingCo"
    assert not (tmp_path / "data" / "research-results").exists()


def test_prepare_public_research_results_command_writes_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "sec-form-d-results.json"
    sam_results_path = tmp_path / "sam-gov-results.json"
    data_dir = tmp_path / "Hail Mary Data"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )
    _write_public_source_results(
        sam_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI SAM.gov result",
                "text": "Acme AI has a public SAM.gov result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://sam.gov/search/?index=opp&keywords=Acme+AI",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--company",
            "MissingCo",
            "--sec-form-d-results",
            str(sec_results_path),
            "--sam-gov-results",
            str(sam_results_path),
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Public research results prepared" in result.output
    assert "Prepared 2 public research results for 2 companies" in result.output
    assert "No public research results were prepared for: MissingCo" in result.output
    assert "No websites or software data feeds were contacted" in result.output
    assert "import-research-results" in result.output
    output = result.output.replace("\n", "")
    assert f"--data-dir {shlex.quote(str(data_dir))} --dry-run" in output
    assert "Hail Mary Data --dry-run" not in output
    results = list((data_dir / "research-results").glob("*.json"))
    assert len(results) == 1
    assert results[0].name in output


def test_prepare_public_research_results_command_requires_local_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "Add at least one local public-source file" in result.output
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_reports_bad_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "bad-sec-form-d-results.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://example.com/form-d",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sec-form-d-results",
            str(sec_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "source_url must use an SEC website host" in _plain_cli_output(result.output)
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_reports_bad_sam_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sam_results_path = tmp_path / "bad-sam-gov-results.json"
    _write_public_source_results(
        sam_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI SAM.gov result",
                "text": "Acme AI has a public SAM.gov result.",
                "retrieved_at": "2025-12-31T12:00:00Z",
                "source_url": "https://example.com/sam-result",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sam-gov-results",
            str(sam_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "source_url must use a SAM.gov website host" in _plain_cli_output(result.output)
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_requires_results_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "missing-results-list.json"
    sec_results_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sec-form-d-results",
            str(sec_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "results" in result.output
    assert "Field required" in result.output
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_rejects_top_level_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "array-results.json"
    sec_results_path.write_text("[]", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sec-form-d-results",
            str(sec_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "must be a JSON object with a `results` list" in _plain_cli_output(result.output)
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_requires_retrieved_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "missing-retrieved-at.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sec-form-d-results",
            str(sec_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    output = _plain_cli_output(result.output)
    assert "retrieved_at" in output
    assert "time the source was retrieved or viewed" in output
    assert "Traceback" not in result.output


def test_prepare_public_research_results_command_reports_malformed_retrieved_at_as_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sec_results_path = tmp_path / "bad-retrieved-at.json"
    _write_sec_form_d_results(
        sec_results_path,
        [
            {
                "company_name": "Acme AI",
                "title": "Acme AI Form D",
                "text": "Acme AI filed a Form D.",
                "retrieved_at": "not a timestamp",
                "source_url": "https://www.sec.gov/Archives/edgar/data/acme/form-d",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "prepare-public-research-results",
            "--company",
            "Acme AI",
            "--sec-form-d-results",
            str(sec_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    output = _plain_cli_output(result.output)
    assert "retrieved_at is invalid" in output
    assert "retrieved_at is required" not in output
    assert "Traceback" not in result.output


def test_import_research_results_skips_untouched_template_rows(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    plan_result = prepare_research_plan(config=config, created_at=BUILT_AT)
    template_result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    template_payload = json.loads(template_result.output_path.read_text(encoding="utf-8"))
    template_payload["results"][1].update(
        {
            "provider_id": "meridian",
            "provider_name": "Meridian deal page",
            "source_url": "https://portal.angellist.com/m/example/invest",
            "licensing_notes": (
                "Authenticated source. Generated by Hail Mary "
                "prepare-meridian-workflow."
            ),
            "source_kind": "meridian",
            "document_type": "platform_deal_page",
        }
    )
    template_payload["results"][0].update(
        {
            "title": "Exact public source excerpt",
            "text": (
                "Acme AI reports revenue growth from customers. "
                "Minimum investment $2,500."
            ),
            "retrieved_at": "2026-01-01T12:00:00Z",
            "source_url": "https://example.com/exact-acme-source",
            "confidence": "high: exact source",
        }
    )
    template_result.output_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    result = import_research_results(
        config=config,
        results_path=template_result.output_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
        dry_run=True,
    )

    assert result.imported_count == 1
    assert result.skipped_duplicate_count == 0
    assert result.skipped_blank_template_row_count == len(template_payload["results"]) - 1
    assert result.deal_count == 1


def test_import_research_results_rejects_source_only_template_rows(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    plan_result = prepare_research_plan(config=config, created_at=BUILT_AT)
    template_result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    template_payload = json.loads(template_result.output_path.read_text(encoding="utf-8"))
    template_payload["results"][0]["source_url"] = "https://example.com/source-only"
    template_result.output_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"row 1: .*title"):
        import_research_results(
            config=config,
            results_path=template_result.output_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_unmarked_meridian_source_only_template_rows(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    plan_result = prepare_research_plan(
        config=config,
        company_names=["Acme AI"],
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    template_payload = json.loads(template_result.output_path.read_text(encoding="utf-8"))
    meridian_result = next(
        item for item in template_payload["results"] if item["provider_id"] == "meridian"
    )
    meridian_result["source_url"] = "https://portal.angellist.com/m/example/invest"
    template_result.output_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"row \d+: .*title"):
        import_research_results(
            config=config,
            results_path=template_result.output_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_imports_edited_meridian_placeholder_in_dry_run(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    result = import_research_results(
        config=config,
        results_path=workflow.result_template_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
        dry_run=True,
    )

    assert result.imported_count == 1
    assert result.skipped_duplicate_count == 0
    assert result.skipped_blank_template_row_count == len(template_payload["results"]) - 1
    assert result.deal_count == 1


def test_import_research_results_skips_legacy_meridian_placeholder_confidence(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
        }
    )
    template_payload["results"][1]["confidence"] = MERIDIAN_LEGACY_PLACEHOLDER_CONFIDENCE
    template_payload["results"][1]["licensing_notes"] = template_payload["results"][1][
        "licensing_notes"
    ].replace(
        MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER,
        MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER,
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    result = import_research_results(
        config=config,
        results_path=workflow.result_template_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
        dry_run=True,
    )

    assert result.imported_count == 1
    assert result.skipped_blank_template_row_count == len(template_payload["results"]) - 1


def test_import_research_results_rejects_source_only_meridian_placeholder_edits(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0]["source_url"] = (
        "https://portal.angellist.com/m/other-example/invest"
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"row 1: .*text"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_title_only_meridian_placeholder_edits(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0]["title"] = "Acme AI reports a $2,500 minimum."
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"row 1: .*text"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_completed_meridian_placeholder_url_edits(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
            "source_url": "https://portal.angellist.com/m/other-example/invest",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="generated Meridian deal page URL"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_reports_unsafe_completed_meridian_placeholder_source_url(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
            "source_url": "https://portal.angellist.com/m/example/invest?debug=1",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError) as exc_info:
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )

    message = str(exc_info.value)
    assert "row 1 source_url is not a safe Meridian deal page URL" in message
    assert "generated Meridian source URL" not in message


def test_import_research_results_rejects_completed_meridian_placeholder_source_api(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
            "source_api": "https://portal.angellist.com/m/example/invest?debug=1",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError) as exc_info:
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )

    message = str(exc_info.value)
    assert "source_api must be blank" in message
    assert "Clear source_api" in message
    assert "partly completed Meridian placeholder" not in message


def test_import_research_results_rejects_completed_meridian_placeholder_missing_marker(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
            "source_url": "https://portal.angellist.com/m/other-example/invest",
            "licensing_notes": "Authenticated source. Use only permitted facts.",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="generated placeholder marker"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_template_marker_without_source_url_marker(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "title": "Meridian deal page excerpt",
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page text",
            "source_url": "https://portal.angellist.com/m/other-example/invest",
            "licensing_notes": (
                f"{MERIDIAN_WORKFLOW_TEMPLATE_MARKER} Authenticated source."
            ),
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="generated source URL marker"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_meridian_placeholder_confidence(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="placeholder confidence"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_legacy_meridian_placeholder_confidence(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "text": "Acme AI reports a $2,500 minimum investment.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": MERIDIAN_LEGACY_PLACEHOLDER_CONFIDENCE,
        }
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="placeholder confidence"):
        import_research_results(
            config=config,
            results_path=workflow.result_template_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_strips_meridian_workflow_marker_from_evidence(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    workflow = prepare_meridian_workflow(
        config=config,
        company_name="Acme AI",
        meridian_url="https://portal.angellist.com/m/example/invest",
        created_at=BUILT_AT,
    )
    template_payload = json.loads(
        workflow.result_template_path.read_text(encoding="utf-8")
    )
    template_payload["results"][0].update(
        {
            "title": "Meridian deal page excerpt",
            "text": "Acme AI reports revenue growth from customers.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "confidence": "high: exact page excerpt",
        }
    )
    template_payload["results"][0]["licensing_notes"] = template_payload["results"][0][
        "licensing_notes"
    ].replace(
        MERIDIAN_WORKFLOW_PLACEHOLDER_MARKER,
        MERIDIAN_LEGACY_WORKFLOW_PLACEHOLDER_MARKER,
    )
    workflow.result_template_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    import_research_results(
        config=config,
        results_path=workflow.result_template_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence = next(
        evidence for evidence in saved_store.evidence if evidence.provider_id == "meridian"
    )
    assert meridian_evidence.licensing_notes is not None
    assert "Generated by Hail Mary prepare-meridian-workflow" not in (
        meridian_evidence.licensing_notes
    )
    assert "Generated Meridian placeholder" not in meridian_evidence.licensing_notes
    assert "Generated Meridian source URL" not in meridian_evidence.licensing_notes
    assert "Do not bypass" in meridian_evidence.licensing_notes


@pytest.mark.parametrize(
    ("source_url", "message"),
    [
        (
            "http://portal.angellist.com/m/example/invest",
            "must start with https://",
        ),
        (
            "https://portal.angellist.com/m/example/invest;jsessionid=secret",
            "query strings",
        ),
        (
            "https://portal.angellist.com/m/example;jsessionid=secret/invest",
            "query strings",
        ),
        (
            "https://portal.angellist.com/m/example/session-token/invest",
            "Meridian deal page",
        ),
        (
            "https://portal.angellist.com/m/example/invest?token=secret",
            "query strings",
        ),
        (
            "https://portal.angellist.com/m/example/invest#details",
            "query strings",
        ),
        (
            "https://portal.angellist.com//m/example/invest",
            "Meridian deal page",
        ),
        (
            "https://portal.angellist.com/m/example/invest/",
            "Meridian deal page",
        ),
        (
            "https://portal.angellist.com/m/example%3Ftoken=secret/invest",
            "encoded query strings",
        ),
        (
            "https://portal.angellist.com/m/example%253Ftoken=secret/invest",
            "encoded query strings",
        ),
        (
            "https://portal.angellist.com/m/example%2525253Ftoken=secret/invest",
            "encoded query strings",
        ),
        (
            " https://portal.angellist.com/m/example/invest",
            "cannot contain spaces",
        ),
        (
            "https://portal.angellist.com/m/example/invest ",
            "cannot contain spaces",
        ),
        (
            "https://user:token@portal.angellist.com/m/example/invest",
            "username or password",
        ),
        (
            "https://portal.angellist.com:444/m/example/invest",
            "cannot include a port",
        ),
        (
            "https://portal.angellist.com:/m/example/invest",
            "cannot include a port",
        ),
        (
            "https://example.com/m/example/invest",
            "portal.angellist.com",
        ),
    ],
)
def test_import_research_results_rejects_unsafe_meridian_source_urls(
    tmp_path: Path,
    source_url: str,
    message: str,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-meridian-url.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                source_url=source_url,
                source_kind="meridian",
                document_type="platform_deal_page",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match=message):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_meridian_source_api(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-meridian-source-api.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                source_url="https://portal.angellist.com/m/example/invest",
                source_api="https://portal.angellist.com/m/example/invest?token=secret",
                source_kind="meridian",
                document_type="platform_deal_page",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="source_api must be blank"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_marker_only_meridian_licensing_notes(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-marker-only-licensing.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                source_url="https://portal.angellist.com/m/example/invest",
                source_kind="meridian",
                document_type="platform_deal_page",
                licensing_notes="Generated by Hail Mary prepare-meridian-workflow.",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="licensing_notes must explain"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_unsafe_meridian_alternate_source_fields(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-meridian-extra-source.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                source_url="https://portal.angellist.com/m/example/invest",
                source_kind="meridian",
                document_type="platform_deal_page",
                portal_source_url="https://portal.angellist.com/m/example/invest?token=secret",
            )
        ],
    )

    with pytest.raises(
        ResearchImportError,
        match="portal_source_url is not an allowed research result field",
    ):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_reports_original_template_row_number(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    plan_result = prepare_research_plan(config=config, created_at=BUILT_AT)
    template_result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    template_payload = json.loads(template_result.output_path.read_text(encoding="utf-8"))
    template_payload["results"][1].update(
        {
            "title": "Bad source URL excerpt",
            "text": "Acme AI reports revenue growth from customers.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "source_url": "https://example .com/bad",
            "confidence": "high: exact source",
        }
    )
    template_result.output_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"Research result 2 .*source_url"):
        import_research_results(
            config=config,
            results_path=template_result.output_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_reports_malformed_retrieved_at_as_invalid(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "malformed-retrieved-at-results.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                retrieved_at="not a timestamp",
            )
        ],
    )

    with pytest.raises(ResearchImportError) as exc_info:
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )

    message = str(exc_info.value)
    assert "retrieved_at is invalid" in message
    assert "retrieved_at is required" not in message


def test_import_research_results_does_not_skip_incomplete_handwritten_rows(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "incomplete-handwritten-results.json"
    bad_results_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "company_name": "Acme AI",
                        "provider_id": "sec_form_d",
                        "licensing_notes": "Public government source.",
                    },
                    _research_result(),
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match=r"row 1: .*title"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_rejects_operator_supplied_template_skip_count(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "spoofed-skip-count-results.json"
    bad_results_path.write_text(
        json.dumps(
            {
                "skipped_blank_template_row_count": 99,
                "results": [_research_result()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="skipped_blank_template_row_count"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 3, tzinfo=UTC),
            dry_run=True,
        )


def test_import_research_results_appends_source_linked_external_evidence(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)

    result = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert result.imported_count == 1
    assert result.skipped_duplicate_count == 0
    assert result.deals[0].company_name == "Acme AI"
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    external_evidence = saved_store.evidence[-1]
    assert external_evidence.provider_id == "sec_form_d"
    assert external_evidence.provider_name == "SEC EDGAR Form D search"
    assert external_evidence.source_url == "https://www.sec.gov/example/acme-ai"
    assert external_evidence.external_confidence == "high: exact company match"
    assert external_evidence.licensing_notes == "Public government source."
    assert external_evidence.source_span_start == 0
    assert external_evidence.source_span_end == len(external_evidence.text)
    assert {claim.label for claim in saved_store.claims} >= {
        "minimum investment",
        "valuation cap",
    }
    scored_deal = score_evidence_store(saved_store, config=config)
    memo = render_markdown_memo(scored_deal, saved_store)
    assert "provider: SEC EDGAR Form D search" in memo
    assert "source page: https://www.sec.gov/example/acme-ai" in memo
    packet = build_agent_input_packet(
        saved_store,
        scored_deal,
        role=AgentRole.PRODUCT_MARKET_FIT,
    )
    packet_payload = packet.model_dump(mode="json")
    assert "SEC EDGAR Form D search" not in json.dumps(packet_payload)
    assert "high: exact company match" not in json.dumps(packet_payload)
    assert "Public government source." not in json.dumps(packet_payload)

    summary = json.loads((tmp_path / "data" / "processed" / "ingestion_summary.json").read_text())
    assert summary["deals"][0]["evidence_count"] == saved_store.evidence_count
    assert summary["deals"][0]["claim_count"] == saved_store.claim_count


def test_import_research_results_dry_run_does_not_write_private_outputs(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    assert deal.evidence_store_path is not None
    summary_path = tmp_path / "data" / "processed" / "ingestion_summary.json"
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")
    before_summary = summary_path.read_text(encoding="utf-8")

    result = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.imported_count == 1
    assert result.skipped_duplicate_count == 0
    assert result.updated_store_paths == []
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store
    assert summary_path.read_text(encoding="utf-8") == before_summary


def test_import_research_results_skips_duplicate_records(tmp_path: Path) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    result = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert result.imported_count == 0
    assert result.skipped_duplicate_count == 1
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    assert len([evidence for evidence in saved_store.evidence if evidence.provider_id]) == 1


def test_import_research_results_skips_legacy_meridian_duplicate_after_canonicalizing_url(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    meridian_results_path = tmp_path / "research-results-meridian.json"
    meridian_source_url = "https://PORTAL.ANGELLIST.com/m/example/invest"
    _write_results(
        meridian_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                title="Meridian deal page excerpt",
                source_url=meridian_source_url,
                source_api=None,
                source_kind="meridian",
                document_type="platform_deal_page",
                licensing_notes="Authenticated source.",
            )
        ],
    )
    import_research_results(
        config=config,
        results_path=meridian_results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence = next(
        evidence for evidence in saved_store.evidence if evidence.provider_id == "meridian"
    )
    legacy_evidence_id, legacy_document_id = _legacy_meridian_external_ids(
        deal_id=deal.id,
        source_url=meridian_source_url,
        text=meridian_evidence.text,
    )
    legacy_evidence = meridian_evidence.model_copy(
        update={
            "id": legacy_evidence_id,
            "document_id": legacy_document_id,
            "source_url": meridian_source_url,
        }
    )
    legacy_store = saved_store.model_copy(
        update={
            "evidence": [
                legacy_evidence if evidence.id == meridian_evidence.id else evidence
                for evidence in saved_store.evidence
            ]
        }
    )
    deal.evidence_store_path.write_text(
        legacy_store.model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = import_research_results(
        config=config,
        results_path=meridian_results_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert result.imported_count == 0
    assert result.skipped_duplicate_count == 1
    saved_again = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence_records = [
        evidence for evidence in saved_again.evidence if evidence.provider_id == "meridian"
    ]
    assert len(meridian_evidence_records) == 1
    assert meridian_evidence_records[0].source_url == meridian_source_url


def test_import_research_results_reuses_legacy_meridian_document_id(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    meridian_results_path = tmp_path / "research-results-meridian.json"
    meridian_source_url = "https://PORTAL.ANGELLIST.com/m/example/invest"
    _write_results(
        meridian_results_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                title="Meridian deal page excerpt",
                source_url=meridian_source_url,
                source_api=None,
                source_kind="meridian",
                document_type="platform_deal_page",
                licensing_notes="Authenticated source.",
            )
        ],
    )
    import_research_results(
        config=config,
        results_path=meridian_results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence = next(
        evidence for evidence in saved_store.evidence if evidence.provider_id == "meridian"
    )
    legacy_evidence_id, legacy_document_id = _legacy_meridian_external_ids(
        deal_id=deal.id,
        source_url=meridian_source_url,
        text=meridian_evidence.text,
    )
    legacy_evidence = meridian_evidence.model_copy(
        update={
            "id": legacy_evidence_id,
            "document_id": legacy_document_id,
            "source_url": meridian_source_url,
        }
    )
    legacy_store = saved_store.model_copy(
        update={
            "evidence": [
                legacy_evidence if evidence.id == meridian_evidence.id else evidence
                for evidence in saved_store.evidence
            ]
        }
    )
    deal.evidence_store_path.write_text(
        legacy_store.model_dump_json(indent=2),
        encoding="utf-8",
    )

    second_result_path = tmp_path / "research-results-meridian-second.json"
    _write_results(
        second_result_path,
        [
            _research_result(
                provider_id="meridian",
                provider_name="Meridian deal page",
                title="Meridian deal page traction excerpt",
                text="Acme AI reports customer growth from paid pilots.",
                source_url=meridian_source_url,
                source_api=None,
                source_kind="meridian",
                document_type="platform_deal_page",
                licensing_notes="Authenticated source.",
            )
        ],
    )
    result = import_research_results(
        config=config,
        results_path=second_result_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert result.imported_count == 1
    saved_again = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence_records = [
        evidence for evidence in saved_again.evidence if evidence.provider_id == "meridian"
    ]
    assert len(meridian_evidence_records) == 2
    assert {evidence.document_id for evidence in meridian_evidence_records} == {
        legacy_document_id
    }
    assert {evidence.source_url for evidence in meridian_evidence_records} == {
        meridian_source_url,
        "https://portal.angellist.com/m/example/invest",
    }


def test_import_research_results_dry_run_reports_duplicates_without_writing(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert deal.evidence_store_path is not None
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")

    result = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.imported_count == 0
    assert result.skipped_duplicate_count == 1
    assert result.updated_store_paths == []
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store


def test_import_research_results_skips_duplicate_with_different_title(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    retitled_results_path = tmp_path / "research-results-retitled.json"
    _write_results(
        retitled_results_path,
        [
            _research_result(
                title="Retitled copy of same source excerpt",
            )
        ],
    )

    result = import_research_results(
        config=config,
        results_path=retitled_results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert result.imported_count == 0
    assert result.skipped_duplicate_count == 1
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    assert len([evidence for evidence in saved_store.evidence if evidence.provider_id]) == 1


def test_import_research_results_skips_cross_provider_duplicate_records(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    cross_provider_results_path = tmp_path / "research-results-cross-provider.json"
    _write_results(
        cross_provider_results_path,
        [
            _research_result(
                provider_id="public_web",
                provider_name="Public web and press search",
                title="Retitled public copy of same source excerpt",
                text=(
                    "Acme AI reports revenue growth from customers.\n"
                    "Minimum investment $2,500."
                ),
            )
        ],
    )

    result = import_research_results(
        config=config,
        results_path=cross_provider_results_path,
        imported_at=datetime(2026, 1, 3, tzinfo=UTC),
    )

    assert result.imported_count == 0
    assert result.skipped_duplicate_count == 1
    assert result.provider_imported_counts == {}
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    imported_evidence = [evidence for evidence in saved_store.evidence if evidence.provider_id]
    assert len(imported_evidence) == 1
    assert imported_evidence[0].provider_id == "sec_form_d"


def test_import_research_results_keeps_one_document_id_per_external_source(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(
        tmp_path,
        extra_results=[
            _research_result(
                title="Acme AI Form D snippet two",
                text="Acme AI has customers and a minimum investment $2,500.",
            )
        ],
    )

    result = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert result.imported_count == 2
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    imported_evidence = [
        evidence for evidence in saved_store.evidence if evidence.provider_id == "sec_form_d"
    ]
    assert len(imported_evidence) == 2
    assert len({evidence.id for evidence in imported_evidence}) == 2
    assert len({evidence.document_id for evidence in imported_evidence}) == 1


def test_import_research_results_defaults_meridian_to_meridian_source_kind(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(
        tmp_path,
        extra_results=[
            _research_result(
                provider_id="meridian",
                provider_name=None,
                title="Meridian deal page excerpt",
                source_url="https://PORTAL.ANGELLIST.com/m/example/invest",
                source_api=None,
                licensing_notes="Authenticated source.",
            )
        ],
    )

    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    meridian_evidence = next(
        evidence for evidence in saved_store.evidence if evidence.provider_id == "meridian"
    )
    assert meridian_evidence.source_kind == SourceKind.MERIDIAN
    assert meridian_evidence.document_type == DocumentType.PLATFORM_DEAL_PAGE
    assert meridian_evidence.provider_id == "meridian"
    assert meridian_evidence.provider_name == "Meridian deal page"
    assert meridian_evidence.source_url == "https://portal.angellist.com/m/example/invest"
    assert meridian_evidence.source_api is None
    assert meridian_evidence.retrieved_at == datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert meridian_evidence.external_confidence == "high: exact company match"
    assert meridian_evidence.licensing_notes == "Authenticated source."
    assert meridian_evidence.text == (
        "Acme AI reports revenue growth from customers. Minimum investment $2,500."
    )


def test_import_research_results_escapes_external_metadata_in_memos(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(
        tmp_path,
        extra_results=[
            _research_result(
                title="Metadata escape test",
                text="Acme AI has customers and a minimum investment $2,500.",
                source_url="https://www.sec.gov/example/acme-ai-metadata",
                provider_name="SEC source\n# Fake Heading",
                confidence="high\n## Fake Confidence",
                licensing_notes="[fake](https://example.com)\n- Fake claim",
            )
        ],
    )

    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    memo = render_markdown_memo(score_evidence_store(saved_store, config=config), saved_store)
    assert "\n# Fake Heading" not in memo
    assert "\n## Fake Confidence" not in memo
    assert "\\# Fake Heading" in memo
    assert "\\[fake\\]\\(https://example.com\\)" in memo


def test_import_research_results_fails_before_writing_for_unknown_deal(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(
        tmp_path,
        extra_results=[
            _research_result(company_name="MissingCo", title="Unknown company source"),
        ],
    )
    assert deal.evidence_store_path is not None
    before = deal.evidence_store_path.read_text(encoding="utf-8")

    with pytest.raises(ResearchImportError, match="no ingested deal has that exact name"):
        import_research_results(
            config=config,
            results_path=results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )

    after = deal.evidence_store_path.read_text(encoding="utf-8")
    assert after == before


def test_import_research_results_requires_source_url_or_api(tmp_path: Path) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-missing-source.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                source_url=None,
                source_api=None,
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="source_url or source_api"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_imports_stale_sources_as_stale(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(
        tmp_path,
        extra_results=[
            _research_result(
                title="Older public source",
                text="Acme AI reported customer traction in an older public source.",
                retrieved_at="2024-01-01T12:00:00Z",
                source_url="https://www.sec.gov/example/acme-ai-old",
            )
        ],
    )

    summary = import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert summary.imported_count == 2
    assert summary.stale_count == 1
    assert summary.provider_imported_counts == {"sec_form_d": 2}
    assert summary.provider_stale_counts == {"sec_form_d": 1}
    assert summary.provider_names == {"sec_form_d": "SEC EDGAR Form D search"}
    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    stale_records = [
        evidence
        for evidence in saved_store.evidence
        if evidence.source_freshness == SourceFreshness.STALE
    ]
    assert len(stale_records) == 1
    assert "older public source" in stale_records[0].text


def test_import_research_results_saves_reliability_and_identity_tags(
    tmp_path: Path,
) -> None:
    config, deal, results_path = _ingest_deal_and_write_results(tmp_path)

    import_research_results(
        config=config,
        results_path=results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert deal.evidence_store_path is not None
    saved_store = EvidenceStore.model_validate_json(
        deal.evidence_store_path.read_text(encoding="utf-8")
    )
    external_evidence = next(
        evidence for evidence in saved_store.evidence if evidence.provider_id == "sec_form_d"
    )
    assert external_evidence.source_reliability == SourceReliability.GOVERNMENT_FILING
    assert external_evidence.identity_match_kind == CompanyMatchKind.EXACT.value
    assert external_evidence.identity_match_reason

    quality = research_quality_status(saved_store)

    assert quality.status == "usable"
    assert quality.imported_record_count == 1
    assert quality.source_reliability[0].label == SourceReliability.GOVERNMENT_FILING.value
    assert quality.identity_matches[0].label == CompanyMatchKind.EXACT.value


def test_import_research_results_rejects_non_exact_identity_without_safe_confidence(
    tmp_path: Path,
) -> None:
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-related-identity.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                deal_id=deal.id,
                company_name=None,
                identity_match_kind="product_name",
                identity_match_reason="The source was for a similarly named platform.",
                confidence="medium: related product name",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="identity_match_kind product name"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_requires_plain_english_licensing_notes(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-missing-licensing.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                licensing_notes="",
            )
        ],
    )

    with pytest.raises(
        ResearchImportError,
        match="licensing_notes is required",
    ):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("licensing_notes", "message"),
    [
        ("n/a", "not a placeholder"),
        ("unknown", "not a placeholder"),
        ("todo", "not a placeholder"),
        ("https://example.com/license", "not only provide a URL"),
        ("[license](https://example.com/license)", "not only contain Markdown"),
        ("[license](https://example.com/license).", "not only contain Markdown"),
    ],
)
def test_import_research_results_rejects_placeholder_or_markdown_only_licensing_notes(
    tmp_path: Path,
    licensing_notes: str,
    message: str,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-licensing.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                licensing_notes=licensing_notes,
            )
        ],
    )

    with pytest.raises(ResearchImportError, match=message):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("source_url", "message"),
    [
        ("https://example.com:bad/path", "invalid port"),
        ("https://example.com:99999/path", "invalid port"),
        ("https://user:token@example.com/path", "username or password"),
        (
            "https://www.sec.gov/example/acme-ai?token=secret",
            "token, signature, credential",
        ),
        (
            "https://www.sec.gov/example/acme-ai?access%255Ftoken=secret",
            "token, signature, credential",
        ),
        (
            "https://www.sec.gov/example/acme-ai?api-key=secret",
            "token, signature, credential",
        ),
        (
            "https://www.sec.gov/example/acme-ai?redirect_url=https%3A%2F%2Fexample.com",
            "credential, redirect",
        ),
        (
            "https://www.sec.gov/example/acme-ai?file=x;access_token=secret",
            "semicolon query delimiters",
        ),
        (
            "https://www.sec.gov/example/acme-ai?file=x%253Baccess_token=secret",
            "semicolon query delimiters",
        ),
        (
            "https://www.sec.gov/example/acme-ai#access_token=secret",
            "URL fragments",
        ),
        (
            "https://www.sec.gov/example/acme%3Ftoken=secret",
            "encoded query, fragment, or parameter delimiters",
        ),
    ],
)
def test_import_research_results_rejects_unsafe_source_urls(
    tmp_path: Path,
    source_url: str,
    message: str,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-url.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                source_url=source_url,
            )
        ],
    )

    with pytest.raises(ResearchImportError, match=message):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("source_api", "message"),
    [
        ("https://api.example.com:bad/result", "source_api has an invalid port"),
        (
            "https://api.example.com/result?X-Amz-Signature=secret",
            "source_api cannot include token, signature, credential",
        ),
        (
            "https://api.example.com/result?x%252Damz%252Dsignature=secret",
            "source_api cannot include token, signature, credential",
        ),
        (
            "https://api.example.com/result?access-token=secret",
            "source_api cannot include token, signature, credential",
        ),
        (
            "https://api.example.com/result?next=https%3A%2F%2Fexample.com",
            "source_api cannot include token, signature, credential, redirect",
        ),
        (
            "https://api.example.com/result?file=x;access_token=secret",
            "source_api cannot include semicolon query delimiters",
        ),
        (
            "https://api.example.com/result#access_token=secret",
            "source_api cannot include URL fragments",
        ),
        (
            "https://user:token@api.example.com/result",
            "source_api cannot include a username or password",
        ),
    ],
)
def test_import_research_results_rejects_unsafe_url_like_source_apis(
    tmp_path: Path,
    source_api: str,
    message: str,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-source-api.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                source_url=None,
                source_api=source_api,
            )
        ],
    )

    with pytest.raises(ResearchImportError, match=message):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "source_url",
    [
        "https://example.com/result?q=https%3A%2F%2Fprivate.example%2Fdeal%3Ftoken%3Dsecret",
        "https://example.com/result?q=token%3Dsecret",
        "https://example.com/result?q=company%26api_key%3Dsecret",
    ],
)
def test_source_url_validation_rejects_sensitive_nested_query_values(
    source_url: str,
) -> None:
    with pytest.raises(ValueError, match="token, signature, credential"):
        validate_http_url(source_url, field_name="source_url")


def test_source_url_validation_allows_plain_search_query_value() -> None:
    validate_http_url(
        "https://example.com/search?q=tokenization%20market%20analysis",
        field_name="source_url",
    )


def test_import_research_results_rejects_wrong_provider_source_url_host(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-wrong-provider-url.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="sam_gov",
                provider_name="SAM.gov search",
                source_url="https://example.com/sam-result",
            )
        ],
    )

    with pytest.raises(
        ResearchImportError,
        match=r"provider_id sam_gov.*source_url must use a SAM\.gov website host",
    ):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_unsafe_source_url_with_quality_fields(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-tokenized-quality-url.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="custom_public_source",
                provider_name="Custom public source",
                source_url="https://example.com/acme-ai?token=secret",
                source_reliability="unknown",
                identity_match_kind="exact",
                identity_match_reason="Synthetic exact company identity fixture.",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="token, signature, credential"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_wrong_provider_source_api_host(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-wrong-provider-api.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="sec_form_d",
                source_url=None,
                source_api="https://example.com/api/sec-result",
            )
        ],
    )

    with pytest.raises(
        ResearchImportError,
        match=r"provider_id sec_form_d.*source_api must use an SEC website host",
    ):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_allows_unknown_provider_source_url_host(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    custom_results_path = tmp_path / "research-results-custom-provider.json"
    _write_results(
        custom_results_path,
        [
            _research_result(
                provider_id="custom_public_source",
                provider_name="Custom public source",
                source_url="https://example.com/acme-ai",
            )
        ],
    )

    result = import_research_results(
        config=config,
        results_path=custom_results_path,
        imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        dry_run=True,
    )

    assert result.imported_count == 1


def test_import_research_results_rejects_builtin_provider_source_kind_mismatch(
    tmp_path: Path,
) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-bad-source-kind.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                provider_id="meridian",
                source_url="https://portal.angellist.com/m/example/invest",
                source_kind="web",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="source_kind meridian"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_unknown_result_fields(tmp_path: Path) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-extra-field.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                unsupported_note="This field should fail instead of being ignored.",
            )
        ],
    )

    with pytest.raises(ResearchImportError, match="unsupported_note"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_requires_results_key(tmp_path: Path) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-missing-results.json"
    bad_results_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ResearchImportError, match="results"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_rejects_top_level_array(tmp_path: Path) -> None:
    config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "research-results-array.json"
    bad_results_path.write_text(
        json.dumps([_research_result()]),
        encoding="utf-8",
    )

    with pytest.raises(ResearchImportError, match="JSON object with a `results` list"):
        import_research_results(
            config=config,
            results_path=bad_results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_requires_ingested_deals(tmp_path: Path) -> None:
    results_path = tmp_path / "research-results.json"
    _write_results(results_path, [_research_result()])

    with pytest.raises(ResearchImportError, match="No ingested deals"):
        import_research_results(
            config=AppConfig(data_dir=tmp_path / "data"),
            results_path=results_path,
            imported_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_import_research_results_command_has_plain_english_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _config, _deal, results_path = _ingest_deal_and_write_results(tmp_path)

    result = runner.invoke(
        app,
        [
            "import-research-results",
            str(results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Research results imported" in result.output
    assert "Imported 1 external research evidence record into 1 deal" in result.output
    assert "No websites or APIs were contacted" in result.output


def test_import_research_results_command_dry_run_has_plain_english_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _config, deal, results_path = _ingest_deal_and_write_results(tmp_path)
    assert deal.evidence_store_path is not None
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "import-research-results",
            str(results_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Research import preview" in result.output
    assert "Dry run: 1 external research evidence record would be imported" in result.output
    assert "- Acme AI: would add 1 record." in result.output
    assert "No evidence stores were changed." in result.output
    assert "No websites or APIs were contacted." in result.output
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store


def test_import_research_results_command_reports_untouched_template_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config, deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    plan_result = prepare_research_plan(config=config, created_at=BUILT_AT)
    template_result = prepare_research_results_template(
        config=config,
        plan_path=plan_result.output_path,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    template_payload = json.loads(template_result.output_path.read_text(encoding="utf-8"))
    template_payload["results"][0].update(
        {
            "title": "Exact public source excerpt",
            "text": "Acme AI reports revenue growth from customers.",
            "retrieved_at": "2026-01-01T12:00:00Z",
            "source_url": "https://example.com/exact-acme-source",
            "confidence": "high: exact source",
        }
    )
    template_result.output_path.write_text(
        json.dumps(template_payload),
        encoding="utf-8",
    )
    assert deal.evidence_store_path is not None
    before_store = deal.evidence_store_path.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "import-research-results",
            str(template_result.output_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Skipped 7 untouched template rows." in result.output
    assert deal.evidence_store_path.read_text(encoding="utf-8") == before_store


def test_import_research_results_command_has_plain_english_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _config, _deal, _results_path = _ingest_deal_and_write_results(tmp_path)
    bad_results_path = tmp_path / "bad-research-results.json"
    _write_results(
        bad_results_path,
        [
            _research_result(
                source_url="https://example .com",
            )
        ],
    )

    result = runner.invoke(
        app,
        [
            "import-research-results",
            str(bad_results_path),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "source_url cannot contain spaces" in result.output
    assert "Traceback" not in result.output


def _ingest_deal_and_write_results(
    tmp_path: Path,
    *,
    extra_results: list[dict[str, object]] | None = None,
) -> tuple[AppConfig, IngestedDeal, Path]:
    root = tmp_path / "pitch-decks"
    company = root / "Acme AI"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    config = AppConfig(data_dir=tmp_path / "data")
    summary = ingest_folder(root, config=config)
    results_path = tmp_path / "research-results.json"
    results = [_research_result(company_name="Acme AI")]
    if extra_results:
        results.extend(extra_results)
    _write_results(results_path, results)
    return config, summary.deals[0], results_path


def _research_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "company_name": "Acme AI",
        "provider_id": "sec_form_d",
        "title": "Acme AI Form D",
        "text": (
            "Acme AI reports revenue growth from customers. Minimum investment $2,500."
        ),
        "retrieved_at": "2026-01-01T12:00:00Z",
        "source_url": "https://www.sec.gov/example/acme-ai",
        "confidence": "high: exact company match",
        "licensing_notes": "Public government source.",
    }
    result.update(overrides)
    return result


def _paid_fact(**overrides: object) -> PaidProviderFact:
    payload: dict[str, object] = {
        "company_name": "Acme AI",
        "title": "Acme AI paid provider profile",
        "text": "Acme AI has a synthetic paid-provider company profile.",
        "retrieved_at": "2025-12-31T12:00:00Z",
        "source_url": "https://www.crunchbase.com/organization/acme-ai",
        "source_api": "https://api.crunchbase.com/api/v4/entities/organizations/acme-ai",
        "confidence": "high: exact company match from a licensed paid provider",
        "licensing_notes": (
            "Licensed paid-provider account permits saving this short diligence fact."
        ),
    }
    payload.update(overrides)
    return PaidProviderFact.model_validate(payload)


def _write_results(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def _legacy_meridian_external_ids(
    *,
    deal_id: str,
    source_url: str,
    text: str,
) -> tuple[str, str]:
    evidence_payload = "\0".join(
        [
            deal_id,
            "meridian",
            SourceKind.MERIDIAN.value,
            source_url,
            text,
        ]
    )
    evidence_digest = hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
    document_payload = "\0".join(
        [
            deal_id,
            "meridian",
            SourceKind.MERIDIAN.value,
            source_url,
        ]
    )
    document_digest = hashlib.sha256(document_payload.encode("utf-8")).hexdigest()
    return (
        f"ev_external_meridian_{evidence_digest[:16]}",
        f"doc_external_meridian_{document_digest[:12]}",
    )


def _plain_cli_output(output: str) -> str:
    return " ".join(output.replace("│", " ").split())


class _FakeWebResearchClient:
    def __init__(self, responses: dict[str, WebFetchResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch(
        self,
        url: str,
        *,
        provider_id: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        _ = (provider_id, timeout_seconds, max_bytes)
        self.calls.append(url)
        return self.responses[url]


class _FailingWebResearchClient:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls: list[str] = []

    def fetch(
        self,
        url: str,
        *,
        provider_id: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebFetchResponse:
        _ = (provider_id, timeout_seconds, max_bytes)
        self.calls.append(url)
        raise WebResearchFetchError(self.message)


class _FakePaidProviderClient:
    def __init__(
        self,
        *,
        provider_id: str,
        facts: list[PaidProviderFact] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.facts = facts or []
        self.calls: list[str] = []

    def search_company(
        self,
        request: PaidProviderSearchRequest,
    ) -> PaidProviderSearchResponse:
        self.calls.append(request.company_name)
        return PaidProviderSearchResponse(provider_id=self.provider_id, results=self.facts)


class _FakeHeaders:
    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeUsaspendingAwardsClient:
    def __init__(
        self,
        responses: dict[tuple[str, int], UsaspendingAwardsResponse],
        *,
        error: UsaspendingApiError | None = None,
    ) -> None:
        self.responses = responses
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    def search_awards(
        self,
        company_name: str,
        *,
        limit: int,
        page: int,
        timeout_seconds: float,
    ) -> UsaspendingAwardsResponse:
        _ = timeout_seconds
        self.calls.append((company_name, limit, page))
        if self.error is not None:
            raise self.error
        try:
            return self.responses[(company_name, page)]
        except KeyError as exc:
            raise AssertionError(
                f"No fake USAspending response for {company_name} page {page}."
            ) from exc


class _FakeSbirAwardsClient:
    def __init__(
        self,
        responses: dict[tuple[str, int], SbirAwardsResponse],
        *,
        error: SbirApiError | None = None,
    ) -> None:
        self.responses = responses
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    def search_awards(
        self,
        company_name: str,
        *,
        rows: int,
        start: int,
        timeout_seconds: float,
    ) -> SbirAwardsResponse:
        _ = timeout_seconds
        self.calls.append((company_name, rows, start))
        if self.error is not None:
            raise self.error
        try:
            return self.responses[(company_name, start)]
        except KeyError as exc:
            raise AssertionError(
                f"No fake SBIR/STTR response for {company_name} start {start}."
            ) from exc


class _FakeSecFormDFilingsClient:
    def __init__(
        self,
        responses: dict[tuple[str, int], SecFormDFilingsResponse],
        *,
        error: SecFormDApiError | None = None,
    ) -> None:
        self.responses = responses
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    def search_filings(
        self,
        company_name: str,
        *,
        count: int,
        start: int,
        timeout_seconds: float,
    ) -> SecFormDFilingsResponse:
        _ = timeout_seconds
        self.calls.append((company_name, count, start))
        if self.error is not None:
            raise self.error
        try:
            return self.responses[(company_name, start)]
        except KeyError as exc:
            raise AssertionError(
                f"No fake SEC Form D response for {company_name} start {start}."
            ) from exc


class _FakeGitHubRepositorySearchClient:
    def __init__(
        self,
        responses: dict[tuple[str, int], GitHubRepositorySearchResponse],
        *,
        error: GitHubApiError | None = None,
    ) -> None:
        self.responses = responses
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    def search_repositories(
        self,
        company_name: str,
        *,
        per_page: int,
        page: int,
        timeout_seconds: float,
    ) -> GitHubRepositorySearchResponse:
        _ = timeout_seconds
        self.calls.append((company_name, per_page, page))
        if self.error is not None:
            raise self.error
        try:
            return self.responses[(company_name, page)]
        except KeyError as exc:
            raise AssertionError(
                f"No fake GitHub response for {company_name} page {page}."
            ) from exc


def _usaspending_response(
    results: list[UsaspendingAwardRecord],
    *,
    has_next: bool = False,
) -> UsaspendingAwardsResponse:
    return UsaspendingAwardsResponse.model_validate(
        {
            "results": [result.model_dump(by_alias=True) for result in results],
            "page_metadata": {"hasNext": has_next},
        }
    )


def _usaspending_raw_response(
    results: list[dict[str, object]],
    *,
    has_next: bool = False,
) -> UsaspendingAwardsResponse:
    return UsaspendingAwardsResponse.model_validate(
        {
            "results": results,
            "page_metadata": {"hasNext": has_next},
        }
    )


def _sbir_response(results: list[SbirAwardRecord]) -> SbirAwardsResponse:
    return SbirAwardsResponse.model_validate(
        {
            "results": [result.model_dump() for result in results],
        }
    )


def _sbir_raw_response(results: list[dict[str, object]]) -> SbirAwardsResponse:
    return SbirAwardsResponse.model_validate({"results": results})


def _sec_form_d_response(
    results: list[SecFormDFilingRecord],
    *,
    has_next: bool = False,
) -> SecFormDFilingsResponse:
    return SecFormDFilingsResponse.model_validate(
        {
            "results": [result.model_dump() for result in results],
            "has_next": has_next,
        }
    )


def _sec_form_d_raw_response(
    results: list[dict[str, object]],
    *,
    has_next: bool = False,
) -> SecFormDFilingsResponse:
    return SecFormDFilingsResponse.model_validate(
        {"results": results, "has_next": has_next}
    )


def _github_repository_response(
    results: list[dict[str, object]],
    *,
    incomplete_results: bool = False,
    has_next: bool = False,
    total_count: int | None = None,
) -> GitHubRepositorySearchResponse:
    return GitHubRepositorySearchResponse.model_validate(
        {
            "total_count": len(results) if total_count is None else total_count,
            "incomplete_results": incomplete_results,
            "has_next": has_next,
            "items": results,
        }
    )


def _github_raw_repository_response(
    results: list[dict[str, object]],
    *,
    incomplete_results: bool = False,
) -> GitHubRepositorySearchResponse:
    return _github_repository_response(
        results,
        incomplete_results=incomplete_results,
    )


def _usaspending_award(
    *,
    recipient_name: str,
    award_id: str = "FAKE-123",
    generated_internal_id: str = "CONT_AWD_FAKE_123",
    award_amount: float | None = None,
    description: str | None = None,
) -> UsaspendingAwardRecord:
    return UsaspendingAwardRecord.model_validate(
        {
            "Award ID": award_id,
            "Recipient Name": recipient_name,
            "Recipient UEI": "UEI123",
            "generated_internal_id": generated_internal_id,
            "Award Amount": award_amount,
            "Award Type": "Contract",
            "Awarding Agency": "Department of Example",
            "Awarding Sub Agency": "Example Office",
            "Funding Agency": "Department of Example",
            "Funding Sub Agency": "Example Funding Office",
            "Start Date": "2025-01-01",
            "End Date": "2025-12-31",
            "Description": description,
        }
    )


def _sbir_award(
    *,
    firm: str,
    award_title: str,
    award_link: str | None = "https://www.sbir.gov/awards/123",
    award_amount: float | None = None,
    abstract: str | None = None,
) -> SbirAwardRecord:
    return SbirAwardRecord.model_validate(
        {
            "firm": firm,
            "award_title": award_title,
            "agency": "NSF",
            "branch": "Example Branch",
            "phase": "Phase I",
            "program": "SBIR",
            "agency_tracking_number": f"TRACK-{award_title}",
            "contract": f"CONTRACT-{award_title}",
            "proposal_award_date": "2025-01-01",
            "contract_end_date": "2025-12-31",
            "solicitation_number": "SOL-123",
            "solicitation_year": "2025",
            "topic_code": "AI",
            "award_year": "2025",
            "award_amount": award_amount,
            "uei": "UEI123",
            "research_area_keywords": "synthetic fixtures",
            "abstract": abstract,
            "award_link": award_link,
            "poc_email": "founder@example.com",
            "poc_phone": "555-0100",
        }
    )


def _sec_form_d_filing(
    *,
    issuer_name: str,
    filing_type: str = "D",
    accession_number: str = "0001234567-26-000001",
    total_offering_amount: str | None = None,
) -> SecFormDFilingRecord:
    accession_digits = accession_number.replace("-", "")
    accession_filename = collection_module._sec_accession_filename(accession_number)
    assert accession_filename is not None
    return SecFormDFilingRecord.model_validate(
        {
            "issuer_name": issuer_name,
            "filing_type": filing_type,
            "accession_number": accession_number,
            "source_url": (
                "https://www.sec.gov/Archives/edgar/data/1234567890/"
                f"{accession_digits}/{accession_filename}.txt"
            ),
            "source_api": collection_module._sec_form_d_atom_api_url(
                issuer_name,
                count=5,
                start=0,
            ),
            "filing_date": "2026-01-01",
            "form_name": "Notice of Exempt Offering of Securities",
            "total_offering_amount": total_offering_amount,
            "total_amount_sold": "$250,000",
            "total_remaining": "$750,000",
            "minimum_investment_accepted": "$2,500",
            "total_investors": "5",
            "industry_group": "Other Technology",
            "revenue_range": "Decline to Disclose",
            "federal_exemptions": ["06b"],
            "contact_email": "founder@example.com",
            "contact_phone": "555-0100",
            "street_address": "1 Main Street",
        }
    )


def _github_repository(
    *,
    name: str,
    full_name: str,
    owner_login: str,
) -> dict[str, object]:
    return {
        "name": name,
        "full_name": full_name,
        "owner": {"login": owner_login},
        "html_url": f"https://github.com/{full_name}",
        "url": f"https://api.github.com/repos/{full_name}",
        "description": "Synthetic public repository metadata.",
        "language": "Python",
        "stargazers_count": 42,
        "forks_count": 7,
        "open_issues_count": 3,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "pushed_at": "2026-01-02T00:00:00Z",
        "license": {"name": "MIT License"},
        "private": False,
        "fork": False,
        "archived": False,
        "disabled": False,
    }


def _write_public_source_results(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def _write_sec_form_d_results(path: Path, results: list[dict[str, object]]) -> None:
    _write_public_source_results(path, results)
