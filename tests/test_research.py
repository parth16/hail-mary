from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.research import (
    ResearchPlanError,
    ResearchProviderCategory,
    ResearchTaskStatus,
    builtin_research_providers,
    prepare_research_plan,
)

runner = CliRunner()
BUILT_AT = datetime(2026, 1, 1, tzinfo=UTC)


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
        "pitchbook",
    }
    assert all(provider.default_enabled is False for provider in paid_providers)


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
