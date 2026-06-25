from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary.agents.packets import build_agent_input_packet
from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.research import (
    MeridianWorkflowError,
    ResearchImportError,
    ResearchPlanError,
    ResearchProviderCategory,
    ResearchTaskStatus,
    ResearchTemplateError,
    builtin_research_providers,
    import_research_results,
    prepare_meridian_workflow,
    prepare_public_research_results,
    prepare_research_plan,
    prepare_research_results_template,
)
from hailmary.research.schemas import ResearchResultInput
from hailmary.schemas.agents import AgentRole
from hailmary.schemas.documents import DocumentType, IngestedDeal, SourceKind
from hailmary.schemas.evidence import EvidenceStore
from hailmary.scoring.memo import render_markdown_memo
from hailmary.scoring.scorer import score_evidence_store

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


def test_research_result_source_kind_defaults_match_builtin_registry() -> None:
    providers = builtin_research_providers(include_paid=True, include_meridian=True)
    base_payload = _research_result()

    for provider in providers:
        payload = {**base_payload, "provider_id": provider.id}

        result = ResearchResultInput.model_validate(payload)

        assert result.source_kind == provider.source_kind


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
    assert workflow["company_name"] == "Acme AI"
    assert workflow["meridian_url"] == "https://portal.angellist.com/m/acme-ai/invest"
    assert workflow["result_template_path"] == str(result.result_template_path)
    assert "import-research-results" in workflow["import_command"]
    assert any("Do not bypass" in rule for rule in workflow["safety_rules"])
    assert any("normal sign-in" in step for step in workflow["manual_steps"])

    template = json.loads(result.result_template_path.read_text(encoding="utf-8"))
    assert list(template) == ["results"]
    assert len(template["results"]) == 1
    row = template["results"][0]
    assert row["company_name"] == "Acme AI"
    assert row["provider_id"] == "meridian"
    assert row["provider_name"] == "Meridian deal page"
    assert row["source_url"] == "https://portal.angellist.com/m/acme-ai/invest"
    assert row["retrieved_at"] == ""
    assert row["source_kind"] == "meridian"
    assert row["document_type"] == "platform_deal_page"
    assert "Do not bypass" in row["licensing_notes"]
    assert "Generated by Hail Mary prepare-meridian-workflow" in row["licensing_notes"]


@pytest.mark.parametrize(
    "meridian_url",
    [
        "",
        "mailto:founder@example.com",
        "http://portal.angellist.com/m/acme-ai/invest",
        "https://example.com/m/acme-ai/invest",
        "https://portal.angellist.com/not-m/acme-ai/invest",
        "https://portal.angellist.com/m/acme-ai/profile",
        "https://portal.angellist.com/m/acme-ai/session-token/invest",
        "https://user:token@portal.angellist.com/m/acme-ai/invest",
        "https://portal.angellist.com:bad/m/acme-ai/invest",
        "https://portal.angellist.com/m/acme ai/invest",
        "https://portal.angellist.com/m/acme-ai;jsessionid=secret/invest",
        "https://portal.angellist.com/m/acme-ai/invest;jsessionid=secret",
        "https://portal.angellist.com/m/acme-ai/invest?token=secret",
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
    assert "Prepared a Meridian manual workflow for Acme AI" in result.output
    assert "did not open Meridian, sign in, bypass access controls" in result.output
    assert "import-research-results" in result.output
    assert (tmp_path / "data" / "meridian-workflows").is_dir()
    workflows = list((tmp_path / "data" / "meridian-workflows").glob("*.json"))
    templates = list((tmp_path / "data" / "research-results-templates").glob("*.json"))
    assert len(workflows) == 1
    assert len(templates) == 1
    assert workflows[0].name in result.output.replace("\n", "")
    assert templates[0].name in result.output.replace("\n", "")


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
    assert "Prepared a fillable external research results template" in result.output
    assert "No websites, APIs, paid databases, or Meridian pages were contacted" in result.output
    assert "import-research-results" in result.output
    assert "--dry-run" in result.output
    assert f"--data-dir {tmp_path / 'data'}" in result.output.replace("\n", "")
    templates = list((tmp_path / "data" / "research-results-templates").glob("*.json"))
    assert len(templates) == 1
    assert templates[0].name in result.output.replace("\n", "")


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
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Prepared 1 public research result for 2 companies" in result.output
    assert "No public research results were prepared for: MissingCo" in result.output
    assert "No websites or software data feeds were contacted" in result.output
    assert "import-research-results" in result.output
    results = list((tmp_path / "data" / "research-results").glob("*.json"))
    assert len(results) == 1
    assert results[0].name in result.output.replace("\n", "")


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
    assert "source_url must use an SEC website host" in result.output
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
    assert "retrieved_at" in result.output
    assert "Field" in result.output
    assert "required" in result.output
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
            "extra text after",
        ),
        (
            "https://portal.angellist.com/m/example;jsessionid=secret/invest",
            "extra text after",
        ),
        (
            "https://portal.angellist.com/m/example/session-token/invest",
            "Meridian deal page",
        ),
        (
            "https://portal.angellist.com/m/example/invest?token=secret",
            "extra text after",
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
                source_url="https://portal.angellist.com/m/example/invest",
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


@pytest.mark.parametrize(
    ("source_url", "message"),
    [
        ("https://example.com:bad/path", "invalid port"),
        ("https://example.com:99999/path", "invalid port"),
        ("https://user:token@example.com/path", "username or password"),
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


def _write_results(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def _write_sec_form_d_results(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
