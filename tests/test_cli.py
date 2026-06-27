from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from hailmary.cli import _format_dollars, app
from hailmary.config import AppConfig
from hailmary.evidence.actions import action_log_path
from hailmary.ingest import folder_loader
from hailmary.ingest.extractors import ExtractionResult
from hailmary.ingest.extractors import extract_document as real_extract_document
from hailmary.ingest.ocr import LocalOcrResult
from hailmary.schemas.documents import (
    DocumentType,
    ExtractedPage,
    ExtractionQuality,
    FileType,
    IngestedDeal,
    IngestedDocument,
    IngestionSummary,
    SourceDocument,
    SourceKind,
)
from hailmary.schemas.evidence import (
    ClaimConflict,
    ClaimRecord,
    ClaimType,
    EvidenceCitation,
    EvidenceKind,
    EvidenceQuality,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)

runner = CliRunner()


class CliFakeOcrEngine:
    def image_to_text(
        self,
        path: Path,
        *,
        page_number: int | None = None,
    ) -> LocalOcrResult:
        del path
        assert page_number == 1
        return LocalOcrResult(
            text="Valuation cap $8M. Minimum investment $1,000.",
            confidence=0.9,
        )

    def pdf_page_to_text(self, path: Path, *, page_number: int) -> LocalOcrResult:
        del path, page_number
        return LocalOcrResult(text="")


def test_bin_wrapper_runs_without_uv_run() -> None:
    wrapper = Path(__file__).parents[1] / "bin" / "hailmary"

    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert "uv run" not in wrapper_text
    assert 'cd "$PROJECT_ROOT"' in wrapper_text
    assert "PYTHONPATH" in wrapper_text


def test_top_level_help_shows_only_operator_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "evaluate-deal" in result.output
    assert "review-evidence" in result.output
    hidden_commands = [
        "init",
        "ingest-folder",
        "score-deals",
        "prepare-agent-packets",
        "validate-agent-output",
        "run-evals",
        "list-research-providers",
        "research-workflow",
        "prepare-research-plan",
        "collect-web-research",
        "prepare-research-results-template",
        "prepare-public-research-results",
        "collect-sec-form-d-filings",
        "collect-github-repositories",
        "collect-usaspending-awards",
        "collect-sbir-awards",
        "prepare-meridian-workflow",
        "import-research-results",
        "portfolio",
    ]
    for command in hidden_commands:
        assert command not in result.output


def test_init_creates_local_state(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "local-data"

    result = runner.invoke(app, ["init", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Init complete" in result.output
    assert "Status" in result.output
    assert "Location" in result.output
    assert "Created Hail Mary local folders" in result.output
    assert (data_dir / "raw").is_dir()
    assert (data_dir / "processed").is_dir()
    assert (data_dir / "reports").is_dir()
    assert (data_dir / "portfolio").is_dir()
    assert (tmp_path / ".hailmary" / "config.yaml").is_file()


def test_portfolio_commands_record_and_show_investments(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"

    add_result = runner.invoke(
        app,
        [
            "portfolio",
            "add-investment",
            "--company",
            "ExampleCo",
            "--amount",
            "5000",
            "--date",
            "2026-06-23",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert add_result.exit_code == 0, add_result.output
    assert "Investment recorded" in add_result.output
    assert "ExampleCo" in add_result.output
    assert "$5,000" in add_result.output
    assert (data_dir / "portfolio" / "ledger.json").is_file()

    status_result = runner.invoke(
        app,
        ["portfolio", "status", "--data-dir", str(data_dir)],
    )

    assert status_result.exit_code == 0, status_result.output
    assert "Portfolio status" in status_result.output
    assert "Recorded investments" in status_result.output
    assert "ExampleCo" in status_result.output
    assert "$5,000" in status_result.output

    plan_result = runner.invoke(
        app,
        ["portfolio", "plan", "--data-dir", str(data_dir)],
    )

    assert plan_result.exit_code == 0, plan_result.output
    assert "Portfolio plan" in plan_result.output
    assert "Recorded investments are subtracted" in plan_result.output


def test_portfolio_add_investment_bad_date_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "portfolio",
            "add-investment",
            "--company",
            "ExampleCo",
            "--amount",
            "5000",
            "--date",
            "06/23/2026",
        ],
    )

    assert result.exit_code == 1
    assert "investment date must use YYYY-MM-DD" in result.output
    assert "Traceback" not in result.output


def test_portfolio_add_investment_negative_amount_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "portfolio",
            "add-investment",
            "--company",
            "ExampleCo",
            "--amount",
            "-1",
            "--date",
            "2026-06-23",
        ],
    )

    assert result.exit_code == 1
    normalized_output = " ".join(result.output.split())
    assert "investment amount must be greater than zero" in normalized_output
    assert "Traceback" not in result.output


def test_portfolio_dollar_formatting_preserves_large_integers() -> None:
    amount = 10**100 + 123_456_789

    assert _format_dollars(amount) == f"${amount:,}"


def test_ingest_folder_command_writes_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "Acme memo.txt").write_text("Memo about Acme customer traction.", encoding="utf-8")
    data_dir = tmp_path / "data"

    result = runner.invoke(app, ["ingest-folder", str(source.parent), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Scan complete" in result.output
    assert "Metric" in result.output
    assert "Value" in result.output
    assert "Found 1 deal and 1 document" in result.output
    assert "source-linked evidence" in result.output

    summary_path = data_dir / "processed" / "ingestion_summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["deals"][0]["company_name"] == "Acme"


def test_ingest_folder_warns_when_documents_need_image_text_reading(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "ScanCo"
    source.mkdir(parents=True)
    (source / "ScanCo pitch deck.pdf").write_bytes(b"%PDF-1.4")

    def fake_extract_document(path: Path, **_: object) -> ExtractionResult:
        assert path.name == "ScanCo pitch deck.pdf"
        page = ExtractedPage(
            page_number=1,
            raw_text="",
            clean_text="",
            needs_ocr=True,
            vision_recommended=True,
        )
        return ExtractionResult(
            pages=[page],
            page_count=1,
            extraction_quality=ExtractionQuality.LOW,
            ocr_recommended=True,
            vision_recommended=True,
        )

    monkeypatch.setattr(folder_loader, "extract_document", fake_extract_document)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code == 0, result.output
    assert "Review needed" in result.output
    normalized_output = " ".join(result.output.split())
    assert "may need image-based text reading (OCR)" in normalized_output
    assert "before Hail Mary can use all of their content" in normalized_output


def test_ingest_folder_warns_for_image_only_deal(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "ImageOnlyCo"
    source.mkdir(parents=True)
    (source / "scan.png").write_bytes(b"synthetic image placeholder")
    data_dir = tmp_path / "data"

    result = runner.invoke(app, ["ingest-folder", str(source.parent), "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Found 1 deal and 1 document" in normalized_output
    assert "No usable evidence text was built for deal: ImageOnlyCo" in normalized_output
    assert "may need image-based text reading (OCR)" in normalized_output

    summary_path = data_dir / "processed" / "ingestion_summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_document = saved_summary["deals"][0]["documents"][0]
    assert saved_document["source"]["ocr_recommended"]
    assert saved_document["source"]["vision_recommended"]


def test_ingest_folder_enable_ocr_uses_local_image_text_reading(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "OcrCliCo"
    source.mkdir(parents=True)
    (source / "scan.png").write_bytes(b"synthetic image placeholder")
    data_dir = tmp_path / "data"
    monkeypatch.setattr(folder_loader, "SubprocessLocalOcrEngine", CliFakeOcrEngine)

    result = runner.invoke(
        app,
        [
            "ingest-folder",
            str(source.parent),
            "--data-dir",
            str(data_dir),
            "--enable-ocr",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Used image-based text reading (OCR) on 1 document" in normalized_output
    assert "OCR means reading text from images" in normalized_output
    assert "source-linked evidence" in normalized_output

    summary_path = data_dir / "processed" / "ingestion_summary.json"
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_document = saved_summary["deals"][0]["documents"][0]
    assert saved_document["source"]["ocr_applied"]
    assert saved_document["source"]["ocr_confidence"] == 0.9
    assert not saved_document["source"]["ocr_recommended"]


def test_ingest_folder_warns_when_no_usable_evidence_is_built(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "EmptyCo"
    source.mkdir(parents=True)
    (source / "empty.pdf").write_bytes(b"%PDF-1.4")

    def fake_extract_document(path: Path, **_: object) -> ExtractionResult:
        assert path.name == "empty.pdf"
        return ExtractionResult(
            pages=[],
            page_count=1,
            extraction_quality=ExtractionQuality.LOW,
        )

    monkeypatch.setattr(folder_loader, "extract_document", fake_extract_document)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code == 0, result.output
    assert "No usable evidence text was built" in result.output
    assert "cannot use their text yet" in result.output


def test_ingest_folder_warns_for_each_deal_without_usable_evidence(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "pitch-decks"
    empty_source = root / "EmptyCo"
    full_source = root / "FullCo"
    empty_source.mkdir(parents=True)
    full_source.mkdir(parents=True)
    (empty_source / "empty.pdf").write_bytes(b"%PDF-1.4")
    (full_source / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")

    def fake_extract_document(path: Path, **_: object) -> ExtractionResult:
        if path.name == "empty.pdf":
            return ExtractionResult(
                pages=[],
                page_count=1,
                extraction_quality=ExtractionQuality.LOW,
            )
        return real_extract_document(path)

    monkeypatch.setattr(folder_loader, "extract_document", fake_extract_document)

    result = runner.invoke(app, ["ingest-folder", str(root)])

    assert result.exit_code == 0, result.output
    assert "source-linked evidence" in result.output
    assert "No usable evidence text was built for deal: EmptyCo" in result.output
    assert "FullCo" not in result.output.split("No usable evidence text was built", 1)[1]


def test_evaluate_deal_local_only_cli_prints_safe_run_summary(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "CliEvalCo"
    source.mkdir()
    (source / "memo.txt").write_text(
        "Valuation cap $8M. Discount 20%. Round size $1M. "
        "ARR revenue growth with paid customers and retention. "
        "Lead investor committed and seed round is active. PRIVATE_FULL_TEXT_MARKER_AT_END",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["evaluate-deal", str(source), "--data-dir", str(tmp_path / "data")],
    )

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Deal evaluation complete" in normalized_output
    assert "Local-only mode was used" in normalized_output
    assert "rule-based scoring" in normalized_output
    assert "Documents ingested" in normalized_output
    assert "Evidence records" in normalized_output
    assert "Claims found" in normalized_output
    assert "Conflicts found" in normalized_output
    assert "Evidence health" in normalized_output
    assert "Evidence health review found" in normalized_output
    assert "saved source records are complete and safe enough" in normalized_output
    assert "Rule-based recommendation" in normalized_output
    assert "Final recommendation" in normalized_output
    assert "Failed model roles none" in normalized_output
    assert "OCR means reading text from images" in normalized_output
    assert "PRIVATE_FULL_TEXT_MARKER_AT_END" not in result.output
    assert "Valuation cap $8M" not in result.output


def test_evaluate_deal_cli_imports_research_results_before_final_decision(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "CliResearchCo"
    source.mkdir()
    (source / "memo.txt").write_text(
        "Valuation cap $8M. Round size $1M. Lead investor committed.",
        encoding="utf-8",
    )
    research_results = tmp_path / "research-results.json"
    research_results.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "company_name": "CliResearchCo",
                        "provider_id": "company_website",
                        "provider_name": "Company website",
                        "title": "CliResearchCo traction page",
                        "text": (
                            "CliResearchCo public site reports ARR revenue growth, "
                            "paid customers, and strong retention."
                        ),
                        "retrieved_at": "2026-01-01T12:00:00Z",
                        "source_url": "https://example.com/cliresearchco/traction",
                        "confidence": "high: exact synthetic company match",
                        "licensing_notes": "Synthetic public page fixture.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate-deal",
            str(source),
            "--data-dir",
            str(tmp_path / "data"),
            "--results-file",
            str(research_results),
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "external research workflow" in normalized_output
    assert "External research imported 1" in normalized_output
    assert "External research planned" in normalized_output
    assert "CliResearchCo public site reports" not in result.output
    memo_paths = list((tmp_path / "data" / "reports").glob("*-final-evaluation.md"))
    assert len(memo_paths) == 1
    memo_text = memo_paths[0].read_text(encoding="utf-8")
    assert "## External Research" in memo_text
    assert "Imported 1 external research evidence record before scoring." in memo_text


def test_ingest_folder_unreadable_path_has_plain_english_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "pitch-decks"
    company = root / "HiddenCo"
    secret_dir = company / "Secret"
    company.mkdir(parents=True)
    secret_dir.mkdir()
    (company / "memo.txt").write_text("Memo about HiddenCo.", encoding="utf-8")

    def fake_walk(
        top: Path,
        topdown: bool,
        onerror: Callable[[OSError], None] | None,
        followlinks: bool,
    ) -> Iterator[tuple[Path, list[str], list[str]]]:
        assert top == root.resolve()
        assert topdown is True
        assert followlinks is False
        yield root.resolve(), ["HiddenCo"], []
        if callable(onerror):
            onerror(PermissionError(13, "Permission denied", str(secret_dir)))
        yield company.resolve(), ["Secret"], ["memo.txt"]

    monkeypatch.setattr(os, "walk", fake_walk)

    result = runner.invoke(app, ["ingest-folder", str(root), "--data-dir", str(tmp_path / "data")])

    assert result.exit_code == 0, result.output
    assert "Could not read 1 path" in result.output
    assert "documents may be missing" in result.output
    assert "unsupported or ignored" not in result.output


def test_ingest_folder_missing_folder_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    missing_folder = tmp_path / "missing"

    result = runner.invoke(app, ["ingest-folder", str(missing_folder)])

    assert result.exit_code != 0
    assert "The folder does not exist" in result.output
    assert "Traceback" not in result.output


def test_invalid_boolean_env_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "treu")

    result = runner.invoke(app, ["init", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "must be true or false" in result.output
    assert "privacy settings should fail closed" in result.output
    assert "Traceback" not in result.output


def test_invalid_numeric_env_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_MAX_CHECK", "ten")

    result = runner.invoke(app, ["init", "--data-dir", str(tmp_path / "data")])

    assert result.exit_code != 0
    assert "must be a whole number" in result.output
    assert "Traceback" not in result.output


def test_init_unsafe_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)

    result = runner.invoke(app, ["init", "--data-dir", "."])

    assert result.exit_code != 0
    assert "cannot be the current folder" in result.output
    assert "Traceback" not in result.output


def test_init_file_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local-data").write_text("not a folder", encoding="utf-8")

    result = runner.invoke(app, ["init", "--data-dir", "local-data"])

    assert result.exit_code != 0
    assert "needs local-data to be a folder" in result.output
    assert "Traceback" not in result.output


def test_init_bad_config_path_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".hailmary" / "config.yaml"
    config_path.mkdir(parents=True)

    result = runner.invoke(app, ["init", "--force"])

    assert result.exit_code != 0
    assert "needs .hailmary/config.yaml to be a file" in result.output
    assert "Traceback" not in result.output


def test_init_force_recovers_invalid_saved_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".hailmary"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("local_only: treu\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "--force", "--data-dir", "local-data"])

    assert result.exit_code == 0, result.output
    assert "Created local config" in result.output
    config_text = (config_dir / "config.yaml").read_text(encoding="utf-8")
    assert 'data_dir: "local-data"' in config_text
    assert "local_only: true" in config_text


def test_ingest_folder_unsafe_data_dir_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git" / "info").mkdir(parents=True)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")

    result = runner.invoke(app, ["ingest-folder", str(source.parent), "--data-dir", "."])

    assert result.exit_code != 0
    assert "cannot be the current folder" in result.output
    assert "Traceback" not in result.output


def test_ingest_folder_summary_write_error_has_plain_english_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pitch-decks" / "Acme"
    source.mkdir(parents=True)
    (source / "memo.txt").write_text("Memo about Acme.", encoding="utf-8")
    summary_path = tmp_path / "data" / "processed" / "ingestion_summary.json"
    summary_path.mkdir(parents=True)

    result = runner.invoke(app, ["ingest-folder", str(source.parent)])

    assert result.exit_code != 0
    assert "Could not write scan summary" in result.output
    assert "Traceback" not in result.output


def test_review_evidence_single_deal_latest_summary_hides_text_by_default(
    tmp_path: Path,
) -> None:
    evidence = _review_evidence_record(
        "ev_secret",
        "SECRET CUSTOMER LIST. Valuation cap $8M.",
        deal_id="deal_secret",
    )
    store = _review_store(
        deal_id="deal_secret",
        company_name="SecretCo",
        evidence=[evidence],
        claims=[_review_claim("valuation cap", "$8M", evidence)],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Evidence review" in result.output
    assert "SecretCo" in result.output
    assert "Evidence health summary" in result.output
    assert "Evidence by source document" in result.output
    assert "Claims by label and status" in result.output
    assert "Evidence text is hidden by default" in result.output
    assert "source span" in result.output
    assert "citation span" in result.output
    assert "SECRET CUSTOMER LIST" not in result.output


def test_evidence_actions_cli_writes_private_action_and_hides_notes_by_default(
    tmp_path: Path,
) -> None:
    evidence = _review_evidence_record(
        "ev_secret",
        "SECRET CUSTOMER LIST. Valuation cap $8M.",
        deal_id="deal_secret",
    )
    store = _review_store(
        deal_id="deal_secret",
        company_name="SecretCo",
        evidence=[evidence],
        claims=[_review_claim("valuation cap", "$8M", evidence)],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    action_result = runner.invoke(
        app,
        [
            "evidence-actions",
            "exclude",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
            "--evidence-id",
            "ev_secret",
            "--note",
            "Operator checked the source locally.",
        ],
    )
    list_result = runner.invoke(
        app,
        [
            "evidence-actions",
            "list",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
        ],
    )
    notes_result = runner.invoke(
        app,
        [
            "evidence-actions",
            "list",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
            "--show-notes",
        ],
    )

    assert action_result.exit_code == 0, action_result.output
    assert "No evidence text was copied" in action_result.output
    assert "SECRET CUSTOMER LIST" not in action_result.output
    action_path = action_log_path(
        config=AppConfig(data_dir=data_dir),
        deal_id="deal_secret",
    )
    action_text = action_path.read_text(encoding="utf-8")
    assert "Operator checked the source locally" in action_text
    assert "SECRET CUSTOMER LIST" not in action_text
    assert list_result.exit_code == 0, list_result.output
    assert "excluded" in list_result.output
    assert "Operator checked the source locally" not in list_result.output
    assert "SECRET CUSTOMER LIST" not in list_result.output
    assert notes_result.exit_code == 0, notes_result.output
    assert "Operator checked the source locally" in notes_result.output
    assert "SECRET CUSTOMER LIST" not in notes_result.output


def test_review_evidence_shows_action_status_but_hides_notes_by_default(
    tmp_path: Path,
) -> None:
    evidence = _review_evidence_record(
        "ev_secret",
        "SECRET CUSTOMER LIST. Valuation cap $8M.",
        deal_id="deal_secret",
    )
    store = _review_store(
        deal_id="deal_secret",
        company_name="SecretCo",
        evidence=[evidence],
        claims=[_review_claim("valuation cap", "$8M", evidence)],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])
    action_result = runner.invoke(
        app,
        [
            "evidence-actions",
            "needs-review",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
            "--evidence-id",
            "ev_secret",
            "--note",
            "Operator note should stay private by default.",
        ],
    )

    review_result = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--deal-id", "deal_secret"],
    )
    review_notes_result = runner.invoke(
        app,
        [
            "review-evidence",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
            "--show-notes",
        ],
    )

    assert action_result.exit_code == 0, action_result.output
    assert review_result.exit_code == 0, review_result.output
    assert "needs review" in review_result.output
    assert "Operator note should stay private" not in review_result.output
    assert "SECRET CUSTOMER LIST" not in review_result.output
    assert review_notes_result.exit_code == 0, review_notes_result.output
    assert "Operator note should stay private" in review_notes_result.output
    assert "SECRET CUSTOMER LIST" not in review_notes_result.output


def test_evidence_actions_cli_unknown_target_has_plain_english_error(
    tmp_path: Path,
) -> None:
    store = _review_store(deal_id="deal_secret", company_name="SecretCo")
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "evidence-actions",
            "exclude",
            "--data-dir",
            str(data_dir),
            "--deal-id",
            "deal_secret",
            "--evidence-id",
            "missing_ev",
        ],
    )

    assert result.exit_code != 0
    assert "No evidence record missing_ev" in result.output
    assert "Traceback" not in result.output


def test_review_evidence_selects_by_deal_id_company_and_all(tmp_path: Path) -> None:
    alpha = _review_store(deal_id="deal_alpha", company_name="AlphaCo")
    beta = _review_store(deal_id="deal_beta", company_name="BetaCo")
    data_dir = _write_review_ingestion_summary(tmp_path, [alpha, beta])

    by_deal_id = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--deal-id", "deal_beta"],
    )
    by_company = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--company", "alphaco"],
    )
    all_deals = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--all"],
    )

    assert by_deal_id.exit_code == 0, by_deal_id.output
    assert "BetaCo" in by_deal_id.output
    assert "AlphaCo" not in by_deal_id.output
    assert by_company.exit_code == 0, by_company.output
    assert "AlphaCo" in by_company.output
    assert "BetaCo" not in by_company.output
    assert all_deals.exit_code == 0, all_deals.output
    assert "AlphaCo" in all_deals.output
    assert "BetaCo" in all_deals.output


def test_review_evidence_multiple_deals_needs_explicit_selector(tmp_path: Path) -> None:
    data_dir = _write_review_ingestion_summary(
        tmp_path,
        [
            _review_store(deal_id="deal_alpha", company_name="AlphaCo"),
            _review_store(deal_id="deal_beta", company_name="BetaCo"),
        ],
    )

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])
    normalized_output = " ".join(result.output.split())

    assert result.exit_code != 0
    assert "latest ingestion summary has 2 deals" in normalized_output
    assert "--deal-id" in normalized_output
    assert "--company" in normalized_output
    assert "--all" in normalized_output
    assert "Traceback" not in result.output


def test_review_evidence_no_local_data_initialized_has_plain_english_error(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(tmp_path / "missing-data")],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(result.output.split())
    assert "No local generated-data folder was found" in normalized_output
    assert "hailmary" in normalized_output
    assert "evaluate-deal" in normalized_output
    assert "Traceback" not in result.output


def test_review_evidence_missing_ingestion_summary_has_plain_english_error(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code != 0
    normalized_output = " ".join(result.output.split())
    assert "No generated evidence was found" in normalized_output
    assert "hailmary" in normalized_output
    assert "evaluate-deal" in normalized_output
    assert "Traceback" not in result.output


def test_review_evidence_bad_selector_and_duplicate_company_have_plain_english_errors(
    tmp_path: Path,
) -> None:
    data_dir = _write_review_ingestion_summary(
        tmp_path,
        [
            _review_store(deal_id="deal_first", company_name="TwinCo"),
            _review_store(deal_id="deal_second", company_name="TwinCo"),
        ],
    )

    bad_deal_id = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--deal-id", "missing"],
    )
    duplicate_company = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(data_dir), "--company", "TwinCo"],
    )

    assert bad_deal_id.exit_code != 0
    assert "No ingested deal has deal ID missing" in bad_deal_id.output
    assert "Traceback" not in bad_deal_id.output
    assert duplicate_company.exit_code != 0
    normalized_duplicate_output = " ".join(duplicate_company.output.split())
    assert "matches more than one ingested deal" in normalized_duplicate_output
    assert "--deal-id instead" in normalized_duplicate_output
    assert "Traceback" not in duplicate_company.output


def test_review_evidence_malformed_evidence_store_has_plain_english_error(
    tmp_path: Path,
) -> None:
    store = _review_store(deal_id="deal_broken", company_name="BrokenCo")
    data_dir = _write_review_ingestion_summary(tmp_path, [store])
    store_path = data_dir / "processed" / "deals" / store.deal_id / "evidence_store.json"
    store_path.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code != 0
    normalized_output = " ".join(result.output.split())
    assert "evidence store for BrokenCo could not be read" in normalized_output
    assert "hailmary" in normalized_output
    assert "evaluate-deal" in normalized_output
    assert "Traceback" not in result.output


def test_review_evidence_rejects_symlinked_data_dir(tmp_path: Path) -> None:
    real_data_dir = _write_review_ingestion_summary(
        tmp_path,
        [_review_store(deal_id="deal_link", company_name="LinkCo")],
    )
    symlink_data_dir = tmp_path / "linked-data"
    symlink_data_dir.symlink_to(real_data_dir, target_is_directory=True)

    result = runner.invoke(
        app,
        ["review-evidence", "--data-dir", str(symlink_data_dir)],
    )

    assert result.exit_code != 0
    assert "data directory" in result.output
    assert "symlink" in result.output
    assert "Traceback" not in result.output


def test_review_evidence_flags_empty_evidence_store(tmp_path: Path) -> None:
    store = _review_store(
        deal_id="deal_empty",
        company_name="EmptyReviewCo",
        evidence=[],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "No usable evidence" in normalized_output
    assert "Re-run ingestion" in normalized_output
    assert "No review issues found" not in normalized_output


def test_review_evidence_flags_invalid_source_spans(tmp_path: Path) -> None:
    evidence = _review_evidence_record(
        "ev_invalid_span",
        "Valuation cap $8M.",
        deal_id="deal_invalid_span",
    ).model_copy(update={"source_span_start": 5, "source_span_end": 5})
    store = _review_store(
        deal_id="deal_invalid_span",
        company_name="InvalidSpanCo",
        evidence=[evidence],
        claims=[_review_claim("valuation cap", "$8M", evidence)],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "missing_spans" in normalized_output
    assert "warning" in normalized_output
    assert "missing source span" in normalized_output


def test_review_evidence_quote_limit_shows_only_bounded_excerpt(tmp_path: Path) -> None:
    evidence = _review_evidence_record(
        "ev_excerpt",
        "ALPHA-BETA-GAMMA-DELTA Valuation cap $8M.",
        deal_id="deal_excerpt",
    )
    store = _review_store(
        deal_id="deal_excerpt",
        company_name="ExcerptCo",
        evidence=[evidence],
        claims=[_review_claim("valuation cap", "$8M", evidence)],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(
        app,
        [
            "review-evidence",
            "--data-dir",
            str(data_dir),
            "--quote-limit",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Showing short evidence excerpts capped at 12 characters" in result.output
    assert "ALPHA-BET..." in result.output
    assert "GAMMA-DELTA" not in result.output


def test_review_evidence_summarizes_review_issues_and_conflicts(tmp_path: Path) -> None:
    first_evidence = _review_evidence_record(
        "ev_first",
        "Valuation cap $8M.",
        deal_id="deal_issues",
        document_id="doc_scan",
        document_path=Path("scan.pdf"),
        source_span=False,
        ocr_applied=True,
        ocr_confidence=0.2,
        source_freshness=SourceFreshness.STALE,
    )
    second_evidence = _review_evidence_record(
        "ev_second",
        "Valuation cap $10M.",
        deal_id="deal_issues",
        document_id="doc_web",
        document_path=Path("web.txt"),
        source_kind=SourceKind.WEB,
        source_freshness=SourceFreshness.UNKNOWN,
        external_confidence=None,
    )
    first_claim = _review_claim(
        "valuation cap",
        "$8M",
        first_evidence,
        status=VerificationStatus.CONFLICTED,
        confidence=0.2,
    )
    second_claim = _review_claim(
        "valuation cap",
        "$10M",
        second_evidence,
        status=VerificationStatus.CONFLICTED,
        confidence=0.2,
    )
    missing_citation_claim = _review_claim(
        "round size",
        "$1M",
        first_evidence,
        include_citation=False,
        status=VerificationStatus.MISSING_CITATION,
        confidence=0.2,
    )
    invalid_citation_claim = _review_claim(
        "discount",
        "20%",
        first_evidence,
        citation_quote="Wrong quote",
        status=VerificationStatus.VERIFIED,
        confidence=0.35,
    )
    store = _review_store(
        deal_id="deal_issues",
        company_name="IssueCo",
        evidence=[first_evidence, second_evidence],
        claims=[
            first_claim,
            second_claim,
            missing_citation_claim,
            invalid_citation_claim,
        ],
        conflicts=[
            ClaimConflict(
                id="conflict_active",
                deal_id="deal_issues",
                claim_type=ClaimType.DEAL_TERM,
                label="valuation cap",
                normalized_values=["usd_cents:800000000", "usd_cents:1000000000"],
                claim_ids=[first_claim.id, second_claim.id],
                notes="Multiple valuation caps were extracted.",
            ),
            ClaimConflict(
                id="conflict_stale",
                deal_id="deal_issues",
                claim_type=ClaimType.DEAL_TERM,
                label="discount",
                normalized_values=["basis_points:2000", "basis_points:2500"],
                claim_ids=["missing_claim_id"],
                notes="Stored stale conflict.",
            ),
        ],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])
    normalized_output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "Conflicts and why they matter" in normalized_output
    assert "active" in normalized_output
    assert "stale" in normalized_output
    assert "image_text" in normalized_output
    assert "low_image_text" in normalized_output
    assert "stale_evidence" in normalized_output
    assert "unknown_freshness" in normalized_output
    assert "missing_spans" in normalized_output
    assert "missing_citations" in normalized_output
    assert "invalid_citations" in normalized_output
    assert "quote mismatch" in normalized_output
    assert "low_claim_conf" in normalized_output
    assert "external_conf" in normalized_output
    assert "blocking" in normalized_output
    assert "warning" in normalized_output


def test_review_evidence_shows_table_index_with_page_number(tmp_path: Path) -> None:
    evidence = _review_evidence_record(
        "ev_table",
        "Valuation cap | $8M",
        deal_id="deal_table",
    ).model_copy(
        update={
            "evidence_kind": EvidenceKind.TABLE_TEXT,
            "page_number": 2,
            "table_index": 3,
        }
    )
    store = _review_store(
        deal_id="deal_table",
        company_name="TableLocationCo",
        evidence=[evidence],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "page 2, table 3" in normalized_output


def test_review_evidence_flags_documents_without_evidence_and_ocr_needed(
    tmp_path: Path,
) -> None:
    evidence = _review_evidence_record(
        "ev_readable",
        "Valuation cap $8M.",
        deal_id="deal_mixed_docs",
        document_id="doc_readable",
        document_path=Path("readable-memo.txt"),
    )
    store = _review_store(
        deal_id="deal_mixed_docs",
        company_name="MixedDocsCo",
        evidence=[evidence],
    )
    documents = [
        _review_ingested_document(
            deal_id="deal_mixed_docs",
            document_id="doc_readable",
            path=Path("readable-memo.txt"),
        ),
        _review_ingested_document(
            deal_id="deal_mixed_docs",
            document_id="doc_scan",
            path=Path("scan-only.png"),
            file_type=FileType.PNG,
            extraction_quality=ExtractionQuality.LOW,
            ocr_recommended=True,
            vision_recommended=True,
        ),
    ]
    data_dir = _write_review_ingestion_summary(
        tmp_path,
        [store],
        documents_by_deal_id={"deal_mixed_docs": documents},
    )

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])
    normalized_output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "scan-only.png" in normalized_output
    assert "doc_no_evidence" in normalized_output
    assert "doc_needs_image" in normalized_output
    assert "add readable files" in normalized_output
    assert "manual review" in normalized_output
    assert "No review issues found" not in normalized_output


def test_review_evidence_shows_exact_external_source_reference(tmp_path: Path) -> None:
    source_url = "https://www.sec.gov/Archives/edgar/data/example"
    evidence = _review_evidence_record(
        "ev_external_lineage",
        "ExampleCo filed a Form D.",
        deal_id="deal_external_lineage",
        document_id="external_doc",
        document_path=Path("external-research/sec/example.json"),
        source_kind=SourceKind.WEB,
        source_url=source_url,
    ).model_copy(
        update={
            "page_number": None,
            "table_index": None,
        }
    )
    store = _review_store(
        deal_id="deal_external_lineage",
        company_name="ExternalLineageCo",
        evidence=[evidence],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "Exact source" in result.output
    assert source_url in result.output
    assert "missing page/table location" not in result.output


def test_review_evidence_does_not_flag_text_file_without_page_location(
    tmp_path: Path,
) -> None:
    evidence = _review_evidence_record(
        "ev_text_lineage",
        "Synthetic memo says valuation cap $8M.",
        deal_id="deal_text_lineage",
        document_id="doc_text_lineage",
        document_path=Path("memo.txt"),
        file_type=FileType.TXT,
        page_number=None,
    )
    store = _review_store(
        deal_id="deal_text_lineage",
        company_name="TextLineageCo",
        evidence=[evidence],
    )
    data_dir = _write_review_ingestion_summary(tmp_path, [store])

    result = runner.invoke(app, ["review-evidence", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert "missing page/table location" not in result.output
    assert "missing_location" not in result.output
    assert "complete source lineage" in result.output


def _review_evidence_record(
    evidence_id: str,
    text: str,
    *,
    deal_id: str = "deal_test",
    document_id: str = "doc_test",
    document_path: Path = Path("memo.txt"),
    source_kind: SourceKind = SourceKind.LOCAL_FILE,
    source_freshness: SourceFreshness = SourceFreshness.CURRENT,
    source_span: bool = True,
    file_type: FileType = FileType.TXT,
    page_number: int | None = 1,
    ocr_applied: bool = False,
    ocr_confidence: float | None = None,
    external_confidence: str | None = "high: exact synthetic source",
    source_url: str | None = None,
    source_api: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        id=evidence_id,
        deal_id=deal_id,
        document_id=document_id,
        document_path=document_path,
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=source_kind,
        document_type=DocumentType.MEMO,
        file_type=file_type,
        text=text,
        page_number=page_number,
        source_span_start=0 if source_span else None,
        source_span_end=len(text) if source_span else None,
        ocr_applied=ocr_applied,
        ocr_confidence=ocr_confidence,
        source_freshness=source_freshness,
        external_confidence=external_confidence if source_kind != SourceKind.LOCAL_FILE else None,
        source_url=source_url,
        source_api=source_api,
    )


def _review_claim(
    label: str,
    value: str,
    evidence: EvidenceRecord,
    *,
    include_citation: bool = True,
    citation_quote: str | None = None,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    confidence: float = 0.72,
) -> ClaimRecord:
    quote = citation_quote or f"{label.capitalize()} {value}"
    start = evidence.text.find(quote)
    if start < 0:
        start = 0
    end = start + len(quote)
    citations = (
        [
            EvidenceCitation(
                evidence_id=evidence.id,
                quote=quote,
                source_span_start=start,
                source_span_end=end,
                verification_status=VerificationStatus.VERIFIED,
            )
        ]
        if include_citation
        else []
    )
    return ClaimRecord(
        id=f"claim_{label.replace(' ', '_')}_{evidence.id}",
        deal_id=evidence.deal_id,
        claim_type=ClaimType.DEAL_TERM,
        label=label,
        value=value,
        normalized_value=f"{label}:{value}",
        unit=None,
        raw_text=quote,
        citations=citations,
        verification_status=status,
        quality=EvidenceQuality(
            claim_type=ClaimType.DEAL_TERM,
            source_type=evidence.source_kind,
            verification_status=status,
            recency=evidence.source_freshness,
            reliability="synthetic_test_source",
            confidence=confidence,
            materiality="high",
        ),
    )


def _review_store(
    *,
    deal_id: str,
    company_name: str,
    evidence: list[EvidenceRecord] | None = None,
    claims: list[ClaimRecord] | None = None,
    conflicts: list[ClaimConflict] | None = None,
) -> EvidenceStore:
    evidence_records = (
        evidence
        if evidence is not None
        else [
            _review_evidence_record(
                "ev_default",
                "Valuation cap $8M.",
                deal_id=deal_id,
            )
        ]
    )
    return EvidenceStore(
        deal_id=deal_id,
        company_name=company_name,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        evidence=evidence_records,
        claims=claims or [],
        conflicts=conflicts or [],
    )


def _write_review_ingestion_summary(
    tmp_path: Path,
    stores: list[EvidenceStore],
    *,
    documents_by_deal_id: dict[str, list[IngestedDocument]] | None = None,
) -> Path:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    deals: list[IngestedDeal] = []
    for store in stores:
        store_dir = processed_dir / "deals" / store.deal_id
        store_dir.mkdir(parents=True)
        store_path = store_dir / "evidence_store.json"
        store_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
        deals.append(
            IngestedDeal(
                id=store.deal_id,
                company_name=store.company_name,
                documents=(documents_by_deal_id or {}).get(store.deal_id, []),
                evidence_store_path=store_path,
                evidence_count=store.evidence_count,
                claim_count=store.claim_count,
                conflict_count=store.conflict_count,
            )
        )

    summary = IngestionSummary(
        root_path=tmp_path / "pitch-decks",
        scanned_at=datetime(2026, 1, 1, tzinfo=UTC),
        deals=deals,
        summary_path=processed_dir / "ingestion_summary.json",
    )
    summary.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return data_dir


def _review_ingested_document(
    *,
    deal_id: str,
    document_id: str,
    path: Path,
    file_type: FileType = FileType.TXT,
    extraction_quality: ExtractionQuality = ExtractionQuality.HIGH,
    ocr_recommended: bool = False,
    vision_recommended: bool = False,
) -> IngestedDocument:
    return IngestedDocument(
        source=SourceDocument(
            id=document_id,
            deal_id=deal_id,
            path=path,
            source_kind=SourceKind.LOCAL_FILE,
            document_type=DocumentType.MEMO,
            file_type=file_type,
            title=path.name,
            ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
            sha256="synthetic-test-sha",
            extraction_quality=extraction_quality,
            ocr_recommended=ocr_recommended,
            vision_recommended=vision_recommended,
        ),
        pages=[],
        tables=[],
        output_path=Path("processed") / f"{document_id}.json",
    )
