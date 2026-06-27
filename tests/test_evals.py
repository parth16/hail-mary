from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hailmary.cli as cli_module
from hailmary.cli import app
from hailmary.evals import EvalCategory, EvalHarnessError, builtin_eval_cases, fixtures
from hailmary.evals.runner import run_builtin_evals
from hailmary.evals.schemas import EvalCaseResult, EvalRunSummary

runner = CliRunner()


def test_builtin_eval_metadata_covers_required_phase_6_categories() -> None:
    categories = {case.category for case in builtin_eval_cases()}

    assert categories == {
        EvalCategory.EXTRACTION,
        EvalCategory.OCR,
        EvalCategory.CITATION,
        EvalCategory.CONTRADICTION,
        EvalCategory.RESEARCH_IMPORT,
        EvalCategory.PUBLIC_COLLECTORS,
        EvalCategory.MERIDIAN,
        EvalCategory.PROMPT_INJECTION,
        EvalCategory.SCORE_CALIBRATION,
        EvalCategory.MISSING_DATA,
        EvalCategory.MEMO_SNAPSHOT,
        EvalCategory.PRIVACY,
    }


def test_run_builtin_evals_passes_all_synthetic_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(work_dir=tmp_path)

    assert summary.total_count == 47
    assert summary.passed
    assert summary.failed_results == []


def test_run_builtin_evals_filters_evaluate_deal_golden_case(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        case_ids=["evaluate-deal-golden-workflow"],
        work_dir=tmp_path,
    )

    assert [result.id for result in summary.results] == ["evaluate-deal-golden-workflow"]
    assert summary.passed


def test_run_builtin_evals_filters_by_category(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.SCORE_CALIBRATION],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "score-strong-invest",
        "score-borderline-pass",
        "score-missing-terms-pass",
        "score-high-valuation-pass",
        "score-stale-conflicting-pass",
        "score-stage-aware-v2",
        "score-return-math-missing-v2",
        "score-calibration-guards",
        "score-strong-team-weak-pmf-pass",
        "score-high-traction-overvalued-pass",
        "score-missing-deal-terms-v3-pass",
        "score-conflicting-revenue-customers-pass",
        "score-stale-public-validation-pass",
        "score-model-invest-guardrail-pass",
        "score-no-or-unsafe-evidence-pass",
        "score-small-budget-pass",
        "score-platform-minimum-above-capital-pass",
        "score-negated-traction-funding-pass",
    }
    assert summary.passed


def test_run_builtin_evals_filters_by_ocr_category(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.OCR],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "ocr-image-unavailable",
        "ocr-fake-success-source-linked",
        "ocr-prompt-injection-untrusted",
    }
    assert summary.passed


def test_run_builtin_evals_filters_prompt_injection_document_cases(
    tmp_path: Path,
) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.PROMPT_INJECTION],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "prompt-injection-recommendation",
        "prompt-injection-pdf-recommendation",
        "prompt-injection-docx-recommendation",
        "prompt-injection-boundaries",
    }
    assert summary.passed


def test_run_builtin_evals_filters_public_collector_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.PUBLIC_COLLECTORS],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "public-collectors-source-guards",
    }
    assert summary.passed


def test_run_builtin_evals_filters_meridian_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.MERIDIAN],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "meridian-workflow-guards",
    }
    assert summary.passed


def test_run_builtin_evals_filters_privacy_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.PRIVACY],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "privacy-output-guards",
    }
    assert summary.passed


def test_run_builtin_evals_filters_research_import_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.RESEARCH_IMPORT],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "research-free-public-collectors-v2",
        "research-public-source-import",
        "research-workflow-v2",
        "research-workflow-v4",
        "research-usaspending-api-pagination",
    }
    assert summary.passed


def test_run_builtin_evals_filters_by_case_id(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        case_ids=["citation-span-mismatch"],
        work_dir=tmp_path,
    )

    assert [result.id for result in summary.results] == ["citation-span-mismatch"]
    assert summary.passed


def test_run_builtin_evals_rejects_filters_that_match_nothing(tmp_path: Path) -> None:
    with pytest.raises(EvalHarnessError, match="No evals matched"):
        run_builtin_evals(
            case_ids=["citation-span-mismatch"],
            categories=[EvalCategory.SCORE_CALIBRATION],
            work_dir=tmp_path,
        )


def test_run_builtin_evals_rejects_unknown_case_even_with_valid_case(
    tmp_path: Path,
) -> None:
    with pytest.raises(EvalHarnessError, match="Unknown eval case ID: missing-case"):
        run_builtin_evals(
            case_ids=["citation-span-mismatch", "missing-case"],
            work_dir=tmp_path,
        )


def test_run_builtin_evals_reports_fixture_failure_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fixture() -> None:
        raise fixtures.EvalFixtureFailure(
            "Expected a specific scoring result.",
            {"expected": "PASS", "actual": "INVEST"},
        )

    monkeypatch.setattr(fixtures, "run_citation_fixture", fail_fixture)

    summary = run_builtin_evals(case_ids=["citation-span-mismatch"], work_dir=tmp_path)

    assert not summary.passed
    assert summary.failed_results[0].message == "Expected a specific scoring result."
    assert summary.failed_results[0].details["expected"] == "PASS"
    assert summary.failed_results[0].details["actual"] == "INVEST"


def test_run_evals_command_reports_passes() -> None:
    result = runner.invoke(app, ["run-evals", "--case", "citation-span-mismatch"])

    assert result.exit_code == 0, result.output
    assert "Eval results" in result.output
    assert "Ran 1 synthetic eval. 1 passed, 0 failed." in result.output
    assert "Traceback" not in result.output


def test_run_evals_command_prints_json() -> None:
    result = runner.invoke(
        app,
        ["run-evals", "--case", "citation-span-mismatch", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["results"][0]["id"] == "citation-span-mismatch"
    assert payload["results"][0]["passed"] is True


def test_run_evals_command_rejects_unknown_category() -> None:
    result = runner.invoke(app, ["run-evals", "--category", "unknown"])

    assert result.exit_code != 0
    assert "Unknown eval category 'unknown'" in result.output
    assert "Traceback" not in result.output


def test_run_evals_command_rejects_unknown_case() -> None:
    result = runner.invoke(
        app,
        ["run-evals", "--case", "citation-span-mismatch", "--case", "missing-case"],
    )

    assert result.exit_code != 0
    assert "Unknown eval case ID: missing-case" in result.output
    assert "Traceback" not in result.output


def test_run_evals_command_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_builtin_evals(
        *,
        case_ids: Sequence[str] | None = None,
        categories: Sequence[EvalCategory] | None = None,
    ) -> EvalRunSummary:
        assert case_ids == []
        assert categories == []
        return EvalRunSummary(
            results=[
                EvalCaseResult(
                    id="synthetic-failure",
                    category=EvalCategory.CITATION,
                    name="Synthetic failure",
                    passed=False,
                    message="The expected behavior did not happen.",
                    details={
                        "description": "Hidden in normal output.",
                        "expected": "PASS",
                        "actual": "INVEST",
                    },
                )
            ]
        )

    monkeypatch.setattr(cli_module, "run_builtin_evals", fake_run_builtin_evals)

    result = runner.invoke(app, ["run-evals"])

    assert result.exit_code != 0
    assert "Eval results" in result.output
    assert "Ran 1 synthetic eval. 0 passed, 1 failed." in result.output
    assert "- synthetic-failure: The expected behavior did not happen." in result.output
    assert "expected: PASS" in result.output
    assert "actual: INVEST" in result.output
    assert "Hidden in normal output" not in result.output
    assert "Traceback" not in result.output
