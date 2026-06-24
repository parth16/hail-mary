from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hailmary.cli as cli_module
from hailmary.cli import app
from hailmary.evals import EvalCategory, EvalHarnessError, builtin_eval_cases
from hailmary.evals.runner import run_builtin_evals
from hailmary.evals.schemas import EvalCaseResult, EvalRunSummary

runner = CliRunner()


def test_builtin_eval_metadata_covers_required_phase_6_categories() -> None:
    categories = {case.category for case in builtin_eval_cases()}

    assert categories == {
        EvalCategory.EXTRACTION,
        EvalCategory.CITATION,
        EvalCategory.CONTRADICTION,
        EvalCategory.PROMPT_INJECTION,
        EvalCategory.SCORE_CALIBRATION,
    }


def test_run_builtin_evals_passes_all_synthetic_cases(tmp_path: Path) -> None:
    summary = run_builtin_evals(work_dir=tmp_path)

    assert summary.total_count == 6
    assert summary.passed
    assert summary.failed_results == []


def test_run_builtin_evals_filters_by_category(tmp_path: Path) -> None:
    summary = run_builtin_evals(
        categories=[EvalCategory.SCORE_CALIBRATION],
        work_dir=tmp_path,
    )

    assert {result.id for result in summary.results} == {
        "score-strong-invest",
        "score-borderline-pass",
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
        run_builtin_evals(case_ids=["missing-case"], work_dir=tmp_path)


def test_run_evals_command_reports_passes() -> None:
    result = runner.invoke(app, ["run-evals", "--case", "citation-span-mismatch"])

    assert result.exit_code == 0, result.output
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
                )
            ]
        )

    monkeypatch.setattr(cli_module, "run_builtin_evals", fake_run_builtin_evals)

    result = runner.invoke(app, ["run-evals"])

    assert result.exit_code != 0
    assert "Ran 1 synthetic eval. 0 passed, 1 failed." in result.output
    assert "- synthetic-failure: The expected behavior did not happen." in result.output
    assert "Traceback" not in result.output
