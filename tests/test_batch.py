from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary import batch as batch_module
from hailmary.batch import (
    BatchAllocationRow,
    BatchDealOutcome,
    BatchEvaluationError,
    BatchEvaluationResult,
    BatchPortfolioConstraints,
    batch_evaluate_folder,
    render_batch_portfolio_report,
)
from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.evaluation import DealEvaluationResult
from hailmary.schemas.evidence import EvidenceStore
from hailmary.schemas.scoring import Recommendation

runner = CliRunner()
PRIVATE_MARKER = "PRIVATE_BATCH_SOURCE_TEXT_DO_NOT_PRINT"


def test_batch_evaluate_allocates_and_continues_after_failed_deal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "batch-deals"
    _write_deal(
        root,
        "A StrongCo",
        _strong_investable_text(),
    )
    _write_deal(
        root,
        "B BudgetCo",
        _strong_investable_text(),
    )
    _write_deal(root, "MissingEvidenceCo", "Company overview only.")
    _write_deal(
        root,
        "HighValuationCo",
        (
            "Seed company with a beta design partner. Valuation cap $60M. "
            "Discount 20%. Round size $1M. Lead investor committed."
        ),
    )
    broken_dir = root / "BrokenCo"
    broken_dir.mkdir(parents=True)
    (broken_dir / "malformed.bin").write_bytes(b"\x00synthetic unsupported input")
    data_dir = tmp_path / "private-data"

    result = batch_evaluate_folder(
        root,
        config=AppConfig(
            data_dir=data_dir,
            local_only=True,
            capital_budget=1_000,
        ),
        max_concurrency=1,
        run_research=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.deal_count == 5
    assert result.evaluated_count == 4
    assert result.failed_count == 1
    assert result.allocated_count == 1
    rows = {row.company_name: row for row in result.allocation_rows}
    assert rows["A StrongCo"].evaluation_status == "allocated"
    assert rows["A StrongCo"].final_recommendation == Recommendation.INVEST
    assert rows["A StrongCo"].batch_check_size == 1_000
    assert rows["B BudgetCo"].evaluation_status == "skipped"
    assert rows["B BudgetCo"].batch_check_size == 0
    assert rows["B BudgetCo"].skipped_reason is not None
    assert "capital" in rows["B BudgetCo"].skipped_reason.casefold()
    assert rows["MissingEvidenceCo"].final_recommendation == Recommendation.PASS
    assert rows["MissingEvidenceCo"].batch_check_size == 0
    assert rows["HighValuationCo"].final_recommendation == Recommendation.PASS
    assert rows["HighValuationCo"].batch_check_size == 0
    assert rows["BrokenCo"].evaluation_status == "failed"
    assert rows["BrokenCo"].final_recommendation == Recommendation.PASS
    assert rows["BrokenCo"].batch_check_size == 0

    for path in (result.report_path, result.json_path):
        path.resolve(strict=True).relative_to(data_dir.resolve(strict=True))

    export = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert export["privacy"]["contains_raw_evidence_text"] is False
    assert export["privacy"]["contains_model_excerpts"] is False
    assert export["portfolio_constraints"]["allowed_check_sizes"] == [0, 1_000]
    serialized_export = json.dumps(export, sort_keys=True)
    report_text = result.report_path.read_text(encoding="utf-8")
    assert PRIVATE_MARKER not in serialized_export
    assert PRIVATE_MARKER not in report_text
    assert "Valuation cap $8M" not in serialized_export
    assert "Valuation cap $8M" not in report_text


def test_batch_evaluate_cli_prints_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "true")
    monkeypatch.setenv("HAILMARY_MOCK_LLM", "true")
    root = tmp_path / "batch-cli-deals"
    _write_deal(root, "CliStrongCo", _strong_investable_text())
    broken_dir = root / "CliBrokenCo"
    broken_dir.mkdir(parents=True)
    (broken_dir / "malformed.bin").write_bytes(b"\x00synthetic unsupported input")

    result = runner.invoke(
        app,
        [
            "batch-evaluate",
            str(root),
            "--data-dir",
            str(tmp_path / "private-data"),
            "--skip-research",
            "--max-concurrency",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "Batch evaluation complete" in normalized_output
    assert "Deals found" in normalized_output
    assert "Deal-level failures" in normalized_output
    assert "CliStrongCo" in normalized_output
    assert "CliBrokenCo" in normalized_output
    assert "Final" in normalized_output
    assert "Batch check" in normalized_output
    assert "Batch JSON" in normalized_output
    assert "raw evidence text" in normalized_output
    assert PRIVATE_MARKER not in result.output
    assert "Valuation cap $8M" not in result.output
    assert "\x1b[" not in result.output


def test_batch_evaluate_accepts_private_raw_batch_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    raw_root = data_dir / "raw"
    _write_deal(raw_root, "RawStrongCo", _strong_investable_text())

    result = batch_evaluate_folder(
        raw_root,
        config=AppConfig(
            data_dir=data_dir,
            local_only=True,
            capital_budget=1_000,
        ),
        max_concurrency=1,
        run_research=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.deal_count == 1
    assert result.evaluated_count == 1
    assert result.allocated_count == 1
    assert result.allocation_rows[0].company_name == "RawStrongCo"


def test_batch_ranks_before_applying_scarce_capital_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "ranked-deals"
    _write_deal(
        root,
        "HighMinimumCo",
        (
            f"{_strong_investable_text()} Investor ownership 100%. "
            "Estimated dilution 20%. SPV expenses 5%. Carry 20%. "
            "Exit value $1B. Minimum investment $5,000."
        ),
    )
    _write_deal(root, "LowerMinimumCo", _strong_investable_text())

    result = batch_evaluate_folder(
        root,
        config=AppConfig(
            data_dir=tmp_path / "private-data",
            local_only=True,
            capital_budget=1_000,
        ),
        max_concurrency=1,
        run_research=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    rows = {row.company_name: row for row in result.allocation_rows}
    assert rows["HighMinimumCo"].portfolio_rank == 1
    assert rows["HighMinimumCo"].evaluation_status == "skipped"
    assert rows["HighMinimumCo"].batch_check_size == 0
    assert rows["HighMinimumCo"].skipped_reason is not None
    assert "budget" in rows["HighMinimumCo"].skipped_reason.casefold()
    assert rows["LowerMinimumCo"].portfolio_rank == 2
    assert rows["LowerMinimumCo"].evaluation_status == "allocated"
    assert rows["LowerMinimumCo"].batch_check_size == 1_000


def test_batch_evaluate_fails_when_no_child_deal_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "all-broken"
    for company_name in ("BrokenOne", "BrokenTwo"):
        broken_dir = root / company_name
        broken_dir.mkdir(parents=True)
        (broken_dir / "malformed.bin").write_bytes(b"\x00synthetic unsupported input")

    with pytest.raises(BatchEvaluationError, match="BrokenOne.*BrokenTwo"):
        batch_evaluate_folder(
            root,
            config=AppConfig(
                data_dir=tmp_path / "private-data",
                local_only=True,
            ),
            max_concurrency=1,
            run_research=False,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_batch_evaluate_cli_fails_when_no_child_deal_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "true")
    monkeypatch.setenv("HAILMARY_MOCK_LLM", "true")
    root = tmp_path / "all-broken-cli"
    for company_name in ("BrokenOne", "BrokenTwo"):
        broken_dir = root / company_name
        broken_dir.mkdir(parents=True)
        (broken_dir / "malformed.bin").write_bytes(b"\x00synthetic unsupported input")

    result = runner.invoke(
        app,
        [
            "batch-evaluate",
            str(root),
            "--data-dir",
            str(tmp_path / "private-data"),
            "--skip-research",
            "--max-concurrency",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "No deal could be evaluated successfully" in result.output
    assert "BrokenOne" in result.output
    assert "BrokenTwo" in result.output
    assert "Batch evaluation complete" not in result.output


def test_batch_evaluate_fails_when_all_allocation_rows_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "allocation-fails"
    _write_deal(root, "StrongCo", _strong_investable_text())

    def fail_store_load(*args: object, **kwargs: object) -> None:
        raise BatchEvaluationError("Synthetic allocation failure.")

    monkeypatch.setattr(batch_module, "_load_actioned_store_for_result", fail_store_load)

    with pytest.raises(BatchEvaluationError, match="No deal could be evaluated"):
        batch_evaluate_folder(
            root,
            config=AppConfig(
                data_dir=tmp_path / "private-data",
                local_only=True,
            ),
            max_concurrency=1,
            run_research=False,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_batch_counts_allocation_stage_failures_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "allocation-mixed"
    _write_deal(root, "ActionFailureCo", _strong_investable_text())
    _write_deal(root, "HealthyCo", _strong_investable_text())
    original_loader = batch_module._load_actioned_store_for_result

    def fail_one_store_load(
        result: DealEvaluationResult,
        *,
        config: AppConfig,
    ) -> EvidenceStore:
        if result.company_name == "ActionFailureCo":
            raise BatchEvaluationError("Synthetic allocation reload failure.")
        return original_loader(result, config=config)

    monkeypatch.setattr(
        batch_module,
        "_load_actioned_store_for_result",
        fail_one_store_load,
    )

    result = batch_evaluate_folder(
        root,
        config=AppConfig(
            data_dir=tmp_path / "private-data",
            local_only=True,
            capital_budget=2_000,
        ),
        max_concurrency=1,
        run_research=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.deal_count == 2
    assert result.evaluated_count == 1
    assert result.failed_count == 1
    rows = {row.company_name: row for row in result.allocation_rows}
    assert rows["ActionFailureCo"].evaluation_status == "failed"
    assert rows["HealthyCo"].evaluation_status == "allocated"


def test_batch_rejects_symlinked_parent_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    real_parent = tmp_path / "real-parent"
    deals_root = real_parent / "deals"
    _write_deal(deals_root, "StrongCo", _strong_investable_text())
    symlink_parent = tmp_path / "linked-parent"
    try:
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks are not available in this environment: {exc}")

    with pytest.raises(BatchEvaluationError, match="symlinked parent"):
        batch_evaluate_folder(
            symlink_parent / "deals",
            config=AppConfig(
                data_dir=tmp_path / "private-data",
                local_only=True,
            ),
            max_concurrency=1,
            run_research=False,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_batch_final_pass_rows_report_final_pass_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "final-pass-row"
    folder = _write_deal(root, "StrongCo", _strong_investable_text())

    result = batch_evaluate_folder(
        root,
        config=AppConfig(
            data_dir=tmp_path / "private-data",
            local_only=True,
            capital_budget=1_000,
        ),
        max_concurrency=1,
        run_research=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    original_row = next(row for row in result.allocation_rows if row.company_name == "StrongCo")
    assert original_row.final_recommendation == Recommendation.INVEST
    evaluation_result = next(outcome.result for outcome in result.outcomes if outcome.result)
    final_pass = evaluation_result.final_recommendation.model_copy(
        update={
            "recommendation": Recommendation.PASS,
            "check_size": 0,
            "reason": "Final review found a blocking source-linked risk.",
        }
    )
    overridden_result = replace(evaluation_result, final_recommendation=final_pass)

    row = batch_module._row_from_success(
        BatchDealOutcome(
            folder=folder,
            company_name="StrongCo",
            result=overridden_result,
        ),
        portfolio_rank=1,
        ranking_score=overridden_result.deterministic_score,
    )

    assert row.skipped_reason == "Final recommendation was PASS."
    assert row.key_blockers == [
        "Final PASS: review the child memo for the cited final-decision blocker."
    ]


def test_batch_markdown_report_escapes_untrusted_html_delimiters() -> None:
    root = Path("batch <img src=x onerror=alert(1)>")
    report = render_batch_portfolio_report(
        BatchEvaluationResult(
            root_folder=root,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            report_path=Path("data/reports/batch.md"),
            json_path=Path("data/reports/batch.json"),
            outcomes=[
                BatchDealOutcome(
                    folder=root / "Bad <script>alert(1)</script>",
                    company_name="Bad <script>alert(1)</script>",
                    failure_reason="Failed <img src=x onerror=alert(1)> | [bad](url)",
                )
            ],
            allocation_rows=[
                BatchAllocationRow(
                    company_name="Bad <script>alert(1)</script>",
                    deal_id=None,
                    folder=root / "Bad <script>alert(1)</script>",
                    evaluation_status="failed",
                    final_recommendation=Recommendation.PASS,
                    single_deal_check_size=0,
                    batch_check_size=0,
                    score=None,
                    max_score=None,
                    confidence=None,
                    key_blockers=["Failed <img src=x onerror=alert(1)> | [bad](url)"],
                    failure_reason="Failed <img src=x onerror=alert(1)> | [bad](url)",
                )
            ],
            constraints=BatchPortfolioConstraints(
                starting_capital=10_000,
                reserve_amount=0,
                allocatable_capital=10_000,
                existing_invested_capital=0,
                available_capital_before_batch=10_000,
                new_allocated_capital=0,
                remaining_allocatable_capital=10_000,
                allowed_check_sizes=[0, 1_000, 2_500, 5_000, 7_500, 10_000],
                configured_min_check=1_000,
                configured_max_check=10_000,
                max_company_exposure_percent="10",
                max_category_exposure_percent="25",
                max_stage_exposure_percent="30",
                max_low_confidence_exposure_percent="5",
                max_medium_confidence_exposure_percent="10",
                max_high_confidence_exposure_percent="20",
            ),
        )
    )

    assert "<img" not in report
    assert "<script" not in report
    assert "&lt;img" in report
    assert "&lt;script" in report
    assert "\\|" in report
    assert "\\[bad\\]\\(url\\)" in report


def _write_deal(root: Path, company_name: str, body: str) -> Path:
    company_dir = root / company_name
    company_dir.mkdir(parents=True)
    (company_dir / "memo.txt").write_text(body, encoding="utf-8")
    return company_dir


def _strong_investable_text() -> str:
    return (
        "Seed stage. Category: synthetic software. Valuation cap $8M. "
        "Discount 20%. Round size $1M. ARR revenue growth with paid customers "
        "and retention. Lead investor committed and seed round is active. "
        f"{PRIVATE_MARKER}"
    )
