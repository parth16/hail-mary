from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hailmary.batch import BatchEvaluationError, batch_evaluate_folder
from hailmary.cli import app
from hailmary.config import AppConfig
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
