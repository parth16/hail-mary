from __future__ import annotations

import os
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from hailmary.config import AppConfig
from hailmary.portfolio import (
    PortfolioError,
    add_portfolio_investment,
    load_portfolio_ledger,
    portfolio_status,
)


def test_add_portfolio_investment_writes_private_ledger(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        capital_budget=10_000,
        reserve_dollars=1_000,
    )

    ledger, investment, ledger_path = add_portfolio_investment(
        config=config,
        company_name="  ExampleCo   AI ",
        amount=5_000,
        invested_on=date(2026, 6, 23),
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )

    assert investment.company_name == "ExampleCo AI"
    assert investment.amount == 5_000
    assert investment.invested_on == date(2026, 6, 23)
    assert ledger.investment_count == 1
    assert ledger.invested_amount == 5_000
    assert ledger_path == tmp_path / "data" / "portfolio" / "ledger.json"
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "data" / "portfolio").stat().st_mode) == 0o700
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600

    reloaded = load_portfolio_ledger(config)
    assert reloaded.invested_amount == 5_000
    status = portfolio_status(config)
    assert status.invested_amount == 5_000
    assert status.available_capital == 4_000
    assert status.allowed_check_tiers == [0, 1_000, 2_500]


def test_portfolio_status_handles_missing_ledger(tmp_path: Path) -> None:
    status = portfolio_status(AppConfig(data_dir=tmp_path / "data"))

    assert status.investment_count == 0
    assert status.invested_amount == 0
    assert status.available_capital == 100_000


def test_add_portfolio_investment_blocks_duplicates(tmp_path: Path) -> None:
    config = AppConfig(data_dir=tmp_path / "data")
    add_portfolio_investment(
        config=config,
        company_name="ExampleCo",
        amount=2_500,
        invested_on=date(2026, 6, 23),
    )

    with pytest.raises(PortfolioError, match="already recorded"):
        add_portfolio_investment(
            config=config,
            company_name=" exampleco ",
            amount=2_500,
            invested_on=date(2026, 6, 23),
        )


def test_load_portfolio_ledger_rejects_bad_json(tmp_path: Path) -> None:
    ledger_path = tmp_path / "data" / "portfolio" / "ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(PortfolioError, match="not valid JSON"):
        load_portfolio_ledger(AppConfig(data_dir=tmp_path / "data"))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="Symlinks are not supported here")
def test_load_portfolio_ledger_rejects_symlinked_portfolio_folder(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    outside_dir = tmp_path / "outside"
    data_dir.mkdir()
    outside_dir.mkdir()
    (data_dir / "portfolio").symlink_to(outside_dir, target_is_directory=True)
    (outside_dir / "ledger.json").write_text(
        '{"version":"1","investments":[]}',
        encoding="utf-8",
    )

    with pytest.raises(PortfolioError, match="cannot be a symlink"):
        load_portfolio_ledger(AppConfig(data_dir=data_dir))


def test_load_portfolio_ledger_rejects_portfolio_folder_as_file(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "portfolio").write_text("not a folder", encoding="utf-8")

    with pytest.raises(PortfolioError, match="must be a folder"):
        load_portfolio_ledger(AppConfig(data_dir=data_dir))
