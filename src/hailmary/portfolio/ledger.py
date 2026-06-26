from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hailmary.config import AppConfig, validate_local_state
from hailmary.portfolio.scenario import (
    PortfolioScenario,
    allowed_check_tiers_for_available_capital,
    portfolio_scenario,
)
from hailmary.utils.slug import slugify


class PortfolioError(RuntimeError):
    """The private portfolio ledger could not be used safely."""


class PortfolioInvestment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    company_name: str
    amount: int
    invested_on: date
    created_at: datetime

    @field_validator("id")
    @classmethod
    def _clean_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("id cannot be blank")
        return cleaned

    @field_validator("company_name")
    @classmethod
    def _clean_company_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("company_name cannot be blank")
        return cleaned

    @field_validator("amount")
    @classmethod
    def _positive_amount(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("amount must be greater than zero")
        return value


class PortfolioLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    updated_at: datetime | None = None
    investments: list[PortfolioInvestment] = Field(default_factory=list)

    @property
    def investment_count(self) -> int:
        return len(self.investments)

    @property
    def invested_amount(self) -> int:
        return sum(investment.amount for investment in self.investments)


@dataclass(frozen=True)
class PortfolioStatus:
    ledger_path: Path
    ledger: PortfolioLedger
    scenario: PortfolioScenario
    invested_amount: int
    available_capital: int
    over_allocated_amount: int
    allowed_check_tiers: list[int]

    @property
    def investment_count(self) -> int:
        return self.ledger.investment_count


def portfolio_ledger_path(config: AppConfig) -> Path:
    config = validate_local_state(config, update_git_exclude=False)
    return config.data_dir / "portfolio" / "ledger.json"


def load_portfolio_ledger(config: AppConfig) -> PortfolioLedger:
    config = validate_local_state(config, update_git_exclude=False)
    ledger_path = config.data_dir / "portfolio" / "ledger.json"
    _validate_private_directory(ledger_path.parent, private_root=config.data_dir)
    if ledger_path.is_symlink():
        raise PortfolioError("The portfolio ledger file cannot be a symlink.")
    if not ledger_path.exists():
        return PortfolioLedger()
    if not ledger_path.is_file():
        raise PortfolioError(
            f"Hail Mary needs {ledger_path} to be a file, but it is a folder."
        )
    try:
        raw_text = ledger_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PortfolioError(
            "The portfolio ledger is not plain UTF-8 text. Restore it from a valid JSON backup."
        ) from exc
    except OSError as exc:
        raise PortfolioError(
            f"Could not read the portfolio ledger at {ledger_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PortfolioError(f"The portfolio ledger is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise PortfolioError("The portfolio ledger must be a JSON object.")
    try:
        return PortfolioLedger.model_validate(payload)
    except ValidationError as exc:
        raise PortfolioError(
            f"The portfolio ledger is incomplete: {_first_validation_detail(exc)}"
        ) from exc


def add_portfolio_investment(
    *,
    config: AppConfig,
    company_name: str,
    amount: int,
    invested_on: date,
    created_at: datetime | None = None,
) -> tuple[PortfolioLedger, PortfolioInvestment, Path]:
    config = validate_local_state(config)
    cleaned_company_name = " ".join(company_name.split())
    if not cleaned_company_name:
        raise PortfolioError("The company name cannot be blank.")
    if amount <= 0:
        raise PortfolioError("The investment amount must be greater than zero.")
    ledger_path = config.data_dir / "portfolio" / "ledger.json"
    _ensure_private_directory(config.data_dir, private_root=config.data_dir)
    _ensure_private_directory(ledger_path.parent, private_root=config.data_dir)
    ledger = load_portfolio_ledger(config)
    try:
        investment = PortfolioInvestment(
            id=_investment_id(
                company_name=cleaned_company_name,
                amount=amount,
                invested_on=invested_on,
            ),
            company_name=cleaned_company_name,
            amount=amount,
            invested_on=invested_on,
            created_at=created_at or datetime.now(UTC),
        )
    except ValidationError as exc:
        raise PortfolioError(
            f"The portfolio investment is incomplete: {_first_validation_detail(exc)}"
        ) from exc
    _ensure_not_duplicate(ledger, investment)
    updated_ledger = ledger.model_copy(
        update={
            "updated_at": investment.created_at,
            "investments": [*ledger.investments, investment],
        }
    )
    _write_private_ledger(ledger_path, updated_ledger)
    return updated_ledger, investment, ledger_path


def portfolio_status(config: AppConfig) -> PortfolioStatus:
    config = validate_local_state(config, update_git_exclude=False)
    ledger_path = config.data_dir / "portfolio" / "ledger.json"
    ledger = load_portfolio_ledger(config)
    scenario = portfolio_scenario(config)
    invested_amount = ledger.invested_amount
    available_capital = max(0, scenario.allocatable_capital - invested_amount)
    over_allocated_amount = max(0, invested_amount - scenario.allocatable_capital)
    return PortfolioStatus(
        ledger_path=ledger_path,
        ledger=ledger,
        scenario=scenario,
        invested_amount=invested_amount,
        available_capital=available_capital,
        over_allocated_amount=over_allocated_amount,
        allowed_check_tiers=allowed_check_tiers_for_available_capital(
            config,
            available_capital=available_capital,
        ),
    )


def _investment_id(*, company_name: str, amount: int, invested_on: date) -> str:
    cleaned_company_name = " ".join(company_name.split())
    digest = hashlib.sha256(
        f"{cleaned_company_name.casefold()}|{amount}|{invested_on.isoformat()}".encode()
    ).hexdigest()[:12]
    return f"inv_{slugify(cleaned_company_name)}_{invested_on:%Y%m%d}_{digest}"


def _ensure_not_duplicate(
    ledger: PortfolioLedger,
    investment: PortfolioInvestment,
) -> None:
    for existing in ledger.investments:
        if existing.id == investment.id:
            raise PortfolioError(
                "That investment is already recorded. Hail Mary matched the same company, "
                "amount, and investment date."
            )


def _write_private_ledger(path: Path, ledger: PortfolioLedger) -> None:
    if path.is_symlink():
        raise PortfolioError("The portfolio ledger file cannot be a symlink.")
    tmp_path = path.with_suffix(".tmp")
    if tmp_path.is_symlink():
        raise PortfolioError("The temporary portfolio ledger file cannot be a symlink.")
    payload = ledger.model_dump_json(indent=2)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.chmod(0o600)
        os.replace(tmp_path, path)
        path.chmod(0o600)
    except OSError as exc:
        raise PortfolioError(f"Could not write the portfolio ledger at {path}: {exc}") from exc
    finally:
        if tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


def _ensure_private_directory(path: Path, *, private_root: Path) -> None:
    _validate_private_directory(path, private_root=private_root)
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError as exc:
        raise PortfolioError(
            f"Could not create the portfolio ledger folder at {path}: {exc}"
        ) from exc


def _validate_private_directory(path: Path, *, private_root: Path) -> None:
    if path.is_symlink():
        raise PortfolioError(f"The portfolio ledger folder {path} cannot be a symlink.")
    root = private_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PortfolioError(
            f"The portfolio ledger folder {path} resolves outside the private data directory."
        ) from exc
    if path.exists() and not path.is_dir():
        raise PortfolioError(
            f"The portfolio ledger folder {path} must be a folder, but it is a file."
        )


def _first_validation_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid value"
    first_error = errors[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "invalid value"))
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ")
    if location:
        return f"{location}: {message}"
    return message
