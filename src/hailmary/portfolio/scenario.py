from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext

from hailmary.config import CHECK_SIZE_TIERS, AppConfig

PERCENT_BASE = Decimal("100")


@dataclass(frozen=True)
class PortfolioScenario:
    starting_capital: int
    reserve_amount: int
    allocatable_capital: int
    min_check: int
    max_check: int
    reserve_percent: Decimal
    reserve_dollars: int
    estimated_dilution_percent: Decimal
    platform_fee_percent: Decimal
    carry_percent: Decimal
    gross_return_multiple: Decimal


def portfolio_scenario(config: AppConfig) -> PortfolioScenario:
    reserve_amount = _reserve_amount(config)
    return PortfolioScenario(
        starting_capital=config.capital_budget,
        reserve_amount=reserve_amount,
        allocatable_capital=max(0, config.capital_budget - reserve_amount),
        min_check=config.min_check,
        max_check=config.max_check,
        reserve_percent=config.reserve_percent,
        reserve_dollars=config.reserve_dollars,
        estimated_dilution_percent=config.estimated_dilution_percent,
        platform_fee_percent=config.platform_fee_percent,
        carry_percent=config.carry_percent,
        gross_return_multiple=config.gross_return_multiple,
    )


def allowed_check_tiers_for_available_capital(
    config: AppConfig,
    *,
    available_capital: int,
) -> list[int]:
    maximum_nonzero_check = min(config.max_check, max(0, available_capital))
    return [
        tier
        for tier in CHECK_SIZE_TIERS
        if tier == 0 or config.min_check <= tier <= maximum_nonzero_check
    ]


def _reserve_amount(config: AppConfig) -> int:
    if config.reserve_dollars > 0:
        return config.reserve_dollars
    if config.reserve_percent <= 0:
        return 0
    capital_budget = Decimal(config.capital_budget)
    with localcontext() as context:
        context.prec = max(
            context.prec,
            _decimal_digit_count(capital_budget)
            + _decimal_digit_count(config.reserve_percent)
            + _decimal_digit_count(PERCENT_BASE)
            + 4,
        )
        reserve = capital_budget * config.reserve_percent / PERCENT_BASE
    return int(reserve.to_integral_value(rounding=ROUND_CEILING))


def _decimal_digit_count(value: Decimal) -> int:
    return len(value.as_tuple().digits)
