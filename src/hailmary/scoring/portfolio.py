from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from hailmary.config import CHECK_SIZE_TIERS, AppConfig
from hailmary.schemas.scoring import Recommendation, ScoredDeal

CONFIDENCE_RANK = {
    "high": 3,
    "medium": 2,
    "low": 1,
}
INVEST_MINIMUM_SCORE = 75
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


@dataclass(frozen=True)
class SkippedDeal:
    rank: int
    company_name: str
    score_text: str
    reason: str


@dataclass(frozen=True)
class PortfolioReturnCase:
    label: str
    gross_return_multiple: Decimal
    invested_capital: int
    gross_value_before_dilution: Decimal
    value_after_dilution: Decimal
    platform_fee: Decimal
    carry: Decimal
    net_cash_returned: Decimal
    net_profit_after_fees: Decimal
    net_multiple: Decimal


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


def portfolio_rank_key(deal: ScoredDeal) -> tuple[int, int, int, int, str, str]:
    return (
        0 if deal.recommendation == Recommendation.INVEST else 1,
        -deal.total_score,
        -CONFIDENCE_RANK.get(str(deal.confidence), 0),
        -deal.check_size,
        deal.company_name.casefold(),
        deal.deal_id.casefold(),
    )


def portfolio_report_order_key(deal: ScoredDeal) -> tuple[int, int, int, int, int, str, str]:
    if deal.portfolio_rank is not None:
        return (0, deal.portfolio_rank, 0, 0, 0, "", "")
    recommendation_rank, score_rank, confidence_rank, check_rank, company, deal_id = (
        portfolio_rank_key(deal)
    )
    return (
        1,
        recommendation_rank,
        score_rank,
        confidence_rank,
        check_rank,
        company,
        deal_id,
    )


def ranked_deals(scored_deals: list[ScoredDeal]) -> list[ScoredDeal]:
    return sorted(scored_deals, key=portfolio_report_order_key)


def allowed_check_tiers(config: AppConfig) -> list[int]:
    scenario = portfolio_scenario(config)
    maximum_nonzero_check = min(config.max_check, scenario.allocatable_capital)
    return [
        tier
        for tier in CHECK_SIZE_TIERS
        if tier == 0 or config.min_check <= tier <= maximum_nonzero_check
    ]


def skipped_deals(scored_deals: list[ScoredDeal]) -> list[SkippedDeal]:
    skipped: list[SkippedDeal] = []
    for index, deal in enumerate(ranked_deals(scored_deals), start=1):
        if deal.recommendation == Recommendation.INVEST:
            continue
        skipped.append(
            SkippedDeal(
                rank=index,
                company_name=deal.company_name,
                score_text=f"{deal.total_score}/{deal.max_score}",
                reason=skip_reason(deal),
            )
        )
    return skipped


def skip_reason(deal: ScoredDeal) -> str:
    triggered_gate_names = [gate.name for gate in deal.triggered_kill_gates]
    if "No usable source-linked evidence" in triggered_gate_names:
        return "No usable source-linked evidence was available."
    if "Platform minimum above maximum check" in triggered_gate_names:
        return "The platform minimum check is above the configured maximum check size."
    if "No available check size" in triggered_gate_names:
        if (deal.capital_remaining_before or 0) <= 0:
            return "No allocatable capital remained for an allowed nonzero check."
        return "No configured check size fits the platform minimum and remaining budget."
    if triggered_gate_names:
        gate_text = "; ".join(triggered_gate_names)
        return f"Triggered kill gate: {gate_text}."
    if deal.total_score < INVEST_MINIMUM_SCORE:
        return f"Score below the {INVEST_MINIMUM_SCORE}/100 INVEST threshold."
    return "Deterministic guardrails require PASS."


def portfolio_return_cases(
    *,
    invested_capital: int,
    config: AppConfig,
) -> list[PortfolioReturnCase]:
    cases = [
        ("Configured", config.gross_return_multiple),
        ("Sensitivity 1x", Decimal("1")),
        ("Sensitivity 3x", Decimal("3")),
        ("Sensitivity 10x", Decimal("10")),
    ]
    return [
        portfolio_return_case(
            label=label,
            gross_return_multiple=multiple,
            invested_capital=invested_capital,
            config=config,
        )
        for label, multiple in cases
    ]


def portfolio_return_case(
    *,
    label: str,
    gross_return_multiple: Decimal,
    invested_capital: int,
    config: AppConfig,
) -> PortfolioReturnCase:
    invested = Decimal(invested_capital)
    gross_value_before_dilution = invested * gross_return_multiple
    dilution_factor = Decimal("1") - (config.estimated_dilution_percent / PERCENT_BASE)
    value_after_dilution = gross_value_before_dilution * dilution_factor
    platform_fee = invested * config.platform_fee_percent / PERCENT_BASE
    profit_after_dilution = max(Decimal("0"), value_after_dilution - invested)
    carry = profit_after_dilution * config.carry_percent / PERCENT_BASE
    net_cash_returned = value_after_dilution - carry
    net_profit_after_fees = net_cash_returned - invested - platform_fee
    cash_in = invested + platform_fee
    net_multiple = Decimal("0") if cash_in == 0 else net_cash_returned / cash_in
    return PortfolioReturnCase(
        label=label,
        gross_return_multiple=gross_return_multiple,
        invested_capital=invested_capital,
        gross_value_before_dilution=gross_value_before_dilution,
        value_after_dilution=value_after_dilution,
        platform_fee=platform_fee,
        carry=carry,
        net_cash_returned=net_cash_returned,
        net_profit_after_fees=net_profit_after_fees,
        net_multiple=net_multiple,
    )


def _reserve_amount(config: AppConfig) -> int:
    if config.reserve_dollars > 0:
        return config.reserve_dollars
    if config.reserve_percent <= 0:
        return 0
    reserve = Decimal(config.capital_budget) * config.reserve_percent / PERCENT_BASE
    return int(reserve.to_integral_value(rounding=ROUND_CEILING))
