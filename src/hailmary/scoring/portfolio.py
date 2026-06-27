from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext

from hailmary.config import CHECK_SIZE_TIERS, AppConfig
from hailmary.portfolio.ledger import PortfolioLedger
from hailmary.portfolio.scenario import (
    PERCENT_BASE,
    allowed_check_tiers_for_available_capital,
    portfolio_scenario,
)
from hailmary.schemas.evidence import EvidenceRecord, EvidenceStore
from hailmary.schemas.scoring import (
    CompanyStage,
    ConfidenceLevel,
    PortfolioExposureCheck,
    PortfolioExposureDimension,
    Recommendation,
    ScoredDeal,
)

CONFIDENCE_RANK = {
    "high": 3,
    "medium": 2,
    "low": 1,
}
INVEST_MINIMUM_SCORE = 75
CATEGORY_LABEL_PATTERN = re.compile(
    r"\b(?:sector|category|industry)\s*(?:is|:|-)\s*"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9 &/+.-]{0,60})",
    re.IGNORECASE,
)
CATEGORY_STOP_PATTERN = re.compile(r"[\n\r.;,|#<>\[\]{}()]")


@dataclass(frozen=True)
class PortfolioExposureState:
    company: dict[str, int]
    category: dict[str, int]
    stage: dict[str, int]
    source_confidence: dict[str, int]

    def exposure_for(self, dimension: PortfolioExposureDimension, key: str) -> int:
        bucket = self._bucket(dimension)
        exposure = bucket.get(key, 0)
        if dimension != PortfolioExposureDimension.COMPANY and key != "unknown":
            exposure += bucket.get("unknown", 0)
        return exposure

    def with_exposure(
        self,
        *,
        dimension: PortfolioExposureDimension,
        key: str,
        amount: int,
    ) -> PortfolioExposureState:
        if amount <= 0:
            return self
        updates = {
            PortfolioExposureDimension.COMPANY: dict(self.company),
            PortfolioExposureDimension.CATEGORY: dict(self.category),
            PortfolioExposureDimension.STAGE: dict(self.stage),
            PortfolioExposureDimension.SOURCE_CONFIDENCE: dict(self.source_confidence),
        }
        bucket = updates[dimension]
        bucket[key] = bucket.get(key, 0) + amount
        return PortfolioExposureState(
            company=updates[PortfolioExposureDimension.COMPANY],
            category=updates[PortfolioExposureDimension.CATEGORY],
            stage=updates[PortfolioExposureDimension.STAGE],
            source_confidence=updates[PortfolioExposureDimension.SOURCE_CONFIDENCE],
        )

    def _bucket(self, dimension: PortfolioExposureDimension) -> dict[str, int]:
        if dimension == PortfolioExposureDimension.COMPANY:
            return self.company
        if dimension == PortfolioExposureDimension.CATEGORY:
            return self.category
        if dimension == PortfolioExposureDimension.STAGE:
            return self.stage
        return self.source_confidence


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


def empty_portfolio_exposure_state() -> PortfolioExposureState:
    return PortfolioExposureState(
        company={},
        category={},
        stage={},
        source_confidence={},
    )


def portfolio_exposure_state_from_ledger(
    ledger: PortfolioLedger,
    *,
    config: AppConfig | None = None,
) -> PortfolioExposureState:
    state = empty_portfolio_exposure_state()
    for investment in ledger.investments:
        state = state.with_exposure(
            dimension=PortfolioExposureDimension.COMPANY,
            key=portfolio_company_key(investment.company_name),
            amount=investment.amount,
        )
    if config is None or ledger.invested_amount <= 0:
        return state
    for dimension, limit_percent in _ledger_unknown_exposure_limits(config):
        if limit_percent <= 0:
            continue
        state = state.with_exposure(
            dimension=dimension,
            key="unknown",
            amount=ledger.invested_amount,
        )
    return state


def portfolio_exposure_state_after_score(
    state: PortfolioExposureState,
    scored_deal: ScoredDeal,
) -> PortfolioExposureState:
    updated_state = state
    if scored_deal.check_size <= 0:
        return updated_state
    for check in scored_deal.check_sizing.exposure_checks:
        if not check.applied:
            continue
        updated_state = updated_state.with_exposure(
            dimension=check.dimension,
            key=check.key,
            amount=scored_deal.check_size,
        )
    return updated_state


def portfolio_exposure_checks(
    *,
    config: AppConfig,
    store: EvidenceStore,
    company_stage: CompanyStage,
    confidence: ConfidenceLevel,
    exposure_state: PortfolioExposureState | None,
) -> list[PortfolioExposureCheck]:
    state = exposure_state or empty_portfolio_exposure_state()
    category_key = portfolio_category_key_with_evidence(store.evidence)
    checks: list[PortfolioExposureCheck] = []
    checks.extend(
        _exposure_check(
            config=config,
            state=state,
            dimension=PortfolioExposureDimension.COMPANY,
            key=portfolio_company_key(store.company_name),
            limit_percent=config.max_company_exposure_percent,
            reason_code="company_exposure",
        )
    )
    checks.extend(
        _exposure_check(
            config=config,
            state=state,
            dimension=PortfolioExposureDimension.CATEGORY,
            key=category_key.key,
            limit_percent=config.max_category_exposure_percent,
            reason_code="category_exposure",
            evidence_ids=category_key.evidence_ids,
        )
    )
    checks.extend(
        _exposure_check(
            config=config,
            state=state,
            dimension=PortfolioExposureDimension.STAGE,
            key=(
                None
                if company_stage == CompanyStage.UNKNOWN
                else str(company_stage.value)
            ),
            limit_percent=config.max_stage_exposure_percent,
            reason_code="stage_exposure",
        )
    )
    checks.extend(
        _exposure_check(
            config=config,
            state=state,
            dimension=PortfolioExposureDimension.SOURCE_CONFIDENCE,
            key=str(confidence.value),
            limit_percent=_confidence_exposure_percent(config, confidence),
            reason_code="source_confidence_exposure",
        )
    )
    return checks


def portfolio_exposure_cap(checks: list[PortfolioExposureCheck]) -> int | None:
    capacities = [check.available_capacity for check in checks if check.applied]
    if not capacities:
        return None
    return min(capacities)


def portfolio_company_key(company_name: str) -> str:
    return " ".join(company_name.split()).casefold()


def portfolio_category_key(evidence: list[EvidenceRecord]) -> str | None:
    return portfolio_category_key_with_evidence(evidence).key


@dataclass(frozen=True)
class PortfolioExposureKey:
    key: str | None
    evidence_ids: list[str]


def portfolio_category_key_with_evidence(
    evidence: list[EvidenceRecord],
) -> PortfolioExposureKey:
    for record in evidence:
        match = CATEGORY_LABEL_PATTERN.search(record.text)
        if match is None:
            continue
        raw_value = CATEGORY_STOP_PATTERN.split(match.group("value"), maxsplit=1)[0]
        cleaned = " ".join(raw_value.split()).strip(" -:/")
        if not cleaned:
            continue
        if len(cleaned) > 50:
            cleaned = cleaned[:50].rstrip()
        return PortfolioExposureKey(key=cleaned.casefold(), evidence_ids=[record.id])
    return PortfolioExposureKey(key=None, evidence_ids=[])


def _exposure_check(
    *,
    config: AppConfig,
    state: PortfolioExposureState,
    dimension: PortfolioExposureDimension,
    key: str | None,
    limit_percent: Decimal,
    reason_code: str,
    evidence_ids: list[str] | None = None,
) -> list[PortfolioExposureCheck]:
    if limit_percent <= 0:
        return []
    limit_amount = _exposure_limit_amount(config, limit_percent)
    evidence_ids = evidence_ids or []
    if key is None:
        exposure_before = state.exposure_for(dimension, "unknown")
        available_capacity = max(0, limit_amount - exposure_before)
        minimum_nonzero_check = _minimum_nonzero_check(config)
        return [
            PortfolioExposureCheck(
                dimension=dimension,
                key="unknown",
                configured_limit_percent=limit_percent,
                exposure_before=exposure_before,
                limit_amount=limit_amount,
                available_capacity=available_capacity,
                applied=True,
                blocking=available_capacity < minimum_nonzero_check,
                reason_code=f"{reason_code}_missing_key",
                evidence_ids=evidence_ids,
            )
        ]
    exposure_before = state.exposure_for(dimension, key)
    available_capacity = max(0, limit_amount - exposure_before)
    minimum_nonzero_check = _minimum_nonzero_check(config)
    return [
        PortfolioExposureCheck(
            dimension=dimension,
            key=key,
            configured_limit_percent=limit_percent,
            exposure_before=exposure_before,
            limit_amount=limit_amount,
            available_capacity=available_capacity,
            applied=True,
            blocking=available_capacity < minimum_nonzero_check,
            reason_code=reason_code,
            evidence_ids=evidence_ids,
        )
    ]


def _exposure_limit_amount(config: AppConfig, limit_percent: Decimal) -> int:
    scenario = portfolio_scenario(config)
    basis = Decimal(scenario.allocatable_capital)
    with localcontext() as context:
        context.prec = max(
            context.prec,
            len(basis.as_tuple().digits)
            + len(limit_percent.as_tuple().digits)
            + len(PERCENT_BASE.as_tuple().digits)
            + 4,
        )
        limit = basis * limit_percent / PERCENT_BASE
    return int(limit.to_integral_value(rounding=ROUND_FLOOR))


def _confidence_exposure_percent(
    config: AppConfig,
    confidence: ConfidenceLevel,
) -> Decimal:
    if confidence == ConfidenceLevel.LOW:
        return config.max_low_confidence_exposure_percent
    if confidence == ConfidenceLevel.MEDIUM:
        return config.max_medium_confidence_exposure_percent
    return config.max_high_confidence_exposure_percent


def _minimum_nonzero_check(config: AppConfig) -> int:
    nonzero_tiers = [tier for tier in CHECK_SIZE_TIERS if tier > 0]
    configured_tiers = [tier for tier in nonzero_tiers if tier >= config.min_check]
    if configured_tiers:
        return min(configured_tiers)
    return min(nonzero_tiers)


def _ledger_unknown_exposure_limits(
    config: AppConfig,
) -> list[tuple[PortfolioExposureDimension, Decimal]]:
    confidence_limit = max(
        config.max_low_confidence_exposure_percent,
        config.max_medium_confidence_exposure_percent,
        config.max_high_confidence_exposure_percent,
    )
    return [
        (PortfolioExposureDimension.CATEGORY, config.max_category_exposure_percent),
        (PortfolioExposureDimension.STAGE, config.max_stage_exposure_percent),
        (PortfolioExposureDimension.SOURCE_CONFIDENCE, confidence_limit),
    ]


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
    return allowed_check_tiers_for_available_capital(
        config,
        available_capital=scenario.allocatable_capital,
    )


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
    if deal.total_score < INVEST_MINIMUM_SCORE:
        return f"Score below the {INVEST_MINIMUM_SCORE}/100 INVEST threshold."
    if "No available check size" in triggered_gate_names:
        reason_codes = set(deal.check_sizing.reason_codes)
        if "exposure_cap_below_minimum" in reason_codes:
            return "Exposure limits left no room for an allowed nonzero check."
        if "exposure_limit_no_tier" in reason_codes:
            return "No configured check size fits the portfolio exposure limits."
        if (deal.capital_remaining_before or 0) <= 0:
            return "No allocatable capital remained for an allowed nonzero check."
        return "No configured check size fits the platform minimum and remaining budget."
    if triggered_gate_names:
        gate_text = "; ".join(triggered_gate_names)
        return f"Triggered kill gate: {gate_text}."
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
    with localcontext() as context:
        max_result_adjusted = (
            max(invested.adjusted(), 0) + max(gross_return_multiple.adjusted(), 0) + 10
        )
        context.Emax = max(context.Emax, max_result_adjusted)
        context.Emin = min(context.Emin, -max_result_adjusted)
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
