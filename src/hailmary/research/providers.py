from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote, quote_plus

from hailmary.schemas.documents import SourceKind

from .schemas import (
    ResearchAccessMode,
    ResearchProvider,
    ResearchProviderCategory,
)


@dataclass(frozen=True)
class ProviderAdapter:
    provider: ResearchProvider
    build_url: Callable[[str, str | None], str | None]


def builtin_provider_adapters(
    *,
    include_paid: bool = False,
    include_meridian: bool = False,
) -> list[ProviderAdapter]:
    adapters = [*_free_public_adapters()]
    if include_meridian:
        adapters.append(_meridian_adapter())
    if include_paid:
        adapters.extend(_paid_optional_adapters())
    return adapters


def builtin_research_providers(
    *,
    include_paid: bool = False,
    include_meridian: bool = True,
) -> list[ResearchProvider]:
    return [
        adapter.provider
        for adapter in builtin_provider_adapters(
            include_paid=include_paid,
            include_meridian=include_meridian,
        )
    ]


def _free_public_adapters() -> list[ProviderAdapter]:
    return [
        ProviderAdapter(
            provider=ResearchProvider(
                id="company_website",
                name="Company website",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Company-controlled public pages, blogs, docs, and pricing pages.",
                licensing_notes=(
                    "Use public pages for diligence notes. Save the exact URL and access time "
                    "before turning anything into evidence."
                ),
                operator_note="Use the official company site when known.",
            ),
            build_url=lambda _company_name, website_url: website_url,
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="sec_form_d",
                name="SEC EDGAR Form D search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public SEC filings that can show securities offerings.",
                licensing_notes=(
                    "Public government source. Record the filing URL and timestamp if facts "
                    "are later imported."
                ),
                operator_note=(
                    "The live collector searches SEC EDGAR for exact issuer-name matches. "
                    "Use the file workflow for related legal entity names."
                ),
            ),
            build_url=lambda company_name, _website_url: (
                "https://www.sec.gov/edgar/search/#/q="
                f"{quote(company_name)}&category=form-cat7"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="sam_gov",
                name="SAM.gov search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public U.S. government entity and opportunity search.",
                licensing_notes=(
                    "Public government source. Record the exact result URL and timestamp "
                    "before importing any fact."
                ),
                operator_note=(
                    "SAM.gov public APIs require API keys, so Hail Mary keeps this as a "
                    "manual or local-file workflow. Check exact company names, subsidiaries, "
                    "and founder entities."
                ),
            ),
            build_url=lambda company_name, _website_url: (
                "https://sam.gov/search/?index=opp&keywords="
                f"{quote_plus(company_name)}"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="usaspending",
                name="USAspending search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public U.S. federal awards, grants, loans, and contracts.",
                licensing_notes=(
                    "Public government source. Record award URLs, timestamps, and whether "
                    "the company name match is exact."
                ),
                operator_note=(
                    "Look for grants or contracts that support customer or funding claims."
                ),
            ),
            build_url=lambda company_name, _website_url: (
                "https://www.usaspending.gov/search/?keywords="
                f"{quote_plus(company_name)}"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="sbir",
                name="SBIR/STTR award search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public small-business research award search.",
                licensing_notes=(
                    "Public government source. Record the exact award page, timestamp, and "
                    "recipient match confidence."
                ),
                operator_note="Useful for deep-tech companies and government-backed research.",
            ),
            build_url=lambda company_name, _website_url: (
                "https://www.sbir.gov/award?search="
                f"{quote_plus(company_name)}"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="uspto",
                name="USPTO trademark search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public trademark records for company, product, and brand names.",
                licensing_notes=(
                    "Public government source. Record the trademark result URL and timestamp "
                    "before importing any fact."
                ),
                operator_note=(
                    "Official USPTO Open Data Portal APIs require API keys, so Hail Mary "
                    "keeps this as a manual or local-file workflow. Check important brands "
                    "or products."
                ),
            ),
            build_url=lambda company_name, _website_url: (
                "https://tmsearch.uspto.gov/search/search-results?query="
                f"{quote_plus(company_name)}"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="github",
                name="GitHub repository search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public code repositories, release activity, and developer signals.",
                licensing_notes=(
                    "Public web source. Respect repository licenses and record the exact URL "
                    "and timestamp for any later evidence."
                ),
                operator_note=(
                    "The live collector saves public repository metadata only when the "
                    "owner or repository slug exactly matches the requested company."
                ),
            ),
            build_url=lambda company_name, _website_url: (
                "https://github.com/search?q="
                f"{quote_plus(company_name)}&type=repositories"
            ),
        ),
        ProviderAdapter(
            provider=ResearchProvider(
                id="public_web",
                name="Public web and press search",
                category=ResearchProviderCategory.FREE_PUBLIC,
                access_mode=ResearchAccessMode.WEB_PAGE,
                source_kind=SourceKind.WEB,
                description="Public company pages, press, customer pages, and benchmark reports.",
                licensing_notes=(
                    "Public web sources vary by license. Record the provider, exact URL, "
                    "timestamp, and confidence before importing any fact."
                ),
                operator_note="Search manually and prefer primary sources over summaries.",
            ),
            build_url=lambda _company_name, _website_url: None,
        ),
    ]


def _meridian_adapter() -> ProviderAdapter:
    return ProviderAdapter(
        provider=ResearchProvider(
            id="meridian",
            name="Meridian deal page",
            category=ResearchProviderCategory.AUTHENTICATED_PORTAL,
            access_mode=ResearchAccessMode.MANUAL_PORTAL,
            source_kind=SourceKind.MERIDIAN,
            description="Authenticated platform deal page supplied by the operator.",
            default_enabled=False,
            requires_authenticated_session=True,
            licensing_notes=(
                "Authenticated source. Do not bypass login, CAPTCHA, two-factor checks, "
                "paywalls, or platform restrictions. Save only permitted exports locally."
            ),
            operator_note=(
                "Open the supplied URL in an authenticated browser session and copy only "
                "short allowed facts into the generated template."
            ),
        ),
        build_url=lambda _company_name, website_url: website_url,
    )


def _paid_optional_adapters() -> list[ProviderAdapter]:
    return [
        _paid_api_provider(
            provider_id="crunchbase",
            name="Crunchbase",
            description="Company profiles, funding rounds, investors, and acquisition data.",
        ),
        _paid_api_provider(
            provider_id="people_data_labs",
            name="People Data Labs",
            description="People and company enrichment data for team and employment checks.",
        ),
        _paid_api_provider(
            provider_id="newsapi",
            name="NewsAPI",
            description="Programmatic news search for press and announcement coverage.",
        ),
        _paid_api_provider(
            provider_id="similarweb",
            name="Similarweb",
            description="Web traffic estimates and digital market signals.",
        ),
        _paid_api_provider(
            provider_id="sensor_tower",
            name="Sensor Tower",
            description="Mobile app market and download estimates.",
        ),
        _paid_manual_provider(
            provider_id="pitchbook",
            name="PitchBook",
            description="Private-company financings, investors, comparables, and fund data.",
        ),
        _paid_manual_provider(
            provider_id="cb_insights",
            name="CB Insights",
            description="Private-company market maps, financings, and competitive data.",
        ),
    ]


def _paid_api_provider(
    *,
    provider_id: str,
    name: str,
    description: str,
) -> ProviderAdapter:
    return ProviderAdapter(
        provider=ResearchProvider(
            id=provider_id,
            name=name,
            category=ResearchProviderCategory.PAID_OPTIONAL,
            access_mode=ResearchAccessMode.API,
            source_kind=SourceKind.WEB,
            description=description,
            default_enabled=False,
            requires_api_key=True,
            licensing_notes=(
                "Paid optional source. Use only with an active license, keep API keys out "
                "of the repository, and store provider terms with imported facts."
            ),
            operator_note="Configure a licensed account before using this source.",
        ),
        build_url=lambda _company_name, _website_url: None,
    )


def _paid_manual_provider(
    *,
    provider_id: str,
    name: str,
    description: str,
) -> ProviderAdapter:
    return ProviderAdapter(
        provider=ResearchProvider(
            id=provider_id,
            name=name,
            category=ResearchProviderCategory.PAID_OPTIONAL,
            access_mode=ResearchAccessMode.MANUAL_PORTAL,
            source_kind=SourceKind.WEB,
            description=description,
            default_enabled=False,
            requires_authenticated_session=True,
            licensing_notes=(
                "Paid optional source. Use only with an active license and record source, "
                "timestamp, confidence, and licensing notes for every imported fact."
            ),
            operator_note="Use a licensed account and export only what the license permits.",
        ),
        build_url=lambda _company_name, _website_url: None,
    )
