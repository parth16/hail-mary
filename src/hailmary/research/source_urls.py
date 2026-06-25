from __future__ import annotations

from urllib.parse import urlparse

PUBLIC_SOURCE_HOSTS: dict[str, tuple[str, str]] = {
    "sec_form_d": ("sec.gov", "an SEC website host such as www.sec.gov or data.sec.gov"),
    "sam_gov": ("sam.gov", "a SAM.gov website host such as sam.gov or www.sam.gov"),
    "usaspending": (
        "usaspending.gov",
        "a USAspending website host such as www.usaspending.gov",
    ),
    "sbir": ("sbir.gov", "an SBIR website host such as www.sbir.gov"),
    "uspto": ("uspto.gov", "a USPTO website host such as tmsearch.uspto.gov"),
    "github": ("github.com", "the GitHub website host github.com"),
}


def validate_http_url(url: str, *, field_name: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{field_name} must start with http:// or https://")
    try:
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid URL") from exc
    if not parsed.netloc or host is None:
        raise ValueError(f"{field_name} must include a website host")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} cannot include a username or password")
    if any(character.isspace() for character in url):
        raise ValueError(f"{field_name} cannot contain spaces")


def validate_provider_source_url(
    provider_id: str,
    url: str,
    *,
    field_name: str = "source_url",
) -> None:
    validate_http_url(url, field_name=field_name)
    host_rule = PUBLIC_SOURCE_HOSTS.get(provider_id)
    if host_rule is None:
        return
    allowed_suffix, description = host_rule
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    allowed_host = allowed_suffix.casefold()
    if host == allowed_host or host.endswith(f".{allowed_host}"):
        return
    raise ValueError(f"{field_name} must use {description}")
