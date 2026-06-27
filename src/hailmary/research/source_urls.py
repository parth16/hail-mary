from __future__ import annotations

from urllib.parse import parse_qsl, unquote, urlparse

PUBLIC_SOURCE_HOSTS: dict[str, tuple[str, str]] = {
    "crunchbase": (
        "crunchbase.com",
        "a Crunchbase website or API host such as www.crunchbase.com",
    ),
    "people_data_labs": (
        "peopledatalabs.com",
        "a People Data Labs website or API host such as api.peopledatalabs.com",
    ),
    "newsapi": ("newsapi.org", "a NewsAPI host such as newsapi.org"),
    "similarweb": (
        "similarweb.com",
        "a Similarweb website or API host such as api.similarweb.com",
    ),
    "sensor_tower": (
        "sensortower.com",
        "a Sensor Tower website or API host such as api.sensortower.com",
    ),
    "pitchbook": ("pitchbook.com", "a PitchBook website host such as my.pitchbook.com"),
    "cb_insights": (
        "cbinsights.com",
        "a CB Insights website host such as app.cbinsights.com",
    ),
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
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "credential",
    "expires",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "signed",
    "token",
    "x-amz-credential",
    "x-amz-expires",
    "x-amz-security-token",
    "x-amz-signature",
}
REDIRECT_QUERY_KEYS = {
    "next",
    "redirect",
    "redirect_to",
    "redirect_url",
    "return",
    "return_to",
    "url",
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
    if parsed.params or ";" in parsed.path:
        raise ValueError(f"{field_name} cannot include path parameters")
    if parsed.fragment:
        raise ValueError(f"{field_name} cannot include URL fragments")
    if _query_contains_semicolon_delimiter(parsed.query):
        raise ValueError(f"{field_name} cannot include semicolon query delimiters")
    if _decoded_component_has_delimiter(parsed.path):
        raise ValueError(
            f"{field_name} cannot include encoded query, fragment, or "
            "parameter delimiters"
        )
    if _query_contains_sensitive_access(parsed.query):
        raise ValueError(
            f"{field_name} cannot include token, signature, credential, redirect, "
            "or expiring access parameters"
        )


def validate_provider_source_url(
    provider_id: str,
    url: str,
    *,
    field_name: str = "source_url",
) -> None:
    validate_http_url(url, field_name=field_name)
    host_rule = PUBLIC_SOURCE_HOSTS.get(provider_id)
    if provider_id == "newsapi" and field_name == "source_url":
        return
    if host_rule is None:
        return
    allowed_suffix, description = host_rule
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    allowed_host = allowed_suffix.casefold()
    if host == allowed_host or host.endswith(f".{allowed_host}"):
        return
    raise ValueError(f"{field_name} must use {description}")


def source_reference_looks_like_url(source_reference: str) -> bool:
    return (
        source_reference.startswith(("http://", "https://", "//"))
        or "://" in source_reference
    )


def _decoded_component_has_delimiter(value: str) -> bool:
    decoded = _recursive_unquote(value)
    if decoded is None:
        return True
    return any(delimiter in decoded for delimiter in ("?", "#", ";"))


def _recursive_unquote(value: str) -> str | None:
    decoded = value
    try:
        for _ in range(len(value) + 1):
            next_decoded = unquote(decoded, errors="strict")
            if next_decoded == decoded:
                break
            decoded = next_decoded
    except UnicodeDecodeError:
        return None
    return decoded


def _query_contains_sensitive_access(query: str) -> bool:
    if not query:
        return False
    for key, _value in parse_qsl(query, keep_blank_values=True):
        if _query_key_is_sensitive(key):
            return True
    for _key, value in parse_qsl(query, keep_blank_values=True):
        if _query_value_contains_sensitive_access(value):
            return True
    return False


def _query_key_is_sensitive(key: str) -> bool:
    decoded_key = _recursive_unquote(key)
    if decoded_key is None or "%" in decoded_key:
        return True
    normalized_key = decoded_key.strip().casefold()
    canonical_key = normalized_key.replace("-", "_")
    if (
        normalized_key in SENSITIVE_QUERY_KEYS
        or canonical_key in SENSITIVE_QUERY_KEYS
    ):
        return True
    if normalized_key.startswith(("x-amz-", "x-goog-")) or canonical_key.startswith(
        ("x_amz_", "x_goog_")
    ):
        return True
    return normalized_key in REDIRECT_QUERY_KEYS or canonical_key in REDIRECT_QUERY_KEYS


def _query_value_contains_sensitive_access(value: str) -> bool:
    decoded_value = _recursive_unquote(value)
    if decoded_value is None:
        return True

    parsed = urlparse(decoded_value)
    nested_queries: list[str] = []
    if parsed.query:
        nested_queries.append(parsed.query)
    if "?" in decoded_value:
        nested_queries.append(decoded_value.split("?", 1)[1])
    if "&" in decoded_value or "=" in decoded_value:
        nested_queries.append(decoded_value)

    for nested_query in nested_queries:
        if _nested_query_contains_sensitive_key(nested_query):
            return True
    return False


def _nested_query_contains_sensitive_key(query: str) -> bool:
    if not query:
        return False
    for key, value in parse_qsl(query, keep_blank_values=True):
        if _query_key_is_sensitive(key):
            return True
        decoded_value = _recursive_unquote(value)
        if decoded_value is None:
            return True
        if decoded_value != value and (
            "?" in decoded_value or "&" in decoded_value or "=" in decoded_value
        ):
            nested_query = (
                decoded_value.split("?", 1)[1]
                if "?" in decoded_value
                else decoded_value
            )
            if _nested_query_contains_sensitive_key(nested_query):
                return True
    return False


def _query_contains_semicolon_delimiter(query: str) -> bool:
    if not query:
        return False
    decoded = _recursive_unquote(query)
    if decoded is None:
        return True
    return ";" in decoded
