from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel


class CompanyMatchKind(StrEnum):
    EXACT = "exact"
    LEGAL_ENTITY = "legal_entity"
    LIKELY = "likely"
    PRODUCT_NAME = "product_name"
    FOUNDER_RELATED = "founder_related"
    RELATED = "related"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class CompanyMatch(BaseModel):
    requested_name: str
    candidate_name: str
    kind: CompanyMatchKind
    reason: str
    normalized_requested: str
    normalized_candidate: str

    @property
    def import_ready(self) -> bool:
        return self.kind in {CompanyMatchKind.EXACT, CompanyMatchKind.LEGAL_ENTITY}


LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "l.l.c",
    "llc",
    "ltd",
    "limited",
    "lp",
    "llp",
    "plc",
}
PRODUCT_MARKERS = {
    "app",
    "api",
    "platform",
    "product",
    "service",
    "software",
    "suite",
    "tool",
}
FOUNDER_MARKERS = {
    "ceo",
    "cofounder",
    "founder",
    "founding",
    "president",
}


def normalize_company_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    lowered = folded.casefold()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9.]+", " ", lowered)
    tokens = [token for token in lowered.split() if token]
    return " ".join(token.rstrip(".") for token in tokens).strip()


def normalize_company_slug(value: str) -> str:
    normalized = normalize_company_name(value)
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def classify_company_match(
    requested_name: str,
    candidate_name: str,
) -> CompanyMatch:
    requested = normalize_company_name(requested_name)
    candidate = normalize_company_name(candidate_name)
    if not requested or not candidate:
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.REJECTED,
            reason="One of the company names was blank after normalization.",
            normalized_requested=requested,
            normalized_candidate=candidate,
        )
    if requested == candidate:
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.EXACT,
            reason="Normalized company names match exactly.",
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    requested_base, requested_suffix = _split_legal_suffix(requested)
    candidate_base, candidate_suffix = _split_legal_suffix(candidate)
    if requested_base and requested_base == candidate_base:
        if requested_suffix and candidate_suffix and requested_suffix != candidate_suffix:
            return CompanyMatch(
                requested_name=requested_name,
                candidate_name=candidate_name,
                kind=CompanyMatchKind.AMBIGUOUS,
                reason=(
                    "The base company name matches, but the legal-entity suffix differs. "
                    "Treat this as an ambiguous entity match until an operator validates it."
                ),
                normalized_requested=requested,
                normalized_candidate=candidate,
            )
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.LEGAL_ENTITY,
            reason=(
                "The base company name matches and the legal-entity suffix does not conflict."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    requested_compact = re.sub(r"[^a-z0-9]+", "", requested)
    candidate_compact = re.sub(r"[^a-z0-9]+", "", candidate)
    if requested_compact and requested_compact == candidate_compact:
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.LIKELY,
            reason=(
                "The names match after removing word breaks, but this still needs "
                "operator validation before import."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    requested_base_compact = re.sub(r"[^a-z0-9]+", "", requested_base)
    candidate_base_compact = re.sub(r"[^a-z0-9]+", "", candidate_base)
    if requested_base_compact and requested_base_compact == candidate_base_compact:
        if requested_suffix and candidate_suffix and requested_suffix != candidate_suffix:
            return CompanyMatch(
                requested_name=requested_name,
                candidate_name=candidate_name,
                kind=CompanyMatchKind.AMBIGUOUS,
                reason=(
                    "The base company name matches after removing word breaks, but the "
                    "legal-entity suffix differs. Operator validation is required before import."
                ),
                normalized_requested=requested,
                normalized_candidate=candidate,
            )
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.LIKELY,
            reason=(
                "The base names match after removing word breaks, but this still needs "
                "operator validation before import."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    requested_tokens = requested.split()
    candidate_tokens = candidate.split()
    if _contains_token_phrase(candidate_tokens, requested_tokens) or _contains_token_phrase(
        requested_tokens,
        candidate_tokens,
    ):
        if _looks_founder_related(requested_tokens, candidate_tokens):
            return CompanyMatch(
                requested_name=requested_name,
                candidate_name=candidate_name,
                kind=CompanyMatchKind.FOUNDER_RELATED,
                reason=(
                    "One normalized name includes the company plus founder or executive "
                    "wording. Treat this as founder-related context until the company "
                    "identity is validated."
                ),
                normalized_requested=requested,
                normalized_candidate=candidate,
            )
        if _looks_product_name(requested_tokens, candidate_tokens):
            return CompanyMatch(
                requested_name=requested_name,
                candidate_name=candidate_name,
                kind=CompanyMatchKind.PRODUCT_NAME,
                reason=(
                    "One normalized name includes the other plus product wording. Treat "
                    "this as a product-name match until an operator validates the entity."
                ),
                normalized_requested=requested,
                normalized_candidate=candidate,
            )
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.RELATED,
            reason=(
                "One normalized company name contains the other. Treat this as a "
                "related-name match until an operator validates the entity."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    overlap = set(requested_tokens) & set(candidate_tokens)
    smaller_size = min(len(set(requested_tokens)), len(set(candidate_tokens)))
    if smaller_size >= 2 and len(overlap) / smaller_size >= 0.75:
        return CompanyMatch(
            requested_name=requested_name,
            candidate_name=candidate_name,
            kind=CompanyMatchKind.AMBIGUOUS,
            reason=(
                "The names share most normalized words, but do not match exactly. "
                "Treat this as ambiguous until an operator validates the entity."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    return CompanyMatch(
        requested_name=requested_name,
        candidate_name=candidate_name,
        kind=CompanyMatchKind.REJECTED,
        reason="The normalized company names appear unrelated.",
        normalized_requested=requested,
        normalized_candidate=candidate,
    )


def best_company_match(
    requested_names: list[str],
    candidate_name: str,
) -> CompanyMatch:
    if not requested_names:
        return classify_company_match("", candidate_name)
    matches = [
        classify_company_match(requested_name, candidate_name)
        for requested_name in requested_names
    ]
    return min(matches, key=lambda match: _match_rank(match.kind))


def _contains_token_phrase(tokens: list[str], phrase: list[str]) -> bool:
    if not tokens or not phrase or len(phrase) > len(tokens):
        return False
    for index in range(0, len(tokens) - len(phrase) + 1):
        if tokens[index : index + len(phrase)] == phrase:
            return True
    return False


def _match_rank(kind: CompanyMatchKind) -> int:
    return {
        CompanyMatchKind.EXACT: 0,
        CompanyMatchKind.LEGAL_ENTITY: 1,
        CompanyMatchKind.LIKELY: 2,
        CompanyMatchKind.PRODUCT_NAME: 3,
        CompanyMatchKind.FOUNDER_RELATED: 4,
        CompanyMatchKind.RELATED: 5,
        CompanyMatchKind.AMBIGUOUS: 6,
        CompanyMatchKind.REJECTED: 7,
    }[kind]


def _split_legal_suffix(value: str) -> tuple[str, str]:
    tokens = value.split()
    suffix_tokens: list[str] = []
    while tokens and tokens[-1].rstrip(".") in LEGAL_SUFFIXES:
        suffix_tokens.insert(0, tokens.pop().rstrip("."))
    return " ".join(tokens), " ".join(suffix_tokens)


def _looks_product_name(requested_tokens: list[str], candidate_tokens: list[str]) -> bool:
    extra_tokens = _extra_phrase_tokens(requested_tokens, candidate_tokens)
    return any(token in PRODUCT_MARKERS for token in extra_tokens)


def _looks_founder_related(
    requested_tokens: list[str],
    candidate_tokens: list[str],
) -> bool:
    extra_tokens = _extra_phrase_tokens(requested_tokens, candidate_tokens)
    if any(token in FOUNDER_MARKERS for token in extra_tokens):
        return True
    if not _contains_token_phrase(candidate_tokens, requested_tokens):
        return False
    first_index = _phrase_index(candidate_tokens, requested_tokens)
    return first_index is not None and 0 < first_index <= 4


def _extra_phrase_tokens(tokens_one: list[str], tokens_two: list[str]) -> list[str]:
    if _contains_token_phrase(tokens_one, tokens_two):
        index = _phrase_index(tokens_one, tokens_two)
        if index is None:
            return []
        return [*tokens_one[:index], *tokens_one[index + len(tokens_two) :]]
    if _contains_token_phrase(tokens_two, tokens_one):
        index = _phrase_index(tokens_two, tokens_one)
        if index is None:
            return []
        return [*tokens_two[:index], *tokens_two[index + len(tokens_one) :]]
    return []


def _phrase_index(tokens: list[str], phrase: list[str]) -> int | None:
    if not tokens or not phrase or len(phrase) > len(tokens):
        return None
    for index in range(0, len(tokens) - len(phrase) + 1):
        if tokens[index : index + len(phrase)] == phrase:
            return index
    return None
