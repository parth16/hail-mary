from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel


class CompanyMatchKind(StrEnum):
    EXACT = "exact"
    LIKELY = "likely"
    RELATED = "related"
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
        return self.kind == CompanyMatchKind.EXACT


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


def normalize_company_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    lowered = folded.casefold()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9.]+", " ", lowered)
    tokens = [token for token in lowered.split() if token]
    while tokens and tokens[-1].rstrip(".") in LEGAL_SUFFIXES:
        tokens.pop()
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

    requested_tokens = requested.split()
    candidate_tokens = candidate.split()
    if _contains_token_phrase(candidate_tokens, requested_tokens) or _contains_token_phrase(
        requested_tokens,
        candidate_tokens,
    ):
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
            kind=CompanyMatchKind.RELATED,
            reason=(
                "The names share most normalized words. Treat this as a related-name "
                "match until an operator validates the entity."
            ),
            normalized_requested=requested,
            normalized_candidate=candidate,
        )

    return CompanyMatch(
        requested_name=requested_name,
        candidate_name=candidate_name,
        kind=CompanyMatchKind.REJECTED,
        reason="The normalized company names do not match closely enough.",
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
        CompanyMatchKind.LIKELY: 1,
        CompanyMatchKind.RELATED: 2,
        CompanyMatchKind.REJECTED: 3,
    }[kind]
