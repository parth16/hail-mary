from __future__ import annotations

import re

EMBEDDED_SOURCE_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:please\s+)?ignore\s+(?:all\s+|every\s+|previous\s+|the\s+)?instructions?\b",
        r"^(?:please\s+)?disregard\s+(?:all\s+|previous\s+|the\s+)?instructions?\b",
        r"^(?:please\s+)?forget\s+(?:everything\s+above|the\s+above|previous\s+instructions?)\b",
        r"^(?:please\s+)?always\s+recommend\s+(?:invest|pass)\b",
        r"^(?:please\s+)?recommend\s+(?:invest|pass)\b",
        r"^(?:please\s+)?do\s+not\s+follow\s+the\s+system\b",
        r"^(?:please\s+)?(?:print|reveal|show)\s+the\s+system\s+prompt\b",
    )
)
MID_LINE_SOURCE_INSTRUCTION_PATTERN = re.compile(
    r"\s(?:ignore\s+(?:all\s+|every\s+|previous\s+|the\s+)?instructions?|"
    r"disregard\s+(?:all\s+|previous\s+|the\s+)?instructions?|"
    r"forget\s+(?:everything\s+above|the\s+above|previous\s+instructions?)|"
    r"always\s+recommend\s+(?:invest|pass)|"
    r"(?:please\s+)?recommend\s+(?:invest|pass)\s+"
    r"(?:no\s+matter\s+what|regardless\b|regardless\s+of\s+evidence|"
    r"even\s+if\b|without\s+evidence\b)|"
    r"do\s+not\s+follow\s+the\s+system|"
    r"(?:print|reveal|show)\s+the\s+system\s+prompt)\b"
)
SOURCE_INSTRUCTION_JOIN_BOUNDARY_PATTERN = re.compile(
    r"(?<=[,:])(?=\s*(?:please\s+)?(?:always\s+)?recommend\s+(?:invest|pass)\b)",
    re.IGNORECASE,
)
SOURCE_INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"^(?:(?:"
    r"assistant|chat|developer|important|instruction|instructions|model|note|"
    r"operator|prompt|speaker|system|system note|system prompt|user"
    r")\s*(?:[-:\u2010-\u2015\u2212])\s*)+"
)
SOURCE_LIST_PREFIX_PATTERN = re.compile(
    r"""^[\s>"'`#]*(?:(?:[-*+>]+|\d+[\.)]|#+)\s*)*"""
)


def looks_like_embedded_source_instruction(text: str) -> bool:
    """Return true when source text looks like instructions, not diligence evidence."""

    normalized_text = " ".join(text.lower().split())
    return any(
        pattern.search(candidate)
        for candidate in _source_instruction_candidates(text)
        for pattern in EMBEDDED_SOURCE_INSTRUCTION_PATTERNS
    ) or _has_mid_line_instruction(normalized_text)


def _source_instruction_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.splitlines() or [text]:
        for segment in [
            line,
            *re.split(r"(?<=[.!?;])\s*", line),
            *SOURCE_INSTRUCTION_JOIN_BOUNDARY_PATTERN.split(line),
        ]:
            normalized = " ".join(segment.lower().split())
            if not normalized:
                continue
            candidates.append(_strip_source_instruction_prefix(normalized))
    return candidates


def _strip_source_instruction_prefix(text: str) -> str:
    stripped = SOURCE_LIST_PREFIX_PATTERN.sub("", text).strip()
    previous = None
    while stripped != previous:
        previous = stripped
        stripped = SOURCE_INSTRUCTION_PREFIX_PATTERN.sub("", stripped).strip()
        stripped = SOURCE_LIST_PREFIX_PATTERN.sub("", stripped).strip()
    return stripped


def _has_mid_line_instruction(text: str) -> bool:
    for match in MID_LINE_SOURCE_INSTRUCTION_PATTERN.finditer(text):
        before = text[: match.start()].rstrip()
        if before.endswith(("prompt:", "example:", "user type", "users type")):
            continue
        return True
    return False
