from __future__ import annotations

from collections import Counter


def clean_extracted_text(raw_text: str) -> str:
    """Normalize extracted text while preserving the original elsewhere."""

    lines = [" ".join(line.split()) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    counts = Counter(lines)
    cleaned: list[str] = []
    for line in lines:
        if _is_repeated_boilerplate(line, counts[line]):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _is_repeated_boilerplate(line: str, count: int) -> bool:
    if count < 3:
        return False

    lowered = line.lower()
    boilerplate_markers = [
        "confidential:",
        "disclosing deal information",
        "not for distribution",
    ]
    return any(marker in lowered for marker in boilerplate_markers)
