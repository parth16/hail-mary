from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TextCleaningResult:
    clean_text: str
    removed_boilerplate_lines: int
    removed_boilerplate_samples: tuple[str, ...]


def clean_extracted_text(raw_text: str) -> str:
    """Normalize extracted text while preserving the original elsewhere."""

    return clean_extracted_text_with_metadata(raw_text).clean_text


def clean_extracted_text_with_metadata(raw_text: str) -> TextCleaningResult:
    """Normalize extracted text and report repeated boilerplate that was removed."""

    lines = [" ".join(line.split()) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return TextCleaningResult(
            clean_text="",
            removed_boilerplate_lines=0,
            removed_boilerplate_samples=(),
        )

    counts = Counter(lines)
    cleaned: list[str] = []
    removed_samples: list[str] = []
    removed_count = 0
    for line in lines:
        if _is_repeated_boilerplate(line, counts[line]):
            removed_count += 1
            if line not in removed_samples:
                removed_samples.append(line)
            continue
        cleaned.append(line)

    return TextCleaningResult(
        clean_text="\n".join(cleaned),
        removed_boilerplate_lines=removed_count,
        removed_boilerplate_samples=tuple(removed_samples[:5]),
    )


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
