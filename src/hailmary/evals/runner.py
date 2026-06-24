from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hailmary.evals import fixtures
from hailmary.evals.schemas import (
    EvalCaseMetadata,
    EvalCaseResult,
    EvalCategory,
    EvalRunSummary,
)


class EvalHarnessError(RuntimeError):
    """The local eval harness could not run safely."""


@dataclass(frozen=True)
class EvalDefinition:
    metadata: EvalCaseMetadata
    run: Callable[[Path], bool]


def builtin_eval_cases() -> list[EvalCaseMetadata]:
    return [definition.metadata for definition in _eval_definitions()]


def run_builtin_evals(
    *,
    case_ids: Sequence[str] | None = None,
    categories: Sequence[EvalCategory] | None = None,
    work_dir: Path | None = None,
) -> EvalRunSummary:
    selected = _select_eval_definitions(
        _eval_definitions(),
        case_ids=set(case_ids or []),
        categories=set(categories or []),
    )
    if not selected:
        raise EvalHarnessError(
            "No evals matched the requested filters. Run without filters to see "
            "whether the built-in synthetic evals pass."
        )

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        results = [_run_one(definition, work_dir) for definition in selected]
        return EvalRunSummary(results=results)

    with tempfile.TemporaryDirectory(prefix="hailmary-evals-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        results = [_run_one(definition, temp_dir) for definition in selected]
    return EvalRunSummary(results=results)


def _eval_definitions() -> list[EvalDefinition]:
    return [
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="extraction-text-ingestion",
                category=EvalCategory.EXTRACTION,
                name="Text extraction and ingestion",
                description=(
                    "Builds a synthetic local deal folder and checks that text extraction, "
                    "evidence creation, and deal-term claim extraction work together."
                ),
            ),
            run=fixtures.run_extraction_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="citation-span-mismatch",
                category=EvalCategory.CITATION,
                name="Citation span mismatch rejection",
                description=(
                    "Checks that a claim with a stale or mismatched source span is not "
                    "treated as verified evidence."
                ),
            ),
            run=lambda _: fixtures.run_citation_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="contradiction-valid-conflict",
                category=EvalCategory.CONTRADICTION,
                name="Conflicting deal-term gate",
                description=(
                    "Checks that two valid but conflicting valuation claims trigger a "
                    "PASS decision and a conflict warning."
                ),
            ),
            run=lambda _: fixtures.run_contradiction_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="prompt-injection-recommendation",
                category=EvalCategory.PROMPT_INJECTION,
                name="Prompt-injection recommendation rejection",
                description=(
                    "Checks that source text is marked as untrusted and an uncited final "
                    "recommendation is rejected."
                ),
            ),
            run=lambda _: fixtures.run_prompt_injection_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="score-strong-invest",
                category=EvalCategory.SCORE_CALIBRATION,
                name="Strong deal score calibration",
                description=(
                    "Checks that strong synthetic terms, traction, and funding evidence "
                    "produce INVEST with an allowed nonzero check size."
                ),
            ),
            run=lambda _: fixtures.run_strong_score_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="score-borderline-pass",
                category=EvalCategory.SCORE_CALIBRATION,
                name="Borderline deal score calibration",
                description=(
                    "Checks that a synthetic 65-74 score stays PASS with a $0 check."
                ),
            ),
            run=lambda _: fixtures.run_borderline_score_fixture(),
        ),
    ]


def _select_eval_definitions(
    definitions: list[EvalDefinition],
    *,
    case_ids: set[str],
    categories: set[EvalCategory],
) -> list[EvalDefinition]:
    selected = definitions
    if case_ids:
        selected = [
            definition
            for definition in selected
            if definition.metadata.id in case_ids
        ]
    if categories:
        selected = [
            definition
            for definition in selected
            if definition.metadata.category in categories
        ]
    return selected


def _run_one(definition: EvalDefinition, work_dir: Path) -> EvalCaseResult:
    metadata = definition.metadata
    case_work_dir = work_dir / metadata.id
    case_work_dir.mkdir(parents=True, exist_ok=True)
    try:
        passed = definition.run(case_work_dir)
    except Exception as exc:
        return EvalCaseResult(
            id=metadata.id,
            category=metadata.category,
            name=metadata.name,
            passed=False,
            message=f"The eval crashed before it could finish: {exc}",
            details={"description": metadata.description},
        )

    return EvalCaseResult(
        id=metadata.id,
        category=metadata.category,
        name=metadata.name,
        passed=passed,
        message="Passed." if passed else "The expected behavior did not happen.",
        details={"description": metadata.description},
    )
