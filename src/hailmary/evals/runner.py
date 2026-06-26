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
    run: Callable[[Path], None]


def builtin_eval_cases() -> list[EvalCaseMetadata]:
    return [definition.metadata for definition in _eval_definitions()]


def run_builtin_evals(
    *,
    case_ids: Sequence[str] | None = None,
    categories: Sequence[EvalCategory] | None = None,
    work_dir: Path | None = None,
) -> EvalRunSummary:
    definitions = _eval_definitions()
    _reject_unknown_case_ids(case_ids or [], definitions)
    selected = _select_eval_definitions(
        definitions,
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
                id="extraction-ocr-low-text-documents",
                category=EvalCategory.EXTRACTION,
                name="OCR and low-text PDF extraction",
                description=(
                    "Checks that synthetic PDFs distinguish divider pages, repeated "
                    "low-text pages, empty pages, OCR flags, vision flags, and source spans."
                ),
            ),
            run=fixtures.run_ocr_low_text_documents_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="extraction-table-edge-cases",
                category=EvalCategory.EXTRACTION,
                name="Table extraction edge cases",
                description=(
                    "Checks nested HTML tables, row and column spans, sparse XLSX gaps, "
                    "and empty far-right spreadsheet cells."
                ),
            ),
            run=fixtures.run_table_edge_cases_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="ocr-image-unavailable",
                category=EvalCategory.OCR,
                name="Image-only OCR unavailable path",
                description=(
                    "Checks that image-only synthetic documents stay evidence-less "
                    "with explicit OCR and vision-needed metadata when OCR is unavailable."
                ),
            ),
            run=fixtures.run_ocr_image_unavailable_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="ocr-fake-success-source-linked",
                category=EvalCategory.OCR,
                name="Fake OCR success source lineage",
                description=(
                    "Checks that fake local OCR creates source-linked evidence with "
                    "OCR lineage and claim extraction from synthetic image text."
                ),
            ),
            run=fixtures.run_ocr_fake_success_source_linkage_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="ocr-prompt-injection-untrusted",
                category=EvalCategory.OCR,
                name="OCR prompt injection remains untrusted",
                description=(
                    "Checks that prompt-injection text returned by fake OCR remains "
                    "untrusted source evidence and cannot support agent recommendations."
                ),
            ),
            run=fixtures.run_ocr_prompt_injection_untrusted_fixture,
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
                id="citation-packet-quote-preservation",
                category=EvalCategory.CITATION,
                name="Packet truncation preserves cited quotes",
                description=(
                    "Checks that agent packet truncation preserves selected claim quotes "
                    "instead of blindly keeping leading text."
                ),
            ),
            run=lambda _: fixtures.run_packet_quote_preservation_fixture(),
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
                id="contradiction-stale-conflict-cleanup",
                category=EvalCategory.CONTRADICTION,
                name="Stale conflict citation cleanup",
                description=(
                    "Checks that stale conflict citations do not trigger the conflict "
                    "kill gate or displace still-valid score-factor evidence."
                ),
            ),
            run=lambda _: fixtures.run_stale_conflict_cleanup_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="research-public-source-import",
                category=EvalCategory.RESEARCH_IMPORT,
                name="Public-source research import lineage",
                description=(
                    "Checks local public-source result preparation, exact-match import, "
                    "source lineage, refreshed claims, and packet metadata minimization."
                ),
            ),
            run=fixtures.run_public_source_import_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="research-usaspending-api-pagination",
                category=EvalCategory.RESEARCH_IMPORT,
                name="USAspending pagination and page-cap handling",
                description=(
                    "Checks fake USAspending pagination, exact-match retention, page-cap "
                    "warnings, and multi-company result preservation without network calls."
                ),
            ),
            run=fixtures.run_usaspending_pagination_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="research-free-public-collectors-v2",
                category=EvalCategory.RESEARCH_IMPORT,
                name="Free public collector exact matching",
                description=(
                    "Checks fake SEC Form D and GitHub collectors, exact-match "
                    "filtering, source lineage, and import dry-run compatibility."
                ),
            ),
            run=fixtures.run_free_public_collectors_v2_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="research-workflow-v2",
                category=EvalCategory.RESEARCH_IMPORT,
                name="Research workflow V2 loop",
                description=(
                    "Checks workflow artifact generation, local public-source exact "
                    "matching, related-entity skip reporting, no-result summaries, "
                    "and import dry-run behavior."
                ),
            ),
            run=fixtures.run_research_workflow_v2_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="public-collectors-source-guards",
                category=EvalCategory.PUBLIC_COLLECTORS,
                name="Public collector exact-match and source guards",
                description=(
                    "Checks exact company matching, related-entity skips, bad provider "
                    "URL rejection, and no-result summaries naming the company."
                ),
            ),
            run=fixtures.run_public_collectors_source_guards_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="meridian-workflow-guards",
                category=EvalCategory.MERIDIAN,
                name="Meridian workflow guardrails",
                description=(
                    "Checks unsafe Meridian URL rejection, placeholder row protections, "
                    "and marker stripping before saved evidence."
                ),
            ),
            run=fixtures.run_meridian_workflow_guards_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="prompt-injection-recommendation",
                category=EvalCategory.PROMPT_INJECTION,
                name="HTML prompt-injection recommendation rejection",
                description=(
                    "Checks that HTML source text is marked as untrusted and a final "
                    "recommendation that cites an embedded instruction is rejected."
                ),
            ),
            run=fixtures.run_prompt_injection_html_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="prompt-injection-pdf-recommendation",
                category=EvalCategory.PROMPT_INJECTION,
                name="PDF prompt-injection recommendation rejection",
                description=(
                    "Checks that PDF source text is marked as untrusted and a final "
                    "recommendation that cites an embedded instruction is rejected."
                ),
            ),
            run=fixtures.run_prompt_injection_pdf_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="prompt-injection-docx-recommendation",
                category=EvalCategory.PROMPT_INJECTION,
                name="DOCX prompt-injection recommendation rejection",
                description=(
                    "Checks that DOCX source text is marked as untrusted and a final "
                    "recommendation that cites an embedded instruction is rejected."
                ),
            ),
            run=fixtures.run_prompt_injection_docx_fixture,
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="prompt-injection-boundaries",
                category=EvalCategory.PROMPT_INJECTION,
                name="Prompt-injection boundary detection",
                description=(
                    "Checks prefixed, punctuation-joined, and mid-line source "
                    "instructions while allowing benign prompt-product examples."
                ),
            ),
            run=lambda _: fixtures.run_prompt_injection_boundaries_fixture(),
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
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="score-stage-aware-v2",
                category=EvalCategory.SCORE_CALIBRATION,
                name="Stage-aware v2 score calibration",
                description=(
                    "Checks that the same early PMF evidence scores differently by "
                    "company stage and cites the supporting synthetic evidence."
                ),
            ),
            run=lambda _: fixtures.run_stage_aware_score_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="score-return-math-missing-v2",
                category=EvalCategory.SCORE_CALIBRATION,
                name="Missing return-math input handling",
                description=(
                    "Checks that v2 net-return math reports missing dilution, fees or "
                    "carry, and exit assumptions without inventing values."
                ),
            ),
            run=lambda _: fixtures.run_return_math_missing_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="score-calibration-guards",
                category=EvalCategory.SCORE_CALIBRATION,
                name="Scoring calibration guardrails",
                description=(
                    "Checks missing key terms, negated traction and funding language, "
                    "and stage and return-math packet instructions for missing inputs."
                ),
            ),
            run=lambda _: fixtures.run_score_calibration_guards_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="missing-data-pass",
                category=EvalCategory.MISSING_DATA,
                name="Missing data pass gate",
                description=(
                    "Checks that missing source-linked evidence produces PASS, a $0 "
                    "check, a no-evidence kill gate, and diligence questions."
                ),
            ),
            run=lambda _: fixtures.run_missing_data_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="memo-required-sections",
                category=EvalCategory.MEMO_SNAPSHOT,
                name="Memo required section snapshot",
                description=(
                    "Checks that Markdown memo and portfolio report rendering keep "
                    "decision, score, evidence, diligence, and disclaimer sections."
                ),
            ),
            run=lambda _: fixtures.run_memo_snapshot_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="memo-v2-score-evidence",
                category=EvalCategory.MEMO_SNAPSHOT,
                name="Memo v2 score impact evidence",
                description=(
                    "Checks that memos render v2 stage, valuation, return math, support "
                    "status, and evidence IDs for score impacts."
                ),
            ),
            run=lambda _: fixtures.run_memo_v2_score_evidence_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="memo-cited-conflict-evidence",
                category=EvalCategory.MEMO_SNAPSHOT,
                name="Memo includes cited and valid conflict evidence",
                description=(
                    "Checks that memos include required sections, cited evidence beyond "
                    "the first 25 records, and valid conflict evidence while excluding "
                    "stale conflict evidence."
                ),
            ),
            run=lambda _: fixtures.run_memo_cited_conflict_evidence_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="memo-output-guards",
                category=EvalCategory.MEMO_SNAPSHOT,
                name="Memo and portfolio output guardrails",
                description=(
                    "Checks final memo decision ordering, cited evidence beyond the "
                    "first 25 records, and Markdown escaping for untrusted text."
                ),
            ),
            run=lambda _: fixtures.run_memo_output_guards_fixture(),
        ),
        EvalDefinition(
            metadata=EvalCaseMetadata(
                id="privacy-output-guards",
                category=EvalCategory.PRIVACY,
                name="Private output and ignored-folder guardrails",
                description=(
                    "Checks generated folder permissions, ignored local-state scanning, "
                    "and exclusion of browser profiles, cookies, and local database text."
                ),
            ),
            run=fixtures.run_privacy_output_guards_fixture,
        ),
    ]


def _reject_unknown_case_ids(
    case_ids: Sequence[str],
    definitions: list[EvalDefinition],
) -> None:
    known_case_ids = {definition.metadata.id for definition in definitions}
    unknown_case_ids = sorted(set(case_ids) - known_case_ids)
    if not unknown_case_ids:
        return
    valid_values = ", ".join(sorted(known_case_ids))
    unknown_values = ", ".join(unknown_case_ids)
    case_word = "ID" if len(unknown_case_ids) == 1 else "IDs"
    raise EvalHarnessError(
        f"Unknown eval case {case_word}: {unknown_values}. "
        f"Valid case IDs are: {valid_values}."
    )


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
        definition.run(case_work_dir)
    except fixtures.EvalFixtureFailure as exc:
        return EvalCaseResult(
            id=metadata.id,
            category=metadata.category,
            name=metadata.name,
            passed=False,
            message=str(exc),
            details={
                "description": metadata.description,
                **exc.details,
            },
        )
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
        passed=True,
        message="Passed.",
        details={"description": metadata.description},
    )
