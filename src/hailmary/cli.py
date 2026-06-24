from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from hailmary.config import (
    AppConfig,
    ConfigError,
    create_local_state,
    load_config,
)
from hailmary.ingest.folder_loader import (
    IngestionError,
)
from hailmary.ingest.folder_loader import (
    ingest_folder as ingest_folder_path,
)
from hailmary.scoring.memo import ScoringError, score_latest_ingestion

app = typer.Typer(
    help="Evaluate private startup deals from local diligence documents.",
    no_args_is_help=True,
)
console = Console()


def _exit_with_config_error(exc: ConfigError) -> NoReturn:
    console.print(f"Error: {exc}")
    raise typer.Exit(1) from None


def _config_from_options(data_dir: Path | None, *, ignore_saved: bool = False) -> AppConfig:
    try:
        return load_config(data_dir=data_dir, ignore_saved=ignore_saved)
    except ConfigError as exc:
        _exit_with_config_error(exc)


@app.command("init")
def init(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should store local generated files.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace the local config file if it already exists.",
        ),
    ] = False,
) -> None:
    """Create local folders for generated Hail Mary files."""

    config = _config_from_options(data_dir, ignore_saved=force)
    try:
        result = create_local_state(config, force=force)
    except ConfigError as exc:
        _exit_with_config_error(exc)

    console.print(f"Created Hail Mary local folders in {result.data_dir}.")
    if result.config_created:
        console.print(f"Created local config at {result.config_path}.")
    else:
        console.print(f"Kept existing local config at {result.config_path}.")


@app.command("ingest-folder")
def ingest_folder(
    folder: Annotated[
        Path,
        typer.Argument(help="Folder containing one or more deal document folders."),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should store local generated files.",
        ),
    ] = None,
) -> None:
    """Scan a local folder and save source-linked document metadata."""

    config = _config_from_options(data_dir)
    try:
        create_local_state(config, force=False)
    except ConfigError as exc:
        _exit_with_config_error(exc)
    try:
        summary = ingest_folder_path(folder, config=config)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise typer.BadParameter(str(exc), param_hint="folder") from None
    except IngestionError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    deal_word = "deal" if len(summary.deals) == 1 else "deals"
    doc_word = "document" if summary.document_count == 1 else "documents"
    console.print(
        f"Scanned {summary.root_path}. Found {len(summary.deals)} {deal_word} "
        f"and {summary.document_count} {doc_word}."
    )
    console.print(f"Saved the scan summary to {summary.summary_path}.")

    evidence_count = sum(deal.evidence_count for deal in summary.deals)
    claim_count = sum(deal.claim_count for deal in summary.deals)
    conflict_count = sum(deal.conflict_count for deal in summary.deals)
    deals_without_evidence = [
        deal for deal in summary.deals if deal.documents and deal.evidence_count == 0
    ]
    if evidence_count:
        evidence_word = "record" if evidence_count == 1 else "records"
        claim_word = "claim" if claim_count == 1 else "claims"
        console.print(
            f"Built {evidence_count} source-linked evidence {evidence_word} "
            f"and {claim_count} deal-term {claim_word}."
        )
    if deals_without_evidence:
        deal_names = ", ".join(deal.company_name for deal in deals_without_evidence)
        deal_word = "deal" if len(deals_without_evidence) == 1 else "deals"
        console.print(
            f"No usable evidence text was built for {deal_word}: {deal_names}. "
            "Hail Mary stored the files it could read, but cannot use their text yet."
        )
    if conflict_count:
        conflict_word = "conflict" if conflict_count == 1 else "conflicts"
        console.print(
            f"Found {conflict_count} deal-term {conflict_word}. "
            "Review the cited evidence before relying on those terms."
        )

    image_text_documents = sum(
        1
        for deal in summary.deals
        for document in deal.documents
        if document.source.ocr_recommended or document.source.vision_recommended
    )
    if image_text_documents:
        document_word = "document" if image_text_documents == 1 else "documents"
        console.print(
            f"{image_text_documents} {document_word} may need image-based text reading "
            "(OCR) before Hail Mary can use all of their content."
        )

    if summary.skipped_files:
        console.print(
            f"Skipped {len(summary.skipped_files)} unsupported or ignored files. "
            "These were not treated as diligence documents."
        )
    if summary.unreadable_paths:
        path_word = "path" if len(summary.unreadable_paths) == 1 else "paths"
        console.print(
            f"Could not read {len(summary.unreadable_paths)} {path_word}. "
            "Hail Mary did not scan those locations, so diligence documents may be missing."
        )


@app.command("score-deals")
def score_deals(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read generated evidence and write reports.",
        ),
    ] = None,
) -> None:
    """Score ingested deals and write local Markdown memos."""

    config = _config_from_options(data_dir)
    try:
        result = score_latest_ingestion(config=config)
    except ScoringError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    deal_word = "deal" if result.deal_count == 1 else "deals"
    memo_word = "memo" if result.deal_count == 1 else "memos"
    console.print(f"Scored {result.deal_count} {deal_word}.")
    console.print(f"Saved Markdown {memo_word} to {result.report_dir}.")
    for scored_deal in result.scored_deals:
        console.print(
            f"{scored_deal.company_name}: {scored_deal.recommendation}, "
            f"check size {_format_check_size(scored_deal.check_size)}, "
            f"score {scored_deal.total_score}/{scored_deal.max_score}."
        )


def _format_check_size(check_size: int) -> str:
    if check_size == 0:
        return "$0"
    if check_size % 1_000 == 0:
        return f"${check_size // 1_000}K"
    return f"${check_size / 1_000:g}K"
