from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

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

app = typer.Typer(
    help="Evaluate private startup deals from local diligence documents.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console(highlight=False)


def _plain(message: str, *, style: str | None = None) -> Text:
    if style is None:
        return Text(message)
    return Text(message, style=style)


def _print_panel(
    title: str,
    renderables: Sequence[RenderableType],
    *,
    border_style: str,
) -> None:
    console.print(
        Panel(
            Group(*renderables),
            title=title,
            border_style=border_style,
            padding=(1, 2),
            expand=False,
        )
    )


def _print_error(message: str) -> None:
    _print_panel(
        "Error",
        [_plain(f"Error: {message}", style="bold red")],
        border_style="red",
    )


def _print_section(title: str, lines: Sequence[Text], *, style: str) -> None:
    console.print(Rule(title, style=style))
    for line in lines:
        console.print(line)


def _exit_with_config_error(exc: ConfigError) -> NoReturn:
    _print_error(str(exc))
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

    table = Table(
        box=box.SIMPLE,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("Status", style="bold green")
    table.add_column("Location")
    table.add_row(
        _plain("Created Hail Mary local folders"),
        _plain(str(result.data_dir)),
    )
    if result.config_created:
        table.add_row(_plain("Created local config"), _plain(str(result.config_path)))
    else:
        table.add_row(_plain("Kept existing local config"), _plain(str(result.config_path)))
    _print_panel("Init complete", [table], border_style="green")


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
        _print_error(str(exc))
        raise typer.Exit(1) from None
    except IngestionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    deal_word = "deal" if len(summary.deals) == 1 else "deals"
    doc_word = "document" if summary.document_count == 1 else "documents"
    summary_lines: list[Text] = [
        _plain(
            f"Found {len(summary.deals)} {deal_word} and {summary.document_count} "
            f"{doc_word}. Scanned {summary.root_path}."
        ),
        _plain(f"Saved the scan summary to {summary.summary_path}."),
    ]

    metrics = Table(
        box=box.SIMPLE,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
    )
    metrics.add_column("Metric", style="bold cyan")
    metrics.add_column("Value", justify="right")
    metrics.add_row(_plain("Deals"), _plain(str(len(summary.deals))))
    metrics.add_row(_plain("Documents"), _plain(str(summary.document_count)))

    evidence_count = sum(deal.evidence_count for deal in summary.deals)
    claim_count = sum(deal.claim_count for deal in summary.deals)
    conflict_count = sum(deal.conflict_count for deal in summary.deals)
    deals_without_evidence = [
        deal for deal in summary.deals if deal.documents and deal.evidence_count == 0
    ]
    if evidence_count:
        evidence_word = "record" if evidence_count == 1 else "records"
        claim_word = "claim" if claim_count == 1 else "claims"
        summary_lines.append(
            _plain(
                f"Built {evidence_count} source-linked evidence {evidence_word} "
                f"and {claim_count} deal-term {claim_word}."
            )
        )
        metrics.add_row(_plain("Evidence records"), _plain(str(evidence_count)))
        metrics.add_row(_plain("Deal-term claims"), _plain(str(claim_count)))

    _print_section("Scan complete", summary_lines, style="green")
    console.print(metrics)

    warning_lines: list[Text] = []
    if deals_without_evidence:
        deal_names = ", ".join(deal.company_name for deal in deals_without_evidence)
        deal_word = "deal" if len(deals_without_evidence) == 1 else "deals"
        warning_lines.append(
            _plain(
                f"No usable evidence text was built for {deal_word}: {deal_names}. "
                "Hail Mary stored the files it could read, but cannot use their text yet."
            )
        )
    if conflict_count:
        conflict_word = "conflict" if conflict_count == 1 else "conflicts"
        warning_lines.append(
            _plain(
                f"Found {conflict_count} deal-term {conflict_word}. "
                "Review the cited evidence before relying on those terms."
            )
        )

    image_text_documents = sum(
        1
        for deal in summary.deals
        for document in deal.documents
        if document.source.ocr_recommended or document.source.vision_recommended
    )
    if image_text_documents:
        document_word = "document" if image_text_documents == 1 else "documents"
        warning_lines.append(
            _plain(
                f"{image_text_documents} {document_word} may need image-based text reading "
                "(OCR) before Hail Mary can use all of their content."
            )
        )

    if summary.skipped_files:
        warning_lines.append(
            _plain(
                f"Skipped {len(summary.skipped_files)} unsupported or ignored files. "
                "These were not treated as diligence documents."
            )
        )
    if summary.unreadable_paths:
        path_word = "path" if len(summary.unreadable_paths) == 1 else "paths"
        warning_lines.append(
            _plain(
                f"Could not read {len(summary.unreadable_paths)} {path_word}. "
                "Hail Mary did not scan those locations, so diligence documents may be missing."
            )
        )

    if warning_lines:
        _print_section("Review needed", warning_lines, style="yellow")
