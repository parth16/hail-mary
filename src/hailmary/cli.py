from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from hailmary.config import AppConfig, create_local_state, load_config
from hailmary.ingest.folder_loader import ingest_folder as ingest_folder_path

app = typer.Typer(
    help="Evaluate private startup deals from local diligence documents.",
    no_args_is_help=True,
)
console = Console()


def _config_from_options(data_dir: Path | None) -> AppConfig:
    return load_config(data_dir=data_dir)


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

    config = _config_from_options(data_dir)
    result = create_local_state(config, force=force)

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
    create_local_state(config, force=False)
    try:
        summary = ingest_folder_path(folder, config=config)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise typer.BadParameter(str(exc), param_hint="folder") from None

    deal_word = "deal" if len(summary.deals) == 1 else "deals"
    doc_word = "document" if summary.document_count == 1 else "documents"
    console.print(
        f"Scanned {summary.root_path}. Found {len(summary.deals)} {deal_word} "
        f"and {summary.document_count} {doc_word}."
    )
    console.print(f"Saved the scan summary to {summary.summary_path}.")

    if summary.skipped_files:
        console.print(
            f"Skipped {len(summary.skipped_files)} unsupported or ignored files. "
            "These were not treated as diligence documents."
        )
