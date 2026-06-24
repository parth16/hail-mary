from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from hailmary.agents.packets import (
    AgentPacketError,
    load_agent_input_packet,
    load_agent_review_output,
    prepare_agent_packets,
)
from hailmary.agents.validation import validate_agent_output
from hailmary.config import (
    AppConfig,
    ConfigError,
    create_local_state,
    load_config,
)
from hailmary.evals import EvalCategory, EvalHarnessError, run_builtin_evals
from hailmary.ingest.folder_loader import (
    IngestionError,
)
from hailmary.ingest.folder_loader import (
    ingest_folder as ingest_folder_path,
)
from hailmary.research import (
    ResearchImportError,
    ResearchPlanError,
    ResearchProvider,
    ResearchProviderCategory,
    ResearchTaskStatus,
    ResearchTemplateError,
    builtin_research_providers,
    import_research_results,
    prepare_research_plan,
    prepare_research_results_template,
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


@app.command("prepare-agent-packets")
def prepare_agent_packets_command(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read generated evidence and write agent packets.",
        ),
    ] = None,
) -> None:
    """Prepare local JSON packets for structured model review."""

    config = _config_from_options(data_dir)
    try:
        result = prepare_agent_packets(config=config)
    except AgentPacketError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    packet_word = "packet" if result.packet_count == 1 else "packets"
    console.print(f"Prepared {result.packet_count} local agent input {packet_word}.")
    console.print(f"Saved JSON {packet_word} to {result.output_dir}.")
    console.print(
        "These files contain generated diligence material and should stay private."
    )


@app.command("validate-agent-output")
def validate_agent_output_command(
    output_path: Annotated[
        Path,
        typer.Argument(help="JSON output returned by the review model."),
    ],
    packet_path: Annotated[
        Path,
        typer.Argument(help="Agent packet that was given to the review model."),
    ],
) -> None:
    """Validate structured model output against one agent packet."""

    try:
        packet = load_agent_input_packet(packet_path)
        output = load_agent_review_output(output_path)
    except AgentPacketError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    result = validate_agent_output(output, packet)
    if not result.valid:
        issue_word = "problem" if len(result.issues) == 1 else "problems"
        console.print(
            f"Agent output did not pass validation. Found {len(result.issues)} "
            f"{issue_word}."
        )
        for issue in result.issues:
            console.print(f"- {issue.location}: {issue.message}")
        raise typer.Exit(1) from None

    console.print(
        "Agent output passed validation. Every cited evidence ID is in the packet."
    )


@app.command("run-evals")
def run_evals_command(
    case: Annotated[
        list[str] | None,
        typer.Option(
            "--case",
            help="Run one built-in synthetic eval case by ID. Can be used more than once.",
        ),
    ] = None,
    category: Annotated[
        list[str] | None,
        typer.Option(
            "--category",
            help=(
                "Run one eval category. Can be used more than once. Valid values: "
                "extraction, citation, contradiction, prompt_injection, "
                "score_calibration, missing_data, memo_snapshot."
            ),
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the eval results as JSON.",
        ),
    ] = False,
) -> None:
    """Run local synthetic correctness evals."""

    try:
        categories = _parse_eval_categories(category or [])
        summary = run_builtin_evals(case_ids=case or [], categories=categories)
    except EvalHarnessError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    if json_output:
        console.out(summary.model_dump_json(indent=2))
    else:
        eval_word = "eval" if summary.total_count == 1 else "evals"
        console.print(
            f"Ran {summary.total_count} synthetic {eval_word}. "
            f"{summary.passed_count} passed, {summary.failed_count} failed."
        )
        for result in summary.failed_results:
            console.print(f"- {result.id}: {result.message}")
            for detail_name, detail_value in _operator_eval_details(result.details):
                console.print(f"  {detail_name}: {detail_value}")

    if not summary.passed:
        raise typer.Exit(1) from None


@app.command("list-research-providers")
def list_research_providers_command(
    include_paid: Annotated[
        bool,
        typer.Option(
            "--include-paid",
            help="Also list optional paid data sources. These need a paid account or license.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print provider metadata as JSON.",
        ),
    ] = False,
) -> None:
    """List external research sources Hail Mary can plan for."""

    providers = builtin_research_providers(include_paid=include_paid)
    if json_output:
        payload = [provider.model_dump(mode="json") for provider in providers]
        console.out(json.dumps(payload, indent=2))
        return

    free_providers = _providers_by_category(
        providers,
        ResearchProviderCategory.FREE_PUBLIC,
    )
    portal_providers = _providers_by_category(
        providers,
        ResearchProviderCategory.AUTHENTICATED_PORTAL,
    )
    paid_providers = _providers_by_category(
        providers,
        ResearchProviderCategory.PAID_OPTIONAL,
    )
    console.print("Free and public sources:")
    for provider in free_providers:
        console.print(f"- {provider.name}: {provider.description}")
    console.print("Authenticated sources:")
    for provider in portal_providers:
        console.print(f"- {provider.name}: {provider.operator_note}")
    if paid_providers:
        console.print("Optional paid sources:")
        for provider in paid_providers:
            console.print(f"- {provider.name}: {provider.description}")
    else:
        console.print("Optional paid sources are hidden. Use --include-paid to list them.")


@app.command("prepare-research-plan")
def prepare_research_plan_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to plan research for. Use more than once for multiple companies. "
                "If omitted, Hail Mary uses the latest ingestion summary."
            ),
        ),
    ] = None,
    website: Annotated[
        str | None,
        typer.Option(
            "--website",
            help="Official company website. Use only when planning for one company.",
        ),
    ] = None,
    meridian_url: Annotated[
        str | None,
        typer.Option(
            "--meridian-url",
            help=(
                "Authenticated Meridian deal URL. Hail Mary records it as a manual task "
                "and does not open it."
            ),
        ),
    ] = None,
    include_paid: Annotated[
        bool,
        typer.Option(
            "--include-paid",
            help="Include optional paid sources as manual tasks. No paid source is contacted.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read generated evidence and write the plan.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the research plan as JSON.",
        ),
    ] = False,
) -> None:
    """Prepare a private external research checklist without contacting any source."""

    config = _config_from_options(data_dir)
    try:
        result = prepare_research_plan(
            config=config,
            company_names=company or [],
            website_url=website,
            meridian_url=meridian_url,
            include_paid=include_paid,
        )
    except ResearchPlanError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    if json_output:
        console.out(result.plan.model_dump_json(indent=2))
        return

    deal_word = "deal" if result.deal_count == 1 else "deals"
    task_word = "task" if result.task_count == 1 else "tasks"
    console.print(
        f"Prepared an external research plan for {result.deal_count} {deal_word} "
        f"with {result.task_count} {task_word}."
    )
    console.print(f"Saved the private JSON plan to {result.output_path}.")
    console.print("No websites, APIs, paid databases, or Meridian pages were contacted.")

    manual_count = sum(
        1
        for task in result.plan.tasks
        if task.status == ResearchTaskStatus.NEEDS_OPERATOR
    )
    if manual_count:
        console.print(f"{manual_count} {task_word} need your manual action before use.")
    if result.plan.local_only:
        console.print("Local-only mode is on, so this plan is a checklist only.")


@app.command("prepare-research-results-template")
def prepare_research_results_template_command(
    research_plan: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Research plan JSON to turn into a fillable results file. If omitted, "
                "Hail Mary uses the latest private research plan."
            ),
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read plans and write the template.",
        ),
    ] = None,
) -> None:
    """Prepare a private fillable JSON template for external research results."""

    config = _config_from_options(data_dir)
    try:
        result = prepare_research_results_template(
            config=config,
            plan_path=research_plan,
        )
    except ResearchTemplateError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    result_word = "result" if result.result_count == 1 else "results"
    console.print(
        f"Prepared a fillable external research results template with "
        f"{result.result_count} {result_word}."
    )
    console.print(f"Saved the private JSON template to {result.output_path}.")
    console.print("No websites, APIs, paid databases, or Meridian pages were contacted.")
    data_dir_option = f" --data-dir {config.data_dir}" if data_dir is not None else ""
    console.print(
        "Fill in source-backed facts, then run "
        f"`hailmary import-research-results {result.output_path}{data_dir_option} --dry-run`."
    )


@app.command("import-research-results")
def import_research_results_command(
    results_file: Annotated[
        Path,
        typer.Argument(
            help="Local JSON file with manually collected external research results.",
        ),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read generated evidence and write updates.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the file and show what would change without writing evidence.",
        ),
    ] = False,
) -> None:
    """Import source-linked external research evidence from a local JSON file."""

    config = _config_from_options(data_dir)
    try:
        result = import_research_results(
            config=config,
            results_path=results_file,
            dry_run=dry_run,
        )
    except ResearchImportError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    record_word = "record" if result.imported_count == 1 else "records"
    deal_word = "deal" if result.deal_count == 1 else "deals"
    if result.dry_run:
        console.print(
            f"Dry run: {result.imported_count} external research evidence {record_word} "
            f"would be imported into {result.deal_count} {deal_word}."
        )
    else:
        console.print(
            f"Imported {result.imported_count} external research evidence {record_word} "
            f"into {result.deal_count} {deal_word}."
        )
    if result.skipped_duplicate_count:
        duplicate_word = "record" if result.skipped_duplicate_count == 1 else "records"
        console.print(
            f"Skipped {result.skipped_duplicate_count} duplicate {duplicate_word}."
        )
    if result.skipped_blank_template_row_count:
        row_word = "row" if result.skipped_blank_template_row_count == 1 else "rows"
        console.print(
            f"Skipped {result.skipped_blank_template_row_count} untouched template "
            f"{row_word}."
        )
    if result.dry_run:
        for deal in result.deals:
            if deal.imported_count:
                deal_record_word = "record" if deal.imported_count == 1 else "records"
                console.print(
                    f"- {deal.company_name}: would add {deal.imported_count} "
                    f"{deal_record_word}."
                )
        console.print("No evidence stores were changed.")
        console.print("No websites or APIs were contacted.")
        return
    if result.updated_store_paths:
        store_word = "store" if len(result.updated_store_paths) == 1 else "stores"
        console.print(f"Updated {len(result.updated_store_paths)} evidence {store_word}.")
        for deal in result.deals:
            if deal.imported_count:
                deal_record_word = "record" if deal.imported_count == 1 else "records"
                console.print(
                    f"- {deal.company_name}: added {deal.imported_count} "
                    f"{deal_record_word}."
                )
    else:
        console.print("No new evidence records were added.")
    console.print("No websites or APIs were contacted.")


def _parse_eval_categories(raw_categories: list[str]) -> list[EvalCategory]:
    categories: list[EvalCategory] = []
    valid_values = ", ".join(category.value for category in EvalCategory)
    for raw_category in raw_categories:
        try:
            categories.append(EvalCategory(raw_category))
        except ValueError as exc:
            raise EvalHarnessError(
                f"Unknown eval category {raw_category!r}. Valid categories are: "
                f"{valid_values}."
            ) from exc
    return categories


def _operator_eval_details(details: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (name.replace("_", " "), value)
        for name, value in details.items()
        if name != "description" and value
    ]


def _providers_by_category(
    providers: list[ResearchProvider],
    category: ResearchProviderCategory,
) -> list[ResearchProvider]:
    return [provider for provider in providers if provider.category == category]


def _format_check_size(check_size: int) -> str:
    if check_size == 0:
        return "$0"
    if check_size % 1_000 == 0:
        return f"${check_size // 1_000}K"
    return f"${check_size / 1_000:g}K"
