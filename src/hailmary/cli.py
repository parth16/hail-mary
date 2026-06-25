from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

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
from hailmary.evaluation import EvaluationError, evaluate_deal_folder
from hailmary.ingest.folder_loader import (
    IngestionError,
)
from hailmary.ingest.folder_loader import (
    ingest_folder as ingest_folder_path,
)
from hailmary.research import (
    MeridianWorkflowError,
    ResearchCollectionError,
    ResearchImportError,
    ResearchPlanError,
    ResearchProvider,
    ResearchProviderCategory,
    ResearchTaskStatus,
    ResearchTemplateError,
    WebResearchError,
    builtin_research_providers,
    collect_github_repositories,
    collect_sbir_awards,
    collect_sec_form_d_filings,
    collect_usaspending_awards,
    collect_web_research,
    import_research_results,
    prepare_meridian_workflow,
    prepare_public_research_results,
    prepare_research_plan,
    prepare_research_results_template,
)
from hailmary.scoring.memo import ScoringError, score_latest_ingestion

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


def _print_json(json_text: str) -> None:
    console.print(json_text, markup=False, highlight=False, soft_wrap=True)


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
        console.print(line, soft_wrap=True)


def _two_column_table(
    first_column: str,
    second_column: str,
    *,
    second_justify: Literal["default", "left", "center", "right", "full"] = "left",
) -> Table:
    table = Table(
        box=box.SIMPLE,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column(first_column, style="bold cyan")
    table.add_column(second_column, justify=second_justify)
    return table


def _research_provider_table(
    title: str,
    providers: Sequence[ResearchProvider],
    *,
    use_operator_note: bool = False,
) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("Source", style="bold cyan")
    table.add_column("Notes")
    for provider in providers:
        notes = provider.operator_note if use_operator_note else provider.description
        table.add_row(_plain(provider.name), _plain(notes))
    return table


def _exit_with_config_error(exc: ConfigError) -> NoReturn:
    _print_error(str(exc))
    raise typer.Exit(1) from None


def _config_from_options(data_dir: Path | None, *, ignore_saved: bool = False) -> AppConfig:
    try:
        return load_config(data_dir=data_dir, ignore_saved=ignore_saved)
    except ConfigError as exc:
        _exit_with_config_error(exc)


def _config_with_ocr_override(
    config: AppConfig,
    *,
    enable_ocr: bool | None,
) -> AppConfig:
    if enable_ocr is None:
        return config
    return config.model_copy(update={"enable_ocr": enable_ocr})


def _has_ocr_warning(notes: str | None) -> bool:
    if not notes:
        return False
    lowered_notes = notes.lower()
    if "image-based text reading (ocr)" not in lowered_notes:
        return False
    return any(
        marker in lowered_notes
        for marker in [
            "could not",
            "found no readable text",
            "low confidence",
            "needs the local",
        ]
    )


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

    table = _two_column_table("Status", "Location")
    table.add_row(
        _plain("Created Hail Mary local folders", style="bold green"),
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
    enable_ocr: Annotated[
        bool | None,
        typer.Option(
            "--enable-ocr/--disable-ocr",
            help=(
                "Use local image-based text reading (OCR) when local tools are installed. "
                "OCR means reading text from images."
            ),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Scan a local folder and save source-linked document metadata."""

    config = _config_with_ocr_override(
        _config_from_options(data_dir),
        enable_ocr=enable_ocr,
    )
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
    summary_lines = [
        _plain(
            f"Found {len(summary.deals)} {deal_word} and {summary.document_count} "
            f"{doc_word}. Scanned {summary.root_path}."
        ),
        _plain(f"Saved the scan summary to {summary.summary_path}."),
    ]
    metrics = _two_column_table("Metric", "Value", second_justify="right")
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

    ocr_applied_documents = sum(
        1
        for deal in summary.deals
        for document in deal.documents
        if document.source.ocr_applied
    )
    if ocr_applied_documents:
        document_word = "document" if ocr_applied_documents == 1 else "documents"
        summary_lines.append(
            _plain(
                f"Used image-based text reading (OCR) on {ocr_applied_documents} "
                f"{document_word}. OCR means reading text from images."
            )
        )

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
                "(OCR) before Hail Mary can use all of their content. OCR means reading "
                "text from images."
            )
        )

    ocr_warning_documents = sum(
        1
        for deal in summary.deals
        for document in deal.documents
        if _has_ocr_warning(document.source.notes)
    )
    if ocr_warning_documents:
        document_word = "document" if ocr_warning_documents == 1 else "documents"
        warning_lines.append(
            _plain(
                f"{ocr_warning_documents} {document_word} had image-based text reading "
                "(OCR) warnings. OCR means reading text from images. Review the saved "
                "document metadata before relying on that text."
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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    deal_word = "deal" if result.deal_count == 1 else "deals"
    memo_word = "memo" if result.deal_count == 1 else "memos"
    result_lines = [
        _plain(f"Scored {result.deal_count} {deal_word}."),
        _plain(f"Saved Markdown {memo_word} to {result.report_dir}."),
    ]
    if result.portfolio_report_path is not None:
        result_lines.append(
            _plain(f"Saved the portfolio comparison report to {result.portfolio_report_path}.")
        )
        result_lines.append(
            _plain(
                "Portfolio comparison report file: "
                f"{result.portfolio_report_path.name}."
            )
        )
    deals = Table(
        box=box.SIMPLE,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
    )
    deals.add_column("Company", style="bold cyan")
    deals.add_column("Recommendation")
    deals.add_column("Check size", justify="right")
    deals.add_column("Score", justify="right")
    for scored_deal in result.scored_deals:
        deals.add_row(
            _plain(scored_deal.company_name),
            _plain(scored_deal.recommendation),
            _plain(_format_check_size(scored_deal.check_size)),
            _plain(f"{scored_deal.total_score}/{scored_deal.max_score}"),
        )
    _print_panel("Scoring complete", [*result_lines, deals], border_style="green")


@app.command("evaluate-deal")
def evaluate_deal(
    folder: Annotated[
        Path,
        typer.Argument(help="One company folder containing local diligence documents."),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should store private generated files.",
        ),
    ] = None,
    max_concurrency: Annotated[
        int,
        typer.Option(
            "--max-concurrency",
            help="Maximum number of specialist model reviews to run at once.",
        ),
    ] = 3,
    enable_ocr: Annotated[
        bool | None,
        typer.Option(
            "--enable-ocr/--disable-ocr",
            help=(
                "Use local image-based text reading (OCR) during ingestion when local "
                "tools are installed. OCR means reading text from images."
            ),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Evaluate one deal end to end and write a final Markdown memo."""

    config = _config_with_ocr_override(
        _config_from_options(data_dir),
        enable_ocr=enable_ocr,
    )
    stages = [
        "local setup and privacy checks",
        "ingestion",
        "deterministic scoring",
        "agent packet preparation",
        "specialist committee review",
        "final decision review",
        "final memo write",
    ]
    stage_numbers = {stage: index for index, stage in enumerate(stages, start=1)}

    def print_stage(stage: str) -> None:
        number = stage_numbers.get(stage)
        prefix = f"{number}. " if number is not None else ""
        console.print(_plain(f"{prefix}{stage}", style="bold cyan"))

    try:
        result = evaluate_deal_folder(
            folder,
            config=config,
            max_concurrency=max_concurrency,
            stage_callback=print_stage,
        )
    except EvaluationError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    summary = _two_column_table("Result", "Value")
    summary.add_row(
        _plain("Recommendation"),
        _plain(str(result.final_recommendation.recommendation)),
    )
    summary.add_row(
        _plain("Check size"),
        _plain(_format_check_size(result.final_recommendation.check_size)),
    )
    summary.add_row(
        _plain("Score"),
        _plain(f"{result.deterministic_score.total_score}/{result.deterministic_score.max_score}"),
    )
    summary.add_row(_plain("Confidence"), _plain(str(result.deterministic_score.confidence)))
    summary.add_row(_plain("Final memo"), _plain(str(result.final_memo_path)))

    renderables: list[RenderableType] = [
        _plain(f"Evaluated {result.company_name}."),
        summary,
    ]
    if result.warnings:
        warning_table = _two_column_table("Warning", "Detail")
        for index, warning in enumerate(result.warnings, start=1):
            warning_table.add_row(_plain(str(index)), _plain(warning))
        renderables.append(warning_table)
    else:
        renderables.append(_plain("Validation warnings: none."))

    _print_panel("Deal evaluation complete", renderables, border_style="green")


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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    packet_word = "packet" if result.packet_count == 1 else "packets"
    _print_panel(
        "Agent packets prepared",
        [
            _plain(f"Prepared {result.packet_count} local agent input {packet_word}."),
            _plain(f"Saved JSON {packet_word} to {result.output_dir}."),
            _plain("These files contain generated diligence material and should stay private."),
        ],
        border_style="green",
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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    result = validate_agent_output(output, packet)
    if not result.valid:
        issue_word = "problem" if len(result.issues) == 1 else "problems"
        issue_lines = [
            _plain(
                f"Agent output did not pass validation. Found {len(result.issues)} {issue_word}."
            )
        ]
        for issue in result.issues:
            issue_lines.append(_plain(f"- {issue.location}: {issue.message}"))
        _print_panel(
            "Validation failed",
            issue_lines,
            border_style="red",
        )
        raise typer.Exit(1) from None

    _print_panel(
        "Validation passed",
        [_plain("Agent output passed validation. Every cited evidence ID is in the packet.")],
        border_style="green",
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
                "extraction, citation, contradiction, research_import, prompt_injection, "
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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    if json_output:
        _print_json(summary.model_dump_json(indent=2))
    else:
        eval_word = "eval" if summary.total_count == 1 else "evals"
        result_lines = [
            _plain(
                f"Ran {summary.total_count} synthetic {eval_word}. "
                f"{summary.passed_count} passed, {summary.failed_count} failed."
            )
        ]
        for failed_result in summary.failed_results:
            result_lines.append(_plain(f"- {failed_result.id}: {failed_result.message}"))
            for detail_name, detail_value in _operator_eval_details(failed_result.details):
                result_lines.append(_plain(f"  {detail_name}: {detail_value}"))
        _print_panel(
            "Eval results",
            result_lines,
            border_style="green" if summary.passed else "red",
        )

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
        _print_json(json.dumps(payload, indent=2))
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
    renderables: list[RenderableType] = [
        _research_provider_table("Free and public sources", free_providers),
        _research_provider_table(
            "Authenticated sources",
            portal_providers,
            use_operator_note=True,
        ),
    ]
    if paid_providers:
        renderables.append(_research_provider_table("Optional paid sources", paid_providers))
    else:
        renderables.append(
            _plain("Optional paid sources are hidden. Use --include-paid to list them.")
        )
    _print_panel("Research providers", renderables, border_style="cyan")


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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    if json_output:
        _print_json(result.plan.model_dump_json(indent=2))
        return

    deal_word = "deal" if result.deal_count == 1 else "deals"
    task_word = "task" if result.task_count == 1 else "tasks"
    plan_lines = [
        _plain(
            f"Prepared an external research plan for {result.deal_count} {deal_word} "
            f"with {result.task_count} {task_word}."
        ),
        _plain(f"Saved the private JSON plan to {result.output_path}."),
        _plain("No websites, APIs, paid databases, or Meridian pages were contacted."),
    ]

    manual_count = sum(
        1 for task in result.plan.tasks if task.status == ResearchTaskStatus.NEEDS_OPERATOR
    )
    if manual_count:
        plan_lines.append(_plain(f"{manual_count} {task_word} need your manual action before use."))
    if result.plan.local_only:
        plan_lines.append(_plain("Local-only mode is on, so this plan is a checklist only."))
    _print_panel("Research plan prepared", plan_lines, border_style="green")


@app.command("collect-web-research")
def collect_web_research_command(
    research_plan: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Research plan JSON to collect public web pages from. If omitted, "
                "Hail Mary uses the latest private research plan."
            ),
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            "--provider",
            help=(
                "Only collect tasks for this provider ID. Use more than once for "
                "multiple providers."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show which public URLs would be fetched without contacting websites.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should read plans and write private results.",
        ),
    ] = None,
) -> None:
    """Collect import-ready evidence text from planned public web pages."""

    config = _config_from_options(data_dir)
    try:
        result = collect_web_research(
            config=config,
            plan_path=research_plan,
            provider_ids=provider or [],
            dry_run=dry_run,
        )
    except WebResearchError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    fetch_word = "page" if result.fetched_count == 1 else "pages"
    planned_word = "page" if result.planned_count == 1 else "pages"
    failed_word = "task" if result.failed_count == 1 else "tasks"
    skipped_word = "task" if result.skipped_count == 1 else "tasks"
    if result.dry_run:
        lines = [
            _plain(
                f"Dry run: {result.planned_count} public web {planned_word} would be fetched."
            ),
            _plain("No websites were contacted and no results file was saved."),
        ]
        border_style = "yellow"
        title = "Web research preview"
    else:
        lines = [
            _plain(f"Fetched {result.fetched_count} public web {fetch_word}.")
        ]
        if result.output_path is not None:
            data_dir_option = (
                f" --data-dir {shlex.quote(str(config.data_dir))}"
                if data_dir is not None
                else ""
            )
            next_command = (
                f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
                f"{data_dir_option} --dry-run`."
            )
            lines.append(_plain(f"Saved the private JSON results file to {result.output_path}."))
            lines.append(_plain(f"Next, run {next_command}"))
        else:
            lines.append(_plain("No results file was saved."))
        lines.append(
            _plain("No authenticated, paid, Meridian, or local-only sources were contacted.")
        )
        border_style = "green" if result.failed_count == 0 else "yellow"
        title = "Web research collected"

    if result.failed_count:
        lines.append(_plain(f"{result.failed_count} web research {failed_word} failed."))
    if result.skipped_count:
        lines.append(_plain(f"Skipped {result.skipped_count} ineligible {skipped_word}."))
    for task in result.tasks:
        if task.status in {"failed", "planned"}:
            lines.append(
                _plain(
                    f"- {task.company_name} / {task.provider_id}: {task.reason}"
                )
            )

    _print_panel(title, lines, border_style=border_style)
    if result.failed_count:
        raise typer.Exit(1)


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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    result_word = "result" if result.result_count == 1 else "results"
    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    _print_section(
        "Research template prepared",
        [
            _plain(
                f"Prepared a fillable external research results template with "
                f"{result.result_count} {result_word}."
            ),
            _plain(f"Saved the private JSON template to {result.output_path}."),
            _plain("No websites, APIs, paid databases, or Meridian pages were contacted."),
            _plain(f"Fill in source-backed facts, then run {next_command}"),
        ],
        style="green",
    )


@app.command("prepare-public-research-results")
def prepare_public_research_results_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to prepare public research results for. Use more than once "
                "for multiple companies."
            ),
        ),
    ] = None,
    sec_form_d_results: Annotated[
        Path | None,
        typer.Option(
            "--sec-form-d-results",
            help=(
                "Local JSON file of SEC Form D search results. Hail Mary reads this "
                "file and does not contact SEC."
            ),
        ),
    ] = None,
    sam_gov_results: Annotated[
        Path | None,
        typer.Option(
            "--sam-gov-results",
            help=(
                "Local JSON file of SAM.gov results. Hail Mary reads this file "
                "and does not contact SAM.gov."
            ),
        ),
    ] = None,
    usaspending_results: Annotated[
        Path | None,
        typer.Option(
            "--usaspending-results",
            help=(
                "Local JSON file of USAspending results. Hail Mary reads this file "
                "and does not contact USAspending."
            ),
        ),
    ] = None,
    sbir_results: Annotated[
        Path | None,
        typer.Option(
            "--sbir-results",
            help=(
                "Local JSON file of SBIR/STTR award results. Hail Mary reads this "
                "file and does not contact SBIR.gov."
            ),
        ),
    ] = None,
    uspto_results: Annotated[
        Path | None,
        typer.Option(
            "--uspto-results",
            help=(
                "Local JSON file of USPTO results. Hail Mary reads this file "
                "and does not contact USPTO."
            ),
        ),
    ] = None,
    github_results: Annotated[
        Path | None,
        typer.Option(
            "--github-results",
            help=(
                "Local JSON file of GitHub results. Hail Mary reads this file "
                "and does not contact GitHub."
            ),
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write the private results file.",
        ),
    ] = None,
) -> None:
    """Prepare import-ready public research results from local source files."""

    config = _config_from_options(data_dir)
    try:
        result = prepare_public_research_results(
            config=config,
            company_names=company or [],
            sec_form_d_results_path=sec_form_d_results,
            sam_gov_results_path=sam_gov_results,
            usaspending_results_path=usaspending_results,
            sbir_results_path=sbir_results,
            uspto_results_path=uspto_results,
            github_results_path=github_results,
        )
    except ResearchCollectionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    result_word = "result" if result.result_count == 1 else "results"
    company_word = "company" if result.deal_count == 1 else "companies"
    if result.output_path is None:
        _print_section(
            "Public research results",
            [
                _plain(
                    f"No matching public research {result_word} were found for "
                    f"{result.deal_count} {company_word}."
                ),
                _plain("No results file was saved."),
                _plain("No websites or software data feeds were contacted."),
            ],
            style="yellow",
        )
        return

    result_lines = [
        _plain(
            f"Prepared {result.result_count} public research {result_word} for "
            f"{result.deal_count} {company_word}."
        )
    ]
    zero_result_companies = [deal.company_name for deal in result.deals if deal.result_count == 0]
    if zero_result_companies:
        result_lines.append(
            _plain(
                f"No public research results were prepared for: {', '.join(zero_result_companies)}."
            )
        )
    result_lines.extend(
        [
            _plain(f"Saved the private JSON results file to {result.output_path}."),
            _plain("No websites or software data feeds were contacted."),
        ]
    )
    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    result_lines.append(
        _plain(
            f"After ingesting the matching deal folders, run {next_command}"
        )
    )
    _print_section("Public research results prepared", result_lines, style="green")


@app.command("collect-sec-form-d-filings")
def collect_sec_form_d_filings_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to search for in SEC Form D filings. Use more than once "
                "for multiple companies."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=25,
            help="Maximum SEC Form D filing records to request per company.",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be sent to SEC EDGAR without contacting SEC.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write the private results file.",
        ),
    ] = None,
) -> None:
    """Collect public SEC Form D evidence for exact issuer-name matches."""

    config = _config_from_options(data_dir)
    try:
        result = collect_sec_form_d_filings(
            config=config,
            company_names=company or [],
            limit=limit,
            dry_run=dry_run,
        )
    except ResearchCollectionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    company_word = "company" if result.deal_count == 1 else "companies"
    result_word = "result" if result.result_count == 1 else "results"
    if result.dry_run:
        _print_section(
            "SEC Form D preview",
            [
                _plain(
                    f"Dry run: Hail Mary would send {result.deal_count} "
                    f"{company_word} to SEC EDGAR public filing search."
                ),
                _plain(
                    f"A live run can request up to {limit} filing records per page "
                    f"for up to 20 pages per company while looking for exact issuer-name "
                    "matches."
                ),
                _plain("No SEC requests were sent and no results file was saved."),
            ],
            style="yellow",
        )
        return

    if result.output_path is None:
        zero_result_companies = [
            deal.company_name for deal in result.deals if deal.result_count == 0
        ]
        lines = [
            _plain(
                f"No exact issuer-name SEC Form D {result_word} were found "
                f"for {result.deal_count} {company_word}."
            ),
            _plain(
                f"No exact SEC Form D matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            ),
            _plain("No results file was saved."),
        ]
        for warning in result.warnings:
            lines.append(_plain(f"Warning: {warning}"))
        _print_section("SEC Form D results", lines, style="yellow")
        return

    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    lines = [
        _plain(
            f"Collected {result.result_count} SEC Form D {result_word} for "
            f"{result.deal_count} {company_word}."
        ),
        _plain(f"Saved the private JSON results file to {result.output_path}."),
        _plain(
            "Only exact issuer-name matches were prepared. Hail Mary saved parsed "
            "filing metadata, not raw filings or contact details."
        ),
        _plain(f"Next, run {next_command}"),
    ]
    for warning in result.warnings:
        lines.append(_plain(f"Warning: {warning}"))
    zero_result_companies = [deal.company_name for deal in result.deals if deal.result_count == 0]
    if zero_result_companies:
        lines.append(
            _plain(
                f"No exact SEC Form D matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            )
        )
    _print_section("SEC Form D results collected", lines, style="green")


@app.command("collect-github-repositories")
def collect_github_repositories_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to search for in GitHub public repositories. Use more than "
                "once for multiple companies."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=25,
            help="Maximum GitHub repository records to request per company.",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be sent to GitHub without contacting the API.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write the private results file.",
        ),
    ] = None,
) -> None:
    """Collect public GitHub repository evidence for exact owner or repository matches."""

    config = _config_from_options(data_dir)
    try:
        result = collect_github_repositories(
            config=config,
            company_names=company or [],
            limit=limit,
            dry_run=dry_run,
        )
    except ResearchCollectionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    company_word = "company" if result.deal_count == 1 else "companies"
    result_word = "result" if result.result_count == 1 else "results"
    if result.dry_run:
        _print_section(
            "GitHub repository preview",
            [
                _plain(
                    f"Dry run: Hail Mary would send {result.deal_count} "
                    f"{company_word} to the GitHub public repository search API."
                ),
                _plain(
                    f"A live run can make repository-name, user-owner, and "
                    f"organization-owner searches, requesting up to {limit} repository "
                    "records per page for up to 5 pages per company while looking for "
                    "exact GitHub owner or repository-name matches."
                ),
                _plain("No GitHub API requests were sent and no results file was saved."),
            ],
            style="yellow",
        )
        return

    if result.output_path is None:
        zero_result_companies = [
            deal.company_name for deal in result.deals if deal.result_count == 0
        ]
        lines = [
            _plain(
                f"No exact GitHub owner or repository-name {result_word} were found "
                f"for {result.deal_count} {company_word}."
            ),
            _plain(
                f"No exact GitHub matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            ),
            _plain("No results file was saved."),
        ]
        for warning in result.warnings:
            lines.append(_plain(f"Warning: {warning}"))
        _print_section("GitHub repository results", lines, style="yellow")
        return

    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    lines = [
        _plain(
            f"Collected {result.result_count} GitHub repository {result_word} for "
            f"{result.deal_count} {company_word}."
        ),
        _plain(f"Saved the private JSON results file to {result.output_path}."),
        _plain(
            "Only exact GitHub owner or repository-name matches were prepared. "
            "Hail Mary saved repository metadata only and did not clone code or fetch "
            "README files."
        ),
        _plain(f"Next, run {next_command}"),
    ]
    for warning in result.warnings:
        lines.append(_plain(f"Warning: {warning}"))
    zero_result_companies = [deal.company_name for deal in result.deals if deal.result_count == 0]
    if zero_result_companies:
        lines.append(
            _plain(
                f"No exact GitHub matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            )
        )
    _print_section("GitHub repository results collected", lines, style="green")


@app.command("collect-usaspending-awards")
def collect_usaspending_awards_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to search for in USAspending. Use more than once for "
                "multiple companies."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=25,
            help="Maximum USAspending award records to request per company.",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be sent to USAspending without contacting the API.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write the private results file.",
        ),
    ] = None,
) -> None:
    """Collect public USAspending award evidence for exact recipient-name matches."""

    config = _config_from_options(data_dir)
    try:
        result = collect_usaspending_awards(
            config=config,
            company_names=company or [],
            limit=limit,
            dry_run=dry_run,
        )
    except ResearchCollectionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    company_word = "company" if result.deal_count == 1 else "companies"
    result_word = "result" if result.result_count == 1 else "results"
    if result.dry_run:
        _print_section(
            "USAspending preview",
            [
                _plain(
                    f"Dry run: Hail Mary would send {result.deal_count} "
                    f"{company_word} to the USAspending public API."
                ),
                _plain(
                    f"A live run can request up to {limit} award records per page "
                    f"for up to 20 pages per company while looking for exact matches."
                ),
                _plain("No API requests were sent and no results file was saved."),
            ],
            style="yellow",
        )
        return

    if result.output_path is None:
        lines = [
            _plain(
                f"No exact recipient-name USAspending {result_word} were found "
                f"for {result.deal_count} {company_word}."
            ),
            _plain("No results file was saved."),
        ]
        for warning in result.warnings:
            lines.append(_plain(f"Warning: {warning}"))
        _print_section(
            "USAspending results",
            lines,
            style="yellow",
        )
        return

    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    lines = [
        _plain(
            f"Collected {result.result_count} USAspending {result_word} for "
            f"{result.deal_count} {company_word}."
        ),
        _plain(f"Saved the private JSON results file to {result.output_path}."),
        _plain(
            "Only exact recipient-name matches were prepared. Confirm entity identity "
            "before relying on the evidence."
        ),
        _plain(f"Next, run {next_command}"),
    ]
    for warning in result.warnings:
        lines.append(_plain(f"Warning: {warning}"))
    zero_result_companies = [deal.company_name for deal in result.deals if deal.result_count == 0]
    if zero_result_companies:
        lines.append(
            _plain(
                f"No exact USAspending matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            )
        )
    _print_section("USAspending results collected", lines, style="green")


@app.command("collect-sbir-awards")
def collect_sbir_awards_command(
    company: Annotated[
        list[str] | None,
        typer.Option(
            "--company",
            help=(
                "Company to search for in SBIR/STTR awards. Use more than once "
                "for multiple companies."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=25,
            help="Maximum SBIR/STTR award records to request per company.",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be sent to SBIR/STTR without contacting the API.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write the private results file.",
        ),
    ] = None,
) -> None:
    """Collect public SBIR/STTR award evidence for exact firm-name matches."""

    config = _config_from_options(data_dir)
    try:
        result = collect_sbir_awards(
            config=config,
            company_names=company or [],
            limit=limit,
            dry_run=dry_run,
        )
    except ResearchCollectionError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    company_word = "company" if result.deal_count == 1 else "companies"
    result_word = "result" if result.result_count == 1 else "results"
    if result.dry_run:
        _print_section(
            "SBIR/STTR preview",
            [
                _plain(
                    f"Dry run: Hail Mary would send {result.deal_count} "
                    f"{company_word} to the SBIR/STTR public API."
                ),
                _plain(
                    f"A live run can request up to {limit} award records per page "
                    f"for up to 20 pages per company while looking for exact matches."
                ),
                _plain("No API requests were sent and no results file was saved."),
            ],
            style="yellow",
        )
        return

    if result.output_path is None:
        lines = [
            _plain(
                f"No exact firm-name SBIR/STTR {result_word} were found "
                f"for {result.deal_count} {company_word}."
            ),
            _plain("No results file was saved."),
        ]
        for warning in result.warnings:
            lines.append(_plain(f"Warning: {warning}"))
        _print_section(
            "SBIR/STTR results",
            lines,
            style="yellow",
        )
        return

    data_dir_option = (
        f" --data-dir {shlex.quote(str(config.data_dir))}" if data_dir is not None else ""
    )
    next_command = (
        f"`hailmary import-research-results {shlex.quote(str(result.output_path))}"
        f"{data_dir_option} --dry-run`."
    )
    lines = [
        _plain(
            f"Collected {result.result_count} SBIR/STTR {result_word} for "
            f"{result.deal_count} {company_word}."
        ),
        _plain(f"Saved the private JSON results file to {result.output_path}."),
        _plain(
            "Only exact firm-name matches were prepared. Confirm entity identity "
            "before relying on the evidence."
        ),
        _plain(f"Next, run {next_command}"),
    ]
    for warning in result.warnings:
        lines.append(_plain(f"Warning: {warning}"))
    zero_result_companies = [deal.company_name for deal in result.deals if deal.result_count == 0]
    if zero_result_companies:
        lines.append(
            _plain(
                f"No exact SBIR/STTR matches were prepared for: "
                f"{', '.join(zero_result_companies)}."
            )
        )
    _print_section("SBIR/STTR results collected", lines, style="green")


@app.command("prepare-meridian-workflow")
def prepare_meridian_workflow_command(
    company: Annotated[
        str,
        typer.Option(
            "--company",
            help="Company to prepare a Meridian manual workflow for.",
        ),
    ],
    meridian_url: Annotated[
        str,
        typer.Option(
            "--meridian-url",
            help=(
                "Authenticated Meridian deal URL. Hail Mary records it for manual use "
                "and does not open it."
            ),
        ),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Where Hail Mary should write private Meridian workflow files.",
        ),
    ] = None,
) -> None:
    """Prepare a private manual workflow for a Meridian deal page."""

    config = _config_from_options(data_dir)
    try:
        result = prepare_meridian_workflow(
            config=config,
            company_name=company,
            meridian_url=meridian_url,
        )
    except MeridianWorkflowError as exc:
        _print_error(str(exc))
        raise typer.Exit(1) from None

    next_command = f"`{result.workflow.dry_run_command}`."
    _print_section(
        "Meridian workflow prepared",
        [
            _plain(f"Prepared a Meridian manual workflow for {result.workflow.company_name}."),
            _plain(f"Saved the private workflow to {result.output_path}."),
            _plain(f"Saved the fillable results template to {result.result_template_path}."),
            _plain(
                "Hail Mary did not open Meridian, sign in, bypass access controls, "
                "scrape pages, or save portal content."
            ),
            _plain(
                "Use normal authenticated access and paste only short allowed evidence "
                "snippets into the template, not screenshots, raw page dumps, hidden "
                "page data, browser profiles, cookies, tokens, signed URLs, or "
                "unrelated account data."
            ),
            _plain(
                "Review the before-import checklist in the workflow before running "
                "the dry run."
            ),
            _plain(f"After filling the template, run {next_command}"),
        ],
        style="green",
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
        _print_error(str(exc))
        raise typer.Exit(1) from None

    record_word = "record" if result.imported_count == 1 else "records"
    deal_word = "deal" if result.deal_count == 1 else "deals"
    if result.dry_run:
        result_lines = [
            _plain(
                f"Dry run: {result.imported_count} external research evidence {record_word} "
                f"would be imported into {result.deal_count} {deal_word}."
            )
        ]
        border_style = "yellow"
    else:
        result_lines = [
            _plain(
                f"Imported {result.imported_count} external research evidence {record_word} "
                f"into {result.deal_count} {deal_word}."
            )
        ]
        border_style = "green"
    if result.skipped_duplicate_count:
        duplicate_word = "record" if result.skipped_duplicate_count == 1 else "records"
        result_lines.append(
            _plain(f"Skipped {result.skipped_duplicate_count} duplicate {duplicate_word}.")
        )
    if result.skipped_blank_template_row_count:
        row_word = "row" if result.skipped_blank_template_row_count == 1 else "rows"
        result_lines.append(
            _plain(
                f"Skipped {result.skipped_blank_template_row_count} untouched template {row_word}."
            )
        )
    if result.dry_run:
        for deal in result.deals:
            if deal.imported_count:
                deal_record_word = "record" if deal.imported_count == 1 else "records"
                result_lines.append(
                    _plain(
                        f"- {deal.company_name}: would add {deal.imported_count} "
                        f"{deal_record_word}."
                    )
                )
        result_lines.append(_plain("No evidence stores were changed."))
        result_lines.append(_plain("No websites or APIs were contacted."))
        _print_panel("Research import preview", result_lines, border_style=border_style)
        return
    if result.updated_store_paths:
        store_word = "store" if len(result.updated_store_paths) == 1 else "stores"
        result_lines.append(
            _plain(f"Updated {len(result.updated_store_paths)} evidence {store_word}.")
        )
        for deal in result.deals:
            if deal.imported_count:
                deal_record_word = "record" if deal.imported_count == 1 else "records"
                result_lines.append(
                    _plain(
                        f"- {deal.company_name}: added {deal.imported_count} {deal_record_word}."
                    )
                )
    else:
        result_lines.append(_plain("No new evidence records were added."))
    result_lines.append(_plain("No websites or APIs were contacted."))
    _print_panel("Research results imported", result_lines, border_style=border_style)


def _parse_eval_categories(raw_categories: list[str]) -> list[EvalCategory]:
    categories: list[EvalCategory] = []
    valid_values = ", ".join(category.value for category in EvalCategory)
    for raw_category in raw_categories:
        try:
            categories.append(EvalCategory(raw_category))
        except ValueError as exc:
            raise EvalHarnessError(
                f"Unknown eval category {raw_category!r}. Valid categories are: {valid_values}."
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
