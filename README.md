# Hail Mary

Hail Mary is a local-first personal diligence project for evaluating private startup investment opportunities from pitch decks, platform deal pages, legal documents, memos, spreadsheets, and optional web research.

The current repository state is the build prompt and workflow scaffold. The implementation should proceed in phases from repository hygiene and local ingestion toward evidence validation, deterministic scoring, report generation, and finally optional LLM agents and external data adapters.

## Privacy First

This repository may sit beside confidential investment materials, but those materials must not be committed.

Ignored by default:

- `pitch-decks/`
- `pitch-decks.zip`
- `data/`
- `.hailmary/`
- `.env`
- browser profiles, cookies, generated reports, extracted text, and local databases
- `.DS_Store`

Committed tests should use synthetic fixtures only.

## Build Prompt

The implementation prompt is in `hail-mary-codex-prompt.md`. It defines:

- local-first architecture
- evidence and citation discipline
- stage-aware underwriting
- investment scoring and check sizing
- phased delivery plan
- GitHub branch and PR workflow
- optional future data-provider adapters

## Current Commands

This first build includes:

```bash
uv sync
uv run hailmary init
uv run hailmary ingest-folder ./pitch-decks
```

`init` creates ignored local folders for generated files. `ingest-folder` scans local deal folders, groups documents by company folder, extracts basic text when supported, and saves a JSON summary under `data/processed/`.

## GitHub Workflow

The first safe build may be pushed directly to `main`.

After the first push:

- create branches named `codex/<short-description>`
- open PRs as ready for review, not drafts
- self-review changes before pushing
- include validation commands, test results, privacy notes, and known limitations in PR bodies
- never stage raw investment docs or generated confidential output

## Product Standards

All operator-facing output should use plain English. Avoid unnecessary jargon; if a finance, legal, or technical term is needed, explain it the first time it appears.

Testing is a first-class part of the project. New code should include tests for logical and functional correctness, including success paths, important edge cases, and failure modes. User-facing commands should be robust, handle invalid input deliberately, and explain failures clearly.

## MVP Direction

The first implementation phases should focus on:

1. repository hygiene and local-only privacy guardrails
2. CLI scaffold and config
3. local folder ingestion
4. deterministic PDF/DOCX extraction
5. evidence store and citation validation
6. deterministic scoring and Markdown memos

No paid data source is required for the MVP. External data should be added later through optional provider adapters.

## Not Advice

Hail Mary is a diligence aid. It does not provide legal, tax, financial, or investment advice, and it must never place investments or take irreversible actions.
