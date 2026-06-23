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

## GitHub Workflow

The first safe build may be pushed directly to `main`.

After the first push:

- create branches named `codex/<short-description>`
- open draft PRs by default
- include validation commands and privacy notes in PR bodies
- never stage raw investment docs or generated confidential output

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
