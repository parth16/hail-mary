# Hail Mary

Hail Mary is a local-first personal diligence project for evaluating private startup investment opportunities from pitch decks, platform deal pages, legal documents, memos, spreadsheets, and optional web research.

The current repository state is a local-first diligence scaffold with ingestion, evidence validation, deterministic scoring, memo generation, and structured packet preparation for later model review. The implementation should continue in phases toward richer local analysis and optional external data adapters.

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

The current build includes:

```bash
uv sync
uv run hailmary init
uv run hailmary ingest-folder ./pitch-decks
uv run hailmary score-deals
uv run hailmary prepare-agent-packets
uv run hailmary validate-agent-output output.json packet.json
uv run hailmary run-evals
```

`init` creates ignored local folders for generated files. `ingest-folder` scans local deal folders, groups documents by company folder, extracts text and tables when supported, writes per-document JSON, builds a source-linked evidence store, extracts basic deal-term claims, and saves a JSON summary under `data/processed/`. `score-deals` reads the local evidence stores and writes deterministic Markdown memos under `data/reports/`.

`prepare-agent-packets` writes local JSON packets under `data/agent-packets/` for structured model review. These packets include selected evidence excerpts, allowed evidence IDs, verified claims, deterministic score context, and the required output schema. `validate-agent-output` checks a model's JSON output against the packet, rejecting invented evidence IDs, unsupported findings that are not marked unsupported, and quotes that do not appear in the cited evidence record.

`run-evals` runs local synthetic correctness checks. The built-in evals cover text extraction and ingestion, citation span validation, conflicting deal terms, prompt-injection safeguards, and score calibration. They do not use real deal documents.

## GitHub Workflow

The first safe build may be pushed directly to `main`.

After the first push:

- create branches named `codex/<short-description>`
- open PRs as ready for review, not drafts
- self-review changes before pushing
- include validation commands, test results, privacy notes, and known limitations in PR bodies
- when GitHub Codex automatic reviews are enabled, rely on the automatic review trigger; comment `@codex review` only if the trigger does not run and an immediate manual review is needed
- never stage raw investment docs or generated confidential output

After publishing a PR, monitor Codex review activity every minute. An eyes reaction means Codex has started reviewing. If Codex leaves actionable comments, address them, run the relevant tests, push the fixes, and keep monitoring. Cap automatic review loops at five. After the fifth loop, run another loop only for P1 feedback; otherwise merge and move ahead.

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
7. structured local packets for later model review

No paid data source is required for the MVP. External data should be added later through optional provider adapters.

## Not Advice

Hail Mary is a diligence aid. It does not provide legal, tax, financial, or investment advice, and it must never place investments or take irreversible actions.
