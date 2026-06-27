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

## Operator Commands

For direct terminal use from any directory, add the wrapper to `~/.zshrc` before
running Hail Mary commands:

```bash
export HAILMARY_ROOT="$HOME/Documents/hail-mary"
hailmary() { "$HAILMARY_ROOT/bin/hailmary" "$@"; }
```

Then sync the project environment and run commands directly:

```bash
uv sync
hailmary evaluate-deal ./pitch-decks/ExampleCo
hailmary evaluate-deal ./pitch-decks/ExampleCo --enable-ocr
hailmary review-evidence
hailmary evidence-actions needs-review --deal-id exampleco --evidence-id ev_example --note "Check source before relying on this."
```

`evaluate-deal` is the main operator-facing command. It evaluates one company folder
end to end: local privacy checks, ingestion, optional external research workflow,
evidence import, deterministic scoring, optional model review, final guardrails, and
a final memo. Deterministic scoring means fixed rules applied to source-linked
evidence. Guardrails mean Hail Mary keeps the final decision inside the allowed
`INVEST` or `PASS` choices and the allowed check sizes.

`evaluate-deal --enable-ocr` turns on local image-based text reading. OCR means
reading text from images. This can extract text from standalone PNG/JPG files and
from PDF pages that look empty or image-backed. Hail Mary uses local `tesseract` and
Poppler `pdftoppm` commands when they are available on `PATH`; it does not call cloud
OCR services.

`review-evidence` reads ignored local evidence stores and shows plain-English
evidence health summaries by source document, source kind, claim status, source
freshness, materiality, confidence, source lineage, conflicts, image-based text
reading, source spans, and citation gaps without printing confidential evidence text
by default. Health issues are labeled as `blocking`, `warning`, or `info` so
operators know what must be fixed before trusting a memo. The command is read-only;
use `--show-text` or `--quote-limit` only when you intentionally want short local
excerpts.

`evidence-actions` records local review actions for evidence records and claims:
usable, approved, excluded, or needs review. Action files live under the private
generated data folder and store IDs, timestamps, status, and optional operator notes.
They do not copy source text. Operator notes are hidden by default; use `--show-notes`
only when you intentionally want to view them locally. `evaluate-deal` honors excluded
evidence before scoring and model review, and surfaces needs-review actions in warnings
and memo limitations.

### Internal Commands

Hail Mary also keeps internal maintenance commands for testing individual pipeline
stages, portfolio ledger maintenance, research fixtures, synthetic evals, and model
packet debugging. They are hidden from top-level help and are not the normal operator
workflow, but they remain callable by exact command name for development and recovery
work.

## GitHub Workflow

The first safe build may be pushed directly to `main`.

After the first push:

- create branches named `codex/<short-description>`
- open PRs as ready for review, not drafts
- after modifying code, push the `codex/<short-description>` branch and open a ready-for-review PR before ending the session
- self-review changes before pushing
- include validation commands, test results, privacy notes, and known limitations in PR bodies
- when GitHub Codex automatic reviews are enabled, rely on the automatic review trigger; comment `@codex review` only if the trigger does not run and an immediate manual review is needed
- never stage raw investment docs or generated confidential output

After publishing a PR, start a background monitor that checks every minute for Codex review activity, CI status, test failures, and merge conflicts. An eyes reaction means Codex has started reviewing. If Codex leaves actionable comments, CI fails, tests fail, or merge conflicts appear, address them automatically, run the relevant tests, push the fixes, and keep monitoring. Cap Codex feedback loops at five. After the fifth loop, run another loop only for outstanding P1 feedback; otherwise, if CI and mergeability are clean, merge and move ahead.

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
