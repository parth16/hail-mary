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

For direct terminal use from any directory, add the wrapper to `~/.zshrc` before
running Hail Mary commands:

```bash
export HAILMARY_ROOT="$HOME/Documents/hail-mary"
hailmary() { "$HAILMARY_ROOT/bin/hailmary" "$@"; }
```

Then sync the project environment and run commands directly:

```bash
uv sync
hailmary init
hailmary ingest-folder ./pitch-decks
hailmary score-deals
hailmary prepare-agent-packets
hailmary validate-agent-output output.json packet.json
hailmary run-evals
hailmary list-research-providers
hailmary prepare-research-plan --company "ExampleCo"
hailmary prepare-research-results-template
hailmary prepare-public-research-results \
  --company "ExampleCo" \
  --sec-form-d-results sec-form-d-results.json \
  --sam-gov-results sam-gov-results.json
hailmary prepare-meridian-workflow \
  --company "ExampleCo" \
  --meridian-url "https://portal.angellist.com/m/example/invest"
hailmary import-research-results research-results.json --dry-run
hailmary import-research-results research-results.json
```

`init` creates ignored local folders for generated files. `ingest-folder` scans local deal folders, groups documents by company folder, extracts text and tables when supported, writes per-document JSON, builds a source-linked evidence store, extracts basic deal-term claims, and saves a JSON summary under `data/processed/`. `score-deals` reads the local evidence stores and writes deterministic Markdown memos under `data/reports/`.

`prepare-agent-packets` writes local JSON packets under `data/agent-packets/` for structured model review. These packets include selected evidence excerpts, allowed evidence IDs, verified claims, deterministic score context, and the required output schema. `validate-agent-output` checks a model's JSON output against the packet, rejecting invented evidence IDs, unsupported findings that are not marked unsupported, and quotes that do not appear in the cited evidence record.

`run-evals` runs local synthetic correctness checks. The built-in evals cover text extraction and ingestion, citation span validation, conflicting deal terms, prompt-injection safeguards, missing evidence, score calibration, and Markdown memo snapshot checks. They do not use real deal documents.

`list-research-providers` shows free public, authenticated, and optional paid sources that Hail Mary can plan around. `prepare-research-plan` writes a private JSON checklist under `data/research-plans/` from either the latest ingestion summary or manually supplied `--company` values. It does not contact websites, APIs, paid databases, or Meridian. Any external fact imported later must record the provider, timestamp, exact URL or API source, confidence, and licensing notes.

`prepare-research-results-template` turns a private research plan into a fillable JSON file under `data/research-results-templates/`. It copies company names, provider IDs, source kinds, licensing notes, and ingested deal IDs when available, but leaves fact fields and citation URLs blank so the file cannot be mistaken for validated evidence.

`prepare-public-research-results` normalizes local public-source JSON files into an import-ready results file under `data/research-results/`. It supports local files for SEC Form D, SAM.gov, USAspending, SBIR/STTR, USPTO, and GitHub. It only imports exact company-name matches and checks that each result URL belongs to the expected source. It does not fetch websites, browse pages, call software data feeds, or use paid data. A minimal input file for any supported source looks like:

```json
{
  "results": [
    {
      "company_name": "ExampleCo",
      "title": "ExampleCo Form D",
      "text": "ExampleCo filed a Form D for a $1,000,000 offering.",
      "retrieved_at": "2026-01-01T12:00:00Z",
      "source_url": "https://www.sec.gov/example"
    }
  ]
}
```

Use the matching option for each local source file: `--sec-form-d-results`, `--sam-gov-results`, `--usaspending-results`, `--sbir-results`, `--uspto-results`, or `--github-results`.

`prepare-meridian-workflow` writes a private manual workflow under `data/meridian-workflows/` and a fillable Meridian results template under `data/research-results-templates/`. It does not open Meridian, sign in, bypass access controls, or save portal content. Use normal authenticated access, collect only facts you are allowed to save locally, fill in the template with the time viewed and Meridian page URL, then run `import-research-results --dry-run`. Use the base Meridian deal URL without extra text after `?` or `#`.

`import-research-results` reads a local JSON file of manually collected external research and appends validated records to the ignored evidence stores under `data/processed/`. Use `--dry-run` first to validate the file and preview new or duplicate records without writing anything. The command does not fetch websites or call APIs. Each imported result must name the deal by `deal_id` or exact `company_name`, include provider details, `retrieved_at`, either `source_url` or `source_api`, confidence, licensing notes, and the evidence text to cite later. A minimal file looks like:

```json
{
  "results": [
    {
      "company_name": "ExampleCo",
      "provider_id": "sec_form_d",
      "provider_name": "SEC EDGAR Form D search",
      "title": "ExampleCo Form D",
      "text": "ExampleCo filed a Form D for a $1,000,000 offering.",
      "retrieved_at": "2026-01-01T12:00:00Z",
      "source_url": "https://www.sec.gov/example",
      "confidence": "high: exact company match",
      "licensing_notes": "Public government source."
    }
  ]
}
```

For built-in provider IDs, Hail Mary uses the provider's source kind automatically. If
you provide `source_kind`, it must match the built-in provider. `source_api` may be a
plain provider source label or an `http://` or `https://` endpoint, but endpoint URLs
cannot include an inline username or password.

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
