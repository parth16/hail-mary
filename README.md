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
hailmary ingest-folder ./pitch-decks --enable-ocr
hailmary review-evidence
hailmary score-deals
hailmary prepare-agent-packets
hailmary validate-agent-output output.json packet.json
hailmary run-evals
hailmary list-research-providers
hailmary prepare-research-plan --company "ExampleCo"
HAILMARY_LOCAL_ONLY=false HAILMARY_ENABLE_WEB_RESEARCH=true \
  hailmary collect-web-research --dry-run
hailmary prepare-research-results-template
hailmary prepare-public-research-results \
  --company "ExampleCo" \
  --sec-form-d-results sec-form-d-results.json \
  --sam-gov-results sam-gov-results.json
HAILMARY_LOCAL_ONLY=false HAILMARY_ENABLE_WEB_RESEARCH=true \
  hailmary collect-usaspending-awards --company "ExampleCo" --dry-run
HAILMARY_LOCAL_ONLY=false HAILMARY_ENABLE_WEB_RESEARCH=true \
  hailmary collect-sbir-awards --company "ExampleCo" --dry-run
HAILMARY_LOCAL_ONLY=false HAILMARY_ENABLE_WEB_RESEARCH=true \
  hailmary collect-sec-form-d-filings --company "ExampleCo" --dry-run
HAILMARY_LOCAL_ONLY=false HAILMARY_ENABLE_WEB_RESEARCH=true \
  hailmary collect-github-repositories --company "ExampleCo" --dry-run
hailmary prepare-meridian-workflow \
  --company "ExampleCo" \
  --meridian-url "https://portal.angellist.com/m/example/invest"
hailmary import-research-results research-results.json --dry-run
hailmary import-research-results research-results.json
```

`init` creates ignored local folders for generated files. `ingest-folder` scans local deal folders, groups documents by company folder, extracts text and tables when supported, writes per-document JSON, builds a source-linked evidence store, extracts basic deal-term claims, and saves a JSON summary under `data/processed/`. `review-evidence` reads those ignored local evidence stores and shows plain-English summaries of evidence records, claim status, conflicts, OCR use, source freshness, source spans, and citation gaps without printing confidential evidence text by default. Use `--show-text` or `--quote-limit` only when you intentionally want short local excerpts. `score-deals` reads the local evidence stores and writes deterministic Markdown memos under `data/reports/`.

`ingest-folder --enable-ocr` turns on local image-based text reading (OCR). OCR means reading text from images. This can extract text from standalone PNG/JPG files and from PDF pages that look empty or image-backed. Hail Mary uses local `tesseract` and Poppler `pdftoppm` commands when they are available on `PATH`; it does not call cloud OCR services. If those commands are missing or a page cannot be read, ingestion keeps the current OCR-needed warning, saves plain-English notes in the private generated metadata, and continues without crashing. You can also set `HAILMARY_ENABLE_OCR=true` or `enable_ocr: true` in `.hailmary/config.yaml`.

`prepare-agent-packets` writes local JSON packets under `data/agent-packets/` for structured model review. The default committee uses focused product/customer traction, market/competition, team/execution, financing/next-round risk, and final-decision roles. Packets include selected evidence excerpts, allowed evidence IDs, verified claims, deterministic score context, conflicts, limitations, and the required output schema. `validate-agent-output` checks a model's JSON output against the packet, rejecting invented evidence IDs, specialist recommendations, unsupported findings that are not marked unsupported, and quotes that do not appear in the cited evidence record.

`run-evals` runs local synthetic correctness checks. The built-in evals cover text extraction and ingestion, citation span validation, conflicting deal terms, prompt-injection safeguards, missing evidence, score calibration, and Markdown memo snapshot checks. They do not use real deal documents.

`list-research-providers` shows free public, authenticated, and optional paid sources that Hail Mary can plan around. `prepare-research-plan` writes a private JSON checklist under `data/research-plans/` from either the latest ingestion summary or manually supplied `--company` values. It does not contact websites, APIs, paid databases, or Meridian. Any external fact imported later must record the provider, timestamp, exact URL or API source, confidence, and licensing notes.

`collect-web-research` can fetch direct public web-page tasks from a research plan after you explicitly turn off local-only mode and enable web research with `HAILMARY_LOCAL_ONLY=false` and `HAILMARY_ENABLE_WEB_RESEARCH=true`. Start with `--dry-run` to see which public URLs would be fetched. The command skips paid, authenticated, Meridian, local-only, missing-URL, generated search-result, localhost, and private-network sources. It writes exact source pages to a private JSON file under `data/research-results/`; run `import-research-results --dry-run` before adding that text to evidence stores.

`prepare-research-results-template` turns a private research plan into a fillable JSON file under `data/research-results-templates/`. It copies company names, provider IDs, source kinds, licensing notes, and ingested deal IDs when available, but leaves fact fields and citation URLs blank so the file cannot be mistaken for validated evidence.

`prepare-public-research-results` normalizes local public-source JSON files into an import-ready results file under `data/research-results/`. It supports local files for SEC Form D, SAM.gov, USAspending, SBIR/STTR, USPTO, and GitHub. It only imports exact company-name matches and checks that each result URL or URL-like API source belongs to the expected source. It does not fetch websites, browse pages, call software data feeds, or use paid data. A minimal input file for any supported source looks like:

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

`collect-sec-form-d-filings` is an optional live public collector for SEC EDGAR Form D filings. It requires `HAILMARY_LOCAL_ONLY=false`, `HAILMARY_ENABLE_WEB_RESEARCH=true`, and `HAILMARY_SEC_USER_AGENT` set to an application or company name plus a contact email address. Start with `--dry-run`; a real run sends only the `--company` values you provide to SEC EDGAR, keeps only exact issuer-name matches for Form D or amended Form D filings, saves parsed filing metadata, and writes a private JSON results file under `data/research-results/`. Hail Mary does not save raw SEC filings, contact people, phone numbers, email addresses, or addresses.

`collect-github-repositories` is an optional live public API collector for GitHub repository metadata. It requires `HAILMARY_LOCAL_ONLY=false` and `HAILMARY_ENABLE_WEB_RESEARCH=true`. Start with `--dry-run`; a real run sends only the `--company` values you provide to the GitHub public repository search API, searches repository names plus exact user and organization owner scopes, keeps only repositories where the owner slug or repository slug exactly matches the requested company slug, and writes a private JSON results file under `data/research-results/`. Hail Mary does not clone repositories, fetch code, or fetch README files.

`collect-usaspending-awards` is an optional live public API collector for USAspending award records. It requires `HAILMARY_LOCAL_ONLY=false` and `HAILMARY_ENABLE_WEB_RESEARCH=true`. Start with `--dry-run`; a real run sends only the `--company` values you provide to the USAspending public API, keeps only exact recipient-name matches, and writes a private JSON results file under `data/research-results/`.

SAM.gov and USPTO stay in the manual or local-file workflow in this phase because their official public API documentation requires API keys. Use `prepare-public-research-results` with local JSON files after you manually confirm exact source URLs and licensing notes.

`prepare-meridian-workflow` writes a private Meridian manual workflow under `data/meridian-workflows/` and a fillable Meridian results template under `data/research-results-templates/`. It does not open Meridian, sign in, scrape pages, bypass access controls, save browser profiles, save cookies, save tokens, save signed URLs, save screenshots, save hidden page data, save unrelated account data, or save raw portal pages. Use normal authenticated access in your own browser, manually copy only short allowed facts or excerpts into the generated placeholder rows, and keep the safe Meridian deal URL in `source_url`. The workflow groups required-when-visible facts such as deal terms, customer traction, revenue, team, risks, deadlines, and allocation, plus optional product, market, and use-of-funds facts. It also includes a before-import checklist and a shell-quoted `import-research-results --dry-run` command. The workflow includes plain-English reminders for terms such as SAFE, convertible note, ARR, MRR, allocation, valuation cap, pre-money valuation, discount, minimum investment, target raise, lead investor, and closing date. Use the exact base Meridian deal URL without extra path text, extra slashes, usernames, passwords, unsafe ports, encoded delimiters, `?`, or `#`.

`collect-sbir-awards` is an optional live public API collector for SBIR/STTR award records. It requires `HAILMARY_LOCAL_ONLY=false` and `HAILMARY_ENABLE_WEB_RESEARCH=true`. Start with `--dry-run`; a real run sends only the `--company` values you provide to the SBIR/STTR public API, keeps only exact firm-name matches, saves the API request URL as the source, and writes a private JSON results file under `data/research-results/`. Hail Mary does not save SBIR/STTR contact phone or email fields into generated evidence text.

`import-research-results` reads a local JSON file of manually collected external research and appends validated records to the ignored evidence stores under `data/processed/`. Use `--dry-run` first to validate the file and preview new or duplicate records without writing anything. The command does not fetch websites or call APIs. Each imported result must name the deal by `deal_id` or exact `company_name`, include provider details, `retrieved_at`, either `source_url` or `source_api`, confidence, licensing notes, and the evidence text to cite later. For known public providers such as SEC, SAM.gov, USAspending, SBIR/STTR, USPTO, and GitHub, Hail Mary checks that URL-like source references use that provider's website host. A minimal file looks like:

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
