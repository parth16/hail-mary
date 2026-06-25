# End-to-End Deal Evaluation CLI Plan

## Summary

Add `hailmary evaluate-deal <deal-folder>` as the end-to-end operator command for evaluating one private startup deal from a local pitch-deck folder.

The command should ingest the folder, run deterministic scoring, prepare agent packets, run a full OpenAI-backed diligence committee, validate model outputs against source-linked evidence, and write one final Markdown memo for the operator.

This plan is documentation only. The implementation should happen in a separate PR after review.

## Operator Experience

The primary command should be:

```zsh
hailmary evaluate-deal "$HAILMARY_ROOT/pitch-decks/ExampleCo"
```

The input is one company folder. The folder name becomes the company name for the run. The command should reject inputs that produce zero deals or more than one deal, with a plain-English error that explains how to fix the folder shape.

All operator-facing output should go through Rich. The command should show each major stage:

1. local setup and privacy checks
2. ingestion
3. deterministic scoring
4. agent packet preparation
5. specialist committee review
6. final decision review
7. final memo write

The final terminal output should include the recommendation, check size, score, confidence, final memo path, and any validation warnings.

## LLM Review

Version 1 should support OpenAI only.

The command should require:

- `HAILMARY_LLM_PROVIDER=openai`
- `HAILMARY_MODEL`
- `OPENAI_API_KEY`
- `HAILMARY_LOCAL_ONLY=false`
- `HAILMARY_MOCK_LLM=false`

If any setting is missing or incompatible, the command should fail before making an LLM call and explain the exact setting the operator needs to change.

The command should upload only selected evidence excerpts from generated agent packets, not raw pitch decks or full local source documents.

Use the official OpenAI Python SDK and structured JSON output. The model output must still pass Hail Mary's local `validate_agent_output` checks before it can influence the final memo.

## Committee Workflow

The command should run a full committee review for the single deal.

Specialist role packets should run first, using capped parallelism. Default concurrency should be `--max-concurrency 3`.

Each specialist output should be validated. If validation fails, retry that role once with the validation errors included as repair instructions. If the repaired output still fails, continue the workflow, mark that role as failed, and include the failure as a limitation in the final memo.

After specialist reviews complete, run a final-decision model call. The final-decision output must validate successfully. If it does not validate after one repair attempt, the command should fail and should not write a final memo.

The final decision must remain constrained to:

- recommendation: `INVEST` or `PASS`
- check size: `$0`, `$1K`, `$2.5K`, `$5K`, `$7.5K`, or `$10K`

The model may downgrade any deal to `PASS`. It must not override deterministic kill gates into `INVEST`.

## Outputs

Generated outputs should stay private and ignored by git.

Use these output locations:

- `data/agent-packets/` for generated model input packets
- `data/agent-outputs/<deal-id>/` for validated model JSON and invalid raw attempts
- `data/reports/<deal-id>-final-evaluation.md` for the operator-facing final memo

The implementation must add `data/agent-outputs/` to local-state creation and data-directory validation. It should be created with owner-only permissions and reserved as a first-class generated Hail Mary folder, so a successful run does not make later commands reject `data/` as containing unknown files.

The final memo should combine:

- the existing opening decision box with recommendation, suggested check, score, confidence, one-line reason, and deal terms
- deterministic score and kill gates
- validated specialist findings
- final LLM recommendation
- cited evidence IDs and quotes
- limitations, including failed specialist roles
- diligence questions

The memo must preserve evidence lineage. Material claims must cite allowed evidence IDs or be explicitly labeled `UNVERIFIED`, `INFERRED`, or `NEEDS_DILIGENCE`.

## Data Safety

Do not commit raw pitch decks, local source documents, extracted confidential text, generated reports, agent outputs, `.hailmary`, `.env`, local databases, browser profiles, cookies, tokens, or signed URLs.

The data directory validator should tolerate harmless OS metadata such as `.DS_Store`, so Finder metadata does not block valid Hail Mary workflows.

Source documents remain untrusted input. The model instructions must tell the model not to follow instructions embedded in evidence excerpts, and local validation must reject citations that rely on embedded source-document instructions.

## Tests

Use synthetic fixtures only.

Required tests:

- `evaluate-deal` succeeds for one synthetic company folder with mocked OpenAI responses, and the mocked OpenAI client asserts the request payload contains only selected packet excerpts, not raw deck bytes, full extracted documents, or local source paths.
- Missing or incompatible `HAILMARY_LLM_PROVIDER`, `OPENAI_API_KEY`, `HAILMARY_MODEL`, `HAILMARY_LOCAL_ONLY=false`, or `HAILMARY_MOCK_LLM=false` fails before model calls.
- A specialist validation failure retries once, then continues with a warning and records the failed role as a limitation in the final memo.
- A final-decision validation failure exits without writing a final memo.
- Deterministic kill gates force final `PASS` even when mocked model output tries `INVEST`.
- The final memo starts with the established decision box, including recommendation, suggested check, score, confidence, one-line reason, and deal terms.
- `.DS_Store` inside `data/` does not block valid workflows.
- CLI output uses Rich and does not print tracebacks for expected operator errors.

Before implementation PRs merge, run:

```zsh
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## References

- OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
- OpenAI Python SDK: https://pypi.org/project/openai/
