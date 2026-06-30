from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


def test_live_evaluate_deal_cli_smoke_uses_real_wrapper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "bin" / "hailmary"
    assert wrapper.is_file()

    config_dir = repo_root / ".hailmary"
    exclude_path = _git_exclude_path(repo_root)
    original_exclude = _read_optional_text(exclude_path)

    with tempfile.TemporaryDirectory(
        prefix="evaluate-deal-",
        dir=repo_root.parent,
    ) as raw_work_dir:
        work_dir = Path(raw_work_dir).resolve(strict=True)
        saved_config_dir = work_dir / "saved-hailmary-config"
        config_was_moved = False
        try:
            if config_dir.exists() or config_dir.is_symlink():
                config_dir.rename(saved_config_dir)
                config_was_moved = True
            company_dir = work_dir / "SyntheticLiveCliCo"
            company_dir.mkdir()
            (company_dir / "memo.txt").write_text(
                "Valuation cap $8M. Discount 20%. Round size $1M. "
                "ARR revenue growth with paid customers and retention. "
                "Lead investor committed and seed round is active. "
                "PRIVATE_FULL_TEXT_MARKER_AT_END",
                encoding="utf-8",
            )
            data_dir = work_dir / "data"

            env = _live_cli_env(os.environ, data_dir=data_dir)
            result = subprocess.run(
                [
                    str(wrapper),
                    "evaluate-deal",
                    str(company_dir),
                    "--data-dir",
                    str(data_dir),
                    "--skip-research",
                ],
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            output = result.stdout + result.stderr
            assert result.returncode == 0, output
            assert "Traceback" not in output
            assert "\x1b[" not in output
            assert "Deal evaluation complete" in output
            normalized_output = " ".join(output.split())
            assert "Recommendation" in normalized_output
            assert "Check size" in normalized_output
            assert "Bottom line Final guarded recommendation" in normalized_output
            assert "Evaluation mode: local-only" in normalized_output
            assert "External research was skipped" in normalized_output
            assert "Final memo" in normalized_output
            assert "Final JSON" in normalized_output
            assert "PRIVATE_FULL_TEXT_MARKER_AT_END" not in output
            assert "Valuation cap $8M" not in output
            assert "ARR revenue growth" not in output

            memo_paths = list((data_dir / "reports").glob("*-final-evaluation.md"))
            assert len(memo_paths) == 1
            memo_text = memo_paths[0].read_text(encoding="utf-8")
            assert memo_text.startswith(
                "# Hail Mary Final Evaluation: SyntheticLiveCliCo\n\n## Decision"
            )

            json_paths = list((data_dir / "reports").glob("*-final-evaluation.json"))
            assert len(json_paths) == 1
            export = json.loads(json_paths[0].read_text(encoding="utf-8"))
            assert export["final_decision"]["recommendation"] in {"INVEST", "PASS"}
            assert export["final_decision"]["check_size"] in {
                0,
                1_000,
                2_500,
                5_000,
                7_500,
                10_000,
            }
            assert export["deal"]["evaluation_mode"] == "local-only"
            assert export["privacy"]["contains_raw_evidence_text"] is False
            assert export["privacy"]["contains_model_excerpts"] is False
            assert _read_optional_text(exclude_path) == original_exclude
        finally:
            try:
                if config_was_moved:
                    shutil.rmtree(config_dir, ignore_errors=True)
                    saved_config_dir.rename(config_dir)
                else:
                    shutil.rmtree(config_dir, ignore_errors=True)
            finally:
                _restore_git_exclude(exclude_path, original_exclude)


def _git_exclude_path(repo_root: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--git-path", "info/exclude"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return Path(result.stdout.strip())


def _read_optional_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _restore_git_exclude(path: Path, original_text: str | None) -> None:
    if original_text is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(original_text, encoding="utf-8")


def _live_cli_env(
    base_env: Mapping[str, str],
    *,
    data_dir: Path,
) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "HAILMARY_DATA_DIR": str(data_dir),
            "HAILMARY_MERIDIAN_PROFILE_DIR": str(
                data_dir / "browser-profiles" / "meridian"
            ),
            "HAILMARY_LOCAL_ONLY": "true",
            "HAILMARY_MOCK_LLM": "true",
            "HAILMARY_ENABLE_WEB_RESEARCH": "false",
            "HAILMARY_ENABLE_OCR": "false",
            "HAILMARY_CALCULATED_RISK_MODE": "true",
            "HAILMARY_CAPITAL_BUDGET": "100000",
            "HAILMARY_MIN_CHECK": "1000",
            "HAILMARY_MAX_CHECK": "10000",
            "HAILMARY_RESERVE_PERCENT": "0",
            "HAILMARY_RESERVE_DOLLARS": "0",
            "HAILMARY_ESTIMATED_DILUTION_PERCENT": "0",
            "HAILMARY_PLATFORM_FEE_PERCENT": "0",
            "HAILMARY_CARRY_PERCENT": "0",
            "HAILMARY_GROSS_RETURN_MULTIPLE": "5",
            "HAILMARY_MAX_COMPANY_EXPOSURE_PERCENT": "0",
            "HAILMARY_MAX_CATEGORY_EXPOSURE_PERCENT": "0",
            "HAILMARY_MAX_STAGE_EXPOSURE_PERCENT": "0",
            "HAILMARY_MAX_LOW_CONFIDENCE_EXPOSURE_PERCENT": "0",
            "HAILMARY_MAX_MEDIUM_CONFIDENCE_EXPOSURE_PERCENT": "0",
            "HAILMARY_MAX_HIGH_CONFIDENCE_EXPOSURE_PERCENT": "0",
            "HAILMARY_ENABLED_PAID_PROVIDERS": "",
            "HAILMARY_LLM_SPECIALIST_TOKEN_BUDGET": "100000",
            "HAILMARY_LLM_FINAL_TOKEN_BUDGET": "100000",
            "HAILMARY_LLM_SPECIALIST_MAX_OUTPUT_TOKENS": "1000",
            "HAILMARY_LLM_FINAL_MAX_OUTPUT_TOKENS": "1000",
            "HAILMARY_LLM_INPUT_COST_PER_MILLION_TOKENS_CENTS": "1",
            "HAILMARY_LLM_OUTPUT_COST_PER_MILLION_TOKENS_CENTS": "1",
            "HAILMARY_LLM_SPECIALIST_COST_BUDGET_CENTS": "100000",
            "HAILMARY_LLM_FINAL_COST_BUDGET_CENTS": "100000",
        }
    )
    for name in (
        "HAILMARY_LLM_PROVIDER",
        "HAILMARY_MODEL",
        "OPENAI_API_KEY",
    ):
        env.pop(name, None)
    return env
