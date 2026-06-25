from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hailmary.evaluation as evaluation
from hailmary.cli import app
from hailmary.config import AppConfig
from hailmary.evaluation import EvaluationError, evaluate_deal_folder, openai_review_messages
from hailmary.schemas.agents import (
    AgentEvidenceReference,
    AgentFinding,
    AgentInputPacket,
    AgentRecommendationRationale,
    AgentReviewOutput,
    AgentRole,
    AgentSummaryPoint,
    AgentValidationIssue,
)
from hailmary.schemas.scoring import ConfidenceLevel, Recommendation

runner = CliRunner()


ReviewFactory = Callable[[AgentInputPacket], str]


class RecordingReviewClient:
    def __init__(
        self,
        outputs_by_role: dict[AgentRole, list[str | ReviewFactory]] | None = None,
    ) -> None:
        self.outputs_by_role = {
            role: list(outputs) for role, outputs in (outputs_by_role or {}).items()
        }
        self.calls: list[tuple[AgentInputPacket, tuple[AgentValidationIssue, ...], str]] = []
        self.request_payloads: list[str] = []

    def create_review(
        self,
        packet: AgentInputPacket,
        *,
        repair_issues: Sequence[AgentValidationIssue] = (),
        committee_context: str | None = None,
    ) -> str:
        repair_tuple = tuple(repair_issues)
        self.calls.append((packet, repair_tuple, committee_context or ""))
        self.request_payloads.append(
            json.dumps(
                openai_review_messages(
                    packet,
                    repair_issues=repair_tuple,
                    committee_context=committee_context,
                ),
                sort_keys=True,
            )
        )
        outputs = self.outputs_by_role.get(packet.agent_role)
        if outputs:
            output = outputs.pop(0)
            if callable(output):
                return output(packet)
            return output
        return _valid_output_json(packet)


class FakeOpenAIReviewClient(RecordingReviewClient):
    instances: list[FakeOpenAIReviewClient] = []

    def __init__(self, *, model: str, api_key: str) -> None:
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.instances.append(self)


def test_evaluate_deal_command_succeeds_with_mocked_openai_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    FakeOpenAIReviewClient.instances = []
    monkeypatch.setattr(evaluation, "OpenAIAgentReviewClient", FakeOpenAIReviewClient)
    company_dir = _write_company_folder(tmp_path, company_name="ExampleCo", include_long_tail=True)
    local_path_text = str(company_dir / "memo.txt")

    result = runner.invoke(
        app,
        [
            "evaluate-deal",
            str(company_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--max-concurrency",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1. local setup and privacy checks" in result.output
    assert "7. final memo write" in result.output
    assert "Deal evaluation complete" in result.output
    assert "Recommendation" in result.output
    assert "Check size" in result.output
    assert "Final memo" in result.output

    memo_paths = list((tmp_path / "data" / "reports").glob("*-final-evaluation.md"))
    assert len(memo_paths) == 1
    memo_text = memo_paths[0].read_text(encoding="utf-8")
    assert memo_text.startswith("# Hail Mary Final Evaluation: ExampleCo\n\n## Decision")
    assert "**Recommendation:** INVEST" in memo_text
    assert "**Suggested check:**" in memo_text
    assert "**Score:**" in memo_text
    assert "**Confidence:**" in memo_text
    assert "**One-line reason:**" in memo_text
    assert "**Round / Instrument:**" in memo_text
    assert "**Valuation / Cap:**" in memo_text

    assert FakeOpenAIReviewClient.instances
    fake_client = FakeOpenAIReviewClient.instances[0]
    assert fake_client.model == "gpt-test"
    assert fake_client.api_key == "test-openai-key"
    serialized_payloads = "\n".join(fake_client.request_payloads)
    assert "Valuation cap $8M" in serialized_payloads
    assert "PRIVATE_FULL_TEXT_MARKER_AT_END" not in serialized_payloads
    assert local_path_text not in serialized_payloads
    assert "input_file" not in serialized_payloads


@pytest.mark.parametrize(
    ("env_updates", "config", "expected_message"),
    [
        (
            {"HAILMARY_LLM_PROVIDER": None},
            AppConfig(data_dir=Path("data"), local_only=False, mock_llm=False),
            "HAILMARY_LLM_PROVIDER is missing",
        ),
        (
            {"HAILMARY_LLM_PROVIDER": "anthropic"},
            AppConfig(data_dir=Path("data"), local_only=False, mock_llm=False),
            "HAILMARY_LLM_PROVIDER must be openai",
        ),
        (
            {"HAILMARY_MODEL": None},
            AppConfig(data_dir=Path("data"), local_only=False, mock_llm=False),
            "HAILMARY_MODEL is missing",
        ),
        (
            {"OPENAI_API_KEY": None},
            AppConfig(data_dir=Path("data"), local_only=False, mock_llm=False),
            "OPENAI_API_KEY is missing",
        ),
        (
            {},
            AppConfig(data_dir=Path("data"), local_only=True, mock_llm=False),
            "HAILMARY_LOCAL_ONLY must be false",
        ),
        (
            {},
            AppConfig(data_dir=Path("data"), local_only=False, mock_llm=True),
            "HAILMARY_MOCK_LLM must be false",
        ),
    ],
)
def test_evaluate_deal_preflight_settings_fail_before_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_updates: dict[str, str | None],
    config: AppConfig,
    expected_message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    for name, value in env_updates.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient()

    with pytest.raises(EvaluationError, match=expected_message):
        evaluate_deal_folder(
            company_dir,
            config=config,
            model_client=client,
            max_concurrency=1,
        )

    assert client.calls == []


def test_evaluate_deal_specialist_validation_failure_retries_then_records_limitation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient(
        outputs_by_role={
            AgentRole.TEAM: [_unknown_evidence_output_json, _unknown_evidence_output_json]
        }
    )

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    team_calls = [call for call in client.calls if call[0].agent_role == AgentRole.TEAM]
    assert len(team_calls) == 2
    assert team_calls[1][1]
    assert any("Team failed validation" in warning for warning in result.warnings)
    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "Team failed validation after one repair attempt" in memo_text
    assert len(list(result.agent_output_dir.glob("team-attempt-*-invalid.json"))) == 2


def test_evaluate_deal_final_decision_validation_failure_does_not_write_final_memo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient(
        outputs_by_role={
            AgentRole.FINAL_DECISION: [
                _unknown_evidence_output_json,
                _unknown_evidence_output_json,
            ]
        }
    )

    with pytest.raises(EvaluationError, match="final-decision review did not pass"):
        evaluate_deal_folder(
            company_dir,
            config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
            model_client=client,
            max_concurrency=1,
        )

    assert list((tmp_path / "data" / "reports").glob("*-final-evaluation.md")) == []


def test_evaluate_deal_deterministic_pass_overrides_model_invest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(
        tmp_path,
        body=(
            "Round size $1M. Discount 20%. ARR revenue growth with paid customers "
            "and retention. Lead investor committed and seed round is active."
        ),
    )
    client = RecordingReviewClient(
        outputs_by_role={AgentRole.FINAL_DECISION: [_invest_output_json]}
    )

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    assert result.deterministic_score.recommendation == Recommendation.PASS
    assert result.final_recommendation.recommendation == Recommendation.PASS
    assert result.final_recommendation.check_size == 0
    assert any("forced final PASS" in warning for warning in result.warnings)
    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "**Recommendation:** PASS" in memo_text
    assert "Deterministic scoring forced PASS" in memo_text
    assert "Valuation cap" not in result.final_recommendation.reason


def test_evaluate_deal_clamps_final_invest_check_to_deterministic_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient(
        outputs_by_role={AgentRole.FINAL_DECISION: [_invest_output_json]}
    )

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(
            data_dir=tmp_path / "data",
            local_only=False,
            mock_llm=False,
            min_check=5_000,
        ),
        model_client=client,
        max_concurrency=1,
    )

    assert result.deterministic_score.recommendation == Recommendation.INVEST
    assert result.deterministic_score.check_size == 5_000
    assert result.final_recommendation.recommendation == Recommendation.INVEST
    assert result.final_recommendation.check_size == 5_000
    assert any("deterministic allocation" in warning for warning in result.warnings)


def test_evaluate_deal_final_memo_includes_conflict_evidence_for_forced_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(
        tmp_path,
        body=(
            "Valuation cap $8M. Valuation cap $10M. Discount 20%. Round size $1M. "
            "ARR revenue growth with paid customers and retention. "
            "Lead investor committed and seed round is active."
        ),
    )
    client = RecordingReviewClient(
        outputs_by_role={AgentRole.FINAL_DECISION: [_invest_output_json]}
    )

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    assert result.final_recommendation.recommendation == Recommendation.PASS
    assert result.final_recommendation.evidence
    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "Conflicting material deal terms" in memo_text
    assert "Valuation cap $8M" in memo_text
    assert "Valuation cap $10M" in memo_text


def test_evaluate_deal_no_evidence_writes_pass_memo_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = tmp_path / "EmptyCo"
    company_dir.mkdir()
    (company_dir / "empty.txt").write_text("", encoding="utf-8")
    client = RecordingReviewClient()

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    assert client.calls == []
    assert result.final_recommendation.recommendation == Recommendation.PASS
    assert result.final_recommendation.check_size == 0
    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "**Recommendation:** PASS" in memo_text
    assert "NEEDS\\_DILIGENCE: No usable source-linked evidence was available" in memo_text
    assert "skipped model committee review" in memo_text


def test_evaluate_deal_renders_final_decision_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient(
        outputs_by_role={AgentRole.FINAL_DECISION: [_final_finding_output_json]}
    )

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "Final caveat: Validate customer concentration before wiring funds." in memo_text


def test_evaluate_deal_final_memo_includes_source_document_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    company_dir = _write_company_folder(tmp_path)
    client = RecordingReviewClient()

    result = evaluate_deal_folder(
        company_dir,
        config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
        model_client=client,
        max_concurrency=1,
    )

    memo_text = result.final_memo_path.read_text(encoding="utf-8")
    assert "document: memo.txt" in memo_text
    assert "source kind: local_file" in memo_text


def test_evaluate_deal_rejects_collection_folder_with_multiple_deals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    root = tmp_path / "pitch-decks"
    _write_company_folder(root, company_name="OneCo")
    _write_company_folder(root, company_name="TwoCo")
    client = RecordingReviewClient()

    with pytest.raises(EvaluationError, match="produced 2 deals"):
        evaluate_deal_folder(
            root,
            config=AppConfig(data_dir=tmp_path / "data", local_only=False, mock_llm=False),
            model_client=client,
            max_concurrency=1,
        )

    assert client.calls == []


def test_evaluate_deal_missing_provider_cli_error_has_no_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _set_openai_env(monkeypatch)
    monkeypatch.delenv("HAILMARY_LLM_PROVIDER", raising=False)
    company_dir = _write_company_folder(tmp_path)

    result = runner.invoke(
        app,
        [
            "evaluate-deal",
            str(company_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code != 0
    assert "HAILMARY_LLM_PROVIDER is missing" in result.output
    assert "Traceback" not in result.output


def _set_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAILMARY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HAILMARY_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("HAILMARY_LOCAL_ONLY", "false")
    monkeypatch.setenv("HAILMARY_MOCK_LLM", "false")


def _write_company_folder(
    root: Path,
    *,
    company_name: str = "AcmeCo",
    body: str | None = None,
    include_long_tail: bool = False,
) -> Path:
    company_dir = root / company_name
    company_dir.mkdir(parents=True)
    text = body or (
        "Valuation cap $8M. Discount 20%. Round size $1M. "
        "ARR revenue growth with paid customers and retention. "
        "Lead investor committed and seed round is active."
    )
    if include_long_tail:
        text = f"{text} " + ("filler " * 500) + "PRIVATE_FULL_TEXT_MARKER_AT_END"
    (company_dir / "memo.txt").write_text(text, encoding="utf-8")
    return company_dir


def _valid_output(packet: AgentInputPacket) -> AgentReviewOutput:
    evidence = packet.evidence[0]
    quote = _quote(evidence.text)
    reference = AgentEvidenceReference(evidence_id=evidence.id, quote=quote)
    recommendation = None
    if packet.agent_role == AgentRole.FINAL_DECISION:
        if packet.score.recommendation == Recommendation.INVEST:
            recommendation = AgentRecommendationRationale(
                recommendation=Recommendation.INVEST,
                check_size=packet.score.check_size,
                reason=(
                    "The packet has enough cited evidence to support the deterministic "
                    "invest case."
                ),
                evidence=[reference],
            )
        else:
            recommendation = AgentRecommendationRationale(
                recommendation=Recommendation.PASS,
                check_size=0,
                reason="The deterministic score or kill gates require a pass.",
                evidence=[reference],
            )
    return AgentReviewOutput(
        deal_id=packet.deal_id,
        company_name=packet.company_name,
        agent_role=packet.agent_role,
        summary=[
            AgentSummaryPoint(
                summary=f"{packet.agent_role} reviewed source-linked evidence.",
                evidence=[reference],
            )
        ],
        findings=[
            AgentFinding(
                title="Source-backed review",
                finding="The review used the selected packet evidence only.",
                confidence=ConfidenceLevel.MEDIUM,
                materiality="medium",
                evidence=[reference],
            )
        ],
        limitations=["Synthetic fixture output."],
        recommendation=recommendation,
    )


def _valid_output_json(packet: AgentInputPacket) -> str:
    return _valid_output(packet).model_dump_json()


def _unknown_evidence_output_json(packet: AgentInputPacket) -> str:
    return AgentReviewOutput(
        deal_id=packet.deal_id,
        company_name=packet.company_name,
        agent_role=packet.agent_role,
        summary=[
            AgentSummaryPoint(
                summary="This output cites an invented evidence record.",
                evidence=[AgentEvidenceReference(evidence_id="ev_missing")],
            )
        ],
    ).model_dump_json()


def _invest_output_json(packet: AgentInputPacket) -> str:
    evidence = packet.evidence[0]
    reference = AgentEvidenceReference(evidence_id=evidence.id, quote=_quote(evidence.text))
    return AgentReviewOutput(
        deal_id=packet.deal_id,
        company_name=packet.company_name,
        agent_role=packet.agent_role,
        summary=[
            AgentSummaryPoint(
                summary="The model tries to invest despite deterministic gates.",
                evidence=[reference],
            )
        ],
        recommendation=AgentRecommendationRationale(
            recommendation=Recommendation.INVEST,
            check_size=1_000,
            reason="The model says invest, but deterministic gates should override it.",
            evidence=[reference],
        ),
    ).model_dump_json()


def _final_finding_output_json(packet: AgentInputPacket) -> str:
    evidence = packet.evidence[0]
    reference = AgentEvidenceReference(evidence_id=evidence.id, quote=_quote(evidence.text))
    return AgentReviewOutput(
        deal_id=packet.deal_id,
        company_name=packet.company_name,
        agent_role=packet.agent_role,
        findings=[
            AgentFinding(
                title="Final caveat",
                finding="Validate customer concentration before wiring funds.",
                confidence=ConfidenceLevel.MEDIUM,
                materiality="high",
                evidence=[reference],
            )
        ],
        recommendation=AgentRecommendationRationale(
            recommendation=Recommendation.INVEST,
            check_size=packet.score.check_size,
            reason="The final decision is supported by source-linked packet evidence.",
            evidence=[reference],
        ),
    ).model_dump_json()


def _quote(text: str) -> str:
    first_sentence = text.split(".", 1)[0].strip()
    return first_sentence or text[:40]
