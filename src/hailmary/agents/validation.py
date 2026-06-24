from __future__ import annotations

from hailmary.schemas.agents import (
    AgentEvidenceItem,
    AgentEvidenceReference,
    AgentInputPacket,
    AgentReviewOutput,
    AgentRole,
    AgentValidationIssue,
    AgentValidationResult,
    is_allowed_check_size,
)
from hailmary.schemas.scoring import Recommendation


def validate_agent_output(
    output: AgentReviewOutput,
    packet: AgentInputPacket,
) -> AgentValidationResult:
    issues: list[AgentValidationIssue] = []
    evidence_by_id = {evidence.id: evidence for evidence in packet.evidence}

    if output.deal_id != packet.deal_id:
        issues.append(
            AgentValidationIssue(
                location="deal_id",
                message="The output deal ID does not match the agent packet.",
            )
        )
    if output.company_name != packet.company_name:
        issues.append(
            AgentValidationIssue(
                location="company_name",
                message="The output company name does not match the agent packet.",
            )
        )
    if output.agent_role != packet.agent_role:
        issues.append(
            AgentValidationIssue(
                location="agent_role",
                message="The output agent role does not match the agent packet.",
            )
        )
    if (
        not output.findings
        and not output.diligence_questions
        and not output.limitations
        and output.recommendation is None
    ):
        issues.append(
            AgentValidationIssue(
                location="findings",
                message=(
                    "The output needs at least one finding, diligence question, or "
                    "limitation."
                ),
            )
        )
    if packet.agent_role == AgentRole.FINAL_DECISION and output.recommendation is None:
        issues.append(
            AgentValidationIssue(
                location="recommendation",
                message="Final-decision agent output needs an INVEST or PASS recommendation.",
            )
        )

    for finding_index, finding in enumerate(output.findings):
        if not finding.unsupported and not finding.evidence:
            issues.append(
                AgentValidationIssue(
                    location=f"findings[{finding_index}].evidence",
                    message=(
                        "This finding needs at least one evidence ID or must be marked "
                        "unsupported."
                    ),
                )
            )
        if finding.unsupported and finding.score_delta > 0:
            issues.append(
                AgentValidationIssue(
                    location=f"findings[{finding_index}].score_delta",
                    message="Unsupported findings cannot increase the score.",
                )
            )
        for reference_index, reference in enumerate(finding.evidence):
            _validate_evidence_reference(
                reference,
                evidence_by_id=evidence_by_id,
                location=f"findings[{finding_index}].evidence[{reference_index}]",
                issues=issues,
            )

    for question_index, question in enumerate(output.diligence_questions):
        for reference_index, reference in enumerate(question.evidence):
            _validate_evidence_reference(
                reference,
                evidence_by_id=evidence_by_id,
                location=(
                    f"diligence_questions[{question_index}].evidence[{reference_index}]"
                ),
                issues=issues,
            )

    if output.recommendation is not None:
        if not is_allowed_check_size(output.recommendation.check_size):
            issues.append(
                AgentValidationIssue(
                    location="recommendation.check_size",
                    message=(
                        "The suggested check size must be $0, $1K, $2.5K, "
                        "$5K, $7.5K, or $10K."
                    ),
                )
            )
        if (
            output.recommendation.recommendation == Recommendation.INVEST
            and output.recommendation.check_size == 0
        ):
            issues.append(
                AgentValidationIssue(
                    location="recommendation.check_size",
                    message="An INVEST recommendation cannot use a $0 check size.",
                )
            )
        if (
            output.recommendation.recommendation == Recommendation.INVEST
            and not output.recommendation.evidence
        ):
            issues.append(
                AgentValidationIssue(
                    location="recommendation.evidence",
                    message="An INVEST recommendation needs at least one cited evidence ID.",
                )
            )
        for reference_index, reference in enumerate(output.recommendation.evidence):
            _validate_evidence_reference(
                reference,
                evidence_by_id=evidence_by_id,
                location=f"recommendation.evidence[{reference_index}]",
                issues=issues,
            )

    return AgentValidationResult(issues=issues)


def _validate_evidence_reference(
    reference: AgentEvidenceReference,
    *,
    evidence_by_id: dict[str, AgentEvidenceItem],
    location: str,
    issues: list[AgentValidationIssue],
) -> None:
    evidence = evidence_by_id.get(reference.evidence_id)
    if evidence is None:
        issues.append(
            AgentValidationIssue(
                location=location,
                message=f"Unknown evidence ID: {reference.evidence_id}.",
            )
        )
        return

    quote = reference.quote
    if quote is not None and quote not in evidence.text:
        issues.append(
            AgentValidationIssue(
                location=location,
                message="The quoted text was not found in the cited evidence record.",
            )
        )
