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
from hailmary.utils.source_instructions import looks_like_embedded_source_instruction


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
        not output.summary
        and not output.findings
        and not output.diligence_questions
        and not output.limitations
        and output.recommendation is None
    ):
        issues.append(
            AgentValidationIssue(
                location="summary",
                message=(
                    "The output needs at least one summary point, finding, diligence "
                    "question, limitation, or recommendation."
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
    if packet.agent_role != AgentRole.FINAL_DECISION and output.recommendation is not None:
        issues.append(
            AgentValidationIssue(
                location="recommendation",
                message=(
                    "Only the final-decision agent may return an INVEST or PASS "
                    "recommendation."
                ),
            )
        )

    for summary_index, summary in enumerate(output.summary):
        if not summary.unsupported and not summary.evidence:
            issues.append(
                AgentValidationIssue(
                    location=f"summary[{summary_index}].evidence",
                    message=(
                        "This summary point needs at least one evidence ID or must be "
                        "marked unsupported."
                    ),
                )
            )
        for reference_index, reference in enumerate(summary.evidence):
            _validate_evidence_reference(
                reference,
                evidence_by_id=evidence_by_id,
                location=f"summary[{summary_index}].evidence[{reference_index}]",
                issues=issues,
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
        if finding.unsupported and finding.score_delta != 0:
            issues.append(
                AgentValidationIssue(
                    location=f"findings[{finding_index}].score_delta",
                    message="Unsupported findings cannot change the score.",
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
        if not output.recommendation.evidence:
            if packet.evidence:
                recommendation_text = (
                    "An INVEST recommendation"
                    if output.recommendation.recommendation == Recommendation.INVEST
                    else "A PASS recommendation"
                )
                issues.append(
                    AgentValidationIssue(
                        location="recommendation.evidence",
                        message=(
                            f"{recommendation_text} needs at least one cited evidence ID "
                            "when packet evidence exists."
                        ),
                    )
                )
            elif not _is_no_evidence_final_pass(output, packet):
                issues.append(
                    AgentValidationIssue(
                        location="recommendation.evidence",
                        message=(
                            "A no-evidence final recommendation must be PASS with a $0 "
                            "check and must clearly say NEEDS_DILIGENCE."
                        ),
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


def _is_no_evidence_final_pass(
    output: AgentReviewOutput,
    packet: AgentInputPacket,
) -> bool:
    recommendation = output.recommendation
    if recommendation is None:
        return False
    return (
        packet.agent_role == AgentRole.FINAL_DECISION
        and not packet.evidence
        and recommendation.recommendation == Recommendation.PASS
        and recommendation.check_size == 0
        and _mentions_needs_diligence(output)
    )


def _mentions_needs_diligence(output: AgentReviewOutput) -> bool:
    values: list[str] = []
    values.extend(summary.summary for summary in output.summary)
    values.extend(finding.finding for finding in output.findings)
    values.extend(output.limitations)
    if output.recommendation is not None:
        values.append(output.recommendation.reason)
    return any("NEEDS_DILIGENCE" in value for value in values)


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
    if quote is not None and not quote.strip():
        issues.append(
            AgentValidationIssue(
                location=location,
                message="The quoted text is empty. Add a precise quote or omit the quote.",
            )
        )
        quote = None

    if quote is not None and quote not in evidence.text:
        issues.append(
            AgentValidationIssue(
                location=location,
                message="The quoted text was not found in the cited evidence record.",
            )
        )
    if quote is not None and looks_like_embedded_source_instruction(quote):
        issues.append(
            AgentValidationIssue(
                location=location,
                message=(
                    "The quoted text looks like an instruction embedded in a source "
                    "document, not investment evidence."
                ),
            )
        )
    if (
        quote is None
        and looks_like_embedded_source_instruction(evidence.text)
    ):
        issues.append(
            AgentValidationIssue(
                location=location,
                message=(
                    "This citation uses an evidence record that contains an "
                    "instruction embedded in a source document. Add a precise quote "
                    "from the investment evidence instead."
                ),
            )
        )
