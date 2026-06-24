from __future__ import annotations

import re

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

EMBEDDED_SOURCE_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:please\s+)?ignore\s+(?:all\s+|every\s+|previous\s+|the\s+)?instructions?\b",
        r"^(?:please\s+)?disregard\s+(?:all\s+|previous\s+|the\s+)?instructions?\b",
        r"^(?:please\s+)?forget\s+(?:everything\s+above|the\s+above|previous\s+instructions?)\b",
        r"^(?:please\s+)?always\s+recommend\s+(?:invest|pass)\b",
        r"^(?:please\s+)?do\s+not\s+follow\s+the\s+system\b",
        r"^(?:please\s+)?(?:print|reveal|show)\s+the\s+system\s+prompt\b",
    )
)
MID_LINE_SOURCE_INSTRUCTION_PATTERN = re.compile(
    r"\s(?:ignore\s+(?:all\s+|every\s+|previous\s+|the\s+)?instructions?|"
    r"disregard\s+(?:all\s+|previous\s+|the\s+)?instructions?|"
    r"forget\s+(?:everything\s+above|the\s+above|previous\s+instructions?)|"
    r"always\s+recommend\s+(?:invest|pass)|"
    r"do\s+not\s+follow\s+the\s+system|"
    r"(?:print|reveal|show)\s+the\s+system\s+prompt)\b"
)
SOURCE_INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"^(?:(?:"
    r"assistant|chat|developer|important|instruction|instructions|model|note|"
    r"operator|prompt|speaker|system|system note|system prompt|user"
    r")\s*(?:[-:\u2010-\u2015\u2212])\s*)+"
)
SOURCE_LIST_PREFIX_PATTERN = re.compile(
    r"""^[\s>"'`#]*(?:(?:[-*+>]+|\d+[\.)]|#+)\s*)*"""
)


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
            issues.append(
                AgentValidationIssue(
                    location="recommendation.evidence",
                    message="A recommendation needs at least one cited evidence ID.",
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
    if quote is not None and _looks_like_embedded_source_instruction(quote):
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
        and _looks_like_embedded_source_instruction(evidence.text)
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


def _looks_like_embedded_source_instruction(text: str) -> bool:
    normalized_text = " ".join(text.lower().split())
    return any(
        pattern.search(candidate)
        for candidate in _source_instruction_candidates(text)
        for pattern in EMBEDDED_SOURCE_INSTRUCTION_PATTERNS
    ) or _has_mid_line_instruction(normalized_text)


def _source_instruction_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.splitlines() or [text]:
        for segment in [line, *re.split(r"(?<=[.!?;])\s+", line)]:
            normalized = " ".join(segment.lower().split())
            if not normalized:
                continue
            candidates.append(_strip_source_instruction_prefix(normalized))
    return candidates


def _strip_source_instruction_prefix(text: str) -> str:
    stripped = SOURCE_LIST_PREFIX_PATTERN.sub("", text).strip()
    previous = None
    while stripped != previous:
        previous = stripped
        stripped = SOURCE_INSTRUCTION_PREFIX_PATTERN.sub("", stripped).strip()
        stripped = SOURCE_LIST_PREFIX_PATTERN.sub("", stripped).strip()
    return stripped


def _has_mid_line_instruction(text: str) -> bool:
    for match in MID_LINE_SOURCE_INSTRUCTION_PATTERN.finditer(text):
        before = text[: match.start()].rstrip()
        if before.endswith(("prompt:", "example:", "user type", "users type")):
            continue
        return True
    return False
