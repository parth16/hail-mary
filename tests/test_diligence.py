from __future__ import annotations

from datetime import UTC, datetime

from hailmary.evidence import (
    DiligenceAnswerLog,
    DiligenceAnswerRecord,
    DiligenceAnswerStatus,
    DiligenceQuestionItem,
    DiligenceQuestionQueue,
    DiligenceQuestionSource,
    DiligenceResolutionPath,
    DiligenceTriageStatus,
    apply_diligence_answers,
    build_diligence_triage,
)

BUILT_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_diligence_triage_groups_duplicates_and_routes_resolution_paths() -> None:
    queue = _queue(
        [
            _question(
                "dq_return_rule",
                priority=1,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Collect the missing return-math inputs before sizing the check.",
                reason="Ownership, dilution, fees, carry, and exit scenario are missing.",
                evidence_ids=["ev_terms"],
            ),
            _question(
                "dq_return_model",
                priority=101,
                source=DiligenceQuestionSource.FINAL_REVIEW,
                question="Please provide expected investor ownership and any pro rata rights.",
                reason="The final review cannot size the check without investment economics.",
                evidence_ids=["ev_terms"],
            ),
            _question(
                "dq_valuation",
                priority=2,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Confirm why the valuation is justified by current evidence.",
                reason="The entry valuation is high relative to verified support.",
            ),
            _question(
                "dq_external",
                priority=4,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question=(
                    "Finish unresolved external research before relying on "
                    "public-source gaps."
                ),
                reason="Public-source research has not verified market and financing facts.",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert triage.total_question_count == 4
    assert triage.unresolved_question_count == 4
    assert triage.decision_blocker_count == 3
    by_title = {item.title: item for item in triage.items}
    assert by_title["Return math and ownership"].unresolved_question_count == 2
    assert by_title["Return math and ownership"].resolution_path == (
        DiligenceResolutionPath.MERIDIAN_EMAIL
    )
    assert by_title["Return math and ownership"].status == (
        DiligenceTriageStatus.DECISION_BLOCKER
    )
    assert by_title["Valuation support"].resolution_path == (
        DiligenceResolutionPath.PAID_DATA_SOURCE
    )
    assert by_title["External validation"].resolution_path == (
        DiligenceResolutionPath.WEB_RESEARCH
    )
    assert triage.meridian_email_draft is not None
    assert "Return math and ownership" in triage.meridian_email_draft.body
    assert "screenshots" in triage.meridian_email_draft.body
    assert triage.meridian_email_draft.question_ids == [
        "dq_return_rule",
        "dq_return_model",
    ]


def test_diligence_triage_excludes_resolved_question_groups() -> None:
    queue = _queue(
        [
            _question(
                "dq_return_rule",
                priority=1,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Collect the missing return-math inputs before sizing the check.",
                reason="Ownership and dilution are missing.",
            ),
            _question(
                "dq_return_model",
                priority=101,
                source=DiligenceQuestionSource.FINAL_REVIEW,
                question="Please provide expected investor ownership and pro rata rights.",
                reason="The final review cannot size the check without economics.",
            ),
            _question(
                "dq_source_review",
                priority=3,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Verify low-confidence extracted claims against the source.",
                reason="Extraction confidence was low.",
            ),
        ]
    )
    answer_log = DiligenceAnswerLog(
        deal_id="synthetic-deal",
        answers=[
            _answer("ans_1", "dq_return_rule"),
            _answer("ans_2", "dq_return_model"),
        ],
    )

    updated_queue = apply_diligence_answers(queue, answer_log)

    assert updated_queue.resolved_count == 2
    assert updated_queue.triage is not None
    assert [item.title for item in updated_queue.triage.items] == ["Source quality review"]
    assert updated_queue.triage.meridian_email_draft is None


def test_diligence_triage_routes_meridian_workflow_fields_to_email() -> None:
    queue = _queue(
        [
            _question(
                "dq_meridian_valuation",
                priority=21,
                source=DiligenceQuestionSource.MERIDIAN_WORKFLOW,
                question="Resolve the Meridian field: Valuation.",
                reason=(
                    "The Meridian manual workflow still needs this portal field "
                    "before its coverage can be treated as complete."
                ),
                category="meridian:valuation",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert len(triage.items) == 1
    item = triage.items[0]
    assert item.title == "Valuation support"
    assert item.resolution_path == DiligenceResolutionPath.MERIDIAN_EMAIL
    assert triage.meridian_email_draft is not None
    assert "Valuation support" in triage.meridian_email_draft.body


def test_diligence_triage_classifies_fundability_as_investment_terms() -> None:
    queue = _queue(
        [
            _question(
                "dq_fundability",
                priority=5,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Check whether the company can raise the next round.",
                reason="Limited investor or growth signals make fundability unclear.",
                category="financing terms",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert len(triage.items) == 1
    item = triage.items[0]
    assert item.title == "Investment terms"
    assert item.resolution_path == DiligenceResolutionPath.MERIDIAN_EMAIL
    assert item.status == DiligenceTriageStatus.DECISION_BLOCKER


def test_diligence_triage_does_not_match_fee_inside_feedback() -> None:
    queue = _queue(
        [
            _question(
                "dq_feedback",
                priority=30,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Collect customer feedback and retention evidence.",
                reason="Customer feedback should support retention before scoring traction.",
                category="customers",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert len(triage.items) == 1
    item = triage.items[0]
    assert item.title == "Traction and customer metrics"
    assert item.resolution_path == DiligenceResolutionPath.PAID_DATA_SOURCE


def test_diligence_triage_includes_follow_up_meridian_items_in_email() -> None:
    queue = _queue(
        [
            _question(
                "dq_meridian_traction",
                priority=40,
                source=DiligenceQuestionSource.MERIDIAN_WORKFLOW,
                question="Resolve the Meridian field: Traction, revenue, and customers.",
                reason=(
                    "The Meridian manual workflow still needs this portal field "
                    "before its coverage can be treated as complete."
                ),
                category="meridian:traction_revenue_customers",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert len(triage.items) == 1
    item = triage.items[0]
    assert item.status == DiligenceTriageStatus.FOLLOW_UP
    assert item.resolution_path == DiligenceResolutionPath.MERIDIAN_EMAIL
    assert triage.meridian_email_draft is not None
    assert "Traction and customer metrics" in triage.meridian_email_draft.body
    assert triage.meridian_email_draft.question_ids == ["dq_meridian_traction"]


def test_diligence_triage_financing_category_beats_valuation_keywords() -> None:
    queue = _queue(
        [
            _question(
                "dq_mixed_terms",
                priority=8,
                source=DiligenceQuestionSource.RULE_BASED_SCORING,
                question="Confirm valuation, round size, discount, and minimum check.",
                reason="No source-backed financing terms were available.",
                category="financing terms",
            ),
        ]
    )

    triage = build_diligence_triage(queue)

    assert len(triage.items) == 1
    item = triage.items[0]
    assert item.title == "Investment terms"
    assert item.resolution_path == DiligenceResolutionPath.MERIDIAN_EMAIL


def _queue(questions: list[DiligenceQuestionItem]) -> DiligenceQuestionQueue:
    return DiligenceQuestionQueue(
        deal_id="synthetic-deal",
        company_name="SyntheticCo",
        created_at=BUILT_AT,
        questions=questions,
    )


def _question(
    question_id: str,
    *,
    priority: int,
    source: DiligenceQuestionSource,
    question: str,
    reason: str,
    category: str | None = None,
    evidence_ids: list[str] | None = None,
) -> DiligenceQuestionItem:
    return DiligenceQuestionItem(
        question_id=question_id,
        priority=priority,
        source=source,
        question=question,
        reason=reason,
        category=category,
        evidence_ids=evidence_ids or [],
    )


def _answer(answer_id: str, question_id: str) -> DiligenceAnswerRecord:
    return DiligenceAnswerRecord(
        answer_id=answer_id,
        deal_id="synthetic-deal",
        question_id=question_id,
        status=DiligenceAnswerStatus.RESOLVED,
        answer="Synthetic source-backed answer.",
        evidence_ids=[],
        created_at=BUILT_AT,
    )
