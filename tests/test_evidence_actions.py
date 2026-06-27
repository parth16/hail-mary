from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hailmary.config import AppConfig
from hailmary.evidence.actions import (
    EvidenceActionError,
    EvidenceActionLog,
    EvidenceActionRecord,
    EvidenceActionStatus,
    EvidenceActionTarget,
    action_log_path,
    apply_evidence_actions,
    load_action_log,
    prune_stale_evidence_actions,
    record_evidence_action,
    summarize_evidence_actions,
    write_action_log,
)
from hailmary.schemas.documents import (
    DocumentType,
    FileType,
    IngestedDeal,
    IngestionSummary,
    SourceKind,
)
from hailmary.schemas.evidence import (
    ClaimRecord,
    ClaimType,
    EvidenceCitation,
    EvidenceKind,
    EvidenceQuality,
    EvidenceRecord,
    EvidenceStore,
    SourceFreshness,
    VerificationStatus,
)

BUILT_AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_action_file_write_and_read_stores_only_ids_status_and_note(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store()
    action = _action(
        action_id="act_first",
        target_id="ev_traction",
        status=EvidenceActionStatus.NEEDS_REVIEW,
        note="Operator wants a source check.",
    )

    path = write_action_log(
        config=config,
        log=EvidenceActionLog(deal_id=store.deal_id, actions=[action]),
    )
    loaded = load_action_log(config=config, deal_id=store.deal_id)

    assert loaded.actions == [action]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    raw_text = path.read_text(encoding="utf-8")
    assert "Operator wants a source check" in raw_text
    assert "paid customers" not in raw_text
    assert "Valuation cap" not in raw_text


def test_malformed_action_file_has_plain_english_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = action_log_path(config=config, deal_id="deal_action")
    path.parent.mkdir(parents=True)
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(EvidenceActionError, match="could not be read"):
        load_action_log(config=config, deal_id="deal_action")


def test_duplicate_action_ids_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = action_log_path(config=config, deal_id="deal_action")
    path.parent.mkdir(parents=True)
    duplicate = _action(
        action_id="act_duplicate",
        target_id="ev_traction",
        status=EvidenceActionStatus.USABLE,
    )
    path.write_text(
        EvidenceActionLog(
            deal_id="deal_action",
            actions=[duplicate],
        )
        .model_copy(update={"actions": [duplicate, duplicate]})
        .model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(EvidenceActionError, match="duplicate action IDs"):
        load_action_log(config=config, deal_id="deal_action")


def test_last_action_wins_and_excluded_evidence_filters_claims(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _store()
    write_action_log(
        config=config,
        log=EvidenceActionLog(
            deal_id=store.deal_id,
            actions=[
                _action(
                    action_id="act_first",
                    target_id="ev_traction",
                    status=EvidenceActionStatus.NEEDS_REVIEW,
                ),
                _action(
                    action_id="act_second",
                    target_id="ev_traction",
                    status=EvidenceActionStatus.EXCLUDED,
                ),
            ],
        ),
    )

    summary = summarize_evidence_actions(config=config, store=store)
    application = apply_evidence_actions(config=config, store=store)

    assert summary.evidence_states["ev_traction"].status == EvidenceActionStatus.EXCLUDED
    assert application.excluded_evidence_ids == {"ev_traction"}
    assert [evidence.id for evidence in application.store.evidence] == ["ev_terms"]
    assert [claim.id for claim in application.store.claims] == ["claim_terms"]


def test_excluded_claim_preserves_evidence_for_packet_quote_suppression(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = _store()
    write_action_log(
        config=config,
        log=EvidenceActionLog(
            deal_id=store.deal_id,
            actions=[
                _action(
                    action_id="act_exclude_claim",
                    target_type=EvidenceActionTarget.CLAIM,
                    target_id="claim_terms",
                    status=EvidenceActionStatus.EXCLUDED,
                ),
            ],
        ),
    )

    application = apply_evidence_actions(config=config, store=store)

    assert application.excluded_claim_ids == {"claim_terms"}
    assert application.excluded_evidence_ids == set()
    assert application.packet_quote_only_evidence_ids == {"ev_terms"}
    assert [evidence.id for evidence in application.store.evidence] == [
        "ev_traction",
        "ev_terms",
    ]
    assert [claim.id for claim in application.store.claims] == ["claim_traction"]


def test_exclusion_preserves_unrelated_missing_evidence_claim(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = _store()
    missing_claim = _claim("claim_missing", "$12M", store.evidence[1]).model_copy(
        update={
            "citations": [
                EvidenceCitation(
                    evidence_id="ev_missing",
                    quote="Valuation cap $12M.",
                    source_span_start=0,
                    source_span_end=len("Valuation cap $12M."),
                    verification_status=VerificationStatus.VERIFIED,
                )
            ],
        }
    )
    store = store.model_copy(update={"claims": [*store.claims, missing_claim]})
    write_action_log(
        config=config,
        log=EvidenceActionLog(
            deal_id=store.deal_id,
            actions=[
                _action(
                    action_id="act_exclude_traction",
                    target_id="ev_traction",
                    status=EvidenceActionStatus.EXCLUDED,
                )
            ],
        ),
    )

    application = apply_evidence_actions(config=config, store=store)
    claims_by_id = {claim.id: claim for claim in application.store.claims}

    assert "claim_traction" not in claims_by_id
    assert "claim_terms" in claims_by_id
    assert claims_by_id["claim_missing"].citations[0].verification_status == (
        VerificationStatus.EVIDENCE_NOT_FOUND
    )


def test_excluded_conflict_side_resets_surviving_score_impact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    low_evidence = _evidence("ev_low", "Valuation cap $8M.")
    high_evidence = _evidence("ev_high", "Valuation cap $10M.")
    low_claim = _conflicted_claim(_claim("claim_low", "$8M", low_evidence))
    high_claim = _conflicted_claim(_claim("claim_high", "$10M", high_evidence))
    store = EvidenceStore(
        deal_id="deal_action",
        company_name="ActionCo",
        created_at=BUILT_AT,
        evidence=[low_evidence, high_evidence],
        claims=[low_claim, high_claim],
    )
    write_action_log(
        config=config,
        log=EvidenceActionLog(
            deal_id=store.deal_id,
            actions=[
                _action(
                    action_id="act_exclude_high",
                    target_type=EvidenceActionTarget.CLAIM,
                    target_id="claim_high",
                    status=EvidenceActionStatus.EXCLUDED,
                )
            ],
        ),
    )

    application = apply_evidence_actions(config=config, store=store)
    surviving_claim = application.store.claims[0]

    assert surviving_claim.id == "claim_low"
    assert surviving_claim.verification_status == VerificationStatus.VERIFIED
    assert surviving_claim.quality.score_impact == "not_scored_yet"


def test_prune_stale_evidence_actions_removes_obsolete_targets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = _store()
    _write_ingestion_summary(tmp_path, store)
    write_action_log(
        config=config,
        log=EvidenceActionLog(
            deal_id=store.deal_id,
            actions=[
                _action(
                    action_id="act_valid",
                    target_id="ev_terms",
                    status=EvidenceActionStatus.APPROVED,
                ),
                _action(
                    action_id="act_stale_evidence",
                    target_id="ev_missing",
                    status=EvidenceActionStatus.NEEDS_REVIEW,
                    note="Stale note stays private.",
                ),
                _action(
                    action_id="act_stale_claim",
                    target_type=EvidenceActionTarget.CLAIM,
                    target_id="claim_missing",
                    status=EvidenceActionStatus.EXCLUDED,
                ),
            ],
        ),
    )

    result = prune_stale_evidence_actions(config=config, deal_id=store.deal_id)
    loaded = load_action_log(config=config, deal_id=store.deal_id)
    summary = summarize_evidence_actions(config=config, store=store)

    assert result.removed_action_count == 2
    assert result.removed_target_count == 2
    assert [action.action_id for action in loaded.actions] == ["act_valid"]
    assert summary.stale_action_count == 0


def test_unknown_evidence_id_is_rejected_before_writing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_ingestion_summary(tmp_path, _store())

    with pytest.raises(EvidenceActionError, match="No evidence record missing_ev"):
        record_evidence_action(
            config=config,
            deal_id="deal_action",
            evidence_id="missing_ev",
            status=EvidenceActionStatus.EXCLUDED,
        )

    assert not action_log_path(config=config, deal_id="deal_action").exists()


def test_unknown_claim_id_is_rejected_before_writing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_ingestion_summary(tmp_path, _store())

    with pytest.raises(EvidenceActionError, match="No claim missing_claim"):
        record_evidence_action(
            config=config,
            deal_id="deal_action",
            claim_id="missing_claim",
            status=EvidenceActionStatus.EXCLUDED,
        )

    assert not action_log_path(config=config, deal_id="deal_action").exists()


def test_action_state_rejects_symlinked_data_dir(tmp_path: Path) -> None:
    real_data_dir = tmp_path / "real-data"
    real_data_dir.mkdir()
    symlink_data_dir = tmp_path / "symlink-data"
    try:
        symlink_data_dir.symlink_to(real_data_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(EvidenceActionError, match="symlink"):
        load_action_log(
            config=AppConfig(data_dir=symlink_data_dir),
            deal_id="deal_action",
        )


def test_action_state_rejects_symlinked_action_folder(tmp_path: Path) -> None:
    config = _config(tmp_path)
    external_dir = tmp_path / "external-actions"
    external_dir.mkdir()
    action_dir = config.data_dir / "evidence-actions"
    try:
        action_dir.symlink_to(external_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    with pytest.raises(EvidenceActionError, match="symlink"):
        write_action_log(
            config=config,
            log=EvidenceActionLog(
                deal_id="deal_action",
                actions=[
                    _action(
                        action_id="act_first",
                        target_id="ev_traction",
                        status=EvidenceActionStatus.USABLE,
                    )
                ],
            ),
        )


def _config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AppConfig(data_dir=data_dir)


def _store() -> EvidenceStore:
    traction = _evidence(
        "ev_traction",
        "ARR revenue growth with paid customers and retention.",
    )
    terms = _evidence("ev_terms", "Valuation cap $8M.")
    return EvidenceStore(
        deal_id="deal_action",
        company_name="ActionCo",
        created_at=BUILT_AT,
        evidence=[traction, terms],
        claims=[
            _claim("claim_traction", "growth", traction),
            _claim("claim_terms", "$8M", terms),
        ],
    )


def _evidence(evidence_id: str, text: str) -> EvidenceRecord:
    return EvidenceRecord(
        id=evidence_id,
        deal_id="deal_action",
        document_id=f"doc_{evidence_id}",
        document_path=Path(f"{evidence_id}.txt"),
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        text=text,
        source_span_start=0,
        source_span_end=len(text),
        source_freshness=SourceFreshness.CURRENT,
    )


def _claim(claim_id: str, value: str, evidence: EvidenceRecord) -> ClaimRecord:
    quote = evidence.text
    return ClaimRecord(
        id=claim_id,
        deal_id=evidence.deal_id,
        claim_type=ClaimType.DEAL_TERM,
        label="valuation cap" if value.startswith("$") else "traction",
        value=value,
        normalized_value=value,
        raw_text=quote,
        citations=[
            EvidenceCitation(
                evidence_id=evidence.id,
                quote=quote,
                source_span_start=0,
                source_span_end=len(quote),
                verification_status=VerificationStatus.VERIFIED,
            )
        ],
        verification_status=VerificationStatus.VERIFIED,
        quality=EvidenceQuality(
            claim_type=ClaimType.DEAL_TERM,
            source_type=evidence.source_kind,
            verification_status=VerificationStatus.VERIFIED,
            recency=evidence.source_freshness,
            reliability="synthetic_test_source",
            confidence=0.72,
            materiality="high",
        ),
    )


def _conflicted_claim(claim: ClaimRecord) -> ClaimRecord:
    return claim.model_copy(
        update={
            "verification_status": VerificationStatus.CONFLICTED,
            "quality": claim.quality.model_copy(
                update={
                    "verification_status": VerificationStatus.CONFLICTED,
                    "confidence": 0.2,
                    "score_impact": "excluded_until_conflict_is_resolved",
                }
            ),
        }
    )


def _action(
    *,
    action_id: str,
    target_id: str,
    status: EvidenceActionStatus,
    note: str | None = None,
    target_type: EvidenceActionTarget = EvidenceActionTarget.EVIDENCE,
) -> EvidenceActionRecord:
    return EvidenceActionRecord(
        action_id=action_id,
        deal_id="deal_action",
        target_type=target_type,
        target_id=target_id,
        status=status,
        created_at=BUILT_AT,
        operator_note=note,
    )


def _write_ingestion_summary(tmp_path: Path, store: EvidenceStore) -> None:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    store_dir = processed_dir / "deals" / store.deal_id
    store_dir.mkdir(parents=True)
    store_path = store_dir / "evidence_store.json"
    store_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    summary = IngestionSummary(
        root_path=tmp_path / "pitch-decks",
        scanned_at=BUILT_AT,
        deals=[
            IngestedDeal(
                id=store.deal_id,
                company_name=store.company_name,
                documents=[],
                evidence_store_path=store_path,
                evidence_count=store.evidence_count,
                claim_count=store.claim_count,
                conflict_count=store.conflict_count,
            )
        ],
        summary_path=processed_dir / "ingestion_summary.json",
    )
    summary.summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
