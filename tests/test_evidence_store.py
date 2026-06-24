from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hailmary.config import AppConfig
from hailmary.evidence import build_evidence_store, verify_citation
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.schemas.documents import (
    DocumentType,
    FileType,
    SourceKind,
)
from hailmary.schemas.evidence import (
    EvidenceCitation,
    EvidenceKind,
    EvidenceRecord,
    SourceFreshness,
    VerificationStatus,
)


def test_ingest_folder_writes_evidence_store_with_verified_deal_terms(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "TermCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap $8M. Discount 20%. Minimum investment $1,000.",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    assert deal.evidence_store_path.exists()
    assert deal.evidence_count == 1
    assert deal.claim_count == 3
    assert deal.conflict_count == 0

    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    labels = {claim["label"] for claim in saved_store["claims"]}
    assert labels == {"discount", "minimum investment", "valuation cap"}
    assert {
        claim["verification_status"] for claim in saved_store["claims"]
    } == {VerificationStatus.VERIFIED}
    for claim in saved_store["claims"]:
        citation = claim["citations"][0]
        evidence = saved_store["evidence"][0]
        assert (
            evidence["text"][citation["source_span_start"] : citation["source_span_end"]]
            == citation["quote"]
        )
        assert citation["quote"] in evidence["text"]


def test_evidence_store_flags_conflicting_deal_terms(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "ConflictCo"
    company.mkdir(parents=True)
    (company / "memo-a.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    (company / "memo-b.txt").write_text("Valuation cap $10M.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.conflict_count == 1
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    assert saved_store["conflicts"][0]["label"] == "valuation cap"
    assert len(saved_store["conflicts"][0]["normalized_values"]) == 2
    assert {
        claim["verification_status"] for claim in saved_store["claims"]
    } == {VerificationStatus.CONFLICTED}
    assert {
        claim["quality"]["verification_status"] for claim in saved_store["claims"]
    } == {VerificationStatus.CONFLICTED}


def test_verify_citation_checks_evidence_id_span_and_quote() -> None:
    evidence = EvidenceRecord(
        id="ev_test",
        deal_id="deal_test",
        document_id="doc_test",
        document_path=Path("memo.txt"),
        evidence_kind=EvidenceKind.PAGE_TEXT,
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        text="Valuation cap $8M.",
    )
    evidence_by_id = {evidence.id: evidence}

    assert (
        verify_citation(
            EvidenceCitation(
                evidence_id=evidence.id,
                quote="Valuation cap $8M",
                source_span_start=0,
                source_span_end=17,
            ),
            evidence_by_id,
        )
        == VerificationStatus.VERIFIED
    )
    assert (
        verify_citation(
            EvidenceCitation(
                evidence_id="missing",
                quote="Valuation cap $8M",
                source_span_start=0,
                source_span_end=17,
            ),
            evidence_by_id,
        )
        == VerificationStatus.EVIDENCE_NOT_FOUND
    )
    assert (
        verify_citation(
            EvidenceCitation(
                evidence_id=evidence.id,
                quote="Valuation cap $8M",
                source_span_start=0,
                source_span_end=200,
            ),
            evidence_by_id,
        )
        == VerificationStatus.SPAN_MISMATCH
    )
    assert (
        verify_citation(
            EvidenceCitation(
                evidence_id=evidence.id,
                quote="Different quote",
                source_span_start=0,
                source_span_end=17,
            ),
            evidence_by_id,
        )
        == VerificationStatus.QUOTE_MISMATCH
    )


def test_evidence_store_marks_old_sources_stale(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "StaleCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")
    stale_created_at = datetime(2024, 1, 1, tzinfo=UTC)

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))
    deal = summary.deals[0]
    document = deal.documents[0]
    document.source.created_at = stale_created_at

    store = build_evidence_store(deal, created_at=datetime(2026, 1, 2, tzinfo=UTC))

    assert store.evidence[0].source_freshness == SourceFreshness.STALE
    assert store.claims[0].quality.recency == SourceFreshness.STALE
