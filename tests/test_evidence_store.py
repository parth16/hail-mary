from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hailmary.config import AppConfig
from hailmary.evidence import build_evidence_store, verify_citation
from hailmary.ingest.folder_loader import ingest_folder
from hailmary.schemas.documents import (
    DocumentType,
    ExtractedPage,
    ExtractedTable,
    ExtractionQuality,
    FileType,
    IngestedDeal,
    IngestedDocument,
    SourceDocument,
    SourceKind,
)
from hailmary.schemas.evidence import (
    EvidenceCitation,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStore,
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


def test_evidence_store_v2_default_preserves_v1_store_version() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    new_store = EvidenceStore(
        deal_id="deal_test",
        company_name="VersionCo",
        created_at=created_at,
    )
    old_store = EvidenceStore.model_validate(
        {
            "version": "1",
            "deal_id": "deal_test",
            "company_name": "VersionCo",
            "created_at": created_at.isoformat(),
            "evidence": [],
            "claims": [],
            "conflicts": [],
        }
    )

    assert new_store.version == "2"
    assert old_store.version == "1"
    assert old_store.evidence_count == 0


def test_deal_terms_parse_table_separators_and_full_money_suffixes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "TableTermCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap | $8 million\nDiscount | 20% \n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    claims_by_label = {claim["label"]: claim for claim in saved_store["claims"]}

    assert claims_by_label["valuation cap"]["value"] == "$8 million"
    assert claims_by_label["valuation cap"]["normalized_value"] == "usd_cents:800000000"
    assert claims_by_label["valuation cap"]["raw_text"] == "Valuation cap | $8 million"
    assert claims_by_label["valuation cap"]["verification_status"] == VerificationStatus.VERIFIED
    assert claims_by_label["discount"]["raw_text"] == "Discount | 20%"
    assert claims_by_label["discount"]["verification_status"] == VerificationStatus.VERIFIED


def test_deal_terms_parse_post_money_cap_with_usd_suffix(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "PostMoneyCapCo"
    company.mkdir(parents=True)
    (company / "terms.txt").write_text(
        "Round Seed\n"
        "Instrument SAFE\n"
        "Estimated round size $4M USD\n"
        "Post-money cap $20M USD\n"
        "Discount 0%\n"
        "Minimum investment $1,000 USD\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    claims_by_label = {claim["label"]: claim for claim in saved_store["claims"]}

    assert set(claims_by_label) == {
        "discount",
        "minimum investment",
        "round size",
        "valuation cap",
    }
    valuation_cap = claims_by_label["valuation cap"]
    assert valuation_cap["value"] == "$20M USD"
    assert valuation_cap["normalized_value"] == "usd_cents:2000000000"
    assert valuation_cap["raw_text"] == "Post-money cap $20M USD"
    assert valuation_cap["verification_status"] == VerificationStatus.VERIFIED
    assert claims_by_label["round size"]["normalized_value"] == "usd_cents:400000000"
    assert claims_by_label["discount"]["normalized_value"] == "basis_points:0"
    citation = valuation_cap["citations"][0]
    evidence = next(
        evidence
        for evidence in saved_store["evidence"]
        if evidence["id"] == citation["evidence_id"]
    )
    assert (
        evidence["text"][citation["source_span_start"] : citation["source_span_end"]]
        == "Post-money cap $20M USD"
    )


def test_deal_terms_parse_standalone_cap_only_in_financing_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "StandaloneCapCo"
    company.mkdir(parents=True)
    (company / "terms.txt").write_text(
        "Instrument SAFE\nRound Seed\nCap | $150M\nDiscount 15%\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    claims_by_label = {claim["label"]: claim for claim in saved_store["claims"]}

    assert claims_by_label["valuation cap"]["value"] == "$150M"
    assert claims_by_label["valuation cap"]["normalized_value"] == "usd_cents:15000000000"
    assert claims_by_label["valuation cap"]["raw_text"] == "Cap | $150M"


def test_deal_terms_parse_standalone_cap_with_split_safe_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "SplitSafeCapCo"
    company.mkdir(parents=True)
    (company / "terms.txt").write_text(
        "Instrument\nSAFE\nCap $20M\nDiscount 0%\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    claims_by_label = {claim["label"]: claim for claim in saved_store["claims"]}

    assert claims_by_label["valuation cap"]["value"] == "$20M"
    assert claims_by_label["valuation cap"]["normalized_value"] == "usd_cents:2000000000"
    assert claims_by_label["valuation cap"]["raw_text"] == "Cap $20M"


def test_deal_terms_do_not_parse_unrelated_caps_as_valuation_caps(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "UnrelatedCapCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Round Seed\n"
        "Discount 15%\n"
        "Market cap $150M.\n"
        "Expense cap $2M.\n"
        "Exposure cap $5M.\n"
        "Budget cap $8M.\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))

    assert "valuation cap" not in {claim["label"] for claim in saved_store["claims"]}


def test_deal_terms_do_not_parse_standalone_cap_without_financing_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "NoContextCapCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Cap $150M\nGeneral market commentary without financing terms.\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))

    assert saved_store["claims"] == []


def test_deal_terms_do_not_parse_wrapped_market_cap_in_financing_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "WrappedMarketCapCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Round Seed\n"
        "Market\n"
        "Cap $150M\n"
        "Discount 15%\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))

    assert "valuation cap" not in {claim["label"] for claim in saved_store["claims"]}
    assert {claim["label"] for claim in saved_store["claims"]} == {"discount"}


def test_deal_terms_do_not_truncate_usdc_as_usd(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "UsdcCapCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "Valuation cap $20M USDC. Discount 15%.\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))

    assert "valuation cap" not in {claim["label"] for claim in saved_store["claims"]}
    assert {claim["label"] for claim in saved_store["claims"]} == {"discount"}


def test_deal_terms_do_not_extract_negated_post_money_cap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "NegatedPostMoneyCapCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text(
        "No post-money cap of $20M is included; the SAFE is uncapped. Discount 0%.\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))

    assert "valuation cap" not in {claim["label"] for claim in saved_store["claims"]}
    assert {claim["label"] for claim in saved_store["claims"]} == {"discount"}


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
    assert {claim["quality"]["confidence"] for claim in saved_store["claims"]} == {0.2}
    assert {
        claim["quality"]["score_impact"] for claim in saved_store["claims"]
    } == {"excluded_until_conflict_is_resolved"}


def test_table_evidence_is_not_duplicated_as_page_evidence(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "TableOnlyCo"
    company.mkdir(parents=True)
    (company / "terms.csv").write_text("Valuation cap,$8M\n", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    assert [evidence["evidence_kind"] for evidence in saved_store["evidence"]] == [
        EvidenceKind.TABLE_TEXT
    ]
    assert deal.evidence_count == 1
    assert deal.claim_count == 1


def test_multi_table_page_text_is_not_duplicated_as_page_evidence(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "MultiTableCo"
    company.mkdir(parents=True)
    (company / "terms.csv").write_text(
        "Valuation cap,$8M\nDiscount,20%\n",
        encoding="utf-8",
    )

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    assert [evidence["evidence_kind"] for evidence in saved_store["evidence"]] == [
        EvidenceKind.TABLE_TEXT
    ]
    assert deal.evidence_count == 1
    assert deal.claim_count == 2


def test_page_evidence_keeps_safe_clean_text_source_span() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = SourceDocument(
        id="doc_test",
        deal_id="deal_test",
        path=Path("memo.txt"),
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        title="memo",
        ingested_at=created_at,
        sha256="abc",
        extraction_quality=ExtractionQuality.HIGH,
    )
    document = IngestedDocument(
        source=source,
        pages=[
            ExtractedPage(
                page_number=None,
                raw_text="  Valuation cap $8M.  ",
                clean_text="Valuation cap $8M.",
                word_count=3,
                source_span_start=40,
                source_span_end=62,
            )
        ],
        tables=[],
        output_path=Path("out.json"),
    )
    deal = IngestedDeal(id="deal_test", company_name="SpanCo", documents=[document])

    store = build_evidence_store(deal, created_at=created_at)

    assert store.evidence[0].source_span_start == 42
    assert store.evidence[0].source_span_end == 60


def test_page_evidence_keeps_span_when_ocr_merge_adds_blank_separator() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = SourceDocument(
        id="doc_test",
        deal_id="deal_test",
        path=Path("deck.pdf"),
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.PITCH_DECK,
        file_type=FileType.PDF,
        title="deck",
        ingested_at=created_at,
        sha256="abc",
        extraction_quality=ExtractionQuality.MEDIUM,
        ocr_applied=True,
    )
    raw_text = "Acme investor deck\n\nValuation cap $8M. Minimum investment $1,000."
    document = IngestedDocument(
        source=source,
        pages=[
            ExtractedPage(
                page_number=1,
                raw_text=raw_text,
                clean_text="Acme investor deck\nValuation cap $8M. Minimum investment $1,000.",
                word_count=8,
                source_span_start=30,
                source_span_end=30 + len(raw_text),
                ocr_applied=True,
                ocr_confidence=0.91,
            )
        ],
        tables=[],
        output_path=Path("out.json"),
    )
    deal = IngestedDeal(id="deal_test", company_name="SpanCo", documents=[document])

    store = build_evidence_store(deal, created_at=created_at)

    assert store.evidence[0].source_span_start == 30
    assert store.evidence[0].source_span_end == 30 + len(raw_text)
    assert store.evidence[0].ocr_applied
    assert store.evidence[0].ocr_confidence == 0.91


def test_page_evidence_drops_span_when_clean_text_changes_raw_text() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = SourceDocument(
        id="doc_test",
        deal_id="deal_test",
        path=Path("memo.txt"),
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.TXT,
        title="memo",
        ingested_at=created_at,
        sha256="abc",
        extraction_quality=ExtractionQuality.MEDIUM,
    )
    document = IngestedDocument(
        source=source,
        pages=[
            ExtractedPage(
                page_number=None,
                raw_text="Valuation   cap $8M.",
                clean_text="Valuation cap $8M.",
                word_count=3,
            )
        ],
        tables=[],
        output_path=Path("out.json"),
    )
    deal = IngestedDeal(id="deal_test", company_name="SpanCo", documents=[document])

    store = build_evidence_store(deal, created_at=created_at)

    assert store.evidence[0].source_span_start is None
    assert store.evidence[0].source_span_end is None


def test_mixed_page_and_table_text_does_not_duplicate_table_claims() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = SourceDocument(
        id="doc_test",
        deal_id="deal_test",
        path=Path("memo.docx"),
        source_kind=SourceKind.LOCAL_FILE,
        document_type=DocumentType.MEMO,
        file_type=FileType.DOCX,
        title="memo",
        ingested_at=created_at,
        sha256="abc",
        extraction_quality=ExtractionQuality.HIGH,
    )
    document = IngestedDocument(
        source=source,
        pages=[
            ExtractedPage(
                page_number=None,
                raw_text="Company memo\nValuation cap | $8M",
                clean_text="Company memo\nValuation cap | $8M",
                word_count=5,
            )
        ],
        tables=[
            ExtractedTable(
                table_index=1,
                rows=[["Valuation cap", "$8M"]],
                clean_text="Valuation cap | $8M",
                row_count=1,
                column_count=2,
            )
        ],
        output_path=Path("out.json"),
    )
    deal = IngestedDeal(id="deal_test", company_name="MixedCo", documents=[document])

    store = build_evidence_store(deal, created_at=created_at)

    assert [evidence.evidence_kind for evidence in store.evidence] == [
        EvidenceKind.PAGE_TEXT,
        EvidenceKind.TABLE_TEXT,
    ]
    assert store.evidence[0].text == "Company memo"
    assert [claim.label for claim in store.claims] == ["valuation cap"]


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
    document.source.retrieved_at = datetime(2025, 12, 31, tzinfo=UTC)

    store = build_evidence_store(deal, created_at=datetime(2026, 1, 2, tzinfo=UTC))

    assert store.evidence[0].source_freshness == SourceFreshness.STALE
    assert store.claims[0].quality.recency == SourceFreshness.STALE


def test_ingested_local_files_use_file_timestamp_for_freshness(tmp_path: Path) -> None:
    root = tmp_path / "pitch-decks"
    company = root / "FreshCo"
    company.mkdir(parents=True)
    (company / "memo.txt").write_text("Valuation cap $8M.", encoding="utf-8")

    summary = ingest_folder(root, config=AppConfig(data_dir=tmp_path / "data"))

    deal = summary.deals[0]
    document = deal.documents[0]
    assert document.source.created_at is not None
    assert deal.evidence_store_path is not None
    saved_store = json.loads(deal.evidence_store_path.read_text(encoding="utf-8"))
    assert saved_store["evidence"][0]["source_freshness"] == SourceFreshness.CURRENT
