"""tests/test_stage1.py — Unit tests for stage1_preprocess.py"""
import json
import sys
import os
from pathlib import Path
import pytest

# Ensure finreg10k-kg is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import stage1_preprocess as s1


# ── chunk_item1a: ALL CAPS header splitting ───────────────────────────────────

def test_chunk_item1a_splits_on_all_caps_header():
    """chunk_item1a splits on ALL CAPS headers (10+ uppercase chars)."""
    # Build two well-separated sections; each body >= 200 chars
    section1_body = (
        "We are exposed to interest rate risk. Changes in prevailing interest rates "
        "affect our net interest income and the fair value of our fixed-rate assets "
        "and liabilities. Rising rates reduce bond values and increase funding costs."
    )
    section2_body = (
        "We maintain adequate liquidity to meet our obligations. Our liquidity "
        "management framework requires that we hold sufficient high-quality liquid "
        "assets to survive a 30-day stress scenario as required under the LCR rules."
    )
    text = (
        "Some preamble text about the company.\n\n"
        "\nINTEREST RATE RISK AND MARKET CONDITIONS\n"
        + section1_body + "\n\n"
        "\nLIQUIDITY RISK MANAGEMENT\n"
        + section2_body
    )
    chunks = s1.chunk_item1a(text, "12345", "202212")
    # Should produce at least 1 chunk (sections are long enough)
    assert len(chunks) >= 1
    texts = [c["text"] for c in chunks]
    combined = " ".join(texts)
    assert "interest rate risk" in combined.lower() or "INTEREST RATE" in combined


def test_chunk_item1a_splits_on_risk_factor_n():
    """chunk_item1a splits on 'Risk Factor N' explicit numbering."""
    body1 = (
        "Regulatory compliance is critical for our operations. We must adhere to "
        "the Bank Secrecy Act, OFAC requirements, and other applicable federal and "
        "state laws. Failure to comply could result in material penalties and harm "
        "to our reputation. We devote significant resources to compliance programs."
    )
    body2 = (
        "Market conditions can adversely affect our business, financial condition, "
        "and results of operations. Deterioration in economic conditions, including "
        "increased unemployment or reduced consumer confidence, may reduce demand "
        "for our products and services and increase credit losses in our portfolio."
    )
    text = (
        "Overview of risk factors.\n"
        "\nRisk Factor 1\n"
        + body1 + "\n"
        "\nRisk Factor 2\n"
        + body2
    )
    chunks = s1.chunk_item1a(text, "12345", "202212")
    assert len(chunks) >= 2
    texts = [c["text"] for c in chunks]
    combined = " ".join(texts)
    assert "regulatory compliance" in combined.lower()
    assert "market conditions" in combined.lower()


def test_chunk_item1a_splits_on_bullet():
    """chunk_item1a splits on bullet-initiated risks."""
    bullet1_body = (
        "Interest rate risk could cause material losses in our securities portfolio "
        "and affect net interest income. We are exposed to both repricing risk and "
        "basis risk, and use interest rate swaps and other derivatives to manage "
        "this exposure in accordance with our Asset Liability Management policy."
    )
    bullet2_body = (
        "Credit risk arises from our lending activities and represents the risk of "
        "financial loss due to a borrower or counterparty failing to meet its "
        "contractual obligations. We manage credit risk through underwriting "
        "standards, concentration limits, and ongoing portfolio monitoring processes."
    )
    text = (
        "The following risks apply:\n"
        "\n• " + bullet1_body + "\n"
        "\n• " + bullet2_body
    )
    chunks = s1.chunk_item1a(text, "12345", "202212")
    # At least some of the bullet content should survive the 200-char filter
    assert len(chunks) >= 1


def test_chunk_item1a_drops_short_chunks():
    """Chunks shorter than MIN_CHUNK_CHARS (200) are dropped."""
    text = (
        "\nSHORT SECTION\n"
        "Too short.\n"  # less than 200 chars — will be dropped
        "\nLONG SECTION WITH SUFFICIENT DETAIL\n"
        "This is a much longer risk factor that describes in detail the various "
        "regulatory requirements that the bank must comply with, including capital "
        "adequacy, liquidity management, and stress testing requirements under DFAST. " * 3
    )
    chunks = s1.chunk_item1a(text, "12345", "202212")
    for c in chunks:
        assert len(c["text"]) >= s1.MIN_CHUNK_CHARS, (
            f"Chunk shorter than {s1.MIN_CHUNK_CHARS} chars: {len(c['text'])}"
        )


def test_chunk_item1a_chunk_id_format():
    """chunk_id must follow format {cik}_{period}_1a_{i:04d}."""
    text = "\nREGULATORY CAPITAL REQUIREMENTS AND COMPLIANCE OBLIGATIONS\n" + "x " * 200
    chunks = s1.chunk_item1a(text, "98765", "202306")
    for c in chunks:
        assert c["chunk_id"].startswith("98765_202306_1a_"), (
            f"Unexpected chunk_id format: {c['chunk_id']}"
        )
        # The numeric suffix should be zero-padded to 4 digits
        suffix = c["chunk_id"].rsplit("_", 1)[-1]
        assert len(suffix) == 4 and suffix.isdigit(), (
            f"chunk_id suffix not 4-digit zero-padded: {c['chunk_id']}"
        )


def test_chunk_item1a_source_field():
    """All item1a chunks have source='item1a'."""
    text = "\nOPERATIONAL RISK MANAGEMENT FRAMEWORK\n" + "risk " * 100
    chunks = s1.chunk_item1a(text, "11111", "202212")
    for c in chunks:
        assert c["source"] == "item1a"


def test_chunk_item1a_preserves_cik_period():
    """cik and period metadata fields are preserved in every chunk."""
    text = "\nCREDIT RISK AND LOAN PORTFOLIO QUALITY\n" + "loan " * 100
    cik, period = "54321", "202206"
    chunks = s1.chunk_item1a(text, cik, period)
    for c in chunks:
        assert c["cik"]    == cik
        assert c["period"] == period


def test_chunk_item1a_empty_text():
    """Empty text returns empty list."""
    chunks = s1.chunk_item1a("", "12345", "202212")
    assert chunks == []


def test_chunk_item1a_no_headers_long_text():
    """Text without headers but long enough is kept as single chunk."""
    text = "We are subject to extensive regulatory requirements. " * 50
    chunks = s1.chunk_item1a(text, "12345", "202212")
    # Single block — either kept (if >= 200 chars) or empty
    combined_len = len(text.strip())
    if combined_len >= s1.MIN_CHUNK_CHARS:
        assert len(chunks) >= 1
    else:
        assert len(chunks) == 0


# ── chunk_item1: paragraph splitting ─────────────────────────────────────────

def test_chunk_item1_splits_on_double_newline():
    """chunk_item1 splits on double newlines (paragraph boundaries)."""
    para1 = "We operate as a commercial bank providing services to retail customers. " * 5
    para2 = "Our investment banking division advises corporations on capital markets. " * 5
    text  = para1 + "\n\n" + para2
    chunks = s1.chunk_item1(text, "12345", "202212")
    assert len(chunks) == 2


def test_chunk_item1_drops_short_paragraphs():
    """Paragraphs shorter than 200 chars are dropped."""
    short = "Short paragraph."
    long  = "We provide comprehensive banking services to commercial and retail clients. " * 5
    text  = short + "\n\n" + long + "\n\n" + short
    chunks = s1.chunk_item1(text, "12345", "202212")
    for c in chunks:
        assert len(c["text"]) >= s1.MIN_CHUNK_CHARS


def test_chunk_item1_chunk_id_format():
    """chunk_id must follow format {cik}_{period}_1_{i:04d}."""
    para = "We are a diversified financial services company. " * 10
    text = para + "\n\n" + para
    chunks = s1.chunk_item1(text, "77777", "202106")
    for c in chunks:
        assert c["chunk_id"].startswith("77777_202106_1_")
        suffix = c["chunk_id"].rsplit("_", 1)[-1]
        assert len(suffix) == 4 and suffix.isdigit()


def test_chunk_item1_source_field():
    """All item1 chunks have source='item1'."""
    para = "We operate a national branch network serving consumer customers. " * 10
    chunks = s1.chunk_item1(para + "\n\n" + para, "12345", "202212")
    for c in chunks:
        assert c["source"] == "item1"


# ── Manifest schema compatibility ─────────────────────────────────────────────

def test_manifest_schema_matches_finwiki(tmp_path, monkeypatch):
    """
    The generated manifest must contain doc_id, corpus, chunk_count, chunks,
    and metadata fields — matching FinWiki manifest schema exactly.
    """
    # Create a minimal raw file
    raw_dir = tmp_path / "finreg10k_raw"
    raw_dir.mkdir()
    doc = {
        "cik":         "12345",
        "company":     "Test Bank",
        "sic":         "6022",
        "period":      "2022-12",
        "filing_date": "2023-03-01",
        "item1":       "We are a commercial bank providing financial services. " * 30,
        "item1a":      (
            "\nINTEREST RATE RISK AND MARKET EXPOSURE\n"
            "Interest rate changes can materially affect our net interest income. " * 20
        ),
        "char_count":  5000,
    }
    (raw_dir / "12345_202212.json").write_text(json.dumps(doc))

    chunk_dir   = tmp_path / "finreg10k_chunks"
    pipeline_dir = tmp_path / "data" / "chunks"
    manifest_path = tmp_path / "finreg10k_manifest.json"

    monkeypatch.setattr(s1, "RAW_DIR",            raw_dir)
    monkeypatch.setattr(s1, "CHUNK_DIR",           chunk_dir)
    monkeypatch.setattr(s1, "PIPELINE_CHUNKS_DIR", pipeline_dir)
    monkeypatch.setattr(s1, "MANIFEST",            manifest_path)

    s1.run()

    assert manifest_path.exists(), "Manifest file not created"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) >= 1

    entry = manifest[0]
    # Required top-level keys
    assert "doc_id"      in entry, "Missing 'doc_id'"
    assert "corpus"      in entry, "Missing 'corpus'"
    assert "chunk_count" in entry, "Missing 'chunk_count'"
    assert "chunks"      in entry, "Missing 'chunks'"
    assert "metadata"    in entry, "Missing 'metadata'"

    # corpus value
    assert entry["corpus"] == "FinReg10K"

    # chunk_count consistency
    assert entry["chunk_count"] == len(entry["chunks"])

    # metadata required fields
    meta = entry["metadata"]
    assert "cik"    in meta
    assert "period" in meta
    assert "source" in meta
    assert meta["source"] == "SEC EDGAR 10-K"


def test_manifest_chunks_have_required_fields(tmp_path, monkeypatch):
    """Each chunk in the manifest must have chunk_id, source, cik, period, text, char_count."""
    raw_dir = tmp_path / "finreg10k_raw"
    raw_dir.mkdir()
    doc = {
        "cik":         "99999",
        "company":     "Another Bank",
        "sic":         "6021",
        "period":      "2023-06",
        "filing_date": "2023-09-01",
        "item1":       "Commercial banking operations description. " * 20,
        "item1a":      (
            "\nCOMPLIANCE RISK AND REGULATORY ENVIRONMENT\n"
            "We face significant compliance risk under various regulations. " * 20
        ),
        "char_count":  4000,
    }
    (raw_dir / "99999_202306.json").write_text(json.dumps(doc))

    chunk_dir    = tmp_path / "finreg10k_chunks"
    pipeline_dir = tmp_path / "data" / "chunks"
    manifest_path = tmp_path / "finreg10k_manifest.json"

    monkeypatch.setattr(s1, "RAW_DIR",            raw_dir)
    monkeypatch.setattr(s1, "CHUNK_DIR",           chunk_dir)
    monkeypatch.setattr(s1, "PIPELINE_CHUNKS_DIR", pipeline_dir)
    monkeypatch.setattr(s1, "MANIFEST",            manifest_path)

    s1.run()

    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        for chunk in entry["chunks"]:
            assert "chunk_id"   in chunk
            assert "source"     in chunk
            assert "cik"        in chunk
            assert "period"     in chunk
            assert "text"       in chunk
            assert "char_count" in chunk
            assert chunk["char_count"] >= s1.MIN_CHUNK_CHARS


def test_pipeline_chunks_written(tmp_path, monkeypatch):
    """Pipeline chunk files (data/chunks/*.json) are written in FinWiki format."""
    raw_dir = tmp_path / "finreg10k_raw"
    raw_dir.mkdir()
    doc = {
        "cik":         "44444",
        "company":     "Pipeline Test Bank",
        "sic":         "6035",
        "period":      "2022-12",
        "filing_date": "2023-02-01",
        "item1":       "We are a savings institution. " * 30,
        "item1a":      (
            "\nINTEREST RATE RISK AND NET INTEREST MARGIN\n"
            "Rising interest rates may compress our net interest margin. " * 20
        ),
        "char_count":  4000,
    }
    (raw_dir / "44444_202212.json").write_text(json.dumps(doc))

    chunk_dir    = tmp_path / "finreg10k_chunks"
    pipeline_dir = tmp_path / "data" / "chunks"
    manifest_path = tmp_path / "finreg10k_manifest.json"

    monkeypatch.setattr(s1, "RAW_DIR",            raw_dir)
    monkeypatch.setattr(s1, "CHUNK_DIR",           chunk_dir)
    monkeypatch.setattr(s1, "PIPELINE_CHUNKS_DIR", pipeline_dir)
    monkeypatch.setattr(s1, "MANIFEST",            manifest_path)

    s1.run()

    pipeline_files = list(pipeline_dir.glob("*.json"))
    assert len(pipeline_files) >= 1

    for pf in pipeline_files:
        chunks = json.loads(pf.read_text())
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # Each chunk must have FinWiki pipeline-compatible fields
        for c in chunks:
            assert "chunk_id"       in c
            assert "document_id"    in c
            assert "sequence"       in c
            assert "section_title"  in c
            assert "content"        in c
            assert "token_estimate" in c
            assert "title"          in c
            assert "url"            in c
            assert "domain"         in c


def test_chunk_id_uniqueness():
    """All chunk_ids within a document must be unique."""
    text_1a = (
        "\nINTEREST RATE RISK\n" + "risk text. " * 50 +
        "\nCREDIT RISK AND LOAN LOSSES\n" + "credit text. " * 50 +
        "\nLIQUIDITY RISK MANAGEMENT\n" + "liquidity text. " * 50
    )
    text_1 = "\n\n".join(["paragraph " + str(i) + " " + "x " * 60 for i in range(5)])

    chunks_1a = s1.chunk_item1a(text_1a, "55555", "202212")
    chunks_1  = s1.chunk_item1(text_1,   "55555", "202212")

    all_ids = [c["chunk_id"] for c in chunks_1a + chunks_1]
    assert len(all_ids) == len(set(all_ids)), "Duplicate chunk_ids found"


def test_infer_domain_banking():
    """Banking SIC codes map to 'banking' domain."""
    assert s1.infer_domain("6022") == "banking"
    assert s1.infer_domain("6035") == "banking"
    assert s1.infer_domain("6099") == "banking"
    assert s1.infer_domain("6141") == "banking"


def test_infer_domain_securities():
    """Securities broker SIC codes map to 'securities' domain."""
    assert s1.infer_domain("6211") == "securities"
    assert s1.infer_domain("6200") == "securities"


def test_infer_domain_insurance():
    """Insurance SIC codes map to 'insurance' domain."""
    assert s1.infer_domain("6311") == "insurance"
    assert s1.infer_domain("6411") == "insurance"
