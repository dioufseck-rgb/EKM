"""tests/test_stage0.py — Unit tests for stage0_acquire.py"""
import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest

# Ensure finreg10k-kg is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import stage0_acquire as s0


# ── SIC list completeness ─────────────────────────────────────────────────────

def test_sic_list_complete():
    """All SIC codes from spec Section 2 must be present."""
    required = {
        # Commercial banks
        "6020", "6021", "6022", "6025", "6026", "6027", "6029",
        # Savings institutions
        "6035", "6036",
        # Functions related to depository banking
        "6099",
        # Personal credit institutions
        "6141",
        # Short-term business credit, federal-sponsored credit
        "6153", "6154", "6159",
        # Security and commodity brokers
        "6200", "6211",
        # Life and accident insurance
        "6311", "6321",
        # Insurance agents and brokers
        "6411",
    }
    assert required.issubset(set(s0.FINANCIAL_SICS)), (
        f"Missing SIC codes: {required - set(s0.FINANCIAL_SICS)}"
    )


def test_sic_list_no_extra_unexpected():
    """SIC list should not contain obviously wrong codes (e.g. non-financial)."""
    for sic in s0.FINANCIAL_SICS:
        code = int(sic)
        # All codes should be in the financial services SIC range 6000-6999
        assert 6000 <= code <= 6999, f"Non-financial SIC code: {sic}"


def test_sic_list_strings():
    """All SIC entries must be strings (EDGAR API expects string comparison)."""
    for sic in s0.FINANCIAL_SICS:
        assert isinstance(sic, str), f"SIC code is not a string: {sic!r}"


# ── Deduplication logic ───────────────────────────────────────────────────────

def test_deduplicate_keeps_most_recent(tmp_path, monkeypatch):
    """For a firm with multiple filings, deduplicate keeps the most recent."""
    monkeypatch.setattr(s0, "CORPUS_CSV", tmp_path / "corpus.csv")
    df = pd.DataFrame([
        {"cik": "123", "company": "BankA", "sic": "6022",
         "filing_date": "2023-03-15", "accession_no": "0000123-23-001", "period": "2022-12-31"},
        {"cik": "123", "company": "BankA", "sic": "6022",
         "filing_date": "2022-03-10", "accession_no": "0000123-22-001", "period": "2021-12-31"},
        {"cik": "456", "company": "BankB", "sic": "6021",
         "filing_date": "2022-04-01", "accession_no": "0000456-22-001", "period": "2021-12-31"},
    ])
    result = s0.deduplicate(df)
    assert len(result) == 2
    bank_a_row = result[result["cik"] == "123"]
    assert len(bank_a_row) == 1
    assert bank_a_row.iloc[0]["filing_date"] == "2023-03-15"


def test_deduplicate_single_filing_per_firm(tmp_path, monkeypatch):
    """Each CIK appears exactly once in deduplicated output."""
    monkeypatch.setattr(s0, "CORPUS_CSV", tmp_path / "corpus.csv")
    df = pd.DataFrame([
        {"cik": str(i), "company": f"Bank{i}", "sic": "6022",
         "filing_date": f"202{i%4+0}-01-01", "accession_no": f"ACC{i}", "period": f"202{i%4+0}"}
        for i in range(10)
    ])
    # Add duplicates
    df = pd.concat([df, df]).reset_index(drop=True)

    result = s0.deduplicate(df)
    assert result["cik"].nunique() == result["cik"].count()  # no duplicates
    assert len(result) == 10


def test_deduplicate_respects_corpus_cap(tmp_path, monkeypatch):
    """Deduplicate caps at CORPUS_CAP (200) firms."""
    monkeypatch.setattr(s0, "CORPUS_CSV", tmp_path / "corpus.csv")
    df = pd.DataFrame([
        {"cik": str(i), "company": f"Bank{i}", "sic": "6022",
         "filing_date": "2023-01-01", "accession_no": f"ACC{i}", "period": "2022"}
        for i in range(300)
    ])
    result = s0.deduplicate(df)
    assert len(result) <= s0.CORPUS_CAP


def test_deduplicate_empty_dataframe(tmp_path, monkeypatch):
    """Deduplicate on empty DataFrame returns empty DataFrame."""
    monkeypatch.setattr(s0, "CORPUS_CSV", tmp_path / "corpus.csv")
    df = pd.DataFrame(columns=["cik", "company", "sic", "filing_date", "accession_no", "period"])
    result = s0.deduplicate(df)
    assert result.empty


# ── Boilerplate filter ────────────────────────────────────────────────────────

def test_filter_boilerplate_drops_low_charcount(tmp_path, monkeypatch):
    """Documents with char_count < 2000 must be dropped."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)
    doc = {
        "cik": "111", "company": "TinyBank", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "item1":  "Short text.",
        "item1a": "Some risk factor.",
        "char_count": 100,  # below threshold
    }
    fp = tmp_path / "111_202212.json"
    fp.write_text(json.dumps(doc))

    kept, dropped = s0.filter_boilerplate()
    assert dropped == 1
    assert kept == 0
    assert not fp.exists()


def test_filter_boilerplate_drops_empty_item1a(tmp_path, monkeypatch):
    """Documents with empty item1a must be dropped."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)
    doc = {
        "cik": "222", "company": "NoRiskBank", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "item1":  "A" * 3000,
        "item1a": "",           # empty
        "char_count": 3000,
    }
    fp = tmp_path / "222_202212.json"
    fp.write_text(json.dumps(doc))

    kept, dropped = s0.filter_boilerplate()
    assert dropped == 1
    assert kept == 0
    assert not fp.exists()


def test_filter_boilerplate_keeps_valid_doc(tmp_path, monkeypatch):
    """Valid documents (char_count >= 2000 and non-empty item1a) must be kept."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)
    doc = {
        "cik": "333", "company": "GoodBank", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "item1":  "B" * 2000,
        "item1a": "We are subject to extensive regulation under the Bank Secrecy Act.",
        "char_count": 2500,
    }
    fp = tmp_path / "333_202212.json"
    fp.write_text(json.dumps(doc))

    kept, dropped = s0.filter_boilerplate()
    assert kept == 1
    assert dropped == 0
    assert fp.exists()


def test_filter_boilerplate_exactly_at_threshold(tmp_path, monkeypatch):
    """Documents with char_count == 2000 are below threshold (< 2000 is drop, == 2000 is keep)."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)
    doc = {
        "cik": "444", "company": "EdgeBank", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "item1":  "C" * 1000,
        "item1a": "D" * 1000,
        "char_count": 2000,
    }
    fp = tmp_path / "444_202212.json"
    fp.write_text(json.dumps(doc))

    kept, dropped = s0.filter_boilerplate()
    # char_count == 2000 is NOT < 2000, so should be kept
    assert kept == 1
    assert dropped == 0


def test_filter_boilerplate_whitespace_only_item1a(tmp_path, monkeypatch):
    """item1a containing only whitespace is treated as empty and dropped."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)
    doc = {
        "cik": "555", "company": "WhitespaceBank", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "item1":  "E" * 3000,
        "item1a": "   \n\t  ",   # whitespace only
        "char_count": 3100,
    }
    fp = tmp_path / "555_202212.json"
    fp.write_text(json.dumps(doc))

    kept, dropped = s0.filter_boilerplate()
    assert dropped == 1
    assert kept == 0


# ── EDGAR call mocking ────────────────────────────────────────────────────────

def test_extract_filing_mocked():
    """extract_filing calls edgar.Filing().obj() and returns correct record dict."""
    mock_tenk = MagicMock()
    mock_tenk.item1  = "Business description text. " * 100
    mock_tenk.item1a = "Risk factor: We are subject to BSA. " * 50

    mock_filing = MagicMock()
    mock_filing.obj.return_value = mock_tenk

    row = {
        "cik":         "12345",
        "company":     "Test Bank Corp",
        "sic":         "6022",
        "period":      "202212",
        "filing_date": "2023-03-01",
        "accession_no": "0000012345-23-000001",
    }

    with patch.dict("sys.modules", {"edgar": MagicMock()}):
        import edgar as mock_edgar
        mock_edgar.Filing.return_value = mock_filing

        record = s0.extract_filing(row)

    assert record is not None
    assert record["cik"]     == "12345"
    assert record["company"] == "Test Bank Corp"
    assert len(record["item1"])  > 0
    assert len(record["item1a"]) > 0
    assert record["char_count"] == len(record["item1"]) + len(record["item1a"])


def test_extract_filing_no_items_returns_none():
    """extract_filing returns None when both item1 and item1a are empty."""
    mock_tenk = MagicMock()
    mock_tenk.item1  = ""
    mock_tenk.item1a = ""

    mock_filing = MagicMock()
    mock_filing.obj.return_value = mock_tenk

    row = {
        "cik": "99999", "company": "EmptyFiler", "sic": "6022",
        "period": "202212", "filing_date": "2023-01-01",
        "accession_no": "ACC99999",
    }

    with patch.dict("sys.modules", {"edgar": MagicMock()}):
        import edgar as mock_edgar
        mock_edgar.Filing.return_value = mock_filing
        result = s0.extract_filing(row)

    assert result is None


def test_acquire_corpus_skips_existing(tmp_path, monkeypatch):
    """acquire_corpus skips files that already exist (checkpoint behavior)."""
    monkeypatch.setattr(s0, "RAW_DIR", tmp_path)

    cik    = "77777"
    period = "202212"
    existing = tmp_path / f"{cik}_{period}.json"
    existing.write_text(json.dumps({"cik": cik, "item1": "x", "item1a": "y", "char_count": 3000}))

    corpus = pd.DataFrame([{
        "cik": cik, "company": "ExistingBank", "sic": "6022",
        "filing_date": "2023-01-01", "accession_no": "ACC77777",
        "period": f"{period[:4]}-{period[4:6]}",
    }])

    call_count = {"n": 0}
    original_extract = s0.extract_filing

    def mock_extract(row):
        call_count["n"] += 1
        return original_extract(row)

    monkeypatch.setattr(s0, "extract_filing", mock_extract)

    # Should not call extract_filing at all since file exists
    failures = s0.acquire_corpus(corpus)
    assert call_count["n"] == 0
    assert failures == []
