"""tests/test_chunker.py — Tests for Stage 2 chunking logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.stage2_chunk import chunk_text, split_by_sections, chunk_article

MAX_T  = 400
OVER   = 50


# ── chunk_text ─────────────────────────────────────────────────────────────────

def test_short_text_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, max_tokens=MAX_T, overlap=OVER)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_multiple_chunks():
    text = "word " * 600
    chunks = chunk_text(text, max_tokens=MAX_T, overlap=OVER)
    assert len(chunks) >= 2


def test_chunk_max_length():
    text = "word " * 600
    chunks = chunk_text(text, max_tokens=MAX_T, overlap=OVER)
    for c in chunks:
        assert len(c.split()) <= MAX_T + 5  # +5 tolerance for boundary rounding


def test_overlap_content():
    """Last <overlap> words of chunk N should appear at start of chunk N+1."""
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(text, max_tokens=200, overlap=50)
    assert len(chunks) > 2
    last_50_of_first  = chunks[0].split()[-50:]
    first_50_of_second = chunks[1].split()[:50]
    assert last_50_of_first == first_50_of_second


def test_exactly_max_tokens():
    text = "word " * MAX_T
    chunks = chunk_text(text, max_tokens=MAX_T, overlap=OVER)
    assert len(chunks) == 1


def test_chunk_text_empty():
    chunks = chunk_text("", max_tokens=MAX_T, overlap=OVER)
    assert chunks == [""] or len(chunks) == 1


# ── split_by_sections ──────────────────────────────────────────────────────────

def test_split_no_sections():
    text = "Just a plain paragraph without headings."
    sections = split_by_sections(text)
    assert len(sections) == 1
    assert sections[0] == text


def test_split_two_sections():
    text = "Intro text\n== Section One ==\nContent one here.\n== Section Two ==\nContent two here."
    sections = split_by_sections(text)
    assert len(sections) >= 2
    assert any("Content one here" in s for s in sections)
    assert any("Content two here" in s for s in sections)


def test_split_preserves_heading():
    text = "== Capital Requirements ==\nBanks must hold adequate capital."
    sections = split_by_sections(text)
    # Heading should be included in section text
    combined = " ".join(sections)
    assert "Capital Requirements" in combined


def test_split_nested_headings():
    text = "== Level 2 ==\nContent\n=== Level 3 ===\nNested content"
    sections = split_by_sections(text)
    assert len(sections) >= 1


# ── chunk_article ──────────────────────────────────────────────────────────────

def test_chunk_article_basic():
    article = {
        "title":   "Basel III",
        "url":     "https://en.wikipedia.org/wiki/Basel_III",
        "content": "word " * 100,
        "domain":  "banking",
    }
    chunks = chunk_article("Basel_III", article)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "Basel_III"
        assert c.chunk_id    != ""
        assert c.content     != ""


def test_chunk_article_sequences():
    article = {
        "title":   "Test",
        "content": "== Section A ==\n" + "word " * 200 + "\n== Section B ==\n" + "word " * 200,
    }
    chunks = chunk_article("test_doc", article)
    sequences = [c.sequence for c in chunks]
    assert sequences == sorted(sequences)  # sequences must be non-decreasing
    assert len(set(c.chunk_id for c in chunks)) == len(chunks)  # all IDs unique
