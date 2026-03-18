"""
stage1_preprocess.py — Chunking and manifest generation for FinReg10K.

Stage 1: Split raw 10-K documents into chunks and build the manifest that
         feeds into the existing FinWiki pipeline stages 2-8.

Chunking strategy:
  - Item 1A (Risk Factors): split by risk factor headers (ALL CAPS, "Risk Factor N",
    bullet patterns)
  - Item 1 (Business): paragraph splitting on double newlines
  - Drop chunks < 200 chars

Outputs:
  finreg10k_chunks/             — per-document chunk JSON files
  finreg10k_manifest.json       — manifest in FinWiki manifest schema (stages 2-8 ready)

The manifest also writes individual chunk files per document into
data/chunks/ (relative to BASE_DIR) so the existing pipeline stages can
find them without modification.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RAW_DIR     = BASE_DIR / "finreg10k_raw"
CHUNK_DIR   = BASE_DIR / "finreg10k_chunks"
MANIFEST    = BASE_DIR / "finreg10k_manifest.json"

# Pipeline chunk directory — consumed by stages 3-8 (mirrors finwiki-kg layout)
PIPELINE_CHUNKS_DIR = BASE_DIR / "data" / "chunks"

# Minimum chunk size (chars) — matches spec Section 4
MIN_CHUNK_CHARS = 200

# ── Risk factor header patterns ────────────────────────────────────────────────
# Matches:
#   - ALL CAPS headers: lines of 10+ uppercase chars (with spaces/punctuation)
#   - "Risk Factor N" explicit numbering
#   - Bullet-initiated risks (• or unicode bullets)
HEADER_RE = re.compile(
    r'(?=\n[A-Z][A-Z\s,;:\-\.]{10,}\n)'  # ALL CAPS header (10+ chars)
    r'|(?=\nRisk Factor \d+)'              # "Risk Factor N" pattern
    r'|(?=\n[•●]\s)',                      # bullet-initiated risk
    re.MULTILINE,
)


# ── Chunking functions ─────────────────────────────────────────────────────────

def chunk_item1a(text: str, cik: str, period: str) -> List[Dict]:
    """
    Split Item 1A into individual risk factor chunks.

    Each headed risk factor block becomes one chunk.
    Chunks shorter than MIN_CHUNK_CHARS are dropped.

    chunk_id format: {cik}_{period}_1a_{i:04d}
    """
    raw_chunks = HEADER_RE.split(text)
    chunks: List[Dict] = []

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        chunks.append({
            "chunk_id":    f"{cik}_{period}_1a_{i:04d}",
            "source":      "item1a",
            "cik":         cik,
            "period":      period,
            "text":        chunk,
            "char_count":  len(chunk),
        })

    return chunks


def chunk_item1(text: str, cik: str, period: str) -> List[Dict]:
    """
    Split Item 1 (Business) into paragraph-level chunks.

    Analogous to FinWiki paragraph chunking.
    Splits on double newlines; drops paragraphs < MIN_CHUNK_CHARS chars.

    chunk_id format: {cik}_{period}_1_{i:04d}
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_CHUNK_CHARS]
    return [
        {
            "chunk_id":    f"{cik}_{period}_1_{i:04d}",
            "source":      "item1",
            "cik":         cik,
            "period":      period,
            "text":        para,
            "char_count":  len(para),
        }
        for i, para in enumerate(paragraphs)
    ]


def chunks_to_pipeline_format(
    raw_chunks: List[Dict],
    doc_id: str,
    company: str,
    cik: str,
    period: str,
    domain: str = "banking",
) -> List[Dict]:
    """
    Convert FinReg10K raw chunks into the format expected by the FinWiki
    pipeline stages 3-8.

    FinWiki chunk file format (from stage2_chunk.py / stage3_classify.py):
      chunk_id, document_id, sequence, section_title, content,
      token_estimate, title, url, domain
    """
    pipeline_chunks = []
    for seq, rc in enumerate(raw_chunks):
        token_estimate = max(1, len(rc["text"].split()))
        pipeline_chunks.append({
            "chunk_id":       rc["chunk_id"],
            "document_id":    doc_id,
            "sequence":       seq,
            "section_title":  rc.get("source", ""),   # "item1a" or "item1"
            "content":        rc["text"],
            "token_estimate": token_estimate,
            "title":          company,
            "url":            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
            "domain":         domain,
            # FinReg10K extras (ignored by downstream pipeline)
            "cik":            cik,
            "period":         period,
        })
    return pipeline_chunks


def infer_domain(sic: str) -> str:
    """Map SIC code to domain label used by FinWiki pipeline."""
    sic_int = int(sic) if sic.isdigit() else 0
    if 6020 <= sic_int <= 6099:
        return "banking"
    if 6141 <= sic_int <= 6159:
        return "banking"
    if 6200 <= sic_int <= 6211:
        return "securities"
    if 6311 <= sic_int <= 6411:
        return "insurance"
    return "finance"


# ── Stage runner ──────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger.info("=== Stage 1: Preprocessing and Chunking ===")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        logger.warning(f"No raw files found in {RAW_DIR} — run stage0_acquire.py first")
        return

    all_chunks: List[Dict] = []
    doc_chunks: Dict[str, List[Dict]] = defaultdict(list)
    doc_count = 0

    for fp in raw_files:
        try:
            doc = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"  Skip {fp.name}: {e}")
            continue

        cik    = str(doc.get("cik", ""))
        period = str(doc.get("period", "")).replace("-", "")[:6]
        company = str(doc.get("company", f"CIK_{cik}"))
        sic     = str(doc.get("sic", "6020"))
        domain  = infer_domain(sic)
        doc_id  = f"{cik}_{period}"

        # Chunk both sections
        raw_1a = chunk_item1a(doc.get("item1a", ""), cik, period)
        raw_1  = chunk_item1(doc.get("item1",  ""), cik, period)
        raw    = raw_1a + raw_1

        if not raw:
            logger.warning(f"  No chunks produced: {fp.name}")
            continue

        # Write per-document FinReg10K chunk file
        chunk_out = CHUNK_DIR / f"{cik}_{period}_chunks.json"
        chunk_out.write_text(json.dumps(raw, indent=2))

        # Convert to pipeline format and write to data/chunks/
        pipeline_chunks = chunks_to_pipeline_format(raw, doc_id, company, cik, period, domain)
        pipeline_out = PIPELINE_CHUNKS_DIR / f"{cik}_{period}.json"
        pipeline_out.write_text(json.dumps(pipeline_chunks, indent=2))

        for rc in raw:
            doc_chunks[doc_id].append(rc)
        all_chunks.extend(raw)
        doc_count += 1
        logger.debug(f"  {doc_id}: {len(raw)} chunks ({len(raw_1a)} item1a, {len(raw_1)} item1)")

    # Build manifest
    manifest = []
    for doc_id, chunks in doc_chunks.items():
        manifest.append({
            "doc_id":      doc_id,
            "corpus":      "FinReg10K",
            "chunk_count": len(chunks),
            "chunks":      chunks,
            "metadata": {
                "cik":    chunks[0]["cik"],
                "period": chunks[0]["period"],
                "source": "SEC EDGAR 10-K",
            },
        })

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # Stats
    total_chunks = len(all_chunks)
    if total_chunks > 0:
        avg_len = sum(c["char_count"] for c in all_chunks) / total_chunks
    else:
        avg_len = 0.0

    logger.info(f"=== Stage 1 complete ===")
    logger.info(f"  Documents processed: {doc_count}")
    logger.info(f"  Total chunks:        {total_chunks}")
    logger.info(f"  Avg chunk length:    {avg_len:.0f} chars")
    logger.info(f"  Manifest written:    {MANIFEST}")
    logger.info(f"  Pipeline chunks dir: {PIPELINE_CHUNKS_DIR}")

    print(f"\nStage 1 stats:")
    print(f"  Document count:   {doc_count}")
    print(f"  Chunk count:      {total_chunks}")
    print(f"  Avg chunk length: {avg_len:.0f} chars")


if __name__ == "__main__":
    run()
