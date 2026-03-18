"""
stage0_acquire.py — SEC EDGAR 10-K corpus acquisition for FinReg10K.

Stage 0: Identify target filings, deduplicate, extract Item 1 + Item 1A text,
         and filter boilerplate documents.

Outputs:
  finreg10k_index.csv         — all SIC-filtered filings found
  finreg10k_corpus_index.csv  — deduplicated corpus (one per firm, most recent)
  finreg10k_raw/              — extracted JSON documents
  extraction_failures.json    — CIKs that failed extraction

Checkpoint: file-level. Already-extracted files are skipped on re-run.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── SIC codes from spec Section 2 ─────────────────────────────────────────────
FINANCIAL_SICS = [
    "6020", "6021", "6022", "6025", "6026", "6027", "6029",
    "6035", "6036",
    "6099",
    "6141",
    "6153", "6154", "6159",
    "6200", "6211",
    "6311", "6321",
    "6411",
]

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
RAW_DIR      = BASE_DIR / "finreg10k_raw"
INDEX_CSV    = BASE_DIR / "finreg10k_index.csv"
CORPUS_CSV   = BASE_DIR / "finreg10k_corpus_index.csv"
FAILURES_JSON = BASE_DIR / "extraction_failures.json"

CORPUS_CAP  = 200   # max filings — parity with FinWiki 106-doc corpus
SEC_SLEEP   = 0.3   # seconds between EDGAR requests (SEC rate-limit policy)
BOILERPLATE_THRESHOLD = 2000  # chars


# ── Step 1: Build index from EDGAR ────────────────────────────────────────────

def build_index() -> pd.DataFrame:
    """
    Query EDGAR for all 10-K filings 2018-2023 from the SIC codes in scope.
    Returns a DataFrame with columns: cik, company, sic, filing_date,
    accession_no, period.
    """
    try:
        from edgar import get_filings
        import edgar
    except ImportError:
        logger.error("edgartools not installed. Run: pip install edgartools")
        sys.exit(1)

    # SEC requires a User-Agent identity: https://www.sec.gov/os/accessing-edgar-data
    identity = os.environ.get("EDGAR_IDENTITY", "FinReg10K Research dioufseck@gmail.com")
    edgar.set_identity(identity)

    records: List[Dict[str, Any]] = []

    for year in range(2018, 2024):
        for quarter in range(1, 5):
            logger.info(f"  Querying {year} Q{quarter} ...")
            try:
                filings = get_filings(year, quarter, form="10-K")
            except Exception as e:
                logger.warning(f"  get_filings({year}, {quarter}) failed: {e}")
                continue

            for filing in filings:
                # edgartools v5: SIC is on the entity, not the filing object
                try:
                    sic_str = str(getattr(filing, "sic", None) or filing.get_entity().sic or "").strip()
                except Exception:
                    sic_str = ""
                if sic_str not in FINANCIAL_SICS:
                    continue
                records.append({
                    "cik":          str(getattr(filing, "cik", "")),
                    "company":      str(getattr(filing, "company", "")),
                    "sic":          sic_str,
                    "filing_date":  str(getattr(filing, "filing_date", "")),
                    "accession_no": str(getattr(filing, "accession_no", "")),
                    "period":       str(getattr(filing, "period_of_report", "")),
                })

    if not records:
        logger.warning("No filings found — check edgartools and EDGAR connectivity")
        return pd.DataFrame(columns=["cik", "company", "sic", "filing_date", "accession_no", "period"])

    df = pd.DataFrame(records).drop_duplicates(subset=["cik", "period"])
    df.to_csv(INDEX_CSV, index=False)
    logger.info(f"Index written: {INDEX_CSV}  ({len(df)} rows)")
    return df


# ── Step 2: Deduplicate to one filing per firm ────────────────────────────────

def deduplicate(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain the most recent filing per CIK (firm), cap at CORPUS_CAP.
    Returns the corpus DataFrame.
    """
    if index_df.empty:
        return index_df

    corpus = (
        index_df
        .sort_values("filing_date", ascending=False)
        .drop_duplicates(subset=["cik"])
        .reset_index(drop=True)
        .head(CORPUS_CAP)
    )
    corpus.to_csv(CORPUS_CSV, index=False)
    logger.info(f"Corpus index written: {CORPUS_CSV}  ({len(corpus)} firms)")
    return corpus


# ── Step 3: Extract Item 1 and Item 1A text ───────────────────────────────────

def extract_filing(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Download and extract Item 1 and Item 1A from a single 10-K filing.
    Returns a record dict or None on failure.
    """
    try:
        from edgar import Filing
    except ImportError:
        raise RuntimeError("edgartools not installed")

    filing = Filing(
        form="10-K",
        filing_date=row["filing_date"],
        company=row["company"],
        cik=row["cik"],
        accession_no=row["accession_no"],
    )
    tenk = filing.obj()

    item1_text  = str(getattr(tenk, "item1",  "") or "")
    item1a_text = str(getattr(tenk, "item1a", "") or "")

    if not item1_text and not item1a_text:
        return None

    return {
        "cik":         row["cik"],
        "company":     row["company"],
        "sic":         row["sic"],
        "period":      row["period"],
        "filing_date": row["filing_date"],
        "accession_no": row["accession_no"],
        "item1":       item1_text,
        "item1a":      item1a_text,
        "char_count":  len(item1_text) + len(item1a_text),
    }


def acquire_corpus(corpus_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Iterate over corpus_df rows, extract text, write JSON files.
    Skips already-extracted files (file-level checkpoint).
    Returns list of failure records.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, Any]] = []

    rows = corpus_df.to_dict("records")
    logger.info(f"Acquiring {len(rows)} filings ...")

    for row in tqdm(rows, desc="Extracting 10-Ks"):
        cik    = str(row["cik"])
        period = str(row["period"]).replace("-", "")[:6]  # YYYYMM
        out_path = RAW_DIR / f"{cik}_{period}.json"

        if out_path.exists():
            logger.debug(f"  Skip (exists): {out_path.name}")
            continue

        try:
            record = extract_filing(row)
        except Exception as e:
            logger.warning(f"  FAILED {cik}: {e}")
            failures.append({"cik": cik, "reason": str(e)})
            continue

        if record is None:
            logger.warning(f"  No items extracted: {cik}")
            failures.append({"cik": cik, "reason": "no_items_extracted"})
            continue

        try:
            out_path.write_text(json.dumps(record, indent=2))
            logger.debug(f"  Saved: {out_path.name}  ({record['char_count']} chars)")
        except OSError as e:
            logger.warning(f"  Write failed {cik}: {e}")
            failures.append({"cik": cik, "reason": f"write_error: {e}"})
            continue

        time.sleep(SEC_SLEEP)

    return failures


# ── Step 4: Filter boilerplate documents ──────────────────────────────────────

def filter_boilerplate() -> tuple[int, int]:
    """
    Remove documents where char_count < 2000 or item1a is empty.
    Modifies the raw directory in-place.
    Returns (kept_count, dropped_count).
    """
    kept = 0
    dropped = 0

    for fp in list(RAW_DIR.glob("*.json")):
        try:
            doc = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"  Could not read {fp.name}: {e}")
            fp.unlink(missing_ok=True)
            dropped += 1
            continue

        char_count  = doc.get("char_count", 0)
        item1a_text = doc.get("item1a", "").strip()

        if char_count < BOILERPLATE_THRESHOLD or not item1a_text:
            logger.info(f"  Dropping boilerplate: {fp.name}  (chars={char_count}, item1a={'empty' if not item1a_text else 'ok'})")
            fp.unlink()
            dropped += 1
        else:
            kept += 1

    return kept, dropped


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger.info("=== Stage 0: Corpus Acquisition ===")

    # Step 1: Build / reload index
    if INDEX_CSV.exists():
        logger.info(f"  Loading existing index: {INDEX_CSV}")
        index_df = pd.read_csv(INDEX_CSV, dtype=str)
        logger.info(f"  Index loaded: {len(index_df)} filings")
    else:
        logger.info("  Building EDGAR index (this may take several minutes) ...")
        index_df = build_index()

    if index_df.empty:
        logger.error("  Empty index — aborting")
        sys.exit(1)

    # Step 2: Deduplicate
    if CORPUS_CSV.exists():
        logger.info(f"  Loading existing corpus index: {CORPUS_CSV}")
        corpus_df = pd.read_csv(CORPUS_CSV, dtype=str)
    else:
        corpus_df = deduplicate(index_df)

    logger.info(f"  Corpus size: {len(corpus_df)} firms")

    # Step 3: Extract text
    failures = acquire_corpus(corpus_df)

    # Write failures
    FAILURES_JSON.write_text(json.dumps(failures, indent=2))
    logger.info(f"  Extraction failures written: {FAILURES_JSON}  ({len(failures)} failures)")

    # Step 4: Filter boilerplate
    kept, dropped = filter_boilerplate()
    logger.info(f"  Boilerplate filter: kept={kept} dropped={dropped}")

    # Summary
    raw_count = len(list(RAW_DIR.glob("*.json")))
    logger.info(f"=== Stage 0 complete: {raw_count} documents in {RAW_DIR} ===")


if __name__ == "__main__":
    run()
