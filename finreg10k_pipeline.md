# FinReg10K: EDGAR Instantiation Pipeline
## Second Corpus Implementation Guide for Epistemic Type Alignment Paper

This document specifies the implementation of a second corpus instantiation using SEC
EDGAR 10-K filings. FinWiki remains the primary instantiation. FinReg10K is the
replication corpus, added to Section 5 and Section 6 to demonstrate that the
architecture's properties generalize from encyclopedic financial text (FinWiki) to
primary regulatory disclosure text (FinReg10K).

The theoretical and architectural contribution is unchanged. The pipeline (Stages 1–8)
is unchanged. All work is in corpus acquisition, preprocessing, and evaluation harness
adaptation.

---

## 1. Motivation and Expected Contribution

### Why a second corpus strengthens the paper

FinWiki's encyclopedic composition — 87% constative by assertion count — understates
the normative density of primary enterprise corpora in regulated industries. This is
the first limitation stated in Section 7.3. A reviewer reading that limitation will
ask: does the architecture perform better or differently on a corpus that actually
looks like what regulated enterprises have?

FinReg10K answers that question directly. 10-K Item 1A (Risk Factors) and Item 1
(Business) sections contain dense normative language: regulatory obligations, compliance
requirements, and enforcement risk disclosures. The architecture's warrant monopoly
and conflict detection properties should be more pronounced on this corpus, not less.

### Concrete claims the second corpus enables

| Claim | FinWiki alone | FinReg10K adds |
|---|---|---|
| Architecture generalizes beyond encyclopedic text | Cannot claim | Demonstrated |
| Warrant monopoly appears in primary regulatory corpora | Hypothesized | Tested |
| Normative query VTP improvement holds under higher normative density | Shown for sparse case | Shown for dense case |
| Conflict detection rate > 16.7% when normative content is primary | Predicted in §7.3 | Measured |
| Effect sizes are corpus-composition dependent | Argued theoretically | Empirically decomposed |

---

## 2. Corpus Design

### Target filing population

```
Form type:   10-K (annual report)
Sections:    Item 1 (Business) + Item 1A (Risk Factors)
SIC codes:   6020–6029  Commercial banks
             6035–6036  Savings institutions
             6099       Functions related to depository banking
             6141       Personal credit institutions
             6153–6159  Short-term business credit, federal-sponsored credit
             6200–6211  Security and commodity brokers
             6311–6321  Life and accident insurance
             6411       Insurance agents and brokers
Date range:  2018–2023  (post-Dodd-Frank maturity, pre-SVB)
Target size: ~200 filings (comparability with FinWiki's 106 documents)
```

### Why Item 1 and Item 1A specifically

Item 1A (Risk Factors) is the highest-normative-density section of any 10-K. It is
structured as a series of declared obligations and failure conditions:

- "We are subject to extensive regulation..."
- "Failure to comply with the BSA or OFAC regulations could result in..."
- "We must maintain capital ratios above regulatory minimums..."

This is exactly the normative-warrant-heavy structure the architecture is designed to
index. Item 1 (Business) provides the constative substrate — descriptions of what the
firm does — that grounds the normative claims in Item 1A. Together they replicate the
constative/normative mixture at much higher normative density than FinWiki.

Item 7 (MD&A) is excluded: it is primarily constative financial narrative with
expressive forward-looking statements and adds noise without normative signal.

### Expected corpus composition vs. FinWiki

| Property | FinWiki | FinReg10K (predicted) |
|---|---|---|
| Constative % | ~87% | ~55–65% |
| Normative % | ~8% | ~25–35% |
| Expressive % | ~5% | ~10–15% |
| Assertions per document | ~120 | ~80–150 |
| CONTRADICTS edge density | Low | Higher (multi-firm, regulatory change) |
| Warrant monopoly Gini | ~1.00 (Basel III cluster) | Expected ~0.85–1.00 |

---

## 3. Stage 0 — Corpus Acquisition

### Dependencies

```bash
pip install edgartools pandas tqdm
```

### 3.1 Identify target filings

```python
from edgar import get_filings
import pandas as pd

FINANCIAL_SICS = [
    "6020", "6021", "6022", "6025", "6026", "6027", "6029",
    "6035", "6036", "6099",
    "6141", "6153", "6154", "6159",
    "6200", "6211",
    "6311", "6321",
    "6411"
]

records = []

for year in range(2018, 2024):
    for quarter in range(1, 5):
        filings = get_filings(year, quarter, form="10-K")
        for filing in filings:
            if str(filing.sic) in FINANCIAL_SICS:
                records.append({
                    "cik":          filing.cik,
                    "company":      filing.company,
                    "sic":          filing.sic,
                    "filing_date":  filing.filing_date,
                    "accession_no": filing.accession_no,
                    "period":       filing.period_of_report,
                })

index_df = pd.DataFrame(records).drop_duplicates(subset=["cik", "period"])
index_df.to_csv("finreg10k_index.csv", index=False)
print(f"Total filings identified: {len(index_df)}")
```

### 3.2 Deduplicate to one filing per firm

For firms with multiple years, retain the most recent filing per CIK unless
multi-year comparison is the explicit goal (see Section 6 note on cross-year
conflict detection).

```python
# One filing per firm — most recent
deduped = (
    index_df
    .sort_values("filing_date", ascending=False)
    .drop_duplicates(subset=["cik"])
    .reset_index(drop=True)
)

# Cap at 200 filings for comparability with FinWiki
corpus_df = deduped.head(200)
corpus_df.to_csv("finreg10k_corpus_index.csv", index=False)
print(f"Corpus size: {len(corpus_df)} firms")
```

### 3.3 Extract Item 1 and Item 1A text

```python
from edgar import Filing
from pathlib import Path
import json, time

OUTPUT_DIR = Path("./finreg10k_raw/")
OUTPUT_DIR.mkdir(exist_ok=True)
FAILED = []

for _, row in corpus_df.iterrows():
    out_path = OUTPUT_DIR / f"{row['cik']}_{row['period']}.json"
    if out_path.exists():
        continue  # checkpoint — skip already processed

    try:
        filing = Filing(
            form="10-K",
            filing_date=row["filing_date"],
            company=row["company"],
            cik=row["cik"],
            accession_no=row["accession_no"]
        )
        tenk = filing.obj()

        item1_text  = tenk.item1  if hasattr(tenk, "item1")  else ""
        item1a_text = tenk.item1a if hasattr(tenk, "item1a") else ""

        if not item1_text and not item1a_text:
            FAILED.append({"cik": row["cik"], "reason": "no_items_extracted"})
            continue

        record = {
            "cik":         row["cik"],
            "company":     row["company"],
            "sic":         row["sic"],
            "period":      row["period"],
            "filing_date": row["filing_date"],
            "item1":       str(item1_text),
            "item1a":      str(item1a_text),
            "char_count":  len(str(item1_text)) + len(str(item1a_text)),
        }

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        time.sleep(0.3)  # respect SEC rate limits

    except Exception as e:
        FAILED.append({"cik": row["cik"], "reason": str(e)})

print(f"Extracted: {len(list(OUTPUT_DIR.glob('*.json')))}")
print(f"Failed:    {len(FAILED)}")

with open("extraction_failures.json", "w") as f:
    json.dump(FAILED, f, indent=2)
```

### 3.4 Filter boilerplate documents

Replicate the FinWiki navigation filtering step. Drop documents where:
- `char_count` < 2000 (pure boilerplate, no substantive content)
- Item 1A is empty (incorporation by reference or pre-mandatory period)

```python
raw_files = list(Path("./finreg10k_raw/").glob("*.json"))
kept, dropped = [], []

for fp in raw_files:
    with open(fp) as f:
        doc = json.load(f)

    if doc["char_count"] < 2000 or not doc["item1a"].strip():
        dropped.append(doc["cik"])
        fp.unlink()
    else:
        kept.append(doc)

print(f"Kept: {len(kept)} | Dropped: {len(dropped)}")
```

---

## 4. Stage 1 — Preprocessing and Chunking

### 4.1 Section-aware chunking

Unlike FinWiki (Wikipedia articles chunked by paragraph), 10-K Item 1A sections are
structured with numbered risk factor headers. Use this structure as the primary chunk
boundary — each risk factor is a natural substantive unit analogous to a FinWiki
paragraph chunk.

```python
import re
import json
from pathlib import Path

def chunk_item1a(text: str, cik: str, period: str) -> list[dict]:
    """
    Split Item 1A into individual risk factor chunks.
    Each headed risk factor block becomes one chunk.
    """
    pattern = re.compile(
        r'(?=\n[A-Z][A-Z\s,;]{10,}\n)'   # ALL CAPS headers
        r'|(?=\nRisk Factor \d+)'          # "Risk Factor N" pattern
        r'|(?=\n•\s)',                     # bullet-initiated risks
        re.MULTILINE
    )

    raw_chunks = pattern.split(text)
    chunks = []

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if len(chunk) < 200:
            continue
        chunks.append({
            "chunk_id":   f"{cik}_{period}_1a_{i:04d}",
            "source":     "item1a",
            "cik":        cik,
            "period":     period,
            "text":       chunk,
            "char_count": len(chunk),
        })

    return chunks


def chunk_item1(text: str, cik: str, period: str) -> list[dict]:
    """
    Split Item 1 (Business) into paragraph-level chunks.
    Analogous to FinWiki paragraph chunking.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    return [
        {
            "chunk_id":   f"{cik}_{period}_1_{i:04d}",
            "source":     "item1",
            "cik":        cik,
            "period":     period,
            "text":       para,
            "char_count": len(para),
        }
        for i, para in enumerate(paragraphs)
    ]


CHUNK_DIR = Path("./finreg10k_chunks/")
CHUNK_DIR.mkdir(exist_ok=True)
all_chunks = []

for fp in Path("./finreg10k_raw/").glob("*.json"):
    with open(fp) as f:
        doc = json.load(f)

    chunks = (
        chunk_item1a(doc["item1a"], doc["cik"], doc["period"]) +
        chunk_item1(doc["item1"],   doc["cik"], doc["period"])
    )

    out_path = CHUNK_DIR / f"{doc['cik']}_{doc['period']}_chunks.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")
print(f"Avg chunk length: {sum(c['char_count'] for c in all_chunks) / len(all_chunks):.0f} chars")
```

### 4.2 Document manifest

Generate the manifest that feeds into Stage 2 (the existing pipeline entry point).
The manifest format must match the FinWiki manifest schema exactly so the downstream
pipeline requires no modification.

```python
from collections import defaultdict

doc_chunks = defaultdict(list)
for fp in Path("./finreg10k_chunks/").glob("*.json"):
    with open(fp) as f:
        chunks = json.load(f)
    for chunk in chunks:
        doc_chunks[f"{chunk['cik']}_{chunk['period']}"].append(chunk)

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
        }
    })

with open("finreg10k_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest: {len(manifest)} documents, "
      f"{sum(d['chunk_count'] for d in manifest)} chunks")
```

---

## 5. Stages 2–8 — Existing LLM Pipeline

**No changes to the pipeline code.**

Pass `finreg10k_manifest.json` as the corpus input in place of the FinWiki manifest.
All eight stages — assertion extraction, classification, relation mining, conflict
detection, regulation node construction, quality output generation — run unchanged.

### One adaptation: Regulation node gazetteer

The FinWiki pipeline seeded Regulation nodes from a financial regulation gazetteer.
The same gazetteer applies to FinReg10K and will yield denser matches because 10-K
filings cite regulatory instruments by name far more frequently than Wikipedia articles.
Add the following entries if not already present:

```yaml
# regulation_gazetteer.yaml — additions for FinReg10K
- "Bank Secrecy Act"
- "BSA"
- "OFAC"
- "Community Reinvestment Act"
- "CRA"
- "Volcker Rule"
- "CECL"
- "DFAST"
- "CCAR"
- "Consumer Financial Protection Bureau"
- "CFPB"
- "OCC"
- "Sarbanes-Oxley"
- "SOX"
```

### Carry forward the Stage 3b batch size fix

10-K Item 1A sections can be long. The batch size fix from FinWiki (groups of 50
assertions with overlap window) is required here. Do not remove it.

---

## 6. Stage E — Evaluation Harness Adaptation

The Section 5.2 evaluation design carries over with two adaptations.

### 6.1 Query stratification

FinWiki's normative stratum was underrepresented (8% of assertions), so 33/33/33
stratification was appropriate. FinReg10K's normative density is substantially higher.
Adjust the stratum weights to concentrate evaluation power where the corpora differ:

```python
QUERY_STRATA = {
    "normative":  40,   # "What must [firm type] do under [regulation]?"
    "constative": 40,   # "What does [firm type] do in [business area]?"
    "mixed":      20,   # "What are the requirements and practices for [topic]?"
}
# Total: 100 queries (vs. 120 for FinWiki — acceptable for a replication corpus)
```

### 6.2 Optional: cross-firm conflict stratum

FinReg10K contains filings from multiple firms covering the same regulations. The
pipeline may surface CONTRADICTS edges across documents — two firms asserting
incompatible interpretations of the same regulatory requirement. This is not present
in FinWiki (single-author encyclopedic articles). If observed, report it as a novel
finding: the architecture detects inter-firm regulatory disagreement as a first-class
quality signal.

### 6.3 Outcome measures — unchanged

All four measures (VTP, AC, CSR, CWE) apply without modification. Report them in a
parallel results table alongside the FinWiki results.

### 6.4 Expected results pattern

| Measure | FinWiki result | FinReg10K prediction | What this shows |
|---|---|---|---|
| VTP A vs D (all) | 0.458 vs 0.873 | Similar magnitude | Architecture generalizes |
| VTP normative stratum | 0.115 vs 1.000 | Smaller gap | Effect size is corpus-composition dependent, as predicted |
| CSR (conflict surface rate) | 16.7% | Higher | More conflicts in primary regulatory text |
| Warrant monopoly Gini | 1.00 (Basel III) | ~0.85–1.00 | Warrant monopoly holds in primary regulatory corpora |
| AC two-pass vs validity-gated | d=0.375 | Similar or larger | Two-pass benefit holds across corpus types |

The key narrative: FinWiki demonstrates the architecture under adverse conditions
(sparse normative content). FinReg10K demonstrates it under conditions closer to
actual enterprise regulatory corpora. The architecture should perform better on
FinReg10K, not worse — and if it does, that resolves the primary limitation.

---

## 7. Paper Integration

### Where FinReg10K appears in the paper

| Section | Change |
|---|---|
| §1 Introduction | One sentence: "Results replicate on FinReg10K, a second corpus of 200 SEC 10-K filings from regulated financial firms." |
| §5.1 | New subsection 5.2: "The FinReg10K Corpus" — mirrors the FinWiki corpus description |
| §5.2 Evaluation Design | Note that the 120-query harness runs on both corpora |
| §6 Results | Parallel column or parallel table for each DP reporting both corpora |
| §7.1 Theoretical contributions | Strengthen generalizability claim: "Results replicate on primary regulatory text, confirming that epistemic type alignment is a property of the architecture, not of corpus composition." |
| §7.3 Limitations | Remove or substantially weaken Limitation 1 (encyclopedic corpus) |

### Results table structure

Report FinWiki and FinReg10K side by side for each design proposition. Example for
Table 2 (VTP):

| Condition | FinWiki VTP | FinReg10K VTP | FinWiki Norm. | FinReg10K Norm. |
|---|---|---|---|---|
| A — Standard RAG | 0.458 | TBD | 0.115 | TBD |
| D — Full discourse-typed | 0.873 | TBD | 0.980 | TBD |
| E — Two-pass | 0.858 | TBD | 1.000 | TBD |
| A vs D: p, d | p<0.001, d=0.776 | TBD | — | — |

---

## 8. Implementation Checklist

```
CORPUS ACQUISITION
  [ ] Run SIC-filtered 10-K index query
  [ ] Deduplicate to one filing per firm (most recent)
  [ ] Extract Item 1 and Item 1A text for ~200 firms
  [ ] Filter boilerplate / empty Item 1A documents
  [ ] Log extraction failures → extraction_failures.json

PREPROCESSING
  [ ] Chunk Item 1A by risk factor header
  [ ] Chunk Item 1 by paragraph
  [ ] Drop chunks < 200 chars
  [ ] Generate finreg10k_manifest.json in FinWiki manifest schema
  [ ] Verify manifest: document count, chunk count, avg chunk length

PIPELINE
  [ ] Add financial regulation gazetteer entries
  [ ] Confirm Stage 3b batch size fix is active (groups of 50)
  [ ] Run Stages 1–8 on FinReg10K manifest
  [ ] Verify: assertion count, relation count, CONTRADICTS edge count
  [ ] Compare assertion type distribution to FinWiki (expected: higher normative %)

EVALUATION
  [ ] Generate 100 queries (40 normative / 40 constative / 20 mixed)
  [ ] Run all five retrieval conditions
  [ ] Compute VTP, AC, CSR, CWE
  [ ] Compare to FinWiki results
  [ ] Check for cross-firm CONTRADICTS edges — report if observed

PAPER
  [ ] Add §5.2 FinReg10K corpus description
  [ ] Add parallel results columns to Tables 2, 3, 4
  [ ] Update §7.1 to strengthen generalizability claim
  [ ] Update §7.3 to retire Limitation 1
  [ ] Update abstract with FinReg10K headline numbers
```

---

## 9. Reproducibility Notes

SEC EDGAR is publicly available without authentication. All 10-K filings are public
disclosures. The `edgartools` library accesses EDGAR directly — no WRDS subscription,
no institutional access, fully reproducible by any reviewer.

The corpus is defined deterministically by SIC code list, date range, deduplication
rule (most recent filing per CIK), and extraction failure log. Any researcher with
`edgartools` installed can reconstruct the identical corpus from `finreg10k_corpus_index.csv`.

Commit to the replication package:
- `finreg10k_index.csv`
- `finreg10k_corpus_index.csv`
- `extraction_failures.json`
- `regulation_gazetteer.yaml`

Do not commit raw filing text to the repository — the acquisition script is sufficient
for reproduction.
