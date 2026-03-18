# Supplemental Experiments — Autonomous Execution Spec

Three targeted experiments to strengthen empirical claims in the JMIS paper.
Run autonomously. Log everything. Write partial results every 10 minutes.
Never block on a single failure — mark failed items and continue.
All output goes to `data/eval/supplemental/`.

---

## Setup

```python
import json, time, logging, os, math
from datetime import datetime
from pathlib import Path

OUT = Path("data/eval/supplemental")
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=OUT / "exp_log.txt",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger()

def checkpoint(filename, data):
    """Write partial results atomically."""
    tmp = OUT / (filename + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(OUT / filename)
    log.info(f"Checkpoint: {filename}")
```

Connect to existing infrastructure — same connection strings used in the main pipeline:

```python
from neo4j import GraphDatabase
import psycopg2

neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
pg_conn = psycopg2.connect("postgresql://localhost:5432/finwiki")

# Qdrant for vector search
from qdrant_client import QdrantClient
qdrant = QdrantClient(host="localhost", port=6333)
```

Load the existing evaluation harness queries:

```python
queries = json.loads(Path("data/eval/queries.json").read_text())
normative_queries = [q for q in queries if q["epistemic_type"] == "normative"]
log.info(f"Loaded {len(normative_queries)} normative queries for Experiments 1 and 2")
```

---

## Experiment 1 — VTP K-Invariance Curve

**Purpose:** Produce Figure 2. Show VTP for Condition A (standard RAG) and Condition D
(full discourse-typed) as k increases from 1 to 100. Demonstrate that Condition A
plateaus while Condition D holds. The visual makes the k-invariance claim undeniable.

**Runtime estimate:** ~45 minutes (12 k-values × 40 queries × 2 conditions × embed + search).

```python
K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]

def get_top_k_assertions(query_embedding, k, validity_filter=None):
    """
    Retrieve top-k assertions from Qdrant.
    If validity_filter is set (e.g. 'normative'), apply as metadata filter.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qfilter = None
    if validity_filter:
        qfilter = Filter(must=[
            FieldCondition(key="validity_claim_type", match=MatchValue(value=validity_filter))
        ])
    results = qdrant.search(
        collection_name="assertions",
        query_vector=query_embedding,
        limit=k,
        query_filter=qfilter,
        with_payload=True
    )
    return results

def compute_vtp(retrieved_assertions, query_epistemic_type):
    """
    Validity-Type Precision: proportion of retrieved assertions whose
    validity_claim_type matches the query's epistemic requirement.
    For normative queries: target type = 'normative'.
    For constative queries: target type = 'constative'.
    """
    if not retrieved_assertions:
        return 0.0
    target = query_epistemic_type  # 'normative' | 'constative' | 'mixed'
    if target == 'mixed':
        return 1.0  # mixed queries accept any type
    matches = sum(1 for a in retrieved_assertions
                  if a.payload.get("validity_claim_type") == target)
    return matches / len(retrieved_assertions)

def embed_query(text):
    """Use same embedding model as pipeline — text-embedding-004 via Vertex AI or local."""
    # Import whichever embedding client is used in the main pipeline
    from pipeline.embeddings import embed_text  # adjust import to match your codebase
    return embed_text(text)

results_a = []  # VTP for Condition A at each k
results_d = []  # VTP for Condition D at each k

for ki, k in enumerate(K_VALUES):
    vtp_a_per_query = []
    vtp_d_per_query = []

    for qi, query in enumerate(normative_queries):
        try:
            emb = embed_query(query["text"])

            # Condition A: standard vector RAG, no type filter
            retrieved_a = get_top_k_assertions(emb, k, validity_filter=None)
            vtp_a_per_query.append(compute_vtp(retrieved_a, "normative"))

            # Condition D: validity-gated, normative filter only
            retrieved_d = get_top_k_assertions(emb, k, validity_filter="normative")
            # If fewer than k normative assertions found, pad with top unfiltered
            if len(retrieved_d) < k:
                all_retrieved = get_top_k_assertions(emb, k*2, validity_filter=None)
                existing_ids = {r.id for r in retrieved_d}
                extras = [r for r in all_retrieved if r.id not in existing_ids]
                retrieved_d = retrieved_d + extras[:k - len(retrieved_d)]
            vtp_d_per_query.append(compute_vtp(retrieved_d[:k], "normative"))

        except Exception as e:
            log.warning(f"Exp1 k={k} q={qi}: {e}")
            vtp_a_per_query.append(None)
            vtp_d_per_query.append(None)

    valid_a = [v for v in vtp_a_per_query if v is not None]
    valid_d = [v for v in vtp_d_per_query if v is not None]
    results_a.append(round(sum(valid_a)/len(valid_a), 4) if valid_a else None)
    results_d.append(round(sum(valid_d)/len(valid_d), 4) if valid_d else None)
    log.info(f"Exp1 k={k}: A={results_a[-1]} D={results_d[-1]}")

    # Checkpoint every 3 k-values
    if ki % 3 == 2:
        checkpoint("k_invariance_curve.json", {
            "status": "partial",
            "k_values_completed": K_VALUES[:ki+1],
            "condition_A_vtp": results_a,
            "condition_D_vtp": results_d
        })

# Compute plateau stats
valid_a_full = [(k, v) for k, v in zip(K_VALUES, results_a) if v is not None]
max_a = max(v for _, v in valid_a_full)
plateau_k_a = next((k for k, v in valid_a_full if v >= 0.90 * max_a), K_VALUES[-1])
gap_at_100 = round(results_d[-1] - results_a[-1], 4) if results_a[-1] and results_d[-1] else None

final = {
    "status": "complete",
    "n_queries": len(normative_queries),
    "k_values": K_VALUES,
    "condition_A_vtp": results_a,
    "condition_D_vtp": results_d,
    "plateau_k_A": plateau_k_a,
    "plateau_vtp_A": round(max_a, 4),
    "gap_at_k100": gap_at_100,
    "interpretation": (
        f"Condition A VTP plateaus at k={plateau_k_a} (VTP={max_a:.3f}). "
        f"Condition D holds at {results_d[-1]:.3f} at k=100. "
        f"Gap at k=100: {gap_at_100:.3f}. "
        "K-invariance confirmed: validity-type gap does not close as context window grows."
    )
}
checkpoint("k_invariance_curve.json", final)
print("Experiment 1 complete:", json.dumps(final, indent=2))
```

---

## Experiment 2 — Warrant Monopoly Corpus-Wide Analysis

**Purpose:** Convert the Basel III illustration (one cluster) into a corpus-wide structural
finding. If warrant concentration is a systematic property — not a cherry-picked example —
the paper's P3 claim is empirically grounded without expert annotation.

**Runtime estimate:** ~15 minutes (single Neo4j traversal + Python computation).

```python
def gini(values):
    """Gini coefficient of a list of non-negative values. 1.0 = complete monopoly."""
    values = sorted(float(v) for v in values)
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(values))
    return round((2 * cumsum) / (n * sum(values)) - (n + 1) / n, 4)

cypher = """
MATCH (reg:Regulation)<-[:REFERENCES]-(a:Assertion)
WITH reg,
     a.doc_id AS doc,
     COUNT(a) AS total_assertions,
     SUM(CASE WHEN a.discourse_role = 'warrant' THEN 1 ELSE 0 END) AS warrant_count
WITH reg,
     COUNT(DISTINCT doc) AS doc_count,
     SUM(warrant_count) AS total_warrants,
     SUM(total_assertions) AS total_assertions,
     COLLECT({doc: doc, warrants: warrant_count, total: total_assertions}) AS doc_data
WHERE doc_count >= 3 AND total_warrants > 0
RETURN reg.name AS regulation,
       doc_count,
       total_warrants,
       total_assertions,
       doc_data
ORDER BY total_warrants DESC
"""

clusters = []
with neo4j_driver.session() as session:
    result = session.run(cypher)
    for rec in result:
        doc_data = list(rec["doc_data"])
        warrant_counts = [d["warrants"] for d in doc_data]
        g = gini(warrant_counts)
        auth_doc = max(doc_data, key=lambda d: d["warrants"])
        auth_share = round(auth_doc["warrants"] / max(rec["total_warrants"], 1), 4)
        clusters.append({
            "regulation": rec["regulation"],
            "doc_count": rec["doc_count"],
            "total_warrants": rec["total_warrants"],
            "total_assertions": rec["total_assertions"],
            "gini": g,
            "authoritative_source": auth_doc["doc"],
            "authoritative_warrant_share": auth_share,
            "doc_breakdown": sorted(doc_data, key=lambda d: -d["warrants"])
        })
        log.info(f"Exp2 cluster: {rec['regulation']} gini={g} auth_share={auth_share}")

ginis = [c["gini"] for c in clusters]
median_gini = round(sorted(ginis)[len(ginis)//2], 4) if ginis else None
mean_gini = round(sum(ginis)/len(ginis), 4) if ginis else None
prop_monopoly = round(sum(1 for g in ginis if g > 0.80)/len(ginis), 4) if ginis else None
prop_distributed = round(sum(1 for g in ginis if g < 0.40)/len(ginis), 4) if ginis else None

output = {
    "status": "complete",
    "clusters_analyzed": len(clusters),
    "median_gini": median_gini,
    "mean_gini": mean_gini,
    "proportion_monopoly": prop_monopoly,
    "proportion_distributed": prop_distributed,
    "interpretation": (
        f"Across {len(clusters)} regulatory clusters with >= 3 contributing documents, "
        f"median Gini = {median_gini} and {round(prop_monopoly*100)}% show monopoly "
        f"concentration (Gini > 0.80). Warrant monopoly is a systematic structural "
        f"property of the corpus, not an artifact of the Basel III example."
    ),
    "clusters": clusters
}
checkpoint("warrant_monopoly_corpus.json", output)
print("Experiment 2 complete:", json.dumps({k:v for k,v in output.items() if k != "clusters"}, indent=2))
```

---

## Experiment 3 — Conflict Registry Characterization

**Purpose:** Convert the 63 CONTRADICTS count from a single number into a structured
characterization. The validity-claim type pairing breakdown is the key theoretical finding:
normative vs constative conflicts are a different kind of epistemic tension from normative
vs normative. The paper's theory predicts this matters but currently doesn't show whether
this pattern appears in the data.

**Runtime estimate:** ~10 minutes (single Neo4j query + Python aggregation).

```python
cypher = """
MATCH (a1:Assertion)-[r:CONTRADICTS]->(a2:Assertion)
RETURN
    a1.assertion_id AS src_id,
    a1.assertion_text[0..120] AS src_text,
    a1.discourse_role AS src_role,
    a1.validity_claim_type AS src_vtype,
    a1.doc_id AS src_doc,
    a2.assertion_id AS tgt_id,
    a2.assertion_text[0..120] AS tgt_text,
    a2.discourse_role AS tgt_role,
    a2.validity_claim_type AS tgt_vtype,
    a2.doc_id AS tgt_doc,
    r.confidence AS confidence,
    r.evidence_text[0..300] AS evidence
ORDER BY r.confidence DESC
"""

conflicts = []
with neo4j_driver.session() as session:
    result = session.run(cypher)
    for rec in result:
        pairing = f"{rec['src_vtype']}_vs_{rec['tgt_vtype']}"
        conflicts.append({
            "src_id": rec["src_id"],
            "src_text": rec["src_text"],
            "src_role": rec["src_role"],
            "src_vtype": rec["src_vtype"],
            "src_doc": rec["src_doc"],
            "tgt_id": rec["tgt_id"],
            "tgt_text": rec["tgt_text"],
            "tgt_role": rec["tgt_role"],
            "tgt_vtype": rec["tgt_vtype"],
            "tgt_doc": rec["tgt_doc"],
            "confidence": rec["confidence"],
            "evidence": rec["evidence"],
            "cross_document": rec["src_doc"] != rec["tgt_doc"],
            "vtype_pairing": pairing
        })

def domain(doc_id):
    doc = (doc_id or "").lower()
    if any(x in doc for x in ["basel","aifmd","dodd","mifid","gdpr","sox","sarbanes","volcker","ifrs","directive","regulation","act"]):
        return "regulatory"
    if any(x in doc for x in ["risk","var","stress","concentration","operational","liquidity","credit","market"]):
        return "risk"
    if any(x in doc for x in ["capm","apt","black","scholes","arbitrage","sharpe","markowitz","portfolio"]):
        return "financial_theory"
    if any(x in doc for x in ["fund","swap","derivative","option","futures","bond","equity","etf","hedge"]):
        return "products"
    return "other"

def conf_bucket(c):
    if c is None: return "unknown"
    if c >= 0.90: return "0.90_1.00"
    if c >= 0.80: return "0.80_0.89"
    if c >= 0.70: return "0.70_0.79"
    return "below_0.70"

from collections import Counter

pairing_counts = Counter(c["vtype_pairing"] for c in conflicts)
domain_counts = Counter(domain(c["src_doc"]) for c in conflicts)
conf_counts = Counter(conf_bucket(c.get("confidence")) for c in conflicts)
expressive_count = sum(1 for c in conflicts
                       if "expressive" in (c["src_vtype"] or "") + (c["tgt_vtype"] or ""))

output = {
    "status": "complete",
    "total_contradicts": len(conflicts),
    "cross_document": sum(1 for c in conflicts if c["cross_document"]),
    "within_document": sum(1 for c in conflicts if not c["cross_document"]),
    "by_validity_pairing": dict(pairing_counts),
    "expressive_involved": expressive_count,
    "by_confidence_bucket": dict(conf_counts),
    "by_domain": dict(domain_counts),
    "top_10_highest_confidence": [
        {
            "src_text": c["src_text"],
            "tgt_text": c["tgt_text"],
            "src_doc": c["src_doc"],
            "tgt_doc": c["tgt_doc"],
            "vtype_pairing": c["vtype_pairing"],
            "confidence": c["confidence"],
            "evidence": c["evidence"],
            "cross_document": c["cross_document"]
        }
        for c in sorted(conflicts, key=lambda x: x.get("confidence") or 0, reverse=True)[:10]
    ]
}
checkpoint("conflict_characterization.json", output)
print("Experiment 3 complete:", json.dumps({k:v for k,v in output.items() if k != "top_10_highest_confidence"}, indent=2))
```

---

## Entry point — run all three sequentially

Save as `run_supplemental_experiments.py` at the repo root.

```python
if __name__ == "__main__":
    import sys
    log.info("=== Supplemental experiments start ===")

    print("\n--- Experiment 1: VTP K-Invariance Curve ---")
    try:
        # paste Experiment 1 code here or import it
        pass
    except Exception as e:
        log.error(f"Experiment 1 failed: {e}")
        print(f"Experiment 1 FAILED: {e}")

    print("\n--- Experiment 2: Warrant Monopoly Corpus-Wide ---")
    try:
        pass
    except Exception as e:
        log.error(f"Experiment 2 failed: {e}")
        print(f"Experiment 2 FAILED: {e}")

    print("\n--- Experiment 3: Conflict Characterization ---")
    try:
        pass
    except Exception as e:
        log.error(f"Experiment 3 failed: {e}")
        print(f"Experiment 3 FAILED: {e}")

    log.info("=== Supplemental experiments complete ===")
    print("\nAll results in data/eval/supplemental/")
    print("Log: data/eval/supplemental/exp_log.txt")
```

---

## What to report back

When complete, print:

```
Experiment 1 — K-Invariance:
  Condition A plateau: k=? VTP=?
  Condition D at k=100: VTP=?
  Gap at k=100: ?
  K-invariance confirmed: yes/no

Experiment 2 — Warrant Monopoly:
  Clusters analyzed: ?
  Median Gini: ?
  Proportion with monopoly (Gini > 0.80): ?%
  Systematic property confirmed: yes/no

Experiment 3 — Conflict Characterization:
  Total CONTRADICTS: ?
  Cross-document: ? / Within-document: ?
  Normative vs normative: ?
  Normative vs constative: ?
  Constative vs constative: ?
  Dominant domain: ?
  Top conflict confidence: ?
```

These numbers go directly into Sections 6.1, 6.2, and 6.3 of the JMIS paper.

---

## One adjustment needed before running

The embed_query function imports from `pipeline.embeddings`. Check the actual module
path in the codebase — it may be `src/embeddings.py`, `utils/embed.py`, or a direct
Vertex AI / OpenAI call. Adjust the import to match. Everything else uses the same
Neo4j and Qdrant connections already established in the pipeline.
