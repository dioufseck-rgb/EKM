"""
run_evaluation.py — FinReg10K Evaluation Harness

Adaptation of the FinWiki evaluation harness (eval_harness_v4) for the
FinReg10K corpus.

Key adaptations from Section 6 of finreg10k_pipeline.md:
  - Query stratification: 40 normative / 40 constative / 20 mixed (100 total)
  - Query templates per stratum (firm-type and regulation placeholders)
  - Cross-firm CONTRADICTS edge detection and reporting
  - Parallel results table comparing FinWiki vs FinReg10K metrics

Metrics: VTP, AC, CSR, CWE — identical to FinWiki harness.
Conditions A-E — identical to FinWiki harness.

Usage:
    python run_evaluation.py
    python run_evaluation.py --finwiki-results path/to/finwiki_results_v4.json
    python run_evaluation.py --generate-queries   # generate and exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FINWIKI_PATH = Path(os.environ.get("FINWIKI_KG_PATH", BASE_DIR.parent / "finwiki-kg"))

if str(FINWIKI_PATH) not in sys.path:
    sys.path.insert(0, str(FINWIKI_PATH))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Patch data paths to finreg10k-kg ─────────────────────────────────────────
os.environ.setdefault("POSTGRES_URL",   "postgresql://finwiki:finwiki@localhost:5432/finwiki")
os.environ.setdefault("NEO4J_URL",      "bolt://localhost:7687")
os.environ.setdefault("QDRANT_URL",     "http://localhost:6333")
os.environ.setdefault("NEO4J_PASSWORD", "finwiki123")

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import psycopg2
import psycopg2.pool
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from scipy import stats as scipy_stats

import google.genai as genai
from google.genai import types as genai_types

try:
    from pipeline.config import settings
except ImportError as e:
    print(f"ERROR: Cannot import pipeline.config from {FINWIKI_PATH}: {e}", file=sys.stderr)
    sys.exit(1)

# ── Directory constants ────────────────────────────────────────────────────────
EVAL_DIR        = BASE_DIR / "data" / "eval"
LOG_FILE        = EVAL_DIR / "eval_log_finreg10k.txt"
RESULTS_FILE    = EVAL_DIR / "results_finreg10k.json"
RESULTS_PARTIAL = EVAL_DIR / "results_partial_finreg10k.json"
SUMMARY_FILE    = EVAL_DIR / "results_summary_finreg10k.txt"
QUERIES_FILE    = EVAL_DIR / "queries_finreg10k.json"

MAX_K = 50
AC_THRESHOLD = 0.8

# FinReg10K uses the same Qdrant collections as the shared Neo4j/Qdrant/PG cluster
ASSERTIONS_COLLECTION = "finwiki_assertions"
CHUNKS_COLLECTION     = "finwiki_chunks"

# ── Query strata (spec Section 6.1) ───────────────────────────────────────────
QUERY_STRATA: Dict[str, int] = {
    "normative":  40,
    "constative": 40,
    "mixed":      20,
}

# Templates instantiated with firm_type, regulation, business_area, topic
NORMATIVE_TEMPLATES = [
    "What must {firm_type} do under {regulation}?",
    "What are the compliance obligations for {firm_type} under {regulation}?",
    "What capital requirements does {regulation} impose on {firm_type}?",
    "What reporting requirements apply to {firm_type} under {regulation}?",
    "What restrictions does {regulation} place on {firm_type} activities?",
]
CONSTATIVE_TEMPLATES = [
    "What does {firm_type} do in {business_area}?",
    "How does {firm_type} generate revenue from {business_area}?",
    "What services does {firm_type} offer in {business_area}?",
    "What risks does {firm_type} face in {business_area}?",
    "How does {firm_type} manage operations in {business_area}?",
]
MIXED_TEMPLATES = [
    "What are the requirements and practices for {topic}?",
    "What regulations govern {topic} and how do firms comply?",
    "What are the key rules and operational practices for {topic}?",
    "How is {topic} regulated and how do financial firms implement compliance?",
    "What must firms do and what do they actually do regarding {topic}?",
]

# Placeholder values for template instantiation
FIRM_TYPES     = ["commercial bank", "savings institution", "broker-dealer",
                  "insurance company", "credit institution", "financial holding company"]
REGULATIONS    = ["BSA", "OFAC", "CRA", "Volcker Rule", "CECL", "DFAST", "CCAR",
                  "Sarbanes-Oxley", "Basel III", "Dodd-Frank", "CFPB regulations",
                  "OCC guidelines", "Bank Secrecy Act", "Solvency II"]
BUSINESS_AREAS = ["retail banking", "commercial lending", "wealth management",
                  "investment banking", "treasury operations", "insurance underwriting",
                  "mortgage origination", "capital markets", "trade finance"]
TOPICS         = ["anti-money laundering compliance", "capital adequacy management",
                  "liquidity risk management", "credit risk assessment",
                  "regulatory stress testing", "consumer protection compliance",
                  "sanctions screening", "model risk management"]


def generate_queries() -> List[Dict[str, Any]]:
    """
    Generate 100 evaluation queries: 40 normative, 40 constative, 20 mixed.
    Uses template rotation with placeholder cycling.
    """
    import itertools

    queries: List[Dict[str, Any]] = []

    # Normative: 40 queries
    firm_cycle    = itertools.cycle(FIRM_TYPES)
    reg_cycle     = itertools.cycle(REGULATIONS)
    tmpl_cycle    = itertools.cycle(NORMATIVE_TEMPLATES)
    for _ in range(QUERY_STRATA["normative"]):
        tmpl      = next(tmpl_cycle)
        firm_type = next(firm_cycle)
        regulation = next(reg_cycle)
        text = tmpl.format(firm_type=firm_type, regulation=regulation)
        queries.append({
            "query_id":      str(uuid.uuid4()),
            "query_text":    text,
            "epistemic_type": "normative",
            "firm_type":     firm_type,
            "regulation":    regulation,
            "template":      tmpl,
        })

    # Constative: 40 queries
    firm_cycle = itertools.cycle(FIRM_TYPES)
    area_cycle = itertools.cycle(BUSINESS_AREAS)
    tmpl_cycle = itertools.cycle(CONSTATIVE_TEMPLATES)
    for _ in range(QUERY_STRATA["constative"]):
        tmpl      = next(tmpl_cycle)
        firm_type = next(firm_cycle)
        area      = next(area_cycle)
        text = tmpl.format(firm_type=firm_type, business_area=area)
        queries.append({
            "query_id":      str(uuid.uuid4()),
            "query_text":    text,
            "epistemic_type": "constative",
            "firm_type":     firm_type,
            "business_area": area,
            "template":      tmpl,
        })

    # Mixed: 20 queries
    topic_cycle = itertools.cycle(TOPICS)
    tmpl_cycle  = itertools.cycle(MIXED_TEMPLATES)
    for _ in range(QUERY_STRATA["mixed"]):
        tmpl  = next(tmpl_cycle)
        topic = next(topic_cycle)
        text  = tmpl.format(topic=topic)
        queries.append({
            "query_id":      str(uuid.uuid4()),
            "query_text":    text,
            "epistemic_type": "mixed",
            "topic":         topic,
            "template":      tmpl,
        })

    return queries


# ── Connection helpers ─────────────────────────────────────────────────────────

def make_pg_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return psycopg2.pool.ThreadedConnectionPool(
        1, 8, settings.postgres_url
    )


def make_neo4j():
    return GraphDatabase.driver(
        settings.neo4j_url, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def pg_query(pool, sql: str, params=None) -> List[Tuple]:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()
    finally:
        pool.putconn(conn)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_text(text: str, client: genai.Client) -> List[float]:
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def qdrant_top_assertions(
    vector: List[float], k: int, qdrant: QdrantClient,
    validity_type: Optional[str] = None,
) -> List[Dict]:
    query_filter = None
    if validity_type:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(
                key="validity_claim_type",
                match=MatchValue(value=validity_type),
            )]
        )
    result = qdrant.query_points(
        collection_name=ASSERTIONS_COLLECTION,
        query=vector,
        limit=k,
        with_payload=True,
        query_filter=query_filter,
    )
    return [h.payload for h in result.points if h.payload]


def _row_to_assertion(r) -> Dict:
    return {
        "assertion_id":      r[0],
        "claim_text":        r[1],
        "subject":           r[2] or "",
        "predicate_type":    r[3] or "",
        "object_text":       r[4] or "",
        "source_document":   r[5] or "",
        "source_url":        r[6] or "",
        "epistemic_status":  r[7] or "authoritative",
        "confidence":        float(r[8] or 0.8),
        "domain":            r[9] or "",
        "validity_claim_type": r[10] or "unclassified",
        "discourse_role":    r[11] or "unclassified",
    }


def fetch_assertions_by_ids(assertion_ids: List[str], pool) -> List[Dict]:
    if not assertion_ids:
        return []
    placeholders = ",".join(["%s"] * len(assertion_ids))
    rows = pg_query(
        pool,
        f"""
        SELECT assertion_id, claim_text, subject, predicate_type, object_text,
               source_document, source_url, epistemic_status, confidence, domain,
               validity_claim_type, discourse_role
        FROM assertions
        WHERE assertion_id IN ({placeholders})
          AND epistemic_status NOT IN ('deprecated', 'orphaned')
        ORDER BY confidence DESC
        """,
        assertion_ids,
    )
    return [_row_to_assertion(r) for r in rows]


def _enrich_from_pg(payloads: List[Dict], pool) -> List[Dict]:
    if not payloads:
        return []
    ids = [p["assertion_id"] for p in payloads if "assertion_id" in p]
    if not ids:
        return []
    rows = fetch_assertions_by_ids(ids, pool)
    order = {aid: i for i, aid in enumerate(ids)}
    rows.sort(key=lambda a: order.get(a["assertion_id"], 999))
    return rows


# ── Toulmin neighborhood (undirected, mirrors v4 fix) ────────────────────────

def get_toulmin_neighborhood(assertion_id: str, driver, max_hops: int = 2) -> Set[str]:
    try:
        with driver.session() as s:
            result = s.run(
                f"""
                MATCH path = (a:Assertion {{assertion_id: $id}})
                    -[:ENTAILS|CAUSES|TRIGGERS|SPECIALIZES*1..{max_hops}]-
                    (b:Assertion)
                WHERE NOT (b.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN DISTINCT b.assertion_id AS nid
                """,
                id=assertion_id,
            )
            return {row["nid"] for row in result}
    except Exception:
        return set()


def get_neutral_seed(
    query_id: str, vector: List[float], qdrant: QdrantClient, pool,
    queries: List[Dict],
) -> Optional[str]:
    for q in queries:
        if q["query_id"] == query_id and q.get("query_seed_assertion_id"):
            return q["query_seed_assertion_id"]
    payloads = qdrant_top_assertions(vector, 1, qdrant)
    if not payloads:
        return None
    seed_id = payloads[0].get("assertion_id")
    if not seed_id:
        return None
    for q in queries:
        if q["query_id"] == query_id:
            q["query_seed_assertion_id"] = seed_id
            break
    QUERIES_FILE.write_text(json.dumps(queries, indent=2))
    return seed_id


# ── Cross-firm CONTRADICTS detection (FinReg10K-specific) ─────────────────────

def detect_cross_firm_contradicts(assertion_ids: Set[str], driver) -> List[Dict]:
    """
    Detect CONTRADICTS edges where source and target come from different firms
    (i.e., different CIK-based document IDs). This is a novel FinReg10K signal:
    inter-firm regulatory disagreement detected as a first-class quality finding.
    """
    if len(assertion_ids) < 2:
        return []
    ids = list(assertion_ids)
    cross_firm: List[Dict] = []
    try:
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $ids AS sid
                MATCH (a:Assertion {assertion_id: sid})
                      -[r:CONTRADICTS]->(b:Assertion)
                WHERE b.assertion_id IN $ids
                  AND a.source_document <> b.source_document
                RETURN a.assertion_id AS src, b.assertion_id AS tgt,
                       a.source_document AS src_doc, b.source_document AS tgt_doc
                """,
                ids=ids,
            )
            for row in result:
                cross_firm.append({
                    "source": row["src"],
                    "target": row["tgt"],
                    "source_doc": row["src_doc"],
                    "target_doc": row["tgt_doc"],
                    "type": "cross_firm",
                })
    except Exception:
        pass
    return cross_firm


# ── Retrieval conditions (A-D, mirrors v4) ─────────────────────────────────────

def retrieve_a(vector, k, qdrant, pool) -> List[Dict]:
    payloads = qdrant_top_assertions(vector, k, qdrant)
    return _enrich_from_pg(payloads, pool)


def retrieve_b(vector, k, qdrant, pool, driver) -> List[Dict]:
    seed_payloads = qdrant_top_assertions(vector, k, qdrant)
    seeds = _enrich_from_pg(seed_payloads, pool)
    if not seeds:
        return seeds
    seed_ids = [a["assertion_id"] for a in seeds[:10]]
    try:
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $ids AS sid
                MATCH (a:Assertion {assertion_id: sid})-[r]-(b:Assertion)
                WHERE NOT (b.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN DISTINCT b.assertion_id AS nid LIMIT 200
                """,
                ids=seed_ids,
            )
            neighbor_ids = [row["nid"] for row in result]
    except Exception:
        neighbor_ids = []
    existing = {a["assertion_id"] for a in seeds}
    new_ids = [nid for nid in neighbor_ids if nid not in existing]
    neighbors = fetch_assertions_by_ids(new_ids[:k], pool)
    combined = seeds + neighbors
    return sorted(combined, key=lambda x: x["confidence"], reverse=True)[:k]


def retrieve_c(vector, k, qdrant, pool, epistemic_type) -> List[Dict]:
    if epistemic_type == "mixed":
        norm_p  = qdrant_top_assertions(vector, k, qdrant, validity_type="normative")
        const_p = qdrant_top_assertions(vector, k, qdrant, validity_type="constative")
        norm    = _enrich_from_pg(norm_p, pool)
        const_  = _enrich_from_pg(const_p, pool)
        seen: Set[str] = set()
        merged = []
        for a in norm + const_:
            if a["assertion_id"] not in seen:
                merged.append(a)
                seen.add(a["assertion_id"])
        return sorted(merged, key=lambda x: x["confidence"], reverse=True)[:k]
    vtype = "normative" if epistemic_type == "normative" else "constative"
    payloads = qdrant_top_assertions(vector, k, qdrant, validity_type=vtype)
    return _enrich_from_pg(payloads, pool)[:k]


def _toulmin_expand(seed_ids: List[str], driver, max_hops: int = 2) -> List[str]:
    if not seed_ids:
        return []
    try:
        with driver.session() as s:
            result = s.run(
                f"""
                UNWIND $ids AS sid
                MATCH path = (a:Assertion {{assertion_id: sid}})
                    -[:ENTAILS|CAUSES|TRIGGERS|SPECIALIZES*1..{max_hops}]->
                    (b:Assertion)
                WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
                  AND NOT (b.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN DISTINCT b.assertion_id AS nid LIMIT 100
                """,
                ids=seed_ids[:5],
            )
            return [row["nid"] for row in result]
    except Exception:
        return []


def _detect_contradicts(assertion_ids: Set[str], driver) -> List[Tuple[str, str]]:
    if len(assertion_ids) < 2:
        return []
    ids = list(assertion_ids)
    try:
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $ids AS sid
                MATCH (a:Assertion {assertion_id: sid})
                      -[r:CONTRADICTS]->(b:Assertion)
                WHERE b.assertion_id IN $ids
                RETURN a.assertion_id AS src, b.assertion_id AS tgt
                """,
                ids=ids,
            )
            return [(row["src"], row["tgt"]) for row in result]
    except Exception:
        return []


def retrieve_d(vector, k, qdrant, pool, driver, epistemic_type):
    if epistemic_type == "mixed":
        norm_p  = qdrant_top_assertions(vector, k, qdrant, validity_type="normative")
        const_p = qdrant_top_assertions(vector, k, qdrant, validity_type="constative")
        seeds   = _enrich_from_pg(norm_p + const_p, pool)
        seen: Set[str] = set()
        deduped = []
        for a in seeds:
            if a["assertion_id"] not in seen:
                deduped.append(a)
                seen.add(a["assertion_id"])
        seeds = sorted(deduped, key=lambda x: x["confidence"], reverse=True)[:k]
    else:
        vtype    = "normative" if epistemic_type == "normative" else "constative"
        payloads = qdrant_top_assertions(vector, k, qdrant, validity_type=vtype)
        seeds    = _enrich_from_pg(payloads, pool)[:k]

    seed_ids     = [a["assertion_id"] for a in seeds[:5]]
    expanded_ids = _toulmin_expand(seed_ids, driver)
    expanded     = fetch_assertions_by_ids(expanded_ids, pool)

    existing = {a["assertion_id"] for a in seeds}
    combined = list(seeds)
    for a in expanded:
        if a["assertion_id"] not in existing:
            combined.append(a)
            existing.add(a["assertion_id"])
    combined = sorted(combined, key=lambda x: x["confidence"], reverse=True)[:k]

    subgraph_ids   = {a["assertion_id"] for a in combined}
    conflict_pairs = _detect_contradicts(subgraph_ids, driver)
    conflicts      = len(conflict_pairs) > 0

    # FinReg10K addition: detect cross-firm conflicts
    cross_firm = detect_cross_firm_contradicts(subgraph_ids, driver)

    return combined, conflicts, conflict_pairs, cross_firm


# ── Metrics ───────────────────────────────────────────────────────────────────

def validity_type_precision(assertions: List[Dict], epistemic_type: str, k: int = 10) -> Dict:
    top = assertions[:k]
    if not top:
        return {"precision": 0.0}
    if epistemic_type == "normative":
        p = sum(1 for a in top if a.get("validity_claim_type") == "normative") / len(top)
        return {"precision": p}
    elif epistemic_type == "constative":
        p = sum(1 for a in top if a.get("validity_claim_type") == "constative") / len(top)
        return {"precision": p}
    else:
        has_n = any(a.get("validity_claim_type") == "normative" for a in top)
        has_c = any(a.get("validity_claim_type") == "constative" for a in top)
        return {
            "precision": float(has_n and has_c),
            "normative_present": has_n,
            "constative_present": has_c,
        }


def compute_ac_cwe(
    assertions: List[Dict], neighborhood: Set[str], neighborhood_size: int
) -> Tuple[Optional[float], int]:
    if neighborhood_size == 0:
        return None, MAX_K
    retrieved_ids = {a["assertion_id"] for a in assertions}
    ac = len(retrieved_ids & neighborhood) / neighborhood_size
    sorted_ids = [a["assertion_id"] for a in assertions]
    cwe = MAX_K
    for k_val in range(1, len(sorted_ids) + 1):
        if len(set(sorted_ids[:k_val]) & neighborhood) / neighborhood_size >= AC_THRESHOLD:
            cwe = k_val
            break
    return ac, cwe


# ── Statistical helpers ────────────────────────────────────────────────────────

def cohens_d_paired(a: List[float], b: List[float]) -> float:
    diff = np.array(b) - np.array(a)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def run_statistical_tests(scores_a: List[float], scores_b: List[float]) -> Dict:
    if len(scores_a) < 3 or len(scores_b) < 3:
        return {"error": "insufficient data (n<3)"}
    n = min(len(scores_a), len(scores_b))
    a, b = scores_a[:n], scores_b[:n]
    out: Dict[str, Any] = {
        "n_pairs":          n,
        "mean_baseline":    float(np.mean(a)),
        "mean_treatment":   float(np.mean(b)),
        "median_baseline":  float(np.median(a)),
        "median_treatment": float(np.median(b)),
    }
    try:
        w_stat, p_val = scipy_stats.wilcoxon(a, b)
        out["wilcoxon_p"] = float(p_val)
    except Exception as e:
        out["wilcoxon_p"]    = None
        out["wilcoxon_error"] = str(e)
    out["cohens_d"] = cohens_d_paired(a, b)
    d = abs(out["cohens_d"])
    out["effect_size_label"] = (
        "negligible" if d < 0.2 else
        "small"      if d < 0.5 else
        "medium"     if d < 0.8 else
        "large"
    )
    return out


def _paired_scores(completed: List[Dict], cond_a: str, cond_b: str, metric: str):
    a_vals, b_vals = [], []
    for r in completed:
        va = r["metrics"].get(cond_a, {}).get(metric)
        vb = r["metrics"].get(cond_b, {}).get(metric)
        if va is not None and vb is not None:
            a_vals.append(float(va))
            b_vals.append(float(vb))
    return a_vals, b_vals


# ── Per-query evaluation ───────────────────────────────────────────────────────

def with_retry(fn, retries: int = 3, backoff: int = 10, logger=None):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                if logger:
                    logger.warning(f"Retry {attempt+1}/{retries} after: {e}")
                time.sleep(backoff)
    raise last_exc


def eval_single_query(
    q: Dict, queries: List[Dict],
    qdrant, driver, pool, client, logger,
) -> Dict:
    qid   = q["query_id"]
    qtype = q["epistemic_type"]
    ts    = datetime.utcnow().isoformat()

    vector = embed_text(q["query_text"], client)

    assertions_a = retrieve_a(vector, MAX_K, qdrant, pool)
    assertions_b = retrieve_b(vector, MAX_K, qdrant, pool, driver)
    assertions_c = retrieve_c(vector, MAX_K, qdrant, pool, qtype)
    assertions_d, conflicts_d, conflict_pairs_d, cross_firm_d = retrieve_d(
        vector, MAX_K, qdrant, pool, driver, qtype
    )

    # Condition-neutral seed + Toulmin neighborhood (undirected, mirrors v4)
    seed_id      = get_neutral_seed(qid, vector, qdrant, pool, queries)
    neighborhood = get_toulmin_neighborhood(seed_id, driver) if seed_id else set()
    neighborhood_size = len(neighborhood)
    ac_eligible  = neighborhood_size > 0

    seed_doc = ""
    if seed_id:
        rows = pg_query(pool, "SELECT source_document FROM assertions WHERE assertion_id = %s", [seed_id])
        seed_doc = rows[0][0] if rows else ""

    metrics: Dict[str, Dict] = {}
    for cond, assertions in [
        ("A", assertions_a),
        ("B", assertions_b),
        ("C", assertions_c),
        ("D", assertions_d),
    ]:
        vtp = validity_type_precision(assertions, qtype)
        ac, cwe = compute_ac_cwe(assertions, neighborhood, neighborhood_size)
        if not ac_eligible:
            ac  = None
            cwe = MAX_K
        csr = (1 if conflicts_d else 0) if cond == "D" else 0

        m: Dict[str, Any] = {
            "vtp":        vtp.get("precision", 0.0),
            "vtp_detail": vtp,
            "ac":         ac,
            "cwe":        cwe,
            "csr":        csr,
            "n_retrieved": len(assertions),
        }
        if cond == "D" and conflicts_d:
            m["conflict_pairs"] = [
                {"source": s, "target": t} for s, t in conflict_pairs_d[:5]
            ]
        if cond == "D" and cross_firm_d:
            m["cross_firm_contradicts"] = cross_firm_d[:5]

        metrics[cond] = m
        logger.info(
            f"qid={qid[:8]} type={qtype} cond={cond} "
            f"vtp={m['vtp']:.3f} ac={ac} cwe={cwe} csr={csr} "
            f"n={len(assertions)} nbr={neighborhood_size}"
        )

    # Condition E: two-pass compliance retrieval (import from FinWiki api)
    try:
        from api.reasoning import two_pass_compliance_retrieval
        result_e = two_pass_compliance_retrieval(
            query_text=q["query_text"],
            vector=vector,
            driver=driver,
            qdrant=qdrant,
            pool=pool,
            epistemic_type=qtype,
            k=MAX_K,
        )
        pass1_assertions: List[Dict] = result_e.get("_pass1_assertions", [])
        all_assertions:   List[Dict] = result_e.get("_all_assertions", [])
        conflicts_e:      bool       = result_e.get("_conflicts_detected", False)
        conflict_pairs_e             = result_e.get("_conflict_pairs", [])

        vtp_e  = validity_type_precision(pass1_assertions, qtype)
        ac_e, cwe_e = compute_ac_cwe(all_assertions, neighborhood, neighborhood_size)
        if not ac_eligible:
            ac_e  = None
            cwe_e = MAX_K
        csr_e = 1 if conflicts_e else 0

        metrics["E"] = {
            "vtp":              vtp_e.get("precision", 0.0),
            "vtp_detail":       vtp_e,
            "ac":               ac_e,
            "cwe":              cwe_e,
            "csr":              csr_e,
            "n_retrieved_pass1": len(pass1_assertions),
            "n_retrieved_all":   len(all_assertions),
            "completeness_warning": result_e.get("metadata", {}).get("completeness_warning", False),
        }
        if conflicts_e:
            metrics["E"]["conflict_pairs"] = [
                {"source": s, "target": t} for s, t in conflict_pairs_e[:5]
            ]
        logger.info(
            f"qid={qid[:8]} type={qtype} cond=E "
            f"vtp={metrics['E']['vtp']:.3f} ac={ac_e} cwe={cwe_e} csr={csr_e}"
        )
    except Exception as e:
        logger.error(f"Condition E failed for {qid[:8]}: {e}")
        metrics["E"] = None

    return {
        "query_id":          qid,
        "query_text":        q["query_text"],
        "epistemic_type":    qtype,
        "seed_assertion_id": seed_id,
        "seed_doc_id":       seed_doc,
        "neighborhood_size": neighborhood_size,
        "metrics":           metrics,
        "timestamp":         ts,
        "failed":            False,
    }


# ── Aggregate statistics ───────────────────────────────────────────────────────

def aggregate_results(query_results: List[Dict]) -> Dict:
    completed = [r for r in query_results if not r.get("failed")]
    metrics_names = ["vtp", "ac", "cwe", "csr"]
    conditions    = ["A", "B", "C", "D", "E"]

    ac_eligible_count = sum(1 for r in completed if r.get("neighborhood_size", 0) > 0)

    results_by_condition: Dict[str, Dict] = {}
    for cond in conditions:
        results_by_condition[cond] = {}
        for m in metrics_names:
            vals = [
                float(r["metrics"].get(cond, {}).get(m, 0))
                for r in completed
                if r["metrics"].get(cond) is not None
                and r["metrics"].get(cond, {}).get(m) is not None
            ]
            results_by_condition[cond][f"{m}_mean"]   = float(np.mean(vals))   if vals else None
            results_by_condition[cond][f"{m}_median"] = float(np.median(vals)) if vals else None
            results_by_condition[cond][f"{m}_n"]      = len(vals)

    stat_tests: Dict[str, Dict] = {}
    for bl, tr in [("A", "D"), ("B", "D"), ("C", "D"),
                   ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E")]:
        pk = f"{bl}_vs_{tr}"
        stat_tests[pk] = {}
        for m in metrics_names:
            a_s, b_s = _paired_scores(completed, bl, tr, m)
            stat_tests[pk][m] = run_statistical_tests(a_s, b_s)

    csr_d_values = [r["metrics"].get("D", {}).get("csr", 0) for r in completed if r["metrics"].get("D")]
    csr_e_values = [r["metrics"].get("E", {}).get("csr", 0) for r in completed if r["metrics"].get("E")]

    # Cross-firm conflict rate (FinReg10K-specific)
    cross_firm_rate = 0.0
    cross_firm_total = 0
    for r in completed:
        d_metrics = r["metrics"].get("D", {})
        if d_metrics and d_metrics.get("cross_firm_contradicts"):
            cross_firm_total += 1
    if completed:
        cross_firm_rate = cross_firm_total / len(completed)

    subgroup_type: Dict[str, Dict] = {}
    for qtype in ("normative", "constative", "mixed"):
        subset = [r for r in completed if r["epistemic_type"] == qtype]
        subgroup_type[qtype] = {}
        for bl, tr in [("D", "E"), ("A", "E"), ("A", "D")]:
            pk = f"{bl}_vs_{tr}"
            subgroup_type[qtype][pk] = {}
            for m in metrics_names:
                a_s, b_s = _paired_scores(subset, bl, tr, m)
                subgroup_type[qtype][pk][m] = run_statistical_tests(a_s, b_s)

    return {
        "n_completed":              len(completed),
        "n_failed":                 len(query_results) - len(completed),
        "ac_eligible_queries":      ac_eligible_count,
        "ac_excluded_queries":      len(completed) - ac_eligible_count,
        "csr_d_rate":               float(np.mean(csr_d_values)) if csr_d_values else 0.0,
        "csr_e_rate":               float(np.mean(csr_e_values)) if csr_e_values else 0.0,
        "cross_firm_conflict_rate": cross_firm_rate,
        "results_by_condition":     results_by_condition,
        "statistical_tests":        stat_tests,
        "subgroup_by_epistemic_type": subgroup_type,
    }


# ── Parallel comparison table ──────────────────────────────────────────────────

def build_comparison_table(
    finreg_aggregate: Dict,
    finwiki_results: Optional[Dict] = None,
) -> str:
    """
    Build a side-by-side FinWiki vs FinReg10K results table (spec Section 7).
    """
    rc_fr = finreg_aggregate["results_by_condition"]

    def _f3(v):
        return f"{v:.3f}" if v is not None else "n/a"

    lines = [
        "",
        "=== FinWiki vs FinReg10K Parallel Results ===",
        "",
        "VTP (Validity-Type Precision):",
        f"{'Condition':<25} {'FinWiki VTP':>12} {'FinReg10K VTP':>14}",
        "-" * 55,
    ]

    fw_rc = None
    if finwiki_results:
        fw_rc = finwiki_results.get("results_by_condition", {})

    for cond in ("A", "B", "C", "D", "E"):
        fw_vtp = _f3(fw_rc.get(cond, {}).get("vtp_mean")) if fw_rc else "n/a"
        fr_vtp = _f3(rc_fr.get(cond, {}).get("vtp_mean"))
        cond_label = {
            "A": "A — Standard RAG",
            "B": "B — Entity GraphRAG",
            "C": "C — Validity-gated",
            "D": "D — Full discourse-typed",
            "E": "E — Two-pass",
        }.get(cond, cond)
        lines.append(f"  {cond_label:<23} {fw_vtp:>12} {fr_vtp:>14}")

    lines += [
        "",
        "AC (Assertional Coverage):",
        f"{'Condition':<25} {'FinWiki AC':>12} {'FinReg10K AC':>13}",
        "-" * 55,
    ]
    for cond in ("A", "D", "E"):
        fw_ac = _f3(fw_rc.get(cond, {}).get("ac_mean")) if fw_rc else "n/a"
        fr_ac = _f3(rc_fr.get(cond, {}).get("ac_mean"))
        lines.append(f"  Condition {cond:<20} {fw_ac:>12} {fr_ac:>13}")

    lines += [
        "",
        "CSR (Conflict Surface Rate):",
        f"  FinWiki CSR (D):    {_f3(finwiki_results.get('csr_d_rate')) if finwiki_results else 'n/a'}",
        f"  FinReg10K CSR (D):  {_f3(finreg_aggregate.get('csr_d_rate'))}",
        f"  Cross-firm CONTRADICTS rate: {_f3(finreg_aggregate.get('cross_firm_conflict_rate'))}",
        "",
    ]

    return "\n".join(lines)


# ── Output helpers ─────────────────────────────────────────────────────────────

def save_partial(results: List[Dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PARTIAL.write_text(json.dumps(results, indent=2, default=str))


def save_results(query_results: List[Dict], aggregate: Dict, run_ts: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp":     run_ts,
        "harness_version":   "finreg10k_v1",
        "corpus":            "FinReg10K",
        "description":       (
            "FinReg10K evaluation — 100 queries (40 normative / 40 constative / 20 mixed). "
            "Conditions A-E. Cross-firm CONTRADICTS detection enabled."
        ),
        "query_strata":      QUERY_STRATA,
        "n_queries":         len(query_results),
        "n_failed":          aggregate["n_failed"],
        "ac_eligible_queries": aggregate["ac_eligible_queries"],
        "csr_d_rate":        aggregate["csr_d_rate"],
        "csr_e_rate":        aggregate["csr_e_rate"],
        "cross_firm_conflict_rate": aggregate["cross_firm_conflict_rate"],
        "results_by_condition": aggregate["results_by_condition"],
        "statistical_tests": aggregate["statistical_tests"],
        "subgroup_by_epistemic_type": aggregate["subgroup_by_epistemic_type"],
        "failed_queries": [
            {"query_id": r["query_id"], "error": r.get("error", "unknown")}
            for r in query_results if r.get("failed")
        ],
        "per_query_results": query_results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))


def save_summary(aggregate: Dict, run_ts: str, n_total: int, comparison: str) -> None:
    rc = aggregate["results_by_condition"]

    def _f3(v):
        return f"{v:.3f}" if v is not None else "n/a"

    lines = [
        "FINREG10K EVALUATION HARNESS — RESULTS SUMMARY",
        "=" * 55,
        f"Run timestamp: {run_ts}",
        f"Corpus: FinReg10K (SEC EDGAR 10-K filings)",
        f"Queries: {aggregate['n_completed']}/{n_total} completed",
        f"Strata: 40 normative / 40 constative / 20 mixed",
        "",
        "CONDITION MEANS:",
    ]
    for cond in ("A", "B", "C", "D", "E"):
        c = rc.get(cond, {})
        lines.append(
            f"  [{cond}] VTP={_f3(c.get('vtp_mean'))}  "
            f"AC={_f3(c.get('ac_mean'))}  "
            f"CSR={_f3(c.get('csr_mean'))}"
        )

    lines += [
        "",
        f"CSR D rate:               {_f3(aggregate['csr_d_rate'])}",
        f"CSR E rate:               {_f3(aggregate['csr_e_rate'])}",
        f"Cross-firm conflict rate: {_f3(aggregate['cross_firm_conflict_rate'])}",
        "",
        comparison,
    ]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n")


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_harness_finreg10k")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FinReg10K evaluation harness")
    parser.add_argument(
        "--generate-queries",
        action="store_true",
        help="Generate query file and exit (no evaluation run)",
    )
    parser.add_argument(
        "--finwiki-results",
        metavar="PATH",
        default=str(FINWIKI_PATH / "data" / "eval" / "results_v4.json"),
        help="Path to FinWiki results JSON for parallel table",
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    run_ts = datetime.utcnow().isoformat()

    # Load or generate queries
    if QUERIES_FILE.exists() and not args.generate_queries:
        with open(QUERIES_FILE) as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} queries from {QUERIES_FILE}")
    else:
        queries = generate_queries()
        QUERIES_FILE.write_text(json.dumps(queries, indent=2))
        logger.info(f"Generated {len(queries)} queries → {QUERIES_FILE}")
        if args.generate_queries:
            print(f"Queries written to {QUERIES_FILE}")
            return

    logger.info(f"=== FinReg10K Eval Harness START {run_ts} ===")
    logger.info(f"Queries: {len(queries)} ({QUERY_STRATA})")

    # Connections
    genai_client = genai.Client(api_key=settings.google_api_key)
    qdrant = QdrantClient(url=settings.qdrant_url)
    driver = make_neo4j()
    pool   = make_pg_pool()
    logger.info("Connections established")

    # Load FinWiki results for comparison (optional)
    finwiki_results: Optional[Dict] = None
    fw_path = Path(args.finwiki_results)
    if fw_path.exists():
        try:
            with open(fw_path) as f:
                finwiki_results = json.load(f)
            logger.info(f"Loaded FinWiki results from {fw_path}")
        except Exception as e:
            logger.warning(f"Could not load FinWiki results: {e}")

    # Evaluate
    query_results: List[Dict] = []
    for i, q in enumerate(queries):
        qid = q["query_id"]
        try:
            result = with_retry(
                lambda q=q: eval_single_query(q, queries, qdrant, driver, pool, genai_client, logger),
                retries=3, backoff=10, logger=logger,
            )
            query_results.append(result)
        except Exception as e:
            logger.error(f"FAILED {qid[:8]}: {e}")
            query_results.append({
                "query_id":          qid,
                "query_text":        q["query_text"],
                "epistemic_type":    q["epistemic_type"],
                "seed_assertion_id": None,
                "seed_doc_id":       "",
                "neighborhood_size": 0,
                "metrics":           {},
                "failed":            True,
                "error":             str(e),
            })

        if (i + 1) % 10 == 0:
            save_partial(query_results)
            logger.info(f"Progress: {i+1}/{len(queries)} — partial saved")

    save_partial(query_results)
    aggregate  = aggregate_results(query_results)
    comparison = build_comparison_table(aggregate, finwiki_results)
    save_results(query_results, aggregate, run_ts)
    save_summary(aggregate, run_ts, len(queries), comparison)

    # Console output
    rc = aggregate["results_by_condition"]

    def _f3(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"\n=== FINREG10K RESULTS ===")
    print(f"AC eligible: {aggregate['ac_eligible_queries']}/{len(queries)}")
    print(f"Failed:      {aggregate['n_failed']}")
    print()
    for cond in ("A", "D", "E"):
        c = rc.get(cond, {})
        print(
            f"Condition {cond}: VTP={_f3(c.get('vtp_mean'))}  "
            f"AC={_f3(c.get('ac_mean'))}  "
            f"CSR={_f3(c.get('csr_mean'))}"
        )
    print()
    print(f"Cross-firm conflict rate: {_f3(aggregate['cross_firm_conflict_rate'])}")
    print(comparison)

    logger.info(f"=== FinReg10K Eval Harness COMPLETE ===")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
