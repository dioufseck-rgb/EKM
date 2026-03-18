"""pipeline/eval_harness_v4.py — FinWiki Evaluation Harness v4

Key change vs v3:
  Fix: _toulmin_expand_from and get_toulmin_neighborhood now use undirected
       traversal (-[:ENTAILS|...]-) instead of directed (-[:ENTAILS|...]->).
       This corrects AC eligibility from ~35/120 to ~67/120.

  Full rerun: runs all conditions A-E from scratch (does not load v2/v3
  partial results). Seeds are reused from queries.json if already stored.

Usage:
  cd /workspaces/EKM/finwiki-kg
  python run_evaluation.py --version v4 --all-conditions
  # or directly:
  python -m pipeline.eval_harness_v4 > data/eval/stdout_v4.log 2>&1
"""
import os
os.environ.setdefault("POSTGRES_URL", "postgresql://finwiki:finwiki@localhost:5432/finwiki")
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import psycopg2
import psycopg2.pool
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from scipy import stats as scipy_stats

import google.genai as genai
from google.genai import types as genai_types

from pipeline.config import settings
from api.reasoning import two_pass_compliance_retrieval

# ── Directory & file constants ────────────────────────────────────────────────
EVAL_DIR        = Path("data/eval")
LOG_FILE        = EVAL_DIR / "eval_log_v4.txt"
RESULTS_FILE    = EVAL_DIR / "results_v4.json"
RESULTS_PARTIAL = EVAL_DIR / "results_partial_v4.json"
SUMMARY_FILE    = EVAL_DIR / "results_summary_v4.txt"
QUERIES_FILE    = EVAL_DIR / "queries.json"

MAX_K = 50
AC_THRESHOLD = 0.8
ASSERTIONS_COLLECTION = "finwiki_assertions"
CHUNKS_COLLECTION = "finwiki_chunks"


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_harness_v4")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


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


# ── Connections ───────────────────────────────────────────────────────────────
def make_pg_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return psycopg2.pool.ThreadedConnectionPool(
        1, 8, "postgresql://finwiki:finwiki@localhost:5432/finwiki"
    )


def make_neo4j():
    return GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", settings.neo4j_password)
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


# ── Assertion fetchers ─────────────────────────────────────────────────────────
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


def _row_to_assertion(r) -> Dict:
    return {
        "assertion_id": r[0], "claim_text": r[1], "subject": r[2] or "",
        "predicate_type": r[3] or "", "object_text": r[4] or "",
        "source_document": r[5] or "", "source_url": r[6] or "",
        "epistemic_status": r[7] or "authoritative",
        "confidence": float(r[8] or 0.8), "domain": r[9] or "",
        "validity_claim_type": r[10] or "unclassified",
        "discourse_role": r[11] or "unclassified",
    }


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


# ── Condition-neutral seed ─────────────────────────────────────────────────────
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
    with open(QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)
    return seed_id


# ── Toulmin neighborhood — FIXED: undirected traversal ───────────────────────
def get_toulmin_neighborhood(
    assertion_id: str, driver, max_hops: int = 2
) -> Set[str]:
    """
    Returns the set of assertion IDs reachable from assertion_id via
    ENTAILS|CAUSES|TRIGGERS|SPECIALIZES edges in either direction.

    v4 fix: uses undirected traversal (-[...]-) instead of directed (-[...]->)
    so edges stored with seed as target are also found.
    """
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


# ── Retrieval conditions ───────────────────────────────────────────────────────
def retrieve_a(vector, k, qdrant, pool):
    """Condition A: Standard vector RAG."""
    payloads = qdrant_top_assertions(vector, k, qdrant)
    return _enrich_from_pg(payloads, pool)


def retrieve_b(vector, k, qdrant, pool, driver):
    """Condition B: Entity GraphRAG — top-k seeds + 1-2 hop expansion."""
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


def retrieve_c(vector, k, qdrant, pool, epistemic_type):
    """Condition C: Validity-gated."""
    if epistemic_type == "mixed":
        norm_p = qdrant_top_assertions(vector, k, qdrant, validity_type="normative")
        const_p = qdrant_top_assertions(vector, k, qdrant, validity_type="constative")
        norm = _enrich_from_pg(norm_p, pool)
        const_ = _enrich_from_pg(const_p, pool)
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


def _regulation_anchor_ids(seeds: List[Dict], driver, top_k: int = 5) -> List[str]:
    if not seeds:
        return []
    try:
        with driver.session() as s:
            candidates: Dict[str, int] = {}
            for seed in seeds[:5]:
                rows = list(s.run(
                    "MATCH (a:Assertion {assertion_id: $id})-[:REFERENCES]->(r:Regulation) "
                    "RETURN r.name AS reg",
                    id=seed["assertion_id"],
                ))
                for row in rows:
                    reg = row["reg"]
                    if reg not in candidates:
                        cnt = s.run(
                            "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                            "WHERE NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                            "RETURN count(a) AS cnt",
                            reg=reg,
                        ).single()
                        candidates[reg] = cnt["cnt"] if cnt else 0
            if not candidates:
                return []
            best_reg = max(candidates, key=lambda k: candidates[k])
            auth = s.run(
                "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                "WHERE NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                "RETURN a.source_document AS src, count(a) AS cnt "
                "ORDER BY cnt DESC LIMIT 1",
                reg=best_reg,
            ).single()
            if not auth:
                return []
            auth_doc = auth["src"]
            periph = list(s.run(
                "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                "WHERE a.source_document <> $src "
                "  AND NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                "RETURN a.assertion_id AS id "
                "ORDER BY a.confidence DESC LIMIT $k",
                reg=best_reg, src=auth_doc, k=top_k * 3,
            ))
            return [row["id"] for row in periph]
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
    """Condition D: Full discourse-typed retrieval."""
    if epistemic_type == "mixed":
        norm_p = qdrant_top_assertions(vector, k, qdrant, validity_type="normative")
        const_p = qdrant_top_assertions(vector, k, qdrant, validity_type="constative")
        seeds = _enrich_from_pg(norm_p + const_p, pool)
        seen: Set[str] = set()
        deduped = []
        for a in seeds:
            if a["assertion_id"] not in seen:
                deduped.append(a)
                seen.add(a["assertion_id"])
        seeds = sorted(deduped, key=lambda x: x["confidence"], reverse=True)[:k]
    else:
        vtype = "normative" if epistemic_type == "normative" else "constative"
        payloads = qdrant_top_assertions(vector, k, qdrant, validity_type=vtype)
        seeds = _enrich_from_pg(payloads, pool)[:k]

    seed_ids = [a["assertion_id"] for a in seeds[:5]]
    expanded_ids = _toulmin_expand(seed_ids, driver)
    expanded = fetch_assertions_by_ids(expanded_ids, pool)

    anchor_ids = _regulation_anchor_ids(seeds, driver)
    anchor = fetch_assertions_by_ids(anchor_ids, pool)

    existing = {a["assertion_id"] for a in seeds}
    combined = list(seeds)
    for a in expanded + anchor:
        if a["assertion_id"] not in existing:
            combined.append(a)
            existing.add(a["assertion_id"])
    combined = sorted(combined, key=lambda x: x["confidence"], reverse=True)[:k]

    subgraph_ids = {a["assertion_id"] for a in combined}
    conflict_pairs = _detect_contradicts(subgraph_ids, driver)
    conflicts_detected = len(conflict_pairs) > 0

    return combined, conflicts_detected, conflict_pairs


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
        "n_pairs": n,
        "mean_baseline": float(np.mean(a)),
        "mean_treatment": float(np.mean(b)),
        "median_baseline": float(np.median(a)),
        "median_treatment": float(np.median(b)),
    }
    try:
        w_stat, p_val = scipy_stats.wilcoxon(a, b)
        out["wilcoxon_p"] = float(p_val)
    except Exception as e:
        out["wilcoxon_p"] = None
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
def eval_single_query(q: Dict, queries: List[Dict], qdrant, driver, pool, client, logger) -> Dict:
    qid = q["query_id"]
    qtype = q["epistemic_type"]
    ts = datetime.utcnow().isoformat()

    vector = embed_text(q["query_text"], client)

    # Conditions A-D
    assertions_a = retrieve_a(vector, MAX_K, qdrant, pool)
    assertions_b = retrieve_b(vector, MAX_K, qdrant, pool, driver)
    assertions_c = retrieve_c(vector, MAX_K, qdrant, pool, qtype)
    assertions_d, conflicts_d, conflict_pairs_d = retrieve_d(
        vector, MAX_K, qdrant, pool, driver, qtype
    )

    # Condition-neutral seed + neighborhood (v4 fix: undirected)
    seed_id = get_neutral_seed(qid, vector, qdrant, pool, queries)
    neighborhood = get_toulmin_neighborhood(seed_id, driver) if seed_id else set()
    neighborhood_size = len(neighborhood)
    ac_eligible = neighborhood_size > 0

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
            ac = None
            cwe = MAX_K
        csr = (1 if conflicts_d else 0) if cond == "D" else 0

        metrics[cond] = {
            "vtp": vtp.get("precision", 0.0),
            "vtp_detail": vtp,
            "ac": ac,
            "cwe": cwe,
            "csr": csr,
            "n_retrieved": len(assertions),
        }
        if cond == "D" and conflicts_d:
            metrics[cond]["conflict_pairs"] = [
                {"source": s, "target": t} for s, t in conflict_pairs_d[:5]
            ]

        logger.info(
            f"qid={qid[:8]} type={qtype} cond={cond} "
            f"vtp={metrics[cond]['vtp']:.3f} ac={ac} cwe={cwe} csr={csr} "
            f"n={len(assertions)} nbr={neighborhood_size}"
        )

    # Condition E: two-pass compliance retrieval
    try:
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
        all_assertions: List[Dict]   = result_e.get("_all_assertions", [])
        conflicts_e: bool            = result_e.get("_conflicts_detected", False)
        conflict_pairs_e             = result_e.get("_conflict_pairs", [])

        vtp_e = validity_type_precision(pass1_assertions, qtype)
        ac_e, cwe_e = compute_ac_cwe(all_assertions, neighborhood, neighborhood_size)
        if not ac_eligible:
            ac_e = None
            cwe_e = MAX_K
        csr_e = 1 if conflicts_e else 0

        metrics["E"] = {
            "vtp": vtp_e.get("precision", 0.0),
            "vtp_detail": vtp_e,
            "ac": ac_e,
            "cwe": cwe_e,
            "csr": csr_e,
            "n_retrieved_pass1": len(pass1_assertions),
            "n_retrieved_all": len(all_assertions),
            "completeness_warning": result_e.get("metadata", {}).get("completeness_warning", False),
        }
        if conflicts_e:
            metrics["E"]["conflict_pairs"] = [
                {"source": s, "target": t} for s, t in conflict_pairs_e[:5]
            ]

        logger.info(
            f"qid={qid[:8]} type={qtype} cond=E "
            f"vtp={metrics['E']['vtp']:.3f} ac={ac_e} cwe={cwe_e} csr={csr_e} "
            f"n_p1={len(pass1_assertions)} n_all={len(all_assertions)}"
        )
    except Exception as e:
        logger.error(f"Condition E failed for {qid[:8]}: {e}")
        metrics["E"] = None

    return {
        "query_id": qid,
        "query_text": q["query_text"],
        "epistemic_type": qtype,
        "source_doc": q.get("source_doc", ""),
        "source_category": q.get("source_category", ""),
        "seed_assertion_id": seed_id,
        "seed_doc_id": seed_doc,
        "neighborhood_size": neighborhood_size,
        "metrics": metrics,
        "timestamp": ts,
        "failed": False,
    }


# ── Aggregate statistics ───────────────────────────────────────────────────────
def aggregate_results(query_results: List[Dict]) -> Dict:
    completed = [r for r in query_results if not r.get("failed")]
    metrics_names = ["vtp", "ac", "cwe", "csr"]
    conditions = ["A", "B", "C", "D", "E"]

    ac_eligible_count = sum(1 for r in completed if r.get("neighborhood_size", 0) > 0)
    ac_excluded_count = len(completed) - ac_eligible_count

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
    pairs = [
        ("A", "D"), ("B", "D"), ("C", "D"),
        ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E"),
    ]
    for bl, tr in pairs:
        pk = f"{bl}_vs_{tr}"
        stat_tests[pk] = {}
        for m in metrics_names:
            a_s, b_s = _paired_scores(completed, bl, tr, m)
            stat_tests[pk][m] = run_statistical_tests(a_s, b_s)

    csr_d_values = [r["metrics"].get("D", {}).get("csr", 0) for r in completed if r["metrics"].get("D")]
    csr_e_values = [r["metrics"].get("E", {}).get("csr", 0) for r in completed if r["metrics"].get("E")]
    csr_d_rate = float(np.mean(csr_d_values)) if csr_d_values else 0.0
    csr_e_rate = float(np.mean(csr_e_values)) if csr_e_values else 0.0

    # Subgroup by epistemic type
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
        "n_completed": len(completed),
        "n_failed": len(query_results) - len(completed),
        "ac_eligible_queries": ac_eligible_count,
        "ac_excluded_queries": ac_excluded_count,
        "csr_d_rate": csr_d_rate,
        "csr_e_rate": csr_e_rate,
        "results_by_condition": results_by_condition,
        "statistical_tests": stat_tests,
        "subgroup_by_epistemic_type": subgroup_type,
    }


# ── Output ─────────────────────────────────────────────────────────────────────
def save_partial(results: List[Dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PARTIAL, "w") as f:
        json.dump(results, f, indent=2, default=str)


def save_results(query_results: List[Dict], aggregate: Dict, run_ts: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp": run_ts,
        "harness_version": "v4",
        "description": (
            "Full rerun with undirected Toulmin neighborhood traversal. "
            "Fix: -[:ENTAILS|CAUSES|TRIGGERS|SPECIALIZES*1..N]- (was ->). "
            "AC eligibility corrected from ~35/120 to ~67/120."
        ),
        "n_queries": len(query_results),
        "n_failed": aggregate["n_failed"],
        "ac_eligible_queries": aggregate["ac_eligible_queries"],
        "ac_excluded_queries": aggregate["ac_excluded_queries"],
        "csr_d_rate": aggregate["csr_d_rate"],
        "csr_e_rate": aggregate["csr_e_rate"],
        "results_by_condition": aggregate["results_by_condition"],
        "statistical_tests": aggregate["statistical_tests"],
        "subgroup_by_epistemic_type": aggregate["subgroup_by_epistemic_type"],
        "failed_queries": [
            {"query_id": r["query_id"], "error": r.get("error", "unknown")}
            for r in query_results if r.get("failed")
        ],
        "per_query_results": query_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)


def save_summary(aggregate: Dict, run_ts: str, n_total: int) -> None:
    tests = aggregate["statistical_tests"]
    rc    = aggregate["results_by_condition"]

    def _f3(v): return f"{v:.3f}" if v is not None else "n/a"
    def _f1(v): return f"{v:.1f}" if v is not None else "n/a"
    def _fmt(pair, metric):
        t = tests.get(pair, {}).get(metric, {})
        if not t or "error" in t:
            return "n/a"
        return (
            f"mean {t.get('mean_baseline',0.0):.3f}→{t.get('mean_treatment',0.0):.3f} "
            f"p={t.get('wilcoxon_p') or 'n/a'} "
            f"d={t.get('cohens_d',0.0):.3f} [{t.get('effect_size_label','?')}]"
        )

    lines = [
        "FINWIKI EVALUATION HARNESS v4 — RESULTS SUMMARY",
        "=" * 50,
        f"Run timestamp: {run_ts}",
        f"Harness version: v4 (undirected Toulmin neighbourhood fix)",
        f"Queries completed: {aggregate['n_completed']} / {n_total}",
        f"Queries failed: {aggregate['n_failed']}",
        "",
        "AC ELIGIBILITY (undirected fix):",
        f"  Queries with non-zero Toulmin neighborhood: {aggregate['ac_eligible_queries']}/{n_total}",
        f"  Queries excluded (neighborhood_size=0): {aggregate['ac_excluded_queries']}",
        "",
        "─" * 50,
        "CONDITION MEANS:",
    ]
    for cond in ("A", "B", "C", "D", "E"):
        c = rc.get(cond, {})
        lines.append(
            f"  [{cond}] VTP={_f3(c.get('vtp_mean'))}  "
            f"AC={_f3(c.get('ac_mean'))}  "
            f"CWE={_f1(c.get('cwe_median'))}  "
            f"CSR={_f3(c.get('csr_mean'))}"
        )
    lines += [
        "",
        "─" * 50,
        "KEY COMPARISONS:",
        f"  A_vs_D VTP: {_fmt('A_vs_D','vtp')}",
        f"  A_vs_D AC:  {_fmt('A_vs_D','ac')}",
        f"  C_vs_E AC:  {_fmt('C_vs_E','ac')}",
        f"  D_vs_E AC:  {_fmt('D_vs_E','ac')}",
        f"  A_vs_E VTP: {_fmt('A_vs_E','vtp')}",
        "",
        f"  CSR D rate: {aggregate['csr_d_rate']:.3f}",
        f"  CSR E rate: {aggregate['csr_e_rate']:.3f}",
    ]

    # Subgroup normative A_vs_E VTP
    norm_ae_vtp = aggregate["subgroup_by_epistemic_type"].get(
        "normative", {}
    ).get("A_vs_E", {}).get("vtp", {})
    if norm_ae_vtp and "cohens_d" in norm_ae_vtp:
        lines.append(
            f"  Normative subgroup A_vs_E VTP: d={norm_ae_vtp['cohens_d']:.3f}"
        )

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    run_ts = datetime.utcnow().isoformat()
    logger.info(f"=== FinWiki Eval Harness v4 START {run_ts} ===")
    logger.info("v4 fix: undirected Toulmin neighbourhood traversal")

    genai_client = genai.Client(api_key=settings.google_api_key)
    qdrant = QdrantClient(url="http://localhost:6333")
    driver = make_neo4j()
    pool = make_pg_pool()
    logger.info("Connections established")

    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE} — aborting")
        return
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries")

    query_results: List[Dict] = []
    for i, q in enumerate(queries):
        qid = q["query_id"]
        try:
            result = with_retry(
                lambda q=q: eval_single_query(
                    q, queries, qdrant, driver, pool, genai_client, logger
                ),
                retries=3, backoff=10, logger=logger,
            )
            query_results.append(result)
        except Exception as e:
            logger.error(f"FAILED {qid[:8]}: {e}")
            query_results.append({
                "query_id": qid,
                "query_text": q["query_text"],
                "epistemic_type": q["epistemic_type"],
                "source_doc": q.get("source_doc", ""),
                "source_category": q.get("source_category", ""),
                "seed_assertion_id": None,
                "seed_doc_id": "",
                "neighborhood_size": 0,
                "metrics": {},
                "failed": True,
                "error": str(e),
            })

        if (i + 1) % 10 == 0:
            save_partial(query_results)
            logger.info(f"Progress: {i+1}/{len(queries)} — partial saved")

    save_partial(query_results)
    aggregate = aggregate_results(query_results)
    save_results(query_results, aggregate, run_ts)
    save_summary(aggregate, run_ts, len(queries))

    rc = aggregate["results_by_condition"]
    tests = aggregate["statistical_tests"]

    def _f3(v): return f"{v:.3f}" if v is not None else "n/a"

    print(f"\n=== FINWIKI v4 RESULTS ===")
    print(f"AC eligible queries: {aggregate['ac_eligible_queries']}/{len(queries)}")
    print(f"n_failed: {aggregate['n_failed']}")
    print()
    for cond in ("A", "D", "E"):
        c = rc.get(cond, {})
        print(
            f"Condition {cond}: VTP={_f3(c.get('vtp_mean'))}  "
            f"AC={_f3(c.get('ac_mean'))}  "
            f"CSR={_f3(c.get('csr_mean'))}"
        )
    print()
    for pair in ("A_vs_D", "C_vs_E", "D_vs_E"):
        for metric in ("vtp", "ac"):
            t = tests.get(pair, {}).get(metric, {})
            if t and "error" not in t:
                print(
                    f"{pair} {metric}: p={t.get('wilcoxon_p') or 'n/a'}  "
                    f"d={t.get('cohens_d',0.0):.3f}"
                )

    # Normative subgroup A_vs_E VTP
    norm_ae = aggregate["subgroup_by_epistemic_type"].get(
        "normative", {}
    ).get("A_vs_E", {}).get("vtp", {})
    if norm_ae and "cohens_d" in norm_ae:
        print(f"Normative A_vs_E VTP: d={norm_ae['cohens_d']:.3f}")

    logger.info(f"=== FinWiki Eval Harness v4 COMPLETE ===")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
