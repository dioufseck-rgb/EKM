"""pipeline/eval_harness_v2.py — FinWiki Evaluation Harness v2

Fixes vs v1:
  Fix 1 — Pool cap: all conditions retrieve top-k ASSERTIONS (not k chunks
           expanded to all their assertions). Adds finwiki_assertions Qdrant
           collection, built lazily on first run using chunk-proxy embeddings.
  Fix 2 — Condition-neutral seed: neighborhood is seeded from the top-1
           assertion in the unfiltered finwiki_assertions collection, computed
           before any condition runs. Stored as query_seed_assertion_id in
           queries.json for reproducibility.
  Fix 3 — Neighborhood metadata: logs neighborhood_size, seed_assertion_id,
           and seed_doc_id per query. Queries with neighborhood_size == 0 are
           excluded from AC analysis (AC = null, not 0.0). Proportion excluded
           is reported in the summary.
  Fix 4 — CSR measures structural conflict detection, not passive collision.
           retrieve_d traverses CONTRADICTS edges during graph expansion and
           sets conflicts_detected=True when found. CSR for Conditions A, B, C
           is 0 by definition. k-invariance CSR sweep is removed.

Usage: python pipeline/eval_harness_v2.py > data/eval/stdout_v2.log 2>&1
"""
import os
os.environ.setdefault("POSTGRES_URL", "postgresql://finwiki:finwiki@localhost:5432/finwiki")
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import psycopg2
import psycopg2.pool
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from scipy import stats as scipy_stats

import google.genai as genai
from google.genai import types as genai_types

from pipeline.config import settings

# ── Directory & file constants ────────────────────────────────────────────────
EVAL_DIR       = Path("data/eval")
LOG_FILE       = EVAL_DIR / "eval_log_v2.txt"
RESULTS_FILE   = EVAL_DIR / "results_v2.json"
RESULTS_PARTIAL = EVAL_DIR / "results_partial_v2.json"
SUMMARY_FILE   = EVAL_DIR / "results_summary_v2.txt"
QUERIES_FILE   = EVAL_DIR / "queries.json"   # shared with v1; adds seed field in-place
K_INV_VTP_FILE = EVAL_DIR / "k_invariance_vtp_v2.json"

MAX_K = 50
K_INV_MAX = 100
AC_THRESHOLD = 0.8

ASSERTIONS_COLLECTION = "finwiki_assertions"
CHUNKS_COLLECTION     = "finwiki_chunks"

CATEGORY_KEYWORDS = {
    "regulatory": ["Basel", "Dodd", "MiFID", "AIFMD", "Sarbanes", "GDPR", "Volcker",
                   "regulation", "directive", "compliance", "supervisory"],
    "risk": ["risk", "VaR", "stress", "volatility", "exposure", "concentration",
             "liquidity", "credit risk", "operational risk"],
    "pricing": ["CAPM", "APT", "Black-Scholes", "arbitrage", "valuation", "pricing",
                "expected return", "beta", "factor model"],
    "product": ["derivative", "swap", "option", "futures", "bond", "security",
                "collateral", "CDO", "mortgage"],
}


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_harness_v2")
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


# ── Assertion-level Qdrant collection (Fix 1) ─────────────────────────────────
def ensure_assertion_collection(
    qdrant: QdrantClient, pool, client: genai.Client, logger
) -> None:
    """
    Build finwiki_assertions Qdrant collection if it doesn't exist.

    Each point represents one assertion. The vector is the embedding of the
    assertion's parent chunk (from finwiki_chunks via qdrant_id_map → chunk_id).
    This requires no new LLM embedding calls while giving assertion-level
    granularity for retrieval. Each assertion gets the semantic vector of its
    source chunk.

    Payload: {assertion_id, document_id, confidence, validity_claim_type,
              discourse_role, chunk_id}
    """
    collections = [c.name for c in qdrant.get_collections().collections]
    if ASSERTIONS_COLLECTION in collections:
        count = qdrant.count(ASSERTIONS_COLLECTION).count
        if count > 0:
            logger.info(
                f"Collection '{ASSERTIONS_COLLECTION}' exists with {count} points — skipping build"
            )
            return
        qdrant.delete_collection(ASSERTIONS_COLLECTION)

    logger.info(f"Building '{ASSERTIONS_COLLECTION}' collection from chunk proxy vectors...")

    # Fetch all assertions with their chunk_ids from PG
    rows = pg_query(pool, """
        SELECT a.assertion_id, a.chunk_id, a.document_id, a.confidence,
               a.validity_claim_type, a.discourse_role
        FROM assertions a
        WHERE a.epistemic_status NOT IN ('deprecated', 'orphaned')
    """)
    logger.info(f"  {len(rows)} assertions to index")

    # Fetch chunk_id → qdrant_point_id mapping
    id_map_rows = pg_query(pool, "SELECT chunk_id, qdrant_point_id FROM qdrant_id_map")
    chunk_to_qdrant = {r[0]: r[1] for r in id_map_rows}

    # Fetch vectors from finwiki_chunks in batches
    qdrant_ids_needed = list({
        chunk_to_qdrant[r[1]]
        for r in rows if r[1] in chunk_to_qdrant
    })
    logger.info(f"  Fetching {len(qdrant_ids_needed)} chunk vectors from Qdrant...")

    chunk_vector_map: Dict[int, List[float]] = {}
    FETCH_BATCH = 500
    for i in range(0, len(qdrant_ids_needed), FETCH_BATCH):
        batch_ids = qdrant_ids_needed[i:i + FETCH_BATCH]
        results = qdrant.retrieve(
            collection_name=CHUNKS_COLLECTION,
            ids=batch_ids,
            with_vectors=True,
        )
        for pt in results:
            chunk_vector_map[pt.id] = pt.vector

    # Build points
    vec_size = None
    if chunk_vector_map:
        vec_size = len(next(iter(chunk_vector_map.values())))
    else:
        # Fallback: embed a dummy text to get dimension
        v = embed_text("financial regulation", client)
        vec_size = len(v)

    qdrant.create_collection(
        collection_name=ASSERTIONS_COLLECTION,
        vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
    )
    logger.info(f"  Created collection (dim={vec_size})")

    points = []
    skipped = 0
    for assertion_id, chunk_id, doc_id, conf, vtype, drole in rows:
        qdrant_chunk_id = chunk_to_qdrant.get(chunk_id)
        vector = chunk_vector_map.get(qdrant_chunk_id) if qdrant_chunk_id else None
        if vector is None:
            skipped += 1
            continue
        qdrant_point_id = abs(hash(assertion_id)) % (2**31)
        points.append(PointStruct(
            id=qdrant_point_id,
            vector=vector,
            payload={
                "assertion_id":       assertion_id,
                "document_id":        doc_id or "",
                "confidence":         float(conf or 0.8),
                "validity_claim_type": vtype or "unclassified",
                "discourse_role":     drole or "unclassified",
                "chunk_id":           chunk_id or "",
            },
        ))

    UPSERT_BATCH = 200
    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(collection_name=ASSERTIONS_COLLECTION, points=points[i:i + UPSERT_BATCH])

    logger.info(
        f"  Indexed {len(points)} assertions into '{ASSERTIONS_COLLECTION}' "
        f"({skipped} skipped — no chunk vector)"
    )


# ── Qdrant helpers ─────────────────────────────────────────────────────────────
def qdrant_top_assertions(
    vector: List[float], k: int, qdrant: QdrantClient,
    validity_type: Optional[str] = None,
) -> List[Dict]:
    """
    Query finwiki_assertions collection directly. Returns top-k payloads
    sorted by cosine similarity. Optional validity_type filter.
    """
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


def qdrant_top_chunks(vector: List[float], k: int, qdrant: QdrantClient) -> List[str]:
    result = qdrant.query_points(
        collection_name=CHUNKS_COLLECTION, query=vector, limit=k, with_payload=True,
    )
    return [h.payload.get("chunk_id", "") for h in result.points if h.payload]


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


def fetch_assertions_for_chunks(
    chunk_ids: List[str], pool, validity_type: Optional[str] = None
) -> List[Dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join(["%s"] * len(chunk_ids))
    params: list = list(chunk_ids)
    sql = f"""
        SELECT assertion_id, claim_text, subject, predicate_type, object_text,
               source_document, source_url, epistemic_status, confidence, domain,
               validity_claim_type, discourse_role
        FROM assertions
        WHERE chunk_id IN ({placeholders})
          AND epistemic_status NOT IN ('deprecated', 'orphaned')
    """
    if validity_type:
        sql += " AND validity_claim_type = %s"
        params.append(validity_type)
    sql += " ORDER BY confidence DESC"
    rows = pg_query(pool, sql, params)
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
    """Given Qdrant assertion payloads, fetch full assertion rows from PG."""
    if not payloads:
        return []
    ids = [p["assertion_id"] for p in payloads if "assertion_id" in p]
    if not ids:
        return []
    rows = fetch_assertions_by_ids(ids, pool)
    # Preserve Qdrant ranking order
    order = {aid: i for i, aid in enumerate(ids)}
    rows.sort(key=lambda a: order.get(a["assertion_id"], 999))
    return rows


# ── Condition-neutral seed (Fix 2) ────────────────────────────────────────────
def get_neutral_seed(
    query_id: str,
    vector: List[float],
    qdrant: QdrantClient,
    pool,
    queries: List[Dict],
) -> Optional[str]:
    """
    Return the fixed neutral seed assertion_id for a query.

    On first call for a query: retrieves top-1 assertion from the unfiltered
    finwiki_assertions collection and stores it in queries.json as
    query_seed_assertion_id.

    On subsequent calls: returns the stored value.
    """
    # Check if already stored
    for q in queries:
        if q["query_id"] == query_id and q.get("query_seed_assertion_id"):
            return q["query_seed_assertion_id"]

    # Compute: top-1 from unfiltered assertion collection
    payloads = qdrant_top_assertions(vector, 1, qdrant)
    if not payloads:
        return None
    seed_id = payloads[0].get("assertion_id")
    if not seed_id:
        return None

    # Persist into queries list and file
    for q in queries:
        if q["query_id"] == query_id:
            q["query_seed_assertion_id"] = seed_id
            break
    with open(QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)

    return seed_id


# ── Retrieval conditions (Fix 1: all return exactly k assertions) ─────────────
def retrieve_a(
    vector: List[float], k: int, qdrant: QdrantClient, pool
) -> List[Dict]:
    """Condition A: Standard vector RAG — top-k assertions, no filter."""
    payloads = qdrant_top_assertions(vector, k, qdrant)
    return _enrich_from_pg(payloads, pool)


def retrieve_b(
    vector: List[float], k: int, qdrant: QdrantClient, pool, driver
) -> List[Dict]:
    """Condition B: Entity GraphRAG — top-k assertion seeds + generic 1-2 hop expansion."""
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


def retrieve_c(
    vector: List[float], k: int, qdrant: QdrantClient, pool, epistemic_type: str
) -> List[Dict]:
    """Condition C: Validity-gated — top-k assertions filtered by validity_claim_type."""
    if epistemic_type == "mixed":
        norm_payloads = qdrant_top_assertions(vector, k, qdrant, validity_type="normative")
        const_payloads = qdrant_top_assertions(vector, k, qdrant, validity_type="constative")
        norm = _enrich_from_pg(norm_payloads, pool)
        const_ = _enrich_from_pg(const_payloads, pool)
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


def _detect_contradicts_in_subgraph(
    assertion_ids: Set[str], driver
) -> List[Tuple[str, str]]:
    """
    Traverse CONTRADICTS edges within the retrieved subgraph.
    Returns list of (source_id, target_id) conflict pairs found.
    This is structural conflict detection: only edges explicitly traversed
    during graph expansion count, not passive co-occurrence.
    """
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


def retrieve_d(
    vector: List[float], k: int, qdrant: QdrantClient, pool, driver,
    epistemic_type: str,
) -> Tuple[List[Dict], bool, List[Tuple[str, str]]]:
    """
    Condition D: Full discourse-typed retrieval.
    Returns (assertions, conflicts_detected, conflict_pairs).

    Steps:
      1. Top-k validity-typed assertion seeds (same as Condition C).
      2. Toulmin expansion (truth-preserving inference edges up to 2 hops).
      3. Regulation-anchored cross-document assembly.
      4. Traverse CONTRADICTS edges within assembled subgraph → set conflicts_detected.
    """
    # Step 1: validity-typed seeds
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

    # Step 2: Toulmin expansion
    seed_ids = [a["assertion_id"] for a in seeds[:5]]
    expanded_ids = _toulmin_expand(seed_ids, driver)
    expanded = fetch_assertions_by_ids(expanded_ids, pool)

    # Step 3: Regulation anchor
    anchor_ids = _regulation_anchor_ids(seeds, driver)
    anchor = fetch_assertions_by_ids(anchor_ids, pool)

    # Merge and cap at k
    existing = {a["assertion_id"] for a in seeds}
    combined = list(seeds)
    for a in expanded + anchor:
        if a["assertion_id"] not in existing:
            combined.append(a)
            existing.add(a["assertion_id"])
    combined = sorted(combined, key=lambda x: x["confidence"], reverse=True)[:k]

    # Step 4: structural conflict detection within assembled subgraph
    subgraph_ids = {a["assertion_id"] for a in combined}
    conflict_pairs = _detect_contradicts_in_subgraph(subgraph_ids, driver)
    conflicts_detected = len(conflict_pairs) > 0

    return combined, conflicts_detected, conflict_pairs


# ── Metrics ───────────────────────────────────────────────────────────────────
def validity_type_precision(
    assertions: List[Dict], epistemic_type: str, k: int = 10
) -> Dict:
    top = assertions[:k]
    if not top:
        return {"precision": 0.0}
    if epistemic_type == "normative":
        p = sum(1 for a in top if a["validity_claim_type"] == "normative") / len(top)
        return {"precision": p}
    elif epistemic_type == "constative":
        p = sum(1 for a in top if a["validity_claim_type"] == "constative") / len(top)
        return {"precision": p}
    else:  # mixed
        has_n = any(a["validity_claim_type"] == "normative" for a in top)
        has_c = any(a["validity_claim_type"] == "constative" for a in top)
        return {
            "precision": float(has_n and has_c),
            "normative_present": has_n,
            "constative_present": has_c,
        }


def get_toulmin_neighborhood(
    assertion_id: str, driver, max_hops: int = 2
) -> Set[str]:
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


def repetitiveness_index(vector: List[float], k: int, qdrant: QdrantClient) -> float:
    try:
        result = qdrant.query_points(
            collection_name=CHUNKS_COLLECTION, query=vector, limit=k, with_payload=False,
        )
        scores = [h.score for h in result.points]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


# ── Statistical analysis ──────────────────────────────────────────────────────
def cohens_d_paired(a: List[float], b: List[float]) -> float:
    diff = np.array(b) - np.array(a)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def rank_biserial(scores_a: List[float], scores_b: List[float]) -> float:
    diff = np.array(scores_b) - np.array(scores_a)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = scipy_stats.rankdata(np.abs(nonzero))
    w_pos = float(np.sum(ranks[nonzero > 0])) if np.any(nonzero > 0) else 0.0
    w_neg = float(np.sum(ranks[nonzero < 0])) if np.any(nonzero < 0) else 0.0
    total = len(nonzero) * (len(nonzero) + 1) / 2
    return float((w_pos - w_neg) / (total + 1e-10))


def bootstrap_ci(
    scores_a: List[float], scores_b: List[float], n_boot: int = 1000
) -> Tuple[float, float]:
    diff = np.array(scores_b) - np.array(scores_a)
    n = len(diff)
    boot = [np.mean(diff[np.random.choice(n, n, replace=True)]) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_statistical_tests(
    scores_a: List[float], scores_b: List[float],
) -> Dict:
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
    out["rank_biserial"] = rank_biserial(a, b)
    lo, hi = bootstrap_ci(a, b)
    out["ci_95"] = [lo, hi]
    d = abs(out["cohens_d"])
    out["effect_size_label"] = (
        "negligible" if d < 0.2 else
        "small" if d < 0.5 else
        "medium" if d < 0.8 else
        "large"
    )
    return out


# ── Per-query evaluation ──────────────────────────────────────────────────────
def eval_single_query(
    q: Dict, queries: List[Dict], qdrant, driver, pool, client, logger
) -> Dict:
    qid = q["query_id"]
    qtype = q["epistemic_type"]
    ts = datetime.utcnow().isoformat()

    vector = embed_text(q["query_text"], client)
    rep_idx = repetitiveness_index(vector, 20, qdrant)

    # Fix 1: all conditions retrieve exactly MAX_K assertions
    assertions_a = retrieve_a(vector, MAX_K, qdrant, pool)
    assertions_b = retrieve_b(vector, MAX_K, qdrant, pool, driver)
    assertions_c = retrieve_c(vector, MAX_K, qdrant, pool, qtype)
    assertions_d, conflicts_d, conflict_pairs_d = retrieve_d(
        vector, MAX_K, qdrant, pool, driver, qtype
    )

    # Fix 2: condition-neutral seed for neighborhood
    seed_id = get_neutral_seed(qid, vector, qdrant, pool, queries)
    neighborhood = get_toulmin_neighborhood(seed_id, driver) if seed_id else set()

    # Fix 3: log neighborhood metadata
    seed_doc = ""
    if seed_id:
        rows = pg_query(pool, "SELECT source_document FROM assertions WHERE assertion_id = %s", [seed_id])
        seed_doc = rows[0][0] if rows else ""

    neighborhood_size = len(neighborhood)
    ac_eligible = neighborhood_size > 0  # exclude if no neighborhood

    metrics: Dict[str, Dict] = {}
    for cond, assertions in [
        ("A", assertions_a),
        ("B", assertions_b),
        ("C", assertions_c),
        ("D", assertions_d),
    ]:
        vtp = validity_type_precision(assertions, qtype)

        # Fix 1: pool is exactly k assertions; AC is computed over this fixed pool
        ac = None
        cwe = MAX_K
        if ac_eligible:
            retrieved_ids = {a["assertion_id"] for a in assertions}
            ac = len(retrieved_ids & neighborhood) / neighborhood_size
            sorted_ids = [a["assertion_id"] for a in assertions]
            for k_val in range(1, len(sorted_ids) + 1):
                if len(set(sorted_ids[:k_val]) & neighborhood) / neighborhood_size >= AC_THRESHOLD:
                    cwe = k_val
                    break

        # Fix 4: CSR — structural conflict detection for D; 0 by definition for A/B/C
        if cond == "D":
            csr = 1 if conflicts_d else 0
        else:
            csr = 0

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
            f"{ts} qid={qid[:8]} type={qtype} cond={cond} "
            f"vtp={metrics[cond]['vtp']:.3f} ac={ac} cwe={cwe} csr={csr} "
            f"n={len(assertions)} seed_nbr={neighborhood_size} rep={rep_idx:.3f} status=ok"
        )

    return {
        "query_id": qid,
        "query_text": q["query_text"],
        "epistemic_type": qtype,
        "source_doc": q.get("source_doc", ""),
        "source_category": q.get("source_category", ""),
        "repetitiveness_index": rep_idx,
        "seed_assertion_id": seed_id,
        "seed_doc_id": seed_doc,
        "neighborhood_size": neighborhood_size,
        "metrics": metrics,
        "timestamp": ts,
        "failed": False,
    }


# ── K-invariance (VTP only; CSR removed per Fix 4) ───────────────────────────
def compute_k_invariance_vtp(queries, qdrant, pool, client, driver, logger) -> Dict:
    normative_qs = [q for q in queries if q["epistemic_type"] == "normative"][:30]
    k_vals = list(range(1, K_INV_MAX + 1))
    curves_a: Dict[int, List[float]] = {k: [] for k in k_vals}
    curves_d: Dict[int, List[float]] = {k: [] for k in k_vals}
    logger.info(f"K-invariance VTP: {len(normative_qs)} normative queries to k={K_INV_MAX}")
    for i, q in enumerate(normative_qs):
        try:
            vector = embed_text(q["query_text"], client)
            a_all = retrieve_a(vector, K_INV_MAX, qdrant, pool)
            d_all, _, _ = retrieve_d(vector, K_INV_MAX, qdrant, pool, driver, "normative")
            for k in k_vals:
                curves_a[k].append(validity_type_precision(a_all, "normative", k).get("precision", 0.0))
                curves_d[k].append(validity_type_precision(d_all, "normative", k).get("precision", 0.0))
            if (i + 1) % 5 == 0:
                logger.info(f"K-invariance VTP: {i+1}/{len(normative_qs)}")
        except Exception as e:
            logger.warning(f"K-inv VTP failed q={q['query_id'][:8]}: {e}")
    return {
        "vtp_normative_A": [float(np.mean(curves_a[k])) if curves_a[k] else 0.0 for k in k_vals],
        "vtp_normative_D": [float(np.mean(curves_d[k])) if curves_d[k] else 0.0 for k in k_vals],
        "k_values": k_vals,
    }


# ── Aggregate statistics ──────────────────────────────────────────────────────
def _paired_scores(completed: List[Dict], cond_a: str, cond_b: str, metric: str):
    a_vals, b_vals = [], []
    for r in completed:
        va = r["metrics"].get(cond_a, {}).get(metric)
        vb = r["metrics"].get(cond_b, {}).get(metric)
        if va is not None and vb is not None:
            a_vals.append(float(va))
            b_vals.append(float(vb))
    return a_vals, b_vals


def aggregate_results(query_results: List[Dict]) -> Dict:
    completed = [r for r in query_results if not r.get("failed")]
    metrics_names = ["vtp", "ac", "cwe", "csr"]
    conditions = ["A", "B", "C", "D"]

    # Fix 3: count AC-eligible queries (neighborhood_size > 0)
    ac_eligible_count = sum(1 for r in completed if r.get("neighborhood_size", 0) > 0)
    ac_excluded_count = len(completed) - ac_eligible_count

    results_by_condition: Dict[str, Dict] = {}
    for cond in conditions:
        results_by_condition[cond] = {}
        for m in metrics_names:
            vals = [
                float(r["metrics"].get(cond, {}).get(m, 0))
                for r in completed
                if r["metrics"].get(cond, {}).get(m) is not None
            ]
            results_by_condition[cond][f"{m}_mean"] = float(np.mean(vals)) if vals else None
            results_by_condition[cond][f"{m}_median"] = float(np.median(vals)) if vals else None
            results_by_condition[cond][f"{m}_n"] = len(vals)

    stat_tests: Dict[str, Dict] = {}
    for bl, tr in [("A", "D"), ("B", "D"), ("C", "D")]:
        pk = f"{bl}_vs_{tr}"
        stat_tests[pk] = {}
        for m in metrics_names:
            a_s, b_s = _paired_scores(completed, bl, tr, m)
            stat_tests[pk][m] = run_statistical_tests(a_s, b_s)

    # CSR rate for Condition D
    csr_d_values = [r["metrics"].get("D", {}).get("csr", 0) for r in completed]
    csr_d_rate = float(np.mean(csr_d_values)) if csr_d_values else 0.0

    # Subgroup by epistemic type
    subgroup_type: Dict[str, Dict] = {}
    for qtype in ("normative", "constative", "mixed"):
        subset = [r for r in completed if r["epistemic_type"] == qtype]
        subgroup_type[qtype] = {}
        for bl, tr in [("A", "D"), ("B", "D"), ("C", "D")]:
            pk = f"{bl}_vs_{tr}"
            subgroup_type[qtype][pk] = {}
            for m in metrics_names:
                a_s, b_s = _paired_scores(subset, bl, tr, m)
                subgroup_type[qtype][pk][m] = run_statistical_tests(a_s, b_s)

    # Subgroup by repetitiveness
    rep_vals = [r["repetitiveness_index"] for r in completed]
    median_rep = float(np.median(rep_vals)) if rep_vals else 0.5
    high_rep = [r for r in completed if r["repetitiveness_index"] >= median_rep]
    low_rep = [r for r in completed if r["repetitiveness_index"] < median_rep]
    subgroup_rep: Dict[str, Dict] = {}
    for stratum_name, stratum in [("high", high_rep), ("low", low_rep)]:
        subgroup_rep[stratum_name] = {}
        for bl, tr in [("A", "D"), ("B", "D"), ("C", "D")]:
            pk = f"{bl}_vs_{tr}"
            subgroup_rep[stratum_name][pk] = {}
            for m in metrics_names:
                a_s, b_s = _paired_scores(stratum, bl, tr, m)
                subgroup_rep[stratum_name][pk][m] = run_statistical_tests(a_s, b_s)

    return {
        "n_completed": len(completed),
        "n_failed": len(query_results) - len(completed),
        "ac_eligible_queries": ac_eligible_count,
        "ac_excluded_queries": ac_excluded_count,
        "csr_d_rate": csr_d_rate,
        "results_by_condition": results_by_condition,
        "statistical_tests": stat_tests,
        "subgroup_by_epistemic_type": subgroup_type,
        "subgroup_by_repetitiveness": subgroup_rep,
    }


# ── Output writers ─────────────────────────────────────────────────────────────
def save_partial(results: List[Dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PARTIAL, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _fmt(tests: Dict, pair: str, metric: str) -> str:
    t = tests.get(pair, {}).get(metric, {})
    if not t or "error" in t:
        return "n/a"
    return (
        f"{t.get('mean_baseline', 0.0):.3f} vs {t.get('mean_treatment', 0.0):.3f} "
        f"| p={t.get('wilcoxon_p') or 'n/a'} "
        f"| d={t.get('cohens_d', 0.0):.3f} [{t.get('effect_size_label', '?')}]"
    )


def _verdict(tests: Dict, pair: str, metric: str) -> str:
    t = tests.get(pair, {}).get(metric, {})
    if not t or "error" in t:
        return "NOT SUPPORTED (insufficient data)"
    p = t.get("wilcoxon_p")
    d = abs(t.get("cohens_d") or 0.0)
    if p is None:
        return "NOT SUPPORTED (test failed)"
    if p < 0.05 and d >= 0.5:
        return "SUPPORTED"
    if p < 0.05 and d >= 0.2:
        return "PARTIALLY SUPPORTED"
    if p < 0.1:
        return "PARTIALLY SUPPORTED"
    return "NOT SUPPORTED"


def save_results(query_results, aggregate, k_inv_vtp, run_ts) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp": run_ts,
        "harness_version": "v2",
        "fixes_applied": [
            "Fix1: assertion-level pool cap at k",
            "Fix2: condition-neutral seed via unfiltered top-1 assertion",
            "Fix3: neighborhood_size logged; ac_eligible excludes size==0",
            "Fix4: CSR measures structural conflict detection in D only",
        ],
        "n_queries": len(query_results),
        "n_failed": aggregate["n_failed"],
        "ac_eligible_queries": aggregate["ac_eligible_queries"],
        "ac_excluded_queries": aggregate["ac_excluded_queries"],
        "csr_d_rate": aggregate["csr_d_rate"],
        "query_distribution": {
            t: sum(1 for r in query_results if r.get("epistemic_type") == t)
            for t in ("normative", "constative", "mixed")
        },
        "results_by_condition": aggregate["results_by_condition"],
        "statistical_tests": aggregate["statistical_tests"],
        "subgroup_by_epistemic_type": aggregate["subgroup_by_epistemic_type"],
        "subgroup_by_repetitiveness": aggregate["subgroup_by_repetitiveness"],
        "k_invariance_vtp": k_inv_vtp,
        "failed_queries": [
            {"query_id": r["query_id"], "error": r.get("error", "unknown")}
            for r in query_results if r.get("failed")
        ],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)


def save_summary(aggregate, k_inv_vtp, run_ts, n_total) -> None:
    tests = aggregate["statistical_tests"]
    by_type = aggregate["subgroup_by_epistemic_type"]
    by_rep = aggregate["subgroup_by_repetitiveness"]
    rc = aggregate["results_by_condition"]

    ac_elig = aggregate["ac_eligible_queries"]
    ac_excl = aggregate["ac_excluded_queries"]
    ac_pct_excl = 100 * ac_excl / (ac_elig + ac_excl) if (ac_elig + ac_excl) > 0 else 0

    def cwe_str(cond):
        v = rc.get(cond, {}).get("cwe_median")
        return f"k={v:.1f}" if v is not None else "k=n/a"

    vtp_a = k_inv_vtp.get("vtp_normative_A", [])
    plateau_k = "n/a"
    if len(vtp_a) > 10:
        diffs = [abs(vtp_a[i] - vtp_a[i-1]) for i in range(10, len(vtp_a))]
        idx = next((i for i, d in enumerate(diffs) if d < 0.005), None)
        plateau_k = str(idx + 11) if idx is not None else ">100"

    p3_by_type = {
        t: by_type.get(t, {}).get("C_vs_D", {}).get("vtp", {}).get("cohens_d")
        for t in ("normative", "constative", "mixed")
    }
    p4_high = by_rep.get("high", {}).get("B_vs_D", {}).get("ac", {}).get("cohens_d")
    p4_low  = by_rep.get("low",  {}).get("B_vs_D", {}).get("ac", {}).get("cohens_d")

    n_props_supported = sum(
        "SUPPORTED" in _verdict(tests, pair, metric)
        for pair, metric in [("A_vs_D","vtp"),("A_vs_D","ac"),("C_vs_D","vtp"),("B_vs_D","ac")]
    )
    all_d = [
        abs(tests.get("A_vs_D",{}).get("vtp",{}).get("cohens_d") or 0.0),
        abs(tests.get("A_vs_D",{}).get("ac",{}).get("cohens_d") or 0.0),
        abs(tests.get("C_vs_D",{}).get("vtp",{}).get("cohens_d") or 0.0),
        abs(tests.get("B_vs_D",{}).get("ac",{}).get("cohens_d") or 0.0),
    ]
    if n_props_supported == 4 and min(all_d) > 0.5:
        venue = "MISQ special issue"
    elif n_props_supported >= 2:
        venue = "SIGIR or EMNLP systems track"
    else:
        venue = "Revisit theoretical model before submission"

    csr_d_rate = aggregate.get("csr_d_rate", 0.0)

    lines = [
        "FINWIKI EVALUATION HARNESS v2 — RESULTS SUMMARY",
        "=" * 49,
        f"Run timestamp: {run_ts}",
        f"Harness version: v2 (pool cap + neutral seed + conflict detection)",
        f"Queries completed: {aggregate['n_completed']} / {n_total}",
        f"Queries failed: {aggregate['n_failed']}",
        "",
        "AC ELIGIBILITY (Fix 3):",
        f"  Queries with non-zero Toulmin neighborhood: {ac_elig} / {n_total} ({100-ac_pct_excl:.1f}%)",
        f"  Queries excluded from AC analysis (neighborhood_size=0): {ac_excl} ({ac_pct_excl:.1f}%)",
        "",
        f"PROPOSITION 1 (Epistemic type mismatch → compliance risk): [{_verdict(tests,'A_vs_D','vtp')}]",
        f"  VTP Condition A vs Condition D: {_fmt(tests,'A_vs_D','vtp')}",
        f"  Interpretation: {'Discourse-typed retrieval achieves higher validity-type precision.' if 'SUPPORTED' in _verdict(tests,'A_vs_D','vtp') else 'No significant VTP difference between conditions.'}",
        "",
        f"PROPOSITION 2 (Argumentative truncation → compliance risk): [{_verdict(tests,'A_vs_D','ac')}]",
        f"  AC Condition A vs Condition D: {_fmt(tests,'A_vs_D','ac')}",
        f"  CWE Condition A: {cwe_str('A')} vs Condition D: {cwe_str('D')} | {_fmt(tests,'A_vs_D','cwe')}",
        f"  AC n_pairs (eligible only): {tests.get('A_vs_D',{}).get('ac',{}).get('n_pairs','n/a')}",
        f"  Interpretation: {'Discourse-typed retrieval achieves greater argumentative completeness at smaller k.' if 'SUPPORTED' in _verdict(tests,'A_vs_D','ac') else 'No significant difference in argumentative completeness.'}",
        "",
        f"PROPOSITION 3 (Validity-gated filtering → compliance risk reduction): [{_verdict(tests,'C_vs_D','vtp')}]",
        f"  VTP Condition C vs Condition D: {_fmt(tests,'C_vs_D','vtp')}",
        f"  Effect size by epistemic type — normative: {p3_by_type['normative']} constative: {p3_by_type['constative']} mixed: {p3_by_type['mixed']}",
        f"  Interpretation: {'Full discourse-typed retrieval improves VTP beyond validity-gating alone.' if 'SUPPORTED' in _verdict(tests,'C_vs_D','vtp') else 'Validity-gating alone is sufficient; regulation anchor adds marginal VTP benefit.'}",
        "",
        f"PROPOSITION 4 (Regulation-anchored assembly → compliance risk reduction): [{_verdict(tests,'B_vs_D','ac')}]",
        f"  AC Condition B vs Condition D: {_fmt(tests,'B_vs_D','ac')}",
        f"  Effect size by repetitiveness — high: {p4_high} low: {p4_low}",
        f"  Interpretation: {'Regulation-anchored cross-document assembly improves AC over generic GraphRAG.' if 'SUPPORTED' in _verdict(tests,'B_vs_D','ac') else 'Generic graph expansion is comparable to regulation-anchored assembly.'}",
        "",
        "CONFLICT DETECTION (Fix 4 — structural, not passive):",
        f"  Condition D CSR rate (conflicts structurally detected): {csr_d_rate:.3f}",
        f"  Conditions A, B, C CSR: 0 by definition (no graph traversal)",
        "",
        "K-INVARIANCE (VTP only; CSR sweep removed per Fix 4):",
        f"  VTP plateau for Condition A at k={plateau_k}",
        "",
        "VENUE RECOMMENDATION:",
        f"  {venue}",
    ]
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test(queries, qdrant, driver, pool, client, logger) -> bool:
    logger.info("=== SMOKE TEST START ===")
    passed = True
    for q in queries[:2]:
        try:
            vector = embed_text(q["query_text"], client)
            a_r = retrieve_a(vector, 5, qdrant, pool)
            b_r = retrieve_b(vector, 5, qdrant, pool, driver)
            c_r = retrieve_c(vector, 5, qdrant, pool, q["epistemic_type"])
            d_r, cd, cp = retrieve_d(vector, 5, qdrant, pool, driver, q["epistemic_type"])
            for cname, res in [("A",a_r),("B",b_r),("C",c_r),("D",d_r)]:
                logger.info(f"  [{cname}] n={len(res)} vtp={validity_type_precision(res, q['epistemic_type']).get('precision',0.0):.3f}")
            logger.info(f"  SMOKE query {q['query_id'][:8]}: PASS (D conflicts={cd})")
        except Exception as e:
            logger.warning(f"  SMOKE query {q['query_id'][:8]}: FAIL — {e}")
            passed = False
    status = "SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED — proceeding anyway per spec"
    logger.info(f"=== {status} ===")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    run_ts = datetime.utcnow().isoformat()
    logger.info(f"=== FinWiki Eval Harness v2 START {run_ts} ===")

    genai_client = genai.Client(api_key=settings.google_api_key)
    qdrant = QdrantClient(url="http://localhost:6333")
    driver = make_neo4j()
    pool = make_pg_pool()
    logger.info("Connections established")

    # Fix 1: build assertion-level Qdrant collection if needed
    ensure_assertion_collection(qdrant, pool, genai_client, logger)

    # Load existing queries (do not regenerate)
    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE} — aborting")
        return
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {QUERIES_FILE}")

    smoke_test(queries[:2], qdrant, driver, pool, genai_client, logger)

    query_results: List[Dict] = []
    for i, q in enumerate(queries):
        try:
            result = with_retry(
                lambda q=q: eval_single_query(
                    q, queries, qdrant, driver, pool, genai_client, logger
                ),
                retries=3, backoff=10, logger=logger,
            )
            query_results.append(result)
        except Exception as e:
            logger.error(f"FAILED query {q['query_id'][:8]} type={q['epistemic_type']}: {e}")
            query_results.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "epistemic_type": q["epistemic_type"],
                "source_doc": q.get("source_doc", ""),
                "source_category": q.get("source_category", ""),
                "repetitiveness_index": 0.0,
                "seed_assertion_id": None,
                "seed_doc_id": "",
                "neighborhood_size": 0,
                "metrics": {},
                "failed": True,
                "error": str(e),
            })
        if (i + 1) % 10 == 0:
            save_partial(query_results)
            logger.info(f"Progress: {i+1}/{len(queries)} — partial results saved to {RESULTS_PARTIAL}")

    logger.info("Main eval loop complete. Running k-invariance VTP sweep...")

    k_inv_vtp: Dict = {}
    try:
        k_inv_vtp = compute_k_invariance_vtp(
            queries, qdrant, pool, genai_client, driver, logger
        )
        with open(K_INV_VTP_FILE, "w") as f:
            json.dump(k_inv_vtp, f, indent=2)
        logger.info(f"K-invariance VTP saved to {K_INV_VTP_FILE}")
    except Exception as e:
        logger.error(f"K-invariance VTP failed: {e}")

    logger.info("Computing aggregate statistics...")
    aggregate = aggregate_results(query_results)
    save_partial(query_results)  # final partial = all results
    save_results(query_results, aggregate, k_inv_vtp, run_ts)
    save_summary(aggregate, k_inv_vtp, run_ts, len(queries))

    logger.info(f"=== FinWiki Eval Harness v2 COMPLETE ===")
    logger.info(f"Completed: {aggregate['n_completed']}/{len(queries)} | Failed: {aggregate['n_failed']}")
    logger.info(f"AC-eligible queries: {aggregate['ac_eligible_queries']}/{len(queries)}")
    logger.info(f"Condition D CSR rate: {aggregate['csr_d_rate']:.3f}")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
