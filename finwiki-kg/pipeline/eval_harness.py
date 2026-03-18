"""pipeline/eval_harness.py — FinWiki Evaluation Harness

Runs overnight, fully autonomous. Produces statistics for four theoretical
propositions about discourse-typed retrieval vs. standard RAG.

Usage: nohup python pipeline/eval_harness.py > data/eval/stdout.log 2>&1 &
"""
# Set localhost connection URLs BEFORE any pipeline import so load_dotenv() doesn't override
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
from scipy import stats as scipy_stats

import google.genai as genai
from google.genai import types as genai_types

from pipeline.config import settings

# ── Directory & file constants ────────────────────────────────────────────────
EVAL_DIR = Path("data/eval")
LOG_FILE = EVAL_DIR / "eval_log.txt"
RESULTS_FILE = EVAL_DIR / "results.json"
RESULTS_PARTIAL = EVAL_DIR / "results_partial.json"
SUMMARY_FILE = EVAL_DIR / "results_summary.txt"
QUERIES_FILE = EVAL_DIR / "queries.json"
K_INV_VTP_FILE = EVAL_DIR / "k_invariance_vtp.json"
K_INV_CSR_FILE = EVAL_DIR / "k_invariance_csr.json"

MAX_K = 50
K_INV_MAX = 100
AC_THRESHOLD = 0.8

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

QUERY_GEN_PROMPT = """\
You are generating evaluation queries for a retrieval system benchmark.

Given this financial document, generate exactly 3 queries:
1. One NORMATIVE query — asks what something requires, mandates, prohibits, or permits
2. One CONSTATIVE query — asks what something is, how something works, or what causes something
3. One MIXED query — requires both normative and constative content to answer completely

Document title: {title}
Document excerpt: {excerpt}

Return JSON only:
{{"normative": "...", "constative": "...", "mixed": "..."}}
"""


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_harness")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ── Retry helper ──────────────────────────────────────────────────────────────
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


# ── Connection helpers ────────────────────────────────────────────────────────
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


# ── Assertion fetchers ────────────────────────────────────────────────────────
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


# ── Qdrant helpers ────────────────────────────────────────────────────────────
def qdrant_top_chunks(vector: List[float], k: int, qdrant: QdrantClient) -> List[str]:
    result = qdrant.query_points(
        collection_name="finwiki_chunks", query=vector, limit=k, with_payload=True,
    )
    return [h.payload.get("chunk_id", "") for h in result.points if h.payload]


# ── Retrieval conditions ──────────────────────────────────────────────────────
def retrieve_a(vector: List[float], k: int, qdrant: QdrantClient, pool) -> List[Dict]:
    """Condition A: Standard vector RAG — top-k chunks → all assertions, no filter."""
    chunk_ids = qdrant_top_chunks(vector, k, qdrant)
    return fetch_assertions_for_chunks(chunk_ids, pool)


def retrieve_b(
    vector: List[float], k: int, qdrant: QdrantClient, pool, driver
) -> List[Dict]:
    """Condition B: Entity GraphRAG — seeds + generic neighbor expansion on any edge."""
    chunk_ids = qdrant_top_chunks(vector, k, qdrant)
    seeds = fetch_assertions_for_chunks(chunk_ids, pool)
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
    neighbors = fetch_assertions_by_ids(new_ids[:50], pool)
    combined = seeds + neighbors
    return sorted(combined, key=lambda x: x["confidence"], reverse=True)


def retrieve_c(
    vector: List[float], k: int, qdrant: QdrantClient, pool, epistemic_type: str
) -> List[Dict]:
    """Condition C: Validity-gated — filter by validity_claim_type."""
    chunk_ids = qdrant_top_chunks(vector, k, qdrant)
    if epistemic_type == "mixed":
        norm = fetch_assertions_for_chunks(chunk_ids, pool, validity_type="normative")
        const = fetch_assertions_for_chunks(chunk_ids, pool, validity_type="constative")
        seen: Set[str] = set()
        merged = []
        for a in norm + const:
            if a["assertion_id"] not in seen:
                merged.append(a)
                seen.add(a["assertion_id"])
        return sorted(merged, key=lambda x: x["confidence"], reverse=True)
    vtype = "normative" if epistemic_type == "normative" else "constative"
    return fetch_assertions_for_chunks(chunk_ids, pool, validity_type=vtype)


def _toulmin_expand(seed_ids: List[str], driver, max_hops: int = 2) -> List[str]:
    if not seed_ids:
        return []
    try:
        with driver.session() as s:
            result = s.run(
                f"""
                UNWIND $ids AS sid
                MATCH path = (a:Assertion {{assertion_id: sid}})
                    -[:ENTAILS|DEFINES|TRIGGERS|SPECIALIZES|SUPERSEDES*1..{max_hops}]->
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
    """Traverse REFERENCES → Regulation ← REFERENCES, return peripheral assertion IDs."""
    if not seeds:
        return []
    try:
        with driver.session() as s:
            # Find regulation with highest total assertion count among seeds
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
            # Authoritative source = doc with most REFERENCES to this regulation
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


def retrieve_d(
    vector: List[float], k: int, qdrant: QdrantClient, pool, driver,
    epistemic_type: str,
) -> List[Dict]:
    """Condition D: Full discourse-typed — validity-gated + Toulmin + regulation anchor."""
    chunk_ids = qdrant_top_chunks(vector, k, qdrant)
    if epistemic_type == "mixed":
        norm = fetch_assertions_for_chunks(chunk_ids, pool, validity_type="normative")
        const = fetch_assertions_for_chunks(chunk_ids, pool, validity_type="constative")
        seen: Set[str] = set()
        seeds = []
        for a in norm + const:
            if a["assertion_id"] not in seen:
                seeds.append(a)
                seen.add(a["assertion_id"])
        seeds = sorted(seeds, key=lambda x: x["confidence"], reverse=True)
    else:
        vtype = "normative" if epistemic_type == "normative" else "constative"
        seeds = fetch_assertions_for_chunks(chunk_ids, pool, validity_type=vtype)

    all_assertions: Dict[str, Dict] = {a["assertion_id"]: a for a in seeds}

    # Toulmin expansion
    seed_ids = [a["assertion_id"] for a in seeds[:5]]
    for aid in _toulmin_expand(seed_ids, driver):
        if aid not in all_assertions:
            for a in fetch_assertions_by_ids([aid], pool):
                all_assertions[a["assertion_id"]] = a

    # Regulation anchor
    for aid in _regulation_anchor_ids(seeds, driver, top_k=5):
        if aid not in all_assertions:
            for a in fetch_assertions_by_ids([aid], pool):
                all_assertions[a["assertion_id"]] = a

    return sorted(all_assertions.values(), key=lambda x: x["confidence"], reverse=True)


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


def get_toulmin_neighborhood(assertion_id: str, driver, max_hops: int = 2) -> Set[str]:
    try:
        with driver.session() as s:
            result = s.run(
                f"""
                MATCH path = (a:Assertion {{assertion_id: $id}})
                    -[:ENTAILS|CAUSES|TRIGGERS|SPECIALIZES*1..{max_hops}]->
                    (b:Assertion)
                WHERE NOT (b.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN DISTINCT b.assertion_id AS nid
                """,
                id=assertion_id,
            )
            return {row["nid"] for row in result}
    except Exception:
        return set()


def conflict_surface_rate(assertion_ids: List[str], pool) -> int:
    if len(assertion_ids) < 2:
        return 0
    try:
        placeholders = ",".join(["%s"] * len(assertion_ids))
        rows = pg_query(
            pool,
            f"""
            SELECT COUNT(*) FROM assertion_relationships
            WHERE relationship_type = 'CONTRADICTS'
              AND source_assertion_id IN ({placeholders})
              AND target_assertion_id IN ({placeholders})
              AND review_status != 'false_positive'
            """,
            assertion_ids + assertion_ids,
        )
        return 1 if rows[0][0] > 0 else 0
    except Exception:
        return 0


def repetitiveness_index(vector: List[float], k: int, qdrant: QdrantClient) -> float:
    try:
        result = qdrant.query_points(
            collection_name="finwiki_chunks", query=vector, limit=k, with_payload=False,
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


# ── Query generation ──────────────────────────────────────────────────────────
def categorize_doc(title: str) -> str:
    tl = title.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw.lower() in tl for kw in kws):
            return cat
    return "product"


def sample_documents(pool, n_per_category: int = 15) -> List[Dict]:
    rows = pg_query(pool, "SELECT DISTINCT source_document FROM assertions ORDER BY source_document")
    docs = [r[0] for r in rows if r[0]]
    by_cat: Dict[str, List[str]] = {c: [] for c in CATEGORY_KEYWORDS}
    for doc in docs:
        by_cat[categorize_doc(doc)].append(doc)
    sampled: List[str] = []
    for cat, doc_list in by_cat.items():
        np.random.shuffle(doc_list)
        sampled.extend(doc_list[:n_per_category])
    result = []
    for title in sampled:
        excerpt_rows = pg_query(
            pool,
            "SELECT claim_text FROM assertions WHERE source_document = %s "
            "AND epistemic_status NOT IN ('deprecated','orphaned') "
            "ORDER BY confidence DESC LIMIT 4",
            [title],
        )
        excerpt = " ".join(r[0] for r in excerpt_rows if r[0])[:500]
        if not excerpt:
            continue
        result.append({"title": title, "excerpt": excerpt, "category": categorize_doc(title)})
    return result


def gen_queries_for_doc(doc: Dict, client: genai.Client, logger) -> Optional[Dict]:
    prompt = QUERY_GEN_PROMPT.format(title=doc["title"], excerpt=doc["excerpt"])
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=settings.flash_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.3
                ),
            )
            raw = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if all(k in data for k in ("normative", "constative", "mixed")):
                return data
        except Exception as e:
            logger.warning(f"Query gen attempt {attempt+1} for {doc['title']}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def load_or_generate_queries(pool, client: genai.Client, logger) -> List[Dict]:
    if QUERIES_FILE.exists():
        logger.info(f"Loading existing queries from {QUERIES_FILE}")
        with open(QUERIES_FILE) as f:
            return json.load(f)
    logger.info("Generating queries from corpus...")
    docs = sample_documents(pool)
    logger.info(f"Sampled {len(docs)} documents for query generation")
    pool_by_type: Dict[str, List[Dict]] = {"normative": [], "constative": [], "mixed": []}
    for i, doc in enumerate(docs):
        result = gen_queries_for_doc(doc, client, logger)
        if result:
            for qtype in ("normative", "constative", "mixed"):
                pool_by_type[qtype].append({
                    "query_id": str(uuid.uuid4()),
                    "query_text": result[qtype],
                    "epistemic_type": qtype,
                    "source_doc": doc["title"],
                    "source_category": doc["category"],
                    "generation_method": "llm_prompted",
                })
        if (i + 1) % 10 == 0:
            logger.info(f"Query generation: {i+1}/{len(docs)} documents")
    # Stratify: up to 50 per type, at least 10 per category per type
    queries: List[Dict] = []
    for qtype in ("normative", "constative", "mixed"):
        qpool = pool_by_type[qtype]
        np.random.shuffle(qpool)
        by_cat: Dict[str, List] = {}
        for q in qpool:
            c = q.get("source_category", "product")
            by_cat.setdefault(c, []).append(q)
        selected: List[Dict] = []
        for c in by_cat:
            selected.extend(by_cat[c][:10])
        remaining = [q for q in qpool if q not in selected]
        selected.extend(remaining[: max(0, 50 - len(selected))])
        queries.extend(selected[:50])
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)
    logger.info(f"Saved {len(queries)} queries to {QUERIES_FILE}")
    return queries


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test(queries: List[Dict], qdrant, driver, pool, client, logger) -> bool:
    logger.info("=== SMOKE TEST START ===")
    passed = True
    for q in queries[:3]:
        try:
            vector = embed_text(q["query_text"], client)
            for cond_name, result in [
                ("A", retrieve_a(vector, 5, qdrant, pool)),
                ("B", retrieve_b(vector, 5, qdrant, pool, driver)),
                ("C", retrieve_c(vector, 5, qdrant, pool, q["epistemic_type"])),
                ("D", retrieve_d(vector, 5, qdrant, pool, driver, q["epistemic_type"])),
            ]:
                vtp = validity_type_precision(result, q["epistemic_type"])
                logger.info(f"  [{cond_name}] n={len(result)} vtp={vtp.get('precision', 0.0):.3f}")
            logger.info(f"  SMOKE query {q['query_id'][:8]}: PASS")
        except Exception as e:
            logger.warning(f"  SMOKE query {q['query_id'][:8]}: FAIL — {e}")
            passed = False
    status = "SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED — proceeding anyway per spec"
    logger.info(f"=== {status} ===")
    return passed


# ── Per-query evaluation ──────────────────────────────────────────────────────
def eval_single_query(q: Dict, qdrant, driver, pool, client, logger) -> Dict:
    qid = q["query_id"]
    qtype = q["epistemic_type"]
    ts = datetime.utcnow().isoformat()

    vector = embed_text(q["query_text"], client)
    rep_idx = repetitiveness_index(vector, 20, qdrant)

    results_by_cond = {
        "A": retrieve_a(vector, MAX_K, qdrant, pool),
        "B": retrieve_b(vector, MAX_K, qdrant, pool, driver),
        "C": retrieve_c(vector, MAX_K, qdrant, pool, qtype),
        "D": retrieve_d(vector, MAX_K, qdrant, pool, driver, qtype),
    }

    # Find shared seed (highest-confidence assertion in Condition A baseline)
    # Neighborhood computed once and shared across conditions for comparability
    seed_a = max(results_by_cond["A"], key=lambda x: x["confidence"]) if results_by_cond["A"] else None
    neighborhood = get_toulmin_neighborhood(seed_a["assertion_id"], driver) if seed_a else set()

    metrics: Dict[str, Dict] = {}
    for cond, assertions in results_by_cond.items():
        vtp = validity_type_precision(assertions, qtype)
        # AC
        ac = None
        cwe = MAX_K
        if neighborhood:
            retrieved_ids = {a["assertion_id"] for a in assertions}
            ac = len(retrieved_ids & neighborhood) / len(neighborhood)
            sorted_ids = [a["assertion_id"] for a in assertions]
            for k_val in range(1, min(len(sorted_ids), MAX_K) + 1):
                if len(set(sorted_ids[:k_val]) & neighborhood) / len(neighborhood) >= AC_THRESHOLD:
                    cwe = k_val
                    break
        # CSR
        aids = [a["assertion_id"] for a in assertions]
        csr = conflict_surface_rate(aids, pool)

        metrics[cond] = {
            "vtp": vtp.get("precision", 0.0),
            "vtp_detail": vtp,
            "ac": ac,
            "cwe": cwe,
            "csr": csr,
            "n_retrieved": len(assertions),
        }
        logger.info(
            f"{ts} qid={qid[:8]} type={qtype} cond={cond} "
            f"vtp={metrics[cond]['vtp']:.3f} ac={ac} cwe={cwe} csr={csr} "
            f"n={len(assertions)} rep={rep_idx:.3f} status=ok"
        )

    return {
        "query_id": qid,
        "query_text": q["query_text"],
        "epistemic_type": qtype,
        "source_doc": q.get("source_doc", ""),
        "source_category": q.get("source_category", ""),
        "repetitiveness_index": rep_idx,
        "metrics": metrics,
        "timestamp": ts,
        "failed": False,
    }


# ── K-invariance ──────────────────────────────────────────────────────────────
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
            d_all = retrieve_d(vector, K_INV_MAX, qdrant, pool, driver, "normative")
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


def compute_k_invariance_csr(queries, qdrant, pool, client, driver, logger) -> Dict:
    sample = queries[:20]
    k_vals = [5, 10, 20, 50, 100]
    csr_a: Dict[int, List[int]] = {k: [] for k in k_vals}
    csr_b: Dict[int, List[int]] = {k: [] for k in k_vals}
    logger.info(f"K-invariance CSR: {len(sample)} queries at k={k_vals}")
    for q in sample:
        try:
            vector = embed_text(q["query_text"], client)
            for k in k_vals:
                a_r = retrieve_a(vector, k, qdrant, pool)
                b_r = retrieve_b(vector, k, qdrant, pool, driver)
                csr_a[k].append(conflict_surface_rate([a["assertion_id"] for a in a_r], pool))
                csr_b[k].append(conflict_surface_rate([a["assertion_id"] for a in b_r], pool))
        except Exception as e:
            logger.warning(f"K-inv CSR failed q={q['query_id'][:8]}: {e}")
    return {
        "csr_A_by_k": {str(k): float(np.mean(csr_a[k])) if csr_a[k] else 0.0 for k in k_vals},
        "csr_B_by_k": {str(k): float(np.mean(csr_b[k])) if csr_b[k] else 0.0 for k in k_vals},
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

    results_by_condition: Dict[str, Dict] = {}
    for cond in conditions:
        results_by_condition[cond] = {}
        for m in metrics_names:
            vals = [float(r["metrics"].get(cond, {}).get(m, 0)) for r in completed
                    if r["metrics"].get(cond, {}).get(m) is not None]
            results_by_condition[cond][f"{m}_mean"] = float(np.mean(vals)) if vals else None
            results_by_condition[cond][f"{m}_median"] = float(np.median(vals)) if vals else None

    stat_tests: Dict[str, Dict] = {}
    for bl, tr in [("A", "D"), ("B", "D"), ("C", "D")]:
        pk = f"{bl}_vs_{tr}"
        stat_tests[pk] = {}
        for m in metrics_names:
            a_s, b_s = _paired_scores(completed, bl, tr, m)
            stat_tests[pk][m] = run_statistical_tests(a_s, b_s)

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
        "results_by_condition": results_by_condition,
        "statistical_tests": stat_tests,
        "subgroup_by_epistemic_type": subgroup_type,
        "subgroup_by_repetitiveness": subgroup_rep,
    }


# ── Output writers ────────────────────────────────────────────────────────────
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


def save_results(query_results, aggregate, k_inv_vtp, k_inv_csr, run_ts) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp": run_ts,
        "n_queries": len(query_results),
        "n_failed": aggregate["n_failed"],
        "query_distribution": {
            t: sum(1 for r in query_results if r.get("epistemic_type") == t)
            for t in ("normative", "constative", "mixed")
        },
        "results_by_condition": aggregate["results_by_condition"],
        "statistical_tests": aggregate["statistical_tests"],
        "subgroup_by_epistemic_type": aggregate["subgroup_by_epistemic_type"],
        "subgroup_by_repetitiveness": aggregate["subgroup_by_repetitiveness"],
        "k_invariance_curves": {
            "vtp_normative_A": k_inv_vtp.get("vtp_normative_A", []),
            "vtp_normative_D": k_inv_vtp.get("vtp_normative_D", []),
            "csr_A_by_k": k_inv_csr.get("csr_A_by_k", {}),
            "csr_B_by_k": k_inv_csr.get("csr_B_by_k", {}),
        },
        "failed_queries": [
            {"query_id": r["query_id"], "error": r.get("error", "unknown")}
            for r in query_results if r.get("failed")
        ],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)


def save_summary(aggregate, k_inv_vtp, k_inv_csr, run_ts, n_total) -> None:
    tests = aggregate["statistical_tests"]
    by_type = aggregate["subgroup_by_epistemic_type"]
    by_rep = aggregate["subgroup_by_repetitiveness"]
    rc = aggregate["results_by_condition"]

    def cwe_str(cond):
        v = rc.get(cond, {}).get("cwe_median")
        return f"k={v:.1f}" if v is not None else "k=n/a"

    # K-invariance plateau
    vtp_a = k_inv_vtp.get("vtp_normative_A", [])
    plateau_k = "n/a"
    if len(vtp_a) > 10:
        diffs = [abs(vtp_a[i] - vtp_a[i-1]) for i in range(10, len(vtp_a))]
        idx = next((i for i, d in enumerate(diffs) if d < 0.005), None)
        plateau_k = str(idx + 11) if idx is not None else ">100"

    csr_a = k_inv_csr.get("csr_A_by_k", {})
    csr_b = k_inv_csr.get("csr_B_by_k", {})
    csr_zero = all(v == 0.0 for v in list(csr_a.values()) + list(csr_b.values()))
    csr_note = "CONFIRMED ZERO across all k" if csr_zero else f"A={csr_a} B={csr_b}"

    p3_by_type = {
        t: by_type.get(t, {}).get("C_vs_D", {}).get("vtp", {}).get("cohens_d")
        for t in ("normative", "constative", "mixed")
    }
    p4_high = by_rep.get("high", {}).get("B_vs_D", {}).get("ac", {}).get("cohens_d")
    p4_low = by_rep.get("low", {}).get("B_vs_D", {}).get("ac", {}).get("cohens_d")

    n_props_supported = sum(
        "SUPPORTED" in _verdict(tests, pair, metric)
        for pair, metric in [("A_vs_D", "vtp"), ("A_vs_D", "ac"), ("C_vs_D", "vtp"), ("B_vs_D", "ac")]
    )
    all_d = [
        abs(tests.get("A_vs_D", {}).get("vtp", {}).get("cohens_d") or 0.0),
        abs(tests.get("A_vs_D", {}).get("ac", {}).get("cohens_d") or 0.0),
        abs(tests.get("C_vs_D", {}).get("vtp", {}).get("cohens_d") or 0.0),
        abs(tests.get("B_vs_D", {}).get("ac", {}).get("cohens_d") or 0.0),
    ]
    if n_props_supported == 4 and min(all_d) > 0.5:
        venue = "MISQ special issue"
    elif n_props_supported >= 2:
        venue = "SIGIR or EMNLP systems track"
    else:
        venue = "Revisit theoretical model before submission"

    lines = [
        "FINWIKI EVALUATION HARNESS — RESULTS SUMMARY",
        "=" * 45,
        f"Run timestamp: {run_ts}",
        f"Queries completed: {aggregate['n_completed']} / {n_total}",
        f"Queries failed: {aggregate['n_failed']}",
        "",
        f"PROPOSITION 1 (Epistemic type mismatch → compliance risk): [{_verdict(tests, 'A_vs_D', 'vtp')}]",
        f"  VTP Condition A vs Condition D: {_fmt(tests, 'A_vs_D', 'vtp')}",
        f"  Interpretation: {'Discourse-typed retrieval achieves higher validity-type precision than standard RAG.' if 'SUPPORTED' in _verdict(tests, 'A_vs_D', 'vtp') else 'No significant difference in validity-type precision between conditions.'}",
        "",
        f"PROPOSITION 2 (Argumentative truncation → compliance risk): [{_verdict(tests, 'A_vs_D', 'ac')}]",
        f"  AC Condition A vs Condition D: {_fmt(tests, 'A_vs_D', 'ac')}",
        f"  CWE Condition A: {cwe_str('A')} vs Condition D: {cwe_str('D')} | {_fmt(tests, 'A_vs_D', 'cwe')}",
        f"  Interpretation: {'Discourse-typed retrieval achieves greater argumentative completeness at smaller k.' if 'SUPPORTED' in _verdict(tests, 'A_vs_D', 'ac') else 'No significant difference in argumentative completeness.'}",
        "",
        f"PROPOSITION 3 (Validity-gated filtering → compliance risk reduction): [{_verdict(tests, 'C_vs_D', 'vtp')}]",
        f"  VTP Condition C vs Condition D: {_fmt(tests, 'C_vs_D', 'vtp')}",
        f"  Effect size by epistemic type — normative: {p3_by_type['normative']} constative: {p3_by_type['constative']} mixed: {p3_by_type['mixed']}",
        f"  Interpretation: {'Full discourse-typed retrieval improves VTP beyond validity-gating alone.' if 'SUPPORTED' in _verdict(tests, 'C_vs_D', 'vtp') else 'Validity-gating alone is sufficient; regulation anchor adds marginal VTP benefit.'}",
        "",
        f"PROPOSITION 4 (Regulation-anchored assembly → compliance risk reduction): [{_verdict(tests, 'B_vs_D', 'ac')}]",
        f"  AC Condition B vs Condition D: {_fmt(tests, 'B_vs_D', 'ac')}",
        f"  Effect size by repetitiveness — high: {p4_high} low: {p4_low}",
        f"  Interpretation: {'Regulation-anchored cross-document assembly improves AC over generic GraphRAG.' if 'SUPPORTED' in _verdict(tests, 'B_vs_D', 'ac') else 'Generic graph expansion is comparable to regulation-anchored assembly.'}",
        "",
        "K-INVARIANCE:",
        f"  VTP plateau for Condition A at k={plateau_k}",
        f"  CSR for Conditions A and B at all k: {csr_note}",
        "",
        "VENUE RECOMMENDATION:",
        f"  {venue}",
    ]
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    run_ts = datetime.utcnow().isoformat()
    logger.info(f"=== FinWiki Eval Harness START {run_ts} ===")

    # Connections
    genai_client = genai.Client(api_key=settings.google_api_key)
    qdrant = QdrantClient(url="http://localhost:6333")
    driver = make_neo4j()
    pool = make_pg_pool()
    logger.info("All connections established")

    # Load or generate queries
    try:
        queries = load_or_generate_queries(pool, genai_client, logger)
    except Exception as e:
        logger.error(f"Query generation failed: {e} — writing empty results and exiting")
        agg = aggregate_results([])
        save_results([], agg, {}, {}, run_ts)
        save_summary(agg, {}, {}, run_ts, 0)
        return

    logger.info(f"Queries ready: {len(queries)}")
    smoke_test(queries[:3], qdrant, driver, pool, genai_client, logger)
    # Proceed regardless of smoke test result

    query_results: List[Dict] = []
    for i, q in enumerate(queries):
        try:
            result = with_retry(
                lambda q=q: eval_single_query(q, qdrant, driver, pool, genai_client, logger),
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
                "metrics": {},
                "failed": True,
                "error": str(e),
            })
        if (i + 1) % 10 == 0:
            save_partial(query_results)
            logger.info(f"Progress: {i+1}/{len(queries)} — partial results saved")

    logger.info("Main eval loop complete. Running k-invariance sweeps...")

    k_inv_vtp: Dict = {}
    k_inv_csr: Dict = {}
    try:
        k_inv_vtp = compute_k_invariance_vtp(queries, qdrant, pool, genai_client, driver, logger)
        with open(K_INV_VTP_FILE, "w") as f:
            json.dump(k_inv_vtp, f, indent=2)
        logger.info(f"K-invariance VTP saved to {K_INV_VTP_FILE}")
    except Exception as e:
        logger.error(f"K-invariance VTP failed: {e}")

    try:
        k_inv_csr = compute_k_invariance_csr(queries, qdrant, pool, genai_client, driver, logger)
        with open(K_INV_CSR_FILE, "w") as f:
            json.dump(k_inv_csr, f, indent=2)
        logger.info(f"K-invariance CSR saved to {K_INV_CSR_FILE}")
    except Exception as e:
        logger.error(f"K-invariance CSR failed: {e}")

    logger.info("Computing aggregate statistics...")
    aggregate = aggregate_results(query_results)
    save_results(query_results, aggregate, k_inv_vtp, k_inv_csr, run_ts)
    save_summary(aggregate, k_inv_vtp, k_inv_csr, run_ts, len(queries))

    logger.info(f"=== FinWiki Eval Harness COMPLETE ===")
    logger.info(f"Completed: {aggregate['n_completed']}/{len(queries)} | Failed: {aggregate['n_failed']}")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
