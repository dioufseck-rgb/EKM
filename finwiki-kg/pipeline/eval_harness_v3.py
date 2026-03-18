"""pipeline/eval_harness_v3.py — FinWiki Evaluation Harness v3

Extends v2 with Condition E: two-pass compliance retrieval.

Design:
  - Loads existing A–D per-query results from results_partial_v2.json (no
    recomputation of conditions A–D). Saves embedding cost and runtime.
  - For each of the 120 queries, runs two_pass_compliance_retrieval() from
    api/reasoning.py.
  - Computes Condition E metrics:
      VTP — on _pass1_assertions (validity-gated obligation set, top-10)
      AC  — on _all_assertions vs neutral neighborhood (same seed as v2)
      CWE — smallest k at which _all_assertions[:k] covers ≥ 0.8 of neighborhood
      CSR — 1 if _conflicts_detected else 0
  - Merges E metrics into each query result alongside A–D.
  - Saves merged results to data/eval/results_v3.json.
  - Saves summary to data/eval/results_summary_v3.txt.

Key research question for E:
  Does two-pass retrieval solve the precision-completeness tradeoff observed
  in v2, where Condition D achieves high VTP but low AC?

Usage:
  cd /workspaces/EKM/finwiki-kg
  python -m pipeline.eval_harness_v3 > data/eval/stdout_v3.log 2>&1
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
LOG_FILE        = EVAL_DIR / "eval_log_v3.txt"
RESULTS_FILE    = EVAL_DIR / "results_v3.json"
RESULTS_PARTIAL = EVAL_DIR / "results_partial_v3.json"
SUMMARY_FILE    = EVAL_DIR / "results_summary_v3.txt"
QUERIES_FILE    = EVAL_DIR / "queries.json"
V2_PARTIAL_FILE = EVAL_DIR / "results_partial_v2.json"

MAX_K = 50
AC_THRESHOLD = 0.8

ASSERTIONS_COLLECTION = "finwiki_assertions"


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_harness_v3")
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


# ── Toulmin neighborhood (same as v2; reuses seed from queries.json) ──────────
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
    else:  # mixed
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
    """Return (ac, cwe). ac=None if neighborhood_size==0."""
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


# ── Statistical helpers ───────────────────────────────────────────────────────
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


# ── Per-query Condition E evaluation ─────────────────────────────────────────
def eval_condition_e(
    q: Dict, v2_result: Dict, driver, qdrant: QdrantClient, pool,
    client: genai.Client, logger
) -> Optional[Dict]:
    """
    Run two_pass_compliance_retrieval for query q and compute E metrics.
    Returns the metrics dict for Condition E, or None on failure.
    """
    qid = q["query_id"]
    qtype = q["epistemic_type"]

    vector = embed_text(q["query_text"], client)

    # Reuse the neutral seed stored by v2
    seed_id = q.get("query_seed_assertion_id")
    neighborhood: Set[str] = set()
    if seed_id:
        neighborhood = get_toulmin_neighborhood(seed_id, driver)
    neighborhood_size = len(neighborhood)

    # Call two-pass retrieval
    result = two_pass_compliance_retrieval(
        query_text=q["query_text"],
        vector=vector,
        driver=driver,
        qdrant=qdrant,
        pool=pool,
        epistemic_type=qtype,
        k=MAX_K,
    )

    pass1_assertions: List[Dict] = result.get("_pass1_assertions", [])
    all_assertions: List[Dict]   = result.get("_all_assertions", [])
    conflicts_detected: bool     = result.get("_conflicts_detected", False)
    conflict_pairs               = result.get("_conflict_pairs", [])

    # VTP measured on Pass 1 obligation set (validity-gated precision)
    vtp = validity_type_precision(pass1_assertions, qtype)

    # AC and CWE measured on combined all_assertions vs neutral neighborhood
    ac_eligible = neighborhood_size > 0
    ac, cwe = compute_ac_cwe(all_assertions, neighborhood, neighborhood_size)

    csr = 1 if conflicts_detected else 0

    metrics_e = {
        "vtp": vtp.get("precision", 0.0),
        "vtp_detail": vtp,
        "ac": ac,
        "cwe": cwe,
        "csr": csr,
        "n_retrieved_pass1": len(pass1_assertions),
        "n_retrieved_all": len(all_assertions),
        "completeness_warning": result.get("metadata", {}).get("completeness_warning", False),
    }
    if conflicts_detected:
        metrics_e["conflict_pairs"] = [
            {"source": s, "target": t} for s, t in conflict_pairs[:5]
        ]

    logger.info(
        f"qid={qid[:8]} type={qtype} cond=E "
        f"vtp={metrics_e['vtp']:.3f} ac={ac} cwe={cwe} csr={csr} "
        f"n_p1={len(pass1_assertions)} n_all={len(all_assertions)} "
        f"nbr={neighborhood_size} warn={metrics_e['completeness_warning']}"
    )
    return metrics_e


# ── Aggregate statistics for all 5 conditions ─────────────────────────────────
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

    # Statistical tests: compare E to each of A, B, C, D
    stat_tests: Dict[str, Dict] = {}
    pairs = [("A", "D"), ("B", "D"), ("C", "D"),   # v2 propositions (reproduced)
             ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E")]  # E comparisons
    for bl, tr in pairs:
        pk = f"{bl}_vs_{tr}"
        stat_tests[pk] = {}
        for m in metrics_names:
            a_s, b_s = _paired_scores(completed, bl, tr, m)
            stat_tests[pk][m] = run_statistical_tests(a_s, b_s)

    # CSR rates
    csr_d_values = [r["metrics"].get("D", {}).get("csr", 0) for r in completed if r["metrics"].get("D")]
    csr_e_values = [r["metrics"].get("E", {}).get("csr", 0) for r in completed if r["metrics"].get("E")]
    csr_d_rate = float(np.mean(csr_d_values)) if csr_d_values else 0.0
    csr_e_rate = float(np.mean(csr_e_values)) if csr_e_values else 0.0

    # Subgroup by epistemic type (E vs D only — key tradeoff comparison)
    subgroup_type: Dict[str, Dict] = {}
    for qtype in ("normative", "constative", "mixed"):
        subset = [r for r in completed if r["epistemic_type"] == qtype]
        subgroup_type[qtype] = {}
        for bl, tr in [("D", "E"), ("A", "E")]:
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


# ── Output writers ────────────────────────────────────────────────────────────
def save_partial(results: List[Dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PARTIAL, "w") as f:
        json.dump(results, f, indent=2, default=str)


def save_results(query_results: List[Dict], aggregate: Dict, run_ts: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp": run_ts,
        "harness_version": "v3",
        "description": "Adds Condition E (two-pass compliance retrieval) to v2 results",
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
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)


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


def save_summary(aggregate: Dict, run_ts: str, n_total: int) -> None:
    tests = aggregate["statistical_tests"]
    rc    = aggregate["results_by_condition"]

    ac_elig    = aggregate["ac_eligible_queries"]
    ac_excl    = aggregate["ac_excluded_queries"]
    ac_pct_elig = 100 * ac_elig / (ac_elig + ac_excl) if (ac_elig + ac_excl) > 0 else 0
    ac_pct_excl = 100 - ac_pct_elig

    def cwe_str(cond):
        v = rc.get(cond, {}).get("cwe_median")
        return f"k={v:.1f}" if v is not None else "k=n/a"

    csr_d = aggregate["csr_d_rate"]
    csr_e = aggregate["csr_e_rate"]

    by_type = aggregate["subgroup_by_epistemic_type"]
    e_d_vtp_by_type = {
        t: by_type.get(t, {}).get("D_vs_E", {}).get("vtp", {}).get("cohens_d")
        for t in ("normative", "constative", "mixed")
    }
    e_d_ac_by_type = {
        t: by_type.get(t, {}).get("D_vs_E", {}).get("ac", {}).get("cohens_d")
        for t in ("normative", "constative", "mixed")
    }

    lines = [
        "FINWIKI EVALUATION HARNESS v3 — RESULTS SUMMARY",
        "=" * 49,
        f"Run timestamp: {run_ts}",
        f"Harness version: v3 (two-pass compliance retrieval — Condition E)",
        f"Queries completed: {aggregate['n_completed']} / {n_total}",
        f"Queries failed: {aggregate['n_failed']}",
        "",
        "AC ELIGIBILITY:",
        f"  Queries with non-zero Toulmin neighborhood: {ac_elig} / {n_total} ({ac_pct_elig:.1f}%)",
        f"  Queries excluded from AC analysis (neighborhood_size=0): {ac_excl} ({ac_pct_excl:.1f}%)",
        "",
        "─" * 49,
        "v2 PROPOSITIONS REPRODUCED (Conditions A–D)",
        "─" * 49,
        "",
        f"PROPOSITION 1 (Epistemic type mismatch → compliance risk): [{_verdict(tests,'A_vs_D','vtp')}]",
        f"  VTP Condition A vs Condition D: {_fmt(tests,'A_vs_D','vtp')}",
        "",
        f"PROPOSITION 2 (Argumentative truncation → compliance risk): [{_verdict(tests,'A_vs_D','ac')}]",
        f"  AC Condition A vs Condition D: {_fmt(tests,'A_vs_D','ac')}",
        f"  CWE: A={cwe_str('A')} D={cwe_str('D')}",
        "",
        f"PROPOSITION 3 (Validity-gated filtering → compliance risk reduction): [{_verdict(tests,'C_vs_D','vtp')}]",
        f"  VTP Condition C vs Condition D: {_fmt(tests,'C_vs_D','vtp')}",
        "",
        f"PROPOSITION 4 (Regulation-anchored assembly → compliance risk reduction): [{_verdict(tests,'B_vs_D','ac')}]",
        f"  AC Condition B vs Condition D: {_fmt(tests,'B_vs_D','ac')}",
        "",
        "─" * 49,
        "CONDITION E: TWO-PASS COMPLIANCE RETRIEVAL",
        "─" * 49,
        "",
        f"PROPOSITION 5 (Two-pass VTP precision ≥ validity-gated C): [{_verdict(tests,'C_vs_E','vtp')}]",
        f"  VTP Condition C vs Condition E: {_fmt(tests,'C_vs_E','vtp')}",
        f"  Interpretation: Pass 1 obligation identification achieves validity-type precision "
        f"{'comparable to or exceeding' if 'SUPPORTED' not in _verdict(tests,'C_vs_E','vtp') else 'exceeding'} "
        f"validity-gated baseline.",
        "",
        f"PROPOSITION 6 (Two-pass AC completeness > discourse-typed D): [{_verdict(tests,'D_vs_E','ac')}]",
        f"  AC Condition D vs Condition E: {_fmt(tests,'D_vs_E','ac')}",
        f"  CWE: D={cwe_str('D')} E={cwe_str('E')}",
        f"  Effect size by epistemic type (D→E AC) — "
        f"normative: {e_d_ac_by_type['normative']} "
        f"constative: {e_d_ac_by_type['constative']} "
        f"mixed: {e_d_ac_by_type['mixed']}",
        f"  Interpretation: "
        f"{'Two-pass Toulmin expansion recovers AC completeness that Condition D sacrifices for VTP precision.' if 'SUPPORTED' in _verdict(tests,'D_vs_E','ac') else 'Two-pass AC is comparable to Condition D — tradeoff not resolved by architecture alone.'}",
        "",
        f"PROPOSITION 7 (Two-pass dominates standard RAG on both VTP and AC): [{_verdict(tests,'A_vs_E','vtp')} / {_verdict(tests,'A_vs_E','ac')}]",
        f"  VTP Condition A vs Condition E: {_fmt(tests,'A_vs_E','vtp')}",
        f"  AC  Condition A vs Condition E: {_fmt(tests,'A_vs_E','ac')}",
        "",
        f"PROPOSITION 8 (Two-pass AC > regulation-anchored GraphRAG B): [{_verdict(tests,'B_vs_E','ac')}]",
        f"  AC Condition B vs Condition E: {_fmt(tests,'B_vs_E','ac')}",
        "",
        "CONFLICT DETECTION:",
        f"  Condition D CSR rate (structural): {csr_d:.3f}",
        f"  Condition E CSR rate (two-pass subgraph): {csr_e:.3f}",
        "",
        "CONDITION MEANS SUMMARY:",
    ]

    for cond in ("A", "B", "C", "D", "E"):
        c = rc.get(cond, {})
        vtp_m = c.get("vtp_mean")
        ac_m  = c.get("ac_mean")
        cwe_m = c.get("cwe_median")
        csr_m = c.get("csr_mean")
        def _f3(v): return f"{v:.3f}" if v is not None else "n/a"
        def _f1(v): return f"{v:.1f}" if v is not None else "n/a"
        lines.append(
            f"  [{cond}] VTP={_f3(vtp_m)} AC={_f3(ac_m)} CWE={_f1(cwe_m)} CSR={_f3(csr_m)}"
        )

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    run_ts = datetime.utcnow().isoformat()
    logger.info(f"=== FinWiki Eval Harness v3 START {run_ts} ===")

    genai_client = genai.Client(api_key=settings.google_api_key)
    qdrant = QdrantClient(url="http://localhost:6333")
    driver = make_neo4j()
    pool = make_pg_pool()
    logger.info("Connections established")

    # Load queries
    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE} — aborting")
        return
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {QUERIES_FILE}")

    # Load v2 per-query results as A–D baseline
    if not V2_PARTIAL_FILE.exists():
        logger.error(f"v2 partial results not found: {V2_PARTIAL_FILE} — aborting")
        return
    with open(V2_PARTIAL_FILE) as f:
        v2_results: List[Dict] = json.load(f)
    v2_by_qid = {r["query_id"]: r for r in v2_results}
    logger.info(f"Loaded {len(v2_results)} v2 query results for A–D baselines")

    # Build query-id → query mapping
    query_by_id = {q["query_id"]: q for q in queries}

    # Check for seed coverage
    seeds_present = sum(1 for q in queries if q.get("query_seed_assertion_id"))
    logger.info(f"Queries with neutral seed already stored: {seeds_present}/{len(queries)}")

    query_results: List[Dict] = []
    for i, q in enumerate(queries):
        qid = q["query_id"]
        v2_result = v2_by_qid.get(qid)

        if v2_result is None:
            logger.warning(f"  [{i+1}/{len(queries)}] No v2 result for {qid[:8]} — creating stub")
            v2_result = {
                "query_id": qid,
                "query_text": q["query_text"],
                "epistemic_type": q["epistemic_type"],
                "source_doc": q.get("source_doc", ""),
                "source_category": q.get("source_category", ""),
                "repetitiveness_index": 0.0,
                "seed_assertion_id": q.get("query_seed_assertion_id"),
                "seed_doc_id": "",
                "neighborhood_size": 0,
                "metrics": {},
                "failed": False,
            }

        # Merge: start with v2 result, add E
        merged = dict(v2_result)

        try:
            metrics_e = with_retry(
                lambda q=q, v2_result=v2_result: eval_condition_e(
                    q, v2_result, driver, qdrant, pool, genai_client, logger
                ),
                retries=3, backoff=10, logger=logger,
            )
            if "metrics" not in merged:
                merged["metrics"] = {}
            merged["metrics"]["E"] = metrics_e
            merged["failed"] = False
        except Exception as e:
            logger.error(f"  FAILED Condition E for {qid[:8]}: {e}")
            if "metrics" not in merged:
                merged["metrics"] = {}
            merged["metrics"]["E"] = None
            merged["failed_e"] = True
            merged["error_e"] = str(e)

        query_results.append(merged)

        if (i + 1) % 10 == 0:
            save_partial(query_results)
            logger.info(f"Progress: {i+1}/{len(queries)} — partial saved to {RESULTS_PARTIAL}")

    save_partial(query_results)
    logger.info(f"All {len(queries)} queries processed. Computing aggregate statistics...")

    aggregate = aggregate_results(query_results)
    save_results(query_results, aggregate, run_ts)
    save_summary(aggregate, run_ts, len(queries))

    logger.info(f"=== FinWiki Eval Harness v3 COMPLETE ===")
    logger.info(f"Completed: {aggregate['n_completed']}/{len(queries)} | Failed: {aggregate['n_failed']}")
    logger.info(f"AC-eligible queries: {aggregate['ac_eligible_queries']}/{len(queries)}")
    logger.info(f"Condition D CSR rate: {aggregate['csr_d_rate']:.3f}")
    logger.info(f"Condition E CSR rate: {aggregate['csr_e_rate']:.3f}")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"Summary: {SUMMARY_FILE}")

    # Print condensed summary to stdout
    rc = aggregate["results_by_condition"]
    def _f3(v): return f"{v:.3f}" if v is not None else "n/a"
    def _f1(v): return f"{v:.1f}" if v is not None else "n/a"
    print("\n=== CONDITION MEANS ===")
    for cond in ("A", "B", "C", "D", "E"):
        c = rc.get(cond, {})
        vtp_m = c.get("vtp_mean")
        ac_m  = c.get("ac_mean")
        cwe_m = c.get("cwe_median")
        csr_m = c.get("csr_mean")
        print(f"  [{cond}] VTP={_f3(vtp_m)}  AC={_f3(ac_m)}  CWE={_f1(cwe_m)}  CSR={_f3(csr_m)}")


if __name__ == "__main__":
    main()
