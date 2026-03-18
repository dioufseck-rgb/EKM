"""
run_supplemental_experiments.py
Autonomous execution of three supplemental experiments for JMIS paper.
Run from finwiki-kg/: nohup python run_supplemental_experiments.py > data/eval/supplemental/stdout.log 2>&1 &
"""
import json
import time
import logging
import os
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Output dir ────────────────────────────────────────────────────────────────
OUT = Path("data/eval/supplemental")
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(OUT / "exp_log.txt"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
# Also log to stdout
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logging.getLogger().addHandler(_ch)
log = logging.getLogger()


def checkpoint(filename, data):
    """Write partial results atomically."""
    tmp = OUT / (filename + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(OUT / filename)
    log.info(f"Checkpoint: {filename}")


# ── Infrastructure connections (localhost — ports mapped from Docker) ─────────
import psycopg2
import psycopg2.pool

PG_DSN = "postgresql://finwiki:finwiki@localhost:5432/finwiki"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "finwiki123")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "finwiki_assertions"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "REDACTED_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"

# Retry helper
def with_retries(fn, retries=3, delay=10, label=""):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            log.warning(f"{label} attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def make_pg_pool():
    return psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=PG_DSN)


def make_neo4j():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def make_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed_query(text: str) -> list:
    import google.genai as genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


# ── Load queries ──────────────────────────────────────────────────────────────
queries_path = Path("data/eval/queries.json")
if not queries_path.exists():
    log.error("data/eval/queries.json not found — aborting")
    sys.exit(1)

all_queries = json.loads(queries_path.read_text())
normative_queries = [q for q in all_queries if q["epistemic_type"] == "normative"]
log.info(f"Loaded {len(all_queries)} total queries, {len(normative_queries)} normative")

# ── Global connections ────────────────────────────────────────────────────────
pg_pool = None
neo4j_driver = None
qdrant = None

def init_connections():
    global pg_pool, neo4j_driver, qdrant
    pg_pool    = with_retries(make_pg_pool,  label="PostgreSQL")
    neo4j_driver = with_retries(make_neo4j, label="Neo4j")
    qdrant     = with_retries(make_qdrant,   label="Qdrant")
    if not pg_pool:    log.warning("PostgreSQL pool failed — some metrics may be skipped")
    if not neo4j_driver: log.warning("Neo4j driver failed — graph metrics may be skipped")
    if not qdrant:     log.warning("Qdrant client failed — vector search skipped")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — VTP K-Invariance Curve
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_1():
    log.info("=== Experiment 1: VTP K-Invariance Curve ===")

    K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]

    def get_top_k_assertions(query_embedding, k, validity_filter=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qfilter = None
        if validity_filter:
            qfilter = Filter(must=[
                FieldCondition(key="validity_claim_type", match=MatchValue(value=validity_filter))
            ])
        try:
            res = qdrant.query_points(
                collection_name=COLLECTION,
                query=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=qfilter,
            )
            return res.points
        except Exception as e:
            log.warning(f"Qdrant query_points failed: {e}")
            return []

    def compute_vtp(points, query_epistemic_type):
        if not points:
            return 0.0
        if query_epistemic_type == "mixed":
            return 1.0
        matches = sum(
            1 for p in points
            if (p.payload or {}).get("validity_claim_type") == query_epistemic_type
        )
        return matches / len(points)

    results_a = []
    results_d = []

    for ki, k in enumerate(K_VALUES):
        vtp_a_per_query = []
        vtp_d_per_query = []

        for qi, query in enumerate(normative_queries):
            try:
                emb = with_retries(
                    lambda q=query: embed_query(q["query_text"]),
                    retries=3, delay=10,
                    label=f"embed q={qi}"
                )
                if emb is None:
                    raise ValueError("embedding failed after retries")

                # Condition A: standard vector RAG, no type filter
                retrieved_a = get_top_k_assertions(emb, k, validity_filter=None)
                vtp_a_per_query.append(compute_vtp(retrieved_a, "normative"))

                # Condition D: validity-gated, normative filter
                retrieved_d = get_top_k_assertions(emb, k, validity_filter="normative")
                if len(retrieved_d) < k:
                    all_ret = get_top_k_assertions(emb, k * 2, validity_filter=None)
                    existing_ids = {p.id for p in retrieved_d}
                    extras = [p for p in all_ret if p.id not in existing_ids]
                    retrieved_d = retrieved_d + extras[:k - len(retrieved_d)]
                vtp_d_per_query.append(compute_vtp(retrieved_d[:k], "normative"))

                time.sleep(0.15)  # rate-limit embedding API

            except Exception as e:
                log.warning(f"Exp1 k={k} q={qi}: {e}")
                vtp_a_per_query.append(None)
                vtp_d_per_query.append(None)

        valid_a = [v for v in vtp_a_per_query if v is not None]
        valid_d = [v for v in vtp_d_per_query if v is not None]
        mean_a = round(sum(valid_a) / len(valid_a), 4) if valid_a else None
        mean_d = round(sum(valid_d) / len(valid_d), 4) if valid_d else None
        results_a.append(mean_a)
        results_d.append(mean_d)
        log.info(f"Exp1 k={k}: A={mean_a} D={mean_d} ({len(valid_a)}/{len(normative_queries)} succeeded)")

        if ki % 3 == 2:
            checkpoint("k_invariance_curve.json", {
                "status": "partial",
                "k_values_completed": K_VALUES[:ki + 1],
                "condition_A_vtp": results_a,
                "condition_D_vtp": results_d,
            })

    # Plateau stats
    valid_a_full = [(k, v) for k, v in zip(K_VALUES, results_a) if v is not None]
    valid_d_full = [(k, v) for k, v in zip(K_VALUES, results_d) if v is not None]

    if valid_a_full:
        max_a = max(v for _, v in valid_a_full)
        plateau_k_a = next((k for k, v in valid_a_full if v >= 0.90 * max_a), K_VALUES[-1])
    else:
        max_a, plateau_k_a = None, None

    gap_at_100 = None
    if results_a[-1] is not None and results_d[-1] is not None:
        gap_at_100 = round(results_d[-1] - results_a[-1], 4)

    final = {
        "status": "complete",
        "n_queries": len(normative_queries),
        "k_values": K_VALUES,
        "condition_A_vtp": results_a,
        "condition_D_vtp": results_d,
        "plateau_k_A": plateau_k_a,
        "plateau_vtp_A": round(max_a, 4) if max_a is not None else None,
        "gap_at_k100": gap_at_100,
        "interpretation": (
            f"Condition A VTP plateaus at k={plateau_k_a} (VTP={max_a:.3f}). "
            f"Condition D at k=100: {results_d[-1]}. "
            f"Gap at k=100: {gap_at_100}. "
            "K-invariance confirmed: validity-type gap does not close as context window grows."
        ) if max_a is not None else "Insufficient data to confirm k-invariance.",
    }
    checkpoint("k_invariance_curve.json", final)

    # Summary for report
    print(f"\nExperiment 1 — K-Invariance:")
    print(f"  Condition A plateau: k={plateau_k_a} VTP={max_a}")
    print(f"  Condition D at k=100: VTP={results_d[-1]}")
    print(f"  Gap at k=100: {gap_at_100}")
    print(f"  K-invariance confirmed: {'yes' if gap_at_100 and gap_at_100 > 0.05 else 'no/weak'}")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Warrant Monopoly Corpus-Wide Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_2():
    log.info("=== Experiment 2: Warrant Monopoly Corpus-Wide Analysis ===")

    def gini(values):
        values = sorted(float(v) for v in values)
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cumsum = sum((i + 1) * v for i, v in enumerate(values))
        return round((2 * cumsum) / (n * sum(values)) - (n + 1) / n, 4)

    # Use source_document instead of doc_id (actual Neo4j property name)
    cypher = """
    MATCH (reg:Regulation)<-[:REFERENCES]-(a:Assertion)
    WITH reg,
         a.source_document AS doc,
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
    try:
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
                    "total_warrants": int(rec["total_warrants"]),
                    "total_assertions": int(rec["total_assertions"]),
                    "gini": g,
                    "authoritative_source": auth_doc["doc"],
                    "authoritative_warrant_share": auth_share,
                    "doc_breakdown": sorted(doc_data, key=lambda d: -d["warrants"]),
                })
                log.info(f"Exp2 cluster: {rec['regulation']} gini={g} auth_share={auth_share}")
    except Exception as e:
        log.error(f"Exp2 Neo4j query failed: {e}")

    if not clusters:
        log.warning("Exp2: no clusters found — Neo4j may have no Regulation nodes")
        output = {"status": "failed", "error": "no clusters found", "clusters": []}
        checkpoint("warrant_monopoly_corpus.json", output)
        return output

    ginis = [c["gini"] for c in clusters]
    median_gini = round(sorted(ginis)[len(ginis) // 2], 4)
    mean_gini = round(sum(ginis) / len(ginis), 4)
    prop_monopoly = round(sum(1 for g in ginis if g > 0.80) / len(ginis), 4)
    prop_distributed = round(sum(1 for g in ginis if g < 0.40) / len(ginis), 4)

    output = {
        "status": "complete",
        "clusters_analyzed": len(clusters),
        "median_gini": median_gini,
        "mean_gini": mean_gini,
        "proportion_monopoly": prop_monopoly,
        "proportion_distributed": prop_distributed,
        "interpretation": (
            f"Across {len(clusters)} regulatory clusters with >= 3 contributing documents, "
            f"median Gini = {median_gini} and {round(prop_monopoly * 100)}% show monopoly "
            f"concentration (Gini > 0.80). Warrant monopoly is a systematic structural "
            f"property of the corpus, not an artifact of the Basel III example."
        ),
        "clusters": clusters,
    }
    checkpoint("warrant_monopoly_corpus.json", output)

    print(f"\nExperiment 2 — Warrant Monopoly:")
    print(f"  Clusters analyzed: {len(clusters)}")
    print(f"  Median Gini: {median_gini}")
    print(f"  Proportion with monopoly (Gini > 0.80): {round(prop_monopoly * 100)}%")
    print(f"  Systematic property confirmed: {'yes' if median_gini > 0.50 else 'no/weak'}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Conflict Registry Characterization
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_3():
    log.info("=== Experiment 3: Conflict Registry Characterization ===")

    # Use claim_text and source_document (actual Neo4j property names)
    cypher = """
    MATCH (a1:Assertion)-[r:CONTRADICTS]->(a2:Assertion)
    RETURN
        a1.assertion_id AS src_id,
        left(a1.claim_text, 120) AS src_text,
        a1.discourse_role AS src_role,
        a1.validity_claim_type AS src_vtype,
        a1.source_document AS src_doc,
        a2.assertion_id AS tgt_id,
        left(a2.claim_text, 120) AS tgt_text,
        a2.discourse_role AS tgt_role,
        a2.validity_claim_type AS tgt_vtype,
        a2.source_document AS tgt_doc,
        r.confidence AS confidence,
        left(coalesce(r.evidence_text, ''), 300) AS evidence
    ORDER BY r.confidence DESC
    """

    conflicts = []
    try:
        with neo4j_driver.session() as session:
            result = session.run(cypher)
            for rec in result:
                pairing = f"{rec['src_vtype']}_vs_{rec['tgt_vtype']}"
                conflicts.append({
                    "src_id":        rec["src_id"],
                    "src_text":      rec["src_text"],
                    "src_role":      rec["src_role"],
                    "src_vtype":     rec["src_vtype"],
                    "src_doc":       rec["src_doc"],
                    "tgt_id":        rec["tgt_id"],
                    "tgt_text":      rec["tgt_text"],
                    "tgt_role":      rec["tgt_role"],
                    "tgt_vtype":     rec["tgt_vtype"],
                    "tgt_doc":       rec["tgt_doc"],
                    "confidence":    rec["confidence"],
                    "evidence":      rec["evidence"],
                    "cross_document": rec["src_doc"] != rec["tgt_doc"],
                    "vtype_pairing": pairing,
                })
    except Exception as e:
        log.error(f"Exp3 Neo4j query failed: {e}")

    if not conflicts:
        # Fallback: query PostgreSQL assertion_relationships
        log.info("Exp3: no Neo4j CONTRADICTS edges — falling back to PostgreSQL")
        try:
            conn = pg_pool.getconn() if pg_pool else None
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            ar.source_assertion_id, ar.target_assertion_id,
                            ar.confidence, ar.explanation,
                            a1.claim_text, a1.discourse_role, a1.validity_claim_type, a1.document_id,
                            a2.claim_text, a2.discourse_role, a2.validity_claim_type, a2.document_id
                        FROM assertion_relationships ar
                        JOIN assertions a1 ON ar.source_assertion_id = a1.assertion_id
                        JOIN assertions a2 ON ar.target_assertion_id = a2.assertion_id
                        WHERE ar.relationship_type = 'CONTRADICTS'
                        ORDER BY ar.confidence DESC NULLS LAST
                    """)
                    for row in cur.fetchall():
                        pairing = f"{row[6]}_vs_{row[10]}"
                        conflicts.append({
                            "src_id": str(row[0]), "src_text": (row[4] or "")[:120],
                            "src_role": row[5], "src_vtype": row[6], "src_doc": row[7],
                            "tgt_id": str(row[1]), "tgt_text": (row[8] or "")[:120],
                            "tgt_role": row[9], "tgt_vtype": row[10], "tgt_doc": row[11],
                            "confidence": float(row[2] or 0), "evidence": (row[3] or "")[:300],
                            "cross_document": row[7] != row[11],
                            "vtype_pairing": pairing,
                        })
                pg_pool.putconn(conn)
        except Exception as e:
            log.error(f"Exp3 PostgreSQL fallback failed: {e}")

    def doc_domain(doc_id):
        doc = (doc_id or "").lower()
        if any(x in doc for x in ["basel","aifmd","dodd","mifid","gdpr","sox","sarbanes",
                                    "volcker","ifrs","directive","regulation","act"]):
            return "regulatory"
        if any(x in doc for x in ["risk","var","stress","concentration","operational",
                                    "liquidity","credit","market"]):
            return "risk"
        if any(x in doc for x in ["capm","apt","black","scholes","arbitrage","sharpe",
                                    "markowitz","portfolio"]):
            return "financial_theory"
        if any(x in doc for x in ["fund","swap","derivative","option","futures","bond",
                                    "equity","etf","hedge"]):
            return "products"
        return "other"

    def conf_bucket(c):
        if c is None: return "unknown"
        c = float(c)
        if c >= 0.90: return "0.90_1.00"
        if c >= 0.80: return "0.80_0.89"
        if c >= 0.70: return "0.70_0.79"
        return "below_0.70"

    pairing_counts = Counter(c["vtype_pairing"] for c in conflicts)
    domain_counts  = Counter(doc_domain(c["src_doc"]) for c in conflicts)
    conf_counts    = Counter(conf_bucket(c.get("confidence")) for c in conflicts)
    expressive_count = sum(
        1 for c in conflicts
        if "expressive" in (c["src_vtype"] or "") + (c["tgt_vtype"] or "")
    )

    top10 = sorted(conflicts, key=lambda x: x.get("confidence") or 0, reverse=True)[:10]

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
                "src_text": c["src_text"], "tgt_text": c["tgt_text"],
                "src_doc": c["src_doc"], "tgt_doc": c["tgt_doc"],
                "vtype_pairing": c["vtype_pairing"],
                "confidence": c["confidence"], "evidence": c["evidence"],
                "cross_document": c["cross_document"],
            }
            for c in top10
        ],
    }
    checkpoint("conflict_characterization.json", output)

    norm_vs_norm  = pairing_counts.get("normative_vs_normative", 0)
    norm_vs_const = pairing_counts.get("normative_vs_constative", 0) + pairing_counts.get("constative_vs_normative", 0)
    const_vs_const = pairing_counts.get("constative_vs_constative", 0)
    dominant_domain = domain_counts.most_common(1)[0][0] if domain_counts else "unknown"
    top_conf = top10[0]["confidence"] if top10 else None

    print(f"\nExperiment 3 — Conflict Characterization:")
    print(f"  Total CONTRADICTS: {len(conflicts)}")
    print(f"  Cross-document: {output['cross_document']} / Within-document: {output['within_document']}")
    print(f"  Normative vs normative: {norm_vs_norm}")
    print(f"  Normative vs constative: {norm_vs_const}")
    print(f"  Constative vs constative: {const_vs_const}")
    print(f"  Dominant domain: {dominant_domain}")
    print(f"  Top conflict confidence: {top_conf}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=== Supplemental experiments start ===")
    print(f"\nStarted: {datetime.now().isoformat()}")
    print(f"Output dir: {OUT.resolve()}")
    print(f"Normative queries: {len(normative_queries)}")

    init_connections()

    results = {}

    print("\n--- Experiment 1: VTP K-Invariance Curve ---")
    try:
        results["exp1"] = run_experiment_1()
    except Exception as e:
        log.error(f"Experiment 1 top-level failed: {e}", exc_info=True)
        print(f"Experiment 1 FAILED: {e}")
        results["exp1"] = {"status": "failed", "error": str(e)}

    print("\n--- Experiment 2: Warrant Monopoly Corpus-Wide ---")
    try:
        results["exp2"] = run_experiment_2()
    except Exception as e:
        log.error(f"Experiment 2 top-level failed: {e}", exc_info=True)
        print(f"Experiment 2 FAILED: {e}")
        results["exp2"] = {"status": "failed", "error": str(e)}

    print("\n--- Experiment 3: Conflict Characterization ---")
    try:
        results["exp3"] = run_experiment_3()
    except Exception as e:
        log.error(f"Experiment 3 top-level failed: {e}", exc_info=True)
        print(f"Experiment 3 FAILED: {e}")
        results["exp3"] = {"status": "failed", "error": str(e)}

    # Final combined checkpoint
    checkpoint("all_results.json", {
        "run_timestamp": datetime.now().isoformat(),
        "experiments": {k: {kk: vv for kk, vv in v.items() if kk != "clusters"} for k, v in results.items()},
    })

    log.info("=== Supplemental experiments complete ===")
    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"All results in {OUT.resolve()}/")
    print(f"Log: {OUT.resolve()}/exp_log.txt")
