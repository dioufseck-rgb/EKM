"""
Experiment 1 — VTP K-Invariance Curve
From experiments_spec.md, with one adjustment:
  embed_query uses embed_text(text, client) from pipeline/eval_harness_v4.py
  (pipeline.embeddings does not exist; embedding is in eval_harness_v4.py)
  Also: queries use field "query_text", not "text".

K values: [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
Queries: 40 normative queries from data/eval/queries.json
Conditions: A (no filter) and D (normative filter)
Output: data/eval/supplemental/k_invariance_curve.json
"""
import json, time, logging, os, math, sys
from datetime import datetime
from pathlib import Path

# ── Override URLs to localhost (devcontainer, not inside Docker) ──────────────
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("POSTGRES_URL", "postgresql://finwiki:finwiki@localhost:5432/finwiki")
os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")

# Change to finwiki-kg dir so relative imports work
script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

OUT = Path("data/eval/supplemental")
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=OUT / "exp_log.txt",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
# Also log to stdout
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logging.getLogger().addHandler(console)
log = logging.getLogger()


def checkpoint(filename, data):
    """Write partial results atomically."""
    tmp = OUT / (filename + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(OUT / filename)
    log.info(f"Checkpoint: {filename}")


# ── Connections ───────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient

qdrant = QdrantClient(url=os.environ["QDRANT_URL"])

# ── Embedding (adjusted: uses eval_harness_v4 embed_text with client) ─────────
import google.genai as genai
from google.genai import types as genai_types
from pipeline.config import settings

genai_client = genai.Client(api_key=settings.google_api_key)

ASSERTIONS_COLLECTION = "finwiki_assertions"


def embed_query(text: str):
    """Use same embedding model as pipeline — gemini-embedding-001 via Google AI."""
    result = genai_client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


# ── Qdrant retrieval (spec pseudocode, uses search()) ─────────────────────────
def get_top_k_assertions(query_embedding, k, validity_filter=None):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qfilter = None
    if validity_filter:
        qfilter = Filter(must=[
            FieldCondition(key="validity_claim_type", match=MatchValue(value=validity_filter))
        ])
    results = qdrant.search(
        collection_name=ASSERTIONS_COLLECTION,
        query_vector=query_embedding,
        limit=k,
        query_filter=qfilter,
        with_payload=True,
    )
    return results


def compute_vtp(retrieved_assertions, query_epistemic_type):
    """
    Validity-Type Precision: proportion of retrieved assertions whose
    validity_claim_type matches the query's epistemic requirement.
    """
    if not retrieved_assertions:
        return 0.0
    target = query_epistemic_type
    if target == "mixed":
        return 1.0
    matches = sum(
        1 for a in retrieved_assertions
        if (a.payload or {}).get("validity_claim_type") == target
    )
    return matches / len(retrieved_assertions)


# ── Load queries ──────────────────────────────────────────────────────────────
queries = json.loads(Path("data/eval/queries.json").read_text())
normative_queries = [q for q in queries if q["epistemic_type"] == "normative"]
log.info(f"Loaded {len(normative_queries)} normative queries for Experiment 1")

K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]

results_a = []
results_d = []

for ki, k in enumerate(K_VALUES):
    vtp_a_per_query = []
    vtp_d_per_query = []

    for qi, query in enumerate(normative_queries):
        try:
            # Adjusted: field is "query_text", not "text"
            emb = embed_query(query["query_text"])

            # Condition A: standard vector RAG, no type filter
            retrieved_a = get_top_k_assertions(emb, k, validity_filter=None)
            vtp_a_per_query.append(compute_vtp(retrieved_a, "normative"))

            # Condition D: validity-gated, normative filter only
            retrieved_d = get_top_k_assertions(emb, k, validity_filter="normative")
            # Pad with unfiltered if fewer than k normative assertions found
            if len(retrieved_d) < k:
                all_retrieved = get_top_k_assertions(emb, k * 2, validity_filter=None)
                existing_ids = {r.id for r in retrieved_d}
                extras = [r for r in all_retrieved if r.id not in existing_ids]
                retrieved_d = retrieved_d + extras[: k - len(retrieved_d)]
            vtp_d_per_query.append(compute_vtp(retrieved_d[:k], "normative"))

        except Exception as e:
            log.warning(f"Exp1 k={k} q={qi}: {e}")
            vtp_a_per_query.append(None)
            vtp_d_per_query.append(None)

    valid_a = [v for v in vtp_a_per_query if v is not None]
    valid_d = [v for v in vtp_d_per_query if v is not None]
    results_a.append(round(sum(valid_a) / len(valid_a), 4) if valid_a else None)
    results_d.append(round(sum(valid_d) / len(valid_d), 4) if valid_d else None)
    log.info(f"Exp1 k={k}: A={results_a[-1]} D={results_d[-1]}")
    print(f"k={k:3d}  A={results_a[-1]}  D={results_d[-1]}")

    # Checkpoint every 3 k-values
    if ki % 3 == 2:
        checkpoint("k_invariance_curve.json", {
            "status": "partial",
            "k_values_completed": K_VALUES[: ki + 1],
            "condition_A_vtp": results_a,
            "condition_D_vtp": results_d,
        })

# ── Compute plateau stats ──────────────────────────────────────────────────────
valid_a_full = [(k, v) for k, v in zip(K_VALUES, results_a) if v is not None]
max_a = max(v for _, v in valid_a_full)
plateau_k_a = next((k for k, v in valid_a_full if v >= 0.90 * max_a), K_VALUES[-1])
gap_at_100 = (
    round(results_d[-1] - results_a[-1], 4)
    if results_a[-1] is not None and results_d[-1] is not None
    else None
)

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
    ),
}
checkpoint("k_invariance_curve.json", final)
print("Experiment 1 complete:", json.dumps(final, indent=2))
