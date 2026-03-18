"""
pipeline/stage6_conflicts.py — Three-stage conflict detection funnel.

6a: Vector candidates (Qdrant similarity >= 0.82, top-5 per chunk)
6b: Structural pre-filter (~70% reduction)
6c: LLM adjudication (Gemini Flash, concurrency 5)
"""
import asyncio
import json
import logging
import re
import uuid
from typing import List, Tuple

import google.genai as genai
from google.genai import types as genai_types
import psycopg2.extras

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker
from pipeline.db import db_cursor, get_qdrant_client

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82
TOP_K = 5
NEGATION_RE = re.compile(r"\b(not|no|prohibit\w*|forbid\w*)\b", re.IGNORECASE)

ADJUDICATION_PROMPT = """\
Compare these two assertions for logical conflict.

Assertion A (ID: {id_a}):
Text: {text_a}
Claim: {claim_a}
Validity type: {validity_type_a}
Scope: {scope_a}

Assertion B (ID: {id_b}):
Text: {text_b}
Claim: {claim_b}
Validity type: {validity_type_b}
Scope: {scope_b}

Follow this reasoning sequence EXACTLY:

Step 0 — Classify validity claim types:
  If either validity_type is "expressive":
    → relationship_type = FALSE_POSITIVE
    → reason = "expressive assertions do not conflict logically"
    → STOP. Return immediately without further analysis.
  If one validity_type is "constative" AND the other is "normative":
    → relationship_type = POTENTIAL_GROUND
    → reason = "constative may be ground for normative, not a conflict"
    → STOP. Return immediately without further analysis.
  If both are "constative" OR both are "normative": proceed to Step 1.

Step 1: Extract the propositional CLAIM of each assertion (not the text — the logical claim)
Step 2: Run SCOPE INTERSECTION TEST — do their validity envelopes overlap temporally, geographically, and organizationally?
Step 3: If scopes do NOT overlap → output SPECIALIZES or FALSE_POSITIVE
Step 4: If scopes DO overlap → determine if the claims are irreconcilable
Step 5: Classify: CONTRADICTS | SUPERSEDES | SPECIALIZES | DUPLICATE | COMPLEMENTARY | FALSE_POSITIVE
Step 6: For CONTRADICTS and SPECIALIZES: provide explanation, conflicting_text, governing_assertion_id (which should govern), reviewer_question, conflict_category (logical|normative|factual)
Step 7: confidence: 0.0-1.0

CRITICAL: The scope intersection test is MANDATORY.
Two assertions about the same topic but in different temporal/geographic/org scopes are SPECIALIZES, NOT CONTRADICTS.
Example: "offices open at 8" (default) vs "in winter offices open at 9" (seasonal override) = SPECIALIZES.

Return JSON only. No markdown. No preamble:
{{"relationship_type": "CONTRADICTS",
  "explanation": "...",
  "conflicting_text": "...",
  "governing_assertion_id": "{id_a}",
  "reviewer_question": "Which rule applies when?",
  "conflict_category": "normative",
  "confidence": 0.85,
  "scope_overlap": {{"temporal": "overlaps", "geographic": "overlaps", "organizational": "overlaps"}}}}
"""

PRIORITY_MAP = {
    "CONTRADICTS":    1,
    "SUPERSEDES":     2,
    "SPECIALIZES":    3,
    "DUPLICATE":      4,
    "COMPLEMENTARY":  5,
    "FALSE_POSITIVE": 5,
    "POTENTIAL_GROUND": 5,
}

# ─── Validity-type Step 0 (testable pure function) ────────────────────────────

def apply_validity_step0(validity_type_a: str, validity_type_b: str):
    """
    Habermas validity-type pre-filter.
    Returns a relationship_type string to short-circuit adjudication, or None
    to proceed to Steps 1–7.
    """
    vta = (validity_type_a or "unclassified").lower()
    vtb = (validity_type_b or "unclassified").lower()
    if "expressive" in (vta, vtb):
        return "FALSE_POSITIVE"
    if {vta, vtb} == {"constative", "normative"}:
        return "POTENTIAL_GROUND"
    return None


# ─── 6a: Vector candidates ────────────────────────────────────────────────────

def _assertions_for_chunk(chunk_id: str) -> List[str]:
    """Return assertion_ids for a given chunk_id."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT assertion_id FROM assertions WHERE chunk_id = %s"
            " AND epistemic_status NOT IN ('deprecated','orphaned') LIMIT 5",
            [chunk_id],
        )
        return [r[0] for r in cur.fetchall()]


def build_vector_candidates() -> int:
    """Query Qdrant for top-5 neighbours per chunk at cosine >= 0.82.
    Resolves chunk IDs to assertion IDs before storing candidates."""
    qdrant = get_qdrant_client()
    logger.info("Stage 6a: building vector candidates")

    # Scroll through all Qdrant points
    offset = None
    all_points = []
    while True:
        results, offset = qdrant.scroll(
            collection_name="finwiki_chunks",
            with_vectors=True,
            limit=100,
            offset=offset,
        )
        all_points.extend(results)
        if offset is None:
            break

    logger.info(f"Stage 6a: {len(all_points)} vectors loaded")

    inserted = 0
    for point in all_points:
        if point.vector is None:
            continue
        chunk_id_a  = point.payload.get("chunk_id", "")
        doc_id_a    = point.payload.get("document_id", "")
        assertion_ids_a = _assertions_for_chunk(chunk_id_a)
        if not assertion_ids_a:
            continue

        result = qdrant.query_points(
            collection_name="finwiki_chunks",
            query=point.vector,
            limit=TOP_K + 1,  # +1 because self is included
            score_threshold=SIMILARITY_THRESHOLD,
            with_payload=True,
        )
        for nb in result.points:
            chunk_id_b = nb.payload.get("chunk_id", "")
            doc_id_b   = nb.payload.get("document_id", "")
            if chunk_id_b == chunk_id_a:
                continue
            if doc_id_a == doc_id_b:
                continue  # same document — skip
            assertion_ids_b = _assertions_for_chunk(chunk_id_b)
            if not assertion_ids_b:
                continue
            # Store one candidate pair per assertion combination (up to first 3 each)
            for aid_a in assertion_ids_a[:3]:
                for aid_b in assertion_ids_b[:3]:
                    a, b = sorted([aid_a, aid_b])
                    with db_cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO conflict_candidates (candidate_id, assertion_id_a, assertion_id_b, similarity_score)
                            VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
                            """,
                            (str(uuid.uuid4()), a, b, nb.score),
                        )
                        inserted += cur.rowcount

    logger.info(f"Stage 6a: {inserted} candidate pairs inserted")
    return inserted


# ─── 6b: Structural pre-filter ────────────────────────────────────────────────

def fetch_assertion(assertion_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT assertion_id, claim_text, subject, predicate_type, object_value,
                   source_text, source_document, domain, effective_date, expiry_date,
                   epistemic_status, confidence, scope_coverage,
                   temporal_season, temporal_is_default,
                   geo_is_global, geo_countries,
                   org_is_universal, validity_claim_type
            FROM assertions WHERE assertion_id = %s
            """,
            [assertion_id],
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "assertion_id":      row[0], "claim_text": row[1], "subject": row[2],
        "predicate_type":    row[3], "object_value": row[4],
        "source_text":       row[5][:1000] if row[5] else "",
        "source_document":   row[6], "domain": row[7],
        "effective_date":    row[8], "expiry_date": row[9],
        "epistemic_status":  row[10], "confidence": row[11],
        "scope_coverage":    row[12],
        "temporal_season":   row[13], "temporal_is_default": row[14],
        "geo_is_global":     row[15], "geo_countries": row[16] or [],
        "org_is_universal":  row[17],
        "validity_claim_type": row[18] or "unclassified",
    }


def passes_structural_filter(a: dict, b: dict) -> bool:
    """Return True if the pair should proceed to LLM adjudication."""
    # Same domain, different source document
    if a.get("domain") == b.get("domain") and a.get("source_document") != b.get("source_document"):
        return True
    # Same subject with different object values
    if (a.get("subject") and a["subject"] == b.get("subject") and
            a.get("object_value") is not None and b.get("object_value") is not None and
            a["object_value"] != b["object_value"]):
        return True
    # Negation pattern in one vs other
    text_a = a.get("source_text", "") + " " + a.get("claim_text", "")
    text_b = b.get("source_text", "") + " " + b.get("claim_text", "")
    if bool(NEGATION_RE.search(text_a)) != bool(NEGATION_RE.search(text_b)):
        return True
    # requires vs prohibits on same subject
    if (a.get("subject") == b.get("subject") and
            {a.get("predicate_type"), b.get("predicate_type")} == {"requires", "prohibits"}):
        return True
    return False


def filter_candidates() -> List[Tuple[str, str, float]]:
    """Load candidates from DB, apply structural filter, return survivors."""
    with db_cursor() as cur:
        cur.execute("SELECT assertion_id_a, assertion_id_b, similarity_score FROM conflict_candidates WHERE adjudicated = false")
        rows = cur.fetchall()

    survivors = []
    for id_a, id_b, score in rows:
        a = fetch_assertion(id_a)
        b = fetch_assertion(id_b)
        if a and b and passes_structural_filter(a, b):
            survivors.append((id_a, id_b, score))

    logger.info(f"Stage 6b: {len(survivors)}/{len(rows)} pairs pass structural filter")
    return survivors


# ─── 6c: LLM adjudication ────────────────────────────────────────────────────

def _scope_summary(a: dict) -> str:
    parts = []
    if not a.get("temporal_is_default", True):
        parts.append(f"season={a.get('temporal_season')}")
    if not a.get("geo_is_global", True):
        parts.append(f"countries={a.get('geo_countries')}")
    if not a.get("org_is_universal", True):
        parts.append("org-scoped")
    return ", ".join(parts) if parts else "universal"


async def adjudicate_pair(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    id_a: str,
    id_b: str,
    checkpoint: CheckpointManager,
) -> None:
    pair_key = f"{min(id_a,id_b)}:{max(id_a,id_b)}"

    a = fetch_assertion(id_a)
    b = fetch_assertion(id_b)
    if not a or not b:
        checkpoint.mark_done(pair_key)
        return

    prompt = ADJUDICATION_PROMPT.format(
        id_a=id_a, text_a=a["source_text"][:1000], claim_a=a["claim_text"],
        validity_type_a=a.get("validity_claim_type", "unclassified"),
        scope_a=_scope_summary(a),
        id_b=id_b, text_b=b["source_text"][:1000], claim_b=b["claim_text"],
        validity_type_b=b.get("validity_claim_type", "unclassified"),
        scope_b=_scope_summary(b),
    )

    async with semaphore:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=settings.flash_model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                raw = resp.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                result = json.loads(raw)
                if isinstance(result, list):
                    result = result[0] if result else None

                input_tokens  = int(len(prompt) / 4)
                output_tokens = int(len(raw) / 4)
                tracker.record(
                    model=settings.flash_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stage="stage6_conflicts",
                    record_id=pair_key,
                )
                break
            except json.JSONDecodeError as e:
                import time as _t
                await asyncio.sleep(2 ** attempt)
                result = None
            except CostLimitReached:
                raise
            except Exception as e:
                logger.error(f"LLM error pair={pair_key}: {e}")
                result = None
                break

    if not result:
        checkpoint.mark_done(pair_key)
        return

    rel_type = result.get("relationship_type", "FALSE_POSITIVE")
    if rel_type in ("FALSE_POSITIVE", "COMPLEMENTARY", "POTENTIAL_GROUND"):
        # Mark adjudicated, no conflict item
        with db_cursor() as cur:
            cur.execute(
                "UPDATE conflict_candidates SET adjudicated=true WHERE assertion_id_a=%s AND assertion_id_b=%s",
                [min(id_a,id_b), max(id_a,id_b)],
            )
        checkpoint.mark_done(pair_key)
        return

    # Insert assertion_relationship
    rel_id = str(uuid.uuid4())
    scope_overlap = result.get("scope_overlap", {})
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO assertion_relationships
                (relationship_id, source_assertion_id, target_assertion_id, relationship_type,
                 explanation, conflicting_text, governing_assertion_id, reviewer_question,
                 confidence, scope_overlap, review_status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            (
                rel_id, id_a, id_b, rel_type,
                result.get("explanation", ""),
                result.get("conflicting_text", ""),
                result.get("governing_assertion_id"),
                result.get("reviewer_question", ""),
                float(result.get("confidence", 0.8)),
                json.dumps(scope_overlap),
                "pending",
            ),
        )
        # Insert conflict item
        priority = PRIORITY_MAP.get(rel_type, 5)
        cur.execute(
            "INSERT INTO conflict_items (conflict_id, relationship_id, priority) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (str(uuid.uuid4()), rel_id, priority),
        )
        # Mark candidate adjudicated
        cur.execute(
            "UPDATE conflict_candidates SET adjudicated=true WHERE assertion_id_a=%s AND assertion_id_b=%s",
            [min(id_a,id_b), max(id_a,id_b)],
        )

    checkpoint.mark_done(pair_key)
    logger.debug(f"Adjudicated: {pair_key} → {rel_type}")


async def _run_async() -> None:
    client = genai.Client(api_key=settings.google_api_key)

    checkpoint = CheckpointManager("stage6_conflicts")
    already_done = checkpoint.get_completed_ids()

    # 6a
    build_vector_candidates()

    # 6b
    survivors = filter_candidates()
    work_queue = [
        (id_a, id_b, score) for id_a, id_b, score in survivors
        if f"{min(id_a,id_b)}:{max(id_a,id_b)}" not in already_done
    ]
    checkpoint.set_total(len(survivors))

    logger.info(f"Stage 6c: {len(work_queue)} pairs to adjudicate")

    semaphore = asyncio.Semaphore(settings.pipeline_concurrency)

    try:
        tasks = [
            adjudicate_pair(semaphore, client, id_a, id_b, checkpoint)
            for id_a, id_b, _ in work_queue
        ]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 6: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
