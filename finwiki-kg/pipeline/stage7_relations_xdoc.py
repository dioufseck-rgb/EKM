"""
pipeline/stage7_relations_xdoc.py — Cross-document logical relation extraction.

Uses concept-anchored clustering: for each Concept node in Neo4j, retrieve
all assertions that GOVERNS it, then run pairwise relation extraction on
clusters of size 2–50.

Concurrency: 5. Cost-controlled.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Dict

import google.genai as genai
from google.genai import types as genai_types

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker
from pipeline.db import db_cursor, get_neo4j_driver
from pipeline.stage3b_relations import (
    PROMPT_TEMPLATE as RELATION_PROMPT,
    _parse_relations,
    _build_assertions_list,
)

logger = logging.getLogger(__name__)

MIN_CLUSTER = 2
MAX_CLUSTER = 50


def _fetch_concept_clusters() -> Dict[str, List[dict]]:
    """
    Query Neo4j for all Concept nodes and their governing assertions.
    Returns {concept_name: [assertion_dicts]} for clusters of size 2-50.
    """
    driver = get_neo4j_driver()
    clusters: Dict[str, List[dict]] = {}

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)<-[:GOVERNS]-(a:Assertion)
            WHERE NOT (a.epistemic_status IN ['deprecated','orphaned'])
            RETURN c.name AS concept, collect({
                assertion_id: a.assertion_id,
                claim_text: a.claim_text,
                document_id: a.document_id,
                source_document: a.source_document,
                confidence: a.confidence
            }) AS assertions
        """)
        for record in result:
            concept = record["concept"]
            assertions = record["assertions"]
            count = len(assertions)
            if MIN_CLUSTER <= count <= MAX_CLUSTER:
                clusters[concept] = assertions

    logger.info(f"Stage 7: {len(clusters)} concept clusters in range [{MIN_CLUSTER},{MAX_CLUSTER}]")
    return clusters


def _existing_relation_pairs() -> set:
    """Return set of (source_id, target_id) pairs already in logical_relationships."""
    with db_cursor() as cur:
        cur.execute("SELECT source_assertion_id, target_assertion_id FROM logical_relationships")
        return {(r[0], r[1]) for r in cur.fetchall()}


async def process_concept_cluster(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    concept: str,
    assertions: List[dict],
    existing_pairs: set,
    checkpoint: CheckpointManager,
) -> None:
    # Filter to cross-document pairs only
    cross_doc_pairs = []
    for i, a in enumerate(assertions):
        for j, b in enumerate(assertions):
            if i >= j:
                continue
            if a.get("document_id") == b.get("document_id"):
                continue  # same doc — covered by Stage 3b
            pair = (a["assertion_id"], b["assertion_id"])
            rpair = (b["assertion_id"], a["assertion_id"])
            if pair in existing_pairs or rpair in existing_pairs:
                continue  # already extracted
            cross_doc_pairs.append((a, b))

    if not cross_doc_pairs:
        checkpoint.mark_done(concept)
        return

    # Build prompt with all assertions in cluster
    title = f"Cross-document concept cluster: {concept}"
    assertions_list = _build_assertions_list(assertions)
    prompt = RELATION_PROMPT.format(title=title, assertions_list=assertions_list)

    async with semaphore:
        loop = asyncio.get_event_loop()
        for attempt in range(3):
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=settings.flash_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    ),
                )
                raw = resp.text
                input_tokens  = int(len(prompt) / 4)
                output_tokens = int(len(raw) / 4)
                tracker.record(
                    model=settings.flash_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stage="stage7_relations_xdoc",
                    record_id=concept,
                )
                relations = _parse_relations(raw, concept)
                break
            except json.JSONDecodeError as e:
                import time as _t
                await asyncio.sleep(2 ** attempt)
                relations = []
            except CostLimitReached:
                raise
            except Exception as e:
                logger.error(f"LLM error concept={concept}: {e}")
                relations = []
                break

    # Filter: only cross-document relations not already in DB
    doc_map = {a["assertion_id"]: a.get("document_id") for a in assertions}
    new_relations = []
    for r in relations:
        src_doc = doc_map.get(r.source_assertion_id)
        tgt_doc = doc_map.get(r.target_assertion_id)
        if src_doc == tgt_doc:
            continue
        pair  = (r.source_assertion_id, r.target_assertion_id)
        rpair = (r.target_assertion_id, r.source_assertion_id)
        if pair in existing_pairs or rpair in existing_pairs:
            continue
        new_relations.append(r)
        existing_pairs.add(pair)

    # Persist to PostgreSQL
    if new_relations:
        rows = [
            (
                r.relationship_id,
                r.source_assertion_id, r.target_assertion_id,
                r.relation_type.value,
                r.is_bidirectional, r.is_truth_preserving, r.is_defeasible,
                r.evidence_type, r.evidence_text, r.logical_form,
                r.mechanism, r.strength, r.directionality,
                json.dumps({}),  # scope
                r.confidence, "llm_cross_doc", 0, r.review_status.value,
            )
            for r in new_relations
        ]
        with db_cursor() as cur:
            import psycopg2.extras
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO logical_relationships (
                    relationship_id, source_assertion_id, target_assertion_id,
                    relation_type, is_bidirectional, is_truth_preserving, is_defeasible,
                    evidence_type, evidence_text, logical_form, mechanism, strength,
                    directionality, scope, confidence, extraction_method, derivation_depth, review_status
                ) VALUES %s ON CONFLICT DO NOTHING
                """,
                rows,
            )

    checkpoint.mark_done(concept)
    logger.info(f"Stage 7: {concept} → {len(new_relations)} new cross-doc relations")


async def _run_async() -> None:
    client = genai.Client(api_key=settings.google_api_key)

    checkpoint = CheckpointManager("stage7_relations_xdoc")
    already_done = checkpoint.get_completed_ids()

    clusters = _fetch_concept_clusters()
    work_queue = {k: v for k, v in clusters.items() if k not in already_done}
    checkpoint.set_total(len(clusters))

    logger.info(f"Stage 7: {len(work_queue)} clusters to process ({len(already_done)} done)")

    existing_pairs = _existing_relation_pairs()
    semaphore = asyncio.Semaphore(settings.pipeline_concurrency)

    try:
        tasks = [
            process_concept_cluster(semaphore, client, concept, assertions, existing_pairs, checkpoint)
            for concept, assertions in work_queue.items()
        ]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 7: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
