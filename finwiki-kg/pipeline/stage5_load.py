"""
pipeline/stage5_load.py — Load all pipeline data into PostgreSQL and Qdrant.

No LLM. Free stage.
Loads: documents, chunks, assertions, logical_relationships → PostgreSQL
       embeddings → Qdrant + qdrant_id_map → PostgreSQL
"""
import json
import logging
import os
from typing import List

import numpy as np
import psycopg2.extras
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.db import db_cursor, get_qdrant_client

logger = logging.getLogger(__name__)

POSTGRES_BATCH = 1000
QDRANT_BATCH   = 100


# ─── PostgreSQL loaders ───────────────────────────────────────────────────────

def load_document(article: dict, filename: str) -> str:
    """Insert document row. Returns document_id (= article title slug)."""
    doc_id = article.get("title", filename).replace(" ", "_")
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (document_id, title, url, domain, subdomain, authority_level, word_count, crawled_at, raw_file_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            ON CONFLICT DO NOTHING
            """,
            (
                doc_id,
                article.get("title", ""),
                article.get("url", ""),
                article.get("domain", "finance"),
                "",
                "reference",
                article.get("word_count", 0),
                os.path.join(settings.raw_dir, filename),
            ),
        )
    return doc_id


def load_chunks(chunks: List[dict]) -> None:
    rows = [
        (
            c["chunk_id"], c["document_id"], c["sequence"],
            c.get("section_title", ""), c["content"],
            c.get("token_estimate", 0),
        )
        for c in chunks
    ]
    with db_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (chunk_id, document_id, sequence, section_title, content, token_estimate)
            VALUES %s ON CONFLICT DO NOTHING
            """,
            rows,
        )


def load_assertions(assertions: List[dict], grammar_map: dict = None) -> None:
    if grammar_map is None:
        grammar_map = {}
    rows = []
    for a in assertions:
        scope = a.get("scope", {})
        t     = scope.get("temporal", {})
        g     = scope.get("geographic", {})
        o     = scope.get("organizational", {})
        c     = scope.get("conditional", {})
        # Merge grammar fields (discourse_role, validity_claim_type) from Stage 3c
        grammar = grammar_map.get(a["assertion_id"], {})
        discourse_role      = grammar.get("discourse_role",      a.get("discourse_role",      "unclassified"))
        validity_claim_type = grammar.get("validity_claim_type", a.get("validity_claim_type", "unclassified"))
        rows.append((
            a["assertion_id"], a.get("chunk_id"), a.get("document_id"),
            a.get("claim_text", ""), a.get("subject", ""),
            a.get("predicate_type", ""), a.get("object_text", ""),
            a.get("object_value"), a.get("object_unit"),
            a.get("source_text", ""), a.get("source_document", ""),
            a.get("source_url", ""), a.get("source_section", ""),
            a.get("authority_level", "reference"),
            a.get("epistemic_status", "authoritative"),
            float(a.get("confidence", 0.8)),
            a.get("extraction_method", "llm"),
            a.get("review_status", "pending"),
            json.dumps(a.get("derivation_chain", [])),
            a.get("topics", []), a.get("entities", []),
            a.get("regulations", []), a.get("keywords", []),
            a.get("domain", "finance"),
            # Temporal scope
            t.get("season"), t.get("months", []), t.get("days_of_week", []),
            t.get("is_default", True), t.get("fiscal_period"),
            # Geographic scope
            g.get("countries", []), g.get("states", []), g.get("regions", []),
            g.get("location_types", []), g.get("is_global", True),
            # Org scope
            o.get("roles", []), o.get("business_units", []), o.get("products", []),
            o.get("customer_segments", []), o.get("account_types", []),
            o.get("is_universal", True),
            # Conditional scope
            c.get("conditions", []), json.dumps(c.get("thresholds", {})),
            c.get("prerequisites", []), c.get("trigger_events", []),
            # Scope meta
            scope.get("coverage", "universal"),
            scope.get("completeness", "unknown"),
            scope.get("source", "unknown"),
            # Discourse grammar
            discourse_role,
            validity_claim_type,
        ))

    with db_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO assertions (
                assertion_id, chunk_id, document_id,
                claim_text, subject, predicate_type, object_text, object_value, object_unit,
                source_text, source_document, source_url, source_section, authority_level,
                epistemic_status, confidence, extraction_method, review_status, derivation_chain,
                topics, entities, regulations, keywords, domain,
                temporal_season, temporal_months, temporal_days_of_week, temporal_is_default, temporal_fiscal_period,
                geo_countries, geo_states, geo_regions, geo_location_types, geo_is_global,
                org_roles, org_business_units, org_products, org_customer_segments, org_account_types, org_is_universal,
                cond_conditions, cond_thresholds, cond_prerequisites, cond_trigger_events,
                scope_coverage, scope_completeness, scope_source,
                discourse_role, validity_claim_type
            ) VALUES %s ON CONFLICT DO NOTHING
            """,
            rows,
        )


def load_logical_relationships(relations: List[dict]) -> None:
    if not relations:
        return
    rows = [
        (
            r["relationship_id"],
            r["source_assertion_id"], r["target_assertion_id"],
            r.get("relation_type", ""),
            r.get("is_bidirectional", False),
            r.get("is_truth_preserving", False),
            r.get("is_defeasible", False),
            r.get("evidence_type", "explicit"),
            r.get("evidence_text", ""),
            r.get("logical_form", ""),
            r.get("mechanism", ""),
            r.get("strength", ""),
            r.get("directionality", "A_to_B"),
            json.dumps(r.get("scope", {})),
            float(r.get("confidence", 0.8)),
            r.get("extraction_method", "llm_within_doc"),
            int(r.get("derivation_depth", 0)),
            r.get("review_status", "pending"),
        )
        for r in relations
    ]
    with db_cursor() as cur:
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


# ─── Qdrant loader ────────────────────────────────────────────────────────────

def ensure_qdrant_collection() -> None:
    qdrant = get_qdrant_client()
    collections = [c.name for c in qdrant.get_collections().collections]
    if "finwiki_chunks" not in collections:
        qdrant.create_collection(
            collection_name="finwiki_chunks",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: finwiki_chunks")


def load_embeddings(npz_path: str, chunks: List[dict]) -> None:
    qdrant = get_qdrant_client()

    data = np.load(npz_path, allow_pickle=True)
    chunk_ids = list(data["chunk_ids"])
    embeddings = data["embeddings"]

    # Build chunk metadata lookup
    meta = {c["chunk_id"]: c for c in chunks}

    points = []
    id_map_rows = []
    for i, (cid, vec) in enumerate(zip(chunk_ids, embeddings)):
        qdrant_id = abs(hash(cid)) % (2**31)  # stable integer ID from UUID
        chunk_meta = meta.get(cid, {})
        points.append(PointStruct(
            id=qdrant_id,
            vector=vec.tolist(),
            payload={
                "chunk_id":    cid,
                "document_id": chunk_meta.get("document_id", ""),
                "domain":      chunk_meta.get("domain", "finance"),
                "content":     chunk_meta.get("content", "")[:500],
            },
        ))
        id_map_rows.append((cid, qdrant_id))

    # Upsert in batches of 100
    for i in range(0, len(points), QDRANT_BATCH):
        qdrant.upsert(collection_name="finwiki_chunks", points=points[i:i + QDRANT_BATCH])

    # Store ID map in PostgreSQL
    with db_cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO qdrant_id_map (chunk_id, qdrant_point_id) VALUES %s ON CONFLICT DO NOTHING",
            id_map_rows,
        )


# ─── Stage runner ─────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))

    checkpoint = CheckpointManager("stage5_load")
    already_done = checkpoint.get_completed_ids()

    ensure_qdrant_collection()

    raw_files = [f for f in os.listdir(settings.raw_dir) if f.endswith(".json")]
    work_queue = [f for f in raw_files if f not in already_done]
    checkpoint.set_total(len(raw_files))

    logger.info(f"Stage 5: {len(work_queue)} documents to load ({len(already_done)} done)")

    for filename in work_queue:
        # 1. Load document
        raw_path = os.path.join(settings.raw_dir, filename)
        try:
            with open(raw_path) as f:
                article = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            checkpoint.mark_failed(filename, str(e))
            continue

        doc_id = load_document(article, filename)

        # 2. Load chunks
        chunk_path = os.path.join(settings.chunks_dir, filename)
        chunks = []
        if os.path.exists(chunk_path):
            try:
                with open(chunk_path) as f:
                    chunks = json.load(f)
                load_chunks(chunks)
            except Exception as e:
                logger.warning(f"Chunk load failed {filename}: {e}")

        # 3. Load assertions (merge grammar fields from Stage 3c)
        assertions_path = os.path.join(settings.assertions_dir, f"{doc_id}_assertions.json")
        if os.path.exists(assertions_path):
            try:
                with open(assertions_path) as f:
                    assertions = json.load(f)
                # Load grammar classifications if available
                grammar_map = {}
                grammar_path = os.path.join(settings.grammar_dir, f"{doc_id}_grammar.json")
                if os.path.exists(grammar_path):
                    with open(grammar_path) as f:
                        grammar_map = json.load(f)
                load_assertions(assertions, grammar_map)
            except Exception as e:
                logger.warning(f"Assertions load failed {filename}: {e}")

        # 4. Load logical relationships
        relations_path = os.path.join(settings.relations_dir, f"{doc_id}_relations.json")
        if os.path.exists(relations_path):
            try:
                with open(relations_path) as f:
                    relations = json.load(f)
                load_logical_relationships(relations)
            except Exception as e:
                logger.warning(f"Relations load failed {filename}: {e}")

        # 5. Load embeddings into Qdrant
        base = filename.replace(".json", "")
        npz_path = os.path.join(settings.embeddings_dir, f"{base}.npz")
        if os.path.exists(npz_path) and chunks:
            try:
                load_embeddings(npz_path, chunks)
            except Exception as e:
                logger.warning(f"Embedding load failed {filename}: {e}")

        checkpoint.mark_done(filename)
        logger.info(f"Loaded: {filename}")

    checkpoint.complete()


if __name__ == "__main__":
    run()
