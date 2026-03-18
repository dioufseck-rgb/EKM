"""
pipeline/stage8_graph.py — Load all data into Neo4j.

No LLM. All Cypher uses MERGE not CREATE — safely re-runnable.
Loads: Document, Chunk, Assertion, Concept, Regulation, Topic nodes,
       all structural + logical + conflict edges.
"""
import json
import logging

import psycopg2.extras

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.db import db_cursor, get_neo4j_driver

logger = logging.getLogger(__name__)

BATCH = 500


def _batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ─── Node loaders ─────────────────────────────────────────────────────────────

def load_documents(session) -> int:
    with db_cursor() as cur:
        cur.execute("SELECT document_id, title, url, domain, subdomain, authority_level, word_count FROM documents")
        rows = cur.fetchall()

    count = 0
    for chunk in _batch(rows, BATCH):
        tx = session.begin_transaction()
        for row in chunk:
            tx.run(
                """
                MERGE (d:Document {document_id: $id})
                SET d.title=$title, d.url=$url, d.domain=$domain,
                    d.subdomain=$sub, d.authority_level=$auth, d.word_count=$wc
                """,
                id=row[0], title=row[1], url=row[2], domain=row[3],
                sub=row[4], auth=row[5], wc=row[6],
            )
        tx.commit()
        count += len(chunk)
    logger.info(f"Neo4j: {count} Document nodes")
    return count


def load_chunks(session) -> int:
    with db_cursor() as cur:
        cur.execute("SELECT chunk_id, document_id, sequence, section_title, token_estimate FROM chunks")
        rows = cur.fetchall()

    count = 0
    for chunk in _batch(rows, BATCH):
        tx = session.begin_transaction()
        for row in chunk:
            tx.run(
                """
                MERGE (c:Chunk {chunk_id: $id})
                SET c.document_id=$doc, c.sequence=$seq, c.section_title=$title, c.token_estimate=$te
                WITH c
                MATCH (d:Document {document_id: $doc})
                MERGE (c)-[:CHUNK_OF]->(d)
                """,
                id=row[0], doc=row[1], seq=row[2], title=row[3], te=row[4],
            )
        tx.commit()
        count += len(chunk)
    logger.info(f"Neo4j: {count} Chunk nodes")
    return count


def load_assertions(session) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT assertion_id, chunk_id, document_id, claim_text, subject, predicate_type,
                   object_text, object_value, domain, confidence, epistemic_status,
                   scope_coverage, scope_completeness, source_document, source_url,
                   topics, entities, regulations,
                   discourse_role, validity_claim_type
            FROM assertions
            """
        )
        rows = cur.fetchall()

    count = 0
    for batch in _batch(rows, BATCH):
        tx = session.begin_transaction()
        for row in batch:
            aid, cid, did, claim, subj, pred, obj, oval, domain, conf, epi, cov, comp, sdoc, surl, topics, entities, regs, dr, vct = row
            tx.run(
                """
                MERGE (a:Assertion {assertion_id: $id})
                SET a.claim_text=$claim, a.subject=$subj, a.predicate_type=$pred,
                    a.object_text=$obj, a.object_value=$oval, a.domain=$domain,
                    a.confidence=$conf, a.epistemic_status=$epi,
                    a.coverage=$cov, a.scope_completeness=$comp,
                    a.source_document=$sdoc, a.source_url=$surl,
                    a.discourse_role=$dr, a.validity_claim_type=$vct,
                    a.document_id=$did, a.derivation_depth=0
                WITH a
                MATCH (c:Chunk {chunk_id: $cid})
                MERGE (a)-[:SOURCED_FROM]->(c)
                """,
                id=aid, claim=claim, subj=subj, pred=pred, obj=obj, oval=oval,
                domain=domain, conf=float(conf or 0.8), epi=epi, cov=cov, comp=comp,
                sdoc=sdoc, surl=surl, cid=cid, did=did,
                dr=dr or "unclassified", vct=vct or "unclassified",
            )
            # Topics
            for topic in (topics or []):
                if topic:
                    tx.run(
                        """
                        MERGE (t:Topic {name: $name})
                        WITH t
                        MATCH (a:Assertion {assertion_id: $aid})
                        MERGE (a)-[:TAGGED_WITH]->(t)
                        """,
                        name=topic, aid=aid,
                    )
            # Concepts (from entities)
            for entity in (entities or []):
                if entity:
                    tx.run(
                        """
                        MERGE (c:Concept {name: $name})
                        WITH c
                        MATCH (a:Assertion {assertion_id: $aid})
                        MERGE (a)-[:GOVERNS]->(c)
                        """,
                        name=entity, aid=aid,
                    )
            # Regulations
            for reg in (regs or []):
                if reg:
                    tx.run(
                        """
                        MERGE (r:Regulation {name: $name})
                        WITH r
                        MATCH (a:Assertion {assertion_id: $aid})
                        MERGE (a)-[:REFERENCES]->(r)
                        """,
                        name=reg, aid=aid,
                    )
        tx.commit()
        count += len(batch)
    logger.info(f"Neo4j: {count} Assertion nodes")
    return count


# ─── Edge loaders ─────────────────────────────────────────────────────────────

def load_logical_edges(session) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT source_assertion_id, target_assertion_id, relation_type,
                   is_truth_preserving, is_defeasible, is_bidirectional,
                   confidence, evidence_text, logical_form,
                   mechanism, strength, directionality,
                   extraction_method, derivation_depth, review_status, scope
            FROM logical_relationships
            """
        )
        rows = cur.fetchall()

    count = 0
    for batch in _batch(rows, BATCH):
        tx = session.begin_transaction()
        for row in batch:
            src, tgt, rtype, tp, def_, bid, conf, evid, form, mech, strength, direc, exm, depth, rstatus, scope = row
            # Dynamic relationship type via apoc or parameterized Cypher
            # Neo4j doesn't support dynamic relationship types without apoc,
            # so we use a conditional approach for the most common types
            props = {
                "is_truth_preserving": tp,
                "is_defeasible": def_,
                "is_bidirectional": bid,
                "confidence": float(conf or 0.8),
                "evidence_text": evid or "",
                "logical_form": form or "",
                "mechanism": mech or "",
                "strength": strength or "",
                "directionality": direc or "A_to_B",
                "extraction_method": exm or "llm",
                "derivation_depth": depth or 0,
                "review_status": rstatus or "pending",
                "scope": json.dumps(scope) if scope else "{}",
            }
            # Use APOC if available; else use parameterized relationship creation
            try:
                tx.run(
                    f"""
                    MATCH (a:Assertion {{assertion_id: $src}})
                    MATCH (b:Assertion {{assertion_id: $tgt}})
                    CALL apoc.merge.relationship(a, $rtype, {{source_assertion_id: $src}}, $props, b)
                    YIELD rel RETURN rel
                    """,
                    src=src, tgt=tgt, rtype=rtype, props=props,
                )
            except Exception:
                # Fallback: use a generic RELATES_TO with type as property
                tx.run(
                    """
                    MATCH (a:Assertion {assertion_id: $src})
                    MATCH (b:Assertion {assertion_id: $tgt})
                    MERGE (a)-[r:LOGICAL_RELATION {relation_type: $rtype, source_assertion_id: $src}]->(b)
                    SET r += $props, r.relation_type = $rtype
                    """,
                    src=src, tgt=tgt, rtype=rtype, props=props,
                )
        tx.commit()
        count += len(batch)
    logger.info(f"Neo4j: {count} logical relationship edges")
    return count


def load_conflict_edges(session) -> int:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT source_assertion_id, target_assertion_id, relationship_type,
                   explanation, confidence, review_status, scope_overlap
            FROM assertion_relationships
            """
        )
        rows = cur.fetchall()

    count = 0
    for batch in _batch(rows, BATCH):
        tx = session.begin_transaction()
        for row in batch:
            src, tgt, rtype, expl, conf, rstatus, scope_overlap = row
            if rtype == "DUPLICATE":
                cypher_type = "DUPLICATE_OF"
            else:
                cypher_type = "CONTRADICTS"  # covers CONTRADICTS, SUPERSEDES, SPECIALIZES
            try:
                tx.run(
                    f"""
                    MATCH (a:Assertion {{assertion_id: $src}})
                    MATCH (b:Assertion {{assertion_id: $tgt}})
                    MERGE (a)-[r:{cypher_type} {{source_assertion_id: $src}}]->(b)
                    SET r.explanation=$expl, r.confidence=$conf,
                        r.review_status=$status, r.relationship_type=$rtype,
                        r.scope_overlap=$scope
                    """,
                    src=src, tgt=tgt, expl=expl or "", conf=float(conf or 0.8),
                    status=rstatus or "pending", rtype=rtype,
                    scope=json.dumps(scope_overlap) if scope_overlap else "{}",
                )
            except Exception as e:
                logger.debug(f"Conflict edge error {src}->{tgt}: {e}")
        tx.commit()
        count += len(batch)
    logger.info(f"Neo4j: {count} conflict edges")
    return count


# ─── Stage runner ─────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))

    checkpoint = CheckpointManager("stage8_graph")

    driver = get_neo4j_driver()

    # Run cypher/schema.cypher constraints first
    try:
        import os
        cypher_path = "cypher/schema.cypher"
        if os.path.exists(cypher_path):
            with open(cypher_path) as f:
                statements = [s.strip() for s in f.read().split(";") if s.strip() and not s.strip().startswith("//")]
            with driver.session() as session:
                for stmt in statements:
                    try:
                        session.run(stmt)
                    except Exception as e:
                        logger.debug(f"Schema stmt skipped: {e}")
    except Exception as e:
        logger.warning(f"Schema.cypher init skipped: {e}")

    with driver.session() as session:
        checkpoint.set_status("running")
        load_documents(session)
        load_chunks(session)
        load_assertions(session)
        load_logical_edges(session)
        load_conflict_edges(session)

    checkpoint.complete()
    logger.info("Stage 8: Neo4j graph loaded")


if __name__ == "__main__":
    run()
