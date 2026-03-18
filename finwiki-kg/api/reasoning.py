"""api/reasoning.py — /query and /prove reasoning endpoints."""
import logging
from typing import List, Optional

import google.genai as genai
from fastapi import APIRouter, HTTPException
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from api.models import (
    AssertionResponse, CausalContextItem, ProofChainResponse,
    ProveRequest, ProveResponse, QueryRequest, QueryResponse,
    ReasoningTraceItem, RegulationAnchoredAssertion, RegulationAnchoredContext,
)
from api.search import embed_query, get_assertions_for_chunks
from pipeline.config import settings
from pipeline.db import db_cursor
from pipeline.inference import InferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter()


def _neo4j():
    return GraphDatabase.driver(
        settings.neo4j_url,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _assertion_from_node(node) -> AssertionResponse:
    return AssertionResponse(
        assertion_id    = node.get("assertion_id", ""),
        claim_text      = node.get("claim_text", ""),
        subject         = node.get("subject", ""),
        predicate_type  = node.get("predicate_type", ""),
        object_text     = node.get("object_text", ""),
        source_document = node.get("source_document", ""),
        source_url      = node.get("source_url", ""),
        epistemic_status = node.get("epistemic_status", "authoritative"),
        confidence      = float(node.get("confidence", 1.0)),
        domain          = node.get("domain", ""),
        derivation_chain = [],
        derivation_depth = int(node.get("derivation_depth", 0)),
    )


def regulation_anchored_context(
    seed_assertions: List[AssertionResponse],
    top_k_peripheral: int = 5,
) -> Optional[RegulationAnchoredContext]:
    """
    Given seed assertions from Qdrant, find Regulation node anchors via REFERENCES
    edges and traverse inverse REFERENCES to assemble a cross-document Toulmin
    context window partitioned by discourse_role.

    Returns a RegulationAnchoredContext for the regulation with the highest
    peripheral assertion count, or None if no Regulation anchor is found.
    """
    if not seed_assertions:
        return None

    try:
        driver = _neo4j()
        with driver.session() as session:

            # Step 1: For each seed, discover which Regulation nodes it references
            # and count how many peripheral assertions that regulation anchors.
            # candidates: {reg_name -> {"source_doc": str, "count": int}}
            candidates: dict = {}

            for seed in seed_assertions:
                reg_result = session.run(
                    """
                    MATCH (a:Assertion {assertion_id: $id})-[:REFERENCES]->(r:Regulation)
                    RETURN r.name AS reg_name
                    """,
                    id=seed.assertion_id,
                )
                for row in reg_result:
                    reg_name = row["reg_name"]
                    if reg_name in candidates:
                        continue
                    # Count peripheral assertions (from other docs) co-citing this regulation
                    cnt_result = session.run(
                        """
                        MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                              <-[:REFERENCES]-(periph:Assertion)
                        WHERE a.source_document = $src
                          AND periph.source_document <> $src
                          AND NOT (periph.epistemic_status IN ['deprecated', 'orphaned'])
                        RETURN count(DISTINCT periph) AS cnt
                        """,
                        reg=reg_name,
                        src=seed.source_document,
                    )
                    cnt_row = cnt_result.single()
                    candidates[reg_name] = {
                        "source_doc": seed.source_document,
                        "count": cnt_row["cnt"] if cnt_row else 0,
                    }

            if not candidates:
                return None

            # Step 2: Select the regulation with the most peripheral assertions
            best_reg = max(candidates, key=lambda k: candidates[k]["count"])

            if candidates[best_reg]["count"] == 0:
                return None

            # Determine authoritative source_doc: the document with the most
            # REFERENCES edges to this regulation (highest-assertion-count doc),
            # not the first seed's document — this ensures the regulation's own
            # canonical document acts as the warrant source regardless of which
            # peripheral document the Qdrant seed came from.
            authority_result = list(session.run(
                """
                MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                WHERE NOT (a.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN a.source_document AS src_doc, count(a) AS cnt
                ORDER BY cnt DESC
                LIMIT 1
                """,
                reg=best_reg,
            ))
            source_doc = (
                authority_result[0]["src_doc"]
                if authority_result
                else candidates[best_reg]["source_doc"]
            )

            # Step 3: Warrant layer — from source_doc, discourse_role='warrant'.
            # If none exist, fall back to high-confidence normative assertions from
            # source_doc that reference this regulation (stage 3 often tags
            # self-referential regulatory text as 'ground' rather than 'warrant').
            warrant_rows = list(session.run(
                """
                MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                WHERE a.source_document = $src
                  AND a.discourse_role = 'warrant'
                  AND NOT (a.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN a.assertion_id AS id, a.claim_text AS text,
                       a.source_document AS src_doc, a.discourse_role AS role,
                       a.validity_claim_type AS vtype,
                       toFloat(a.confidence) AS conf
                ORDER BY conf DESC
                """,
                reg=best_reg, src=source_doc,
            ))
            if not warrant_rows:
                # Fallback: normative or high-confidence ground from authoritative doc
                warrant_rows = list(session.run(
                    """
                    MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                    WHERE a.source_document = $src
                      AND NOT (a.epistemic_status IN ['deprecated', 'orphaned'])
                    RETURN a.assertion_id AS id, a.claim_text AS text,
                           a.source_document AS src_doc, a.discourse_role AS role,
                           a.validity_claim_type AS vtype,
                           toFloat(a.confidence) AS conf
                    ORDER BY conf DESC
                    """,
                    reg=best_reg, src=source_doc,
                ))

            # Step 4: Ground layer — from peripheral docs, discourse_role='ground'
            ground_rows = list(session.run(
                """
                MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                WHERE a.source_document <> $src
                  AND a.discourse_role = 'ground'
                  AND NOT (a.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN a.assertion_id AS id, a.claim_text AS text,
                       a.source_document AS src_doc, a.discourse_role AS role,
                       a.validity_claim_type AS vtype,
                       toFloat(a.confidence) AS conf
                ORDER BY conf DESC
                """,
                reg=best_reg, src=source_doc,
            ))

            # Step 5: Backing layer — from peripheral docs, discourse_role='backing'
            backing_rows = list(session.run(
                """
                MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg})
                WHERE a.source_document <> $src
                  AND a.discourse_role = 'backing'
                  AND NOT (a.epistemic_status IN ['deprecated', 'orphaned'])
                RETURN a.assertion_id AS id, a.claim_text AS text,
                       a.source_document AS src_doc, a.discourse_role AS role,
                       a.validity_claim_type AS vtype,
                       toFloat(a.confidence) AS conf
                ORDER BY conf DESC
                """,
                reg=best_reg, src=source_doc,
            ))

        driver.close()

        def _to_anchored(row, path: str) -> RegulationAnchoredAssertion:
            return RegulationAnchoredAssertion(
                assertion_id=row["id"],
                claim_text=row["text"] if isinstance(row["text"], str) else (row["text"][0] if row["text"] else ""),
                source_document=row["src_doc"],
                discourse_role=row["role"] or "unclassified",
                validity_claim_type=row["vtype"] or "unclassified",
                confidence=float(row["conf"] or 0.8),
                retrieval_path=path,
            )

        warrant_layer  = [_to_anchored(r, "regulation_anchor") for r in warrant_rows[:top_k_peripheral]]
        ground_layer   = [_to_anchored(r, "regulation_anchor") for r in ground_rows[:top_k_peripheral]]
        backing_layer  = [_to_anchored(r, "regulation_anchor") for r in backing_rows[:top_k_peripheral]]

        all_docs = list({
            a.source_document
            for layer in (warrant_layer, ground_layer, backing_layer)
            for a in layer
        })

        return RegulationAnchoredContext(
            regulation_name=best_reg,
            warrant_layer=warrant_layer,
            ground_layer=ground_layer,
            backing_layer=backing_layer,
            source_documents=sorted(all_docs),
        )

    except Exception as e:
        logger.warning(f"Regulation anchor traversal failed: {e}")
        return None


def two_pass_compliance_retrieval(
    query_text: str,
    vector: List[float],
    driver,
    qdrant,
    pool,
    epistemic_type: str = "normative",
    k: int = 50,
    confidence_threshold: float = 0.85,
) -> dict:
    """
    Two-pass compliance retrieval architecture.

    Pass 1 — Obligation Identification (validity-gated):
      For normative queries: filter to normative assertions, retrieve top-k,
      run regulation-anchored assembly. Returns obligation_set and anchor
      (highest-confidence normative assertion).
      For constative queries: filter to constative assertions.
      For mixed queries: run Pass 1 twice (normative + constative) and merge.

    Pass 2 — Obligation Grounding (type-relaxed Toulmin expansion):
      Takes the anchor from Pass 1. Removes validity filter entirely.
      Traverses ENTAILS, CAUSES, TRIGGERS, SPECIALIZES edges up to 2 hops
      from the anchor regardless of validity type. Adds cross-document grounds
      via Regulation node traversal. Detects CONTRADICTS edges in the assembled
      subgraph. Partitions context into warrant / ground / backing / rebuttal /
      conflict layers.

    Returns a structured dict matching the design spec output shape. All
    assertion fields are plain dicts compatible with the eval harness.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # ── helpers ──────────────────────────────────────────────────────────────
    def _qdrant_top_assertions(vec, n, validity_filter=None):
        qf = None
        if validity_filter:
            qf = Filter(must=[FieldCondition(
                key="validity_claim_type",
                match=MatchValue(value=validity_filter),
            )])
        try:
            res = qdrant.query_points(
                collection_name="finwiki_assertions",
                query=vec, limit=n, with_payload=True, query_filter=qf,
            )
            return [h.payload for h in res.points if h.payload]
        except Exception:
            return []

    def _fetch_by_ids(ids):
        if not ids:
            return []
        import psycopg2
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(ids))
                cur.execute(f"""
                    SELECT assertion_id, claim_text, subject, predicate_type,
                           object_text, source_document, source_url,
                           epistemic_status, confidence, domain,
                           validity_claim_type, discourse_role
                    FROM assertions
                    WHERE assertion_id IN ({placeholders})
                      AND epistemic_status NOT IN ('deprecated', 'orphaned')
                    ORDER BY confidence DESC
                """, ids)
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)
        def _row(r):
            return {
                "assertion_id": r[0], "claim_text": r[1], "subject": r[2] or "",
                "predicate_type": r[3] or "", "object_text": r[4] or "",
                "source_document": r[5] or "", "source_url": r[6] or "",
                "epistemic_status": r[7] or "authoritative",
                "confidence": float(r[8] or 0.8), "domain": r[9] or "",
                "validity_claim_type": r[10] or "unclassified",
                "discourse_role": r[11] or "unclassified",
            }
        order = {aid: i for i, aid in enumerate(ids)}
        result = [_row(r) for r in rows]
        result.sort(key=lambda a: order.get(a["assertion_id"], 999))
        return result

    def _enrich(payloads):
        ids = [p["assertion_id"] for p in payloads if "assertion_id" in p]
        return _fetch_by_ids(ids)

    def _toulmin_expand_from(anchor_id, hops=2):
        try:
            with driver.session() as s:
                res = s.run(f"""
                    MATCH path = (a:Assertion {{assertion_id: $id}})
                        -[:ENTAILS|CAUSES|TRIGGERS|SPECIALIZES*1..{hops}]-
                        (b:Assertion)
                    WHERE NOT (b.epistemic_status IN ['deprecated', 'orphaned'])
                    RETURN DISTINCT b.assertion_id AS nid
                """, id=anchor_id)
                return [row["nid"] for row in res]
        except Exception:
            return []

    def _regulation_periphery(seed_assertions):
        """Cross-document grounds via Regulation node traversal."""
        if not seed_assertions:
            return [], None
        try:
            with driver.session() as s:
                # Find best regulation anchor
                candidates = {}
                for a in seed_assertions[:5]:
                    rows = list(s.run(
                        "MATCH (a:Assertion {assertion_id: $id})-[:REFERENCES]->(r:Regulation) "
                        "RETURN r.name AS reg", id=a["assertion_id"]
                    ))
                    for row in rows:
                        reg = row["reg"]
                        if reg not in candidates:
                            cnt = s.run(
                                "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                                "WHERE NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                                "RETURN count(a) AS cnt", reg=reg
                            ).single()
                            candidates[reg] = cnt["cnt"] if cnt else 0
                if not candidates:
                    return [], None
                best_reg = max(candidates, key=lambda r: candidates[r])
                auth = s.run(
                    "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                    "WHERE NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                    "RETURN a.source_document AS src, count(a) AS cnt "
                    "ORDER BY cnt DESC LIMIT 1", reg=best_reg
                ).single()
                if not auth:
                    return [], best_reg
                auth_doc = auth["src"]
                periph = list(s.run(
                    "MATCH (a:Assertion)-[:REFERENCES]->(r:Regulation {name: $reg}) "
                    "WHERE a.source_document <> $src "
                    "  AND NOT (a.epistemic_status IN ['deprecated','orphaned']) "
                    "RETURN a.assertion_id AS id "
                    "ORDER BY a.confidence DESC LIMIT 15",
                    reg=best_reg, src=auth_doc
                ))
                return [row["id"] for row in periph], best_reg
        except Exception:
            return [], None

    def _detect_contradicts(ids):
        if len(ids) < 2:
            return []
        try:
            with driver.session() as s:
                res = s.run(
                    "UNWIND $ids AS sid "
                    "MATCH (a:Assertion {assertion_id: sid})-[r:CONTRADICTS]->(b:Assertion) "
                    "WHERE b.assertion_id IN $ids "
                    "RETURN a.assertion_id AS src, b.assertion_id AS tgt",
                    ids=list(ids)
                )
                return [(row["src"], row["tgt"]) for row in res]
        except Exception:
            return []

    # ── Pass 1: Obligation Identification ─────────────────────────────────────
    pass1_assertions = []

    if epistemic_type == "mixed":
        norm_p  = _qdrant_top_assertions(vector, k, validity_filter="normative")
        const_p = _qdrant_top_assertions(vector, k, validity_filter="constative")
        norm_a  = _enrich(norm_p)
        const_a = _enrich(const_p)
        seen: set = set()
        for a in norm_a + const_a:
            if a["assertion_id"] not in seen:
                pass1_assertions.append(a)
                seen.add(a["assertion_id"])
        pass1_assertions.sort(key=lambda a: a["confidence"], reverse=True)
        pass1_assertions = pass1_assertions[:k]
    else:
        vtype = "normative" if epistemic_type == "normative" else "constative"
        payloads = _qdrant_top_assertions(vector, k, validity_filter=vtype)
        pass1_assertions = _enrich(payloads)[:k]

    # Identify anchor: highest-confidence normative assertion meeting threshold
    # For constative queries, anchor is the highest-confidence constative
    anchor = None
    for a in pass1_assertions:
        if epistemic_type == "normative" and a["validity_claim_type"] == "normative":
            anchor = a
            break
        elif epistemic_type == "constative" and a["validity_claim_type"] == "constative":
            anchor = a
            break
        elif epistemic_type == "mixed":
            anchor = a  # highest-confidence from merged set
            break
    # Fallback: any top assertion
    if anchor is None and pass1_assertions:
        anchor = pass1_assertions[0]

    # Pass 1 regulation anchor (warrant layer from authoritative source)
    reg_periph_ids_p1, regulation_anchor_name = _regulation_periphery(pass1_assertions[:5])
    reg_periph_p1 = _fetch_by_ids(reg_periph_ids_p1[:5])

    obligation_set = pass1_assertions[:]
    existing_p1 = {a["assertion_id"] for a in obligation_set}
    for a in reg_periph_p1:
        if a["assertion_id"] not in existing_p1:
            obligation_set.append(a)
            existing_p1.add(a["assertion_id"])

    # ── Pass 2: Obligation Grounding (type-relaxed) ───────────────────────────
    # Graceful degradation: no anchor → return Pass 1 alone with warning
    completeness_warning = False
    pass2_warrant: List[dict] = []
    pass2_ground: List[dict] = []
    pass2_backing: List[dict] = []
    pass2_rebuttal: List[dict] = []
    conflict_pairs: List[tuple] = []
    conflicts_detected = False
    reg_anchor_docs: List[str] = []

    if anchor is None:
        completeness_warning = True
    else:
        # Toulmin expansion from anchor — NO validity filter
        expanded_ids = _toulmin_expand_from(anchor["assertion_id"])
        expanded = _fetch_by_ids(expanded_ids)

        # Completeness warning if no neighborhood
        if not expanded:
            completeness_warning = True

        # Cross-document grounds from Regulation traversal
        # Use anchor + top pass1 as seeds
        all_p2_seeds = [anchor] + [a for a in pass1_assertions if a["assertion_id"] != anchor["assertion_id"]][:4]
        reg_ids_p2, _ = _regulation_periphery(all_p2_seeds)
        reg_periph_p2 = _fetch_by_ids(reg_ids_p2[:10])

        # Collect REBUTS-connected assertions from anchor
        rebuttal_ids: List[str] = []
        try:
            with driver.session() as s:
                res = s.run(
                    "MATCH (a:Assertion {assertion_id: $id})-[:REBUTS]->(b:Assertion) "
                    "WHERE NOT (b.epistemic_status IN ['deprecated','orphaned']) "
                    "RETURN b.assertion_id AS nid LIMIT 10",
                    id=anchor["assertion_id"]
                )
                rebuttal_ids = [row["nid"] for row in res]
        except Exception:
            pass
        rebuttal = _fetch_by_ids(rebuttal_ids)

        # Partition by Toulmin role and validity type
        p2_all_candidates = expanded + reg_periph_p2 + obligation_set
        seen_p2: set = set()
        for a in p2_all_candidates:
            aid = a["assertion_id"]
            if aid in seen_p2:
                continue
            seen_p2.add(aid)
            vtype = a["validity_claim_type"]
            drole = a["discourse_role"]
            if vtype == "normative" or drole == "warrant":
                pass2_warrant.append({**a, "retrieval_path": "pass2_toulmin"})
            elif drole == "backing":
                pass2_backing.append({**a, "retrieval_path": "pass2_regulation_anchor"})
            else:
                # constative + unclassified → ground layer
                pass2_ground.append({**a, "retrieval_path": "pass2_toulmin"})

        for a in rebuttal:
            if a["assertion_id"] not in seen_p2:
                seen_p2.add(a["assertion_id"])
                pass2_rebuttal.append({**a, "retrieval_path": "pass2_rebuttal"})

        # Conflict detection in assembled subgraph
        subgraph_ids = seen_p2
        conflict_pairs = _detect_contradicts(subgraph_ids)
        conflicts_detected = len(conflict_pairs) > 0

        reg_anchor_docs = list({
            a["source_document"]
            for layer in (pass2_warrant, pass2_ground, pass2_backing)
            for a in layer
            if a["source_document"]
        })

    # ── Compliance context summary ────────────────────────────────────────────
    top_normative = next((a for a in obligation_set if a["validity_claim_type"] == "normative"), None)
    top_constative = next((a for a in pass2_ground if a["validity_claim_type"] == "constative"), None)
    top_rebuttal = pass2_rebuttal[0] if pass2_rebuttal else None

    compliance_context = {
        "obligation": top_normative["claim_text"] if top_normative else "",
        "factual_predicate": top_constative["claim_text"] if top_constative else "",
        "inference_warrant": anchor["claim_text"] if anchor else "",
        "exemption_conditions": top_rebuttal["claim_text"] if top_rebuttal else "",
        "conflicts_requiring_adjudication": [
            {"source": s, "target": t} for s, t in conflict_pairs[:5]
        ],
    }

    # All assertions across both passes (for eval metrics)
    all_assertions: List[dict] = []
    seen_all: set = set()
    for a in obligation_set + pass2_warrant + pass2_ground + pass2_backing + pass2_rebuttal:
        if a["assertion_id"] not in seen_all:
            all_assertions.append(a)
            seen_all.add(a["assertion_id"])

    return {
        "query": query_text,
        "pass_1": {
            "obligation_set": [
                {
                    "assertion_id": a["assertion_id"],
                    "text": a["claim_text"],
                    "validity_type": a["validity_claim_type"],
                    "discourse_role": a["discourse_role"],
                    "confidence": a["confidence"],
                    "source_doc": a["source_document"],
                    "retrieval_path": "validity_gated",
                }
                for a in pass1_assertions
            ],
            "anchor_assertion_id": anchor["assertion_id"] if anchor else None,
            "regulation_anchor": regulation_anchor_name,
        },
        "pass_2": {
            "warrant_layer": pass2_warrant,
            "ground_layer": pass2_ground,
            "backing_layer": pass2_backing,
            "rebuttal_layer": pass2_rebuttal,
            "conflict_layer": {
                "conflicts_detected": conflicts_detected,
                "conflict_pairs": [
                    {
                        "assertion_1": s,
                        "assertion_2": t,
                        "contradiction_type": "scope_conflict",
                        "requires_adjudication": True,
                    }
                    for s, t in conflict_pairs[:5]
                ],
            },
        },
        "compliance_context": compliance_context,
        "metadata": {
            "pass_1_assertions_retrieved": len(pass1_assertions),
            "pass_2_assertions_retrieved": len(all_assertions) - len(pass1_assertions),
            "total_context_assertions": len(all_assertions),
            "neighborhood_size": len(expanded_ids) if anchor and not completeness_warning else 0,
            "cross_document_sources": reg_anchor_docs,
            "conflict_detected": conflicts_detected,
            "completeness_warning": completeness_warning,
        },
        # Flat list of all assertions for eval harness metric computation
        "_pass1_assertions": pass1_assertions,
        "_all_assertions": all_assertions,
        "_conflicts_detected": conflicts_detected,
        "_conflict_pairs": conflict_pairs,
    }


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Graph-grounded reasoning query with backward chaining.

    1. Embed question
    2. Qdrant top-10 chunks
    3. Retrieve assertions from PostgreSQL
    4. Neo4j traversal: truth-preserving edges up to max_hops
    5. Collect causal context (CAUSES/INHIBITS/CORRELATES_WITH)
    6. Check consistency
    7. Gemini Pro generates answer
    8. Return answer + full reasoning trace
    """
    genai_client = genai.Client(api_key=settings.google_api_key)

    # 1-2: Embed + Qdrant
    vector = embed_query(request.question)
    qdrant  = QdrantClient(url=settings.qdrant_url)
    hits    = qdrant.query_points(collection_name="finwiki_chunks", query=vector, limit=10, with_payload=True).points
    chunk_ids = [h.payload.get("chunk_id", "") for h in hits if h.payload]

    # 3: Retrieve assertions
    assertions = get_assertions_for_chunks(chunk_ids, request.min_confidence, request.domain)
    if not request.include_contested:
        assertions = [a for a in assertions if a.epistemic_status != "contested"]
    if not request.include_derived:
        assertions = [a for a in assertions if a.epistemic_status != "derived"]

    contested_warnings = [
        f"Warning: assertion {a.assertion_id[:8]} is contested"
        for a in assertions if a.epistemic_status == "contested"
    ]

    # 3b: Regulation-anchored cross-document context (second retrieval path)
    reg_context = regulation_anchored_context(assertions, top_k_peripheral=5)

    # 4-5: Neo4j traversal
    reasoning_trace: List[ReasoningTraceItem] = []
    causal_context:  List[CausalContextItem]  = []
    consistency_status = "consistent"

    try:
        driver = _neo4j()
        with driver.session() as session:
            for a in assertions[:5]:
                # Truth-preserving traversal
                result = session.run(
                    f"""
                    MATCH path = (a:Assertion {{assertion_id: $id}})
                        -[:ENTAILS|DEFINES|TRIGGERS|SPECIALIZES|SUPERSEDES*1..{request.max_hops}]->
                        (b:Assertion)
                    WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
                      AND ALL(n IN nodes(path) WHERE NOT (n.epistemic_status IN ['deprecated','orphaned']))
                    RETURN b, length(path) AS hops,
                           reduce(c=1.0, r IN relationships(path) | c * r.confidence * 0.9) AS chain_conf,
                           [r IN relationships(path) | type(r)] AS rel_types
                    ORDER BY chain_conf DESC LIMIT 10
                    """,
                    id=a.assertion_id,
                )
                for record in result:
                    node = record["b"]
                    if not request.include_derived and node.get("epistemic_status") == "derived":
                        continue
                    reasoning_trace.append(ReasoningTraceItem(
                        assertion=_assertion_from_node(node),
                        relation_used=record["rel_types"][-1] if record["rel_types"] else None,
                        hop_distance=record["hops"],
                        chain_confidence=float(record["chain_conf"]),
                    ))

                # Causal context
                causal = session.run(
                    """
                    MATCH (a:Assertion {assertion_id: $id})-[r:CAUSES|INHIBITS|CORRELATES_WITH]->(b:Assertion)
                    WHERE NOT (b.epistemic_status IN ['deprecated','orphaned'])
                    RETURN b, type(r) AS rel_type, r.mechanism AS mech, r.strength AS strength
                    LIMIT 5
                    """,
                    id=a.assertion_id,
                )
                for record in causal:
                    causal_context.append(CausalContextItem(
                        assertion=_assertion_from_node(record["b"]),
                        relation_type=record["rel_type"],
                        mechanism=record.get("mech"),
                        strength=record.get("strength"),
                    ))

    except Exception as e:
        logger.warning(f"Neo4j traversal skipped: {e}")

    # 6: Consistency check — look for CONTRADICTS edges in assertion set
    if len(assertions) >= 2:
        try:
            ids = [a.assertion_id for a in assertions]
            placeholders = ",".join(["%s"] * len(ids))
            with db_cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM assertion_relationships
                    WHERE relationship_type='CONTRADICTS'
                    AND source_assertion_id IN ({placeholders})
                    AND target_assertion_id IN ({placeholders})
                    AND review_status != 'false_positive'
                    """,
                    ids + ids,
                )
                contradictions = cur.fetchone()[0]
            if contradictions > 0:
                consistency_status = "contradictory"
                contested_warnings.append(
                    f"⚠️ {contradictions} contradiction(s) detected among retrieved assertions"
                )
        except Exception as e:
            logger.debug(f"Consistency check failed: {e}")

    # 7: Gemini Pro synthesis
    context_text = "\n\n".join(
        f"[{a.epistemic_status.upper()}] [toulmin_expansion] {a.claim_text} "
        f"(source: {a.source_document}, conf: {a.confidence:.2f})"
        for a in assertions
    )
    trace_text = "\n".join(
        f"  [{t.hop_distance} hops, conf:{t.chain_confidence:.2f}] "
        f"[toulmin_expansion] via {t.relation_used or 'direct'}: {t.assertion.claim_text[:120]}"
        for t in reasoning_trace[:10]
    )

    reg_context_text = ""
    if reg_context:
        sections = []
        if reg_context.warrant_layer:
            warrants = "\n".join(
                f"    [{a.discourse_role.upper()}] [regulation_anchor] {a.claim_text} "
                f"(source: {a.source_document}, conf: {a.confidence:.2f})"
                for a in reg_context.warrant_layer
            )
            sections.append(f"  Warrant layer (authoritative source):\n{warrants}")
        if reg_context.ground_layer:
            grounds = "\n".join(
                f"    [{a.discourse_role.upper()}] [regulation_anchor] {a.claim_text} "
                f"(source: {a.source_document}, conf: {a.confidence:.2f})"
                for a in reg_context.ground_layer
            )
            sections.append(f"  Ground layer (peripheral documents):\n{grounds}")
        if reg_context.backing_layer:
            backings = "\n".join(
                f"    [{a.discourse_role.upper()}] [regulation_anchor] {a.claim_text} "
                f"(source: {a.source_document}, conf: {a.confidence:.2f})"
                for a in reg_context.backing_layer
            )
            sections.append(f"  Backing layer (institutional authority):\n{backings}")
        if sections:
            reg_context_text = (
                f"Cross-document context anchored on [{reg_context.regulation_name}] "
                f"(contributing docs: {', '.join(reg_context.source_documents)}):\n"
                + "\n".join(sections)
            )

    reg_context_block = (
        "Cross-document regulation-anchored context [regulation_anchor]:\n" + reg_context_text
        if reg_context_text else ""
    )
    prompt = f"""You are a financial services knowledge expert. Answer ONLY from the provided assertions.

Question: {request.question}

Directly retrieved assertions [toulmin_expansion]:
{context_text or "(none)"}

Logically derived assertions (truth-preserving inference chains) [toulmin_expansion]:
{trace_text or "(none)"}

{reg_context_block}

Rules:
- Distinguish facts stated in source documents from conclusions derived by logical inference
- Mark derived conclusions explicitly: "By logical inference from [source]: ..."
- [regulation_anchor] assertions are retrieved via shared regulation node traversal
- [toulmin_expansion] assertions are retrieved via vector similarity or logical chaining
- If assertions conflict, state the conflict explicitly
- If evidence is insufficient, say so clearly
- Include source document citations for every factual claim

Answer:"""

    try:
        response = genai_client.models.generate_content(model=settings.pro_model, contents=prompt)
        answer = response.text
        # Track cost
        from pipeline.cost_tracker import tracker
        input_tokens  = int(len(prompt) / 4)
        output_tokens = int(len(answer) / 4)
        try:
            tracker.record(settings.pro_model, input_tokens, output_tokens, "api_query")
        except Exception:
            pass
        total_cost = tracker.total
    except Exception as e:
        logger.error(f"Gemini Pro error: {e}")
        answer     = f"[API error: {e}]"
        total_cost = 0.0

    return QueryResponse(
        answer=answer,
        reasoning_trace=reasoning_trace[:20],
        causal_context=causal_context[:10],
        sources=assertions[:10],
        consistency_status=consistency_status,
        contested_warnings=contested_warnings,
        total_cost_usd=total_cost,
        regulation_context=reg_context,
    )


@router.post("/prove", response_model=ProveResponse)
def prove(request: ProveRequest) -> ProveResponse:
    """Backward chaining: attempt to prove or disprove a specific claim."""
    engine = InferenceEngine()
    result = engine.prove(request.claim, request.max_depth, request.context)

    return ProveResponse(
        target_claim=result.target_claim,
        conclusion=result.conclusion,
        proof_chains=[
            ProofChainResponse(
                assertion_ids=pc.assertion_ids,
                relation_types=pc.relation_types,
                confidences=pc.confidences,
                chain_confidence=pc.chain_confidence,
                conclusion=pc.conclusion,
                hops=pc.hops,
            )
            for pc in result.proof_chains
        ],
        confidence=result.confidence,
        consistency_status=result.consistency_status,
        contested_warnings=result.contested_warnings,
        derived_assertions=[
            AssertionResponse(
                assertion_id    = da.assertion_id,
                claim_text      = da.claim_text,
                subject         = da.subject,
                predicate_type  = da.predicate_type.value,
                object_text     = da.object_text,
                source_document = da.source_document,
                source_url      = da.source_url,
                epistemic_status = da.epistemic_status.value,
                confidence      = da.confidence,
                domain          = da.domain,
                derivation_chain = da.derivation_chain,
            )
            for da in result.derived_assertions
        ],
    )
