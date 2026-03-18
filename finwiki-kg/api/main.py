"""api/main.py — FastAPI application entry point."""
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AssertionResponse, ConflictCard, ConflictsResponse,
    ConflictUpdateRequest, GraphAssertionResponse, LogicalEdgeResponse,
    NeighborhoodResponse, ScopeOverlapGrid, StatsResponse,
)
from api.search import router as search_router
from api.reasoning import router as reasoning_router
from pipeline.config import settings
from pipeline.db import db_cursor, get_neo4j_driver

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinWiki Knowledge Graph API",
    description="Graph-grounded financial services document intelligence over 2,500 Wikipedia articles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(reasoning_router)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _get_assertion(assertion_id: str) -> AssertionResponse:
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT assertion_id, claim_text, subject, predicate_type, object_text,
                   source_document, source_url, epistemic_status, confidence, domain,
                   derivation_chain
            FROM assertions WHERE assertion_id = %s
            """,
            [assertion_id],
        )
        row = cur.fetchone()
    if not row:
        return AssertionResponse(
            assertion_id=assertion_id, claim_text="(not found)", subject="",
            predicate_type="", object_text="", source_document="", source_url="",
            epistemic_status="", confidence=0.0, domain="",
        )
    return AssertionResponse(
        assertion_id=row[0], claim_text=row[1], subject=row[2],
        predicate_type=row[3], object_text=row[4], source_document=row[5] or "",
        source_url=row[6] or "", epistemic_status=row[7] or "authoritative",
        confidence=float(row[8] or 0.0), domain=row[9] or "",
        derivation_chain=row[10] or [],
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents");           doc_count   = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks");              chunk_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM assertions");          asr_count   = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM logical_relationships"); lr_count  = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM assertion_relationships"); cf_count = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(MAX(running_total_usd),0) FROM llm_cost_log"); total_cost = float(cur.fetchone()[0])
        cur.execute(
            """
            SELECT COUNT(*) FROM conflict_items ci
            JOIN assertion_relationships ar ON ci.relationship_id = ar.relationship_id
            WHERE ar.review_status = 'pending'
            """
        )
        pending = cur.fetchone()[0]
        cur.execute(
            "SELECT stage_name, status, records_total, records_done, last_updated FROM pipeline_checkpoints"
        )
        stages = [
            {"stage": r[0], "status": r[1], "total": r[2], "done": r[3], "updated": str(r[4])}
            for r in cur.fetchall()
        ]

    return StatsResponse(
        pipeline_stages=stages, total_documents=doc_count, total_chunks=chunk_count,
        total_assertions=asr_count, total_logical_relationships=lr_count,
        total_conflicts=cf_count, total_cost_usd=total_cost, pending_conflicts=pending,
    )


@app.get("/conflicts", response_model=ConflictsResponse)
def get_conflicts(
    relationship_type: Optional[str] = None,
    priority:          Optional[int]  = None,
    review_status:     Optional[str]  = None,
    domain:            Optional[str]  = None,
    page:      int = 1,
    page_size: int = 20,
):
    offset = (page - 1) * page_size
    where_clauses, params = [], []

    if relationship_type:
        where_clauses.append("ar.relationship_type = %s");  params.append(relationship_type)
    if priority is not None:
        where_clauses.append("ci.priority = %s");           params.append(priority)
    if review_status:
        where_clauses.append("ar.review_status = %s");      params.append(review_status)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    with db_cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) FROM conflict_items ci
            JOIN assertion_relationships ar ON ci.relationship_id = ar.relationship_id
            {where_sql}
            """,
            params,
        )
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT ci.conflict_id, ci.relationship_id, ar.relationship_type, ci.priority,
                   ar.source_assertion_id, ar.target_assertion_id, ar.explanation,
                   ar.conflicting_text, ar.reviewer_question, ar.scope_overlap,
                   ar.confidence, ar.review_status
            FROM conflict_items ci
            JOIN assertion_relationships ar ON ci.relationship_id = ar.relationship_id
            {where_sql}
            ORDER BY ci.priority ASC, ci.created_at DESC
            LIMIT %s OFFSET %s
            """,
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    items = []
    for row in rows:
        scope_overlap = row[9] or {}
        if isinstance(scope_overlap, str):
            try:
                scope_overlap = json.loads(scope_overlap)
            except Exception:
                scope_overlap = {}
        items.append(ConflictCard(
            conflict_id=row[0], relationship_id=row[1], relationship_type=row[2],
            priority=row[3], assertion_a=_get_assertion(row[4]), assertion_b=_get_assertion(row[5]),
            explanation=row[6] or "", conflicting_text=row[7] or "",
            reviewer_question=row[8] or "",
            scope_overlap=ScopeOverlapGrid(
                temporal=scope_overlap.get("temporal", "unknown"),
                geographic=scope_overlap.get("geographic", "unknown"),
                organizational=scope_overlap.get("organizational", "unknown"),
            ),
            confidence=float(row[10] or 0.8), review_status=row[11] or "pending",
        ))

    return ConflictsResponse(items=items, total=total, page=page, page_size=page_size)


@app.patch("/conflicts/{conflict_id}")
def update_conflict(conflict_id: str, request: ConflictUpdateRequest):
    with db_cursor() as cur:
        cur.execute(
            """
            UPDATE assertion_relationships ar
            SET review_status = %s
            FROM conflict_items ci
            WHERE ci.conflict_id = %s AND ci.relationship_id = ar.relationship_id
            """,
            [request.resolution.value, conflict_id],
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conflict not found")
    return {"status": "updated", "conflict_id": conflict_id, "resolution": request.resolution.value}


@app.get("/graph/assertion/{assertion_id}", response_model=GraphAssertionResponse)
def get_assertion_graph(assertion_id: str):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Assertion {assertion_id: $id})
            OPTIONAL MATCH (a)-[lr]->(b:Assertion)
            WHERE type(lr) IN ['ENTAILS','CAUSES','DEFINES','TRIGGERS','SPECIALIZES',
                               'SUPERSEDES','EQUIVALENT','INHIBITS','CORRELATES_WITH',
                               'GENERALIZES','INSTANTIATES','CLASSIFIES','PRECEDES',
                               'OPERATIONALIZES','LOGICAL_RELATION']
            RETURN a,
                   collect({
                     rel_type: type(lr),
                     target:   b.assertion_id,
                     tp:       lr.is_truth_preserving,
                     def_:     lr.is_defeasible,
                     evid:     lr.evidence_text,
                     mech:     lr.mechanism,
                     strength: lr.strength,
                     conf:     lr.confidence
                   }) AS logical_rels
            """,
            id=assertion_id,
        )
        record = result.single()

    if not record:
        raise HTTPException(status_code=404, detail="Assertion not found")

    node      = record["a"]
    assertion = AssertionResponse(
        assertion_id    = node["assertion_id"],
        claim_text      = node.get("claim_text", ""),
        subject         = node.get("subject", ""),
        predicate_type  = node.get("predicate_type", ""),
        object_text     = node.get("object_text", ""),
        source_document = node.get("source_document", ""),
        source_url      = node.get("source_url", ""),
        epistemic_status = node.get("epistemic_status", "authoritative"),
        confidence      = float(node.get("confidence", 1.0)),
        domain          = node.get("domain", ""),
    )

    logical_rels = [
        LogicalEdgeResponse(
            relation_type        = r.get("rel_type") or r.get("relation_type", ""),
            source_assertion_id  = assertion_id,
            target_assertion_id  = r.get("target", ""),
            is_truth_preserving  = bool(r.get("tp", False)),
            is_defeasible        = bool(r.get("def_", False)),
            evidence_text        = r.get("evid") or "",
            mechanism            = r.get("mech"),
            strength             = r.get("strength"),
            confidence           = float(r.get("conf") or 0.8),
            hop                  = 1,
        )
        for r in record["logical_rels"]
        if r.get("target")
    ]

    return GraphAssertionResponse(
        assertion=assertion,
        logical_relationships=logical_rels,
        conflict_relationships=[],
    )


@app.get("/graph/neighborhood/{assertion_id}", response_model=NeighborhoodResponse)
def get_neighborhood(assertion_id: str, depth: int = 2):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (a:Assertion {{assertion_id: $id}})-[*1..{min(depth,4)}]-(b)
            RETURN nodes(path) AS ns, relationships(path) AS rs
            LIMIT 100
            """,
            id=assertion_id,
        )

        nodes_map: dict = {}
        edges_list: list = []

        for record in result:
            for node in record["ns"]:
                nid = (
                    node.get("assertion_id") or node.get("name") or
                    node.get("chunk_id") or node.get("document_id") or "?"
                )
                label = list(node.labels)[0] if node.labels else "Node"
                nodes_map[nid] = {"id": nid, "label": label, "props": dict(node)}
            for rel in record["rs"]:
                src = (rel.start_node.get("assertion_id") or rel.start_node.get("name") or "?")
                tgt = (rel.end_node.get("assertion_id") or rel.end_node.get("name") or "?")
                edges_list.append({"type": type(rel).__name__, "source": src, "target": tgt})

        # Fetch center node separately
        center_result = session.run(
            "MATCH (a:Assertion {assertion_id: $id}) RETURN a", id=assertion_id
        )
        center_record = center_result.single()
        center_node = center_record["a"] if center_record else {}

    center = AssertionResponse(
        assertion_id    = assertion_id,
        claim_text      = center_node.get("claim_text", ""),
        subject         = center_node.get("subject", ""),
        predicate_type  = center_node.get("predicate_type", ""),
        object_text     = center_node.get("object_text", ""),
        source_document = center_node.get("source_document", ""),
        source_url      = center_node.get("source_url", ""),
        epistemic_status = center_node.get("epistemic_status", "authoritative"),
        confidence      = float(center_node.get("confidence", 1.0)),
        domain          = center_node.get("domain", ""),
    )

    return NeighborhoodResponse(
        center=center,
        nodes=list(nodes_map.values()),
        edges=edges_list,
    )


# ─── Visualization endpoints ──────────────────────────────────────────────────

def _classify_doc(doc_id: str) -> str:
    d = doc_id.lower()
    if any(k in d for k in ["basel","aifmd","dodd","mifid","gdpr","sox","sarbanes",
                              "volcker","fatca","ifrs","directive"," act","regulation"]):
        return "regulatory"
    if any(k in d for k in ["risk","var","stress","concentration","operational",
                              "liquidity","credit","market"]):
        return "risk"
    if any(k in d for k in ["capm","apt","black","scholes","arbitrage","portfolio",
                              "sharpe","markowitz"]):
        return "financial_theory"
    if any(k in d for k in ["fund","swap","derivative","option","futures","bond",
                              "equity","etf","hedge"]):
        return "products"
    return "other"


@app.get("/viz/nodes")
async def viz_nodes():
    with db_cursor() as cur:
        cur.execute("""
            SELECT
                d.document_id,
                d.title,
                COUNT(a.assertion_id)                                                    AS assertion_count,
                SUM(CASE WHEN a.discourse_role = 'warrant'           THEN 1 ELSE 0 END) AS warrant_count,
                SUM(CASE WHEN a.discourse_role = 'ground'            THEN 1 ELSE 0 END) AS ground_count,
                SUM(CASE WHEN a.discourse_role = 'claim'             THEN 1 ELSE 0 END) AS claim_count,
                SUM(CASE WHEN a.validity_claim_type = 'normative'    THEN 1 ELSE 0 END) AS normative_count,
                SUM(CASE WHEN a.validity_claim_type = 'expressive'   THEN 1 ELSE 0 END) AS expressive_count
            FROM documents d
            LEFT JOIN assertions a ON a.document_id = d.document_id
            GROUP BY d.document_id, d.title
            ORDER BY assertion_count DESC
        """)
        rows = cur.fetchall()

        cur.execute("""
            SELECT doc_id, COUNT(*) AS conflict_count FROM (
                SELECT a.document_id AS doc_id FROM assertions a
                JOIN assertion_relationships r ON a.assertion_id = r.source_assertion_id
                WHERE r.relationship_type = 'CONTRADICTS'
                UNION ALL
                SELECT a.document_id AS doc_id FROM assertions a
                JOIN assertion_relationships r ON a.assertion_id = r.target_assertion_id
                WHERE r.relationship_type = 'CONTRADICTS'
            ) sub GROUP BY doc_id
        """)
        conflict_map = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("""
            SELECT
                a.document_id,
                COUNT(*)                                                    AS total,
                COUNT(*) FILTER (WHERE lr.assertion_id IS NULL)             AS isolated
            FROM assertions a
            LEFT JOIN (
                SELECT source_assertion_id AS assertion_id FROM logical_relationships
                UNION
                SELECT target_assertion_id             FROM logical_relationships
            ) lr ON a.assertion_id = lr.assertion_id
            GROUP BY a.document_id
        """)
        isolation_map = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    nodes = []
    for row in rows:
        doc_id, title, ac, wc, gc, cc, nc, ec = row
        ac = ac or 0; wc = wc or 0; gc = gc or 0
        nc = nc or 0; ec = ec or 0

        total, isolated = isolation_map.get(doc_id, (ac, ac))
        isolation_rate    = isolated / max(total, 1)
        assertion_density = min(ac / 150, 1.0)
        type_coverage     = (nc + gc) / max(ac, 1)
        ai_readiness      = assertion_density * 0.3 + type_coverage * 0.5 + (1 - isolation_rate) * 0.2

        nodes.append({
            "doc_id":            doc_id,
            "title":             title,
            "category":          _classify_doc(doc_id),
            "assertion_count":   ac,
            "warrant_count":     wc,
            "ground_count":      gc,
            "claim_count":       cc or 0,
            "normative_count":   nc,
            "expressive_count":  ec,
            "conflict_count":    conflict_map.get(doc_id, 0),
            "ai_readiness_score": round(ai_readiness, 3),
        })
    return {"nodes": nodes}


@app.get("/viz/edges")
async def viz_edges():
    with db_cursor() as cur:
        cur.execute("""
            SELECT
                a1.document_id          AS source_doc,
                a2.document_id          AS target_doc,
                r.rel_type,
                COUNT(*)                AS edge_count,
                AVG(r.confidence)       AS mean_confidence
            FROM (
                SELECT source_assertion_id, target_assertion_id,
                       relation_type    AS rel_type, confidence
                FROM logical_relationships
                UNION ALL
                SELECT source_assertion_id, target_assertion_id,
                       relationship_type AS rel_type, confidence
                FROM assertion_relationships
            ) r
            JOIN assertions a1 ON r.source_assertion_id = a1.assertion_id
            JOIN assertions a2 ON r.target_assertion_id = a2.assertion_id
            WHERE a1.document_id != a2.document_id OR r.rel_type = 'CONTRADICTS'
            GROUP BY a1.document_id, a2.document_id, r.rel_type
            HAVING COUNT(*) >= 1
            ORDER BY edge_count DESC
        """)
        rows = cur.fetchall()
    edges = [
        {
            "source_doc":      r[0],
            "target_doc":      r[1],
            "rel_type":        r[2],
            "edge_count":      r[3],
            "mean_confidence": round(float(r[4] or 0), 3),
        }
        for r in rows
    ]
    return {"edges": edges}


@app.get("/viz/centrality")
async def viz_centrality():
    concepts, regulations = [], []
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                OPTIONAL MATCH (c)-[r]-()
                RETURN c.name AS name, COUNT(r) AS degree
                ORDER BY degree DESC LIMIT 20
            """)
            concepts = [{"name": r["name"], "degree": r["degree"]} for r in result]

            result = session.run("""
                MATCH (reg:Regulation)<-[:REFERENCES]-(a:Assertion)
                RETURN reg.name AS name,
                       COUNT(a) AS assertion_count,
                       COUNT(CASE WHEN a.discourse_role = 'warrant' THEN 1 END) AS warrants,
                       COUNT(DISTINCT a.source_document) AS doc_count
                ORDER BY assertion_count DESC LIMIT 20
            """)
            regulations = [
                {
                    "name":            r["name"],
                    "assertion_count": r["assertion_count"],
                    "warrants":        r["warrants"],
                    "doc_count":       r["doc_count"],
                }
                for r in result
            ]
    except Exception as e:
        logger.warning("Neo4j centrality query failed: %s", e)

    assertions = []
    with db_cursor() as cur:
        cur.execute("""
            SELECT a.assertion_id, a.claim_text, a.discourse_role,
                   a.validity_claim_type, a.document_id, COUNT(r.rid) AS degree
            FROM assertions a
            JOIN (
                SELECT relationship_id AS rid, source_assertion_id AS assertion_id FROM logical_relationships
                UNION ALL
                SELECT relationship_id,        target_assertion_id                FROM logical_relationships
                UNION ALL
                SELECT relationship_id,        source_assertion_id                FROM assertion_relationships
                UNION ALL
                SELECT relationship_id,        target_assertion_id                FROM assertion_relationships
            ) r ON a.assertion_id = r.assertion_id
            GROUP BY a.assertion_id, a.claim_text, a.discourse_role,
                     a.validity_claim_type, a.document_id
            ORDER BY degree DESC LIMIT 20
        """)
        assertions = [
            {
                "assertion_id":       str(r[0]),
                "assertion_text":     r[1],
                "discourse_role":     r[2],
                "validity_claim_type": r[3],
                "doc_id":             r[4],
                "degree":             r[5],
            }
            for r in cur.fetchall()
        ]

    return {"concepts": concepts, "regulations": regulations, "assertions": assertions}


@app.get("/viz/doc/{doc_id}/assertions")
async def viz_doc_assertions(doc_id: str, limit: int = 40):
    """Top assertions for a document, ordered by logical edge degree."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT
                a.assertion_id,
                a.claim_text,
                a.discourse_role,
                a.validity_claim_type,
                a.confidence,
                COUNT(r.rid) AS degree
            FROM assertions a
            LEFT JOIN (
                SELECT relationship_id AS rid, source_assertion_id AS assertion_id FROM logical_relationships
                UNION ALL
                SELECT relationship_id,        target_assertion_id                FROM logical_relationships
                UNION ALL
                SELECT relationship_id,        source_assertion_id                FROM assertion_relationships
                UNION ALL
                SELECT relationship_id,        target_assertion_id                FROM assertion_relationships
            ) r ON a.assertion_id = r.assertion_id
            WHERE a.document_id = %s
            GROUP BY a.assertion_id, a.claim_text, a.discourse_role,
                     a.validity_claim_type, a.confidence
            ORDER BY degree DESC
            LIMIT %s
        """, [doc_id, limit])
        rows = cur.fetchall()

    return {
        "doc_id": doc_id,
        "assertions": [
            {
                "assertion_id":       str(r[0]),
                "claim_text":         r[1],
                "discourse_role":     r[2],
                "validity_claim_type": r[3],
                "confidence":         round(float(r[4] or 1.0), 3),
                "degree":             r[5],
            }
            for r in rows
        ],
    }


@app.get("/graph/causal-chain")
def get_causal_chain(source_concept: str, target_concept: str, max_hops: int = 3):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (src:Concept {{name: $src}})<-[:GOVERNS]-(a:Assertion)
            MATCH (tgt:Concept {{name: $tgt}})<-[:GOVERNS]-(b:Assertion)
            MATCH path = (a)-[:CAUSES|INHIBITS*1..{min(max_hops,4)}]->(b)
            RETURN path LIMIT 5
            """,
            src=source_concept, tgt=target_concept,
        )
        chains = [{"path": str(record["path"])} for record in result]
    return {"chains": chains}
