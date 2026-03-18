"""api/search.py — Semantic search over the assertion vector store."""
import logging
from typing import List, Optional

import google.genai as genai
from google.genai import types as genai_types
from fastapi import APIRouter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from api.models import AssertionResponse, SearchResult
from pipeline.config import settings
from pipeline.db import db_cursor

logger = logging.getLogger(__name__)
router = APIRouter()


def embed_query(text: str) -> list:
    """Embed a query string with task_type=RETRIEVAL_QUERY."""
    client = genai.Client(api_key=settings.google_api_key)
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def get_assertions_for_chunks(
    chunk_ids: List[str],
    min_confidence: float = 0.5,
    domain: Optional[str] = None,
) -> List[AssertionResponse]:
    """Fetch assertions from PostgreSQL for a list of chunk_ids."""
    if not chunk_ids:
        return []

    placeholders = ",".join(["%s"] * len(chunk_ids))
    params: list = list(chunk_ids) + [min_confidence]
    query = f"""
        SELECT assertion_id, claim_text, subject, predicate_type, object_text,
               source_document, source_url, epistemic_status, confidence, domain,
               derivation_chain
        FROM assertions
        WHERE chunk_id IN ({placeholders})
          AND confidence >= %s
          AND epistemic_status NOT IN ('deprecated', 'orphaned')
    """
    if domain:
        query += " AND domain = %s"
        params.append(domain)

    with db_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        AssertionResponse(
            assertion_id=r[0], claim_text=r[1], subject=r[2],
            predicate_type=r[3], object_text=r[4],
            source_document=r[5] or "", source_url=r[6] or "",
            epistemic_status=r[7] or "authoritative",
            confidence=float(r[8] or 0.0), domain=r[9] or "",
            derivation_chain=r[10] or [],
        )
        for r in rows
    ]


@router.get("/search", response_model=List[SearchResult])
def search(
    q: str,
    domain: Optional[str] = None,
    min_confidence: float = 0.5,
    limit: int = 10,
) -> List[SearchResult]:
    """Semantic similarity search over chunk embeddings, returns matching assertions."""
    vector = embed_query(q)

    qdrant = QdrantClient(url=settings.qdrant_url)
    search_filter = None
    if domain:
        search_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    result = qdrant.query_points(
        collection_name="finwiki_chunks",
        query=vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )
    hits = result.points

    chunk_ids  = [h.payload.get("chunk_id", "") for h in hits if h.payload]
    score_map  = {h.payload.get("chunk_id", ""): h.score for h in hits if h.payload}

    assertions = get_assertions_for_chunks(chunk_ids, min_confidence, domain)

    return [
        SearchResult(
            assertion=a,
            similarity_score=score_map.get(a.assertion_id, 0.0),
        )
        for a in assertions
    ]
