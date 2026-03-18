"""pipeline/db.py — Database connection pool and helpers."""
import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.pool
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from pipeline.config import settings

logger = logging.getLogger(__name__)

# ─── PostgreSQL ───────────────────────────────────────────────────────────────

_pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def init_pool() -> None:
    """Initialize the PostgreSQL connection pool."""
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=settings.postgres_url,
        )
        logger.info("PostgreSQL connection pool initialized")


def get_conn():
    """Return a connection from the pool. Initialises pool on first call."""
    global _pg_pool
    if _pg_pool is None:
        init_pool()
    return _pg_pool.getconn()


def release_conn(conn) -> None:
    """Return a connection to the pool."""
    if _pg_pool and conn:
        _pg_pool.putconn(conn)


@contextmanager
def db_cursor():
    """Context manager: yields a cursor and auto-commits / releases connection."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def execute_many(sql: str, params_list: list) -> None:
    """Batch-insert helper using executemany."""
    if not params_list:
        return
    with db_cursor() as cur:
        cur.executemany(sql, params_list)


# ─── Neo4j ────────────────────────────────────────────────────────────────────

_neo4j_driver = None


def get_neo4j_driver():
    """Return (and cache) the Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j driver initialised")
    return _neo4j_driver


# ─── Qdrant ───────────────────────────────────────────────────────────────────

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return (and cache) the Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
        logger.info("Qdrant client initialised")
    return _qdrant_client
