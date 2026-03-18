"""pipeline/config.py — Settings loaded from environment variables."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # APIs
    google_api_key:  str   = os.getenv("GOOGLE_API_KEY", "")
    # Databases
    postgres_url:    str   = os.getenv("POSTGRES_URL", "postgresql://finwiki:finwiki@localhost:5432/finwiki")
    qdrant_url:      str   = os.getenv("QDRANT_URL", "http://localhost:6333")
    neo4j_url:       str   = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user:      str   = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password:  str   = os.getenv("NEO4J_PASSWORD", "finwiki123")
    # Cost control
    cost_ceiling_usd: float = float(os.getenv("COST_CEILING_USD", "9.50"))
    cost_override:    bool  = os.getenv("COST_OVERRIDE", "false").lower() == "true"
    # Pipeline
    pipeline_concurrency: int = int(os.getenv("PIPELINE_CONCURRENCY", "5"))
    log_level:            str = os.getenv("LOG_LEVEL", "INFO")
    api_url:              str = os.getenv("API_URL", "http://localhost:8000")
    # Model names
    flash_model:      str = "gemini-2.0-flash"
    pro_model:        str = "gemini-2.0-flash"
    embedding_model:  str = "gemini-embedding-001"
    # Data paths
    data_dir:         str = "data"
    raw_dir:          str = "data/raw"
    chunks_dir:       str = "data/chunks"
    assertions_dir:   str = "data/assertions"
    grammar_dir:      str = "data/grammar"
    relations_dir:    str = "data/relations"
    embeddings_dir:   str = "data/embeddings"
    checkpoints_dir:  str = "data/checkpoints"


settings = Settings()
