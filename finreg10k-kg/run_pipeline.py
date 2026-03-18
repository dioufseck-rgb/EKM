"""
run_pipeline.py — FinReg10K pipeline orchestrator.

Orchestrates the full FinReg10K pipeline:
  0. Stage 0: corpus acquisition (stage0_acquire.py)
  1. Stage 1: preprocessing + chunking (stage1_preprocess.py)
  2-8. FinWiki pipeline stages 2-8 (imported from finwiki-kg)

The existing FinWiki pipeline stages are consumed unchanged.
Stage 1 produces chunk files in data/chunks/ that the downstream stages
read, ensuring no modification is needed.

Usage:
    python run_pipeline.py [--skip-acquire] [--skip-preprocess] [--stages 3 4 5]

Environment:
    FINWIKI_KG_PATH  — path to finwiki-kg directory (default: ../finwiki-kg)
    GOOGLE_API_KEY   — required for stages 3, 3b, 3c, 4, 6, 7
    POSTGRES_URL     — PostgreSQL connection string
    QDRANT_URL       — Qdrant vector store URL
    NEO4J_URL        — Neo4j bolt URL
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

# ── Resolve finwiki-kg path ────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FINWIKI_PATH = Path(os.environ.get("FINWIKI_KG_PATH", BASE_DIR.parent / "finwiki-kg"))

if not FINWIKI_PATH.exists():
    print(f"ERROR: finwiki-kg not found at {FINWIKI_PATH}", file=sys.stderr)
    print("Set FINWIKI_KG_PATH environment variable to the correct path.", file=sys.stderr)
    sys.exit(1)

# Insert finwiki-kg onto sys.path so its pipeline package is importable
if str(FINWIKI_PATH) not in sys.path:
    sys.path.insert(0, str(FINWIKI_PATH))

# Insert finreg10k-kg itself so stage0/stage1 are importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Override data paths so pipeline writes into finreg10k-kg/data/ ─────────────
# This must happen before importing pipeline.config (which reads env at import time)
_data = str(BASE_DIR / "data")
os.environ.setdefault("POSTGRES_URL",   "postgresql://finwiki:finwiki@localhost:5432/finwiki")
os.environ.setdefault("QDRANT_URL",     "http://localhost:6333")
os.environ.setdefault("NEO4J_URL",      "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER",     "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "finwiki123")

# Monkey-patch settings paths BEFORE the pipeline modules load
# We do this by setting env vars that config.py reads:
#   data_dir, raw_dir, chunks_dir, assertions_dir, grammar_dir,
#   relations_dir, embeddings_dir, checkpoints_dir
os.environ["DATA_DIR"]        = _data
os.environ["RAW_DIR"]         = str(BASE_DIR / "data" / "raw")
os.environ["CHUNKS_DIR"]      = str(BASE_DIR / "data" / "chunks")
os.environ["ASSERTIONS_DIR"]  = str(BASE_DIR / "data" / "assertions")
os.environ["GRAMMAR_DIR"]     = str(BASE_DIR / "data" / "grammar")
os.environ["RELATIONS_DIR"]   = str(BASE_DIR / "data" / "relations")
os.environ["EMBEDDINGS_DIR"]  = str(BASE_DIR / "data" / "embeddings")
os.environ["CHECKPOINTS_DIR"] = str(BASE_DIR / "data" / "checkpoints")


def _patch_settings() -> None:
    """
    After importing pipeline.config, patch Settings paths to point at
    finreg10k-kg/data/ so all pipeline stages write here instead of
    the finwiki-kg data directory.
    """
    try:
        from pipeline.config import settings
        settings.data_dir        = str(BASE_DIR / "data")
        settings.raw_dir         = str(BASE_DIR / "data" / "raw")
        settings.chunks_dir      = str(BASE_DIR / "data" / "chunks")
        settings.assertions_dir  = str(BASE_DIR / "data" / "assertions")
        settings.grammar_dir     = str(BASE_DIR / "data" / "grammar")
        settings.relations_dir   = str(BASE_DIR / "data" / "relations")
        settings.embeddings_dir  = str(BASE_DIR / "data" / "embeddings")
        settings.checkpoints_dir = str(BASE_DIR / "data" / "checkpoints")
    except ImportError as e:
        print(f"ERROR: Could not import pipeline.config: {e}", file=sys.stderr)
        sys.exit(1)


def _ensure_dirs() -> None:
    """Create all required data directories."""
    for subdir in ("raw", "chunks", "assertions", "grammar", "relations",
                   "embeddings", "checkpoints", "eval"):
        (BASE_DIR / "data" / subdir).mkdir(parents=True, exist_ok=True)


# ── Pipeline stage registry ────────────────────────────────────────────────────
# Stages 2-8 are the existing FinWiki pipeline stages (run unchanged).
# Stage 0 and 1 are FinReg10K-specific acquisition/preprocessing.

FINREG_STAGES = [
    ("stage0_acquire",    "stage0_acquire",           "FinReg10K corpus acquisition"),
    ("stage1_preprocess", "stage1_preprocess",         "FinReg10K chunking + manifest"),
]

PIPELINE_STAGES = [
    # stage1_crawl is replaced by stage0+stage1 above — skip it
    ("stage2_chunk",          "pipeline.stage2_chunk",          "chunk articles"),
    ("stage3_classify",       "pipeline.stage3_classify",       "extract assertions"),
    ("stage3c_grammar",       "pipeline.stage3c_grammar",       "discourse grammar"),
    ("stage4_embed",          "pipeline.stage4_embed",          "embed chunks"),
    ("stage5_load",           "pipeline.stage5_load",           "load to DB"),
    ("stage3b_relations",     "pipeline.stage3b_relations",     "within-doc relations"),
    ("stage6_conflicts",      "pipeline.stage6_conflicts",      "conflict detection"),
    ("stage7_relations_xdoc", "pipeline.stage7_relations_xdoc", "cross-doc relations"),
    ("stage8_graph",          "pipeline.stage8_graph",          "build Neo4j graph"),
]


def run_stage(stage_name: str, module_path: str, description: str) -> None:
    """Import and run a single pipeline stage."""
    logger = logging.getLogger("run_pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"  Starting {stage_name}: {description}")
    logger.info(f"{'='*60}")

    try:
        mod = importlib.import_module(module_path)
        mod.run()
    except SystemExit as e:
        logger.error(f"  {stage_name} called sys.exit({e.code})")
        sys.exit(e.code)
    except Exception as e:
        logger.error(f"  {stage_name} FAILED: {e}", exc_info=True)
        sys.exit(1)

    # Report running cost after LLM stages
    try:
        from pipeline.cost_tracker import tracker
        logger.info(f"  {stage_name} COMPLETE  |  running cost: ${tracker.total:.4f}")
    except ImportError:
        logger.info(f"  {stage_name} COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinReg10K pipeline orchestrator"
    )
    parser.add_argument(
        "--skip-acquire",
        action="store_true",
        help="Skip stage 0 (corpus acquisition)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip stage 1 (chunking/manifest generation)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        help="Run only specific downstream stage numbers (e.g. --stages 3 4 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_pipeline")
    logger.info("=== FinReg10K Pipeline ===")
    logger.info(f"  Base dir:     {BASE_DIR}")
    logger.info(f"  FinWiki path: {FINWIKI_PATH}")

    _ensure_dirs()

    # Run FinReg10K-specific stages
    if not args.skip_acquire:
        run_stage("stage0_acquire", "stage0_acquire", "corpus acquisition")
    else:
        logger.info("  Skipping stage 0 (--skip-acquire)")

    if not args.skip_preprocess:
        run_stage("stage1_preprocess", "stage1_preprocess", "chunking + manifest")
    else:
        logger.info("  Skipping stage 1 (--skip-preprocess)")

    # Patch settings to point at finreg10k-kg/data/ before running FinWiki stages
    _patch_settings()

    # Determine which downstream stages to run
    stage_filter = set(args.stages) if args.stages else None

    for stage_name, module_path, description in PIPELINE_STAGES:
        # If --stages filter given, extract the number suffix for matching
        stage_num = stage_name.replace("stage", "").split("_")[0]
        if stage_filter and stage_num not in stage_filter:
            logger.info(f"  Skipping {stage_name} (not in --stages filter)")
            continue

        run_stage(stage_name, module_path, description)

    # Final cost report
    try:
        from pipeline.cost_tracker import tracker
        logger.info(f"{'='*60}")
        logger.info(f"  ALL STAGES COMPLETE  |  final cost: ${tracker.total:.4f}")
        logger.info(f"{'='*60}")
        print(f"\nFinal pipeline cost: ${tracker.total:.4f}")
    except ImportError:
        logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
