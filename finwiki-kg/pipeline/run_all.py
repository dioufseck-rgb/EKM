"""
pipeline/run_all.py — Sequential runner for all 8 pipeline stages.
"""
import importlib
import logging
import sys

from pipeline.config import settings
from pipeline.cost_tracker import tracker


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Stage order per CLAUDE.md: 1→2→3→3c→4→5→3b→6→7→8
    # Note: 3b (within-doc relations) runs AFTER 5 (load) because it needs assertion_ids in DB
    # Note: 3c (grammar) runs AFTER 3 and BEFORE 5 so grammar is available for loading
    stages = [
        ("stage1_crawl",           "pipeline.stage1_crawl"),
        ("stage2_chunk",           "pipeline.stage2_chunk"),
        ("stage3_classify",        "pipeline.stage3_classify"),
        ("stage3c_grammar",        "pipeline.stage3c_grammar"),
        ("stage4_embed",           "pipeline.stage4_embed"),
        ("stage5_load",            "pipeline.stage5_load"),
        ("stage3b_relations",      "pipeline.stage3b_relations"),
        ("stage6_conflicts",       "pipeline.stage6_conflicts"),
        ("stage7_relations_xdoc",  "pipeline.stage7_relations_xdoc"),
        ("stage8_graph",           "pipeline.stage8_graph"),
    ]

    for stage_name, module_path in stages:
        logging.info(f"{'='*60}")
        logging.info(f"  Starting {stage_name}")
        logging.info(f"{'='*60}")
        try:
            mod = importlib.import_module(module_path)
            mod.run()
            logging.info(
                f"  {stage_name} COMPLETE  |  running cost: ${tracker.total:.4f}"
            )
        except SystemExit as e:
            logging.error(f"  {stage_name} called sys.exit({e.code})")
            sys.exit(e.code)
        except Exception as e:
            logging.error(f"  {stage_name} FAILED: {e}", exc_info=True)
            sys.exit(1)

    logging.info(f"{'='*60}")
    logging.info(f"  ALL STAGES COMPLETE  |  final cost: ${tracker.total:.4f}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
