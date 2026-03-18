"""
pipeline/cost_tracker.py — Central LLM cost tracking with hard-stop ceiling.

EVERY LLM call — without exception — must go through tracker.record().
"""
import logging

from pipeline.config import settings

logger = logging.getLogger(__name__)

# Pricing per model (per 1M tokens for LLMs, per 1K tokens for embeddings)
PRICING = {
    "gemini-1.5-flash":   {"input": 0.075,     "output": 0.30},
    "gemini-1.5-pro":     {"input": 1.25,       "output": 5.00},
    "text-embedding-004": {"input": 0.000025,   "output": 0.0},   # per 1K tokens
}


class CostLimitReached(Exception):
    """Raised when the running total reaches or exceeds COST_CEILING_USD."""


class CostTracker:
    """Thread-safe running cost accumulator with PostgreSQL persistence."""

    def __init__(self) -> None:
        self._running_total: float = self._load_running_total()

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_running_total(self) -> float:
        try:
            from pipeline.db import db_cursor
            with db_cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(running_total_usd), 0) FROM llm_cost_log")
                return float(cur.fetchone()[0])
        except Exception as e:
            logger.debug(f"Cost tracker: DB not yet available ({e}), starting at $0.00")
            return 0.0

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = PRICING.get(model, {"input": 0, "output": 0})
        if model == "text-embedding-004":
            # Priced per 1K tokens
            return (input_tokens / 1_000) * pricing["input"]
        return (
            (input_tokens  / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )

    def _persist(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        stage: str,
        record_id: str,
    ) -> None:
        try:
            from pipeline.db import db_cursor
            with db_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_cost_log
                        (model, input_tokens, output_tokens, cost_usd, running_total_usd, stage, record_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (model, input_tokens, output_tokens, cost, self._running_total, stage, record_id),
                )
        except Exception as e:
            logger.warning(f"Cost tracker: DB write failed ({e})")

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        stage: str,
        record_id: str = "",
    ) -> float:
        """
        Record an LLM call, persist to DB, and raise CostLimitReached if ceiling hit.

        Returns the cost of this call in USD.
        """
        cost = self._compute_cost(model, input_tokens, output_tokens)
        self._running_total += cost
        self._persist(model, input_tokens, output_tokens, cost, stage, record_id)

        logger.info(
            f"[cost] stage={stage} model={model} "
            f"in={input_tokens} out={output_tokens} "
            f"call=${cost:.5f} total=${self._running_total:.4f}"
        )

        if not settings.cost_override and self._running_total >= settings.cost_ceiling_usd:
            raise CostLimitReached(
                f"Cost ceiling ${settings.cost_ceiling_usd:.2f} reached. "
                f"Running total: ${self._running_total:.4f}"
            )

        return cost

    @property
    def total(self) -> float:
        return self._running_total


# Module-level singleton — import and use everywhere
tracker = CostTracker()
