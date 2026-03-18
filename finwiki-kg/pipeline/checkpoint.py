"""pipeline/checkpoint.py — CheckpointManager for resumable pipeline stages."""
import json
import logging
import os
from datetime import datetime
from typing import Set

from pipeline.config import settings

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Tracks completed record IDs for a pipeline stage.

    Every stage MUST follow this pattern:
        checkpoint = CheckpointManager("stage_name")
        already_done = checkpoint.get_completed_ids()
        work_queue = [item for item in all_items if item.id not in already_done]
        for item in work_queue:
            process(item)
            checkpoint.mark_done(item.id)
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.path = os.path.join(settings.checkpoints_dir, f"{stage_name}.json")
        self._data = self._load()

    # ── Private ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[{self.stage_name}] Checkpoint load failed ({e}), starting fresh")
        return {
            "stage":         self.stage_name,
            "completed_ids": [],
            "started_at":    datetime.utcnow().isoformat(),
            "last_updated":  datetime.utcnow().isoformat(),
            "records_total": 0,
            "records_done":  0,
            "status":        "running",
        }

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data["last_updated"] = datetime.utcnow().isoformat()
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_completed_ids(self) -> Set[str]:
        return set(self._data.get("completed_ids", []))

    def mark_done(self, record_id: str) -> None:
        if record_id not in self._data["completed_ids"]:
            self._data["completed_ids"].append(record_id)
            self._data["records_done"] = len(self._data["completed_ids"])
            self._save()

    def mark_failed(self, record_id: str, reason: str = "") -> None:
        logger.warning(f"[{self.stage_name}] FAILED record={record_id} reason={reason}")
        # Intentionally NOT added to completed_ids — will be retried on next run

    def set_total(self, total: int) -> None:
        self._data["records_total"] = total
        self._save()

    def set_status(self, status: str) -> None:
        self._data["status"] = status
        self._save()

    def complete(self) -> None:
        self._data["status"] = "complete"
        self._save()
        logger.info(
            f"[{self.stage_name}] Complete — "
            f"{self._data['records_done']}/{self._data['records_total']} records"
        )

    @property
    def status(self) -> str:
        return self._data.get("status", "unknown")

    @property
    def records_done(self) -> int:
        return self._data.get("records_done", 0)

    @property
    def records_total(self) -> int:
        return self._data.get("records_total", 0)
