"""
pipeline/stage4_embed.py — Embed chunks using Google gemini-embedding-001.

Batches of exactly 100 per API call (Google API maximum).
Concurrency: 3 parallel batch calls.
"""
import asyncio
import json
import logging
import os
from typing import List, Tuple

import numpy as np
import google.genai as genai

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # Google embedding API maximum
_client: genai.Client = None  # module-level client, set in _run_async


async def embed_batch(
    semaphore: asyncio.Semaphore,
    texts: List[str],
    chunk_ids: List[str],
    doc_id: str,
) -> Tuple[List[str], List[List[float]]]:
    """Embed a batch of texts and return (chunk_ids, vectors)."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        # Run the synchronous genai call in a thread pool
        result = await loop.run_in_executor(
            None,
            lambda: _client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
            ),
        )
        vectors = [e.values for e in result.embeddings]
        # Track cost — approximate 5 tokens per word average
        total_words  = sum(len(t.split()) for t in texts)
        input_tokens = int(total_words * 5)
        tracker.record(
            model=settings.embedding_model,
            input_tokens=input_tokens,
            output_tokens=0,
            stage="stage4_embed",
            record_id=doc_id,
        )
        return chunk_ids, vectors


async def embed_document(
    semaphore: asyncio.Semaphore,
    filename: str,
    checkpoint: CheckpointManager,
) -> None:
    path = os.path.join(settings.chunks_dir, filename)
    try:
        with open(path) as f:
            chunks = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        checkpoint.mark_failed(filename, str(e))
        return

    if not chunks:
        checkpoint.mark_done(filename)
        return

    doc_id = chunks[0].get("document_id", filename)

    # Prepare batches of 100
    all_ids:     List[str]         = []
    all_vectors: List[List[float]] = []

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    tasks = [
        embed_batch(
            semaphore,
            [c["content"] for c in batch],
            [c["chunk_id"] for c in batch],
            doc_id,
        )
        for batch in batches
    ]

    try:
        results = await asyncio.gather(*tasks)
    except CostLimitReached:
        raise

    for ids, vecs in results:
        all_ids.extend(ids)
        all_vectors.extend(vecs)

    out_path = os.path.join(settings.embeddings_dir, f"{filename.replace('.json', '')}.npz")
    np.savez(
        out_path,
        chunk_ids=np.array(all_ids),
        embeddings=np.array(all_vectors, dtype=np.float32),
    )

    checkpoint.mark_done(filename)
    logger.info(f"Embedded: {filename} → {len(all_ids)} vectors")


async def _run_async() -> None:
    global _client
    _client = genai.Client(api_key=settings.google_api_key)

    checkpoint  = CheckpointManager("stage4_embed")
    already_done = checkpoint.get_completed_ids()

    chunk_files = [f for f in os.listdir(settings.chunks_dir) if f.endswith(".json")]
    work_queue  = [f for f in chunk_files if f not in already_done]
    checkpoint.set_total(len(chunk_files))

    os.makedirs(settings.embeddings_dir, exist_ok=True)

    logger.info(f"Stage 4: {len(work_queue)} files to embed ({len(already_done)} done)")

    semaphore = asyncio.Semaphore(3)  # 3 parallel batch calls

    try:
        tasks = [embed_document(semaphore, fn, checkpoint) for fn in work_queue]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 4: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
