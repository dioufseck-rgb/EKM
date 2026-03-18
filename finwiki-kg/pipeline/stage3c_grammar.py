"""
pipeline/stage3c_grammar.py — Toulmin + Habermas grammar classification.

Runs after Stage 3. For each document, assigns discourse_role and
validity_claim_type to each assertion. One LLM call per document batch
(up to 30 assertions per call).

Concurrency: 5. No DB required.
"""
import asyncio
import json
import logging
import os
from typing import List

import google.genai as genai
from google.genai import types as genai_types

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker

logger = logging.getLogger(__name__)

BATCH_SIZE = 30  # assertions per LLM call

PROMPT_TEMPLATE = """\
You are classifying assertions from a financial services document corpus
using two analytical frameworks.

DOCUMENT: {document_title}

ASSERTIONS:
{assertions_list}

For each assertion assign:

1. DISCOURSE_ROLE (Toulmin structural role):

   warrant    — a rule, principle, or regulation that licenses an inference
                Signs: "must", "shall", "required to", "prohibited from"
                Example: "Banks must maintain Tier 1 capital ratio above 6%"

   ground     — a factual datum or piece of evidence
                Signs: present-tense factual description, statistics, definitions
                Example: "NFCU's Tier 1 capital ratio is 5.8%"

   claim      — a conclusion derived or derivable from warrant + ground
                Signs: follows logically from other assertions in context
                Example: "NFCU is non-compliant with capital requirements"

   backing    — authority or citation supporting a warrant
                Signs: regulatory citations, standards references, "per §...", "under..."
                Example: "Per Basel III section 4.2, capital requirements are..."

   rebuttal   — an exception or condition that defeats a claim
                Signs: "unless", "except", "provided that", "does not apply when"
                Example: "Unless the entity qualifies for an exemption under rule Y"

   qualifier  — a scope hedge or confidence condition
                Signs: "under normal conditions", "as applicable", "where required",
                       "in most cases", "generally"

   unclassified — does not fit any role clearly

2. VALIDITY_CLAIM_TYPE (Habermas):

   constative  — a truth claim about how things ARE
                 Signs: descriptive language, present tense facts, statistics
                 Example: "The Fed funds rate is 5.25%"

   normative   — a rightness claim about what SHOULD or MUST be done
                 Signs: "must", "shall", "required", "prohibited", "permitted"
                 Example: "Banks shall file SARs for transactions over $10,000"

   expressive  — a statement of intent, purpose, or institutional stance
                 Signs: "designed to", "intended to", "the purpose of this policy"
                 Example: "This policy is intended to protect customer data"

   unclassified — cannot be determined

HEURISTICS FOR FINANCIAL SERVICES CORPUS:
- Regulatory requirement sentences → normative warrants
- Definition sentences ("X is defined as...") → constative grounds
- Threshold sentences ("X must be above Y%") → normative warrants
- Historical/background sentences → constative grounds
- Purpose statements → expressive
- Exception clauses → rebuttals (look hard for these — they are underextracted)
- "Unless", "except when", "provided that" → rebuttal
- Qualifier language is often implicit — look for hedging even without explicit markers

Return JSON only:
{{"classifications": [{{"assertion_id": "...", "discourse_role": "...", "validity_claim_type": "..."}}]}}
"""


def _build_assertions_list(assertions: List[dict]) -> str:
    lines = []
    for a in assertions:
        lines.append(f"{a['assertion_id']} | {a['claim_text']}")
    return "\n".join(lines)


async def classify_batch(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    document_title: str,
    batch: List[dict],
    doc_id: str,
) -> List[dict]:
    """Classify one batch of assertions. Returns list of {assertion_id, discourse_role, validity_claim_type}."""
    assertions_list = _build_assertions_list(batch)
    prompt = PROMPT_TEMPLATE.format(
        document_title=document_title,
        assertions_list=assertions_list,
    )

    async with semaphore:
        loop = asyncio.get_event_loop()
        for attempt in range(3):
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=settings.flash_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    ),
                )
                raw = resp.text
                input_tokens  = int(len(prompt) / 4)
                output_tokens = int(len(raw) / 4)
                tracker.record(
                    model=settings.flash_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stage="stage3c_grammar",
                    record_id=doc_id,
                )
                result = json.loads(raw)
                return result.get("classifications", [])

            except json.JSONDecodeError as e:
                wait = 2 ** attempt
                logger.warning(f"JSONDecodeError doc={doc_id} attempt={attempt+1}: {e}. Retrying in {wait}s")
                await asyncio.sleep(wait)
            except CostLimitReached:
                raise
            except Exception as e:
                logger.error(f"LLM error doc={doc_id}: {e}")
                return []

    logger.error(f"All retries failed for grammar doc={doc_id}")
    return []


async def classify_document(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    doc_id: str,
    checkpoint: CheckpointManager,
) -> None:
    assertions_path = os.path.join(settings.assertions_dir, f"{doc_id}_assertions.json")
    if not os.path.exists(assertions_path):
        checkpoint.mark_done(doc_id)
        return

    with open(assertions_path) as f:
        assertions = json.load(f)

    if not assertions:
        checkpoint.mark_done(doc_id)
        return

    doc_title = assertions[0].get("source_document", doc_id) if assertions else doc_id

    # Process in batches of BATCH_SIZE
    batches = [assertions[i:i + BATCH_SIZE] for i in range(0, len(assertions), BATCH_SIZE)]
    tasks = [
        classify_batch(semaphore, client, doc_title, batch, doc_id)
        for batch in batches
    ]

    all_classifications = []
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            all_classifications.extend(result)
        except CostLimitReached:
            raise

    # Build lookup map
    grammar_map = {c["assertion_id"]: c for c in all_classifications if "assertion_id" in c}

    # Write grammar file
    os.makedirs(settings.grammar_dir, exist_ok=True)
    out_path = os.path.join(settings.grammar_dir, f"{doc_id}_grammar.json")
    with open(out_path, "w") as f:
        json.dump(grammar_map, f, indent=2)

    checkpoint.mark_done(doc_id)
    logger.info(f"Grammar: {doc_id} → {len(grammar_map)} classifications")


async def _run_async() -> None:
    client = genai.Client(api_key=settings.google_api_key)

    checkpoint = CheckpointManager("stage3c_grammar")
    already_done = checkpoint.get_completed_ids()

    assertion_files = [
        f.replace("_assertions.json", "")
        for f in os.listdir(settings.assertions_dir)
        if f.endswith("_assertions.json")
    ]
    work_queue = [doc_id for doc_id in assertion_files if doc_id not in already_done]
    checkpoint.set_total(len(assertion_files))

    logger.info(f"Stage 3c: {len(work_queue)} documents to classify ({len(already_done)} done)")

    semaphore = asyncio.Semaphore(settings.pipeline_concurrency)

    try:
        tasks = [classify_document(semaphore, client, doc_id, checkpoint) for doc_id in work_queue]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 3c: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
