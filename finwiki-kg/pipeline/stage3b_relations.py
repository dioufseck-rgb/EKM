"""
pipeline/stage3b_relations.py — Within-document logical relation extraction.

Runs after Stage 3. For each document, splits assertions into batches of
BATCH_SIZE and issues one LLM call per batch, plus one cross-batch call per
adjacent batch pair to catch relations that span batch boundaries. Results
are merged and deduplicated.

Concurrency: semaphore is held per LLM call (not per document) so batch
calls from multiple documents interleave freely up to pipeline_concurrency.

Failure guard: documents with ≥ MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS that
return zero relations are NOT marked complete — they remain in the work
queue so a rerun will retry them.
"""
import asyncio
import json
import logging
import os
import uuid
from typing import List, Set, Tuple

import google.genai as genai
from google.genai import types as genai_types

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker
from pipeline.schema import LogicalRelationship, LogicalRelationType, ReviewStatus

logger = logging.getLogger(__name__)

BATCH_SIZE = 50                         # assertions per within-batch prompt (~2 500 tokens)
CROSS_BATCH_WINDOW = 10                 # assertions from each side for cross-batch prompt
MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS = 30  # docs below this may genuinely have no relations

PROMPT_TEMPLATE = """\
You are identifying logical relationships between assertions extracted from a financial document.

Document: {title}

Assertions:
{assertions_list}

For each PAIR of assertions where a logical relation exists, identify the relation type:

1. ENTAILMENT: If A is true, does B necessarily follow as a matter of logic?
   Only mark ENTAILS if the conclusion is logically guaranteed, not merely likely.

2. CAUSATION: Does A produce or cause B through a real-world mechanism?
   State the mechanism in one sentence. Mark strength: strong | moderate | weak.
   IMPORTANT: CAUSES is NOT ENTAILS. Causal relations are defeasible.

3. DEFINITION dependency: Does A define a term used in B?
   If B uses a term that A defines, mark DEFINES (source=A, target=B).

4. TRIGGERING: Does A being true activate B as a new obligation or requirement?
   Common in regulatory text: condition → obligation.

5. REBUTTAL: Is A an exception condition that defeats B?
   A REBUTS B means: when A's conditions are met, B does not apply.
   This is NOT a contradiction — it is a structural exception. Look for:
   "unless", "except when", "provided that", "does not apply when", "exempt if".

6. TEMPORAL sequence: Must A precede B procedurally? Mark PRECEDES.

7. INHIBITION: Does A make B less likely or harder to achieve? Mark INHIBITS.

8. CORRELATION: Do A and B co-occur without established causation? Mark CORRELATES_WITH.

9. EQUIVALENCE: Are A and B logically the same claim in different words?
   Only mark EQUIVALENT if A entails B AND B entails A.

10. SPECIALIZATION: Is A a scoped (narrower) version of B?
    A's validity envelope must be a proper subset of B's. Mark SPECIALIZES.

11. SUPERSEDES: Does A replace B with respect to authority?

12. CONTRADICTION: Can A and B NOT both be true in overlapping scope? Mark CONTRADICTS.

For each relation return:
{{
  "source_assertion_id": "<full UUID>",
  "target_assertion_id": "<full UUID>",
  "relation_type": "ENTAILS|CAUSES|DEFINES|TRIGGERS|REBUTS|PRECEDES|INHIBITS|CORRELATES_WITH|EQUIVALENT|SPECIALIZES|SUPERSEDES|CONTRADICTS|GENERALIZES",
  "is_truth_preserving": true|false,
  "is_defeasible": true|false,
  "is_bidirectional": true|false,
  "evidence_text": "short quote from document",
  "logical_form": "if A then B (or empty string)",
  "mechanism": "one sentence (CAUSES/INHIBITS only, else empty)",
  "strength": "strong|moderate|weak (CAUSES/INHIBITS/CORRELATES_WITH only, else empty)",
  "directionality": "A_to_B|B_to_A|bidirectional",
  "confidence": 0.0-1.0,
  "scope_condition": "description if relation is itself scoped, else empty"
}}

is_truth_preserving = true ONLY for: ENTAILS, EQUIVALENT, DEFINES, TRIGGERS, SPECIALIZES, GENERALIZES, SUPERSEDES.
is_defeasible = true ONLY for: CAUSES, INHIBITS, CORRELATES_WITH, REBUTS.
is_bidirectional = true ONLY for: CONTRADICTS, EQUIVALENT, CORRELATES_WITH.

Return a JSON array. No markdown. No preamble.
"""

_TRUTH_PRESERVING = {
    "ENTAILS", "EQUIVALENT", "DEFINES", "TRIGGERS",
    "SPECIALIZES", "GENERALIZES", "SUPERSEDES",
}
_DEFEASIBLE    = {"CAUSES", "INHIBITS", "CORRELATES_WITH", "REBUTS"}
_BIDIRECTIONAL = {"CONTRADICTS", "EQUIVALENT", "CORRELATES_WITH"}


def _build_assertions_list(assertions: List[dict]) -> str:
    lines = []
    for a in assertions:
        scope = a.get("scope", {})
        geo   = scope.get("geographic", {}) if isinstance(scope, dict) else {}
        temp  = scope.get("temporal", {}) if isinstance(scope, dict) else {}
        scope_summary = []
        if not geo.get("is_global", True):
            scope_summary.append(f"countries={geo.get('countries')}")
        if not temp.get("is_default", True):
            scope_summary.append(f"season={temp.get('season')}")
        scope_str = ", ".join(scope_summary) if scope_summary else "universal"
        lines.append(
            f"ID: {a['assertion_id']}\n"
            f"Claim: {a['claim_text']}\n"
            f"Scope: {scope_str}\n"
        )
    return "\n".join(lines)


def _parse_relations(raw: str, document_id: str) -> List[LogicalRelationship]:
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    items = json.loads(raw)
    relations = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rtype_str = item.get("relation_type", "")
        try:
            rtype = LogicalRelationType[rtype_str]
        except KeyError:
            logger.warning(f"Unknown relation type: {rtype_str}")
            continue

        lr = LogicalRelationship(
            relationship_id     = str(uuid.uuid4()),
            source_assertion_id = item.get("source_assertion_id", ""),
            target_assertion_id = item.get("target_assertion_id", ""),
            relation_type       = rtype,
            is_truth_preserving = item.get("is_truth_preserving", rtype_str in _TRUTH_PRESERVING),
            is_defeasible       = item.get("is_defeasible",    rtype_str in _DEFEASIBLE),
            is_bidirectional    = item.get("is_bidirectional", rtype_str in _BIDIRECTIONAL),
            evidence_text       = item.get("evidence_text", ""),
            logical_form        = item.get("logical_form", ""),
            mechanism           = item.get("mechanism", ""),
            strength            = item.get("strength", ""),
            directionality      = item.get("directionality", "A_to_B"),
            confidence          = float(item.get("confidence", 0.8)),
            extraction_method   = "llm_within_doc",
            derivation_depth    = 0,
            review_status       = ReviewStatus.pending,
        )
        relations.append(lr)
    return relations


async def _call_llm_for_batch(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    batch: List[dict],
    title: str,
    doc_id: str,
    batch_label: str,
) -> List[LogicalRelationship]:
    """Issue one LLM call for a batch of assertions. Retries up to 3× on JSON errors."""
    if len(batch) < 2:
        return []

    assertions_list = _build_assertions_list(batch)
    prompt = PROMPT_TEMPLATE.format(title=title, assertions_list=assertions_list)

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
                tracker.record(
                    model=settings.flash_model,
                    input_tokens=int(len(prompt) / 4),
                    output_tokens=int(len(raw) / 4),
                    stage="stage3b_relations",
                    record_id=f"{doc_id}:{batch_label}",
                )
                return _parse_relations(raw, doc_id)
            except json.JSONDecodeError as e:
                wait = 2 ** attempt
                logger.warning(
                    f"JSONDecodeError doc={doc_id} batch={batch_label} "
                    f"attempt={attempt + 1}: {e}. Retry in {wait}s"
                )
                await asyncio.sleep(wait)
            except CostLimitReached:
                raise
            except Exception as e:
                logger.error(f"LLM error doc={doc_id} batch={batch_label}: {e}")
                break
    return []


def _deduplicate(relations: List[LogicalRelationship]) -> List[LogicalRelationship]:
    seen: Set[Tuple[str, str, str]] = set()
    unique = []
    for r in relations:
        key = (r.source_assertion_id, r.target_assertion_id, r.relation_type.value)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _should_mark_complete(assertion_count: int, relation_count: int) -> bool:
    """Return False for large docs that produced no relations — they should be retried."""
    if assertion_count < MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS:
        return True   # small docs may genuinely have no relations
    return relation_count > 0


async def process_document(
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

    if len(assertions) < 2:
        checkpoint.mark_done(doc_id)
        return

    title = assertions[0].get("source_document", doc_id) if assertions else doc_id

    # Split into batches of BATCH_SIZE
    batches = [
        assertions[i : i + BATCH_SIZE]
        for i in range(0, len(assertions), BATCH_SIZE)
    ]

    # Schedule within-batch calls
    batch_tasks = [
        _call_llm_for_batch(semaphore, client, batch, title, doc_id, f"batch{idx}")
        for idx, batch in enumerate(batches)
    ]

    # Schedule cross-batch calls for adjacent pairs
    for i in range(len(batches) - 1):
        cross = batches[i][-CROSS_BATCH_WINDOW:] + batches[i + 1][:CROSS_BATCH_WINDOW]
        batch_tasks.append(
            _call_llm_for_batch(semaphore, client, cross, title, doc_id, f"cross{i}-{i+1}")
        )

    results = await asyncio.gather(*batch_tasks)
    all_relations = [r for batch_result in results for r in batch_result]
    relations = _deduplicate(all_relations)

    # Serialize
    out_path = os.path.join(settings.relations_dir, f"{doc_id}_relations.json")
    rel_dicts = [
        {
            "relationship_id":     r.relationship_id,
            "source_assertion_id": r.source_assertion_id,
            "target_assertion_id": r.target_assertion_id,
            "relation_type":       r.relation_type.value,
            "is_bidirectional":    r.is_bidirectional,
            "is_truth_preserving": r.is_truth_preserving,
            "is_defeasible":       r.is_defeasible,
            "evidence_text":       r.evidence_text,
            "logical_form":        r.logical_form,
            "mechanism":           r.mechanism,
            "strength":            r.strength,
            "directionality":      r.directionality,
            "confidence":          r.confidence,
            "extraction_method":   r.extraction_method,
            "derivation_depth":    r.derivation_depth,
            "review_status":       r.review_status.value,
        }
        for r in relations
    ]

    with open(out_path, "w") as f:
        json.dump(rel_dicts, f, indent=2)

    if _should_mark_complete(len(assertions), len(relations)):
        checkpoint.mark_done(doc_id)
    else:
        checkpoint.mark_failed(doc_id, reason=f"zero relations for {len(assertions)}-assertion doc")

    logger.info(
        f"Relations: {doc_id} → {len(relations)} logical relations "
        f"({len(batches)} batches + {max(0, len(batches)-1)} cross-batch calls)"
    )


async def _run_async(doc_ids_override: List[str] = None) -> None:
    client = genai.Client(api_key=settings.google_api_key)

    checkpoint = CheckpointManager("stage3b_relations")
    already_done = checkpoint.get_completed_ids()

    if doc_ids_override is not None:
        work_queue = doc_ids_override
    else:
        assertion_files = [
            f.replace("_assertions.json", "")
            for f in os.listdir(settings.assertions_dir)
            if f.endswith("_assertions.json")
        ]
        work_queue = [doc_id for doc_id in assertion_files if doc_id not in already_done]

    checkpoint.set_total(len(already_done) + len(work_queue))

    os.makedirs(settings.relations_dir, exist_ok=True)

    logger.info(
        f"Stage 3b: {len(work_queue)} documents to process "
        f"({len(already_done)} already done)"
    )

    semaphore = asyncio.Semaphore(settings.pipeline_concurrency)

    try:
        tasks = [
            process_document(semaphore, client, doc_id, checkpoint)
            for doc_id in work_queue
        ]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 3b: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()

    # Load newly written relation files into PostgreSQL
    logger.info("Stage 3b: loading relations into PostgreSQL...")
    from pipeline.stage5_load import load_logical_relationships
    total_loaded = 0
    target_files = (
        {f"{doc_id}_relations.json" for doc_id in work_queue}
        if doc_ids_override
        else None
    )
    for filename in os.listdir(settings.relations_dir):
        if not filename.endswith("_relations.json"):
            continue
        if target_files and filename not in target_files:
            continue
        filepath = os.path.join(settings.relations_dir, filename)
        with open(filepath) as f:
            relations = json.load(f)
        if relations:
            load_logical_relationships(relations)
            total_loaded += len(relations)
    logger.info(f"Stage 3b: loaded {total_loaded} logical relationships into PostgreSQL")


def rerun_failed() -> None:
    """
    Identify documents with ≥ MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS assertions
    that currently have zero relations, remove them from the checkpoint's
    completed_ids, and rerun stage3b on those documents only.
    """
    logging.basicConfig(level=getattr(logging, settings.log_level))

    # Find failed docs: relation file exists but is empty, and doc has enough assertions
    failed_docs = []
    for fname in os.listdir(settings.relations_dir):
        if not fname.endswith("_relations.json"):
            continue
        doc_id = fname.replace("_relations.json", "")
        rel_path = os.path.join(settings.relations_dir, fname)
        with open(rel_path) as f:
            rels = json.load(f)
        if rels:
            continue  # already has relations — skip

        apath = os.path.join(settings.assertions_dir, f"{doc_id}_assertions.json")
        if not os.path.exists(apath):
            continue
        with open(apath) as f:
            assertions = json.load(f)
        if len(assertions) >= MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS:
            failed_docs.append(doc_id)

    logger.info(
        f"rerun_failed: {len(failed_docs)} documents with zero relations "
        f"and ≥{MIN_ASSERTIONS_FOR_REQUIRED_RELATIONS} assertions"
    )

    if not failed_docs:
        logger.info("rerun_failed: nothing to do")
        return

    # Remove failed docs from checkpoint so they re-enter the work queue
    ck_path = os.path.join(settings.checkpoints_dir, "stage3b_relations.json")
    with open(ck_path) as f:
        ck_data = json.load(f)
    failed_set = set(failed_docs)
    ck_data["completed_ids"] = [
        cid for cid in ck_data["completed_ids"] if cid not in failed_set
    ]
    ck_data["records_done"] = len(ck_data["completed_ids"])
    ck_data["status"] = "running"
    with open(ck_path, "w") as f:
        json.dump(ck_data, f, indent=2)

    logger.info(f"rerun_failed: removed {len(failed_docs)} docs from checkpoint")
    asyncio.run(_run_async(doc_ids_override=failed_docs))


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rerun-failed":
        rerun_failed()
    else:
        run()
