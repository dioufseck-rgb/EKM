"""
pipeline/stage3_classify.py — Extract atomic assertions from chunks via Gemini Flash.

Concurrency: 5 parallel LLM calls.
Every LLM call goes through tracker.record().
"""
import asyncio
import json
import logging
import os
import time
import uuid
from typing import List

import google.genai as genai
from google.genai import types as genai_types

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.cost_tracker import CostLimitReached, tracker
from pipeline.schema import (
    Assertion, DiscourseRole, EpistemicStatus, PredicateType, ReviewStatus,
    ScopeEnvelope, TemporalScope, GeographicScope, OrganizationalScope,
    ConditionalScope, CoverageType, CompletenessType, ScopeSource,
    ValidityClaimType,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are extracting atomic factual assertions from a financial services document.

Document title: {title}
Chunk text:
{chunk_text}

Extract ALL atomic assertions — the smallest units that stand independently.
For each assertion:
1. claim_text: complete sentence stating the assertion
2. subject: entity the claim is about
3. predicate_type: one of [defines, requires, prohibits, permits, constrains, classifies, implements, supersedes, causes, governs, relates_to]
   - relates_to: LAST RESORT ONLY. Use only when no other predicate fits.
     If a sentence does multiple things simultaneously, break it into
     multiple assertions with specific predicates rather than using
     relates_to to cover the whole sentence. A high rate of relates_to
     in your output means you are not trying hard enough.
4. object_text: what is claimed about the subject
5. object_value: numeric value if present, else null
6. object_unit: unit (%, $, days, etc.) if present, else null
7. Ask: does this hold always, or only under specific conditions?
   - temporal: season, months, days, date ranges, time of day, fiscal period
   - geographic: countries, states, regions, location types
   - organizational: roles, business units, products, customer segments, account types
   - conditional: conditions, thresholds, prerequisites, trigger events
8. scope_coverage: "universal" | "default" | "conditional_override" | "exception"
9. scope_completeness: "explicit" | "partial" | "implicit" | "unknown"
   NEVER mark "explicit" unless the scope is fully stated in the text.
10. scope_source: "stated" | "inferred" | "unknown"
11. authority_level: "regulatory" | "policy" | "guidance" | "reference"
12. confidence: 0.0-1.0
13. topics: list of topic labels (e.g. ["capital_adequacy", "basel"])
14. entities: list of entity names (organizations, regulations, instruments)
15. regulations: list of specific regulation names cited
16. domain: one of [banking, securities, insurance, derivatives, regulatory, accounting, finance]

ATOMICITY RULE:
Each assertion must express exactly one claim. If the source sentence
contains "and" or "," connecting distinct claims, split into separate
assertions. Never use relates_to to avoid splitting.

Example — wrong:
  claim_text: "The ISA program helps Level 2 merchants meet Mastercard
               requirements and was created by PCI SSC."
  predicate_type: relates_to

Example — correct:
  assertion 1: claim_text: "The ISA program helps Level 2 merchants meet
                             Mastercard compliance requirements."
               predicate_type: implements
  assertion 2: claim_text: "The ISA program was created by PCI SSC."
               predicate_type: governs

DEDUPLICATION RULE:
Do not emit two assertions with identical or near-identical claim_text
from the same chunk, even with different predicate_types.
Pick the most specific predicate and emit once.

Return a JSON array of assertions. No markdown fences. No preamble. Example:
[{{"claim_text":"Banks must maintain a minimum Tier 1 capital ratio of 6%.",
  "subject":"bank","predicate_type":"requires","object_text":"minimum Tier 1 capital ratio of 6%",
  "object_value":6.0,"object_unit":"%",
  "scope":{{"temporal":{{"is_default":true}},"geographic":{{"is_global":true}},
            "organizational":{{"is_universal":true}},"conditional":{{}}}},
  "scope_coverage":"universal","scope_completeness":"explicit","scope_source":"stated",
  "authority_level":"regulatory","confidence":0.95,
  "topics":["capital_adequacy"],"entities":["Basel III"],"regulations":["Basel III"],
  "domain":"banking"}}]
"""

PREDICATE_RANK = {
    "defines": 1, "requires": 1, "prohibits": 1, "permits": 1,
    "constrains": 1, "classifies": 1, "implements": 1,
    "supersedes": 1, "causes": 1, "governs": 1, "relates_to": 99,
}


def dedup_assertions(assertions: List[Assertion]) -> List[Assertion]:
    """Remove near-duplicate assertions within the same chunk, keeping the most specific predicate."""
    seen = {}  # (chunk_id, normalized_claim_120_chars) -> assertion
    for a in assertions:
        key = (a.chunk_id, a.claim_text.strip().lower()[:120])
        if key not in seen:
            seen[key] = a
        else:
            existing_rank = PREDICATE_RANK.get(seen[key].predicate_type.value, 99)
            new_rank = PREDICATE_RANK.get(a.predicate_type.value, 99)
            if new_rank < existing_rank:
                seen[key] = a
    return list(seen.values())


def _parse_response(raw: str, chunk_id: str, document_id: str, doc_title: str) -> List[Assertion]:
    """Parse LLM JSON response into Assertion objects."""
    raw = raw.strip()
    # Strip potential markdown fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("```").strip()

    items = json.loads(raw)
    assertions = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # Build scope
        s = item.get("scope", {})
        t = s.get("temporal", {})
        g = s.get("geographic", {})
        o = s.get("organizational", {})
        c = s.get("conditional", {})

        scope = ScopeEnvelope(
            temporal=TemporalScope(
                season=t.get("season"),
                months=t.get("months", []),
                days_of_week=t.get("days_of_week", []),
                is_default=t.get("is_default", True),
                fiscal_period=t.get("fiscal_period"),
            ),
            geographic=GeographicScope(
                countries=g.get("countries", []),
                states=g.get("states", []),
                regions=g.get("regions", []),
                is_global=g.get("is_global", True),
            ),
            organizational=OrganizationalScope(
                roles=o.get("roles", []),
                business_units=o.get("business_units", []),
                products=o.get("products", []),
                customer_segments=o.get("customer_segments", []),
                account_types=o.get("account_types", []),
                is_universal=o.get("is_universal", True),
            ),
            conditional=ConditionalScope(
                conditions=c.get("conditions", []),
                thresholds=c.get("thresholds", {}),
                prerequisites=c.get("prerequisites", []),
                trigger_events=c.get("trigger_events", []),
            ),
        )

        # Normalize scope_coverage — LLM sometimes returns "conditional" instead of "conditional_override"
        raw_coverage = item.get("scope_coverage", "universal")
        if raw_coverage == "conditional":
            raw_coverage = "conditional_override"
        try:
            coverage = CoverageType(raw_coverage)
        except ValueError:
            coverage = CoverageType.universal

        try:
            completeness = CompletenessType(item.get("scope_completeness", "unknown"))
        except ValueError:
            completeness = CompletenessType.unknown

        try:
            scope_source = ScopeSource(item.get("scope_source", "unknown"))
        except ValueError:
            scope_source = ScopeSource.unknown

        scope = ScopeEnvelope(
            temporal=scope.temporal,
            geographic=scope.geographic,
            organizational=scope.organizational,
            conditional=scope.conditional,
            coverage=coverage,
            completeness=completeness,
            source=scope_source,
        )

        try:
            predicate = PredicateType(item.get("predicate_type", "relates_to"))
        except ValueError:
            predicate = PredicateType.relates_to

        assertions.append(Assertion(
            assertion_id    = str(uuid.uuid4()),
            chunk_id        = chunk_id,
            document_id     = document_id,
            claim_text      = item.get("claim_text", ""),
            subject         = item.get("subject", ""),
            predicate_type  = predicate,
            object_text     = item.get("object_text", ""),
            object_value    = item.get("object_value"),
            object_unit     = item.get("object_unit"),
            scope           = scope,
            source_text     = item.get("claim_text", ""),
            source_document = doc_title,
            source_url      = "",
            authority_level = item.get("authority_level", "reference"),
            epistemic_status = EpistemicStatus.authoritative,
            confidence      = float(item.get("confidence", 0.8)),
            extraction_method = "llm",
            review_status   = ReviewStatus.pending,
            topics          = item.get("topics", []),
            entities        = item.get("entities", []),
            regulations     = item.get("regulations", []),
            keywords        = item.get("keywords", []),
            domain          = item.get("domain", "finance"),
        ))

    return assertions


async def classify_chunk(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
    chunk: dict,
    document_id: str,
) -> List[Assertion]:
    """Classify one chunk — up to 3 retries on JSONDecodeError."""
    chunk_id   = chunk["chunk_id"]
    chunk_text = chunk["content"][:3000]  # cap at 3,000 chars
    doc_title  = chunk.get("title", document_id)

    prompt = PROMPT_TEMPLATE.format(title=doc_title, chunk_text=chunk_text)

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
                # Estimate tokens (rough: 1.3 chars per token)
                input_tokens  = int(len(prompt) / 4)
                output_tokens = int(len(raw) / 4)
                tracker.record(
                    model=settings.flash_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stage="stage3_classify",
                    record_id=chunk_id,
                )
                return _parse_response(raw, chunk_id, document_id, doc_title)

            except json.JSONDecodeError as e:
                wait = 2 ** attempt
                logger.warning(f"JSONDecodeError on chunk {chunk_id[:8]} attempt {attempt+1}: {e}. Retrying in {wait}s")
                await asyncio.sleep(wait)
            except CostLimitReached:
                raise
            except Exception as e:
                logger.error(f"LLM error on chunk {chunk_id[:8]}: {e}")
                return []

    logger.error(f"All retries failed for chunk {chunk_id[:8]} — skipping")
    return []


async def classify_document(
    semaphore: asyncio.Semaphore,
    client: genai.Client,
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

    document_id = chunks[0].get("document_id", filename)

    tasks = [
        classify_chunk(semaphore, client, chunk, document_id)
        for chunk in chunks
    ]

    all_assertions: List[Assertion] = []
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            all_assertions.extend(result)
        except CostLimitReached:
            raise

    # Deduplicate before writing
    all_assertions = dedup_assertions(all_assertions)

    # Serialize
    out_path = os.path.join(settings.assertions_dir, f"{document_id}_assertions.json")
    assertion_dicts = []
    for a in all_assertions:
        d = {
            "assertion_id":       a.assertion_id,
            "chunk_id":           a.chunk_id,
            "document_id":        a.document_id,
            "claim_text":         a.claim_text,
            "subject":            a.subject,
            "predicate_type":     a.predicate_type.value,
            "object_text":        a.object_text,
            "object_value":       a.object_value,
            "object_unit":        a.object_unit,
            "source_text":        a.source_text,
            "source_document":    a.source_document,
            "source_url":         a.source_url,
            "authority_level":    a.authority_level,
            "epistemic_status":   a.epistemic_status.value,
            "confidence":         a.confidence,
            "extraction_method":  a.extraction_method,
            "review_status":      a.review_status.value,
            "topics":             a.topics,
            "entities":           a.entities,
            "regulations":        a.regulations,
            "keywords":           a.keywords,
            "domain":             a.domain,
            "scope": {
                "temporal":      {"is_default": a.scope.temporal.is_default,
                                  "season": a.scope.temporal.season,
                                  "months": a.scope.temporal.months},
                "geographic":    {"is_global": a.scope.geographic.is_global,
                                  "countries": a.scope.geographic.countries},
                "organizational":{"is_universal": a.scope.organizational.is_universal,
                                  "roles": a.scope.organizational.roles},
                "conditional":   {"conditions": a.scope.conditional.conditions,
                                  "thresholds": a.scope.conditional.thresholds},
                "coverage":      a.scope.coverage.value,
                "completeness":  a.scope.completeness.value,
                "source":        a.scope.source.value,
            },
        }
        assertion_dicts.append(d)

    with open(out_path, "w") as f:
        json.dump(assertion_dicts, f, indent=2)

    checkpoint.mark_done(filename)
    logger.info(f"Classified: {filename} → {len(all_assertions)} assertions")


async def _run_async() -> None:
    client = genai.Client(api_key=settings.google_api_key)

    checkpoint = CheckpointManager("stage3_classify")
    already_done = checkpoint.get_completed_ids()

    chunk_files = [f for f in os.listdir(settings.chunks_dir) if f.endswith(".json")]
    work_queue  = [f for f in chunk_files if f not in already_done]
    checkpoint.set_total(len(chunk_files))

    os.makedirs(settings.assertions_dir, exist_ok=True)

    logger.info(f"Stage 3: {len(work_queue)} chunk files to classify ({len(already_done)} done)")

    semaphore = asyncio.Semaphore(settings.pipeline_concurrency)

    try:
        tasks = [
            classify_document(semaphore, client, fn, checkpoint)
            for fn in work_queue
        ]
        await asyncio.gather(*tasks)
    except CostLimitReached as e:
        logger.warning(f"Stage 3: cost limit reached — {e}")
        checkpoint.set_status("paused_cost_limit")
        return

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
