"""
pipeline/inference.py — Backward chaining inference engine.

Answers: "given a target claim, can I prove or disprove it from the
assertion graph using only truth-preserving relations?"

Rules enforced:
- Only is_truth_preserving=True relations may be chained
- Never chain through deprecated or orphaned assertions
- Confidence decay: chain_conf = product(confidences) × 0.9^hops
- Consistency check: contradictory chains → conclusion=conflicted
- Novel derived assertions stored with epistemic_status=derived
"""
import json
import logging
import uuid
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple

import google.genai as genai

from pipeline.config import settings
from pipeline.db import db_cursor, get_neo4j_driver, get_qdrant_client
from pipeline.schema import (
    Assertion, EpistemicStatus, InferenceResult, PredicateType,
    ProofChain, ReviewStatus, ScopeEnvelope, dict_to_scope,
)

logger = logging.getLogger(__name__)

DEAD_STATUSES = {"deprecated", "orphaned"}


class InferenceEngine:
    """Backward chaining inference over the knowledge graph."""

    def __init__(self) -> None:
        self.driver  = get_neo4j_driver()
        self.qdrant  = get_qdrant_client()
        self._genai_client = genai.Client(api_key=settings.google_api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def prove(
        self,
        target_claim: str,
        max_depth: int = 4,
        context: Optional[dict] = None,
    ) -> InferenceResult:
        """
        Attempt to prove or disprove target_claim via backward chaining.

        Returns an InferenceResult with proof_chains, confidence,
        consistency_status, contested_warnings, and any derived_assertions.
        """
        # 1. Find semantically close assertions
        seeds = self._find_seed_assertions(target_claim, top_k=10)
        if not seeds:
            return InferenceResult(
                target_claim=target_claim,
                conclusion="inconclusive",
                proof_chains=[],
                confidence=0.0,
                toulmin_completeness="incomplete",
                consistency_status="incomplete",
                active_rebuttals=[],
                contested_warnings=[],
                derived_assertions=[],
            )

        contested_warnings: List[str] = []
        all_proof_chains:   List[ProofChain] = []
        derived_assertions: List[Assertion]  = []
        active_rebuttals:   List[str]        = []

        for seed_id, seed_claim, seed_status, seed_conf in seeds:
            if seed_status in DEAD_STATUSES:
                continue
            if seed_status == "contested":
                contested_warnings.append(
                    f"Warning: seed assertion {seed_id[:8]} has contested status: '{seed_claim[:80]}'"
                )

                # 2. Check for active REBUTS edges — if rebuttal scope is met, skip seed
            active_rebuttals_for_seed = self._check_rebuttals(seed_id, context or {})
            if active_rebuttals_for_seed:
                active_rebuttals.extend(active_rebuttals_for_seed)
                contested_warnings.append(
                    f"Claim {seed_id[:8]} is defeated by active rebuttal(s): "
                    + ", ".join(r[:8] for r in active_rebuttals_for_seed)
                )
                continue  # rebuttal defeats this seed — do not add to proof chains

            # 3. Traverse ENTAILS edges backward (find what entails this seed)
            chains = self._traverse_entails_backward(seed_id, max_depth)
            for chain_ids, rel_types, confidences, hops in chains:
                # Check context scope at each hop
                if context and not self._check_chain_scope(chain_ids, context):
                    continue

                chain_conf = self._compute_chain_confidence(confidences, hops)
                pc = ProofChain(
                    assertion_ids=chain_ids,
                    relation_types=rel_types,
                    confidences=confidences,
                    chain_confidence=chain_conf,
                    conclusion=f"supports: {seed_claim[:100]}",
                    hops=hops,
                )
                all_proof_chains.append(pc)

            # Also add the seed itself as a 0-hop chain
            all_proof_chains.append(ProofChain(
                assertion_ids=[seed_id],
                relation_types=[],
                confidences=[seed_conf],
                chain_confidence=seed_conf,
                conclusion=f"direct: {seed_claim[:100]}",
                hops=0,
            ))

        if not all_proof_chains:
            return InferenceResult(
                target_claim=target_claim,
                conclusion="inconclusive",
                proof_chains=[],
                confidence=0.0,
                toulmin_completeness="incomplete",
                consistency_status="incomplete",
                active_rebuttals=active_rebuttals,
                contested_warnings=contested_warnings,
                derived_assertions=[],
            )

        # 4. Check consistency
        consistency_status = self._check_consistency(all_proof_chains)
        conclusion = "conflicted" if consistency_status == "contradictory" else "proven"

        # 5. Check Toulmin completeness
        toulmin_completeness = self._check_toulmin_completeness(all_proof_chains)

        # 6. Compute overall confidence as max chain confidence
        overall_confidence = max(pc.chain_confidence for pc in all_proof_chains)

        # 7. Store derived assertions if novel chains produced new conclusions
        novel = self._produce_derived_assertions(target_claim, all_proof_chains, seeds)
        for da in novel:
            self._store_derived_assertion(da)
        derived_assertions.extend(novel)

        return InferenceResult(
            target_claim=target_claim,
            conclusion=conclusion,
            proof_chains=all_proof_chains[:20],  # cap for response size
            confidence=overall_confidence,
            toulmin_completeness=toulmin_completeness,
            consistency_status=consistency_status,
            active_rebuttals=active_rebuttals,
            contested_warnings=contested_warnings,
            derived_assertions=derived_assertions,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_seed_assertions(
        self, claim: str, top_k: int = 10
    ) -> List[Tuple[str, str, str, float]]:
        """Embed claim and find nearest assertions via Qdrant + PostgreSQL."""
        try:
            result = self._genai_client.models.embed_content(
                model=settings.embedding_model,
                contents=claim,
            )
            vector = result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        try:
            hits = self.qdrant.search(
                collection_name="finwiki_chunks",
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

        chunk_ids = [h.payload.get("chunk_id") for h in hits if h.payload]
        if not chunk_ids:
            return []

        placeholders = ",".join(["%s"] * len(chunk_ids))
        with db_cursor() as cur:
            cur.execute(
                f"""
                SELECT assertion_id, claim_text, epistemic_status, confidence
                FROM assertions
                WHERE chunk_id IN ({placeholders})
                AND epistemic_status NOT IN ('deprecated','orphaned')
                ORDER BY confidence DESC
                """,
                chunk_ids,
            )
            rows = cur.fetchall()

        return [(r[0], r[1], r[2], float(r[3])) for r in rows]

    def _traverse_entails_backward(
        self, target_id: str, max_depth: int
    ) -> List[Tuple[List[str], List[str], List[float], int]]:
        """
        Find all proof chains where the terminal assertion is target_id.
        Only truth-preserving edges; no deprecated/orphaned nodes.
        """
        chains = []
        with self.driver.session() as session:
            try:
                result = session.run(
                    f"""
                    MATCH path = (a:Assertion)-[:ENTAILS|DEFINES|TRIGGERS|SPECIALIZES|SUPERSEDES*1..{max_depth}]->(b:Assertion {{assertion_id: $id}})
                    WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
                    AND ALL(n IN nodes(path) WHERE NOT (n.epistemic_status IN ['deprecated','orphaned']))
                    RETURN [n IN nodes(path) | n.assertion_id] as chain,
                           [r IN relationships(path) | type(r)] as rels,
                           [r IN relationships(path) | coalesce(r.confidence, 0.8)] as confs,
                           length(path) as hops
                    ORDER BY hops ASC
                    LIMIT 20
                    """,
                    id=target_id,
                )
                for record in result:
                    chains.append((
                        record["chain"],
                        record["rels"],
                        [float(c) for c in record["confs"]],
                        record["hops"],
                    ))
            except Exception as e:
                logger.debug(f"Neo4j traversal error: {e}")

        return chains

    def _compute_chain_confidence(
        self, confidences: List[float], hops: int
    ) -> float:
        """chain_conf = product(confidences) × 0.9^hops"""
        if not confidences:
            return 0.0
        product = reduce(mul, confidences, 1.0)
        return product * (0.9 ** hops)

    def _check_chain_scope(self, chain_ids: List[str], context: dict) -> bool:
        """
        Check that no relation in the chain has a scope that excludes the context.
        Context keys: season, months, countries, roles, etc.
        """
        if not chain_ids or len(chain_ids) < 2:
            return True

        context_season = context.get("season")

        for i in range(len(chain_ids) - 1):
            src_id = chain_ids[i]
            tgt_id = chain_ids[i + 1]
            with db_cursor() as cur:
                cur.execute(
                    "SELECT scope FROM logical_relationships WHERE source_assertion_id=%s AND target_assertion_id=%s LIMIT 1",
                    [src_id, tgt_id],
                )
                row = cur.fetchone()
            if not row or not row[0]:
                continue

            scope_data = row[0] if isinstance(row[0], dict) else {}
            relation_scope = dict_to_scope(scope_data)

            # Temporal scope check
            if context_season and not relation_scope.temporal.is_default:
                if (relation_scope.temporal.season and
                        relation_scope.temporal.season.lower() != context_season.lower()):
                    return False  # relation is scoped to a different season

        return True

    def _check_consistency(self, proof_chains: List[ProofChain]) -> str:
        """
        Returns 'consistent', 'contradictory', or 'incomplete'.
        Checks for contradicting CONTRADICTS edges between assertions in chains.
        """
        if not proof_chains:
            return "incomplete"

        # Collect all assertion IDs involved in proof chains
        all_ids = set()
        for pc in proof_chains:
            all_ids.update(pc.assertion_ids)

        if len(all_ids) < 2:
            return "consistent"

        # Check for CONTRADICTS edges among these assertions
        id_list = list(all_ids)
        placeholders = ",".join(["%s"] * len(id_list))
        with db_cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) FROM assertion_relationships
                WHERE relationship_type = 'CONTRADICTS'
                AND source_assertion_id IN ({placeholders})
                AND target_assertion_id IN ({placeholders})
                AND review_status != 'false_positive'
                """,
                id_list + id_list,
            )
            contradictions = cur.fetchone()[0]

        return "contradictory" if contradictions > 0 else "consistent"

    def _check_rebuttals(self, assertion_id: str, context: dict) -> List[str]:
        """
        Query Neo4j for REBUTS edges targeting assertion_id.
        Returns list of rebuttal assertion_ids whose scope conditions are met
        in the current context.
        """
        active: List[str] = []
        context_season = context.get("season")
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    MATCH (r:Assertion)-[:REBUTS]->(c:Assertion {assertion_id: $id})
                    WHERE NOT (r.epistemic_status IN ['deprecated', 'orphaned'])
                    RETURN r.assertion_id AS rid, r.scope AS rscope
                    """,
                    id=assertion_id,
                )
                for record in result:
                    rid = record["rid"]
                    scope_raw = record.get("rscope") or "{}"
                    scope_data = json.loads(scope_raw) if isinstance(scope_raw, str) else (scope_raw or {})
                    rebuttal_scope = dict_to_scope(scope_data)
                    # If rebuttal is scoped to a season other than the current context, skip it
                    if context_season and not rebuttal_scope.temporal.is_default:
                        if (rebuttal_scope.temporal.season and
                                rebuttal_scope.temporal.season.lower() != context_season.lower()):
                            continue  # rebuttal not active in this context
                    active.append(rid)
            except Exception as e:
                logger.debug(f"Rebuttal check failed for {assertion_id}: {e}")
        return active

    def _check_toulmin_completeness(self, proof_chains: List[ProofChain]) -> str:
        """
        Check the best proof chain for Toulmin completeness.
        Returns: complete | missing_warrant | missing_ground | incomplete
        A complete argument requires at least one WARRANT (normative) and
        one GROUND (constative) node in the chain.
        """
        if not proof_chains:
            return "incomplete"
        best_chain = max(proof_chains, key=lambda pc: pc.chain_confidence)
        ids = best_chain.assertion_ids
        if not ids:
            return "incomplete"
        placeholders = ",".join(["%s"] * len(ids))
        try:
            with db_cursor() as cur:
                cur.execute(
                    f"SELECT discourse_role FROM assertions WHERE assertion_id IN ({placeholders})",
                    ids,
                )
                roles = {row[0] for row in cur.fetchall() if row[0]}
        except Exception as e:
            logger.debug(f"Toulmin completeness check failed: {e}")
            return "incomplete"
        has_warrant = "warrant" in roles
        has_ground  = "ground"  in roles
        if has_warrant and has_ground:
            return "complete"
        if not has_warrant and not has_ground:
            return "incomplete"
        return "missing_warrant" if not has_warrant else "missing_ground"

    def _produce_derived_assertions(
        self,
        target_claim: str,
        proof_chains: List[ProofChain],
        seeds: List[Tuple[str, str, str, float]],
    ) -> List[Assertion]:
        """
        If inference produces a novel conclusion not in any source document,
        store it as a derived assertion.
        Currently: only produce derived assertions when a multi-hop chain
        is found and the conclusion is not already an explicit assertion.
        """
        derived = []
        seed_claims = {s[1] for s in seeds}
        multi_hop = [pc for pc in proof_chains if pc.hops >= 2]

        for pc in multi_hop[:3]:  # limit derived assertion generation
            derived_claim = f"[Derived via {'-'.join(pc.relation_types)}] {pc.conclusion}"
            if derived_claim in seed_claims:
                continue

            da = Assertion(
                assertion_id    = str(uuid.uuid4()),
                chunk_id        = "",
                document_id     = "",
                claim_text      = derived_claim,
                subject         = "inference_engine",
                predicate_type  = PredicateType.relates_to,
                object_text     = target_claim,
                source_text     = "",
                source_document = "inference_engine",
                source_url      = "",
                epistemic_status = EpistemicStatus.derived,
                confidence       = pc.chain_confidence,
                extraction_method = "inference_engine",
                review_status    = ReviewStatus.pending,
                derivation_chain = [
                    {"assertion_id": aid, "relation_type": rel}
                    for aid, rel in zip(pc.assertion_ids, pc.relation_types + [""])
                ],
                derivation_confidence = pc.chain_confidence,
            )
            derived.append(da)

        return derived

    def _store_derived_assertion(self, assertion: Assertion) -> None:
        """Persist a derived assertion to PostgreSQL."""
        try:
            with db_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO assertions
                        (assertion_id, chunk_id, document_id, claim_text, subject, predicate_type,
                         object_text, source_text, source_document, source_url,
                         epistemic_status, confidence, extraction_method, review_status,
                         derivation_chain, derivation_confidence, domain)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        assertion.assertion_id, "", "", assertion.claim_text,
                        assertion.subject, assertion.predicate_type.value,
                        assertion.object_text, "", "inference_engine", "",
                        assertion.epistemic_status.value, assertion.confidence,
                        assertion.extraction_method, assertion.review_status.value,
                        json.dumps(assertion.derivation_chain),
                        assertion.derivation_confidence, "inference",
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to store derived assertion: {e}")
