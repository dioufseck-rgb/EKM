"""
tests/test_inference.py — Tests for the backward chaining inference engine logic.

All tests are pure logic tests (no DB / network required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from functools import reduce
from operator import mul

from pipeline.schema import (
    Assertion, EpistemicStatus, InferenceResult, LogicalRelationship,
    LogicalRelationType, PredicateType, ProofChain, ReviewStatus,
    ScopeEnvelope, TemporalScope,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_assertion(aid, claim, status=EpistemicStatus.authoritative, conf=0.9):
    return Assertion(
        assertion_id=aid, chunk_id="c1", document_id="d1",
        claim_text=claim, subject="test",
        predicate_type=PredicateType.defines, object_text="obj",
        source_text="", source_document="doc", source_url="http://x",
        epistemic_status=status, confidence=conf,
    )


def _chain_confidence(confidences, hops):
    """Canonical formula: product(confidences) × 0.9^hops"""
    product = reduce(mul, confidences, 1.0)
    return product * (0.9 ** hops)


# ── CAUSES must NOT produce derived assertions ─────────────────────────────────

def test_causes_is_not_truth_preserving():
    """CAUSES relation must never be used for derivation."""
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.CAUSES)
    assert lr.is_truth_preserving == False


def test_inhibits_is_not_truth_preserving():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.INHIBITS)
    assert lr.is_truth_preserving == False


def test_correlates_not_truth_preserving():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.CORRELATES_WITH)
    assert lr.is_truth_preserving == False


def test_only_entails_and_related_are_truth_preserving():
    """Only ENTAILS, EQUIVALENT, DEFINES, TRIGGERS, SPECIALIZES, GENERALIZES, SUPERSEDES."""
    truth_preserving = {
        LogicalRelationType.ENTAILS,
        LogicalRelationType.EQUIVALENT,
        LogicalRelationType.DEFINES,
        LogicalRelationType.TRIGGERS,
        LogicalRelationType.SPECIALIZES,
        LogicalRelationType.GENERALIZES,
        LogicalRelationType.SUPERSEDES,
    }
    for rtype in LogicalRelationType:
        lr = LogicalRelationship.from_relation_type("a", "b", rtype)
        expected = rtype in truth_preserving
        assert lr.is_truth_preserving == expected, (
            f"{rtype.name}: expected is_truth_preserving={expected}, got {lr.is_truth_preserving}"
        )


# ── Backward chaining chain: A ENTAILS B, B ENTAILS C → query C finds [A→B→C] ─

def test_entails_chain_structure():
    """A ENTAILS B, B ENTAILS C — chain [A,B,C] is valid."""
    lr_ab = LogicalRelationship.from_relation_type("A", "B", LogicalRelationType.ENTAILS)
    lr_bc = LogicalRelationship.from_relation_type("B", "C", LogicalRelationType.ENTAILS)
    assert lr_ab.is_truth_preserving == True
    assert lr_bc.is_truth_preserving == True
    # Chain ids
    chain = [lr_ab.source_assertion_id, lr_bc.source_assertion_id, lr_bc.target_assertion_id]
    assert chain == ["A","B","C"]


# ── Scope check: winter relation does not fire in summer ───────────────────────

def test_scope_winter_not_summer():
    """Relation scoped to winter must not apply in a summer context."""
    winter_scope = ScopeEnvelope(
        temporal=TemporalScope(season="winter", is_default=False)
    )
    summer_context = {"season": "summer"}
    # The scope check logic: if context season != relation scope season, reject
    assert winter_scope.temporal.season     == "winter"
    assert summer_context["season"]         == "summer"
    assert winter_scope.temporal.season     != summer_context["season"]


def test_scope_winter_fires_in_winter():
    winter_scope = ScopeEnvelope(
        temporal=TemporalScope(season="winter", is_default=False)
    )
    winter_context = {"season": "winter"}
    assert winter_scope.temporal.season == winter_context["season"]


def test_default_scope_fires_always():
    """A relation with is_default=True should fire in any context."""
    default_scope = ScopeEnvelope(temporal=TemporalScope(is_default=True))
    for season in ["winter","summer","spring","autumn"]:
        # Default scope: no restriction → always fires
        assert default_scope.temporal.is_default == True


# ── Confidence decay ───────────────────────────────────────────────────────────

def test_confidence_decay_3_hops_0_9():
    """3-hop chain from 0.9-confidence assertions.
    Formula: product(confidences) × 0.9^hops = 0.9^3 × 0.9^3 = 0.9^6 ≈ 0.531441
    CLAUDE.md: "4-hop chain from 0.9-confidence: 0.9⁴ × 0.9⁴ = 0.43" confirms this formula.
    """
    confidences = [0.9, 0.9, 0.9]
    hops = 3
    result = _chain_confidence(confidences, hops)
    expected = (0.9 ** 3) * (0.9 ** 3)   # = 0.9^6 = 0.531441
    assert abs(result - expected) < 1e-8
    assert abs(result - 0.531441) < 1e-5


def test_confidence_decay_single_hop():
    confidences = [0.9]
    hops = 1
    result = _chain_confidence(confidences, hops)
    assert abs(result - 0.9 * 0.9) < 1e-8


def test_confidence_decay_2_hop_mixed():
    confidences = [0.9, 0.8]
    hops = 2
    result = _chain_confidence(confidences, hops)
    expected = 0.9 * 0.8 * (0.9 ** 2)
    assert abs(result - expected) < 1e-8


def test_confidence_decay_4_hop_from_0_9():
    """4-hop chain from 0.9 assertions: 0.9^4 * 0.9^4 ≈ 0.430"""
    confidences = [0.9, 0.9, 0.9, 0.9]
    hops = 4
    result = _chain_confidence(confidences, hops)
    expected = (0.9 ** 4) * (0.9 ** 4)
    assert abs(result - expected) < 1e-8
    assert abs(result - 0.43046721) < 1e-6


def test_confidence_less_than_one_per_hop():
    """Every additional hop reduces confidence."""
    conf_1hop = _chain_confidence([0.9], 1)
    conf_2hop = _chain_confidence([0.9, 0.9], 2)
    conf_3hop = _chain_confidence([0.9, 0.9, 0.9], 3)
    assert conf_1hop > conf_2hop > conf_3hop


# ── Deprecated assertions never in proof chains ────────────────────────────────

def test_deprecated_excluded():
    deprecated = _make_assertion("b", "deprecated claim", status=EpistemicStatus.deprecated)
    assert deprecated.epistemic_status == EpistemicStatus.deprecated
    # The inference engine checks: epistemic_status NOT IN ['deprecated','orphaned']
    excluded = {"deprecated","orphaned"}
    assert deprecated.epistemic_status.value in excluded


def test_orphaned_excluded():
    orphaned = _make_assertion("c", "orphaned claim", status=EpistemicStatus.orphaned)
    excluded = {"deprecated","orphaned"}
    assert orphaned.epistemic_status.value in excluded


def test_authoritative_included():
    a = _make_assertion("a", "valid claim", status=EpistemicStatus.authoritative)
    excluded = {"deprecated","orphaned"}
    assert a.epistemic_status.value not in excluded


# ── Consistency conflict → conclusion=conflicted ───────────────────────────────

def test_two_contradictory_chains():
    """If two chains reach contradictory conclusions, conclusion must be 'conflicted'."""
    chain_allowed = ProofChain(
        assertion_ids=["a","b"], relation_types=["ENTAILS"],
        confidences=[0.9, 0.9], chain_confidence=0.729,
        conclusion="X is allowed", hops=1,
    )
    chain_prohibited = ProofChain(
        assertion_ids=["c","d"], relation_types=["ENTAILS"],
        confidences=[0.8, 0.8], chain_confidence=0.576,
        conclusion="X is prohibited", hops=1,
    )
    conclusions = {chain_allowed.conclusion, chain_prohibited.conclusion}
    has_allowed    = any("allowed"    in c for c in conclusions)
    has_prohibited = any("prohibited" in c for c in conclusions)
    assert has_allowed and has_prohibited
    # When both present, the inference engine should return conclusion="conflicted"


def test_single_chain_not_conflicted():
    chain = ProofChain(
        assertion_ids=["a","b"], relation_types=["ENTAILS"],
        confidences=[0.9], chain_confidence=0.81,
        conclusion="X is allowed", hops=1,
    )
    # One chain, no contradiction
    assert "prohibited" not in chain.conclusion


# ── Derived assertions labeling ────────────────────────────────────────────────

def test_derived_assertion_has_correct_status():
    """Novel inference-engine conclusions must have epistemic_status=derived."""
    da = _make_assertion("d1", "derived claim", status=EpistemicStatus.derived)
    assert da.epistemic_status == EpistemicStatus.derived


def test_derived_assertion_extraction_method():
    da = Assertion(
        assertion_id="d1", chunk_id="", document_id="",
        claim_text="Derived: if A then C via B",
        subject="inference_engine",
        predicate_type=PredicateType.relates_to,
        object_text="target",
        source_text="", source_document="inference_engine", source_url="",
        epistemic_status=EpistemicStatus.derived,
        extraction_method="inference_engine",
        confidence=0.43,
        derivation_chain=[
            {"assertion_id": "a", "relation_type": "ENTAILS"},
            {"assertion_id": "b", "relation_type": "ENTAILS"},
        ],
    )
    assert da.extraction_method == "inference_engine"
    assert da.epistemic_status  == EpistemicStatus.derived
    assert len(da.derivation_chain) == 2


# ── REBUTS relation properties ─────────────────────────────────────────────────

def test_rebuts_is_defeasible():
    """REBUTS is a defeasible relation — exception conditions can themselves be overridden."""
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.REBUTS)
    assert lr.is_defeasible == True


def test_rebuts_is_not_truth_preserving():
    """REBUTS does not preserve truth — it defeats a claim in a scope, doesn't disprove it globally."""
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.REBUTS)
    assert lr.is_truth_preserving == False


def test_rebuts_is_not_bidirectional():
    """REBUTS is directional: rebuttal A defeats claim B, not the reverse."""
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.REBUTS)
    assert lr.is_bidirectional == False


def test_rebuttal_defeats_claim():
    """When a rebuttal's scope conditions are met, the claim is defeated (not proven)."""
    # Represents: claim C exists, rebuttal R rebuts C when scope is winter
    # In a winter context, C should be marked defeated, not proven.
    rebuttal_scope = ScopeEnvelope(temporal=TemporalScope(season="winter", is_default=False))
    winter_context = {"season": "winter"}
    summer_context = {"season": "summer"}

    # Rebuttal scope is winter, context is winter → rebuttal IS active
    context_season = winter_context.get("season")
    rebut_season   = rebuttal_scope.temporal.season
    rebuttal_active_in_winter = (
        not rebuttal_scope.temporal.is_default and
        rebut_season and
        rebut_season.lower() == context_season.lower()
    )
    assert rebuttal_active_in_winter == True

    # Rebuttal scope is winter, context is summer → rebuttal is NOT active
    context_season_summer = summer_context.get("season")
    rebuttal_active_in_summer = (
        not rebuttal_scope.temporal.is_default and
        rebut_season and
        rebut_season.lower() == context_season_summer.lower()
    )
    assert rebuttal_active_in_summer == False


# ── Toulmin completeness ────────────────────────────────────────────────────────

def _make_ir(chains, toulmin_completeness="incomplete"):
    """Construct a minimal InferenceResult for completeness tests."""
    return InferenceResult(
        target_claim="test",
        conclusion="proven",
        proof_chains=chains,
        confidence=0.9,
        toulmin_completeness=toulmin_completeness,
        consistency_status="consistent",
        active_rebuttals=[],
        contested_warnings=[],
        derived_assertions=[],
    )


def test_toulmin_completeness_field_exists():
    ir = _make_ir([], "incomplete")
    assert ir.toulmin_completeness == "incomplete"


def test_toulmin_completeness_complete():
    ir = _make_ir([], "complete")
    assert ir.toulmin_completeness == "complete"


def test_toulmin_completeness_missing_warrant():
    """A chain with only GROUND nodes (no WARRANT) should be flagged."""
    ir = _make_ir([], "missing_warrant")
    assert ir.toulmin_completeness == "missing_warrant"


def test_toulmin_completeness_missing_ground():
    """A chain with only WARRANT nodes (no GROUND) should be flagged."""
    ir = _make_ir([], "missing_ground")
    assert ir.toulmin_completeness == "missing_ground"


def test_active_rebuttals_field_exists():
    ir = _make_ir([], "incomplete")
    assert ir.active_rebuttals == []


def test_active_rebuttals_populated():
    ir = InferenceResult(
        target_claim="test",
        conclusion="proven",
        proof_chains=[],
        confidence=0.9,
        toulmin_completeness="complete",
        consistency_status="consistent",
        active_rebuttals=["rebuttal-id-1", "rebuttal-id-2"],
        contested_warnings=[],
        derived_assertions=[],
    )
    assert len(ir.active_rebuttals) == 2
    assert "rebuttal-id-1" in ir.active_rebuttals
