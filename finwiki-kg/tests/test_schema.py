"""tests/test_schema.py — Unit tests for pipeline/schema.py data models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.schema import (
    Assertion, AssertionRelationship, AssertionRelationshipType,
    CoverageType, CompletenessType, DerivationRule, EpistemicStatus,
    InferenceResult, LogicalRelationship, LogicalRelationType,
    PredicateType, ProofChain, ReviewStatus, ScopeEnvelope,
    TemporalScope, GeographicScope, OrganizationalScope, ConditionalScope,
    ScopeSource, dict_to_scope, scope_to_dict,
)


# ── Assertion ──────────────────────────────────────────────────────────────────

def _make_assertion(**kwargs) -> Assertion:
    defaults = dict(
        chunk_id="c1", document_id="d1",
        claim_text="Banks must maintain Tier 1 capital of 6%.",
        subject="bank", predicate_type=PredicateType.requires,
        object_text="Tier 1 capital of 6%",
        source_text="", source_document="Basel III", source_url="http://example.com",
    )
    defaults.update(kwargs)
    return Assertion(**defaults)


def test_assertion_generates_uuid():
    a = _make_assertion()
    assert len(a.assertion_id) == 36
    assert a.assertion_id.count("-") == 4


def test_assertion_defaults():
    a = _make_assertion()
    assert a.epistemic_status == EpistemicStatus.authoritative
    assert a.review_status    == ReviewStatus.pending
    assert a.confidence       == 1.0
    assert a.derivation_chain == []
    assert a.derivation_rule  is None


def test_assertion_unique_ids():
    a1 = _make_assertion()
    a2 = _make_assertion()
    assert a1.assertion_id != a2.assertion_id


def test_assertion_scope_defaults():
    a = _make_assertion()
    assert a.scope.temporal.is_default    == True
    assert a.scope.geographic.is_global   == True
    assert a.scope.organizational.is_universal == True
    assert a.scope.coverage               == CoverageType.universal


# ── PredicateType ──────────────────────────────────────────────────────────────

def test_all_predicate_types():
    expected = {
        "defines","requires","prohibits","permits","constrains",
        "classifies","implements","supersedes","relates_to","causes","governs",
    }
    assert {p.value for p in PredicateType} == expected


# ── LogicalRelationship ────────────────────────────────────────────────────────

def test_entails_properties():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.ENTAILS)
    assert lr.is_truth_preserving == True
    assert lr.is_defeasible       == False
    assert lr.is_bidirectional    == False


def test_causes_properties():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.CAUSES)
    assert lr.is_truth_preserving == False
    assert lr.is_defeasible       == True
    assert lr.is_bidirectional    == False


def test_equivalent_properties():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.EQUIVALENT)
    assert lr.is_truth_preserving == True
    assert lr.is_bidirectional    == True
    assert lr.is_defeasible       == False


def test_contradicts_bidirectional():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.CONTRADICTS)
    assert lr.is_bidirectional    == True
    assert lr.is_truth_preserving == False


def test_correlates_with_properties():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.CORRELATES_WITH)
    assert lr.is_defeasible    == True
    assert lr.is_bidirectional == True


def test_inhibits_properties():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.INHIBITS)
    assert lr.is_defeasible       == True
    assert lr.is_truth_preserving == False


def test_defines_truth_preserving():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.DEFINES)
    assert lr.is_truth_preserving == True
    assert lr.is_defeasible       == False


def test_triggers_truth_preserving():
    lr = LogicalRelationship.from_relation_type("a", "b", LogicalRelationType.TRIGGERS)
    assert lr.is_truth_preserving == True


def test_all_logical_relation_types_exist():
    expected = {
        "ENTAILS","CONTRADICTS","EQUIVALENT","CAUSES","INHIBITS","CORRELATES_WITH",
        "SPECIALIZES","GENERALIZES","INSTANTIATES","CLASSIFIES","DEFINES",
        "OPERATIONALIZES","PRECEDES","TRIGGERS","SUPERSEDES","REBUTS",
    }
    assert {r.name for r in LogicalRelationType} == expected


# ── ScopeEnvelope ──────────────────────────────────────────────────────────────

def test_scope_envelope_defaults():
    scope = ScopeEnvelope()
    assert scope.temporal.is_default         == True
    assert scope.geographic.is_global        == True
    assert scope.organizational.is_universal == True
    assert scope.coverage    == CoverageType.universal
    assert scope.completeness == CompletenessType.unknown
    assert scope.source       == ScopeSource.unknown


def test_coverage_types_complete():
    assert {c.value for c in CoverageType} == {
        "universal","default","conditional_override","exception"
    }


# ── Scope serialization ────────────────────────────────────────────────────────

def test_scope_to_dict_and_back():
    from datetime import date
    scope = ScopeEnvelope(
        temporal=TemporalScope(
            season="winter", months=["Dec","Jan","Feb"],
            is_default=False,
        ),
        geographic=GeographicScope(countries=["US","CA"], is_global=False),
        organizational=OrganizationalScope(roles=["teller"], is_universal=False),
        coverage=CoverageType.conditional_override,
        completeness=CompletenessType.explicit,
        source=ScopeSource.stated,
    )
    d = scope_to_dict(scope)
    restored = dict_to_scope(d)
    assert restored.temporal.season         == "winter"
    assert restored.temporal.months         == ["Dec","Jan","Feb"]
    assert restored.temporal.is_default     == False
    assert restored.geographic.countries    == ["US","CA"]
    assert restored.geographic.is_global    == False
    assert restored.organizational.roles    == ["teller"]
    assert restored.coverage                == CoverageType.conditional_override
    assert restored.completeness            == CompletenessType.explicit
    assert restored.source                  == ScopeSource.stated


def test_dict_to_scope_empty():
    scope = dict_to_scope({})
    assert scope.temporal.is_default         == True
    assert scope.geographic.is_global        == True
    assert scope.organizational.is_universal == True


# ── EpistemicStatus ────────────────────────────────────────────────────────────

def test_epistemic_status_values():
    expected = {"authoritative","inferred","contested","deprecated","orphaned","derived"}
    assert {e.value for e in EpistemicStatus} == expected


# ── ProofChain / InferenceResult ───────────────────────────────────────────────

def test_proof_chain_fields():
    pc = ProofChain(
        assertion_ids=["a","b","c"],
        relation_types=["ENTAILS","DEFINES"],
        confidences=[0.9, 0.8],
        chain_confidence=0.52,
        conclusion="proven",
        hops=2,
    )
    assert len(pc.assertion_ids) == 3
    assert pc.hops == 2
    assert pc.chain_confidence == 0.52


def test_inference_result_fields():
    ir = InferenceResult(
        target_claim="test claim",
        conclusion="inconclusive",
        proof_chains=[],
        confidence=0.0,
        toulmin_completeness="incomplete",
        consistency_status="incomplete",
        contested_warnings=[],
        derived_assertions=[],
    )
    assert ir.conclusion == "inconclusive"
    assert ir.derived_assertions == []
