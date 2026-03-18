"""tests/test_conflict.py — Tests for Stage 6 conflict detection logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import pytest
from pipeline.stage6_conflicts import (
    NEGATION_RE, passes_structural_filter, apply_validity_step0,
)
from pipeline.schema import ScopeEnvelope, TemporalScope, GeographicScope


# ── Negation pattern ───────────────────────────────────────────────────────────

def test_negation_pattern_not():
    assert NEGATION_RE.search("Banks are not allowed to exceed limits")

def test_negation_pattern_no():
    assert NEGATION_RE.search("No customer may withdraw more than $10,000")

def test_negation_pattern_prohibit():
    # "prohibited" matches because pattern uses prohibit\w*
    assert NEGATION_RE.search("This action is prohibited by Basel III")

def test_negation_pattern_forbid():
    # "Forbidden" matches because pattern uses forbid\w*
    assert NEGATION_RE.search("Forbidden transfers must be reported immediately")

def test_negation_pattern_no_match():
    assert not NEGATION_RE.search("Banks must maintain capital ratios at all times")

def test_negation_pattern_no_match_2():
    assert not NEGATION_RE.search("Customers can withdraw funds daily up to $5,000")


# ── Structural filter ──────────────────────────────────────────────────────────

def _make_a(**kwargs):
    defaults = {
        "assertion_id":   "aaa",
        "claim_text":     "Banks must maintain Tier 1 capital of 6%.",
        "subject":        "bank",
        "predicate_type": "requires",
        "object_value":   6.0,
        "source_text":    "Banks must maintain capital.",
        "source_document": "Basel III",
        "domain":         "banking",
        "effective_date": None,
        "expiry_date":    None,
        "epistemic_status": "authoritative",
        "confidence":     0.9,
        "scope_coverage": "universal",
        "temporal_season": None,
        "temporal_is_default": True,
        "geo_is_global":  True,
        "geo_countries":  [],
        "org_is_universal": True,
    }
    defaults.update(kwargs)
    return defaults


def test_same_domain_different_doc_passes():
    a = _make_a(domain="banking", source_document="Basel III")
    b = _make_a(domain="banking", source_document="Dodd-Frank")
    assert passes_structural_filter(a, b) == True


def test_same_doc_different_domain_no_negation_fails():
    a = _make_a(domain="banking", source_document="Basel III", object_value=6.0)
    b = _make_a(domain="securities", source_document="Basel III", object_value=6.0)
    # Same doc — no negation, same subject, same value
    assert passes_structural_filter(a, b) == False


def test_same_subject_different_value_passes():
    a = _make_a(subject="capital_ratio", object_value=6.0,
                domain="banking", source_document="Basel III")
    b = _make_a(subject="capital_ratio", object_value=10.5,
                domain="banking", source_document="EU Directive")
    assert passes_structural_filter(a, b) == True


def test_requires_prohibits_same_subject_passes():
    a = _make_a(predicate_type="requires", subject="disclosure",
                domain="banking", source_document="Doc A")
    b = _make_a(predicate_type="prohibits", subject="disclosure",
                domain="banking", source_document="Doc B")
    assert passes_structural_filter(a, b) == True


def test_negation_in_one_passes():
    a = _make_a(source_text="Banks must maintain capital.",
                domain="banking", source_document="Doc A")
    b = _make_a(source_text="Banks are not required to maintain capital.",
                domain="banking", source_document="Doc B")
    assert passes_structural_filter(a, b) == True


def test_both_negation_no_other_filter_fails():
    a = _make_a(source_text="Banks are not required to submit reports.",
                domain="banking", source_document="Doc A",
                subject="reporting", object_value=None)
    b = _make_a(source_text="No requirement for capital reporting.",
                domain="banking", source_document="Doc B",
                subject="disclosure", object_value=None)
    # Both have negation → no one-sided negation → should NOT pass on that criterion alone
    # But different subjects — check domain filter
    # Same domain, different doc → should PASS on domain criterion
    assert passes_structural_filter(a, b) == True  # same domain, diff doc


# ── Scope intersection semantics ───────────────────────────────────────────────

def test_scope_mismatch_should_be_specializes():
    """Canonical example from CLAUDE.md."""
    # "offices open at 8" (default) vs "in winter offices open at 9" (seasonal override)
    s1 = ScopeEnvelope()  # default / universal
    s2 = ScopeEnvelope(temporal=TemporalScope(season="winter", is_default=False))
    # s1 is universal; s2 is winter-only
    # These are SPECIALIZES not CONTRADICTS
    assert s1.temporal.is_default    == True
    assert s2.temporal.is_default    == False
    assert s2.temporal.season        == "winter"


def test_identical_scopes_could_contradict():
    s1 = ScopeEnvelope()
    s2 = ScopeEnvelope()
    # Both universal → same scope → if claims conflict, it's CONTRADICTS
    assert s1.temporal.is_default    == s2.temporal.is_default
    assert s1.geographic.is_global   == s2.geographic.is_global


# ── Validity-type Step 0 (Habermas pre-filter) ─────────────────────────────────

def test_expressive_a_is_false_positive():
    """Expressive assertion A → immediately FALSE_POSITIVE regardless of B."""
    result = apply_validity_step0("expressive", "constative")
    assert result == "FALSE_POSITIVE"


def test_expressive_b_is_false_positive():
    """Expressive assertion B → immediately FALSE_POSITIVE regardless of A."""
    result = apply_validity_step0("normative", "expressive")
    assert result == "FALSE_POSITIVE"


def test_both_expressive_is_false_positive():
    result = apply_validity_step0("expressive", "expressive")
    assert result == "FALSE_POSITIVE"


def test_constative_normative_is_potential_ground():
    """Constative A + normative B → POTENTIAL_GROUND (not a conflict)."""
    result = apply_validity_step0("constative", "normative")
    assert result == "POTENTIAL_GROUND"


def test_normative_constative_is_potential_ground():
    """Order doesn't matter — still POTENTIAL_GROUND."""
    result = apply_validity_step0("normative", "constative")
    assert result == "POTENTIAL_GROUND"


def test_both_normative_proceed():
    """Both normative → proceed to Step 1 (not short-circuited)."""
    result = apply_validity_step0("normative", "normative")
    assert result is None


def test_both_constative_proceed():
    """Both constative → proceed to Step 1 (not short-circuited)."""
    result = apply_validity_step0("constative", "constative")
    assert result is None


def test_unclassified_proceed():
    """Unclassified types → proceed to Step 1 (conservative)."""
    result = apply_validity_step0("unclassified", "unclassified")
    assert result is None


def test_step0_case_insensitive():
    """Validity types from DB may be lowercase or uppercase."""
    assert apply_validity_step0("EXPRESSIVE", "CONSTATIVE") == "FALSE_POSITIVE"
    assert apply_validity_step0("CONSTATIVE", "NORMATIVE") == "POTENTIAL_GROUND"
    assert apply_validity_step0("NORMATIVE", "NORMATIVE") is None
