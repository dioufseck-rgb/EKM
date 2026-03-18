"""tests/test_scope.py — Tests for scope envelope serialization and semantics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date, time
from pipeline.schema import (
    CoverageType, CompletenessType, ConditionalScope, GeographicScope,
    OrganizationalScope, ScopeEnvelope, ScopeSource, TemporalScope,
    dict_to_scope, scope_to_dict,
)


# ── Round-trip serialization ───────────────────────────────────────────────────

def test_full_scope_roundtrip():
    scope = ScopeEnvelope(
        temporal=TemporalScope(
            season="winter",
            months=["Dec","Jan","Feb"],
            days_of_week=["Monday","Tuesday"],
            date_range_start=date(2025, 12, 1),
            date_range_end=date(2026, 2, 28),
            time_of_day_start=time(9, 0),
            time_of_day_end=time(17, 0),
            fiscal_period="Q4",
            is_default=False,
        ),
        geographic=GeographicScope(
            countries=["US","GB"],
            states=["CA","NY"],
            regions=["North America"],
            location_types=["branch"],
            is_global=False,
        ),
        organizational=OrganizationalScope(
            roles=["manager","teller"],
            business_units=["retail"],
            products=["savings"],
            customer_segments=["retail"],
            account_types=["checking"],
            is_universal=False,
        ),
        conditional=ConditionalScope(
            conditions=["balance > 10000"],
            thresholds={"max_withdrawal": 5000},
            prerequisites=["account_verified"],
            trigger_events=["large_transaction"],
        ),
        coverage=CoverageType.conditional_override,
        completeness=CompletenessType.explicit,
        source=ScopeSource.stated,
        reviewer_note="Seasonal branch hours",
    )

    d = scope_to_dict(scope)
    r = dict_to_scope(d)

    # Temporal
    assert r.temporal.season == "winter"
    assert r.temporal.months == ["Dec","Jan","Feb"]
    assert r.temporal.days_of_week == ["Monday","Tuesday"]
    assert r.temporal.date_range_start == date(2025, 12, 1)
    assert r.temporal.date_range_end   == date(2026, 2, 28)
    assert r.temporal.time_of_day_start == time(9, 0)
    assert r.temporal.time_of_day_end   == time(17, 0)
    assert r.temporal.fiscal_period == "Q4"
    assert r.temporal.is_default    == False

    # Geographic
    assert r.geographic.countries     == ["US","GB"]
    assert r.geographic.states        == ["CA","NY"]
    assert r.geographic.regions       == ["North America"]
    assert r.geographic.location_types == ["branch"]
    assert r.geographic.is_global     == False

    # Organizational
    assert r.organizational.roles             == ["manager","teller"]
    assert r.organizational.business_units    == ["retail"]
    assert r.organizational.products          == ["savings"]
    assert r.organizational.customer_segments == ["retail"]
    assert r.organizational.account_types     == ["checking"]
    assert r.organizational.is_universal      == False

    # Conditional
    assert r.conditional.conditions    == ["balance > 10000"]
    assert r.conditional.thresholds    == {"max_withdrawal": 5000}
    assert r.conditional.prerequisites == ["account_verified"]
    assert r.conditional.trigger_events == ["large_transaction"]

    # Meta
    assert r.coverage     == CoverageType.conditional_override
    assert r.completeness == CompletenessType.explicit
    assert r.source       == ScopeSource.stated
    assert r.reviewer_note == "Seasonal branch hours"


def test_empty_dict_returns_defaults():
    scope = dict_to_scope({})
    assert scope.temporal.is_default         == True
    assert scope.geographic.is_global        == True
    assert scope.organizational.is_universal == True
    assert scope.coverage                    == CoverageType.universal
    assert scope.completeness                == CompletenessType.unknown
    assert scope.source                      == ScopeSource.unknown


def test_partial_dict():
    d = {"temporal": {"season": "summer", "is_default": False}, "geographic": {}}
    scope = dict_to_scope(d)
    assert scope.temporal.season    == "summer"
    assert scope.temporal.is_default == False
    assert scope.geographic.is_global == True  # default preserved


# ── Scope semantics ────────────────────────────────────────────────────────────

def test_universal_scope_overlap():
    """Two universal scopes should both be global/universal."""
    s1 = ScopeEnvelope()
    s2 = ScopeEnvelope()
    assert s1.geographic.is_global    == True
    assert s2.geographic.is_global    == True
    assert s1.temporal.is_default     == True
    assert s2.temporal.is_default     == True


def test_temporal_scopes_different_seasons():
    s1 = ScopeEnvelope(temporal=TemporalScope(season="winter", is_default=False))
    s2 = ScopeEnvelope(temporal=TemporalScope(season="summer", is_default=False))
    # Different seasons means no overlap — this is SPECIALIZES not CONTRADICTS
    assert s1.temporal.season != s2.temporal.season


def test_geographic_scopes_no_overlap():
    s1 = ScopeEnvelope(geographic=GeographicScope(countries=["US"], is_global=False))
    s2 = ScopeEnvelope(geographic=GeographicScope(countries=["EU"], is_global=False))
    us = set(s1.geographic.countries)
    eu = set(s2.geographic.countries)
    assert us.isdisjoint(eu)


def test_coverage_type_values():
    assert CoverageType.universal.value            == "universal"
    assert CoverageType.default.value              == "default"
    assert CoverageType.conditional_override.value == "conditional_override"
    assert CoverageType.exception.value            == "exception"


def test_scope_source_values():
    assert {s.value for s in ScopeSource} == {"stated","inferred","unknown"}


def test_completeness_values():
    assert {c.value for c in CompletenessType} == {"explicit","partial","implicit","unknown"}
