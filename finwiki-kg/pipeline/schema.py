"""
pipeline/schema.py — SOURCE OF TRUTH for all data models.

Never rename, remove, or reorder fields without updating every downstream consumer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, time, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────────────────

class PredicateType(Enum):
    defines     = "defines"
    requires    = "requires"
    prohibits   = "prohibits"
    permits     = "permits"
    constrains  = "constrains"
    classifies  = "classifies"
    implements  = "implements"
    supersedes  = "supersedes"
    relates_to  = "relates_to"
    causes      = "causes"
    governs     = "governs"


class EpistemicStatus(Enum):
    authoritative = "authoritative"
    inferred      = "inferred"
    contested     = "contested"
    deprecated    = "deprecated"
    orphaned      = "orphaned"
    derived       = "derived"


class ReviewStatus(Enum):
    pending   = "pending"
    validated = "validated"
    rejected  = "rejected"
    deferred  = "deferred"


class CoverageType(Enum):
    universal             = "universal"
    default               = "default"
    conditional_override  = "conditional_override"
    exception             = "exception"


class CompletenessType(Enum):
    explicit = "explicit"
    partial  = "partial"
    implicit = "implicit"
    unknown  = "unknown"


class ScopeSource(Enum):
    stated   = "stated"
    inferred = "inferred"
    unknown  = "unknown"


class LogicalRelationType(Enum):
    # Deductive (truth-preserving)
    ENTAILS        = "ENTAILS"
    CONTRADICTS    = "CONTRADICTS"
    EQUIVALENT     = "EQUIVALENT"
    # Probabilistic / empirical (defeasible)
    CAUSES         = "CAUSES"
    INHIBITS       = "INHIBITS"
    CORRELATES_WITH = "CORRELATES_WITH"
    # Structural / taxonomic
    SPECIALIZES    = "SPECIALIZES"
    GENERALIZES    = "GENERALIZES"
    INSTANTIATES   = "INSTANTIATES"
    CLASSIFIES     = "CLASSIFIES"
    # Definitional
    DEFINES        = "DEFINES"
    OPERATIONALIZES = "OPERATIONALIZES"
    # Temporal / sequential
    PRECEDES       = "PRECEDES"
    TRIGGERS       = "TRIGGERS"
    SUPERSEDES     = "SUPERSEDES"
    # Toulmin rebuttal — distinct from CONTRADICTS
    REBUTS         = "REBUTS"


class AssertionRelationshipType(Enum):
    CONTRADICTS   = "CONTRADICTS"
    SUPERSEDES    = "SUPERSEDES"
    SPECIALIZES   = "SPECIALIZES"
    DUPLICATE     = "DUPLICATE"
    COMPLEMENTARY = "COMPLEMENTARY"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class DerivationRule(Enum):
    modus_ponens   = "modus_ponens"
    transitivity   = "transitivity"
    conjunction    = "conjunction"
    specialization = "specialization"


class DiscourseRole(str, Enum):
    WARRANT      = "warrant"
    GROUND       = "ground"
    CLAIM        = "claim"
    BACKING      = "backing"
    REBUTTAL     = "rebuttal"
    QUALIFIER    = "qualifier"
    UNCLASSIFIED = "unclassified"


class ValidityClaimType(str, Enum):
    CONSTATIVE   = "constative"
    NORMATIVE    = "normative"
    EXPRESSIVE   = "expressive"
    UNCLASSIFIED = "unclassified"


# ─── Scope Dataclasses ────────────────────────────────────────────────────────

@dataclass
class TemporalScope:
    season:              Optional[str]       = None
    months:              List[str]           = field(default_factory=list)
    days_of_week:        List[str]           = field(default_factory=list)
    date_range_start:    Optional[date]      = None
    date_range_end:      Optional[date]      = None
    time_of_day_start:   Optional[time]      = None
    time_of_day_end:     Optional[time]      = None
    fiscal_period:       Optional[str]       = None
    is_default:          bool                = True


@dataclass
class GeographicScope:
    countries:     List[str] = field(default_factory=list)
    states:        List[str] = field(default_factory=list)
    regions:       List[str] = field(default_factory=list)
    location_types: List[str] = field(default_factory=list)
    is_global:     bool      = True


@dataclass
class OrganizationalScope:
    roles:             List[str] = field(default_factory=list)
    business_units:    List[str] = field(default_factory=list)
    products:          List[str] = field(default_factory=list)
    customer_segments: List[str] = field(default_factory=list)
    account_types:     List[str] = field(default_factory=list)
    is_universal:      bool      = True


@dataclass
class ConditionalScope:
    conditions:    List[str]       = field(default_factory=list)
    thresholds:    Dict[str, Any]  = field(default_factory=dict)
    prerequisites: List[str]       = field(default_factory=list)
    trigger_events: List[str]      = field(default_factory=list)


@dataclass
class ScopeEnvelope:
    temporal:     TemporalScope      = field(default_factory=TemporalScope)
    geographic:   GeographicScope    = field(default_factory=GeographicScope)
    organizational: OrganizationalScope = field(default_factory=OrganizationalScope)
    conditional:  ConditionalScope   = field(default_factory=ConditionalScope)
    coverage:     CoverageType       = CoverageType.universal
    completeness: CompletenessType   = CompletenessType.unknown
    source:       ScopeSource        = ScopeSource.unknown
    reviewer_note: Optional[str]     = None


# ─── Core Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class Assertion:
    # Identity
    chunk_id:     str
    document_id:  str
    # Claim
    claim_text:   str
    subject:      str
    predicate_type: PredicateType
    object_text:  str
    # Provenance
    source_text:      str
    source_document:  str
    source_url:       str
    # Generated fields with defaults
    assertion_id:     str  = field(default_factory=lambda: str(uuid.uuid4()))
    object_value:     Optional[float] = None
    object_unit:      Optional[str]   = None
    # Scope
    scope:            ScopeEnvelope   = field(default_factory=ScopeEnvelope)
    # Provenance continued
    source_section:   str             = ""
    authority_level:  str             = "reference"
    effective_date:   Optional[date]  = None
    expiry_date:      Optional[date]  = None
    jurisdiction:     Optional[str]   = None
    # Epistemic status
    epistemic_status:      EpistemicStatus  = EpistemicStatus.authoritative
    confidence:            float            = 1.0
    extraction_method:     str              = "llm"
    review_status:         ReviewStatus     = ReviewStatus.pending
    derivation_chain:      List[Dict]       = field(default_factory=list)
    derivation_rule:       Optional[DerivationRule] = None
    derivation_confidence: Optional[float]  = None
    # Discourse grammar (assigned by Stage 3c)
    discourse_role:      DiscourseRole      = DiscourseRole.UNCLASSIFIED
    validity_claim_type: ValidityClaimType  = ValidityClaimType.UNCLASSIFIED
    # Semantic
    topics:      List[str] = field(default_factory=list)
    entities:    List[str] = field(default_factory=list)
    regulations: List[str] = field(default_factory=list)
    keywords:    List[str] = field(default_factory=list)
    domain:      str       = ""
    # Relationships (denormalized)
    contradicts:     Optional[List[str]] = None
    supersedes:      Optional[List[str]] = None
    superseded_by:   Optional[List[str]] = None
    specializes:     Optional[List[str]] = None
    specialized_by:  Optional[List[str]] = None
    requires:        Optional[List[str]] = None
    enables:         Optional[List[str]] = None
    entails:         Optional[List[str]] = None
    entailed_by:     Optional[List[str]] = None
    causes:          Optional[List[str]] = None
    caused_by:       Optional[List[str]] = None
    triggered_by:    Optional[List[str]] = None
    triggers:        Optional[List[str]] = None
    defines:         Optional[List[str]] = None
    defined_by:      Optional[List[str]] = None
    correlates_with: Optional[List[str]] = None


@dataclass
class LogicalRelationship:
    source_assertion_id: str
    target_assertion_id: str
    relation_type:       LogicalRelationType

    relationship_id:     str  = field(default_factory=lambda: str(uuid.uuid4()))
    is_bidirectional:    bool = False
    is_truth_preserving: bool = False
    is_defeasible:       bool = False
    evidence_type:       str  = "explicit"
    evidence_text:       str  = ""
    logical_form:        str  = ""
    mechanism:           str  = ""
    strength:            str  = ""
    directionality:      str  = "A_to_B"
    scope:               ScopeEnvelope = field(default_factory=ScopeEnvelope)
    confidence:          float         = 1.0
    extraction_method:   str           = "llm_within_doc"
    derivation_depth:    int           = 0
    review_status:       ReviewStatus  = ReviewStatus.pending
    validated_by:        Optional[str] = None

    @classmethod
    def from_relation_type(
        cls,
        source_id: str,
        target_id: str,
        rtype: LogicalRelationType,
        **kwargs
    ) -> "LogicalRelationship":
        """
        Create a LogicalRelationship with is_bidirectional, is_truth_preserving,
        and is_defeasible automatically set based on the relation type.
        """
        _TRUTH_PRESERVING = {
            LogicalRelationType.ENTAILS,
            LogicalRelationType.EQUIVALENT,
            LogicalRelationType.DEFINES,
            LogicalRelationType.TRIGGERS,
            LogicalRelationType.SPECIALIZES,
            LogicalRelationType.GENERALIZES,
            LogicalRelationType.SUPERSEDES,
        }
        _DEFEASIBLE = {
            LogicalRelationType.CAUSES,
            LogicalRelationType.INHIBITS,
            LogicalRelationType.CORRELATES_WITH,
            LogicalRelationType.REBUTS,
        }
        _BIDIRECTIONAL = {
            LogicalRelationType.CONTRADICTS,
            LogicalRelationType.EQUIVALENT,
            LogicalRelationType.CORRELATES_WITH,
        }
        return cls(
            source_assertion_id=source_id,
            target_assertion_id=target_id,
            relation_type=rtype,
            is_truth_preserving=rtype in _TRUTH_PRESERVING,
            is_defeasible=rtype in _DEFEASIBLE,
            is_bidirectional=rtype in _BIDIRECTIONAL,
            **kwargs,
        )


@dataclass
class AssertionRelationship:
    source_assertion_id: str
    target_assertion_id: str
    relationship_type:   AssertionRelationshipType

    relationship_id:        str  = field(default_factory=lambda: str(uuid.uuid4()))
    explanation:            str  = ""
    conflicting_text:       str  = ""
    governing_assertion_id: Optional[str] = None
    reviewer_question:      str  = ""
    confidence:             float = 1.0
    scope_overlap:          Dict[str, Any] = field(default_factory=dict)
    review_status:          ReviewStatus   = ReviewStatus.pending


@dataclass
class Document:
    title: str
    url:   str

    document_id:    str               = field(default_factory=lambda: str(uuid.uuid4()))
    domain:         str               = ""
    subdomain:      str               = ""
    authority_level: str              = "reference"
    word_count:     int               = 0
    crawled_at:     Optional[datetime] = None
    raw_file_path:  str               = ""


@dataclass
class Chunk:
    document_id: str
    sequence:    int
    content:     str

    chunk_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    section_title:  str = ""
    token_estimate: int = 0
    chunk_file_path: str = ""


@dataclass
class ConflictItem:
    relationship: AssertionRelationship

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority:    int = 3


@dataclass
class ProofChain:
    assertion_ids:    List[str]
    relation_types:   List[str]
    confidences:      List[float]
    chain_confidence: float
    conclusion:       str
    hops:             int


@dataclass
class InferenceResult:
    target_claim:         str
    conclusion:           str               # proven | refuted | inconclusive | conflicted
    proof_chains:         List[ProofChain]
    confidence:           float
    toulmin_completeness: str               # complete | missing_warrant | missing_ground | incomplete
    consistency_status:   str               # consistent | contradictory | incomplete
    active_rebuttals:     List[str]         = field(default_factory=list)   # assertion_ids
    contested_warnings:   List[str]         = field(default_factory=list)
    derived_assertions:   List[Assertion]   = field(default_factory=list)


# ─── Scope Serialization Utilities ────────────────────────────────────────────

def scope_to_dict(scope: ScopeEnvelope) -> dict:
    """Convert ScopeEnvelope to a flat dict for JSON / JSONB storage."""
    def _date(d):
        return d.isoformat() if d else None

    def _time(t):
        return t.isoformat() if t else None

    return {
        "temporal": {
            "season":             scope.temporal.season,
            "months":             scope.temporal.months,
            "days_of_week":       scope.temporal.days_of_week,
            "date_range_start":   _date(scope.temporal.date_range_start),
            "date_range_end":     _date(scope.temporal.date_range_end),
            "time_of_day_start":  _time(scope.temporal.time_of_day_start),
            "time_of_day_end":    _time(scope.temporal.time_of_day_end),
            "fiscal_period":      scope.temporal.fiscal_period,
            "is_default":         scope.temporal.is_default,
        },
        "geographic": {
            "countries":     scope.geographic.countries,
            "states":        scope.geographic.states,
            "regions":       scope.geographic.regions,
            "location_types": scope.geographic.location_types,
            "is_global":     scope.geographic.is_global,
        },
        "organizational": {
            "roles":             scope.organizational.roles,
            "business_units":    scope.organizational.business_units,
            "products":          scope.organizational.products,
            "customer_segments": scope.organizational.customer_segments,
            "account_types":     scope.organizational.account_types,
            "is_universal":      scope.organizational.is_universal,
        },
        "conditional": {
            "conditions":    scope.conditional.conditions,
            "thresholds":    scope.conditional.thresholds,
            "prerequisites": scope.conditional.prerequisites,
            "trigger_events": scope.conditional.trigger_events,
        },
        "coverage":      scope.coverage.value,
        "completeness":  scope.completeness.value,
        "source":        scope.source.value,
        "reviewer_note": scope.reviewer_note,
    }


def dict_to_scope(d: dict) -> ScopeEnvelope:
    """Reconstruct ScopeEnvelope from a dict."""
    if not d:
        return ScopeEnvelope()

    def _date(s):
        return date.fromisoformat(s) if s else None

    def _time(s):
        return time.fromisoformat(s) if s else None

    t = d.get("temporal", {})
    g = d.get("geographic", {})
    o = d.get("organizational", {})
    c = d.get("conditional", {})

    return ScopeEnvelope(
        temporal=TemporalScope(
            season=t.get("season"),
            months=t.get("months", []),
            days_of_week=t.get("days_of_week", []),
            date_range_start=_date(t.get("date_range_start")),
            date_range_end=_date(t.get("date_range_end")),
            time_of_day_start=_time(t.get("time_of_day_start")),
            time_of_day_end=_time(t.get("time_of_day_end")),
            fiscal_period=t.get("fiscal_period"),
            is_default=t.get("is_default", True),
        ),
        geographic=GeographicScope(
            countries=g.get("countries", []),
            states=g.get("states", []),
            regions=g.get("regions", []),
            location_types=g.get("location_types", []),
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
        coverage=CoverageType(d.get("coverage", "universal")),
        completeness=CompletenessType(d.get("completeness", "unknown")),
        source=ScopeSource(d.get("source", "unknown")),
        reviewer_note=d.get("reviewer_note"),
    )
