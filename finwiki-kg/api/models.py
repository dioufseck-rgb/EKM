"""api/models.py — Pydantic request/response models for all API endpoints."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    domain: Optional[str] = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_hops: int = Field(default=3, ge=1, le=6)
    include_contested: bool = True
    include_derived: bool = True


class ProveRequest(BaseModel):
    claim: str
    max_depth: int = Field(default=4, ge=1, le=6)
    context: Optional[Dict[str, Any]] = None


class ConflictResolution(str, Enum):
    validated     = "validated"
    false_positive = "false_positive"
    deferred      = "deferred"


class ConflictUpdateRequest(BaseModel):
    resolution: ConflictResolution
    reviewer_note: Optional[str] = None


class AssertionResponse(BaseModel):
    assertion_id:    str
    claim_text:      str
    subject:         str
    predicate_type:  str
    object_text:     str
    source_document: str
    source_url:      str
    epistemic_status: str
    confidence:      float
    domain:          str
    derivation_chain: List[Dict] = []
    derivation_depth: int = 0


class LogicalEdgeResponse(BaseModel):
    relation_type:          str
    source_assertion_id:    str
    target_assertion_id:    str
    is_truth_preserving:    bool
    is_defeasible:          bool
    evidence_text:          str
    mechanism:              Optional[str] = None
    strength:               Optional[str] = None
    confidence:             float
    hop:                    int


class ReasoningTraceItem(BaseModel):
    assertion:         AssertionResponse
    relation_used:     Optional[str] = None
    hop_distance:      int = 0
    chain_confidence:  float


class CausalContextItem(BaseModel):
    assertion:    AssertionResponse
    relation_type: str  # CAUSES | INHIBITS | CORRELATES_WITH
    mechanism:    Optional[str] = None
    strength:     Optional[str] = None
    note:         str = "Supporting context (not derived)"


class QueryResponse(BaseModel):
    answer:             str
    reasoning_trace:    List[ReasoningTraceItem]
    causal_context:     List[CausalContextItem]
    sources:            List[AssertionResponse]
    consistency_status: str
    contested_warnings: List[str] = []
    total_cost_usd:     float
    regulation_context: Optional["RegulationAnchoredContext"] = None


class ProofChainResponse(BaseModel):
    assertion_ids:    List[str]
    relation_types:   List[str]
    confidences:      List[float]
    chain_confidence: float
    conclusion:       str
    hops:             int


class ProveResponse(BaseModel):
    target_claim:       str
    conclusion:         str  # proven | refuted | inconclusive | conflicted
    proof_chains:       List[ProofChainResponse]
    confidence:         float
    consistency_status: str
    contested_warnings: List[str]
    derived_assertions: List[AssertionResponse]


class SearchResult(BaseModel):
    assertion:       AssertionResponse
    similarity_score: float


class ScopeOverlapGrid(BaseModel):
    temporal:      str = "unknown"  # overlaps | no_overlap | unknown
    geographic:    str = "unknown"
    organizational: str = "unknown"
    conditional:   str = "unknown"


class ConflictCard(BaseModel):
    conflict_id:       str
    relationship_id:   str
    relationship_type: str
    priority:          int
    assertion_a:       AssertionResponse
    assertion_b:       AssertionResponse
    explanation:       str
    conflicting_text:  str
    reviewer_question: str
    scope_overlap:     ScopeOverlapGrid
    confidence:        float
    review_status:     str


class ConflictsResponse(BaseModel):
    items:     List[ConflictCard]
    total:     int
    page:      int
    page_size: int


class GraphAssertionResponse(BaseModel):
    assertion:              AssertionResponse
    logical_relationships:  List[LogicalEdgeResponse]
    conflict_relationships: List[Dict]


class NeighborhoodResponse(BaseModel):
    center: AssertionResponse
    nodes:  List[Dict]
    edges:  List[Dict]


class RegulationAnchoredAssertion(BaseModel):
    assertion_id:        str
    claim_text:          str
    source_document:     str
    discourse_role:      str
    validity_claim_type: str
    confidence:          float
    retrieval_path:      str  # "regulation_anchor" or "toulmin_expansion"


class RegulationAnchoredContext(BaseModel):
    regulation_name:  str
    warrant_layer:    List["RegulationAnchoredAssertion"]
    ground_layer:     List["RegulationAnchoredAssertion"]
    backing_layer:    List["RegulationAnchoredAssertion"]
    source_documents: List[str]


class StatsResponse(BaseModel):
    pipeline_stages:             List[Dict]
    total_documents:             int
    total_chunks:                int
    total_assertions:            int
    total_logical_relationships: int
    total_conflicts:             int
    total_cost_usd:              float
    pending_conflicts:           int
