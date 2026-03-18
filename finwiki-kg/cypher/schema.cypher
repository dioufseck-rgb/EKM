// FinWiki Knowledge Graph — Neo4j Schema
// All constraints and indexes for the FinWiki knowledge graph
// Run once on startup. All stage8_graph.py writes use MERGE not CREATE.

// ─── Uniqueness Constraints ───────────────────────────────────────────────────

CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document)   REQUIRE d.document_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)      REQUIRE c.chunk_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Assertion)  REQUIRE a.assertion_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept)    REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regulation) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)      REQUIRE t.name IS UNIQUE;

// ─── Indexes ──────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.domain);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.epistemic_status);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.subject);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.predicate_type);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.confidence);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.review_status);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.discourse_role);
CREATE INDEX IF NOT EXISTS FOR (a:Assertion) ON (a.validity_claim_type);
CREATE INDEX IF NOT EXISTS FOR (d:Document)  ON (d.domain);
CREATE INDEX IF NOT EXISTS FOR (d:Document)  ON (d.authority_level);

// ─── Inference Cypher Patterns ────────────────────────────────────────────────

// Finding all assertions entailed by A (transitive closure, max 4 hops):
//
// MATCH path = (a:Assertion {assertion_id: $id})-[:ENTAILS*1..4]->(b:Assertion)
// WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
// AND ALL(n IN nodes(path) WHERE n.epistemic_status NOT IN ['deprecated','orphaned'])
// RETURN b, length(path) as hops,
//        reduce(c=1.0, r IN relationships(path) | c * r.confidence * 0.9) as chain_confidence
// ORDER BY chain_confidence DESC

// Finding backward entailment chain (what entails a given assertion):
//
// MATCH path = (a:Assertion)-[:ENTAILS*1..4]->(b:Assertion {assertion_id: $id})
// WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
// AND ALL(n IN nodes(path) WHERE n.epistemic_status NOT IN ['deprecated','orphaned'])
// RETURN [n IN nodes(path) | n.assertion_id] as chain,
//        [r IN relationships(path) | type(r)] as relations,
//        [r IN relationships(path) | r.confidence] as confidences,
//        length(path) as hops
// ORDER BY hops ASC

// Finding causal chains (non-derivable, for context only):
//
// MATCH path = (a:Assertion)-[:CAUSES|INHIBITS*1..3]->(b:Assertion)
// RETURN path

// Finding all assertions for a concept cluster (for Stage 7):
//
// MATCH (c:Concept {name: $concept_name})<-[:GOVERNS]-(a:Assertion)
// RETURN a.assertion_id, a.claim_text, a.document_id, a.confidence
// ORDER BY a.confidence DESC

// Rebuttal check — must be called before surfacing any claim as proven:
//
// MATCH (r:Assertion)-[:REBUTS]->(c:Assertion {assertion_id: $claim_id})
// WHERE r.epistemic_status NOT IN ['deprecated','orphaned']
// RETURN r.assertion_id, r.claim_text, r.scope

// Detecting contradictions in scope-overlapping assertions:
//
// MATCH (a:Assertion)-[r:CONTRADICTS]-(b:Assertion)
// WHERE r.review_status = 'pending'
// RETURN a, b, r
// ORDER BY r.confidence DESC

// N-hop neighborhood exploration:
//
// MATCH path = (a:Assertion {assertion_id: $id})-[*1..2]-(b)
// RETURN nodes(path), relationships(path)
// LIMIT 100

// Derived assertion chain visualization:
//
// MATCH (a:Assertion {epistemic_status: 'derived'})
// MATCH path = (src:Assertion)-[:ENTAILS*1..4]->(a)
// WHERE ALL(r IN relationships(path) WHERE r.is_truth_preserving = true)
// RETURN path
// LIMIT 20
