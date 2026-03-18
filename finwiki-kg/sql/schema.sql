-- FinWiki Knowledge Graph — PostgreSQL Schema
-- All writes use ON CONFLICT DO NOTHING for idempotency

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Documents
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    domain          TEXT DEFAULT '',
    subdomain       TEXT DEFAULT '',
    authority_level TEXT DEFAULT 'reference',
    word_count      INT  DEFAULT 0,
    crawled_at      TIMESTAMPTZ,
    raw_file_path   TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);

-- 2. Chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    sequence        INT  NOT NULL,
    section_title   TEXT DEFAULT '',
    content         TEXT NOT NULL,
    token_estimate  INT  DEFAULT 0,
    chunk_file_path TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- 3. Assertions
CREATE TABLE IF NOT EXISTS assertions (
    assertion_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id                UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    document_id             TEXT REFERENCES documents(document_id) ON DELETE CASCADE,
    -- Claim
    claim_text              TEXT NOT NULL,
    subject                 TEXT DEFAULT '',
    predicate_type          TEXT DEFAULT '',
    object_text             TEXT DEFAULT '',
    object_value            FLOAT,
    object_unit             TEXT,
    -- Provenance
    source_text             TEXT DEFAULT '',
    source_document         TEXT DEFAULT '',
    source_url              TEXT DEFAULT '',
    source_section          TEXT DEFAULT '',
    authority_level         TEXT DEFAULT 'reference',
    effective_date          DATE,
    expiry_date             DATE,
    jurisdiction            TEXT,
    -- Epistemic
    epistemic_status        TEXT DEFAULT 'authoritative',
    confidence              FLOAT DEFAULT 1.0,
    extraction_method       TEXT DEFAULT 'llm',
    review_status           TEXT DEFAULT 'pending',
    derivation_chain        JSONB DEFAULT '[]',
    derivation_rule         TEXT,
    derivation_confidence   FLOAT,
    -- Semantic
    topics                  TEXT[] DEFAULT '{}',
    entities                TEXT[] DEFAULT '{}',
    regulations             TEXT[] DEFAULT '{}',
    keywords                TEXT[] DEFAULT '{}',
    domain                  TEXT DEFAULT '',
    -- Scope — Temporal
    temporal_season         TEXT,
    temporal_months         TEXT[] DEFAULT '{}',
    temporal_days_of_week   TEXT[] DEFAULT '{}',
    temporal_date_range_start DATE,
    temporal_date_range_end   DATE,
    temporal_time_of_day_start TIME,
    temporal_time_of_day_end   TIME,
    temporal_fiscal_period  TEXT,
    temporal_is_default     BOOLEAN DEFAULT TRUE,
    -- Scope — Geographic
    geo_countries           TEXT[] DEFAULT '{}',
    geo_states              TEXT[] DEFAULT '{}',
    geo_regions             TEXT[] DEFAULT '{}',
    geo_location_types      TEXT[] DEFAULT '{}',
    geo_is_global           BOOLEAN DEFAULT TRUE,
    -- Scope — Organizational
    org_roles               TEXT[] DEFAULT '{}',
    org_business_units      TEXT[] DEFAULT '{}',
    org_products            TEXT[] DEFAULT '{}',
    org_customer_segments   TEXT[] DEFAULT '{}',
    org_account_types       TEXT[] DEFAULT '{}',
    org_is_universal        BOOLEAN DEFAULT TRUE,
    -- Scope — Conditional
    cond_conditions         TEXT[] DEFAULT '{}',
    cond_thresholds         JSONB  DEFAULT '{}',
    cond_prerequisites      TEXT[] DEFAULT '{}',
    cond_trigger_events     TEXT[] DEFAULT '{}',
    -- Scope envelope meta
    scope_coverage          TEXT DEFAULT 'universal',
    scope_completeness      TEXT DEFAULT 'unknown',
    scope_source            TEXT DEFAULT 'unknown',
    scope_reviewer_note     TEXT,
    -- Discourse grammar (assigned by Stage 3c)
    discourse_role          TEXT NOT NULL DEFAULT 'unclassified',
    validity_claim_type     TEXT NOT NULL DEFAULT 'unclassified',
    -- Timestamps
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assertions_document_id     ON assertions(document_id);
CREATE INDEX IF NOT EXISTS idx_assertions_chunk_id        ON assertions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_assertions_domain          ON assertions(domain);
CREATE INDEX IF NOT EXISTS idx_assertions_epistemic_status ON assertions(epistemic_status);
CREATE INDEX IF NOT EXISTS idx_assertions_review_status   ON assertions(review_status);
CREATE INDEX IF NOT EXISTS idx_assertions_subject         ON assertions(subject);
CREATE INDEX IF NOT EXISTS idx_assertions_discourse_role  ON assertions(discourse_role);
CREATE INDEX IF NOT EXISTS idx_assertions_validity_type   ON assertions(validity_claim_type);

-- 4. Logical Relationships
CREATE TABLE IF NOT EXISTS logical_relationships (
    relationship_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_assertion_id     UUID NOT NULL REFERENCES assertions(assertion_id) ON DELETE CASCADE,
    target_assertion_id     UUID NOT NULL REFERENCES assertions(assertion_id) ON DELETE CASCADE,
    relation_type           TEXT NOT NULL,
    is_bidirectional        BOOLEAN DEFAULT FALSE,
    is_truth_preserving     BOOLEAN DEFAULT FALSE,
    is_defeasible           BOOLEAN DEFAULT FALSE,
    evidence_type           TEXT DEFAULT 'explicit',
    evidence_text           TEXT DEFAULT '',
    logical_form            TEXT DEFAULT '',
    mechanism               TEXT DEFAULT '',
    strength                TEXT DEFAULT '',
    directionality          TEXT DEFAULT 'A_to_B',
    scope                   JSONB DEFAULT '{}',
    confidence              FLOAT DEFAULT 1.0,
    extraction_method       TEXT DEFAULT 'llm_within_doc',
    derivation_depth        INT  DEFAULT 0,
    review_status           TEXT DEFAULT 'pending',
    validated_by            TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lr_source     ON logical_relationships(source_assertion_id);
CREATE INDEX IF NOT EXISTS idx_lr_target     ON logical_relationships(target_assertion_id);
CREATE INDEX IF NOT EXISTS idx_lr_type       ON logical_relationships(relation_type);
CREATE INDEX IF NOT EXISTS idx_lr_truth      ON logical_relationships(is_truth_preserving);

-- 5. Assertion Relationships (conflicts)
CREATE TABLE IF NOT EXISTS assertion_relationships (
    relationship_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_assertion_id     UUID NOT NULL REFERENCES assertions(assertion_id) ON DELETE CASCADE,
    target_assertion_id     UUID NOT NULL REFERENCES assertions(assertion_id) ON DELETE CASCADE,
    relationship_type       TEXT NOT NULL,  -- CONTRADICTS|SUPERSEDES|SPECIALIZES|DUPLICATE|COMPLEMENTARY|FALSE_POSITIVE
    explanation             TEXT DEFAULT '',
    conflicting_text        TEXT DEFAULT '',
    governing_assertion_id  UUID,
    reviewer_question       TEXT DEFAULT '',
    confidence              FLOAT DEFAULT 1.0,
    scope_overlap           JSONB DEFAULT '{}',
    review_status           TEXT DEFAULT 'pending',
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ar_source ON assertion_relationships(source_assertion_id);
CREATE INDEX IF NOT EXISTS idx_ar_target ON assertion_relationships(target_assertion_id);
CREATE INDEX IF NOT EXISTS idx_ar_type   ON assertion_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_ar_status ON assertion_relationships(review_status);

-- 6. Conflict Candidates (vector similarity pairs before adjudication)
CREATE TABLE IF NOT EXISTS conflict_candidates (
    candidate_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assertion_id_a  UUID NOT NULL,
    assertion_id_b  UUID NOT NULL,
    similarity_score FLOAT NOT NULL,
    adjudicated     BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (assertion_id_a, assertion_id_b)
);

CREATE INDEX IF NOT EXISTS idx_cc_adjudicated ON conflict_candidates(adjudicated);

-- 7. Conflict Items (human review queue)
CREATE TABLE IF NOT EXISTS conflict_items (
    conflict_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    relationship_id UUID NOT NULL REFERENCES assertion_relationships(relationship_id) ON DELETE CASCADE,
    priority        INT  DEFAULT 3,  -- 1=critical, 2=high, 3=medium, 4=low
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ci_priority ON conflict_items(priority);

-- 8. Qdrant ID Map
CREATE TABLE IF NOT EXISTS qdrant_id_map (
    chunk_id        UUID PRIMARY KEY,
    qdrant_point_id BIGINT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 9. LLM Cost Log
CREATE TABLE IF NOT EXISTS llm_cost_log (
    log_id              SERIAL PRIMARY KEY,
    model               TEXT NOT NULL,
    input_tokens        INT  NOT NULL,
    output_tokens       INT  NOT NULL,
    cost_usd            FLOAT NOT NULL,
    running_total_usd   FLOAT NOT NULL,
    stage               TEXT DEFAULT '',
    record_id           TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cost_stage ON llm_cost_log(stage);

-- 10. Pipeline Checkpoints
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    stage_name      TEXT PRIMARY KEY,
    status          TEXT DEFAULT 'pending',  -- pending|running|complete|failed|paused_cost_limit
    completed_ids   JSONB DEFAULT '[]',
    records_total   INT  DEFAULT 0,
    records_done    INT  DEFAULT 0,
    started_at      TIMESTAMPTZ,
    last_updated    TIMESTAMPTZ DEFAULT NOW()
);
