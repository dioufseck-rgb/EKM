# FinWiki Evaluation Harness Specification

Build a complete evaluation harness for the FinWiki discourse-typed retrieval architecture. The harness must run overnight and produce statistics sufficient to support or falsify four theoretical propositions about epistemic type alignment and regulatory compliance retrieval.

**This harness must run to completion without any human input or intervention.** Never use `input()` or any blocking prompt. Make all design decisions autonomously, log them to `data/eval/eval_log.txt`, and continue. Wrap every query in a try/except — log failures and move on, never exit early. Retry database failures 3 times with 10-second backoff before skipping. Write incremental results to `data/eval/results_partial.json` every 10 queries so progress survives a kill signal. Before the full run, execute a 3-query smoke test across all four conditions, log SMOKE TEST PASSED or FAILED with reason, then proceed automatically without waiting for review. Always write `results.json` and `results_summary.txt` at the end with whatever data was collected, marking failed queries clearly. The harness starts, runs, and finishes on its own.

---

## Architecture Overview

Four retrieval conditions to compare:

- **Condition A:** Standard vector RAG — top-k chunk retrieval from Qdrant, no filtering, no graph traversal
- **Condition B:** Entity GraphRAG — top-k assertion retrieval from Qdrant without validity-type filtering, with generic neighbor expansion on any edge type
- **Condition C:** Validity-gated only — top-k assertion retrieval filtered by validity-claim type, no Toulmin expansion, no regulation anchor
- **Condition D:** Full discourse-typed — validity-gated + Toulmin neighborhood expansion + regulation-anchored cross-document assembly

---

## Query Generation — 150 Queries

Generate queries from the corpus using the following procedure:

```python
QUERY_GEN_PROMPT = """
You are generating evaluation queries for a retrieval system benchmark.

Given this financial document, generate exactly 3 queries:
1. One NORMATIVE query — asks what something requires, mandates, prohibits, or permits
2. One CONSTATIVE query — asks what something is, how something works, or what causes something
3. One MIXED query — requires both normative and constative content to answer completely

Document title: {title}
Document excerpt: {excerpt}

Return JSON only:
{{"normative": "...", "constative": "...", "mixed": "..."}}
"""
```

Sample 50 normative, 50 constative, 50 mixed from the generated pool. Stratify to ensure at least 10 queries per document type:

- Regulatory instrument articles (AIFMD, Basel III, Dodd-Frank, MiFID)
- Risk measurement articles (VaR, concentration risk, stress testing)
- Asset pricing articles (CAPM, APT, Black-Scholes)
- Financial product articles (derivatives, swaps, options)

Save all 150 queries to `data/eval/queries.json` with fields: `query_id`, `query_text`, `epistemic_type`, `source_doc`, `generation_method`.

---

## Outcome Measures — Four Metrics

### Metric 1: Validity-Type Precision (VTP)

For each retrieved assertion, check whether its `validity_claim_type` matches the expected type for the query:

- Normative queries: expected type = `normative`
- Constative queries: expected type = `constative`
- Mixed queries: expected = at least one normative AND at least one constative in top-10

```python
def validity_type_precision(retrieved_assertions, query_epistemic_type, k=10):
    """
    Returns proportion of top-k assertions matching expected validity type.
    For mixed queries: returns proportion of normative + proportion of constative separately.
    """
```

This is a pure metadata lookup — no LLM judge needed.

### Metric 2: Argumentative Completeness (AC)

For each retrieved set, check what proportion of the seed assertion's Toulmin neighborhood is present. Neighborhood is defined as all assertions reachable within 2 hops via ENTAILS, CAUSES, TRIGGERS, SPECIALIZES edges from the highest-confidence seed assertion.

```python
def argumentative_completeness(retrieved_assertions, seed_assertion_id, graph_driver, max_hops=2):
    """
    Returns: components_present / total_neighborhood_size
    If neighborhood size is 0 (isolated assertion), returns None — exclude from AC analysis.
    """
```

This is a graph query — no LLM judge needed.

### Metric 3: Context Window Efficiency (CWE)

For each condition, find the minimum k at which argumentative completeness reaches 0.8. Report median k across all queries with non-null AC.

```python
def context_window_efficiency(query_results_by_k, threshold=0.8):
    """
    Sweep k from 1 to 50. Return minimum k where AC >= threshold.
    If never reached, return 50 (ceiling).
    """
```

### Metric 4: Conflict Surface Rate (CSR)

For each retrieved set, check whether any CONTRADICTS edge exists between any two assertions in the retrieved set.

```python
def conflict_surface_rate(retrieved_assertion_ids, graph_driver):
    """
    Returns 1 if any CONTRADICTS edge exists between retrieved assertions, 0 otherwise.
    Aggregate: proportion of queries where conflict is surfaced.
    """
```

This is a graph query — no LLM judge needed.

---

## Statistical Analysis

Run the following tests on each metric pair (Condition A vs D, B vs D, C vs D):

```python
from scipy import stats
import numpy as np

def run_statistical_tests(scores_baseline, scores_treatment, metric_name, condition_pair):
    """
    1. Wilcoxon signed-rank test (paired, non-parametric — appropriate for ordinal scores)
    2. Cohen's d effect size
    3. Rank-biserial correlation (effect size for Wilcoxon)
    4. Mean and median for each condition
    5. 95% confidence interval via bootstrap (1000 iterations)

    Report: p-value, effect size, CI, and interpretation:
    - d < 0.2: negligible
    - d 0.2-0.5: small
    - d 0.5-0.8: medium
    - d > 0.8: large
    """
```

### Subgroup Analysis 1: By Epistemic Type

Run all statistical tests separately for normative / constative / mixed query strata. This tests the moderating condition in Propositions 3 and 4 — effect sizes should be largest for normative queries.

### Subgroup Analysis 2: By Corpus Repetitiveness

For each query, compute a repetitiveness index as the mean cosine similarity of the top-20 Qdrant results. Split queries into high/low repetitiveness strata at the median. Report effect sizes separately for each stratum. Effect sizes should be larger in high-repetitiveness strata.

---

## K-Invariance Demonstration

For Metric 1 (VTP) on normative queries, sweep k from 1 to 100 for Conditions A and D. Save the curves to `data/eval/k_invariance_vtp.json`. This demonstrates the plateau in standard RAG validity precision — the empirical proof of k-invariance for validity-type routing.

For Metric 4 (CSR), confirm zero conflict surface rate for Conditions A and B at all k values tested (k = 5, 10, 20, 50, 100). Save to `data/eval/k_invariance_csr.json`.

---

## Output Format

Save results to `data/eval/results.json`:

```json
{
  "run_timestamp": "...",
  "n_queries": 150,
  "n_failed": 0,
  "query_distribution": {"normative": 50, "constative": 50, "mixed": 50},
  "results_by_condition": {
    "A": {"vtp_mean": null, "vtp_median": null, "ac_mean": null, "cwe_median": null, "csr_rate": null},
    "B": {"vtp_mean": null, "vtp_median": null, "ac_mean": null, "cwe_median": null, "csr_rate": null},
    "C": {"vtp_mean": null, "vtp_median": null, "ac_mean": null, "cwe_median": null, "csr_rate": null},
    "D": {"vtp_mean": null, "vtp_median": null, "ac_mean": null, "cwe_median": null, "csr_rate": null}
  },
  "statistical_tests": {
    "A_vs_D": {
      "vtp": {"wilcoxon_p": null, "cohens_d": null, "rank_biserial": null, "ci_95": [null, null]},
      "ac":  {"wilcoxon_p": null, "cohens_d": null, "rank_biserial": null, "ci_95": [null, null]},
      "cwe": {"wilcoxon_p": null, "cohens_d": null, "rank_biserial": null, "ci_95": [null, null]},
      "csr": {"wilcoxon_p": null, "cohens_d": null, "rank_biserial": null, "ci_95": [null, null]}
    },
    "B_vs_D": {},
    "C_vs_D": {}
  },
  "subgroup_by_epistemic_type": {
    "normative": {},
    "constative": {},
    "mixed": {}
  },
  "subgroup_by_repetitiveness": {
    "high": {},
    "low": {}
  },
  "k_invariance_curves": {
    "vtp_normative_A": [],
    "vtp_normative_D": [],
    "csr_A_by_k": {},
    "csr_B_by_k": {}
  },
  "failed_queries": []
}
```

Save a human-readable summary to `data/eval/results_summary.txt` stating in plain language whether each proposition is supported, partially supported, or not supported based on the statistical results, using the following template:

```
FINWIKI EVALUATION HARNESS — RESULTS SUMMARY
=============================================
Run timestamp: ...
Queries completed: ... / 150
Queries failed: ...

PROPOSITION 1 (Epistemic type mismatch → compliance risk): [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
  VTP Condition A: ... vs Condition D: ... | p=... | d=...
  Interpretation: ...

PROPOSITION 2 (Argumentative truncation → compliance risk): [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
  AC Condition A: ... vs Condition D: ... | p=... | d=...
  CWE Condition A: k=... vs Condition D: k=... | p=... | d=...
  Interpretation: ...

PROPOSITION 3 (Validity-gated filtering → compliance risk reduction): [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
  VTP Condition C vs D: ... | p=... | d=...
  Effect size by epistemic type — normative: ... constative: ... mixed: ...
  Interpretation: ...

PROPOSITION 4 (Regulation-anchored assembly → compliance risk reduction): [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
  AC Condition B vs D: ... | p=... | d=...
  Effect size by repetitiveness — high: ... low: ...
  Interpretation: ...

K-INVARIANCE:
  VTP plateau for Condition A at k=...: ...
  CSR for Conditions A and B at all k: ...

VENUE RECOMMENDATION:
  If all four propositions supported with d > 0.5: MISQ special issue
  If propositions supported but d < 0.5 or mixed results: SIGIR or EMNLP systems track
  If propositions not supported: revisit theoretical model before submission
```

---

## Implementation Notes

- Use existing `pipeline/db.py` database connections and `api/reasoning.py` retrieval logic
- Condition D must use the `regulation_anchored_context()` function implemented in the previous session
- Scaffold the harness in `pipeline/eval_harness.py`
- Before building, read `api/reasoning.py` and `pipeline/db.py` to confirm available functions and connection patterns
- Log progress to `data/eval/eval_log.txt` — one line per query with timestamp, query_id, condition, metric values, and status
- Estimated runtime: 2–4 hours for query generation + retrieval + analysis
- Run as a background process: `nohup python pipeline/eval_harness.py > data/eval/stdout.log 2>&1 &`
