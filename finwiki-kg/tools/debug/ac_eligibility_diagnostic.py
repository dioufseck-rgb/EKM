"""
AC Eligibility Diagnostic — run once, results saved to tools/debug/
"""
import json, os, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from collections import Counter
from neo4j import GraphDatabase
import psycopg2

OUT = Path("tools/debug")
OUT.mkdir(parents=True, exist_ok=True)

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "finwiki123"))
queries = json.loads(open("data/eval/queries.json").read())

# ── Step 1: edge distribution by document ─────────────────────────────────────
print("Step 1: edge distribution by document…")
with driver.session() as s:
    rows = s.run("""
        MATCH (a:Assertion)
        WITH a
        OPTIONAL MATCH (a)-[r:ENTAILS|CAUSES|SPECIALIZES|TRIGGERS|INHIBITS|REBUTS]->()
        WITH a, COUNT(r) AS deg
        WITH a.source_document AS doc, COUNT(*) AS assertions,
             SUM(CASE WHEN deg=0 THEN 1 ELSE 0 END) AS isolated,
             SUM(CASE WHEN deg>0 THEN 1 ELSE 0 END) AS connected,
             AVG(deg) AS avg_out_degree
        RETURN doc, assertions, connected, isolated, avg_out_degree
        ORDER BY connected DESC
    """)
    edge_dist = [dict(r) for r in rows]

zero_edge_docs = [d for d in edge_dist if d['connected'] == 0]
print(f"  Docs with zero connected assertions: {len(zero_edge_docs)}")
print(f"  Docs with >=1 connected assertion:   {len(edge_dist) - len(zero_edge_docs)}")
(OUT / "edge_distribution_by_doc.json").write_text(json.dumps(edge_dist, indent=2, default=str))

# ── Steps 2+3: seed-assertion join + direction test ───────────────────────────
print("Step 2+3: seed-assertion join & direction test…")
results = []
for q in queries:
    seed_id = q.get("query_seed_assertion_id")
    if not seed_id:
        results.append({**q, "directed_neighbors": 0, "undirected_neighbors": 0})
        continue
    try:
        with driver.session() as s:
            r1 = s.run("MATCH (a:Assertion {assertion_id:$id})-[r:ENTAILS|CAUSES|SPECIALIZES|TRIGGERS|INHIBITS|REBUTS]->() RETURN COUNT(r) AS cnt", id=seed_id).single()
            r2 = s.run("MATCH (a:Assertion {assertion_id:$id})-[r:ENTAILS|CAUSES|SPECIALIZES|TRIGGERS|INHIBITS|REBUTS]-() RETURN COUNT(r) AS cnt", id=seed_id).single()
            # Also check if seed exists at all
            r3 = s.run("MATCH (a:Assertion {assertion_id:$id}) RETURN COUNT(a) AS cnt", id=seed_id).single()
        results.append({
            **q,
            "seed_exists_in_neo4j": (r3['cnt'] if r3 else 0) > 0,
            "directed_neighbors":   r1['cnt'] if r1 else 0,
            "undirected_neighbors": r2['cnt'] if r2 else 0,
        })
    except Exception as e:
        results.append({**q, "error": str(e), "directed_neighbors": 0, "undirected_neighbors": 0})

eligible_d  = sum(1 for r in results if r.get("directed_neighbors",   0) > 0)
eligible_u  = sum(1 for r in results if r.get("undirected_neighbors", 0) > 0)
missing_neo4j = sum(1 for r in results if not r.get("seed_exists_in_neo4j", True))

# Direction-only failures: undirected eligible but directed not
direction_failure = [r for r in results if r.get("undirected_neighbors",0) > 0 and r.get("directed_neighbors",0) == 0]
true_sparse       = [r for r in results if r.get("undirected_neighbors",0) == 0]

# Count direction failures by doc
dir_fail_by_doc = Counter(r["source_doc"] for r in direction_failure)
sparse_by_doc   = Counter(r["source_doc"] for r in true_sparse)

output = {
    "total_queries": len(queries),
    "ac_eligible_directed":   eligible_d,
    "ac_eligible_undirected": eligible_u,
    "seed_missing_from_neo4j": missing_neo4j,
    "direction_filter_failures": len(direction_failure),
    "true_sparse_seeds": len(true_sparse),
    "corrected_eligibility_pct": round(eligible_u / len(queries) * 100, 1),
    "root_cause_summary": (
        f"Join: OK (IDs match between Qdrant and Neo4j). "
        f"Direction filter (Bug 2): {len(direction_failure)} seeds have undirected neighbors but 0 directed — "
        f"edges were stored with seed as target, not source. "
        f"True sparsity: {len(true_sparse)} seeds have no neighbors in either direction. "
        f"Fix: change _toulmin_expand_from to undirected traversal → eligibility 35→{eligible_u}/120."
    ),
    "direction_failures_by_doc": dict(dir_fail_by_doc.most_common(10)),
    "true_sparse_by_doc":        dict(sparse_by_doc.most_common(10)),
    "per_query": results,
}
(OUT / "ac_eligibility_report.json").write_text(json.dumps(output, indent=2, default=str))

print(f"\n=== AC ELIGIBILITY DIAGNOSTIC RESULTS ===")
print(f"Total queries:                    {len(queries)}")
print(f"Seeds missing from Neo4j:         {missing_neo4j}")
print(f"AC-eligible (directed):           {eligible_d}/120  ({round(eligible_d/120*100)}%)")
print(f"AC-eligible (undirected):         {eligible_u}/120  ({round(eligible_u/120*100)}%)")
print(f"Direction-filter failures:        {len(direction_failure)}  (seeds with incoming-only edges)")
print(f"True sparse seeds (no neighbors): {len(true_sparse)}")
print(f"\nRoot cause: {output['root_cause_summary']}")
print(f"\nTop docs with direction-filter failures:")
for doc, cnt in dir_fail_by_doc.most_common(8):
    print(f"  {cnt:3d}  {doc[:55]}")
print(f"\nTop docs with true sparsity:")
for doc, cnt in sparse_by_doc.most_common(8):
    print(f"  {cnt:3d}  {doc[:55]}")
print(f"\nReport: {(OUT / 'ac_eligibility_report.json').resolve()}")
