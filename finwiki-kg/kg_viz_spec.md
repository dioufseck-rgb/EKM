# FinWiki Knowledge Graph — Interactive Visualization

Build a standalone HTML file at `tools/kg_viz/index.html` that renders an interactive force-directed knowledge graph from live Neo4j and PostgreSQL data. The file must be self-contained — one HTML file, no build step, no external dependencies except CDN libraries.

---

## Data Pipeline

On page load, fetch all data from the FinWiki API (FastAPI at `http://localhost:8000`). Add these three endpoints to `api/main.py` if they do not already exist:

### GET /viz/nodes
Returns document-level nodes with centrality metrics.

```python
@app.get("/viz/nodes")
async def viz_nodes():
    """
    Returns all documents as graph nodes with:
    - doc_id, title, category (regulatory/risk/financial_theory/products/other)
    - assertion_count, warrant_count, ground_count
    - conflict_count (number of CONTRADICTS edges touching this doc)
    - ai_readiness_score (0-1): assertion_density * type_coverage * (1 - isolation_rate)
    """
    # PostgreSQL query
    query = """
        SELECT
            a.doc_id,
            COUNT(a.assertion_id) AS assertion_count,
            SUM(CASE WHEN a.discourse_role = 'warrant' THEN 1 ELSE 0 END) AS warrant_count,
            SUM(CASE WHEN a.discourse_role = 'ground' THEN 1 ELSE 0 END) AS ground_count,
            SUM(CASE WHEN a.discourse_role = 'claim' THEN 1 ELSE 0 END) AS claim_count,
            SUM(CASE WHEN a.validity_claim_type = 'normative' THEN 1 ELSE 0 END) AS normative_count,
            SUM(CASE WHEN a.validity_claim_type = 'expressive' THEN 1 ELSE 0 END) AS expressive_count
        FROM assertions a
        GROUP BY a.doc_id
        ORDER BY assertion_count DESC
    """
    # Conflict count per doc from assertion_relationships
    conflict_query = """
        SELECT doc_id, COUNT(*) AS conflict_count FROM (
            SELECT a.doc_id FROM assertions a
            JOIN assertion_relationships r ON a.assertion_id = r.source_assertion_id
            WHERE r.relationship_type = 'CONTRADICTS'
            UNION ALL
            SELECT a.doc_id FROM assertions a
            JOIN assertion_relationships r ON a.assertion_id = r.target_assertion_id
            WHERE r.relationship_type = 'CONTRADICTS'
        ) sub GROUP BY doc_id
    """
```

**Category assignment logic** — classify each `doc_id` by keyword matching:
- `regulatory`: contains Basel, AIFMD, Dodd, MiFID, GDPR, SOX, Sarbanes, Volcker, FATCA, IFRS, Directive, Act, Regulation
- `risk`: contains risk, VaR, stress, concentration, operational, liquidity, credit, market
- `financial_theory`: contains CAPM, APT, Black, Scholes, arbitrage, portfolio, Sharpe, Markowitz
- `products`: contains fund, swap, derivative, option, futures, bond, equity, ETF, hedge
- `other`: everything else

**AI-readiness score formula:**
```
assertion_density = min(assertion_count / 150, 1.0)
type_coverage = (normative_count + ground_count) / max(assertion_count, 1)
isolation_rate = isolated_assertions / assertion_count  # assertions with 0 logical edges
ai_readiness = (assertion_density * 0.3 + type_coverage * 0.5 + (1 - isolation_rate) * 0.2)
```

### GET /viz/edges
Returns document-level edge aggregates (not assertion-level — too many).

```python
@app.get("/viz/edges")
async def viz_edges():
    """
    Aggregates assertion-level logical relations to document pairs.
    Returns one edge per (source_doc, target_doc, relation_type) triple
    with count and mean confidence.
    """
    query = """
        SELECT
            a1.doc_id AS source_doc,
            a2.doc_id AS target_doc,
            r.relationship_type AS rel_type,
            COUNT(*) AS edge_count,
            AVG(r.confidence) AS mean_confidence
        FROM assertion_relationships r
        JOIN assertions a1 ON r.source_assertion_id = a1.assertion_id
        JOIN assertions a2 ON r.target_assertion_id = a2.assertion_id
        WHERE a1.doc_id != a2.doc_id OR r.relationship_type = 'CONTRADICTS'
        GROUP BY a1.doc_id, a2.doc_id, r.relationship_type
        HAVING COUNT(*) >= 1
        ORDER BY edge_count DESC
    """
```

### GET /viz/centrality
Returns top-N centrality rankings for concepts, regulations, and assertions.

```python
@app.get("/viz/centrality")
async def viz_centrality():
    # Neo4j: concept degree centrality
    concept_query = """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        RETURN c.name AS name, COUNT(r) AS degree
        ORDER BY degree DESC LIMIT 20
    """
    # Neo4j: regulation authority (assertion count + warrant count + doc count)
    reg_query = """
        MATCH (reg:Regulation)<-[:REFERENCES]-(a:Assertion)
        RETURN reg.name AS name,
               COUNT(a) AS assertion_count,
               SUM(CASE WHEN a.discourse_role = 'warrant' THEN 1 ELSE 0 END) AS warrants,
               COUNT(DISTINCT a.doc_id) AS doc_count
        ORDER BY assertion_count DESC LIMIT 20
    """
    # PostgreSQL: top assertions by logical edge degree
    assertion_query = """
        SELECT a.assertion_id, a.assertion_text, a.discourse_role,
               a.validity_claim_type, a.doc_id, COUNT(r.relationship_id) AS degree
        FROM assertions a
        JOIN assertion_relationships r
          ON a.assertion_id = r.source_assertion_id OR a.assertion_id = r.target_assertion_id
        GROUP BY a.assertion_id, a.assertion_text, a.discourse_role,
                 a.validity_claim_type, a.doc_id
        ORDER BY degree DESC LIMIT 20
    """
```

---

## Visualization — `tools/kg_viz/index.html`

Self-contained single HTML file. Load D3.js from CDN for the force simulation:
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
```

### Layout

```
┌─────────────────────────────────────────────────────┐
│  Header: title + corpus stats strip                  │
├──────────────────────────┬──────────────────────────┤
│                          │  Controls panel           │
│   Force-directed graph   │  - Filter buttons         │
│       (SVG, fills        │  - Confidence slider      │
│        available         │  - Color-by selector      │
│        height)           │  - Node detail on click   │
│                          │                           │
├──────────────────────────┴──────────────────────────┤
│  Centrality rankings — three columns                 │
│  Concepts | Regulations | Assertions                 │
└─────────────────────────────────────────────────────┘
```

### Graph mechanics (D3 force simulation)

```javascript
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges)
        .id(d => d.doc_id)
        .distance(d => d.rel_type === 'CONTRADICTS' ? 80 : 120)
        .strength(d => d.mean_confidence * 0.4))
    .force("charge", d3.forceManyBody().strength(-400))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 8));
```

Node radius encodes assertion count:
```javascript
const nodeRadius = d => Math.max(8, Math.min(28, Math.sqrt(d.assertion_count) * 2.2));
```

### Visual encoding

**Node fill color** — by category (default) or by AI-readiness score (toggle):

Category mode:
```javascript
const catColor = {
    regulatory:   "#185FA5",   // blue-600
    risk:         "#D85A30",   // coral-600
    financial_theory: "#7F77DD", // purple-400
    products:     "#1D9E75",   // teal-400
    other:        "#888780"    // gray-400
};
```

AI-readiness mode — continuous scale:
```javascript
const readinessColor = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3.interpolateRgb("#E24B4A", "#639922")); // red→green
```

**Node border**: 1.5px white stroke. Nodes with `conflict_count > 0` get an additional 2px `#E24B4A` outer ring.

**Node label**: Show `doc_id` truncated to 14 chars, centered, white, 10px. Only show labels for nodes with `assertion_count >= 30` at default zoom; show all labels when zoomed in past 1.5x.

**Edge stroke**:
- ENTAILS: `#378ADD` at 30% opacity, 1px
- CAUSES: `#1D9E75` at 30% opacity, 1px
- SPECIALIZES: `#7F77DD` at 30% opacity, 1px
- CONTRADICTS: `#E24B4A` at 90% opacity, 2px, `stroke-dasharray: 5,3`
- All others: `#888780` at 20% opacity, 0.5px

Edge thickness encodes `edge_count`: `strokeWidth = 0.5 + Math.min(edge_count / 5, 3)`

### Controls panel

**Filter buttons** (mutually exclusive):
- All edges
- Regulatory only (nodes where cat=regulatory + their immediate neighbors)
- Conflicts only (show only CONTRADICTS edges + their endpoint nodes)
- High AI-readiness (nodes with ai_readiness_score >= 0.7, dimmed otherwise)

**Confidence threshold slider**: 0.50 – 1.00, step 0.05. Hides edges below threshold. Default 0.65.

**Color-by toggle**: "Category" / "AI-readiness"

**Node detail panel** (shown on click, in right panel):
```
[Document title]
Category: regulatory
Assertions: 70  |  Warrants: 16  |  Grounds: 54
Normative: 38%  |  Constative: 58%  |  Expressive: 4%
Conflicts: 0
AI-readiness: 0.82  ████████░░
[Connections list: 8 edges to 6 documents]
```
Each connection listed as: `Doc name — ENTAILS (×3, conf 0.88)`

Add a "Explore in graph ↗" button that calls `sendPrompt('Tell me about the knowledge quality of ' + node.doc_id)`.

### Centrality rankings (bottom strip)

Three equal-width columns. Each shows top 10 with animated horizontal bars.

Column 1 — Concept nodes (degree centrality):
- Bar color: category color of the concept
- Show degree number right-aligned

Column 2 — Regulation nodes (authority = assertion_count, sized by warrant_count):
- Bar fill = `#185FA5`, bar width = assertion_count / max
- Overlay a darker segment for warrant_count proportion
- Tooltip shows: "16 warrants / 83 assertions across 9 documents"

Column 3 — Assertion nodes (edge degree):
- Bar color: blue for warrant, green for ground, gray for other
- Label = first 35 chars of assertion_text
- Click opens the assertion's document in the node detail panel

### Interactions

- **Drag nodes**: standard D3 drag, pins node (fixes position), double-click to unpin
- **Zoom/pan**: D3 zoom on SVG, scroll to zoom, drag background to pan
- **Hover node**: tooltip showing doc_id, assertion_count, conflict_count, ai_readiness
- **Click node**: populate right detail panel
- **Click edge**: tooltip showing rel_type, edge_count, mean_confidence, source → target doc names
- **Double-click background**: reset zoom to fit

### Stats strip (header)

Show five numbers pulled from `/viz/nodes` and `/viz/edges` aggregation:
```
12,601 assertions   |   106 documents   |   6,482 relations   |   63 conflicts   |   Avg AI-readiness: 0.61
```

---

## Error handling

- If API is unreachable on load, show a banner: "API not available — displaying sample data" and load the hardcoded fallback data (use the synthetic data from the centrality analysis script as the fallback).
- Individual endpoint failures should not crash the page — degrade gracefully (e.g., if `/viz/centrality` fails, hide the centrality strip and log the error).
- Show a loading spinner on the canvas while data is fetching.

---

## File output

Single file: `tools/kg_viz/index.html`

After building, start a simple HTTP server to verify it loads without errors:
```bash
cd tools/kg_viz && python3 -m http.server 8080
```
Then confirm `http://localhost:8080` loads the graph with live data from the API.

Report back:
1. Number of nodes and edges loaded from the live API
2. Top 3 nodes by degree centrality
3. Number of CONTRADICTS edges visible at default confidence threshold
4. Any endpoints that needed to be created vs already existed
