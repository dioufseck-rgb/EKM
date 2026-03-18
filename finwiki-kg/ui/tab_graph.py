"""ui/tab_graph.py — Graph Explorer tab using pyvis for interactive visualization."""
import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Node color map keyed by (label, epistemic_status)
NODE_COLORS = {
    "Assertion_authoritative": "#1a7a1a",
    "Assertion_inferred":      "#1a4a7a",
    "Assertion_contested":     "#cc6600",
    "Assertion_derived":       "#6600cc",
    "Assertion_deprecated":    "#999999",
    "Assertion_orphaned":      "#cccccc",
    "Concept":                 "#3d3d8a",
    "Regulation":              "#8a1a1a",
    "Document":                "#1a6b6b",
    "Topic":                   "#5a5a5a",
    "Chunk":                   "#aaaaaa",
}

# Edge color/style map
DEDUCTIVE_TYPES  = {"ENTAILS","EQUIVALENT","DEFINES","TRIGGERS","SUPERSEDES","SPECIALIZES","GENERALIZES"}
CAUSAL_TYPES     = {"CAUSES","INHIBITS"}
EMPIRICAL_TYPES  = {"CORRELATES_WITH"}
CONFLICT_TYPES   = {"CONTRADICTS","DUPLICATE_OF"}
STRUCTURAL_TYPES = {"SOURCED_FROM","CHUNK_OF","GOVERNS","REFERENCES","TAGGED_WITH",
                     "INSTANTIATES","CLASSIFIES","PRECEDES","OPERATIONALIZES","LOGICAL_RELATION"}

EDGE_COLOR = {
    **{t: "#2255cc" for t in DEDUCTIVE_TYPES},
    **{t: "#cc6600" for t in CAUSAL_TYPES},
    **{t: "#cc8800" for t in EMPIRICAL_TYPES},
    **{t: "#cc0000" for t in CONFLICT_TYPES},
    **{t: "#cccccc" for t in STRUCTURAL_TYPES},
}


def render(api_url: str) -> None:
    st.header("🕸️ Knowledge Graph Explorer")

    if not PYVIS_AVAILABLE:
        st.warning("pyvis is not installed. Run `pip install pyvis` to enable graph visualization.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col_search, col_depth = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "Search (assertion_id, subject, or keyword)",
            placeholder="e.g., Basel III capital requirement",
        )
    with col_depth:
        depth = st.slider("Depth", 1, 4, 2)

    # Edge type toggles
    st.caption("Show edge types:")
    et_cols = st.columns(4)
    with et_cols[0]: show_deductive  = st.checkbox("Deductive (ENTAILS…)", value=True)
    with et_cols[1]: show_causal     = st.checkbox("Causal (CAUSES…)",     value=True)
    with et_cols[2]: show_conflicts  = st.checkbox("Conflicts (CONTRADICTS)", value=True)
    with et_cols[3]: show_structural = st.checkbox("Structural / provenance", value=False)

    if not query:
        st.info("Enter a search term to explore the knowledge graph.")
        return

    # ── Search ────────────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{api_url}/search", params={"q": query, "limit": 5}, timeout=30)
        resp.raise_for_status()
        results = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {api_url}")
        return
    except Exception as e:
        st.error(f"Search failed: {e}")
        return

    if not results:
        st.warning("No matching assertions found.")
        return

    options = {
        f"{r['assertion']['claim_text'][:80]}…": r["assertion"]["assertion_id"]
        for r in results
    }
    selected_label = st.selectbox("Select assertion to explore", list(options.keys()))
    assertion_id   = options[selected_label]

    # ── Neighbourhood ─────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            f"{api_url}/graph/neighborhood/{assertion_id}",
            params={"depth": depth},
            timeout=30,
        )
        resp.raise_for_status()
        neighborhood = resp.json()
    except Exception as e:
        st.error(f"Failed to load neighbourhood: {e}")
        return

    nodes = neighborhood.get("nodes", [])
    edges = neighborhood.get("edges", [])

    if not nodes:
        st.info("No neighbourhood data found for this assertion.")
        return

    # ── Build pyvis network ───────────────────────────────────────────────────
    net = Network(
        height="620px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=True,
    )
    net.set_options(json.dumps({
        "physics": {"enabled": True, "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -50}},
        "edges":   {"smooth": {"type": "dynamic"}},
        "interaction": {"hover": True},
    }))

    node_ids_added = set()
    for node in nodes:
        nid    = node["id"]
        label_type = node.get("label", "Node")
        props  = node.get("props", {})
        epi    = props.get("epistemic_status", "authoritative")
        color_key = f"Assertion_{epi}" if label_type == "Assertion" else label_type
        color  = NODE_COLORS.get(color_key, "#555555")
        title  = props.get("claim_text") or props.get("name") or nid
        size   = 22 if label_type == "Assertion" else 14
        net.add_node(str(nid), label=str(nid)[:14], title=str(title)[:250],
                     color=color, size=size)
        node_ids_added.add(str(nid))

    edge_id_set = set()
    for edge in edges:
        etype = edge.get("type", "")
        src   = str(edge.get("source", "?"))
        tgt   = str(edge.get("target", "?"))

        if src not in node_ids_added or tgt not in node_ids_added:
            continue

        # Filter
        if etype in DEDUCTIVE_TYPES  and not show_deductive:  continue
        if etype in CAUSAL_TYPES     and not show_causal:      continue
        if etype in EMPIRICAL_TYPES  and not show_causal:      continue
        if etype in CONFLICT_TYPES   and not show_conflicts:   continue
        if etype in STRUCTURAL_TYPES and not show_structural:  continue

        edge_key = (src, tgt, etype)
        if edge_key in edge_id_set:
            continue
        edge_id_set.add(edge_key)

        color  = EDGE_COLOR.get(etype, "#888888")
        dashes = etype in {*EMPIRICAL_TYPES, *CONFLICT_TYPES}
        width  = 3 if etype in CONFLICT_TYPES else 1

        net.add_edge(src, tgt, label=etype, color=color,
                     dashes=dashes, width=width, title=etype)

    # Render
    html_path = "/tmp/finwiki_graph.html"
    net.save_graph(html_path)
    with open(html_path) as f:
        html_content = f.read()
    components.html(html_content, height=640)

    st.caption(
        f"Showing {len(nodes)} nodes · {len(edges)} edges · depth={depth} · "
        f"Deductive={show_deductive} Causal={show_causal} Conflict={show_conflicts}"
    )

    # Sidebar: edge detail on click (static info)
    with st.sidebar:
        st.subheader("Edge Legend")
        st.markdown("🔵 **Deductive** (ENTAILS, DEFINES, TRIGGERS…) — truth-preserving")
        st.markdown("🟠 **Causal** (CAUSES, INHIBITS) — defeasible, context only")
        st.markdown("🟠 **Empirical** (CORRELATES_WITH) — dashed, defeasible")
        st.markdown("🔴 **Conflict** (CONTRADICTS) — thick dashed")
        st.markdown("⚪ **Structural** (SOURCED_FROM, GOVERNS…) — provenance")
